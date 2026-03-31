"""Single-epoch training utilities for the custom training loop.

This module contains batch transfer helpers, gradient statistics, optional AMP
execution, metric logging, and the full per-epoch optimization routine used by
the higher-level training entry point.
"""

import time
import torch
import torch.nn as nn

from training.autocast import * 
from training.efficent_metrics import *


def move_batch_to_device(batch, device: str):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def compute_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5

def gpu_mem_mb(device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserv = torch.cuda.memory_reserved() / (1024**2)
        return alloc, reserv
    return 0.0, 0.0


def resolve_batch_num_recycles(
    *,
    num_recycles: int = 0,
    stochastic_recycling: bool = False,
    max_recycles: int | None = None,
    device: torch.device | str | None = None,
) -> int:
    """
    Resolve the recycle count to use for a single training batch.

    When stochastic recycling is enabled, the count is sampled uniformly from
    ``[0, upper]`` inclusive using PyTorch RNG so the result follows the same
    seeding policy as the rest of training.
    """
    num_recycles = max(0, int(num_recycles))

    if not stochastic_recycling:
        return num_recycles

    upper = num_recycles if max_recycles is None else int(max_recycles)
    if upper < 0:
        raise ValueError("max_recycles must be >= 0 when stochastic_recycling=True")

    sample_device = device if device is not None else "cpu"
    return int(torch.randint(0, upper + 1, (1,), device=sample_device).item())


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,                      # AlphaFoldLoss
    *,
    device: str = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    scaler=None,
    scheduler=None,
    ema=None,
    ema_target=None,
    grad_clip: float | None = 1.0,
    grad_accum_steps: int = 1,
    max_batches: int | None = None,
    log_every: int = 10,
    log_grad_norm: bool = True,
    log_mem: bool = False,
    on_oom: str = "skip",
    global_step: int = 0,
    ideal_backbone_local: torch.Tensor | None = None,
    num_recycles: int = 0,
    stochastic_recycling: bool = False,
    max_recycles: int | None = None,
):
    """
    Train one epoch for AlphaFold2-like model.

    Notes
    -----
    - Losses are accumulated on every batch.
    - Structural metrics (RMSD/TM/GDT) are computed only on logging steps.
    - Scheduler / EMA / global_step advance only if optimizer step actually happened.
    - ``num_recycles`` controls the fixed number of extra recycling passes.
    - If ``stochastic_recycling=True``, each batch samples its recycle count
      uniformly from ``[0, max_recycles]`` inclusive. When ``max_recycles`` is
      omitted, ``num_recycles`` is used as the upper bound.
    """

    model.train()

    if ema_target is None:
        ema_target = model

    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    running = {
        "loss": 0.0,
        "fape_loss": 0.0,
        "dist_loss": 0.0,
        "plddt_loss": 0.0,
        "torsion_loss": 0.0,
        "num_recycles": 0.0,
        "rmsd_logged": 0.0,
        "tm_score_logged": 0.0,
        "gdt_ts_logged": 0.0,}

    n_seen_batches = 0
    n_optimizer_steps = 0
    n_seen_samples = 0
    n_metric_logs = 0

    if log_every:
        print("┆ In-epoch statistics (AlphaFold2-like)")
        print(
            "┆   {:>8} | {:>8} | {:>9} | {:>9} | {:>9} | {:>8} | {:>8} | {:>8}{}".format(
                "step", "batch", "loss", "fape", "dist", "rmsd", "tm", "gdt",
                (" | grad_norm | mem(MB)" if (log_grad_norm or log_mem) else "")
            )
        )
        print("┆   " + "─" * 118)

    for i, batch in enumerate(dataloader):

        if (max_batches is not None) and (i >= max_batches):
            break
        try:
            t0 = time.perf_counter()

            batch = move_batch_to_device(batch, device)
            B = batch["seq_tokens"].shape[0]
            n_seen_samples += B
            batch_num_recycles = resolve_batch_num_recycles(
                num_recycles=num_recycles,
                stochastic_recycling=stochastic_recycling,
                max_recycles=max_recycles,
                device=batch["seq_tokens"].device,
            )

            # -------------------------
            # Forward
            # -------------------------
            with autocast_ctx(device=device, enabled=amp_enabled, amp_dtype=amp_dtype):
                out = model(
                    seq_tokens=batch["seq_tokens"],
                    msa_tokens=batch["msa_tokens"],
                    seq_mask=batch["seq_mask"],
                    msa_mask=batch["msa_mask"],
                    ideal_backbone_local=ideal_backbone_local,
                    num_recycles=batch_num_recycles,)

                loss_dict = criterion(out, batch)
                loss = loss_dict["loss"] / grad_accum_steps

            # -------------------------
            # backward
            # -------------------------
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_now = ((i + 1) % grad_accum_steps) == 0
            grad_norm = None
            optimizer_step_happened = False

            if step_now:
                did_unscale = False

                if log_grad_norm:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        did_unscale = True
                    grad_norm = compute_grad_norm(model)

                if grad_clip is not None:
                    if scaler is not None and (not did_unscale):
                        scaler.unscale_(optimizer)
                        did_unscale = True
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if scaler is not None:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)     # puede omitirse internamente si hubo overflow
                    scaler.update()
                    new_scale = scaler.get_scale()

                    # overflow -> new_scale < old_scale
                    # no overflow -> new_scale == old_scale o mayor
                    optimizer_step_happened = new_scale >= old_scale
                else:
                    optimizer.step()
                    optimizer_step_happened = True

                if optimizer_step_happened:
                    if scheduler is not None:
                        scheduler.step()

                    if ema is not None:
                        ema.update(ema_target)

                    global_step += 1
                    n_optimizer_steps += 1

                optimizer.zero_grad(set_to_none=True)

            # -------------------------
            # acumular losses
            # -------------------------
            loss_val = float(loss_dict["loss"].detach().item())
            running["loss"] += loss_val
            running["fape_loss"] += float(loss_dict["fape_loss"].detach().item())
            running["dist_loss"] += float(loss_dict["dist_loss"].detach().item())
            running["plddt_loss"] += float(loss_dict["plddt_loss"].detach().item())
            running["torsion_loss"] += float(loss_dict["torsion_loss"].detach().item())
            running["num_recycles"] += float(batch_num_recycles)

            n_seen_batches += 1

            # -------------------------
            # logging: métricas costosas solo aquí
            # -------------------------
            should_log = (
                log_every
                and step_now
                and optimizer_step_happened
                and (global_step % log_every == 0))

            if should_log:
                with torch.no_grad():
                    x_true = batch["coords_ca"].detach()
                    mask = batch["valid_res_mask"].detach()

                    if out.get("backbone_coords", None) is not None:
                        # [B, L, 4, 3] -> tomar CA (índice 1)
                        x_pred = out["backbone_coords"][:, :, 1, :].detach()
                    else:
                        # fallback: translation head
                        x_pred = out["t"].detach()

                    metrics = compute_structure_metrics(
                        x_pred=x_pred,
                        x_true=x_true,
                        mask=mask,
                        align=True)

                running["rmsd_logged"] += float(metrics["rmsd"].item())
                running["tm_score_logged"] += float(metrics["tm_score"].item())
                running["gdt_ts_logged"] += float(metrics["gdt_ts"].item())
                n_metric_logs += 1

                dt_ms = (time.perf_counter() - t0) * 1000.0

                if log_mem:
                    alloc, reserv = gpu_mem_mb(device)
                    mem_msg = f"{alloc:.0f}/{reserv:.0f}"
                else:
                    mem_msg = "—"

                gn_str = f"{grad_norm:.2e}" if grad_norm is not None else "—"

                print(
                    "┆   {:8d} | {:8d} | {:9.4f} | {:9.4f} | {:9.4f} | {:8.3f} | {:8.3f} | {:8.3f} | {:>9} | {:>9} | {:7.1f}ms".format(
                        global_step,
                        i + 1,
                        loss_val,
                        float(loss_dict["fape_loss"].detach().item()),
                        float(loss_dict["dist_loss"].detach().item()),
                        float(metrics["rmsd"].item()),
                        float(metrics["tm_score"].item()),
                        float(metrics["gdt_ts"].item()),
                        gn_str,
                        mem_msg,
                        dt_ms))

        except RuntimeError as e:
            if ("CUDA out of memory" in str(e)) and (on_oom == "skip"):
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[WARN][OOM] Batch {i} omitido. Limpié cache y sigo.")
                optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise

    # -------------------------
    # promedios de epoch
    # -------------------------
    denom_loss = max(1, n_seen_batches)
    denom_metrics = max(1, n_metric_logs)

    epoch_stats = {
        "loss": running["loss"] / denom_loss,
        "fape_loss": running["fape_loss"] / denom_loss,
        "dist_loss": running["dist_loss"] / denom_loss,
        "plddt_loss": running["plddt_loss"] / denom_loss,
        "torsion_loss": running["torsion_loss"] / denom_loss,
        "num_recycles": running["num_recycles"] / denom_loss,
        # promedios solo sobre los puntos donde sí se loguearon métricas
        "rmsd_logged": running["rmsd_logged"] / denom_metrics if n_metric_logs > 0 else float("nan"),
        "tm_score_logged": running["tm_score_logged"] / denom_metrics if n_metric_logs > 0 else float("nan"),
        "gdt_ts_logged": running["gdt_ts_logged"] / denom_metrics if n_metric_logs > 0 else float("nan"),
        "n_seen_batches": n_seen_batches,
        "n_optimizer_steps": n_optimizer_steps,
        "n_seen_samples": n_seen_samples,
        "n_metric_logs": n_metric_logs,}

    return epoch_stats, global_step
