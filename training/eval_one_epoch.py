"""Evaluation utilities for AlphaFold2-like runs.

This module mirrors the train-time epoch loop but keeps the model in eval mode,
computes losses and structure metrics, and never performs optimizer steps.
"""

from __future__ import annotations

import gc
import time

import torch

from training.autocast import autocast_ctx
from training.efficient_metrics import compute_structure_metrics
from training.train_one_epoch import gpu_mem_mb, move_batch_to_device, resolve_batch_num_recycles


def _extract_metric_coords(outputs: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    x_true = batch["coords_ca"].detach()
    if outputs.get("backbone_coords") is not None:
        x_pred = outputs["backbone_coords"][:, :, 1, :].detach()
    else:
        x_pred = outputs["t"].detach()
    return x_pred, x_true


def eval_one_epoch(
    model,
    dataloader,
    criterion,
    *,
    device: str = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    max_batches: int | None = None,
    log_every: int = 10,
    log_mem: bool = False,
    on_oom: str = "skip",
    ideal_backbone_local: torch.Tensor | None = None,
    num_recycles: int = 0,
    stochastic_recycling: bool = False,
    max_recycles: int | None = None,
    is_main_process: bool = True,
):
    """Evaluate one epoch without updating model weights."""
    model.eval()

    running = {
        "loss": 0.0,
        "fape_loss": 0.0,
        "dist_loss": 0.0,
        "msa_loss": 0.0,
        "plddt_loss": 0.0,
        "torsion_loss": 0.0,
        "num_recycles": 0.0,
        "rmsd_logged": 0.0,
        "tm_score_logged": 0.0,
        "gdt_ts_logged": 0.0,
    }

    n_seen_batches = 0
    n_seen_samples = 0
    n_metric_logs = 0

    if log_every and is_main_process:
        print("┆ Eval statistics (AlphaFold2-like)")
        print(
            "┆   {:>8} | {:>8} | {:>9} | {:>9} | {:>9} | {:>9} | {:>8} | {:>8} | {:>8}{}".format(
                "batch",
                "recycles",
                "loss",
                "fape",
                "dist",
                "msa",
                "rmsd",
                "tm",
                "gdt",
                (" | mem(MB)" if log_mem else ""),
            )
        )
        print("┆   " + "─" * 108)

    for index, batch in enumerate(dataloader):
        if max_batches is not None and index >= max_batches:
            break

        try:
            t0 = time.perf_counter()
            batch = move_batch_to_device(batch, device)
            n_seen_samples += int(batch["seq_tokens"].shape[0])
            batch_num_recycles = resolve_batch_num_recycles(
                num_recycles=num_recycles,
                stochastic_recycling=stochastic_recycling,
                max_recycles=max_recycles,
                device=batch["seq_tokens"].device,
            )

            with torch.no_grad():
                with autocast_ctx(device=device, enabled=amp_enabled, amp_dtype=amp_dtype):
                    outputs = model(
                        seq_tokens=batch["seq_tokens"],
                        msa_tokens=batch["msa_tokens"],
                        seq_mask=batch["seq_mask"],
                        msa_mask=batch["msa_mask"],
                        ideal_backbone_local=ideal_backbone_local,
                        num_recycles=batch_num_recycles,
                        extra_msa_feat=batch.get("extra_msa_feat"),
                        extra_msa_mask=batch.get("extra_msa_mask"),
                        template_angle_feat=batch.get("template_angle_feat"),
                        template_pair_feat=batch.get("template_pair_feat"),
                        template_mask=batch.get("template_mask"),
                    )
                    loss_dict = criterion(outputs, batch)

                x_pred, x_true = _extract_metric_coords(outputs, batch)
                metrics = compute_structure_metrics(
                    x_pred=x_pred,
                    x_true=x_true,
                    mask=batch["valid_res_mask"].detach(),
                    align=True,
                )

            running["loss"] += float(loss_dict["loss"].detach().item())
            running["fape_loss"] += float(loss_dict["fape_loss"].detach().item())
            running["dist_loss"] += float(loss_dict["dist_loss"].detach().item())
            running["msa_loss"] += float(loss_dict["msa_loss"].detach().item())
            running["plddt_loss"] += float(loss_dict["plddt_loss"].detach().item())
            running["torsion_loss"] += float(loss_dict["torsion_loss"].detach().item())
            running["num_recycles"] += float(batch_num_recycles)
            running["rmsd_logged"] += float(metrics["rmsd"].item())
            running["tm_score_logged"] += float(metrics["tm_score"].item())
            running["gdt_ts_logged"] += float(metrics["gdt_ts"].item())

            n_seen_batches += 1
            n_metric_logs += 1

            if log_every and is_main_process and ((index + 1) % log_every == 0):
                if log_mem:
                    alloc, reserv = gpu_mem_mb(device)
                    mem_msg = f" | {alloc:.0f}/{reserv:.0f}"
                else:
                    mem_msg = ""

                print(
                    "┆   {:8d} | {:8d} | {:9.4f} | {:9.4f} | {:9.4f} | {:9.4f} | {:8.3f} | {:8.3f} | {:8.3f}{} | {:7.1f}ms".format(
                        index + 1,
                        batch_num_recycles,
                        float(loss_dict["loss"].detach().item()),
                        float(loss_dict["fape_loss"].detach().item()),
                        float(loss_dict["dist_loss"].detach().item()),
                        float(loss_dict["msa_loss"].detach().item()),
                        float(metrics["rmsd"].item()),
                        float(metrics["tm_score"].item()),
                        float(metrics["gdt_ts"].item()),
                        mem_msg,
                        (time.perf_counter() - t0) * 1000.0,
                    )
                )
        except RuntimeError as error:
            if ("CUDA out of memory" in str(error)) and (on_oom == "skip"):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_main_process:
                    print(f"[WARN][OOM] Eval batch {index} omitido. Limpié cache y sigo.")
                continue
            raise

    denom_loss = max(1, n_seen_batches)
    denom_metrics = max(1, n_metric_logs)
    return {
        "loss": running["loss"] / denom_loss,
        "fape_loss": running["fape_loss"] / denom_loss,
        "dist_loss": running["dist_loss"] / denom_loss,
        "msa_loss": running["msa_loss"] / denom_loss,
        "plddt_loss": running["plddt_loss"] / denom_loss,
        "torsion_loss": running["torsion_loss"] / denom_loss,
        "num_recycles": running["num_recycles"] / denom_loss,
        "rmsd_logged": running["rmsd_logged"] / denom_metrics if n_metric_logs > 0 else float("nan"),
        "tm_score_logged": running["tm_score_logged"] / denom_metrics if n_metric_logs > 0 else float("nan"),
        "gdt_ts_logged": running["gdt_ts_logged"] / denom_metrics if n_metric_logs > 0 else float("nan"),
        "n_seen_batches": n_seen_batches,
        "n_seen_samples": n_seen_samples,
        "n_metric_logs": n_metric_logs,
    }
