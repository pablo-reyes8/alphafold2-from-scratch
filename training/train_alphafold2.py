"""High-level training orchestration for AlphaFold2-like experiments.

This module coordinates training, optional evaluation, checkpointing, and
metric selection while reusing the lower-level epoch utilities.
"""

from __future__ import annotations

import os
import time

import torch

from training.checkpoints import (
    get_resume_state,
    load_checkpoint,
    maybe_save_best_and_last,
    save_checkpoint,
)
from training.colab_utils import fmt_hms, rule
from training.eval_one_epoch import eval_one_epoch
from training.train_one_epoch import train_one_epoch
from training.train_parallel.data_parallel import maybe_barrier, sync_epoch_stats


BASE_MONITOR_NAMES = {
    "loss",
    "fape_loss",
    "dist_loss",
    "msa_loss",
    "plddt_loss",
    "torsion_loss",
    "rmsd_logged",
    "tm_score_logged",
    "gdt_ts_logged",
}


def _valid_monitor_names() -> set[str]:
    names = set(BASE_MONITOR_NAMES)
    names.update({f"train_{name}" for name in BASE_MONITOR_NAMES})
    names.update({f"eval_{name}" for name in BASE_MONITOR_NAMES})
    return names


def _resolve_monitor_stats(
    monitor_name: str,
    *,
    train_stats: dict,
    eval_stats: dict | None,
) -> tuple[dict, str]:
    if monitor_name.startswith("train_"):
        return train_stats, monitor_name[len("train_"):]

    if monitor_name.startswith("eval_"):
        if eval_stats is None:
            raise ValueError(
                f"monitor_name='{monitor_name}' requires eval_stats but no eval_loader was provided."
            )
        return eval_stats, monitor_name[len("eval_"):]

    if eval_stats is not None and monitor_name in eval_stats:
        return eval_stats, monitor_name

    return train_stats, monitor_name


def _prefixed_stats(prefix: str, stats: dict | None) -> dict:
    if stats is None:
        return {}
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def train_alphafold2(
    *,
    model,
    train_loader,
    eval_loader=None,
    optimizer,
    criterion,
    scheduler=None,
    ema=None,
    scaler=None,
    device: str = "cuda",
    epochs: int = 10,
    start_epoch: int = 0,
    global_step: int = 0,
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    grad_clip: float | None = 1.0,
    grad_accum_steps: int = 1,
    log_every: int = 10,
    log_grad_norm: bool = True,
    log_mem: bool = False,
    max_batches: int | None = None,
    on_oom: str = "skip",
    ideal_backbone_local: torch.Tensor | None = None,
    num_recycles: int = 0,
    stochastic_recycling: bool = False,
    max_recycles: int | None = None,
    parallel_context=None,
    ckpt_dir: str = "checkpoints",
    run_name: str = "alphafold2",
    save_every: int = 1,
    save_last: bool = True,
    eval_every: int = 1,
    monitor_name: str = "loss",
    monitor_mode: str = "min",
    best_metric: float | None = None,
    config: dict | None = None,
    resume_path: str | None = None,
    strict_resume: bool = True,
    restore_rng_state: bool = False,
    drive_ckpt_dir: str | None = None,
    copy_fixed_to_drive: bool = True,
    fixed_drive_name: str = "latest_alphafold2.pt",
):
    """Run the configured training loop with optional evaluation after epochs."""
    os.makedirs(ckpt_dir, exist_ok=True)
    is_main_process = True if parallel_context is None else bool(parallel_context.is_main_process)
    recycle_upper = num_recycles if max_recycles is None else int(max_recycles)
    recycle_mode = (
        f"uniform[0,{recycle_upper}]" if stochastic_recycling else f"fixed={num_recycles}"
    )

    if resume_path is not None and os.path.exists(resume_path):
        ckpt = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            map_location=device,
            strict=strict_resume,
            load_optimizer_state=True,
            restore_rng_state=restore_rng_state,
        )
        resume_state = get_resume_state(ckpt)
        start_epoch = int(resume_state["epoch"]) + 1
        global_step = int(resume_state["global_step"])
        best_metric = resume_state["best_metric"]

        if is_main_process:
            print(f"[RESUME] {resume_path}")
            print(
                f"[RESUME] start_epoch={start_epoch} | global_step={global_step} | best_metric={best_metric}"
            )

    valid_monitor_names = _valid_monitor_names()
    if monitor_name not in valid_monitor_names:
        raise ValueError(
            f"monitor_name='{monitor_name}' no es válido. Usa uno de: {sorted(valid_monitor_names)}"
        )

    ema_decay_val = getattr(ema, "decay", None)
    ema_str = (
        f"{ema_decay_val:.6f}"
        if isinstance(ema_decay_val, (float, int))
        else ("on" if ema is not None else "off")
    )

    if is_main_process:
        lr_now = optimizer.param_groups[0]["lr"]
        print(rule())
        print(f"AlphaFold2-like run: {run_name}")
        print(
            f"Device: {device} | AMP: {amp_enabled}({amp_dtype}) | EMA: {ema_str} | "
            f"epochs: {epochs} | lr_now: {lr_now:.2e} | grad_clip: {grad_clip} | recycles: {recycle_mode}"
        )
        print(
            f"Monitor: {monitor_name} ({monitor_mode}) | start_epoch: {start_epoch} | global_step: {global_step}"
        )
        if eval_loader is not None:
            print(f"Eval loader: enabled | eval_every: {max(1, int(eval_every))}")
        print(rule())
        print(
            f"{'ep':>3} | {'step':>8} | {'loss':>10} | {'fape':>10} | {'dist':>10} | {'msa':>10} | "
            f"{'plddt':>10} | {'tors':>10} | {'rmsd':>8} | {'tm':>8} | {'gdt':>8} | {'lr':>9} | {'time':>8}"
        )
        print(rule())

    total_time = 0.0
    train_stats = None
    eval_stats = None
    combined_metrics: dict[str, float] = {}
    eval_every = max(1, int(eval_every))

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        sampler = getattr(train_loader, "sampler", None)
        if parallel_context is not None and parallel_context.distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        train_stats, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            scheduler=scheduler,
            ema=ema,
            ema_target=model,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            max_batches=max_batches,
            log_every=log_every,
            log_grad_norm=log_grad_norm,
            log_mem=log_mem,
            on_oom=on_oom,
            global_step=global_step,
            ideal_backbone_local=ideal_backbone_local,
            num_recycles=num_recycles,
            stochastic_recycling=stochastic_recycling,
            max_recycles=max_recycles,
            is_main_process=is_main_process,
        )
        train_stats = sync_epoch_stats(train_stats, parallel_context)

        eval_stats = None
        if eval_loader is not None and ((epoch - start_epoch) % eval_every) == 0:
            eval_stats = eval_one_epoch(
                model=model,
                dataloader=eval_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                max_batches=max_batches,
                log_every=log_every,
                log_mem=log_mem,
                on_oom=on_oom,
                ideal_backbone_local=ideal_backbone_local,
                num_recycles=num_recycles,
                stochastic_recycling=stochastic_recycling,
                max_recycles=max_recycles,
                is_main_process=is_main_process,
            )
            eval_stats = sync_epoch_stats(eval_stats, parallel_context)

        sec = time.time() - t0
        total_time += sec
        lr_now = optimizer.param_groups[0]["lr"]

        combined_metrics = {}
        combined_metrics.update(_prefixed_stats("train", train_stats))
        combined_metrics.update(_prefixed_stats("eval", eval_stats))
        combined_metrics.update(eval_stats if eval_stats is not None else train_stats)

        if is_main_process:
            print(
                f"{epoch:3d} | {global_step:8d} | "
                f"{train_stats['loss']:10.5f} | {train_stats['fape_loss']:10.5f} | {train_stats['dist_loss']:10.5f} | {train_stats['msa_loss']:10.5f} | "
                f"{train_stats['plddt_loss']:10.5f} | {train_stats['torsion_loss']:10.5f} | "
                f"{train_stats['rmsd_logged']:8.3f} | {train_stats['tm_score_logged']:8.3f} | {train_stats['gdt_ts_logged']:8.3f} | "
                f"{lr_now:9.2e} | {fmt_hms(sec):>8}"
            )
            if eval_stats is not None:
                print(
                    "    eval -> "
                    f"loss: {eval_stats['loss']:.5f} | fape: {eval_stats['fape_loss']:.5f} | "
                    f"dist: {eval_stats['dist_loss']:.5f} | msa: {eval_stats['msa_loss']:.5f} | plddt: {eval_stats['plddt_loss']:.5f} | "
                    f"tors: {eval_stats['torsion_loss']:.5f} | rmsd: {eval_stats['rmsd_logged']:.3f} | "
                    f"tm: {eval_stats['tm_score_logged']:.3f} | gdt: {eval_stats['gdt_ts_logged']:.3f}"
                )

        monitor_stats, monitor_key = _resolve_monitor_stats(
            monitor_name,
            train_stats=train_stats,
            eval_stats=eval_stats,
        )
        current_metric = float(monitor_stats[monitor_key])

        if is_main_process:
            best_metric, improved = maybe_save_best_and_last(
                save_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                epoch=epoch,
                global_step=global_step,
                current_metric=current_metric,
                best_metric=best_metric,
                metric_name=monitor_name,
                mode=monitor_mode,
                val_metrics=combined_metrics,
                train_metrics=train_stats,
                eval_metrics=eval_stats,
                config=config,
            )

            if improved:
                print(f"└─ [BEST] improved {monitor_name} -> {best_metric:.6f}")

            if (save_every is not None) and (save_every > 0) and ((epoch % save_every == 0) or (epoch == epochs - 1)):
                ckpt_path = os.path.join(ckpt_dir, f"{run_name}_e{epoch:03d}.pt")
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    monitor_name=monitor_name,
                    metrics=combined_metrics,
                    train_metrics=train_stats,
                    eval_metrics=eval_stats,
                    config=config,
                    save_optimizer_state=True,
                    save_rng_state=True,
                )

                print(f"└─ [CKPT] saved → {ckpt_path}")

                if copy_fixed_to_drive and drive_ckpt_dir:
                    from training.colab_utils import copy_ckpt_to_drive_fixed

                    copy_ckpt_to_drive_fixed(
                        src_path=ckpt_path,
                        drive_dir=drive_ckpt_dir,
                        fixed_name=fixed_drive_name,
                    )

        maybe_barrier(parallel_context)

    if is_main_process and save_last and (train_stats is not None):
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_last_manual.pt")
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            epoch=epochs - 1,
            global_step=global_step,
            best_metric=best_metric,
            monitor_name=monitor_name,
            metrics=combined_metrics,
            train_metrics=train_stats,
            eval_metrics=eval_stats,
            config=config,
            save_optimizer_state=True,
            save_rng_state=True,
        )

        print(f"└─ [CKPT] final saved → {ckpt_path}")

        if copy_fixed_to_drive and drive_ckpt_dir:
            from training.colab_utils import copy_ckpt_to_drive_fixed

            copy_ckpt_to_drive_fixed(
                src_path=ckpt_path,
                drive_dir=drive_ckpt_dir,
                fixed_name=fixed_drive_name,
            )

    maybe_barrier(parallel_context)

    if is_main_process:
        print(rule())
        print(f"Entrenamiento finalizado en {fmt_hms(total_time)}")
        print(rule())

    return {
        "global_step": global_step,
        "best_metric": best_metric,
        "last_train_stats": train_stats,
        "last_eval_stats": eval_stats,
    }
