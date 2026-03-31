"""High-level training orchestration for the AlphaFold2-like model.

This file coordinates resuming, epoch iteration, checkpointing, logging, and
interaction with the per-epoch training routine. It is the main entry point for
scripted training outside the notebook environment.
"""

import torch 
import torch.nn as nn 
from training.chekpoints import * 
from training.seeds import * 
from training.scheduler_warmup import * 
from training.ema import * 
from training.train_one_epoch import *
from training.colab_utils import *

def train_alphafold2(
    *,
    model,
    train_loader,
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

    # checkpoint / monitoring
    ckpt_dir: str = "checkpoints",
    run_name: str = "alphafold2",
    save_every: int = 1,
    save_last: bool = True,
    monitor_name: str = "loss",        # e.g. "loss", "rmsd_logged", "tm_score_logged", "gdt_ts_logged"
    monitor_mode: str = "min",         # "min" for loss/rmsd, "max" for tm/gdt
    best_metric: float | None = None,
    config: dict | None = None,

    # optional resume
    resume_path: str | None = None,
    strict_resume: bool = True,
    restore_rng_state: bool = False,

    # drive mirror
    drive_ckpt_dir: str | None = None,
    copy_fixed_to_drive: bool = True,
    fixed_drive_name: str = "latest_alphafold2.pt"):

    """
    High-level training orchestrator for AlphaFold2 model.

    Assumes model / optimizer / scheduler / ema / scaler / criterion
    are already created externally.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    recycle_upper = num_recycles if max_recycles is None else int(max_recycles)
    recycle_mode = (
        f"uniform[0,{recycle_upper}]"
        if stochastic_recycling
        else f"fixed={num_recycles}"
    )

    # Optional resume
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
            restore_rng_state=restore_rng_state)


        resume_state = get_resume_state(ckpt)
        start_epoch = int(resume_state["epoch"]) + 1
        global_step = int(resume_state["global_step"])
        best_metric = resume_state["best_metric"]

        print(f"[RESUME] {resume_path}")
        print(f"[RESUME] start_epoch={start_epoch} | global_step={global_step} | best_metric={best_metric}")

    # validar monitor_name
    valid_monitor_names = {
        "loss",
        "fape_loss",
        "dist_loss",
        "plddt_loss",
        "torsion_loss",
        "rmsd_logged",
        "tm_score_logged",
        "gdt_ts_logged"}

    if monitor_name not in valid_monitor_names:
        raise ValueError(
            f"monitor_name='{monitor_name}' no es válido. "
            f"Usa uno de: {sorted(valid_monitor_names)}")

    # Header
    ema_decay_val = getattr(ema, "decay", None)
    ema_str = (
        f"{ema_decay_val:.6f}"
        if isinstance(ema_decay_val, (float, int))
        else ("on" if ema is not None else "off"))

    lr_now = optimizer.param_groups[0]["lr"]

    print(rule())
    print(f"AlphaFold2-like run: {run_name}")
    print(
        f"Device: {device} | AMP: {amp_enabled}({amp_dtype}) | EMA: {ema_str} | "
        f"epochs: {epochs} | lr_now: {lr_now:.2e} | grad_clip: {grad_clip} | "
        f"recycles: {recycle_mode}")
    print(
        f"Monitor: {monitor_name} ({monitor_mode}) | "
        f"start_epoch: {start_epoch} | global_step: {global_step}")
    print(rule())
    print(
        f"{'ep':>3} | {'step':>8} | {'loss':>10} | {'fape':>10} | {'dist':>10} | "
        f"{'plddt':>10} | {'tors':>10} | {'rmsd':>8} | {'tm':>8} | {'gdt':>8} | {'lr':>9} | {'time':>8}")
    print(rule())

    # --------------------------------------------------
    # Train loop
    # --------------------------------------------------
    total_time = 0.0
    train_stats = None

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

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
            max_recycles=max_recycles)

        sec = time.time() - t0
        total_time += sec
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:3d} | {global_step:8d} | "
            f"{train_stats['loss']:10.5f} | {train_stats['fape_loss']:10.5f} | {train_stats['dist_loss']:10.5f} | "
            f"{train_stats['plddt_loss']:10.5f} | {train_stats['torsion_loss']:10.5f} | "
            f"{train_stats['rmsd_logged']:8.3f} | {train_stats['tm_score_logged']:8.3f} | {train_stats['gdt_ts_logged']:8.3f} | "
            f"{lr_now:9.2e} | {fmt_hms(sec):>8}")

        # Checkpointing
        current_metric = float(train_stats[monitor_name])

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
            val_metrics=train_stats,   # aquí son train_stats; luego podrías pasar val_stats
            config=config)

        if improved:
            print(f"└─ [BEST] improved {monitor_name} -> {best_metric:.6f}")

        # optional periodic named snapshots
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
                metrics=train_stats,
                config=config,
                save_optimizer_state=True,
                save_rng_state=True)

            print(f"└─ [CKPT] saved → {ckpt_path}")

            if copy_fixed_to_drive and drive_ckpt_dir:
                copy_ckpt_to_drive_fixed(
                    src_path=ckpt_path,
                    drive_dir=drive_ckpt_dir,
                    fixed_name=fixed_drive_name)

    # Final save_last
    if save_last and (train_stats is not None):
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
            metrics=train_stats,
            config=config,
            save_optimizer_state=True,
            save_rng_state=True)

        print(f"└─ [CKPT] final saved → {ckpt_path}")

        if copy_fixed_to_drive and drive_ckpt_dir:
            copy_ckpt_to_drive_fixed(
                src_path=ckpt_path,
                drive_dir=drive_ckpt_dir,
                fixed_name=fixed_drive_name)

    print(rule())
    print(f"Entrenamiento finalizado en {fmt_hms(total_time)}")
    print(rule())

    return {
        "global_step": global_step,
        "best_metric": best_metric,
        "last_train_stats": train_stats}
