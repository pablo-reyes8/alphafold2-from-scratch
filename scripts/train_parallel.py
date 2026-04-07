"""Launch parallel AlphaFold2 training with DDP, model parallelism, or both.

This CLI keeps the existing YAML-driven training stack, but adds runtime
wrapping for three multi-GPU strategies:

- ``ddp``: one full model replica per GPU via ``torchrun``
- ``model``: one two-stage model split across two devices in a single process
- ``hybrid``: DDP over multi-device replicas; requires at least four GPUs
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    build_amp_runtime,
    build_dataset_from_config,
    build_ema_from_config,
    build_ideal_backbone_local,
    build_loss_from_config,
    build_model_from_config,
    build_optimizer_scheduler_from_config,
    count_trainable_parameters,
    load_yaml_config,
    nested_get,
)
from training.seeds import seed_everything
from training.train_alphafold2 import train_alphafold2
from training.train_parallel.data_parallel import (
    build_parallel_context,
    build_parallel_train_eval_loaders,
    cleanup_parallel_context,
    wrap_model_for_data_parallel,
)
from training.train_parallel.model_parallel import build_model_parallel_wrapper


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AlphaFold2-like model with DDP, model parallelism, or hybrid parallelism."
    )
    parser.add_argument("--config", type=str, default="config/experiments/af2_poc.yaml")
    parser.add_argument("--manifest-csv", type=str, default=None)
    parser.add_argument("--parallel-mode", choices=("ddp", "model", "hybrid"), default="ddp")
    parser.add_argument("--device", type=str, default=None, help="Primary device for non-model-parallel runs.")
    parser.add_argument(
        "--model-devices",
        type=str,
        default=None,
        help="Comma-separated stage devices for model mode, for example 'cuda:0,cuda:1'.",
    )
    parser.add_argument("--devices-per-replica", type=int, default=2)
    parser.add_argument("--backend", type=str, default=None, help="Distributed backend, for example nccl or gloo.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-process batch size.")
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default=None)
    parser.add_argument("--num-recycles", type=int, default=None)
    parser.add_argument("--stochastic-recycling", action="store_true")
    parser.add_argument("--max-recycles", type=int, default=None)
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--broadcast-buffers", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    trainer_cfg = dict(config.get("trainer", {}))
    loader_cfg = dict(nested_get(config, "data", "loader", default={}) or {})

    context = build_parallel_context(
        mode=args.parallel_mode,
        device=args.device,
        model_devices=args.model_devices,
        backend=args.backend,
        devices_per_replica=args.devices_per_replica,
    )

    try:
        seed = int(config.get("seed", 42) if args.seed is None else args.seed)
        seed_everything(seed + context.rank, deterministic=args.deterministic)

        dataset = build_dataset_from_config(
            config,
            manifest_csv=args.manifest_csv,
            max_samples=1 if args.dry_run and args.max_samples is None else args.max_samples,
            verbose=context.is_main_process,
        )
        if len(dataset) == 0:
            raise ValueError("Dataset resolved zero valid examples. Check the manifest and local data paths.")

        batch_size = 1 if args.dry_run and args.batch_size is None else int(
            args.batch_size or loader_cfg.get("batch_size", 1)
        )
        loader, eval_loader, split_info_train, split_info_eval = build_parallel_train_eval_loaders(
            dataset,
            batch_size=batch_size,
            shuffle=False if args.dry_run else bool(loader_cfg.get("shuffle", True)),
            num_workers=int(loader_cfg.get("num_workers", 0)),
            pin_memory=bool(loader_cfg.get("pin_memory", False)),
            drop_last=bool(loader_cfg.get("drop_last", False)),
            context=context,
            eval_size=0 if args.dry_run else int(loader_cfg.get("eval_size", 0) if args.eval_size is None else args.eval_size),
            eval_shuffle=bool(loader_cfg.get("eval_shuffle", False)),
            split_seed=int(loader_cfg.get("split_seed", 42)),
            shuffle_before_split=bool(loader_cfg.get("shuffle_before_split", False)),
        )

        if context.model_parallel:
            model = build_model_from_config(config, device="cpu")
            model = build_model_parallel_wrapper(model, context.stage_devices)
        else:
            model = build_model_from_config(config, device=str(context.primary_device))

        model = wrap_model_for_data_parallel(
            model,
            context,
            find_unused_parameters=args.find_unused_parameters,
            broadcast_buffers=args.broadcast_buffers,
        )

        criterion = build_loss_from_config(config, device=str(context.output_device))
        ideal_backbone_local = build_ideal_backbone_local(config, device=str(context.output_device))

        epochs = 1 if args.dry_run else int(trainer_cfg.get("epochs", 1) if args.epochs is None else args.epochs)
        grad_accum_steps = int(trainer_cfg.get("grad_accum_steps", 1))
        max_batches = 1 if args.dry_run else args.max_batches

        optimizer, scheduler = build_optimizer_scheduler_from_config(
            model,
            config,
            num_batches=len(loader),
            epochs=epochs,
            grad_accum_steps=grad_accum_steps,
            max_batches=max_batches,
        )

        ema = None if args.no_ema else build_ema_from_config(model, config)
        amp_runtime = build_amp_runtime(
            config,
            device=str(context.output_device),
            amp_enabled=False if args.no_amp else None,
            amp_dtype=args.amp_dtype,
        )

        num_recycles = int(trainer_cfg.get("num_recycles", 0) if args.num_recycles is None else args.num_recycles)
        stochastic_recycling = bool(trainer_cfg.get("stochastic_recycling", False) or args.stochastic_recycling)
        max_recycles = args.max_recycles
        if max_recycles is None and "max_recycles" in trainer_cfg:
            max_recycles = int(trainer_cfg["max_recycles"])

        on_oom = str(trainer_cfg.get("on_oom", "skip"))
        if context.distributed and on_oom == "skip":
            if context.is_main_process:
                print("[WARN] Switching on_oom from 'skip' to 'raise' for distributed training.")
            on_oom = "raise"

        if context.is_main_process:
            print(
                {
                    "parallel_mode": context.mode,
                    "distributed": context.distributed,
                    "world_size": context.world_size,
                    "rank": context.rank,
                    "stage_devices": [str(device) for device in context.stage_devices],
                    "dataset_examples": len(dataset),
                    "train_examples": len(split_info_train),
                    "eval_examples": len(split_info_eval),
                    "loader_batch_size_per_process": batch_size,
                    "epochs": epochs,
                    "max_batches": max_batches,
                    "trainable_parameters": count_trainable_parameters(model),
                    "num_recycles": num_recycles,
                    "stochastic_recycling": stochastic_recycling,
                    "max_recycles": max_recycles,
                    "amp_enabled_effective": amp_runtime["amp_enabled"],
                    "amp_dtype_effective": str(amp_runtime["amp_dtype_effective"]),
                }
            )

        train_alphafold2(
            model=model,
            train_loader=loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            ema=ema,
            scaler=amp_runtime["scaler"],
            device=str(context.output_device),
            epochs=epochs,
            amp_enabled=bool(amp_runtime["amp_enabled"]),
            amp_dtype=str(args.amp_dtype or trainer_cfg.get("amp_dtype", "bf16")),
            grad_clip=trainer_cfg.get("grad_clip", 1.0),
            grad_accum_steps=grad_accum_steps,
            log_every=int(trainer_cfg.get("log_every", 10)),
            log_grad_norm=bool(trainer_cfg.get("log_grad_norm", True)),
            log_mem=bool(trainer_cfg.get("log_mem", False)),
            max_batches=max_batches,
            on_oom=on_oom,
            ideal_backbone_local=ideal_backbone_local,
            num_recycles=num_recycles,
            stochastic_recycling=stochastic_recycling,
            max_recycles=max_recycles,
            ckpt_dir=str(trainer_cfg.get("ckpt_dir", "checkpoints")),
            run_name=str(trainer_cfg.get("run_name", "alphafold2")),
            save_every=int(trainer_cfg.get("save_every", 1)),
            save_last=bool(trainer_cfg.get("save_last", True)),
            eval_every=int(trainer_cfg.get("eval_every", 1)),
            monitor_name=str(trainer_cfg.get("monitor_name", "loss")),
            monitor_mode=str(trainer_cfg.get("monitor_mode", "min")),
            config=config,
            resume_path=args.resume_path,
            parallel_context=context,
        )
    finally:
        cleanup_parallel_context(context)


if __name__ == "__main__":
    main()
