"""Distributed and data-parallel helpers for multi-GPU AlphaFold2 training.

This module centralizes process-group setup, distributed sampler construction,
DDP wrapping, rank-aware synchronization, and epoch-stat reduction so the core
training loop can stay close to the original single-device implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data.collate_proteins import collate_proteins
from data.loader_wrappers import resolve_train_eval_indices


LOSS_AVERAGE_KEYS = (
    "loss",
    "fape_loss",
    "dist_loss",
    "msa_loss",
    "plddt_loss",
    "torsion_loss",
    "num_recycles",
)
METRIC_AVERAGE_KEYS = (
    "rmsd_logged",
    "tm_score_logged",
    "gdt_ts_logged",
)
COUNT_KEYS = (
    "n_seen_batches",
    "n_optimizer_steps",
    "n_seen_samples",
    "n_metric_logs",
)


@dataclass
class ParallelContext:
    """Describe how the current process participates in parallel training."""

    mode: str = "none"
    distributed: bool = False
    model_parallel: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    primary_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    output_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    stage_devices: tuple[torch.device, ...] = field(default_factory=lambda: (torch.device("cpu"),))

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def parse_device_list(devices: str | list[str] | tuple[str, ...] | None) -> tuple[torch.device, ...]:
    """Normalize comma-separated or sequence device specifications."""
    if devices is None:
        return ()
    if isinstance(devices, str):
        parts = [part.strip() for part in devices.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in devices if str(part).strip()]
    return tuple(torch.device(part) for part in parts)


def _default_cuda_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _resolve_model_stage_devices_for_single_process(
    model_devices: str | list[str] | tuple[str, ...] | None,
) -> tuple[torch.device, ...]:
    parsed = parse_device_list(model_devices)
    if parsed:
        return parsed

    if torch.cuda.device_count() >= 2:
        return (torch.device("cuda:0"), torch.device("cuda:1"))

    return (_default_cuda_device(0),)


def _resolve_model_stage_devices_for_hybrid(local_rank: int, devices_per_replica: int) -> tuple[torch.device, ...]:
    if devices_per_replica < 2:
        raise ValueError("Hybrid parallelism requires at least two stage devices per replica.")
    if not torch.cuda.is_available():
        raise ValueError("Hybrid parallelism requires CUDA devices.")

    base_index = local_rank * devices_per_replica
    max_needed = base_index + devices_per_replica
    if torch.cuda.device_count() < max_needed:
        raise ValueError(
            f"Hybrid mode with local_rank={local_rank} and devices_per_replica={devices_per_replica} "
            f"needs at least {max_needed} visible CUDA devices, found {torch.cuda.device_count()}."
        )

    return tuple(torch.device(f"cuda:{idx}") for idx in range(base_index, max_needed))


def build_parallel_context(
    *,
    mode: str = "none",
    device: str | None = None,
    model_devices: str | list[str] | tuple[str, ...] | None = None,
    backend: str | None = None,
    devices_per_replica: int = 2,
) -> ParallelContext:
    """Create the runtime description for single-device, DDP, model, or hybrid mode."""
    mode = str(mode or "none").lower()
    if mode not in {"none", "ddp", "model", "hybrid"}:
        raise ValueError(f"Unsupported parallel mode: {mode}")

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    if mode in {"ddp", "hybrid"} and world_size <= 1:
        raise ValueError(
            f"parallel mode '{mode}' requires torchrun / WORLD_SIZE > 1. "
            f"Use `torchrun --nproc_per_node=... scripts/train_parallel.py ...`."
        )

    distributed = mode in {"ddp", "hybrid"}
    if distributed and not dist.is_initialized():
        chosen_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=chosen_backend, init_method="env://")

    if mode == "ddp":
        primary_device = torch.device(device) if device is not None else _default_cuda_device(local_rank)
        stage_devices = (primary_device,)
        model_parallel = False
    elif mode == "model":
        stage_devices = _resolve_model_stage_devices_for_single_process(model_devices)
        primary_device = stage_devices[0]
        model_parallel = len(stage_devices) > 1
    elif mode == "hybrid":
        stage_devices = _resolve_model_stage_devices_for_hybrid(local_rank, devices_per_replica)
        primary_device = stage_devices[0]
        model_parallel = True
    else:
        primary_device = torch.device(device) if device is not None else _default_cuda_device(0)
        stage_devices = (primary_device,)
        model_parallel = False

    if primary_device.type == "cuda":
        torch.cuda.set_device(primary_device)

    output_device = stage_devices[-1]
    return ParallelContext(
        mode=mode,
        distributed=distributed,
        model_parallel=model_parallel,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        primary_device=primary_device,
        output_device=output_device,
        stage_devices=stage_devices,
    )


def cleanup_parallel_context(context: ParallelContext | None) -> None:
    """Destroy the process group when this process initialized distributed mode."""
    if context is None:
        return
    if context.distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_parallel_train_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    context: ParallelContext | None = None,
    collate_fn=collate_proteins,
) -> DataLoader:
    """Build a regular or distributed dataloader depending on the context."""
    sampler = None
    if context is not None and context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        collate_fn=collate_fn,
    )


def build_parallel_train_eval_loaders(
    dataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    context: ParallelContext | None = None,
    collate_fn=collate_proteins,
    eval_size: int = 1,
    eval_shuffle: bool = False,
    split_seed: int = 42,
    shuffle_before_split: bool = False,
):
    """Build deterministic train/eval parallel dataloaders from one dataset."""
    resolved_eval_size = int(eval_size or 0)
    if resolved_eval_size <= 0:
        train_loader = build_parallel_train_loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            context=context,
            collate_fn=collate_fn,
        )
        return train_loader, None, tuple(range(len(dataset))), tuple()

    train_indices, eval_indices = resolve_train_eval_indices(
        len(dataset),
        eval_size=resolved_eval_size,
        split_seed=split_seed,
        shuffle_before_split=shuffle_before_split,
    )
    train_loader = build_parallel_train_loader(
        Subset(dataset, list(train_indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        context=context,
        collate_fn=collate_fn,
    )
    eval_loader = build_parallel_train_loader(
        Subset(dataset, list(eval_indices)),
        batch_size=batch_size,
        shuffle=eval_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        context=context,
        collate_fn=collate_fn,
    )
    return train_loader, eval_loader, train_indices, eval_indices


def wrap_model_for_data_parallel(
    model: torch.nn.Module,
    context: ParallelContext | None,
    *,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = False,
) -> torch.nn.Module:
    """Wrap the model in DDP when the context requests distributed execution."""
    if context is None or not context.distributed:
        return model

    if context.model_parallel:
        return DDP(
            model,
            device_ids=None,
            output_device=None,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=broadcast_buffers,
        )

    device_ids = [context.primary_device.index] if context.primary_device.type == "cuda" else None
    output_device = context.primary_device.index if context.primary_device.type == "cuda" else None
    return DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


def sync_epoch_stats(epoch_stats: dict[str, Any], context: ParallelContext | None) -> dict[str, Any]:
    """Average epoch statistics across ranks without changing the single-device API."""
    if context is None or not context.distributed or context.world_size <= 1:
        return epoch_stats

    device = context.output_device if context.output_device.type == "cuda" else torch.device("cpu")
    n_seen_batches = float(epoch_stats.get("n_seen_batches", 0))
    n_metric_logs = float(epoch_stats.get("n_metric_logs", 0))

    values = []
    for key in LOSS_AVERAGE_KEYS:
        values.append(float(epoch_stats.get(key, 0.0)) * n_seen_batches)
    for key in METRIC_AVERAGE_KEYS:
        if n_metric_logs > 0:
            values.append(float(epoch_stats.get(key, 0.0)) * n_metric_logs)
        else:
            values.append(0.0)
    for key in COUNT_KEYS:
        values.append(float(epoch_stats.get(key, 0)))

    payload = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)

    result = dict(epoch_stats)
    cursor = 0
    global_batch_count = float(payload[len(LOSS_AVERAGE_KEYS) + len(METRIC_AVERAGE_KEYS) + 0].item())
    global_metric_count = float(payload[len(LOSS_AVERAGE_KEYS) + len(METRIC_AVERAGE_KEYS) + 3].item())

    for key in LOSS_AVERAGE_KEYS:
        numerator = float(payload[cursor].item())
        result[key] = numerator / max(1.0, global_batch_count)
        cursor += 1

    for key in METRIC_AVERAGE_KEYS:
        numerator = float(payload[cursor].item())
        result[key] = numerator / global_metric_count if global_metric_count > 0 else float("nan")
        cursor += 1

    for key in COUNT_KEYS:
        result[key] = int(round(float(payload[cursor].item())))
        cursor += 1

    return result


def maybe_barrier(context: ParallelContext | None) -> None:
    """Synchronize ranks only when distributed execution is active."""
    if context is None:
        return
    if context.distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
