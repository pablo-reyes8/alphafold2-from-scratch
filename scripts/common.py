"""Shared utilities for the repository CLI scripts.

This module centralizes config loading, path resolution, dataset and dataloader
construction, model/loss instantiation, optimizer setup, synthetic validation
batches, and small formatting helpers so the scripts remain thin wrappers over
the project code.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from data.collate_proteins import collate_proteins
from data.dataloaders import FoldbenchProteinDataset, build_masked_msa_inputs
from data.loader_wrappers import build_protein_dataloader, build_train_eval_protein_dataloaders
from model.alphafold2 import AlphaFold2
from model.alphafold2_full_loss import AlphaFoldLoss
from training.autocast import build_amp_config
from training.ema import EMA
from training.scheduler_warmup import build_optimizer_and_scheduler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDEAL_BACKBONE_LOCAL = [
    [-1.458, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.547, 1.426, 0.0],
    [0.224, 2.617, 0.0],
]


def repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = repo_path(config_path)
    if path is None:
        raise ValueError("config_path must not be None")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    return payload


def nested_get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def choose_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_ideal_backbone_local(
    config: dict[str, Any],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    coords = nested_get(config, "geometry", "ideal_backbone_local", default=None)
    if coords is None:
        coords = DEFAULT_IDEAL_BACKBONE_LOCAL
    return torch.tensor(coords, dtype=torch.float32, device=device)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_total_steps(
    *,
    num_batches: int,
    epochs: int,
    grad_accum_steps: int = 1,
    max_batches: int | None = None,
) -> tuple[int, int]:
    capped_batches = num_batches if max_batches is None else min(num_batches, max_batches)
    capped_batches = max(1, int(capped_batches))
    steps_per_epoch = math.ceil(capped_batches / max(1, int(grad_accum_steps)))
    total_steps = max(1, steps_per_epoch * max(1, int(epochs)))
    return steps_per_epoch, total_steps


def build_dataset_from_config(
    config: dict[str, Any],
    *,
    manifest_csv: str | None = None,
    max_samples: int | None = None,
    verbose: bool = True,
) -> FoldbenchProteinDataset:
    data_cfg = nested_get(config, "data", default={}) or {}
    masked_msa_cfg = (
        nested_get(config, "data", "masked_msa", default=None)
        or nested_get(config, "data", "common", "masked_msa", default=None)
        or {}
    )
    manifest_override = manifest_csv is not None
    return FoldbenchProteinDataset(
        json_path=(
            None
            if manifest_override or not data_cfg.get("json_path")
            else str(repo_path(data_cfg.get("json_path")))
        ),
        msa_root=(
            None
            if manifest_override or not data_cfg.get("msa_root")
            else str(repo_path(data_cfg.get("msa_root")))
        ),
        cif_root=(
            None
            if manifest_override or not data_cfg.get("cif_root")
            else str(repo_path(data_cfg.get("cif_root")))
        ),
        manifest_csv=str(repo_path(manifest_csv or data_cfg.get("manifest_csv")))
        if (manifest_csv or data_cfg.get("manifest_csv"))
        else None,
        max_msa_seqs=int(data_cfg.get("max_msa_seqs", 128)),
        max_extra_msa_seqs=int(data_cfg.get("max_extra_msa_seqs", 256)),
        max_templates=int(data_cfg.get("max_templates", 4)),
        use_a3m_name=str(data_cfg.get("use_a3m_name", "cfdb_hits.a3m")),
        min_identity=float(data_cfg.get("min_identity", 0.85)),
        min_template_identity=float(data_cfg.get("min_template_identity", 0.80)),
        crop_size=(
            None
            if data_cfg.get("crop_size") in (None, "")
            else int(data_cfg.get("crop_size"))
        ),
        random_crop=bool(data_cfg.get("random_crop", True)),
        masked_msa_replace_fraction=float(
            data_cfg.get(
                "masked_msa_replace_fraction",
                masked_msa_cfg.get("replace_fraction", 0.15),
            )
        ),
        masked_msa_profile_prob=float(
            data_cfg.get(
                "masked_msa_profile_prob",
                masked_msa_cfg.get("profile_prob", 0.10),
            )
        ),
        masked_msa_same_prob=float(
            data_cfg.get(
                "masked_msa_same_prob",
                masked_msa_cfg.get("same_prob", 0.10),
            )
        ),
        masked_msa_uniform_prob=float(
            data_cfg.get(
                "masked_msa_uniform_prob",
                masked_msa_cfg.get("uniform_prob", 0.10),
            )
        ),
        single_sequence_mode=bool(data_cfg.get("single_sequence_mode", False)),
        max_samples=max_samples,
        verbose=verbose,
    )


def build_dataloader_from_config(
    dataset,
    config: dict[str, Any],
    *,
    batch_size: int | None = None,
    shuffle: bool | None = None,
) -> Any:
    loader_cfg = nested_get(config, "data", "loader", default={}) or {}
    return build_protein_dataloader(
        dataset,
        batch_size=int(batch_size or loader_cfg.get("batch_size", 1)),
        shuffle=bool(loader_cfg.get("shuffle", True) if shuffle is None else shuffle),
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        drop_last=bool(loader_cfg.get("drop_last", False)),
        collate_fn=collate_proteins,
    )


def build_train_eval_dataloaders_from_config(
    dataset,
    config: dict[str, Any],
    *,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    eval_size: int | None = None,
):
    loader_cfg = nested_get(config, "data", "loader", default={}) or {}
    resolved_eval_size = loader_cfg.get("eval_size", 0) if eval_size is None else eval_size
    resolved_eval_size = int(resolved_eval_size or 0)

    if resolved_eval_size <= 0:
        train_loader = build_dataloader_from_config(
            dataset,
            config,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return train_loader, None, {
            "train_indices": tuple(range(len(dataset))),
            "eval_indices": tuple(),
        }

    split = build_train_eval_protein_dataloaders(
        dataset,
        batch_size=int(batch_size or loader_cfg.get("batch_size", 1)),
        shuffle=bool(loader_cfg.get("shuffle", True) if shuffle is None else shuffle),
        collate_fn=collate_proteins,
        eval_size=resolved_eval_size,
        eval_shuffle=bool(loader_cfg.get("eval_shuffle", False)),
        split_seed=int(loader_cfg.get("split_seed", 42)),
        shuffle_before_split=bool(loader_cfg.get("shuffle_before_split", False)),
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        drop_last=bool(loader_cfg.get("drop_last", False)),
    )
    return split.train_loader, split.eval_loader, {
        "train_indices": split.train_indices,
        "eval_indices": split.eval_indices,
    }


def summarize_dataset(dataset) -> dict[str, Any]:
    dropped_reasons = Counter(reason for _, reason in getattr(dataset, "dropped", []))
    return {
        "valid_examples": int(len(dataset)),
        "dropped_examples": int(len(getattr(dataset, "dropped", []))),
        "drop_reasons_top": dropped_reasons.most_common(10),
        "preview_ids": dataset.df["query_name"].head(5).astype(str).tolist()
        if hasattr(dataset, "df") and not dataset.df.empty
        else [],
    }


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, list):
            summary[key] = {
                "type": "list",
                "len": len(value),
                "preview": value[:3],
            }
        else:
            summary[key] = value
    return summary


def build_model_from_config(
    config: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> AlphaFold2:
    model_cfg = deepcopy(nested_get(config, "model", default={}) or {})
    model = AlphaFold2(**model_cfg)
    return model.to(device)


def build_loss_from_config(
    config: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> AlphaFoldLoss:
    loss_cfg = deepcopy(nested_get(config, "loss", default={}) or {})
    criterion = AlphaFoldLoss(**loss_cfg)
    return criterion.to(device)


def build_optimizer_scheduler_from_config(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    num_batches: int,
    epochs: int,
    grad_accum_steps: int,
    max_batches: int | None = None,
):
    optimizer_cfg = nested_get(config, "optimizer", default={}) or {}
    scheduler_cfg = nested_get(config, "scheduler", default={}) or {}

    optimizer_name = str(optimizer_cfg.get("name", "AdamW"))
    scheduler_name = str(scheduler_cfg.get("name", "warmup_cosine"))

    if optimizer_name != "AdamW":
        raise NotImplementedError(f"Unsupported optimizer in CLI: {optimizer_name}")
    if scheduler_name != "warmup_cosine":
        raise NotImplementedError(f"Unsupported scheduler in CLI: {scheduler_name}")

    _, total_steps = compute_total_steps(
        num_batches=num_batches,
        epochs=epochs,
        grad_accum_steps=grad_accum_steps,
        max_batches=max_batches,
    )

    if scheduler_cfg.get("warmup_steps") is not None:
        warmup_steps = int(scheduler_cfg["warmup_steps"])
    else:
        warmup_fraction = float(scheduler_cfg.get("warmup_fraction", 0.0) or 0.0)
        warmup_steps = max(10, int(warmup_fraction * total_steps)) if warmup_fraction > 0.0 else 0

    return build_optimizer_and_scheduler(
        model=model,
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.95))),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
    )


def build_ema_from_config(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> EMA | None:
    ema_cfg = nested_get(config, "ema", default=None)
    if not isinstance(ema_cfg, dict):
        return None

    return EMA(
        model,
        decay=float(ema_cfg.get("decay", 0.999)),
        device=ema_cfg.get("device"),
        use_num_updates=bool(ema_cfg.get("use_num_updates", True)),
    )


def build_amp_runtime(
    config: dict[str, Any],
    *,
    device: str,
    amp_enabled: bool | None = None,
    amp_dtype: str | None = None,
) -> dict[str, Any]:
    trainer_cfg = nested_get(config, "trainer", default={}) or {}
    return build_amp_config(
        device=device,
        amp_enabled=bool(trainer_cfg.get("amp_enabled", True) if amp_enabled is None else amp_enabled),
        amp_dtype=str(trainer_cfg.get("amp_dtype", "bf16") if amp_dtype is None else amp_dtype),
    )


def make_synthetic_batch(
    config: dict[str, Any],
    *,
    batch_size: int = 1,
    msa_depth: int = 4,
    seq_len: int = 16,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    model_cfg = nested_get(config, "model", default={}) or {}
    masked_msa_cfg = (
        nested_get(config, "data", "masked_msa", default=None)
        or nested_get(config, "data", "common", "masked_msa", default=None)
        or {}
    )
    n_tokens = int(model_cfg.get("n_tokens", 27))
    n_torsions = int(model_cfg.get("n_torsions", 3))

    seq_tokens = torch.randint(1, n_tokens, (batch_size, seq_len), device=device)
    msa_tokens = torch.randint(1, n_tokens, (batch_size, msa_depth, seq_len), device=device)

    seq_mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)
    msa_mask = torch.ones(batch_size, msa_depth, seq_len, dtype=torch.float32, device=device)
    masked_msa_true = torch.zeros(batch_size, msa_depth, seq_len, dtype=torch.long, device=device)
    masked_msa_mask = torch.zeros(batch_size, msa_depth, seq_len, dtype=torch.float32, device=device)
    for batch_index in range(batch_size):
        masked_tokens, masked_true, masked_mask = build_masked_msa_inputs(
            msa_tokens[batch_index],
            msa_mask[batch_index],
            replace_fraction=float(masked_msa_cfg.get("replace_fraction", 0.15)),
            profile_prob=float(masked_msa_cfg.get("profile_prob", 0.10)),
            same_prob=float(masked_msa_cfg.get("same_prob", 0.10)),
            uniform_prob=float(masked_msa_cfg.get("uniform_prob", 0.10)),
        )
        msa_tokens[batch_index] = masked_tokens
        masked_msa_true[batch_index] = masked_true
        masked_msa_mask[batch_index] = masked_mask

    residue_axis = torch.arange(seq_len, dtype=torch.float32, device=device)
    coords_ca = torch.stack(
        [residue_axis, 0.1 * residue_axis, torch.zeros_like(residue_axis)],
        dim=-1,
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    coords_n = coords_ca + torch.tensor([-1.2, 0.4, 0.1], dtype=torch.float32, device=device)
    coords_c = coords_ca + torch.tensor([1.3, 0.5, -0.1], dtype=torch.float32, device=device)

    valid_res_mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)
    valid_backbone_mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)
    pair_mask = valid_res_mask[:, :, None] * valid_res_mask[:, None, :]

    torsion_true = torch.randn(batch_size, seq_len, n_torsions, 2, device=device)
    torsion_true = torsion_true / torch.linalg.norm(
        torsion_true,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    torsion_mask = torch.ones(batch_size, seq_len, n_torsions, dtype=torch.float32, device=device)

    return {
        "seq_tokens": seq_tokens,
        "msa_tokens": msa_tokens,
        "seq_mask": seq_mask,
        "msa_mask": msa_mask,
        "masked_msa_true": masked_msa_true,
        "masked_msa_mask": masked_msa_mask,
        "coords_n": coords_n,
        "coords_ca": coords_ca,
        "coords_c": coords_c,
        "valid_res_mask": valid_res_mask,
        "valid_backbone_mask": valid_backbone_mask,
        "pair_mask": pair_mask,
        "torsion_true": torsion_true,
        "torsion_mask": torsion_mask,
    }
