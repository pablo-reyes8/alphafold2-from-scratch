"""Exercise the lightweight repository CLIs and config plumbing exposed under scripts/."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

import scripts.common as scripts_common
from scripts.common import (
    build_ideal_backbone_local,
    build_loss_from_config,
    build_model_from_config,
    compute_total_steps,
    load_yaml_config,
    make_synthetic_batch,
)
from scripts.inspect_data import parse_args as parse_inspect_args
from scripts.prepare_data import parse_args as parse_prepare_args
from scripts.train_model import parse_args as parse_train_args
from scripts.train_parallel import parse_args as parse_train_parallel_args
from scripts.validate_model import parse_args as parse_validate_args, run_forward_smoke


def _tiny_config() -> dict:
    config = load_yaml_config("config/experiments/af2_poc.yaml")
    config = deepcopy(config)
    config["model"]["num_evoformer_blocks"] = 1
    config["model"]["num_structure_blocks"] = 1
    config["model"]["transition_expansion_evoformer"] = 2
    config["model"]["transition_expansion_structure"] = 2
    return config


def test_compute_total_steps_respects_grad_accumulation():
    steps_per_epoch, total_steps = compute_total_steps(
        num_batches=7,
        epochs=3,
        grad_accum_steps=2,
        max_batches=5,
    )
    assert steps_per_epoch == 3
    assert total_steps == 9


def test_common_builders_support_forward_and_loss_smoke():
    config = _tiny_config()
    device = "cpu"

    model = build_model_from_config(config, device=device)
    criterion = build_loss_from_config(config, device=device)
    batch = make_synthetic_batch(config, batch_size=1, msa_depth=2, seq_len=8, device=device)
    ideal_backbone_local = build_ideal_backbone_local(config, device=device)

    with torch.no_grad():
        outputs = model(
            seq_tokens=batch["seq_tokens"],
            msa_tokens=batch["msa_tokens"],
            seq_mask=batch["seq_mask"],
            msa_mask=batch["msa_mask"],
            ideal_backbone_local=ideal_backbone_local,
            num_recycles=1,
        )
        losses = criterion(outputs, batch)

    assert outputs["distogram_logits"].shape[:3] == (1, 8, 8)
    assert torch.isfinite(outputs["plddt"]).all()
    assert torch.isfinite(losses["loss"])


def test_build_dataset_from_config_does_not_rewrite_paths_when_manifest_is_overridden(monkeypatch):
    captured = {}

    class RecordingDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = _tiny_config()
    monkeypatch.setattr(scripts_common, "FoldbenchProteinDataset", RecordingDataset)

    scripts_common.build_dataset_from_config(
        config,
        manifest_csv="data/smoke_one_manifest.csv",
        max_samples=1,
        verbose=False,
    )

    assert Path(captured["manifest_csv"]).as_posix().endswith("data/smoke_one_manifest.csv")
    assert captured["msa_root"] is None
    assert captured["cif_root"] is None
    assert captured["json_path"] is None


def test_validate_forward_smoke_returns_finite_summary():
    args = parse_validate_args(
        [
            "forward-smoke",
            "--config",
            "config/experiments/af2_poc.yaml",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--msa-depth",
            "2",
            "--seq-len",
            "8",
            "--num-recycles",
            "1",
        ]
    )
    summary = run_forward_smoke(args)

    assert summary["num_recycles"] == 1
    assert summary["output_shapes"]["distogram_logits"] == [1, 8, 8, 64]
    assert all(summary["finite_outputs"].values())
    assert summary["losses"]["loss"] >= 0.0


def test_prepare_cli_parses_bootstrap_command():
    args = parse_prepare_args(
        [
            "bootstrap",
            "--data-config",
            "config/data/foldbench_subset.yaml",
            "--experiment-config",
            "config/experiments/af2_poc.yaml",
            "--skip-download",
            "--skip-manifest",
            "--max-samples",
            "1",
        ]
    )
    assert args.command == "bootstrap"
    assert args.skip_download is True
    assert args.max_samples == 1


def test_prepare_cli_parses_train_eval_loader_smoke_command():
    args = parse_prepare_args(
        [
            "train-eval-loader-smoke",
            "--config",
            "config/experiments/af2_low_vram.yaml",
            "--manifest-csv",
            "data/showcase_manifest.csv",
            "--eval-size",
            "1",
        ]
    )
    assert args.command == "train-eval-loader-smoke"
    assert args.eval_size == 1


def test_inspect_cli_parses_protein_3d_command():
    args = parse_inspect_args(
        [
            "protein-3d",
            "--cif-path",
            "data/example.cif",
            "--chain-id",
            "A",
            "--output",
            "artifacts/protein.png",
        ]
    )
    assert args.command == "protein-3d"
    assert args.chain_id == "A"


def test_validate_cli_parses_pytest_command():
    args = parse_validate_args(
        [
            "pytest",
            "--target",
            "tests/test_full_loss.py",
            "--pytest-arg=-q",
        ]
    )
    assert args.command == "pytest"
    assert args.target == ["tests/test_full_loss.py"]
    assert args.pytest_arg == ["-q"]


def test_train_cli_parses_training_overrides():
    args = parse_train_args(
        [
            "--config",
            "config/experiments/af2_poc.yaml",
            "--device",
            "cpu",
            "--dry-run",
            "--eval-size",
            "1",
            "--stochastic-recycling",
            "--max-recycles",
            "3",
        ]
    )
    assert args.device == "cpu"
    assert args.dry_run is True
    assert args.eval_size == 1
    assert args.stochastic_recycling is True
    assert args.max_recycles == 3


def test_train_parallel_cli_parses_model_parallel_overrides():
    args = parse_train_parallel_args(
        [
            "--config",
            "config/experiments/af2_poc.yaml",
            "--parallel-mode",
            "model",
            "--model-devices",
            "cuda:0,cuda:1",
            "--batch-size",
            "1",
            "--eval-size",
            "1",
        ]
    )
    assert args.parallel_mode == "model"
    assert args.model_devices == "cuda:0,cuda:1"
    assert args.batch_size == 1
    assert args.eval_size == 1
