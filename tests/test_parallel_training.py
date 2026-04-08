"""Exercise model-parallel wrappers and parallel-context helpers on CPU-safe paths."""

from __future__ import annotations

from copy import deepcopy

import torch

from scripts.common import build_ideal_backbone_local, build_model_from_config, load_yaml_config, make_synthetic_batch
from training.train_parallel.data_parallel import build_parallel_context, sync_epoch_stats
from training.train_parallel.model_parallel import AlphaFold2ModelParallel


def _tiny_parallel_config() -> dict:
    config = load_yaml_config("config/experiments/af2_poc.yaml")
    config = deepcopy(config)
    config["model"]["num_evoformer_blocks"] = 1
    config["model"]["num_structure_blocks"] = 1
    config["model"]["transition_expansion_evoformer"] = 2
    config["model"]["transition_expansion_structure"] = 2
    return config


def test_build_parallel_context_model_mode_accepts_explicit_cpu_stages():
    context = build_parallel_context(mode="model", model_devices="cpu,cpu")

    assert context.mode == "model"
    assert context.model_parallel is True
    assert context.distributed is False
    assert context.stage_devices == (torch.device("cpu"), torch.device("cpu"))
    assert context.output_device == torch.device("cpu")


def test_sync_epoch_stats_is_noop_without_distributed_context():
    stats = {
        "loss": 1.0,
        "fape_loss": 0.5,
        "dist_loss": 0.3,
        "msa_loss": 0.2,
        "plddt_loss": 0.1,
        "torsion_loss": 0.1,
        "num_recycles": 2.0,
        "rmsd_logged": 0.9,
        "tm_score_logged": 0.7,
        "gdt_ts_logged": 0.6,
        "n_seen_batches": 3,
        "n_optimizer_steps": 3,
        "n_seen_samples": 6,
        "n_metric_logs": 1,
    }

    synced = sync_epoch_stats(stats, None)
    assert synced == stats


def test_model_parallel_wrapper_preserves_state_dict_keys():
    config = _tiny_parallel_config()
    base_model = build_model_from_config(config, device="cpu")
    wrapper = AlphaFold2ModelParallel(base_model, stage_devices=("cpu", "cpu"))

    assert set(wrapper.state_dict().keys()) == set(base_model.state_dict().keys())


def test_model_parallel_wrapper_forward_cpu_smoke():
    config = _tiny_parallel_config()
    base_model = build_model_from_config(config, device="cpu")
    wrapper = AlphaFold2ModelParallel(base_model, stage_devices=("cpu", "cpu"))
    batch = make_synthetic_batch(config, batch_size=1, msa_depth=2, seq_len=8, device="cpu")
    ideal_backbone_local = build_ideal_backbone_local(config, device="cpu")

    with torch.no_grad():
        outputs = wrapper(
            seq_tokens=batch["seq_tokens"],
            msa_tokens=batch["msa_tokens"],
            seq_mask=batch["seq_mask"],
            msa_mask=batch["msa_mask"],
            ideal_backbone_local=ideal_backbone_local,
            num_recycles=1,
        )

    assert outputs["distogram_logits"].shape == (1, 8, 8, 64)
    assert outputs["masked_msa_logits"].shape == (1, 2, 8, 23)
    assert outputs["torsions"].shape == (1, 8, 3, 2)
    assert outputs["backbone_coords"].shape == (1, 8, 4, 3)
    assert torch.isfinite(outputs["plddt"]).all()
