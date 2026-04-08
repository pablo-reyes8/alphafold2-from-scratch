"""Validate the ablation registry and high-level ablation helpers."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from data.dataloaders import select_msa_sequences
from model.alphafold2 import AlphaFold2
from model.alphafold2_full_loss import AlphaFoldLoss
from model.recycling_module import RecyclingEmbedder
from scripts.common import load_yaml_config
from training.ablations import list_ablation_specs, render_ablation_catalog, resolve_training_variant
from training.train_parallel.model_parallel import AlphaFold2ModelParallel


def _tiny_base_config() -> dict:
    config = deepcopy(load_yaml_config("config/experiments/af2_poc.yaml"))
    config["model"]["num_evoformer_blocks"] = 1
    config["model"]["num_structure_blocks"] = 1
    config["model"]["transition_expansion_evoformer"] = 2
    config["model"]["transition_expansion_structure"] = 2
    config["trainer"]["epochs"] = 1
    return config


def test_ablation_catalog_contains_five_named_presets():
    specs = list_ablation_specs()

    assert [spec.key for spec in specs] == ["AF2_1", "AF2_2", "AF2_3", "AF2_4", "AF2_5"]
    catalog = render_ablation_catalog()
    assert "BASELINE" in catalog
    assert "single_sequence_msa" in catalog


def test_resolve_training_variant_annotates_metadata_and_training_names():
    config, spec = resolve_training_variant(_tiny_base_config(), ablation_name="AF2_3")

    assert spec.key == "AF2_3"
    assert config["metadata"]["ablation_id"] == "AF2_3"
    assert spec.title in config["metadata"]["ablation_title"]
    assert "ablations/" in config["trainer"]["ckpt_dir"]
    assert config["trainer"]["run_name"].startswith("af2_poc_")


def test_baseline_variant_accepts_orthogonal_modifiers():
    config, spec = resolve_training_variant(
        _tiny_base_config(),
        ablation_name="BASELINE",
        single_sequence_msa=True,
        use_block_specific_params=True,
    )

    assert spec.key == "BASELINE"
    assert config["data"]["single_sequence_mode"] is True
    assert config["data"]["max_msa_seqs"] == 1
    assert config["model"]["use_block_specific_params"] is True
    assert config["metadata"]["name"].endswith("single_sequence_structure_untied")


def test_strong_ablation_presets_set_expected_config_overrides():
    af2_1, _ = resolve_training_variant(_tiny_base_config(), ablation_name="AF2_1")
    af2_2, _ = resolve_training_variant(_tiny_base_config(), ablation_name="AF2_2")
    af2_3, _ = resolve_training_variant(_tiny_base_config(), ablation_name="AF2_3")
    af2_4, _ = resolve_training_variant(_tiny_base_config(), ablation_name="AF2_4")
    af2_5, _ = resolve_training_variant(_tiny_base_config(), ablation_name="AF2_5")

    assert af2_1["model"]["ablation"] == 1
    assert af2_1["loss"]["ablation"] == 1
    assert af2_1["trainer"]["num_recycles"] == 0

    assert af2_2["model"]["ablation"] == 2
    assert af2_2["loss"]["ablation"] == 2
    assert af2_2["trainer"]["num_recycles"] == 0

    assert af2_3["model"]["ablation"] == 3
    assert af2_3["loss"]["ablation"] == 3

    assert af2_4["model"]["ablation"] == 4
    assert af2_5["model"]["ablation"] == 5
    assert af2_5["trainer"]["max_recycles"] == 0


def _make_alphafold2_stub(
    *,
    recycle_single_enabled=True,
    recycle_pair_enabled=True,
    structure_pair_context_enabled=True,
):
    model = AlphaFold2.__new__(AlphaFold2)
    nn.Module.__init__(model)
    model.recycle_single_enabled = recycle_single_enabled
    model.recycle_pair_enabled = recycle_pair_enabled
    model.structure_pair_context_enabled = structure_pair_context_enabled
    model.recycling_embedder = RecyclingEmbedder(
        c_m=4,
        c_z=4,
        recycle_single_enabled=recycle_single_enabled,
        recycle_pair_enabled=recycle_pair_enabled,
        recycle_position_enabled=True,
    )
    return model


def _make_model_parallel_stub(
    *,
    recycle_single_enabled=True,
    recycle_pair_enabled=True,
    structure_pair_context_enabled=True,
):
    wrapper = AlphaFold2ModelParallel.__new__(AlphaFold2ModelParallel)
    nn.Module.__init__(wrapper)
    wrapper.recycle_single_enabled = recycle_single_enabled
    wrapper.recycle_pair_enabled = recycle_pair_enabled
    wrapper.structure_pair_context_enabled = structure_pair_context_enabled
    wrapper.recycling_embedder = RecyclingEmbedder(
        c_m=4,
        c_z=4,
        recycle_single_enabled=recycle_single_enabled,
        recycle_pair_enabled=recycle_pair_enabled,
        recycle_position_enabled=True,
    )
    return wrapper


def test_alphafold2_recycle_single_update_respects_flag():
    m = torch.randn(2, 3, 5, 4)
    prev_m1 = torch.randn(2, 5, 4)

    disabled = _make_alphafold2_stub(recycle_single_enabled=False)
    enabled = _make_alphafold2_stub(recycle_single_enabled=True)

    out_disabled = disabled._apply_recycle_single_update(m.clone(), prev_m1)
    out_enabled = enabled._apply_recycle_single_update(m.clone(), prev_m1)

    assert torch.allclose(out_disabled, m)
    assert not torch.allclose(out_enabled, m)


def test_alphafold2_recycle_pair_update_respects_flag():
    z = torch.randn(2, 5, 5, 4)
    prev_pair = torch.randn(2, 5, 5, 4)

    disabled = _make_alphafold2_stub(recycle_pair_enabled=False)
    enabled = _make_alphafold2_stub(recycle_pair_enabled=True)

    out_disabled = disabled._apply_recycle_pair_update(z.clone(), prev_pair)
    out_enabled = enabled._apply_recycle_pair_update(z.clone(), prev_pair)

    assert torch.allclose(out_disabled, z)
    assert not torch.allclose(out_enabled, z)


def test_alphafold2_structure_pair_input_can_be_zeroed():
    z = torch.randn(1, 6, 6, 4)
    disabled = _make_alphafold2_stub(structure_pair_context_enabled=False)
    enabled = _make_alphafold2_stub(structure_pair_context_enabled=True)

    assert torch.allclose(enabled._build_structure_pair_input(z), z)
    assert torch.count_nonzero(disabled._build_structure_pair_input(z)) == 0


def test_model_parallel_helper_flags_match_plain_model_behavior():
    m = torch.randn(1, 2, 4, 4)
    z = torch.randn(1, 4, 4, 4)
    prev_m1 = torch.randn(1, 4, 4)
    prev_pair = torch.randn(1, 4, 4, 4)

    wrapper = _make_model_parallel_stub(
        recycle_single_enabled=False,
        recycle_pair_enabled=False,
        structure_pair_context_enabled=False,
    )

    assert torch.allclose(wrapper._apply_recycle_single_update(m.clone(), prev_m1), m)
    assert torch.allclose(wrapper._apply_recycle_pair_update(z.clone(), prev_pair), z)
    assert torch.count_nonzero(wrapper._build_structure_pair_input(z)) == 0


def test_alphafold2_ablation_defaults_are_explicit_and_baseline_safe():
    assert AlphaFold2.resolve_ablation_defaults(None) == {}
    assert AlphaFold2.resolve_ablation_defaults(1)["recycle_single_enabled"] is False
    assert AlphaFold2.resolve_ablation_defaults(1)["evoformer_pair_stack_enabled"] is False
    assert AlphaFold2.resolve_ablation_defaults(2)["evoformer_triangle_attention_enabled"] is False
    assert AlphaFold2.resolve_ablation_defaults(2)["recycle_single_enabled"] is False
    assert AlphaFold2.resolve_ablation_defaults(3)["masked_msa_head_enabled"] is False
    assert AlphaFold2.resolve_ablation_defaults(3)["plddt_head_enabled"] is False
    assert AlphaFold2.resolve_ablation_defaults(4)["use_block_specific_params"] is True
    assert AlphaFold2.resolve_ablation_defaults(5)["recycle_single_enabled"] is False
    assert AlphaFold2.resolve_ablation_defaults(5)["evoformer_enabled"] is False


def test_alphafold_loss_ablation_defaults_match_named_suite():
    assert AlphaFoldLoss.resolve_ablation_defaults(None) == {}
    assert AlphaFoldLoss.resolve_ablation_defaults(1)["w_plddt"] == 0.0
    assert AlphaFoldLoss.resolve_ablation_defaults(2)["w_plddt"] == 0.0
    assert AlphaFoldLoss.resolve_ablation_defaults(3) == {
        "w_aux": 0.0,
        "w_dist": 0.0,
        "w_msa": 0.0,
        "w_plddt": 0.0,
        "w_torsion": 0.0,
    }


def test_loss_gracefully_skips_disabled_auxiliary_terms():
    criterion = AlphaFoldLoss(w_fape=0.0, w_aux=0.0, w_dist=0.0, w_msa=0.0, w_plddt=0.0, w_torsion=0.0)

    coords_n = torch.tensor([[[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [2.0, -1.0, 0.0]]], dtype=torch.float32)
    coords_ca = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float32)
    coords_c = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]], dtype=torch.float32)

    out = {
        "R": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).expand(1, 3, 3, 3).clone(),
        "t": torch.zeros(1, 3, 3, dtype=torch.float32),
        "distogram_logits": None,
        "masked_msa_logits": None,
        "plddt_logits": None,
        "torsions": None,
    }
    batch = {
        "coords_n": coords_n,
        "coords_ca": coords_ca,
        "coords_c": coords_c,
        "valid_res_mask": torch.ones(1, 3, dtype=torch.float32),
        "valid_backbone_mask": torch.ones(1, 3, dtype=torch.float32),
    }

    losses = criterion(out, batch)

    assert float(losses["loss"].item()) == 0.0
    assert float(losses["fape_loss"].item()) == 0.0
    assert float(losses["aux_loss"].item()) == 0.0
    assert float(losses["dist_loss"].item()) == 0.0
    assert float(losses["msa_loss"].item()) == 0.0
    assert float(losses["plddt_loss"].item()) == 0.0
    assert float(losses["torsion_loss"].item()) == 0.0


def test_single_sequence_msa_selector_collapses_to_target_only():
    selected = select_msa_sequences(
        ["AAAA", "BBBB", "CCCC"],
        target_sequence="WXYZ",
        target_len=4,
        max_msa_seqs=16,
        single_sequence_mode=True,
    )

    assert selected == ["WXYZ"]
