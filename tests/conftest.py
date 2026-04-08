"""Provide shared pytest fixtures for AlphaFold model, loss, and training smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from data.dataloaders import build_masked_msa_inputs
from model.alphafold2 import AlphaFold2
from model.alphafold2_full_loss import AlphaFoldLoss


def _random_unit_vectors(shape):
    values = torch.randn(*shape)
    return values / torch.linalg.norm(values, dim=-1, keepdim=True).clamp_min(1e-8)


@pytest.fixture
def ideal_backbone_local():
    return torch.tensor(
        [
            [-1.458, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.547, 1.426, 0.0],
            [0.224, 2.617, 0.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def toy_batch(ideal_backbone_local):
    torch.manual_seed(123)

    batch_size, msa_depth, length, n_torsions = 2, 3, 6, 3
    seq_tokens = torch.randint(1, 27, (batch_size, length))
    msa_tokens = torch.randint(1, 27, (batch_size, msa_depth, length))

    seq_mask = torch.ones(batch_size, length, dtype=torch.float32)
    seq_mask[0, -1] = 0.0
    msa_mask = seq_mask[:, None, :].repeat(1, msa_depth, 1)
    masked_msa_true = torch.zeros(batch_size, msa_depth, length, dtype=torch.long)
    masked_msa_mask = torch.zeros(batch_size, msa_depth, length, dtype=torch.float32)
    for batch_index in range(batch_size):
        masked_tokens, masked_true, masked_mask = build_masked_msa_inputs(
            msa_tokens[batch_index],
            msa_mask[batch_index],
        )
        msa_tokens[batch_index] = masked_tokens
        masked_msa_true[batch_index] = masked_true
        masked_msa_mask[batch_index] = masked_mask

    residue_axis = torch.arange(length, dtype=torch.float32)
    coords_ca = torch.stack(
        [residue_axis, 0.1 * residue_axis, torch.zeros_like(residue_axis)],
        dim=-1,
    )
    coords_ca = coords_ca.unsqueeze(0).repeat(batch_size, 1, 1)
    coords_ca = coords_ca + 0.01 * torch.randn_like(coords_ca)

    coords_n = coords_ca + torch.tensor([-1.2, 0.4, 0.1], dtype=torch.float32)
    coords_c = coords_ca + torch.tensor([1.3, 0.5, -0.1], dtype=torch.float32)

    valid_res_mask = seq_mask.clone()
    valid_backbone_mask = seq_mask.clone()

    torsion_true = _random_unit_vectors((batch_size, length, n_torsions, 2)).to(torch.float32)
    torsion_mask = valid_backbone_mask.unsqueeze(-1).expand(batch_size, length, n_torsions).clone()

    pair_mask = valid_res_mask[:, :, None] * valid_res_mask[:, None, :]

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
        "ideal_backbone_local": ideal_backbone_local,
    }


@pytest.fixture
def toy_model():
    torch.manual_seed(7)
    model = AlphaFold2(
        n_tokens=27,
        c_m=256,
        c_z=128,
        c_s=256,
        max_relpos=32,
        pad_idx=0,
        num_evoformer_blocks=1,
        num_structure_blocks=1,
        transition_expansion_evoformer=2,
        transition_expansion_structure=2,
        use_block_specific_params=False,
        dist_bins=64,
        plddt_bins=50,
        n_torsions=3,
        num_res_blocks_torsion=1,
    )
    model.eval()
    return model


@pytest.fixture
def toy_criterion():
    return AlphaFoldLoss(
        fape_length_scale=10.0,
        fape_clamp_distance=10.0,
        dist_num_bins=64,
        dist_min_bin=2.0,
        dist_max_bin=22.0,
        plddt_num_bins=50,
        plddt_inclusion_radius=15.0,
        w_fape=0.5,
        w_aux=0.5,
        w_dist=0.3,
        w_msa=2.0,
        w_plddt=0.01,
        w_torsion=0.01,
    )


@pytest.fixture
def loader(toy_batch):
    """Provide a minimal iterable loader for legacy loss smoke tests."""
    return [toy_batch]


def _test_file_name(request) -> str:
    return Path(str(request.fspath)).name


@pytest.fixture
def batch(request):
    """Build synthetic batches for legacy module tests without editing their signatures."""
    test_file = _test_file_name(request)

    if test_file == "test_ipa.py":
        from tests.test_ipa import make_fake_ipa_batch

        return make_fake_ipa_batch(B=2, L=32, c_s=256, c_z=128, device="cpu")

    if test_file == "test_opm.py":
        from tests.test_helpers import make_fake_msa_batch

        return make_fake_msa_batch(B=2, N_msa=16, L=32, c_m=256, device="cpu")

    if test_file == "test_row_column_attention.py":
        from tests.test_row_column_attention import make_fake_msa_pair_batch

        return make_fake_msa_pair_batch(B=2, N=16, L=32, c_m=256, c_z=128, device="cpu")

    if test_file == "test_triangle_attention.py":
        from tests.test_triangle_attention import make_fake_pair_batch

        return make_fake_pair_batch(B=2, L=32, c_z=128, device="cpu")

    if test_file == "test_triangle_multiplication.py":
        from tests.test_triangle_multiplication import make_fake_pair_batch

        return make_fake_pair_batch(B=2, L=32, c_z=128, device="cpu")

    raise LookupError(f"No shared batch fixture is configured for {test_file}.")


@pytest.fixture
def module(request):
    """Instantiate legacy modules for pytest-collected architecture tests."""
    test_file = _test_file_name(request)
    test_name = request.function.__name__

    if test_file == "test_ipa.py":
        from model.invariant_point_attention import InvariantPointAttention

        return InvariantPointAttention(
            c_s=256,
            c_z=128,
            num_heads=8,
            c_hidden=32,
            num_qk_points=4,
            num_v_points=8,
        ).to("cpu")

    if test_file == "test_opm.py":
        from model.outer_product_mean import OuterProductMean

        return OuterProductMean(c_m=256, c_hidden=32, c_z=128).to("cpu")

    if test_file == "test_row_column_attention.py":
        if test_name.startswith("test_msa_row_attention_"):
            from model.msa_row_attention import MSARowAttentionWithPairBias

            return MSARowAttentionWithPairBias(
                c_m=256,
                c_z=128,
                num_heads=8,
                c_hidden=32,
            ).to("cpu")

        from model.msa_column_attention import MSAColumnAttention

        return MSAColumnAttention(
            c_m=256,
            num_heads=8,
            c_hidden=32,
        ).to("cpu")

    if test_file == "test_triangle_attention.py":
        from model.triangle_attention import TriangleAttentionStartingNode

        return TriangleAttentionStartingNode(
            c_z=128,
            num_heads=4,
            c_hidden=32,
        ).to("cpu")

    if test_file == "test_triangle_multiplication.py":
        from model.triangle_multiplication import TriangleMultiplicationOutgoing

        return TriangleMultiplicationOutgoing(
            c_z=128,
            c_hidden=128,
            dropout=0.1,
        ).to("cpu")

    raise LookupError(f"No shared module fixture is configured for {test_file}.")
