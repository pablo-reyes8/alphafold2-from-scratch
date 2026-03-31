from __future__ import annotations

import torch


def test_alphafold2_forward_smoke(toy_model, toy_batch):
    with torch.no_grad():
        outputs = toy_model(
            seq_tokens=toy_batch["seq_tokens"],
            msa_tokens=toy_batch["msa_tokens"],
            seq_mask=toy_batch["seq_mask"],
            msa_mask=toy_batch["msa_mask"],
            ideal_backbone_local=toy_batch["ideal_backbone_local"],
        )

    batch_size, length = toy_batch["seq_tokens"].shape

    assert outputs["m"].shape == (batch_size, toy_batch["msa_tokens"].shape[1], length, 256)
    assert outputs["z"].shape == (batch_size, length, length, 128)
    assert outputs["s"].shape == (batch_size, length, 256)
    assert outputs["R"].shape == (batch_size, length, 3, 3)
    assert outputs["t"].shape == (batch_size, length, 3)
    assert outputs["backbone_coords"].shape == (batch_size, length, 4, 3)
    assert outputs["torsions"].shape == (batch_size, length, 3, 2)
    assert outputs["plddt_logits"].shape == (batch_size, length, 50)
    assert outputs["plddt"].shape == (batch_size, length)
    assert outputs["distogram_logits"].shape == (batch_size, length, length, 64)

    for value in outputs.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()

    assert torch.allclose(
        outputs["distogram_logits"],
        outputs["distogram_logits"].transpose(1, 2),
        atol=1e-5,
    )
    assert torch.all((outputs["plddt"] >= 0.0) & (outputs["plddt"] <= 100.0))


def test_alphafold_loss_orchestrator_returns_finite_components(toy_model, toy_batch, toy_criterion):
    with torch.no_grad():
        outputs = toy_model(
            seq_tokens=toy_batch["seq_tokens"],
            msa_tokens=toy_batch["msa_tokens"],
            seq_mask=toy_batch["seq_mask"],
            msa_mask=toy_batch["msa_mask"],
            ideal_backbone_local=toy_batch["ideal_backbone_local"],
        )
        losses = toy_criterion(outputs, toy_batch)

    for name in ("loss", "fape_loss", "dist_loss", "plddt_loss", "torsion_loss"):
        assert name in losses
        assert torch.isfinite(losses[name])
        assert losses[name].ndim == 0

    expected = (
        toy_criterion.w_fape * losses["fape_loss"]
        + toy_criterion.w_dist * losses["dist_loss"]
        + toy_criterion.w_plddt * losses["plddt_loss"]
        + toy_criterion.w_torsion * losses["torsion_loss"]
    )
    assert torch.allclose(losses["loss"], expected, atol=1e-5)


def test_alphafold2_recycling_changes_final_representations(toy_model, toy_batch):
    with torch.no_grad():
        baseline = toy_model(
            seq_tokens=toy_batch["seq_tokens"],
            msa_tokens=toy_batch["msa_tokens"],
            seq_mask=toy_batch["seq_mask"],
            msa_mask=toy_batch["msa_mask"],
            ideal_backbone_local=toy_batch["ideal_backbone_local"],
            num_recycles=0,
        )
        recycled = toy_model(
            seq_tokens=toy_batch["seq_tokens"],
            msa_tokens=toy_batch["msa_tokens"],
            seq_mask=toy_batch["seq_mask"],
            msa_mask=toy_batch["msa_mask"],
            ideal_backbone_local=toy_batch["ideal_backbone_local"],
            num_recycles=2,
        )

    assert recycled["z"].shape == baseline["z"].shape
    assert recycled["t"].shape == baseline["t"].shape
    assert recycled["backbone_coords"].shape == baseline["backbone_coords"].shape
    assert torch.isfinite(recycled["z"]).all()
    assert torch.isfinite(recycled["distogram_logits"]).all()
    assert not torch.allclose(recycled["z"], baseline["z"])
    assert not torch.allclose(recycled["distogram_logits"], baseline["distogram_logits"])


def test_alphafold_loss_uses_backbone_coords_when_available(toy_batch, toy_criterion):
    batch_size, length = toy_batch["seq_tokens"].shape

    outputs = {
        "R": torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, length, 1, 1),
        "t": torch.zeros(batch_size, length, 3),
        "backbone_coords": torch.zeros(batch_size, length, 4, 3),
        "distogram_logits": torch.randn(batch_size, length, length, 64),
        "plddt_logits": torch.randn(batch_size, length, 50),
        "torsions": toy_batch["torsion_true"].clone(),
    }

    losses = toy_criterion(outputs, toy_batch)
    assert torch.isfinite(losses["loss"])
    assert losses["torsion_loss"].item() < 1e-7
