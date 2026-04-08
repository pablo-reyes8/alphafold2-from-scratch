"""Exercise the full AlphaFold loss orchestrator on real-batch fixtures and synthetic tensors."""

import torch
import pytest

from model.alphafold2_full_loss import *

def move_batch_to_device(batch, device: str):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

torch.manual_seed(11)

def random_unit_vectors(shape, device="cpu", dtype=torch.float32):
    x = torch.randn(*shape, device=device, dtype=dtype)
    x = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    return x

def test_real_batch_plddt_loss(loader, device="cpu"):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    B, L, _ = batch["coords_ca"].shape
    num_bins = 50

    x_true = batch["coords_ca"]
    x_pred = x_true.clone()                      # caso perfecto-ish
    mask = batch["valid_res_mask"]
    logits = torch.randn(B, L, num_bins, device=device)

    loss_fn = PlddtLoss(num_bins=num_bins, inclusion_radius=15.0)
    loss = loss_fn(logits, x_pred, x_true, mask=mask)

    assert torch.isfinite(loss), "PlddtLoss is not finite"
    assert loss.ndim == 0, "PlddtLoss should be scalar"

def test_real_batch_torsion_loss(loader, device="cpu"):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    torsion_true = batch["torsion_true"]        # [B,L,3,2]
    torsion_mask = batch["torsion_mask"]        # [B,L,3]

    torsion_pred = torsion_true.clone()
    loss_fn = TorsionLoss()
    loss = loss_fn(torsion_pred, torsion_true, torsion_mask)

    assert torch.isfinite(loss), "TorsionLoss is not finite"
    assert loss.item() < 1e-7, "TorsionLoss should be ~0 for perfect prediction"

    torsion_pred2 = torsion_true + 0.1 * torch.randn_like(torsion_true)
    torsion_pred2 = torsion_pred2 / torch.linalg.norm(
        torsion_pred2, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    loss2 = loss_fn(torsion_pred2, torsion_true, torsion_mask)

    assert torch.isfinite(loss2), "Perturbed TorsionLoss is not finite"
    assert loss2.item() > loss.item(), "Perturbed torsion loss should be larger"

def test_real_batch_alphafold_loss_orchestrator(loader, device="cpu"):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    B, L = batch["seq_tokens"].shape
    msa_depth = batch["msa_tokens"].shape[1]
    num_dist_bins = 64
    num_msa_classes = 23
    num_plddt_bins = 50
    T = batch["torsion_true"].shape[2]   # debería ser 3

    # out fake, pero consistente con el batch real
    out = {
        "R": torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, L, 1, 1),
        "t": batch["coords_ca"].clone(),   # usar CA real como predicción fácil
        "distogram_logits": torch.randn(B, L, L, num_dist_bins, device=device),
        "masked_msa_logits": torch.randn(B, msa_depth, L, num_msa_classes, device=device),
        "plddt_logits": torch.randn(B, L, num_plddt_bins, device=device),
        "torsions": batch["torsion_true"].clone(),  # predicción perfecta para torsiones
    }

    loss_fn = AlphaFoldLoss(
        fape_length_scale=10.0,
        fape_clamp_distance=10.0,
        dist_num_bins=num_dist_bins,
        dist_min_bin=2.0,
        dist_max_bin=22.0,
        msa_num_classes=num_msa_classes,
        plddt_num_bins=num_plddt_bins,
        plddt_inclusion_radius=15.0,
        w_fape=0.5,
        w_aux=0.5,
        w_dist=0.3,
        w_msa=2.0,
        w_plddt=0.01,
        w_torsion=0.01,
    )

    losses = loss_fn(out, batch)

    assert "loss" in losses
    assert torch.isfinite(losses["loss"]), "Total AlphaFold loss is not finite"
    assert losses["loss"].ndim == 0, "Total loss should be scalar"

    for name in ["fape_loss", "aux_loss", "dist_loss", "msa_loss", "plddt_loss", "torsion_loss"]:
        assert name in losses, f"Missing {name}"
        assert torch.isfinite(losses[name]), f"{name} is not finite"

    # como pusimos torsiones perfectas, debería ser casi cero
    assert losses["torsion_loss"].item() < 1e-7, "torsion_loss should be ~0 for perfect torsion prediction"

    # chequeos extra de shape para estar tranquilos
    assert out["R"].shape == (B, L, 3, 3)
    assert out["t"].shape == (B, L, 3)
    assert out["distogram_logits"].shape == (B, L, L, num_dist_bins)
    assert out["masked_msa_logits"].shape == (B, msa_depth, L, num_msa_classes)
    assert out["plddt_logits"].shape == (B, L, num_plddt_bins)
    assert out["torsions"].shape == batch["torsion_true"].shape
    assert T == 3, f"Expected 3 torsions, got {T}"

def assert_close(x, y, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        max_err = (x - y).abs().max().item()
        raise AssertionError(f"{msg} | max_err={max_err}")

def assert_scalar_finite(x, msg=""):
    assert torch.is_tensor(x), f"{msg} must be a tensor"
    assert x.ndim == 0, f"{msg} must be scalar, got shape {tuple(x.shape)}"
    assert torch.isfinite(x), f"{msg} is not finite"

def random_rotation_matrices(B, L, device="cpu", dtype=torch.float32):
    x = torch.randn(B, L, 3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(x)

    d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d.unsqueeze(-2)

    det = torch.det(q)
    flip = (det < 0).to(dtype).view(B, L, 1)
    q[..., :, 2] = q[..., :, 2] * (1.0 - 2.0 * flip)
    return q

def make_fake_alphafold_batch(B=2, L=20, T=3, device="cpu", dtype=torch.float32):
    coords_ca = torch.randn(B, L, 3, device=device, dtype=dtype)

    # construir N y C razonables alrededor de CA
    e1 = torch.randn(B, L, 3, device=device, dtype=dtype)
    e1 = e1 / torch.linalg.norm(e1, dim=-1, keepdim=True).clamp_min(1e-8)

    e2 = torch.randn(B, L, 3, device=device, dtype=dtype)
    e2 = e2 - (e2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = e2 / torch.linalg.norm(e2, dim=-1, keepdim=True).clamp_min(1e-8)

    coords_c = coords_ca + 1.5 * e1
    coords_n = coords_ca + 1.3 * e2

    valid_res_mask = torch.ones(B, L, device=device, dtype=dtype)
    valid_backbone_mask = torch.ones(B, L, device=device, dtype=dtype)
    valid_res_mask[0, -3:] = 0.0
    valid_backbone_mask[0, -2:] = 0.0

    torsion_true = torch.randn(B, L, T, 2, device=device, dtype=dtype)
    torsion_true = torsion_true / torch.linalg.norm(
        torsion_true, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    torsion_mask = torch.ones(B, L, T, device=device, dtype=dtype)
    torsion_mask[0, -2:, :] = 0.0
    masked_msa_true = torch.randint(0, 23, (B, 4, L), device=device, dtype=torch.long)
    masked_msa_mask = torch.ones(B, 4, L, device=device, dtype=dtype)
    masked_msa_mask[:, :, -1] = 0.0

    batch = {
        "coords_n": coords_n,
        "coords_ca": coords_ca,
        "coords_c": coords_c,
        "valid_res_mask": valid_res_mask,
        "valid_backbone_mask": valid_backbone_mask,
        "masked_msa_true": masked_msa_true,
        "masked_msa_mask": masked_msa_mask,
        "torsion_true": torsion_true,
        "torsion_mask": torsion_mask,
        "seq_tokens": torch.zeros(B, L, dtype=torch.long, device=device),
    }
    return batch

def make_fake_alphafold_out(batch, num_dist_bins=64, num_plddt_bins=50, device="cpu", dtype=torch.float32):
    B, L, _ = batch["coords_ca"].shape

    out = {
        "R": torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(B, L, 1, 1),
        "t": batch["coords_ca"].clone(),
        "distogram_logits": torch.randn(B, L, L, num_dist_bins, device=device, dtype=dtype),
        "masked_msa_logits": torch.randn(
            B,
            batch["masked_msa_true"].shape[1],
            L,
            23,
            device=device,
            dtype=dtype,
        ),
        "plddt_logits": torch.randn(B, L, num_plddt_bins, device=device, dtype=dtype),
        "torsions": batch["torsion_true"].clone(),
    }
    return out

# ============================================================
# build_backbone_frames tests
# ============================================================
def test_build_backbone_frames_validity():
    batch = make_fake_alphafold_batch(B=2, L=12)
    R, t = build_backbone_frames(
        batch["coords_n"],
        batch["coords_ca"],
        batch["coords_c"],
        mask=None,
    )

    assert R.shape == (2, 12, 3, 3)
    assert t.shape == (2, 12, 3)
    assert torch.isfinite(R).all(), "R has NaN/Inf"
    assert torch.isfinite(t).all(), "t has NaN/Inf"

    RtR = torch.matmul(R.transpose(-1, -2), R)
    I = torch.eye(3, device=R.device, dtype=R.dtype).view(1, 1, 3, 3).expand_as(RtR)
    det = torch.det(R)

    assert_close(RtR, I, atol=1e-4, rtol=1e-4, msg="build_backbone_frames gives non-orthogonal R")
    assert_close(det, torch.ones_like(det), atol=1e-4, rtol=1e-4, msg="det(R) != 1")
    assert_close(t, batch["coords_ca"], atol=1e-6, rtol=1e-6, msg="t should equal coords_ca")

def test_build_backbone_frames_mask_behavior():
    batch = make_fake_alphafold_batch(B=2, L=10)
    mask = batch["valid_backbone_mask"]

    R, t = build_backbone_frames(
        batch["coords_n"],
        batch["coords_ca"],
        batch["coords_c"],
        mask=mask,
    )

    eye = torch.eye(3, device=R.device, dtype=R.dtype)   # [3,3]
    masked = (mask == 0)

    if masked.any():
        n_masked = int(masked.sum().item())

        assert_close(
            R[masked],
            eye.unsqueeze(0).expand(n_masked, 3, 3),
            atol=1e-6,
            rtol=1e-6,
            msg="Masked backbone frames should be identity"
        )
        assert_close(
            t[masked],
            torch.zeros_like(t[masked]),
            atol=1e-6,
            rtol=1e-6,
            msg="Masked backbone translations should be zero"
        )

# ============================================================
# AlphaFoldLoss tests
# ============================================================
def test_alphafold_loss_weighted_sum_exact():
    batch = make_fake_alphafold_batch(B=2, L=14)
    out = make_fake_alphafold_out(batch)

    loss_fn = AlphaFoldLoss(
        w_fape=0.5,
        w_aux=0.5,
        w_dist=0.3,
        w_msa=2.0,
        w_plddt=0.01,
        w_torsion=0.01,
    )

    losses = loss_fn(out, batch)

    manual = (
        0.5 * losses["fape_loss"] +
        0.5 * losses["aux_loss"] +
        0.3 * losses["dist_loss"] +
        2.0 * losses["msa_loss"] +
        0.01 * losses["plddt_loss"] +
        0.01 * losses["torsion_loss"]
    )

    assert_close(losses["loss"], manual, atol=1e-6, rtol=1e-6, msg="Weighted total loss mismatch")

def test_alphafold_loss_without_torsion_targets():
    batch = make_fake_alphafold_batch(B=2, L=12)
    out = make_fake_alphafold_out(batch)

    batch_no_torsion = {
        k: v for k, v in batch.items()
        if k not in ("torsion_true", "torsion_mask")
    }

    loss_fn = AlphaFoldLoss()
    losses = loss_fn(out, batch_no_torsion)

    assert "torsion_loss" in losses
    assert "aux_loss" in losses
    assert_close(
        losses["torsion_loss"],
        torch.zeros_like(losses["torsion_loss"]),
        atol=1e-8,
        rtol=1e-8,
        msg="torsion_loss should be zero if torsion supervision is missing"
    )
    assert_scalar_finite(losses["loss"], "total loss without torsion targets")

def test_alphafold_loss_uses_intermediate_aux_outputs():
    batch = make_fake_alphafold_batch(B=2, L=9)
    out = make_fake_alphafold_out(batch)

    R_true, t_true = build_backbone_frames(
        batch["coords_n"],
        batch["coords_ca"],
        batch["coords_c"],
        mask=batch["valid_backbone_mask"],
    )

    num_blocks = 3
    out["aux_R"] = R_true.unsqueeze(0).repeat(num_blocks, 1, 1, 1, 1)
    out["aux_t"] = t_true.unsqueeze(0).repeat(num_blocks, 1, 1, 1)
    out["aux_torsions"] = batch["torsion_true"].unsqueeze(0).repeat(num_blocks, 1, 1, 1, 1)

    loss_fn = AlphaFoldLoss(
        w_fape=0.0,
        w_aux=1.0,
        w_dist=0.0,
        w_msa=0.0,
        w_plddt=0.0,
        w_torsion=0.0,
    )
    losses = loss_fn(out, batch)

    assert "aux_loss" in losses
    assert "aux_fape_loss" in losses
    assert "aux_torsion_loss" in losses
    assert_scalar_finite(losses["aux_loss"], "aux_loss")
    assert losses["aux_fape_loss"].item() < 2e-7, "Perfect intermediate frames should give ~0 aux FAPE"
    assert losses["aux_torsion_loss"].item() < 1e-7, "Perfect intermediate torsions should give ~0 aux torsion"
    assert_close(
        losses["aux_loss"],
        losses["aux_fape_loss"] + losses["aux_torsion_loss"],
        atol=1e-7,
        rtol=1e-7,
        msg="aux_loss should equal aux_fape_loss + aux_torsion_loss",
    )

def test_alphafold_loss_uses_backbone_coords_when_present():
    batch = make_fake_alphafold_batch(B=2, L=10)
    out = make_fake_alphafold_out(batch)

    backbone_coords = torch.zeros(
        2, 10, 3, 3,
        dtype=batch["coords_ca"].dtype,
        device=batch["coords_ca"].device
    )

    # Base: copiar algo razonable
    backbone_coords[:, :, 1, :] = batch["coords_ca"].clone()

    # Perturbación dependiente del residuo para cambiar distancias por pares
    residue_offsets = torch.linspace(
        0.0, 3.0, steps=10,
        device=batch["coords_ca"].device,
        dtype=batch["coords_ca"].dtype
    ).view(1, 10, 1)

    backbone_coords[:, :, 1, 0] += residue_offsets[..., 0]

    out["backbone_coords"] = backbone_coords

    loss_fn = AlphaFoldLoss()

    losses_with_backbone = loss_fn(out, batch)

    out_no_backbone = dict(out)
    del out_no_backbone["backbone_coords"]
    losses_without_backbone = loss_fn(out_no_backbone, batch)

    diff_plddt = abs(losses_with_backbone["plddt_loss"].item() - losses_without_backbone["plddt_loss"].item())
    diff_fape = abs(losses_with_backbone["fape_loss"].item() - losses_without_backbone["fape_loss"].item())

    assert diff_plddt > 1e-8 or diff_fape > 1e-8, (
        "backbone_coords path seems unused; neither pLDDT nor FAPE changed"
    )

def test_alphafold_loss_gradients_finite():
    batch = make_fake_alphafold_batch(B=2, L=10)
    out = make_fake_alphafold_out(batch)

    out["t"] = out["t"].clone().detach().requires_grad_(True)
    out["distogram_logits"] = out["distogram_logits"].clone().detach().requires_grad_(True)
    out["masked_msa_logits"] = out["masked_msa_logits"].clone().detach().requires_grad_(True)
    out["plddt_logits"] = out["plddt_logits"].clone().detach().requires_grad_(True)
    out["torsions"] = out["torsions"].clone().detach().requires_grad_(True)

    loss_fn = AlphaFoldLoss()
    losses = loss_fn(out, batch)
    loss = losses["loss"]

    assert_scalar_finite(loss, "AlphaFoldLoss total loss")
    loss.backward()

    assert out["t"].grad is not None, "No grad for predicted t"
    assert out["distogram_logits"].grad is not None, "No grad for distogram logits"
    assert out["masked_msa_logits"].grad is not None, "No grad for masked MSA logits"
    assert out["plddt_logits"].grad is not None, "No grad for pLDDT logits"
    assert out["torsions"].grad is not None, "No grad for torsions"

    assert torch.isfinite(out["t"].grad).all(), "t grad has NaN/Inf"
    assert torch.isfinite(out["distogram_logits"].grad).all(), "distogram grad has NaN/Inf"
    assert torch.isfinite(out["masked_msa_logits"].grad).all(), "masked MSA grad has NaN/Inf"
    assert torch.isfinite(out["plddt_logits"].grad).all(), "pLDDT grad has NaN/Inf"
    assert torch.isfinite(out["torsions"].grad).all(), "torsion grad has NaN/Inf"

def run_alphafold_orchestrator_tests():

    test_build_backbone_frames_validity()
    test_build_backbone_frames_mask_behavior()

    test_alphafold_loss_weighted_sum_exact()
    test_alphafold_loss_without_torsion_targets()
    test_alphafold_loss_uses_backbone_coords_when_present()
    test_alphafold_loss_gradients_finite()
