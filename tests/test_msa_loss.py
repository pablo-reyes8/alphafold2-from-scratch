"""Validate masked-MSA loss across masking, exact targets, and gradient flow."""

from __future__ import annotations

import torch

from model.losses.masked_msa_loss import MaskedMsaLoss

torch.manual_seed(23)


def assert_close(x, y, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        max_err = (x - y).abs().max().item()
        raise AssertionError(f"{msg} | max_err={max_err}")


def assert_scalar_finite(x, msg=""):
    assert torch.is_tensor(x), f"{msg} | expected tensor"
    assert x.ndim == 0, f"{msg} | expected scalar tensor, got shape {tuple(x.shape)}"
    assert torch.isfinite(x), f"{msg} | scalar is not finite"


def make_fake_masked_msa_batch(B=2, N=5, L=11, C=23, device="cpu", dtype=torch.float32):
    logits = torch.randn(B, N, L, C, device=device, dtype=dtype)
    targets = torch.randint(0, C, (B, N, L), device=device, dtype=torch.long)
    mask = torch.ones(B, N, L, device=device, dtype=dtype)
    mask[0, -1] = 0.0
    mask[1, 0, -2:] = 0.0
    return logits, targets, mask


def test_masked_msa_loss_scalar_and_finite():
    logits, targets, mask = make_fake_masked_msa_batch()

    loss_fn = MaskedMsaLoss(num_classes=23)
    loss = loss_fn(logits, targets, mask)

    assert_scalar_finite(loss, "Masked MSA loss")


def test_masked_msa_loss_perfect_logits_low_loss():
    B, N, L, C = 2, 4, 9, 23
    targets = torch.randint(0, C, (B, N, L), dtype=torch.long)
    mask = torch.ones(B, N, L, dtype=torch.float32)

    logits = torch.full((B, N, L, C), -8.0, dtype=torch.float32)
    logits.scatter_(-1, targets.unsqueeze(-1), 8.0)

    loss_fn = MaskedMsaLoss(num_classes=C)
    loss = loss_fn(logits, targets, mask)

    assert loss.item() < 1e-3, f"Correct masked-MSA logits should give very low loss, got {loss.item()}"


def test_masked_msa_loss_wrong_logits_higher_than_correct_logits():
    B, N, L, C = 2, 4, 8, 23
    targets = torch.randint(0, C, (B, N, L), dtype=torch.long)
    mask = torch.ones(B, N, L, dtype=torch.float32)

    logits_good = torch.full((B, N, L, C), -6.0, dtype=torch.float32)
    logits_good.scatter_(-1, targets.unsqueeze(-1), 6.0)

    wrong_targets = (targets + 1) % C
    logits_bad = torch.full((B, N, L, C), -6.0, dtype=torch.float32)
    logits_bad.scatter_(-1, wrong_targets.unsqueeze(-1), 6.0)

    loss_fn = MaskedMsaLoss(num_classes=C)
    loss_good = loss_fn(logits_good, targets, mask)
    loss_bad = loss_fn(logits_bad, targets, mask)

    assert loss_bad.item() > loss_good.item(), "Wrong masked-MSA logits should have larger loss"


def test_masked_msa_loss_masked_entries_do_not_contribute():
    logits, targets, mask = make_fake_masked_msa_batch(B=2, N=5, L=10, C=23)

    loss_fn = MaskedMsaLoss(num_classes=23)
    loss_ref = loss_fn(logits, targets, mask)

    logits_corrupted = logits.clone()
    targets_corrupted = targets.clone()
    invalid = mask == 0
    logits_corrupted[invalid] = 1e4 * torch.randn_like(logits_corrupted[invalid])
    targets_corrupted[invalid] = (targets_corrupted[invalid] + 7) % 23

    loss_corrupted = loss_fn(logits_corrupted, targets_corrupted, mask)
    assert_close(
        loss_ref,
        loss_corrupted,
        atol=1e-5,
        rtol=1e-5,
        msg="Masked masked-MSA positions should not affect loss",
    )


def test_masked_msa_loss_all_zero_mask_gives_zero():
    logits, targets, _ = make_fake_masked_msa_batch(B=2, N=3, L=7, C=23)
    zero_mask = torch.zeros(2, 3, 7, dtype=torch.float32)

    loss_fn = MaskedMsaLoss(num_classes=23)
    loss = loss_fn(logits, targets, zero_mask)

    assert_scalar_finite(loss, "Masked MSA all-zero-mask loss")
    assert_close(loss, torch.zeros_like(loss), atol=1e-8, msg="All-zero masked-MSA mask should give zero loss")


def test_masked_msa_loss_matches_manual_cross_entropy_average():
    logits, targets, mask = make_fake_masked_msa_batch(B=2, N=4, L=6, C=23)

    loss_fn = MaskedMsaLoss(num_classes=23)
    loss = loss_fn(logits, targets, mask)

    logits_flat = logits.reshape(-1, 23)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)
    manual = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction="none")
    manual = (manual * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)

    assert_close(loss, manual, atol=1e-6, rtol=1e-6, msg="Masked-MSA loss mismatch with manual formula")


def test_masked_msa_loss_joint_row_permutation_invariant():
    logits, targets, mask = make_fake_masked_msa_batch(B=2, N=5, L=7, C=23)
    permutation = torch.tensor([4, 1, 3, 0, 2], dtype=torch.long)

    loss_fn = MaskedMsaLoss(num_classes=23)
    loss_ref = loss_fn(logits, targets, mask)
    loss_perm = loss_fn(logits[:, permutation], targets[:, permutation], mask[:, permutation])

    assert_close(
        loss_ref,
        loss_perm,
        atol=1e-6,
        rtol=1e-6,
        msg="Joint permutation of MSA rows should not change masked-MSA loss",
    )


def test_masked_msa_loss_gradients_finite():
    logits, targets, mask = make_fake_masked_msa_batch(B=2, N=4, L=8, C=23)
    logits = logits.clone().detach().requires_grad_(True)

    loss_fn = MaskedMsaLoss(num_classes=23)
    loss = loss_fn(logits, targets, mask)

    assert_scalar_finite(loss, "Masked MSA gradient loss")
    loss.backward()

    assert logits.grad is not None, "masked_msa_logits got no gradient"
    assert torch.isfinite(logits.grad).all(), "masked_msa_logits grad has NaN/Inf"
