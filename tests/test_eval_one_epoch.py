"""Exercise eval_one_epoch with lightweight fake batches and a mock model."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.eval_one_epoch import eval_one_epoch


def make_fake_eval_batch(batch_size=1, msa_depth=2, seq_len=6):
    seq_tokens = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    msa_tokens = seq_tokens.unsqueeze(1).repeat(1, msa_depth, 1)
    seq_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    msa_mask = torch.ones(batch_size, msa_depth, seq_len, dtype=torch.float32)

    residue_axis = seq_tokens.float()
    coords_ca = torch.stack(
        [residue_axis, 0.1 * residue_axis, torch.zeros_like(residue_axis)],
        dim=-1,
    )
    coords_n = coords_ca + torch.tensor([-1.2, 0.4, 0.1], dtype=torch.float32)
    coords_c = coords_ca + torch.tensor([1.3, 0.5, -0.1], dtype=torch.float32)

    valid_res_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    valid_backbone_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    torsion_true = torch.zeros(batch_size, seq_len, 3, 2, dtype=torch.float32)
    torsion_mask = torch.ones(batch_size, seq_len, 3, dtype=torch.float32)

    return {
        "seq_tokens": seq_tokens,
        "msa_tokens": msa_tokens,
        "seq_mask": seq_mask,
        "msa_mask": msa_mask,
        "masked_msa_true": torch.zeros(batch_size, msa_depth, seq_len, dtype=torch.long),
        "masked_msa_mask": torch.zeros(batch_size, msa_depth, seq_len, dtype=torch.float32),
        "coords_n": coords_n,
        "coords_ca": coords_ca,
        "coords_c": coords_c,
        "valid_res_mask": valid_res_mask,
        "valid_backbone_mask": valid_backbone_mask,
        "torsion_true": torsion_true,
        "torsion_mask": torsion_mask,
        "extra_msa_feat": None,
        "extra_msa_mask": None,
        "template_angle_feat": None,
        "template_pair_feat": None,
        "template_mask": None,
    }


class PerfectBackboneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_num_recycles = []

    def forward(self, *, seq_tokens, num_recycles=0, **kwargs):
        self.seen_num_recycles.append(int(num_recycles))
        residue_axis = seq_tokens.float()
        coords_ca = torch.stack(
            [residue_axis, 0.1 * residue_axis, torch.zeros_like(residue_axis)],
            dim=-1,
        )
        batch_size, seq_len, _ = coords_ca.shape
        backbone_coords = torch.zeros(batch_size, seq_len, 4, 3, dtype=coords_ca.dtype, device=coords_ca.device)
        backbone_coords[:, :, 1, :] = coords_ca
        eye = torch.eye(3, dtype=coords_ca.dtype, device=coords_ca.device).view(1, 1, 3, 3)
        return {
            "R": eye.repeat(batch_size, seq_len, 1, 1),
            "t": coords_ca,
            "backbone_coords": backbone_coords,
        }


class PerfectBackboneCriterion:
    def __call__(self, outputs, batch):
        x_pred = outputs["backbone_coords"][:, :, 1, :]
        loss = torch.nn.functional.mse_loss(x_pred, batch["coords_ca"])
        zero = loss.detach() * 0.0
        return {
            "loss": loss,
            "fape_loss": zero,
            "dist_loss": zero,
            "msa_loss": zero,
            "plddt_loss": zero,
            "torsion_loss": zero,
        }


def test_eval_one_epoch_returns_perfect_metrics_for_matching_backbone():
    model = PerfectBackboneModel()
    dataloader = [make_fake_eval_batch(), make_fake_eval_batch()]

    stats = eval_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=PerfectBackboneCriterion(),
        device="cpu",
        amp_enabled=False,
        log_every=0,
        num_recycles=2,
    )

    assert model.seen_num_recycles == [2, 2]
    assert stats["n_seen_batches"] == 2
    assert stats["n_metric_logs"] == 2
    assert stats["num_recycles"] == 2.0
    assert stats["loss"] < 1e-8
    assert stats["rmsd_logged"] < 2e-4
    assert stats["tm_score_logged"] > 0.999
    assert stats["gdt_ts_logged"] > 0.999
