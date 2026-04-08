"""Masked-MSA supervision for Evoformer MSA representations.

This module applies the AlphaFold-style masked-MSA auxiliary objective by
scoring logits projected from the final MSA representation against the original
pre-corruption residue classes, averaged only over positions selected for the
masked reconstruction task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMsaLoss(nn.Module):
    """Cross-entropy loss over masked MSA positions only."""

    def __init__(self, num_classes=23):
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, masked_msa_logits, masked_msa_true, masked_msa_mask=None):
        batch, depth, length, num_classes = masked_msa_logits.shape
        assert num_classes == self.num_classes, (
            f"Expected num_classes={self.num_classes}, got {num_classes}"
        )

        logits_flat = masked_msa_logits.reshape(batch * depth * length, self.num_classes)
        targets_flat = masked_msa_true.reshape(batch * depth * length)
        per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        if masked_msa_mask is not None:
            valid = masked_msa_mask.reshape(batch * depth * length).to(per_token_loss.dtype)
            denom = valid.sum().clamp_min(1.0)
            return (per_token_loss * valid).sum() / denom

        return per_token_loss.mean()
