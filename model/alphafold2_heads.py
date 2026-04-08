"""Prediction heads built on top of the shared AlphaFold representations.

The classes in this module project internal sequence or pair features into the
single representation, pLDDT logits, and distogram logits used by the model
output dictionary and downstream loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleProjection(nn.Module):
    """
    Project MSA representation to single representation.
    AF2-like simplification: use first row of MSA and project to c_s.
    """
    def __init__(self, c_m=256, c_s=256):
        super().__init__()
        self.ln = nn.LayerNorm(c_m)
        self.linear = nn.Linear(c_m, c_s)

    def forward(self, m):
        # first sequence in MSA as target row
        s = m[:, 0]                  # [B, L, c_m]
        s = self.linear(self.ln(s))  # [B, L, c_s]
        return s


class PlddtHead(nn.Module):
    def __init__(self, c_s=256, hidden=256, num_bins=50):
        super().__init__()
        self.num_bins = num_bins
        self.ln = nn.LayerNorm(c_s)
        self.mlp = nn.Sequential(
            nn.Linear(c_s, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_bins))

    def forward(self, s):
        logits = self.mlp(self.ln(s))                # [B, L, num_bins]
        probs = torch.softmax(logits, dim=-1)
        bin_centers = torch.linspace(
            100.0 / self.num_bins / 2,
            100.0 - 100.0 / self.num_bins / 2,
            self.num_bins,
            device=s.device,
            dtype=s.dtype)

        plddt = (probs * bin_centers).sum(dim=-1)    # [B, L]
        return logits, plddt


class DistogramHead(nn.Module):
    def __init__(self, c_z=128, num_bins=64):
        super().__init__()
        self.num_bins = num_bins
        self.ln = nn.LayerNorm(c_z)
        self.linear = nn.Linear(c_z, num_bins)

    def forward(self, z):
        z_sym = 0.5 * (z + z.transpose(1, 2))        # symmetrize
        logits = self.linear(self.ln(z_sym))         # [B, L, L, num_bins]
        return logits


class MaskedMsaHead(nn.Module):
    def __init__(self, c_m=256, num_classes=23):
        super().__init__()
        self.num_classes = num_classes
        self.ln = nn.LayerNorm(c_m)
        self.linear = nn.Linear(c_m, num_classes)

    def forward(self, m):
        logits = self.linear(self.ln(m))
        return logits
