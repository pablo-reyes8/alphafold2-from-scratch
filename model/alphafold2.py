"""Top-level AlphaFold2-like model assembly.

This module wires the input embedder, Evoformer stack, structure module, and
output heads into a single PyTorch model that returns representations,
geometric predictions, backbone coordinates, torsions, confidence, and
distogram outputs.
"""

import torch 
import torch.nn as nn 
from model.evoformer_block import * 
from model.evoformer_stack import *

from model.alphafold2_heads import * 
from model.torsion_head import *
from model.structure_block import *

class AlphaFold2(nn.Module):
    """
    AF2-like model.

    Outputs:
      - m, z
      - single representation s
      - frames R, t
      - backbone coords from ideal local coords
      - torsion angles
      - pLDDT
      - distogram logits
    """
    def __init__(
        self,
        n_tokens,
        c_m=256,
        c_z=128,
        c_s=256,
        max_relpos=32,
        pad_idx=0,
        num_evoformer_blocks=4,
        num_structure_blocks=8,
        transition_expansion_evoformer = 4, 
        transition_expansion_structure = 4, 
        use_block_specific_params = False, 
        dist_bins=64,
        plddt_bins=50,
        n_torsions=7,
        num_res_blocks_torsion=2,
        recycle_min_bin=3.25,
        recycle_max_bin=20.75,
        recycle_dist_bins=15):

        super().__init__()

        self.c_z = c_z
        self.recycle_min_bin = float(recycle_min_bin)
        self.recycle_max_bin = float(recycle_max_bin)
        self.recycle_dist_bins = int(recycle_dist_bins)


        # Tokens de Entrada
        self.input_embedder = InputEmbedder(
            n_tokens=n_tokens,
            c_m=c_m,
            c_z=c_z,
            c_s=c_s,
            max_relpos=max_relpos,
            pad_idx=pad_idx)


        # Evoformer para m y z
        self.evoformer = EvoformerStack(
            num_blocks=num_evoformer_blocks,
            c_m=c_m,
            c_z=c_z , transition_expansion=transition_expansion_evoformer)

        self.single_proj = SingleProjection(c_m=c_m, c_s=c_s)

        # Structure Model con IPA
        self.structure_module = StructureModule(
            c_s=c_s,
            c_z=c_z,
            num_blocks=num_structure_blocks , transition_expansion=transition_expansion_structure , use_block_specific_params=use_block_specific_params)

        # Cabezas finales para entender el modelo
        self.plddt_head = PlddtHead(c_s=c_s, num_bins=plddt_bins)
        self.distogram_head = DistogramHead(c_z=c_z, num_bins=dist_bins)
        self.torsion_head = TorsionHead(c_s=c_s, n_torsions=n_torsions , num_res_blocks = num_res_blocks_torsion)
        self.recycle_pair_norm = nn.LayerNorm(c_z)
        self.recycle_pos_embedding = nn.Embedding(self.recycle_dist_bins, c_z)

    def _apply_recycle_pair_update(self, z, prev_pair, pair_mask=None):
        if prev_pair is None:
            return z

        z = z + self.recycle_pair_norm(prev_pair)

        if pair_mask is not None:
            z = z * pair_mask.unsqueeze(-1)

        return z

    def _positions_to_recycle_dgram(self, positions, dtype, pair_mask=None):
        deltas = positions[:, :, None, :] - positions[:, None, :, :]
        sq_dist = deltas.pow(2).sum(dim=-1).float()

        boundaries = torch.linspace(
            self.recycle_min_bin,
            self.recycle_max_bin,
            self.recycle_dist_bins - 1,
            device=positions.device,
            dtype=sq_dist.dtype,
        ).pow(2)

        bin_ids = torch.bucketize(sq_dist, boundaries)
        recycle_update = self.recycle_pos_embedding(bin_ids).to(dtype=dtype)

        if pair_mask is not None:
            recycle_update = recycle_update * pair_mask.unsqueeze(-1)

        return recycle_update

    def _extract_recycle_positions(self, backbone_coords, t):
        if backbone_coords is not None:
            ca_index = 1 if backbone_coords.shape[-2] > 1 else 0
            return backbone_coords[:, :, ca_index, :]
        return t


    def forward(
        self,
        seq_tokens,
        msa_tokens,
        seq_mask=None,
        msa_mask=None,
        ideal_backbone_local=None,
        num_recycles: int = 0):
        """
        ideal_backbone_local: [A, 3] or [1,1,A,3] or [B,L,A,3]
          e.g. local ideal coordinates for backbone atoms (N, CA, C, O)

        num_recycles:
          Number of extra recycling passes on the same batch. ``0`` means a
          single forward pass with no recycling.

        returns dict
        """
        if seq_mask is not None:
            pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]
        else:
            pair_mask = None

        num_recycles = max(0, int(num_recycles))
        prev_pair = None
        prev_positions = None
        outputs = None

        for recycle_idx in range(num_recycles + 1):
            # input
            m, z = self.input_embedder(
                seq_tokens=seq_tokens,
                msa_tokens=msa_tokens,
                seq_mask=seq_mask,
                msa_mask=msa_mask)

            z = self._apply_recycle_pair_update(
                z,
                prev_pair=prev_pair,
                pair_mask=pair_mask,
            )

            if prev_positions is not None:
                z = z + self._positions_to_recycle_dgram(
                    prev_positions,
                    dtype=z.dtype,
                    pair_mask=pair_mask,
                )

            # evoformer
            m, z = self.evoformer(
                m,
                z,
                msa_mask=msa_mask,
                pair_mask=pair_mask,)

            # z before structure for distogram
            distogram_logits = self.distogram_head(z)

            # single repr + structure
            s0 = self.single_proj(m)
            s, R, t = self.structure_module(s0, z, mask=seq_mask)

            # backbone coordinates from ideal local atoms
            backbone_coords = None
            if ideal_backbone_local is not None:
                if ideal_backbone_local.dim() == 2:
                    # [A,3] -> [1,1,A,3]
                    ideal_backbone_local = ideal_backbone_local.unsqueeze(0).unsqueeze(0)
                elif ideal_backbone_local.dim() == 4:
                    pass
                else:
                    raise ValueError("ideal_backbone_local must have shape [A,3] or [B,L,A,3]")

                if ideal_backbone_local.shape[0] == 1 and ideal_backbone_local.shape[1] == 1:
                    B, L = seq_tokens.shape
                    ideal_backbone_local = ideal_backbone_local.expand(B, L, -1, -1)

                backbone_coords = apply_transform(
                    R[:, :, None, :, :],     # [B,L,1,3,3]
                    t[:, :, None, :],        # [B,L,1,3]
                    ideal_backbone_local     # [B,L,A,3]
                )

            # torsions and confidence
            s_initial = s0
            s_final = s
            torsions = self.torsion_head(s_initial, s_final, mask=seq_mask)
            plddt_logits, plddt = self.plddt_head(s)

            outputs = {
                "m": m,
                "z": z,
                "s": s,
                "R": R,
                "t": t,
                "backbone_coords": backbone_coords,
                "torsions": torsions,
                "plddt_logits": plddt_logits,
                "plddt": plddt,
                "distogram_logits": distogram_logits,
            }

            if recycle_idx < num_recycles:
                prev_pair = z.detach()
                prev_positions = self._extract_recycle_positions(backbone_coords, t).detach()

        return outputs
