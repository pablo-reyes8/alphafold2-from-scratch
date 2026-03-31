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
        n_torsions=7 , num_res_blocks_torsion = 2):

        super().__init__()


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


    def forward(
        self,
        seq_tokens,
        msa_tokens,
        seq_mask=None,
        msa_mask=None,
        ideal_backbone_local=None):
        """
        ideal_backbone_local: [A, 3] or [1,1,A,3] or [B,L,A,3]
          e.g. local ideal coordinates for backbone atoms (N, CA, C, O)

        returns dict
        """
        if seq_mask is not None:
            pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]
        else:
            pair_mask = None

        # input
        m, z = self.input_embedder(
            seq_tokens=seq_tokens,
            msa_tokens=msa_tokens,
            seq_mask=seq_mask,
            msa_mask=msa_mask)

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

        return {
            "m": m,
            "z": z,
            "s": s,
            "R": R,
            "t": t,
            "backbone_coords": backbone_coords,
            "torsions": torsions,
            "plddt_logits": plddt_logits,
            "plddt": plddt,
            "distogram_logits": distogram_logits}