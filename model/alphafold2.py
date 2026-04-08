"""Top-level AlphaFold2-like model assembly.

This module wires the input embedder, Evoformer stack, structure module, and
output heads into a single PyTorch model that returns representations,
geometric predictions, backbone coordinates, torsions, confidence, and
distogram plus masked-MSA outputs.
"""

import torch 
import torch.nn as nn 
from model.evoformer_block import * 
from model.evoformer_stack import *

from model.alphafold2_heads import * 
from model.torsion_head import *
from model.structure_block import *
from model.recycling_module import RecyclingEmbedder
from model.template_stack import (
    TemplateStack,
    augment_msa_mask_with_template_mask,
    normalize_template_mask)

from model.extra_msa_stack import ExtraMsaStack
from model.msa_transitions import zero_init_linear

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
    @staticmethod
    def _normalize_ablation_id(ablation):
        if ablation is None:
            return None

        if isinstance(ablation, str):
            digits = "".join(character for character in ablation if character.isdigit())
            if digits == "":
                raise ValueError(f"Unsupported ablation identifier: {ablation}")
            return int(digits)

        return int(ablation)

    @classmethod
    def resolve_ablation_defaults(cls, ablation):
        ablation_id = cls._normalize_ablation_id(ablation)
        mapping = {
            None: {},
            1: {
                "evoformer_pair_stack_enabled": False,
                "recycle_single_enabled": False,
                "recycle_pair_enabled": False,
                "recycle_position_enabled": False,
                "plddt_head_enabled": False,
            },
            2: {
                "evoformer_triangle_attention_enabled": False,
                "recycle_single_enabled": False,
                "recycle_pair_enabled": False,
                "recycle_position_enabled": False,
                "plddt_head_enabled": False,
            },
            3: {
                "distogram_head_enabled": False,
                "masked_msa_head_enabled": False,
                "plddt_head_enabled": False,
                "torsion_head_enabled": False,
            },
            4: {
                "use_block_specific_params": True,
            },
            5: {
                "evoformer_enabled": False,
                "recycle_single_enabled": False,
                "recycle_pair_enabled": False,
                "recycle_position_enabled": False,
            },
        }
        if ablation_id not in mapping:
            valid = ", ".join(str(key) for key in sorted(key for key in mapping if key is not None))
            raise ValueError(f"Unsupported AlphaFold2 ablation '{ablation}'. Valid ids: {valid}")
        return mapping[ablation_id]

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
        masked_msa_num_classes=23,
        plddt_bins=50,
        n_torsions=7,
        num_res_blocks_torsion=2,
        recycle_min_bin=3.25,
        recycle_max_bin=20.75,
        recycle_dist_bins=15,
        ablation=None,
        evoformer_enabled=True,
        extra_msa_stack_enabled=True,
        template_stack_enabled=True,
        recycle_single_enabled=True,
        evoformer_pair_stack_enabled=True,
        evoformer_triangle_multiplication_enabled=True,
        evoformer_triangle_attention_enabled=True,
        evoformer_pair_transition_enabled=True,
        recycle_pair_enabled=True,
        recycle_position_enabled=True,
        extra_msa_dim=25,
        extra_msa_c_e=64,
        extra_msa_num_blocks=4,
        template_angle_dim=51,
        template_pair_dim=88,
        template_c_t=64,
        template_num_blocks=2,
        structure_pair_context_enabled=True,
        distogram_head_enabled=True,
        masked_msa_head_enabled=True,
        plddt_head_enabled=True,
        torsion_head_enabled=True):

        super().__init__()
        ablation_defaults = self.resolve_ablation_defaults(ablation)
        use_block_specific_params = ablation_defaults.get("use_block_specific_params", use_block_specific_params)
        evoformer_enabled = ablation_defaults.get("evoformer_enabled", evoformer_enabled)
        evoformer_pair_stack_enabled = ablation_defaults.get(
            "evoformer_pair_stack_enabled",
            evoformer_pair_stack_enabled,
        )
        evoformer_triangle_multiplication_enabled = ablation_defaults.get(
            "evoformer_triangle_multiplication_enabled",
            evoformer_triangle_multiplication_enabled,
        )
        evoformer_triangle_attention_enabled = ablation_defaults.get(
            "evoformer_triangle_attention_enabled",
            evoformer_triangle_attention_enabled,
        )
        evoformer_pair_transition_enabled = ablation_defaults.get(
            "evoformer_pair_transition_enabled",
            evoformer_pair_transition_enabled,
        )
        recycle_single_enabled = ablation_defaults.get("recycle_single_enabled", recycle_single_enabled)
        recycle_pair_enabled = ablation_defaults.get("recycle_pair_enabled", recycle_pair_enabled)
        recycle_position_enabled = ablation_defaults.get("recycle_position_enabled", recycle_position_enabled)
        structure_pair_context_enabled = ablation_defaults.get(
            "structure_pair_context_enabled",
            structure_pair_context_enabled,
        )
        distogram_head_enabled = ablation_defaults.get("distogram_head_enabled", distogram_head_enabled)
        masked_msa_head_enabled = ablation_defaults.get("masked_msa_head_enabled", masked_msa_head_enabled)
        plddt_head_enabled = ablation_defaults.get("plddt_head_enabled", plddt_head_enabled)
        torsion_head_enabled = ablation_defaults.get("torsion_head_enabled", torsion_head_enabled)

        self.ablation = self._normalize_ablation_id(ablation)
        self.c_z = c_z
        self.recycle_min_bin = float(recycle_min_bin)
        self.recycle_max_bin = float(recycle_max_bin)
        self.recycle_dist_bins = int(recycle_dist_bins)
        self.evoformer_enabled = bool(evoformer_enabled)
        self.extra_msa_stack_enabled = bool(extra_msa_stack_enabled)
        self.template_stack_enabled = bool(template_stack_enabled)
        self.recycle_single_enabled = bool(recycle_single_enabled)
        self.evoformer_pair_stack_enabled = bool(evoformer_pair_stack_enabled)
        self.evoformer_triangle_multiplication_enabled = bool(evoformer_triangle_multiplication_enabled)
        self.evoformer_triangle_attention_enabled = bool(evoformer_triangle_attention_enabled)
        self.evoformer_pair_transition_enabled = bool(evoformer_pair_transition_enabled)
        self.recycle_pair_enabled = bool(recycle_pair_enabled)
        self.recycle_position_enabled = bool(recycle_position_enabled)
        self.structure_pair_context_enabled = bool(structure_pair_context_enabled)
        self.distogram_head_enabled = bool(distogram_head_enabled)
        self.masked_msa_head_enabled = bool(masked_msa_head_enabled)
        self.plddt_head_enabled = bool(plddt_head_enabled)
        self.torsion_head_enabled = bool(torsion_head_enabled)


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
            c_z=c_z,
            transition_expansion=transition_expansion_evoformer,
            pair_stack_enabled=self.evoformer_pair_stack_enabled,
            triangle_multiplication_enabled=self.evoformer_triangle_multiplication_enabled,
            triangle_attention_enabled=self.evoformer_triangle_attention_enabled,
            pair_transition_enabled=self.evoformer_pair_transition_enabled)

        self.single_proj = SingleProjection(c_m=c_m, c_s=c_s)

        # Structure Model con IPA
        self.structure_module = StructureModule(
            c_s=c_s,
            c_z=c_z,
            num_blocks=num_structure_blocks,
            use_block_specific_params=use_block_specific_params)

        # Cabezas finales para entender el modelo
        self.plddt_head = PlddtHead(c_s=c_s, num_bins=plddt_bins)
        self.distogram_head = DistogramHead(c_z=c_z, num_bins=dist_bins)
        self.masked_msa_head = MaskedMsaHead(c_m=c_m, num_classes=masked_msa_num_classes)
        self.torsion_head = TorsionHead(c_s=c_s, n_torsions=n_torsions , num_res_blocks = num_res_blocks_torsion)
        self.recycling_embedder = RecyclingEmbedder(
            c_m=c_m,
            c_z=c_z,
            min_bin=self.recycle_min_bin,
            max_bin=self.recycle_max_bin,
            num_bins=self.recycle_dist_bins,
            recycle_single_enabled=self.recycle_single_enabled,
            recycle_pair_enabled=self.recycle_pair_enabled,
            recycle_position_enabled=self.recycle_position_enabled,
        )
        self.template_stack = TemplateStack(
            c_m=c_m,
            c_z=c_z,
            template_angle_dim=template_angle_dim,
            template_pair_dim=template_pair_dim,
            c_t=template_c_t,
            num_blocks=template_num_blocks,
        )
        self.extra_msa_stack = ExtraMsaStack(
            c_m=c_m,
            c_z=c_z,
            extra_dim=extra_msa_dim,
            c_e=extra_msa_c_e,
            num_blocks=extra_msa_num_blocks,
        )
        zero_init_linear(self.plddt_head.mlp[-1])
        zero_init_linear(self.distogram_head.linear)
        zero_init_linear(self.masked_msa_head.linear)

        self._freeze_module(self.evoformer, enabled=self.evoformer_enabled)
        self._freeze_module(self.extra_msa_stack, enabled=self.extra_msa_stack_enabled)
        self._freeze_module(self.template_stack, enabled=self.template_stack_enabled)
        self._freeze_module(self.recycling_embedder.single_norm, enabled=self.recycle_single_enabled)
        self._freeze_module(self.recycling_embedder.pair_norm, enabled=self.recycle_pair_enabled)
        self._freeze_module(self.recycling_embedder.pos_embedding, enabled=self.recycle_position_enabled)
        self._freeze_module(self.distogram_head, enabled=self.distogram_head_enabled)
        self._freeze_module(self.masked_msa_head, enabled=self.masked_msa_head_enabled)
        self._freeze_module(self.plddt_head, enabled=self.plddt_head_enabled)
        self._freeze_module(self.torsion_head, enabled=self.torsion_head_enabled)

    @staticmethod
    def _freeze_module(module, *, enabled):
        if enabled:
            return
        for parameter in module.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _get_target_row_mask(seq_mask=None, msa_mask=None):
        return RecyclingEmbedder.get_target_row_mask(seq_mask=seq_mask, msa_mask=msa_mask)

    def _apply_recycle_single_update(self, m, prev_m1, row_mask=None):
        return self.recycling_embedder._apply_single_recycle(m, prev_m1=prev_m1, row_mask=row_mask)

    def _apply_recycle_pair_update(self, z, prev_pair, pair_mask=None):
        return self.recycling_embedder._apply_pair_recycle(z, prev_z=prev_pair, pair_mask=pair_mask)

    def _positions_to_recycle_dgram(self, positions, dtype, pair_mask=None):
        return self.recycling_embedder._positions_to_dgram_update(
            positions,
            dtype=dtype,
            pair_mask=pair_mask,
        )

    @staticmethod
    def _backbone_to_pseudo_beta(backbone_coords, seq_tokens=None):
        return RecyclingEmbedder.backbone_to_pseudo_beta(backbone_coords, seq_tokens=seq_tokens)

    def _extract_recycle_positions(self, seq_tokens, backbone_coords, t):
        return RecyclingEmbedder.extract_prev_positions(
            seq_tokens=seq_tokens,
            backbone_coords=backbone_coords,
            t=t,
        )

    def _build_structure_pair_input(self, z):
        if self.structure_pair_context_enabled:
            return z
        return torch.zeros_like(z)


    def forward(
        self,
        seq_tokens,
        msa_tokens,
        seq_mask=None,
        msa_mask=None,
        ideal_backbone_local=None,
        num_recycles: int = 0,
        extra_msa_feat=None,
        extra_msa_mask=None,
        template_angle_feat=None,
        template_pair_feat=None,
        template_mask=None,
    ):
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
        target_row_mask = self._get_target_row_mask(seq_mask=seq_mask, msa_mask=msa_mask)

        num_recycles = max(0, int(num_recycles))
        prev_m1 = None
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

            m, z = self.recycling_embedder(
                m,
                z,
                prev_m1=prev_m1,
                prev_z=prev_pair,
                prev_positions=prev_positions,
                seq_mask=seq_mask,
                msa_mask=msa_mask,
            )

            evoformer_msa_mask = msa_mask
            original_msa_depth = m.shape[1]

            if self.template_stack_enabled and (
                template_angle_feat is not None or template_pair_feat is not None):

                template_count = (
                    template_angle_feat.shape[1]
                    if template_angle_feat is not None
                    else template_pair_feat.shape[1])
                
                template_row_mask = normalize_template_mask(
                    template_mask,
                    batch_size=m.shape[0],
                    num_templates=template_count,
                    length=m.shape[2],
                    device=m.device,
                    dtype=m.dtype,)
                
                m, z = self.template_stack(
                    m,
                    z,
                    template_angle_feat=template_angle_feat,
                    template_pair_feat=template_pair_feat,
                    template_mask=template_row_mask)
                
                if template_angle_feat is not None:
                    base_msa_mask = msa_mask
                    if base_msa_mask is None:
                        base_msa_mask = torch.ones(
                            m.shape[0],
                            original_msa_depth,
                            m.shape[2],
                            device=m.device,
                            dtype=m.dtype)
                        
                    evoformer_msa_mask = augment_msa_mask_with_template_mask(
                        base_msa_mask,
                        template_row_mask,
                        length=m.shape[2])

            if self.extra_msa_stack_enabled and extra_msa_feat is not None:
                m, z = self.extra_msa_stack(
                    m,
                    z,
                    extra_msa_feat=extra_msa_feat,
                    seq_mask=seq_mask,
                    extra_msa_mask=extra_msa_mask)

            # evoformer
            if self.evoformer_enabled:
                m, z = self.evoformer(
                    m,
                    z,
                    msa_mask=evoformer_msa_mask,
                    pair_mask=pair_mask,)

            # z before structure for distogram
            distogram_logits = self.distogram_head(z) if self.distogram_head_enabled else None
            masked_msa_logits = None
            if self.masked_msa_head_enabled:
                masked_msa_logits = self.masked_msa_head(m[:, :original_msa_depth])

            # single repr + structure
            s0 = self.single_proj(m)
            structure_pair = self._build_structure_pair_input(z)
            s, R, t, structure_intermediates = self.structure_module(
                s0,
                structure_pair,
                mask=seq_mask,
                return_intermediates=True)

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
            torsions = self.torsion_head(s_initial, s_final, mask=seq_mask) if self.torsion_head_enabled else None
            aux_torsions = None
            if self.torsion_head_enabled:
                aux_torsions = torch.stack(
                    [
                        self.torsion_head(s_initial, s_block, mask=seq_mask)
                        for s_block in structure_intermediates["single"]
                    ],
                    dim=0,
                )
            if self.plddt_head_enabled:
                plddt_logits, plddt = self.plddt_head(s)
            else:
                plddt_logits, plddt = None, None

            outputs = {
                "m": m,
                "z": z,
                "s": s,
                "R": R,
                "t": t,
                "backbone_coords": backbone_coords,
                "torsions": torsions,
                "aux_R": structure_intermediates["R"],
                "aux_t": structure_intermediates["t"],
                "aux_torsions": aux_torsions,
                "plddt_logits": plddt_logits,
                "plddt": plddt,
                "distogram_logits": distogram_logits,
                "masked_msa_logits": masked_msa_logits,
            }

            if recycle_idx < num_recycles:
                prev_m1 = m[:, 0, :, :].detach()
                prev_pair = z.detach()
                prev_positions = self._extract_recycle_positions(
                    seq_tokens=seq_tokens,
                    backbone_coords=backbone_coords,
                    t=t,
                ).detach()

        return outputs
