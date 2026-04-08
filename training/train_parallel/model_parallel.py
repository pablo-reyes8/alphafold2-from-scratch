"""Two-stage model-parallel wrappers for the AlphaFold2 top-level model.

This module keeps Evoformer and recycling updates on a first device, moves the
structure module plus heads to a second device, and preserves the original
state-dict surface so checkpoints remain compatible with the plain model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ipa_transformations import apply_transform
from model.recycling_module import RecyclingEmbedder
from model.template_stack import augment_msa_mask_with_template_mask, normalize_template_mask


class AlphaFold2ModelParallel(nn.Module):
    """Wrap an existing ``AlphaFold2`` instance into a simple two-stage pipeline."""

    def __init__(self, model: nn.Module, stage_devices: tuple[str | torch.device, ...]):
        super().__init__()
        if len(stage_devices) == 0:
            raise ValueError("stage_devices must contain at least one device.")

        devices = tuple(torch.device(device) for device in stage_devices)
        self.stage_devices = devices
        self.input_device = devices[0]
        self.output_device = devices[-1]

        self.c_z = model.c_z
        self.recycle_min_bin = float(model.recycle_min_bin)
        self.recycle_max_bin = float(model.recycle_max_bin)
        self.recycle_dist_bins = int(model.recycle_dist_bins)
        self.evoformer_enabled = bool(getattr(model, "evoformer_enabled", True))
        self.extra_msa_stack_enabled = bool(getattr(model, "extra_msa_stack_enabled", True))
        self.template_stack_enabled = bool(getattr(model, "template_stack_enabled", True))
        self.recycle_single_enabled = bool(getattr(model, "recycle_single_enabled", True))
        self.recycle_pair_enabled = bool(getattr(model, "recycle_pair_enabled", True))
        self.recycle_position_enabled = bool(getattr(model, "recycle_position_enabled", True))
        self.structure_pair_context_enabled = bool(getattr(model, "structure_pair_context_enabled", True))
        self.distogram_head_enabled = bool(getattr(model, "distogram_head_enabled", True))
        self.masked_msa_head_enabled = bool(getattr(model, "masked_msa_head_enabled", True))
        self.plddt_head_enabled = bool(getattr(model, "plddt_head_enabled", True))
        self.torsion_head_enabled = bool(getattr(model, "torsion_head_enabled", True))

        self.input_embedder = model.input_embedder.to(self.input_device)
        self.evoformer = model.evoformer.to(self.input_device)
        self.recycling_embedder = model.recycling_embedder.to(self.input_device)
        self.template_stack = model.template_stack.to(self.input_device)
        self.extra_msa_stack = model.extra_msa_stack.to(self.input_device)

        self.single_proj = model.single_proj.to(self.output_device)
        self.structure_module = model.structure_module.to(self.output_device)
        self.plddt_head = model.plddt_head.to(self.output_device)
        self.distogram_head = model.distogram_head.to(self.output_device)
        self.masked_msa_head = model.masked_msa_head.to(self.output_device)
        self.torsion_head = model.torsion_head.to(self.output_device)

    def _to_input_device(self, tensor):
        if tensor is None:
            return None
        return tensor.to(self.input_device, non_blocking=True)

    def _to_output_device(self, tensor):
        if tensor is None:
            return None
        return tensor.to(self.output_device, non_blocking=True)

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
        seq_tokens = self._to_input_device(seq_tokens)
        msa_tokens = self._to_input_device(msa_tokens)
        seq_mask_input = self._to_input_device(seq_mask)
        msa_mask_input = self._to_input_device(msa_mask)
        extra_msa_feat_input = self._to_input_device(extra_msa_feat)
        extra_msa_mask_input = self._to_input_device(extra_msa_mask)
        template_angle_feat_input = self._to_input_device(template_angle_feat)
        template_pair_feat_input = self._to_input_device(template_pair_feat)
        template_mask_input = self._to_input_device(template_mask)

        if seq_mask_input is not None:
            pair_mask_input = seq_mask_input[:, :, None] * seq_mask_input[:, None, :]
        else:
            pair_mask_input = None

        num_recycles = max(0, int(num_recycles))
        prev_m1 = None
        prev_pair = None
        prev_positions = None
        outputs = None

        for recycle_idx in range(num_recycles + 1):
            m, z = self.input_embedder(
                seq_tokens=seq_tokens,
                msa_tokens=msa_tokens,
                seq_mask=seq_mask_input,
                msa_mask=msa_mask_input,
            )

            m, z = self.recycling_embedder(
                m,
                z,
                prev_m1=prev_m1,
                prev_z=prev_pair,
                prev_positions=prev_positions,
                seq_mask=seq_mask_input,
                msa_mask=msa_mask_input,
            )

            evoformer_msa_mask_input = msa_mask_input
            original_msa_depth = m.shape[1]

            if self.template_stack_enabled and (
                template_angle_feat_input is not None or template_pair_feat_input is not None
            ):
                template_count = (
                    template_angle_feat_input.shape[1]
                    if template_angle_feat_input is not None
                    else template_pair_feat_input.shape[1]
                )
                template_row_mask_input = normalize_template_mask(
                    template_mask_input,
                    batch_size=m.shape[0],
                    num_templates=template_count,
                    length=m.shape[2],
                    device=m.device,
                    dtype=m.dtype,
                )
                m, z = self.template_stack(
                    m,
                    z,
                    template_angle_feat=template_angle_feat_input,
                    template_pair_feat=template_pair_feat_input,
                    template_mask=template_row_mask_input,
                )
                if template_angle_feat_input is not None:
                    base_msa_mask_input = msa_mask_input
                    if base_msa_mask_input is None:
                        base_msa_mask_input = torch.ones(
                            m.shape[0],
                            original_msa_depth,
                            m.shape[2],
                            device=m.device,
                            dtype=m.dtype,
                        )
                    evoformer_msa_mask_input = augment_msa_mask_with_template_mask(
                        base_msa_mask_input,
                        template_row_mask_input,
                        length=m.shape[2],
                    )

            if self.extra_msa_stack_enabled and extra_msa_feat_input is not None:
                m, z = self.extra_msa_stack(
                    m,
                    z,
                    extra_msa_feat=extra_msa_feat_input,
                    seq_mask=seq_mask_input,
                    extra_msa_mask=extra_msa_mask_input,
                )

            if self.evoformer_enabled:
                m, z = self.evoformer(
                    m,
                    z,
                    msa_mask=evoformer_msa_mask_input,
                    pair_mask=pair_mask_input,
                )

            m_output = self._to_output_device(m)
            z_output = self._to_output_device(z)
            seq_mask_output = self._to_output_device(seq_mask_input)

            distogram_logits = self.distogram_head(z_output) if self.distogram_head_enabled else None
            masked_msa_logits = None
            if self.masked_msa_head_enabled:
                masked_msa_logits = self.masked_msa_head(m_output[:, :original_msa_depth])
            s0 = self.single_proj(m_output)
            structure_pair = self._build_structure_pair_input(z_output)
            s, R, t, structure_intermediates = self.structure_module(
                s0,
                structure_pair,
                mask=seq_mask_output,
                return_intermediates=True,
            )

            backbone_coords = None
            if ideal_backbone_local is not None:
                ideal_backbone_output = self._to_output_device(ideal_backbone_local)
                if ideal_backbone_output.dim() == 2:
                    ideal_backbone_output = ideal_backbone_output.unsqueeze(0).unsqueeze(0)
                elif ideal_backbone_output.dim() != 4:
                    raise ValueError("ideal_backbone_local must have shape [A,3] or [B,L,A,3]")

                if ideal_backbone_output.shape[0] == 1 and ideal_backbone_output.shape[1] == 1:
                    batch_size, length = seq_tokens.shape
                    ideal_backbone_output = ideal_backbone_output.expand(batch_size, length, -1, -1)

                backbone_coords = apply_transform(
                    R[:, :, None, :, :],
                    t[:, :, None, :],
                    ideal_backbone_output,
                )

            torsions = self.torsion_head(s0, s, mask=seq_mask_output) if self.torsion_head_enabled else None
            aux_torsions = None
            if self.torsion_head_enabled:
                aux_torsions = torch.stack(
                    [
                        self.torsion_head(s0, s_block, mask=seq_mask_output)
                        for s_block in structure_intermediates["single"]
                    ],
                    dim=0,
                )
            if self.plddt_head_enabled:
                plddt_logits, plddt = self.plddt_head(s)
            else:
                plddt_logits, plddt = None, None

            outputs = {
                "m": m_output,
                "z": z_output,
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
                    seq_tokens=self._to_output_device(seq_tokens),
                    backbone_coords=backbone_coords,
                    t=t,
                ).detach()
                prev_positions = self._to_input_device(prev_positions)

        return outputs


def build_model_parallel_wrapper(
    model: nn.Module,
    stage_devices: tuple[str | torch.device, ...],
) -> AlphaFold2ModelParallel:
    """Create a two-stage model-parallel wrapper around an existing model."""
    return AlphaFold2ModelParallel(model=model, stage_devices=stage_devices)
