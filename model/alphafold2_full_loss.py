"""Composite AlphaFold-style loss orchestration.

This module combines final-structure FAPE, intermediate structure auxiliary
losses, distogram, masked-MSA, pLDDT, and torsion supervision into the single
loss dictionary consumed by the training loop. It acts as the contract between
model outputs and the structural targets prepared by the dataloader.
"""

import torch 
import torch.nn as nn

from model.losses.fape_loss import * 
from model.losses.pLDDT_loss import * 
from model.losses.distogram_loss import * 
from model.losses.masked_msa_loss import *
from model.losses.torsion_loss import * 
from model.losses.loss_helpers import *
from model.losses.structure_aux_loss import StructureAuxLoss

class AlphaFoldLoss(nn.Module):
    """
    Final AlphaFold-style loss orchestrator.

    Weighted sum:
        loss = (
            0.5  * fape_loss +
            0.5  * aux_loss +
            0.3  * dist_loss +
            2.0  * msa_loss +
            0.01 * plddt_loss +
            0.01 * torsion_loss
        )
    """
    @staticmethod
    def _normalize_ablation_id(ablation):
        if ablation is None:
            return None

        if isinstance(ablation, str):
            digits = "".join(character for character in ablation if character.isdigit())
            if digits == "":
                raise ValueError(f"Unsupported loss ablation identifier: {ablation}")
            return int(digits)

        return int(ablation)

    @classmethod
    def resolve_ablation_defaults(cls, ablation):
        ablation_id = cls._normalize_ablation_id(ablation)
        mapping = {
            None: {},
            1: {"w_plddt": 0.0},
            2: {"w_plddt": 0.0},
            3: {"w_aux": 0.0, "w_dist": 0.0, "w_msa": 0.0, "w_plddt": 0.0, "w_torsion": 0.0},
            4: {},
            5: {},
        }
        if ablation_id not in mapping:
            valid = ", ".join(str(key) for key in sorted(key for key in mapping if key is not None))
            raise ValueError(f"Unsupported AlphaFoldLoss ablation '{ablation}'. Valid ids: {valid}")
        return mapping[ablation_id]

    def __init__(
        self,
        fape_length_scale=10.0,
        fape_clamp_distance=10.0,
        dist_num_bins=64,
        dist_min_bin=2.0,
        dist_max_bin=22.0,
        msa_num_classes=23,
        plddt_num_bins=50,
        plddt_inclusion_radius=15.0,
        ablation=None,
        w_fape=0.5,
        w_aux=0.5,
        w_dist=0.3,
        w_msa=2.0,
        w_plddt=0.01,
        w_torsion=0.01,):
      
        super().__init__()
        ablation_defaults = self.resolve_ablation_defaults(ablation)
        self.ablation = self._normalize_ablation_id(ablation)
        w_fape = ablation_defaults.get("w_fape", w_fape)
        w_aux = ablation_defaults.get("w_aux", w_aux)
        w_dist = ablation_defaults.get("w_dist", w_dist)
        w_msa = ablation_defaults.get("w_msa", w_msa)
        w_plddt = ablation_defaults.get("w_plddt", w_plddt)
        w_torsion = ablation_defaults.get("w_torsion", w_torsion)


        self.fape_loss_fn = FAPELoss(
            length_scale=fape_length_scale,
            clamp_distance=fape_clamp_distance)

        self.dist_loss_fn = DistogramLoss(
            num_bins=dist_num_bins,
            min_bin=dist_min_bin,
            max_bin=dist_max_bin,)

        self.msa_loss_fn = MaskedMsaLoss(num_classes=msa_num_classes)

        self.plddt_loss_fn = PlddtLoss(
            num_bins=plddt_num_bins,
            inclusion_radius=plddt_inclusion_radius)

        self.torsion_loss_fn = TorsionLoss()
        self.aux_loss_fn = StructureAuxLoss(
            fape_length_scale=fape_length_scale,
            fape_clamp_distance=fape_clamp_distance,
            fape_eps=1e-12,
        )

        self.w_fape = w_fape
        self.w_aux = w_aux
        self.w_dist = w_dist
        self.w_msa = w_msa
        self.w_plddt = w_plddt
        self.w_torsion = w_torsion

    @staticmethod
    def _zero_scalar(*, device, dtype):
        return torch.zeros((), device=device, dtype=dtype)

    def forward(self, out, batch):
        """
        out: dict from model forward
        batch: dict with ground truth

        Required in out:
            "R", "t", "distogram_logits", "plddt_logits", "torsions"

        Required in batch:
            "coords_n", "coords_ca", "coords_c",
            "valid_res_mask", "valid_backbone_mask"

        Optional in batch:
            "torsion_true", "torsion_mask"
        """

        # Ground truth coordinates
        coords_n = batch["coords_n"]                       # [B,L,3]
        coords_ca = batch["coords_ca"]                     # [B,L,3]
        coords_c = batch["coords_c"]                       # [B,L,3]

        res_mask = batch["valid_res_mask"]                 # [B,L]
        backbone_mask = batch["valid_backbone_mask"]       # [B,L]

        device = coords_ca.device
        dtype = coords_ca.dtype

        # Predicted structure
        R_pred = out["R"]                                  # [B,L,3,3]
        t_pred = out["t"]                                  # [B,L,3]

        # PoC: use predicted translation as predicted CA
        if out.get("backbone_coords", None) is not None:
            x_pred = out["backbone_coords"][:, :, 1, :]
        else:
            x_pred = t_pred                                  # [B,L,3]


        # True structure: canonical backbone frames
        R_true, t_true = build_backbone_frames(
            coords_n=coords_n,
            coords_ca=coords_ca,
            coords_c=coords_c,
            mask=backbone_mask,)

        x_true = coords_ca                                 # [B,L,3]

        # -----------------------------
        # Component losses
        # -----------------------------
        zero = self._zero_scalar(device=device, dtype=dtype)

        if self.w_fape > 0.0:
            fape_loss = self.fape_loss_fn(
                R_pred=R_pred,
                t_pred=t_pred,
                x_pred=x_pred,
                R_true=R_true,
                t_true=t_true,
                x_true=x_true,
                mask=backbone_mask,)
        else:
            fape_loss = zero

        if self.w_aux > 0.0:
            aux_loss_dict = self.aux_loss_fn(
                R_blocks=out.get("aux_R", None),
                t_blocks=out.get("aux_t", None),
                R_true=R_true,
                t_true=t_true,
                coords_ca=coords_ca,
                backbone_mask=backbone_mask,
                torsion_blocks=out.get("aux_torsions", None),
                torsion_true=batch.get("torsion_true", None),
                torsion_mask=batch.get("torsion_mask", None),
            )
            aux_loss = aux_loss_dict["aux_loss"]
            aux_fape_loss = aux_loss_dict["aux_fape_loss"]
            aux_torsion_loss = aux_loss_dict["aux_torsion_loss"]
        else:
            aux_loss = zero
            aux_fape_loss = zero
            aux_torsion_loss = zero

        if (self.w_dist > 0.0) and (out.get("distogram_logits", None) is not None):
            dist_loss = self.dist_loss_fn(
                distogram_logits=out["distogram_logits"],
                x_true=coords_ca,
                mask=res_mask)
        else:
            dist_loss = zero

        if (
            (self.w_msa > 0.0)
            and (out.get("masked_msa_logits", None) is not None)
            and ("masked_msa_true" in batch)
            and ("masked_msa_mask" in batch)
        ):
            msa_loss = self.msa_loss_fn(
                masked_msa_logits=out["masked_msa_logits"],
                masked_msa_true=batch["masked_msa_true"],
                masked_msa_mask=batch["masked_msa_mask"],
            )
        else:
            msa_loss = zero

        if (self.w_plddt > 0.0) and (out.get("plddt_logits", None) is not None):
            plddt_loss = self.plddt_loss_fn(
                plddt_logits=out["plddt_logits"],
                x_pred=x_pred,
                x_true=coords_ca,
                mask=res_mask)
        else:
            plddt_loss = zero

        if (
            (self.w_torsion > 0.0)
            and (out.get("torsions", None) is not None)
            and ("torsion_true" in batch)
            and ("torsion_mask" in batch)
        ):
            torsion_loss = self.torsion_loss_fn(
                torsion_pred=out["torsions"],
                torsion_true=batch["torsion_true"],
                torsion_mask=batch["torsion_mask"])
            
        else:
            torsion_loss = zero


        # Weighted total
        total_loss = (
            self.w_fape * fape_loss +
            self.w_aux * aux_loss +
            self.w_dist * dist_loss +
            self.w_msa * msa_loss +
            self.w_plddt * plddt_loss +
            self.w_torsion * torsion_loss)

        return {
            "loss": total_loss,
            "fape_loss": fape_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "aux_fape_loss": aux_fape_loss.detach(),
            "aux_torsion_loss": aux_torsion_loss.detach(),
            "dist_loss": dist_loss.detach(),
            "msa_loss": msa_loss.detach(),
            "plddt_loss": plddt_loss.detach(),
            "torsion_loss": torsion_loss.detach()}
