import torch 
import torch.nn as nn

from model.losses.fape_loss import * 
from model.losses.pLDDT_loss import * 
from model.losses.distogram_loss import * 
from model.losses.torsion_loss import * 
from model.losses.loss_helpers import *

class AlphaFoldLoss(nn.Module):
    """
    Final AlphaFold-style loss orchestrator.

    Weighted sum:
        loss = (
            0.5  * fape_loss +
            0.3  * dist_loss +
            0.01 * plddt_loss +
            0.01 * torsion_loss
        )
    """

    def __init__(
        self,
        fape_length_scale=10.0,
        fape_clamp_distance=10.0,
        dist_num_bins=64,
        dist_min_bin=2.0,
        dist_max_bin=22.0,
        plddt_num_bins=50,
        plddt_inclusion_radius=15.0,
        w_fape=0.5,
        w_dist=0.3,
        w_plddt=0.01,
        w_torsion=0.01,):
      
        super().__init__()


        self.fape_loss_fn = FAPELoss(
            length_scale=fape_length_scale,
            clamp_distance=fape_clamp_distance)

        self.dist_loss_fn = DistogramLoss(
            num_bins=dist_num_bins,
            min_bin=dist_min_bin,
            max_bin=dist_max_bin,)

        self.plddt_loss_fn = PlddtLoss(
            num_bins=plddt_num_bins,
            inclusion_radius=plddt_inclusion_radius)

        self.torsion_loss_fn = TorsionLoss()

        self.w_fape = w_fape
        self.w_dist = w_dist
        self.w_plddt = w_plddt
        self.w_torsion = w_torsion

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
        fape_loss = self.fape_loss_fn(
            R_pred=R_pred,
            t_pred=t_pred,
            x_pred=x_pred,
            R_true=R_true,
            t_true=t_true,
            x_true=x_true,
            mask=backbone_mask,)

        dist_loss = self.dist_loss_fn(
            distogram_logits=out["distogram_logits"],
            x_true=coords_ca,
            mask=res_mask)

        plddt_loss = self.plddt_loss_fn(
            plddt_logits=out["plddt_logits"],
            x_pred=x_pred,
            x_true=coords_ca,
            mask=res_mask)

        if ("torsion_true" in batch) and ("torsion_mask" in batch):
            torsion_loss = self.torsion_loss_fn(
                torsion_pred=out["torsions"],
                torsion_true=batch["torsion_true"],
                torsion_mask=batch["torsion_mask"])
            
        else:
            torsion_loss = torch.zeros((), device=device, dtype=dtype)


        # Weighted total
        total_loss = (
            self.w_fape * fape_loss +
            self.w_dist * dist_loss +
            self.w_plddt * plddt_loss +
            self.w_torsion * torsion_loss)

        return {
            "loss": total_loss,
            "fape_loss": fape_loss.detach(),
            "dist_loss": dist_loss.detach(),
            "plddt_loss": plddt_loss.detach(),
            "torsion_loss": torsion_loss.detach()}