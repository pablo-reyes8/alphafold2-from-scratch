import torch 
from data.dataloaders import *

def collate_proteins(batch):
    B = len(batch)
    max_L = max(item["seq_tokens"].shape[0] for item in batch)
    max_Nmsa = max(item["msa_tokens"].shape[0] for item in batch)

    seq_pad_token = AA_VOCAB["-"]
    msa_pad_token = AA_VOCAB["-"]

    seq_tokens = torch.full((B, max_L), seq_pad_token, dtype=torch.long)
    seq_mask = torch.zeros((B, max_L), dtype=torch.float32)

    msa_tokens = torch.full((B, max_Nmsa, max_L), msa_pad_token, dtype=torch.long)
    msa_mask = torch.zeros((B, max_Nmsa, max_L), dtype=torch.float32)

    coords_n = torch.zeros((B, max_L, 3), dtype=torch.float32)
    coords_ca = torch.zeros((B, max_L, 3), dtype=torch.float32)
    coords_c = torch.zeros((B, max_L, 3), dtype=torch.float32)

    valid_res_mask = torch.zeros((B, max_L), dtype=torch.float32)
    valid_backbone_mask = torch.zeros((B, max_L), dtype=torch.float32)

    dist_map = torch.zeros((B, max_L, max_L), dtype=torch.float32)
    pair_mask = torch.zeros((B, max_L, max_L), dtype=torch.float32)
    backbone_pair_mask = torch.zeros((B, max_L, max_L), dtype=torch.float32)

    torsion_true = torch.zeros((B, max_L, 3, 2), dtype=torch.float32)
    torsion_mask = torch.zeros((B, max_L, 3), dtype=torch.float32)

    ids = []
    msa_chain_ids = []
    matched_chain_ids = []
    sequence_strs = []
    match_identity = torch.zeros(B, dtype=torch.float32)

    for i, item in enumerate(batch):
        L = item["seq_tokens"].shape[0]
        N = item["msa_tokens"].shape[0]

        seq_tokens[i, :L] = item["seq_tokens"]
        seq_mask[i, :L] = 1.0

        msa_tokens[i, :N, :L] = item["msa_tokens"]
        msa_mask[i, :N, :L] = item["msa_mask"]

        coords_n[i, :L] = item["coords_n"]
        coords_ca[i, :L] = item["coords_ca"]
        coords_c[i, :L] = item["coords_c"]

        valid_res_mask[i, :L] = item["valid_res_mask"]
        valid_backbone_mask[i, :L] = item["valid_backbone_mask"]

        dist_map[i, :L, :L] = item["dist_map"]

        pair_mask[i, :L, :L] = (
            item["valid_res_mask"][:, None] * item["valid_res_mask"][None, :]
        )

        backbone_pair_mask[i, :L, :L] = (
            item["valid_backbone_mask"][:, None] * item["valid_backbone_mask"][None, :]
        )

        torsion_true[i, :L] = item["torsion_true"]
        torsion_mask[i, :L] = item["torsion_mask"]

        ids.append(item["id"])
        msa_chain_ids.append(item["msa_chain_id"])
        matched_chain_ids.append(item["matched_chain_id"])
        sequence_strs.append(item["sequence_str"])
        match_identity[i] = item["match_identity"]

    return {
        "id": ids,
        "msa_chain_id": msa_chain_ids,
        "matched_chain_id": matched_chain_ids,
        "match_identity": match_identity,
        "sequence_str": sequence_strs,
        "seq_tokens": seq_tokens,                     # [B, L]
        "seq_mask": seq_mask,                         # [B, L]
        "msa_tokens": msa_tokens,                     # [B, N_msa, L]
        "msa_mask": msa_mask,                         # [B, N_msa, L]
        "coords_n": coords_n,                         # [B, L, 3]
        "coords_ca": coords_ca,                       # [B, L, 3]
        "coords_c": coords_c,                         # [B, L, 3]
        "dist_map": dist_map,                         # [B, L, L]
        "valid_res_mask": valid_res_mask,             # [B, L]
        "valid_backbone_mask": valid_backbone_mask,   # [B, L]
        "pair_mask": pair_mask,                       # [B, L, L]
        "backbone_pair_mask": backbone_pair_mask,     # [B, L, L]
        "torsion_true": torsion_true,                 # [B, L, 3, 2]
        "torsion_mask": torsion_mask,                 # [B, L, 3]
    }