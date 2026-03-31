import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
from difflib import SequenceMatcher
from Bio import Align


AA_VOCAB = {
    "-": 0,
    "A": 1, "R": 2, "N": 3, "D": 4, "C": 5,
    "Q": 6, "E": 7, "G": 8, "H": 9, "I": 10,
    "L": 11, "K": 12, "M": 13, "F": 14, "P": 15,
    "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20,
    "X": 21, "B": 22, "Z": 23, "U": 24, "O": 25,
    ".": 26
}

UNK_TOKEN = AA_VOCAB["X"]


def tokenize_sequence(seq: str) -> torch.Tensor:
    return torch.tensor([AA_VOCAB.get(ch.upper(), UNK_TOKEN) for ch in seq], dtype=torch.long)


def read_a3m(a3m_path: Path, max_msa_seqs: Optional[int] = None) -> List[str]:
    seqs = []
    with open(a3m_path, "r") as f:
        current_name = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    seq = "".join(current_seq)
                    seq = "".join([c for c in seq if not c.islower()])
                    seqs.append(seq)
                    if max_msa_seqs is not None and len(seqs) >= max_msa_seqs:
                        break
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if (max_msa_seqs is None or len(seqs) < max_msa_seqs) and current_name is not None:
            seq = "".join(current_seq)
            seq = "".join([c for c in seq if not c.islower()])
            seqs.append(seq)

    return seqs


def pad_or_crop_msa(msa_seqs: List[str], target_len: int, max_msa_seqs: int) -> List[str]:
    msa_seqs = msa_seqs[:max_msa_seqs]
    fixed = []

    for s in msa_seqs:
        if len(s) < target_len:
            s = s + "-" * (target_len - len(s))
        elif len(s) > target_len:
            s = s[:target_len]
        fixed.append(s)

    if len(fixed) == 0:
        fixed = ["-" * target_len]

    return fixed


def tokenize_msa(msa_seqs: List[str]) -> torch.Tensor:
    return torch.stack([tokenize_sequence(s) for s in msa_seqs], dim=0)


def safe_residue_to_aa(residue) -> str:
    resname = residue.get_resname().strip()
    try:
        aa = seq1(resname)
        if len(aa) == 1:
            return aa
    except Exception:
        pass
    return "X"


def extract_chain_sequences_and_backbone(cif_path: Path) -> Dict[str, Dict]:
    """
    Extract per-chain sequence plus backbone atom coordinates (N, CA, C).

    Returns
    -------
    out[chain_id] = {
        "sequence": str,
        "coords_n":  np.ndarray [L, 3],
        "coords_ca": np.ndarray [L, 3],
        "coords_c":  np.ndarray [L, 3],
    }
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))

    out = {}
    first_model = next(structure.get_models())

    for chain in first_model:
        seq_chars = []
        coords_n = []
        coords_ca = []
        coords_c = []

        for residue in chain:
            hetflag = residue.id[0]
            if hetflag.strip() != "":
                continue

            aa = safe_residue_to_aa(residue)
            seq_chars.append(aa)

            if "N" in residue:
                coords_n.append(residue["N"].coord)
            else:
                coords_n.append([np.nan, np.nan, np.nan])

            if "CA" in residue:
                coords_ca.append(residue["CA"].coord)
            else:
                coords_ca.append([np.nan, np.nan, np.nan])

            if "C" in residue:
                coords_c.append(residue["C"].coord)
            else:
                coords_c.append([np.nan, np.nan, np.nan])

        if len(seq_chars) == 0:
            continue

        out[chain.id] = {
            "sequence": "".join(seq_chars),
            "coords_n": np.array(coords_n, dtype=np.float32),
            "coords_ca": np.array(coords_ca, dtype=np.float32),
            "coords_c": np.array(coords_c, dtype=np.float32),
        }

    return out


def sequence_identity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_target_to_chain(
    target_seq: str,
    chain_data: Dict[str, Dict],
    min_identity: float = 0.85,
) -> Optional[Tuple[str, float]]:
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"

    best_chain = None
    best_score = -1.0

    for chain_id, info in chain_data.items():
        chain_seq = info["sequence"]
        if len(chain_seq) == 0:
            continue

        score = aligner.score(target_seq, chain_seq)
        coverage = score / len(chain_seq)

        if coverage > best_score:
            best_score = coverage
            best_chain = chain_id

    if best_chain is None or best_score < min_identity:
        return None

    return best_chain, best_score


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    diff = coords[:, None, :] - coords[None, :, :]
    return torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)


def dihedral_angle(p0, p1, p2, p3, eps=1e-8):
    """
    Compute dihedral angle (radians) from 4 points in R^3.
    """
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)
    if b1_norm < eps:
        return np.nan
    b1u = b1 / b1_norm

    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u

    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm < eps or w_norm < eps:
        return np.nan

    x = np.dot(v, w)
    y = np.dot(np.cross(b1u, v), w)

    return np.arctan2(y, x)


def backbone_torsions_from_coords(coords_n, coords_ca, coords_c, valid_backbone_mask, eps=1e-8):
    """
    Compute phi, psi, omega from backbone coordinates.

    Returns
    -------
    torsion_true : torch.Tensor [L, 3, 2]
        order = [phi, psi, omega], encoded as (sin, cos)
    torsion_mask : torch.Tensor [L, 3]
    """
    L = coords_n.shape[0]

    torsion_true = np.zeros((L, 3, 2), dtype=np.float32)
    torsion_mask = np.zeros((L, 3), dtype=np.float32)

    # phi_i   = dihedral(C_{i-1}, N_i,  CA_i, C_i)
    # psi_i   = dihedral(N_i,     CA_i, C_i,  N_{i+1})
    # omega_i = dihedral(CA_i,    C_i,  N_{i+1}, CA_{i+1})

    for i in range(L):
        # phi
        if i > 0 and valid_backbone_mask[i - 1] and valid_backbone_mask[i]:
            ang = dihedral_angle(
                coords_c[i - 1],
                coords_n[i],
                coords_ca[i],
                coords_c[i],
                eps=eps,
            )
            if np.isfinite(ang):
                torsion_true[i, 0, 0] = np.sin(ang)
                torsion_true[i, 0, 1] = np.cos(ang)
                torsion_mask[i, 0] = 1.0

        # psi
        if i < L - 1 and valid_backbone_mask[i] and valid_backbone_mask[i + 1]:
            ang = dihedral_angle(
                coords_n[i],
                coords_ca[i],
                coords_c[i],
                coords_n[i + 1],
                eps=eps,
            )
            if np.isfinite(ang):
                torsion_true[i, 1, 0] = np.sin(ang)
                torsion_true[i, 1, 1] = np.cos(ang)
                torsion_mask[i, 1] = 1.0

        # omega
        if i < L - 1 and valid_backbone_mask[i] and valid_backbone_mask[i + 1]:
            ang = dihedral_angle(
                coords_ca[i],
                coords_c[i],
                coords_n[i + 1],
                coords_ca[i + 1],
                eps=eps,
            )
            if np.isfinite(ang):
                torsion_true[i, 2, 0] = np.sin(ang)
                torsion_true[i, 2, 1] = np.cos(ang)
                torsion_mask[i, 2] = 1.0

    return (
        torch.tensor(torsion_true, dtype=torch.float32),
        torch.tensor(torsion_mask, dtype=torch.float32),
    )


class FoldbenchProteinDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        msa_root: str,
        cif_root: str,
        max_msa_seqs: int = 128,
        use_a3m_name: str = "cfdb_hits.a3m",
        max_samples: Optional[int] = None,
        min_identity: float = 0.90,
        verbose: bool = True,
    ):
        self.json_path = Path(json_path)
        self.msa_root = Path(msa_root)
        self.cif_root = Path(cif_root)
        self.max_msa_seqs = max_msa_seqs
        self.use_a3m_name = use_a3m_name
        self.min_identity = min_identity

        with open(self.json_path, "r") as f:
            data = json.load(f)

        rows = []
        dropped = []

        for qname, q in data["queries"].items():
            chain = q["chains"][0]
            target_seq = chain["sequence"]

            chosen_chain_for_msa = None
            for cid in chain["chain_ids"]:
                if isinstance(cid, str) and len(cid) == 1 and cid.isalpha():
                    chosen_chain_for_msa = cid
                    break

            if chosen_chain_for_msa is None:
                dropped.append((qname, "no_valid_chain_id_in_json"))
                continue

            msa_dir_name = f"{qname.lower()}_{chosen_chain_for_msa}"
            msa_file = self.msa_root / msa_dir_name / self.use_a3m_name

            cif_candidates = list(self.cif_root.glob(f"{qname.lower()}-assembly1_*.cif"))
            if len(cif_candidates) == 0:
                dropped.append((qname, "no_cif"))
                continue
            cif_file = cif_candidates[0]

            if not msa_file.exists():
                dropped.append((qname, "no_msa"))
                continue

            try:
                chain_data = extract_chain_sequences_and_backbone(cif_file)
                match = match_target_to_chain(
                    target_seq=target_seq,
                    chain_data=chain_data,
                    min_identity=self.min_identity,
                )
            except Exception as e:
                dropped.append((qname, f"parse_error:{str(e)}"))
                continue

            if match is None:
                dropped.append((qname, "no_chain_match"))
                continue

            matched_chain_id, match_identity = match
            matched_seq = chain_data[matched_chain_id]["sequence"]

            rows.append({
                "query_name": qname,
                "target_sequence": target_seq,
                "msa_chain_id": chosen_chain_for_msa,
                "matched_chain_id": matched_chain_id,
                "match_identity": match_identity,
                "matched_chain_sequence": matched_seq,
                "msa_file": str(msa_file),
                "cif_file": str(cif_file),
            })

        if max_samples is not None:
            rows = rows[:max_samples]

        self.df = pd.DataFrame(rows).reset_index(drop=True)
        self.dropped = dropped

        if verbose:
            print(f"Dataset valid examples: {len(self.df)}")
            print(f"Dropped examples: {len(self.dropped)}")
            if len(self.df) > 0:
                print(self.df[["query_name", "msa_chain_id", "matched_chain_id", "match_identity"]].head())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        query_name = row["query_name"]
        target_sequence = row["target_sequence"]
        matched_chain_id = row["matched_chain_id"]
        msa_file = Path(row["msa_file"])
        cif_file = Path(row["cif_file"])

        seq_tokens = tokenize_sequence(target_sequence)

        msa_seqs = read_a3m(msa_file, max_msa_seqs=self.max_msa_seqs)
        msa_seqs = pad_or_crop_msa(
            msa_seqs,
            target_len=len(target_sequence),
            max_msa_seqs=self.max_msa_seqs,
        )
        msa_tokens = tokenize_msa(msa_seqs)
        msa_mask = (msa_tokens != AA_VOCAB["-"]).float()

        chain_data = extract_chain_sequences_and_backbone(cif_file)

        coords_n_np = chain_data[matched_chain_id]["coords_n"]
        coords_ca_np = chain_data[matched_chain_id]["coords_ca"]
        coords_c_np = chain_data[matched_chain_id]["coords_c"]

        valid_n = ~np.isnan(coords_n_np).any(axis=1)
        valid_ca = ~np.isnan(coords_ca_np).any(axis=1)
        valid_c = ~np.isnan(coords_c_np).any(axis=1)

        valid_res_mask_np = valid_ca.astype(np.float32)
        valid_backbone_mask_np = (valid_n & valid_ca & valid_c).astype(np.float32)

        coords_n_np = np.nan_to_num(coords_n_np, nan=0.0)
        coords_ca_np = np.nan_to_num(coords_ca_np, nan=0.0)
        coords_c_np = np.nan_to_num(coords_c_np, nan=0.0)

        L = min(
            len(seq_tokens),
            coords_ca_np.shape[0],
            coords_n_np.shape[0],
            coords_c_np.shape[0],
            msa_tokens.shape[1],
        )

        seq_tokens = seq_tokens[:L]
        msa_tokens = msa_tokens[:, :L]
        msa_mask = msa_mask[:, :L]

        coords_n_np = coords_n_np[:L]
        coords_ca_np = coords_ca_np[:L]
        coords_c_np = coords_c_np[:L]

        valid_res_mask_np = valid_res_mask_np[:L]
        valid_backbone_mask_np = valid_backbone_mask_np[:L]

        torsion_true, torsion_mask = backbone_torsions_from_coords(
            coords_n=coords_n_np,
            coords_ca=coords_ca_np,
            coords_c=coords_c_np,
            valid_backbone_mask=valid_backbone_mask_np.astype(bool),
        )

        coords_n = torch.tensor(coords_n_np, dtype=torch.float32)
        coords_ca = torch.tensor(coords_ca_np, dtype=torch.float32)
        coords_c = torch.tensor(coords_c_np, dtype=torch.float32)

        valid_res_mask = torch.tensor(valid_res_mask_np, dtype=torch.float32)
        valid_backbone_mask = torch.tensor(valid_backbone_mask_np, dtype=torch.float32)

        dist_map = pairwise_distances(coords_ca)

        return {
            "id": query_name,
            "msa_chain_id": row["msa_chain_id"],
            "matched_chain_id": matched_chain_id,
            "match_identity": torch.tensor(row["match_identity"], dtype=torch.float32),
            "sequence_str": target_sequence[:L],
            "seq_tokens": seq_tokens,
            "msa_tokens": msa_tokens,
            "msa_mask": msa_mask,
            "coords_n": coords_n,                       # [L, 3]
            "coords_ca": coords_ca,                     # [L, 3]
            "coords_c": coords_c,                       # [L, 3]
            "dist_map": dist_map,                       # [L, L]
            "valid_res_mask": valid_res_mask,           # [L]
            "valid_backbone_mask": valid_backbone_mask, # [L]
            "torsion_true": torsion_true,               # [L, 3, 2] -> phi, psi, omega as (sin, cos)
            "torsion_mask": torsion_mask,               # [L, 3]
        }
