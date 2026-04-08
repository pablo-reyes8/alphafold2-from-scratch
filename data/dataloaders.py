from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from difflib import SequenceMatcher
from torch.utils.data import Dataset

from data.foldbench import build_manifest_dataframe, load_manifest_dataframe


AA_VOCAB = {
    "-": 0,
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "X": 21,
    "B": 22,
    "Z": 23,
    "U": 24,
    "O": 25,
    ".": 26,
}

UNK_TOKEN = AA_VOCAB["X"]
GAP_TOKEN = AA_VOCAB["-"]

FEATURE_AA_STATES = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
    "-",
    "*",
)
FEATURE_AA_TO_INDEX = {token: index for index, token in enumerate(FEATURE_AA_STATES)}
COMMON_AA_STATES = FEATURE_AA_STATES[:20]
MASKED_MSA_MASK_TOKEN = "*"
MASKED_MSA_INPUT_TOKEN = "."
MASKED_MSA_NUM_CLASSES = len(FEATURE_AA_STATES)
EXTRA_MSA_FEATURE_DIM = 25
TEMPLATE_ANGLE_FEATURE_DIM = 51
TEMPLATE_PAIR_FEATURE_DIM = 88
TEMPLATE_PAIR_DIST_BINS = 64
TEMPLATE_REL_POS_CLIP = 10


def _build_token_to_masked_msa_class_lut() -> torch.Tensor:
    lut = torch.full((max(AA_VOCAB.values()) + 1,), FEATURE_AA_TO_INDEX["X"], dtype=torch.long)
    for token in COMMON_AA_STATES:
        lut[AA_VOCAB[token]] = FEATURE_AA_TO_INDEX[token]
    lut[AA_VOCAB["X"]] = FEATURE_AA_TO_INDEX["X"]
    lut[AA_VOCAB["-"]] = FEATURE_AA_TO_INDEX["-"]
    lut[AA_VOCAB[MASKED_MSA_INPUT_TOKEN]] = FEATURE_AA_TO_INDEX[MASKED_MSA_MASK_TOKEN]
    return lut


def _build_masked_msa_class_to_token_lut() -> torch.Tensor:
    lut = torch.full((MASKED_MSA_NUM_CLASSES,), AA_VOCAB["X"], dtype=torch.long)
    for token in COMMON_AA_STATES:
        lut[FEATURE_AA_TO_INDEX[token]] = AA_VOCAB[token]
    lut[FEATURE_AA_TO_INDEX["X"]] = AA_VOCAB["X"]
    lut[FEATURE_AA_TO_INDEX["-"]] = AA_VOCAB["-"]
    lut[FEATURE_AA_TO_INDEX[MASKED_MSA_MASK_TOKEN]] = AA_VOCAB[MASKED_MSA_INPUT_TOKEN]
    return lut


TOKEN_TO_MASKED_MSA_CLASS = _build_token_to_masked_msa_class_lut()
MASKED_MSA_CLASS_TO_TOKEN = _build_masked_msa_class_to_token_lut()
COMMON_AA_CLASS_INDICES = torch.tensor(
    [FEATURE_AA_TO_INDEX[token] for token in COMMON_AA_STATES],
    dtype=torch.long,
)


def build_masked_msa_inputs(
    msa_tokens: torch.Tensor,
    msa_mask: torch.Tensor | None,
    *,
    replace_fraction: float = 0.15,
    profile_prob: float = 0.10,
    same_prob: float = 0.10,
    uniform_prob: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt MSA inputs and emit AF2-style masked-MSA supervision targets."""

    if msa_tokens.ndim != 2:
        raise ValueError(f"msa_tokens must have shape [N, L], got {tuple(msa_tokens.shape)}")

    replace_fraction = float(replace_fraction)
    profile_prob = float(profile_prob)
    same_prob = float(same_prob)
    uniform_prob = float(uniform_prob)
    total_replace_prob = profile_prob + same_prob + uniform_prob

    if not 0.0 <= replace_fraction <= 1.0:
        raise ValueError(f"replace_fraction must be in [0, 1], got {replace_fraction}")
    if min(profile_prob, same_prob, uniform_prob) < 0.0:
        raise ValueError("masked MSA corruption probabilities must be non-negative")
    if total_replace_prob > 1.0 + 1e-8:
        raise ValueError(
            "masked MSA corruption probabilities must sum to <= 1.0: "
            f"profile={profile_prob}, same={same_prob}, uniform={uniform_prob}"
        )

    device = msa_tokens.device
    token_to_class = TOKEN_TO_MASKED_MSA_CLASS.to(device)
    class_to_token = MASKED_MSA_CLASS_TO_TOKEN.to(device)
    true_classes = token_to_class[msa_tokens.long()]

    if msa_mask is None:
        valid_positions = true_classes != FEATURE_AA_TO_INDEX["-"]
    else:
        # Restrict supervision to originally valid residues while keeping the
        # existing trunk mask semantics unchanged for the corrupted input MSA.
        valid_positions = msa_mask > 0

    selected_positions = (
        torch.rand(msa_tokens.shape, device=device, dtype=torch.float32) < replace_fraction
    ) & valid_positions

    corrupted_classes = true_classes.clone()
    if selected_positions.any():
        profile = F.one_hot(true_classes, num_classes=MASKED_MSA_NUM_CLASSES).to(torch.float32).mean(dim=0)
        selection_indices = selected_positions.nonzero(as_tuple=False)
        selection_columns = selection_indices[:, 1]
        draws = torch.rand(selection_indices.shape[0], device=device, dtype=torch.float32)

        same_cutoff = same_prob
        uniform_cutoff = same_cutoff + uniform_prob
        profile_cutoff = uniform_cutoff + profile_prob

        uniform_pick = (draws >= same_cutoff) & (draws < uniform_cutoff)
        profile_pick = (draws >= uniform_cutoff) & (draws < profile_cutoff)
        mask_pick = draws >= profile_cutoff

        common_classes = COMMON_AA_CLASS_INDICES.to(device)

        if uniform_pick.any():
            sampled_uniform = common_classes[
                torch.randint(
                    low=0,
                    high=common_classes.shape[0],
                    size=(int(uniform_pick.sum().item()),),
                    device=device,
                )
            ]
            corrupted_classes[
                selection_indices[uniform_pick, 0],
                selection_indices[uniform_pick, 1],
            ] = sampled_uniform

        if profile_pick.any():
            sampled_profile = torch.multinomial(
                profile[selection_columns[profile_pick]],
                num_samples=1,
            ).squeeze(-1)
            corrupted_classes[
                selection_indices[profile_pick, 0],
                selection_indices[profile_pick, 1],
            ] = sampled_profile

        if mask_pick.any():
            corrupted_classes[
                selection_indices[mask_pick, 0],
                selection_indices[mask_pick, 1],
            ] = FEATURE_AA_TO_INDEX[MASKED_MSA_MASK_TOKEN]

    return (
        class_to_token[corrupted_classes].to(dtype=msa_tokens.dtype),
        true_classes.to(dtype=torch.long),
        selected_positions.to(dtype=torch.float32),
    )


def tokenize_sequence(seq: str) -> torch.Tensor:
    return torch.tensor(
        [AA_VOCAB.get(character.upper(), UNK_TOKEN) for character in seq],
        dtype=torch.long,
    )


def read_a3m(a3m_path: str | Path, max_msa_seqs: int | None = None) -> list[str]:
    sequences: list[str] = []
    current_name: str | None = None
    current_sequence: list[str] = []

    with Path(a3m_path).expanduser().open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_name is not None:
                    sequence = "".join(current_sequence)
                    sequence = "".join(character for character in sequence if not character.islower())
                    sequences.append(sequence)
                    if max_msa_seqs is not None and len(sequences) >= max_msa_seqs:
                        break

                current_name = line[1:]
                current_sequence = []
                continue

            current_sequence.append(line)

    if (max_msa_seqs is None or len(sequences) < max_msa_seqs) and current_name is not None:
        sequence = "".join(current_sequence)
        sequence = "".join(character for character in sequence if not character.islower())
        sequences.append(sequence)

    return sequences


def pad_or_crop_msa(msa_seqs: list[str], target_len: int, max_msa_seqs: int) -> list[str]:
    fixed: list[str] = []

    for sequence in msa_seqs[:max_msa_seqs]:
        if len(sequence) < target_len:
            sequence = sequence + "-" * (target_len - len(sequence))
        elif len(sequence) > target_len:
            sequence = sequence[:target_len]
        fixed.append(sequence)

    if not fixed:
        fixed = ["-" * target_len]

    return fixed


def select_msa_sequences(
    msa_seqs: list[str],
    *,
    target_sequence: str,
    target_len: int,
    max_msa_seqs: int,
    single_sequence_mode: bool = False,
) -> list[str]:
    if single_sequence_mode:
        return pad_or_crop_msa([target_sequence], target_len=target_len, max_msa_seqs=1)

    return pad_or_crop_msa(msa_seqs, target_len=target_len, max_msa_seqs=max_msa_seqs)


def tokenize_msa(msa_seqs: list[str]) -> torch.Tensor:
    return torch.stack([tokenize_sequence(sequence) for sequence in msa_seqs], dim=0)


def canonical_feature_token(character: str) -> str:
    token = character.upper()
    if token in FEATURE_AA_TO_INDEX:
        return token
    if token in {".", "-"}:
        return "-"
    return "X"


def _sequence_to_feature_one_hot(sequence: str) -> np.ndarray:
    one_hot = np.zeros((len(sequence), len(FEATURE_AA_STATES)), dtype=np.float32)
    for index, character in enumerate(sequence):
        one_hot[index, FEATURE_AA_TO_INDEX[canonical_feature_token(character)]] = 1.0
    return one_hot


def _finalize_a3m_sequence(raw_sequence: str) -> tuple[str, np.ndarray]:
    aligned: list[str] = []
    deletion_counts: list[float] = []
    pending_deletions = 0

    for character in raw_sequence:
        if character.islower():
            pending_deletions += 1
            continue

        aligned.append(character.upper())
        deletion_counts.append(float(pending_deletions))
        pending_deletions = 0

    return "".join(aligned), np.asarray(deletion_counts, dtype=np.float32)


def read_a3m_records(
    a3m_path: str | Path,
    max_msa_seqs: int | None = None,
) -> list[tuple[str, np.ndarray]]:
    records: list[tuple[str, np.ndarray]] = []
    current_name: str | None = None
    current_sequence: list[str] = []

    with Path(a3m_path).expanduser().open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_name is not None:
                    records.append(_finalize_a3m_sequence("".join(current_sequence)))
                    if max_msa_seqs is not None and len(records) >= max_msa_seqs:
                        break

                current_name = line[1:]
                current_sequence = []
                continue

            current_sequence.append(line)

    if (max_msa_seqs is None or len(records) < max_msa_seqs) and current_name is not None:
        records.append(_finalize_a3m_sequence("".join(current_sequence)))

    return records


def read_stockholm_records(
    stockholm_path: str | Path,
    max_msa_seqs: int | None = None,
) -> list[tuple[str, np.ndarray]]:
    chunks: dict[str, list[str]] = {}
    order: list[str] = []

    with Path(stockholm_path).expanduser().open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line == "//":
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            name, sequence_chunk = parts[0], parts[1]
            if name not in chunks:
                chunks[name] = []
                order.append(name)
            chunks[name].append(sequence_chunk)

    records: list[tuple[str, np.ndarray]] = []
    for name in order:
        sequence = "".join(chunks[name]).replace(".", "-").upper()
        deletion_counts = np.zeros(len(sequence), dtype=np.float32)
        records.append((sequence, deletion_counts))
        if max_msa_seqs is not None and len(records) >= max_msa_seqs:
            break

    return records


def _normalize_alignment_record(
    sequence: str,
    deletion_counts: np.ndarray,
    target_len: int,
) -> tuple[str, np.ndarray]:
    normalized = [canonical_feature_token(character) for character in sequence[:target_len]]
    deletion = np.asarray(deletion_counts[:target_len], dtype=np.float32)

    if len(normalized) < target_len:
        normalized.extend("-" for _ in range(target_len - len(normalized)))

    if deletion.shape[0] < target_len:
        deletion = np.pad(deletion, (0, target_len - deletion.shape[0]))

    return "".join(normalized), deletion


def _deletion_value_transform(deletion_counts: np.ndarray) -> np.ndarray:
    return np.arctan(deletion_counts / 3.0) * (2.0 / np.pi)


def build_extra_msa_records(
    *,
    msa_dir: str | Path,
    target_sequence: str,
    main_msa_seqs: list[str],
    max_extra_msa_seqs: int,
) -> list[tuple[str, np.ndarray]]:
    if max_extra_msa_seqs <= 0:
        return []

    msa_dir = Path(msa_dir).expanduser()
    target_len = len(target_sequence)
    seen_sequences = {
        _normalize_alignment_record(sequence, np.zeros(len(sequence), dtype=np.float32), target_len)[0]
        for sequence in main_msa_seqs
    }
    seen_sequences.add(
        _normalize_alignment_record(
            target_sequence,
            np.zeros(len(target_sequence), dtype=np.float32),
            target_len,
        )[0]
    )

    records: list[tuple[str, np.ndarray]] = []

    def append_records(source_records: list[tuple[str, np.ndarray]]) -> None:
        for sequence, deletion_counts in source_records:
            normalized_sequence, normalized_deletions = _normalize_alignment_record(
                sequence,
                deletion_counts,
                target_len,
            )
            if not normalized_sequence or set(normalized_sequence) == {"-"}:
                continue
            if normalized_sequence in seen_sequences:
                continue
            seen_sequences.add(normalized_sequence)
            records.append((normalized_sequence, normalized_deletions))
            if len(records) >= max_extra_msa_seqs:
                return

    cfdb_path = msa_dir / "cfdb_hits.a3m"
    if cfdb_path.exists():
        cfdb_records = read_a3m_records(cfdb_path)
        append_records(cfdb_records[len(main_msa_seqs) :])

    for stockholm_name in ("mgnify_hits.sto", "uniprot_hits.sto", "uniref90_hits.sto"):
        if len(records) >= max_extra_msa_seqs:
            break
        stockholm_path = msa_dir / stockholm_name
        if not stockholm_path.exists():
            continue
        append_records(read_stockholm_records(stockholm_path))

    return records[:max_extra_msa_seqs]


def build_extra_msa_features(
    records: list[tuple[str, np.ndarray]],
    *,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_records = len(records)
    features = np.zeros((num_records, target_len, EXTRA_MSA_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((num_records, target_len), dtype=np.float32)

    for record_index, (sequence, deletion_counts) in enumerate(records):
        features[record_index, :, : len(FEATURE_AA_STATES)] = _sequence_to_feature_one_hot(sequence)
        features[record_index, :, 23] = (deletion_counts > 0).astype(np.float32)
        features[record_index, :, 24] = _deletion_value_transform(deletion_counts)
        mask[record_index] = np.asarray(
            [0.0 if canonical_feature_token(character) == "-" else 1.0 for character in sequence],
            dtype=np.float32,
        )

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
    )


def parse_same_structure_template_chain_ids(
    hmm_output_path: str | Path,
    *,
    query_name: str,
) -> list[str]:
    path = Path(hmm_output_path).expanduser()
    if not path.exists():
        return []

    prefix = f"{query_name.lower()}_"
    chain_ids: list[str] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.startswith("#=GS "):
                continue

            token = raw_line.split()[1].split("/")[0]
            token_lower = token.lower()
            if not token_lower.startswith(prefix):
                continue

            parts = token.split("_", maxsplit=1)
            if len(parts) != 2:
                continue

            chain_id = parts[1]
            if chain_id in seen:
                continue

            seen.add(chain_id)
            chain_ids.append(chain_id)

    return chain_ids


def build_alignment_mapping(target_sequence: str, template_sequence: str) -> np.ndarray:
    align_module, _, _ = _require_biopython()
    aligner = align_module.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.1

    alignment = aligner.align(target_sequence, template_sequence)[0]
    mapping = np.full(len(target_sequence), -1, dtype=np.int64)

    for target_span, template_span in zip(alignment.aligned[0], alignment.aligned[1]):
        target_start, target_end = target_span
        template_start, template_end = template_span
        span_length = min(target_end - target_start, template_end - template_start)
        mapping[target_start : target_start + span_length] = np.arange(
            template_start,
            template_start + span_length,
            dtype=np.int64,
        )

    return mapping


def _safe_normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def _compute_backbone_local_geometry(
    n_coord: np.ndarray,
    ca_coord: np.ndarray,
    c_coord: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ex = _safe_normalize(c_coord - ca_coord, eps=eps)
    n_vec = n_coord - ca_coord
    n_proj = n_vec - np.dot(n_vec, ex) * ex
    ey = _safe_normalize(n_proj, eps=eps)
    ez = _safe_normalize(np.cross(ex, ey), eps=eps)
    rotation = np.stack([ex, ey, ez], axis=-1)

    n_local = (n_coord - ca_coord) @ rotation
    c_local = (c_coord - ca_coord) @ rotation
    return n_local, c_local, _safe_normalize(n_local, eps=eps), _safe_normalize(c_local, eps=eps)


def build_template_pair_features(
    coords_ca: np.ndarray,
    residue_mask: np.ndarray,
) -> np.ndarray:
    length = coords_ca.shape[0]
    pair_mask = residue_mask[:, None] * residue_mask[None, :]
    diffs = coords_ca[:, None, :] - coords_ca[None, :, :]
    distances = np.sqrt(np.sum(diffs**2, axis=-1) + 1e-8)

    dist_edges = np.linspace(2.0, 22.0, TEMPLATE_PAIR_DIST_BINS - 1, dtype=np.float32)
    dist_bins = np.digitize(distances, dist_edges, right=False)
    dist_one_hot = np.eye(TEMPLATE_PAIR_DIST_BINS, dtype=np.float32)[dist_bins]
    dist_one_hot *= pair_mask[..., None]

    residue_indices = np.arange(length, dtype=np.int64)
    relpos = residue_indices[:, None] - residue_indices[None, :]
    relpos = np.clip(relpos, -TEMPLATE_REL_POS_CLIP, TEMPLATE_REL_POS_CLIP) + TEMPLATE_REL_POS_CLIP
    relpos_one_hot = np.eye(2 * TEMPLATE_REL_POS_CLIP + 1, dtype=np.float32)[relpos]
    relpos_one_hot *= pair_mask[..., None]

    pair_mask_feat = pair_mask[..., None].astype(np.float32)
    inverse_distance = np.where(pair_mask > 0, 1.0 / (1.0 + distances), 0.0)[..., None].astype(np.float32)
    same_chain_feat = pair_mask[..., None].astype(np.float32)

    return np.concatenate(
        [dist_one_hot, relpos_one_hot, pair_mask_feat, inverse_distance, same_chain_feat],
        axis=-1,
    )


def build_template_features_from_chain(
    *,
    target_sequence: str,
    template_sequence: str,
    coords_n: np.ndarray,
    coords_ca: np.ndarray,
    coords_c: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_len = len(target_sequence)
    mapping = build_alignment_mapping(target_sequence, template_sequence)

    aligned_sequence = ["-"] * target_len
    aligned_coords_n = np.zeros((target_len, 3), dtype=np.float32)
    aligned_coords_ca = np.zeros((target_len, 3), dtype=np.float32)
    aligned_coords_c = np.zeros((target_len, 3), dtype=np.float32)

    source_valid_n = ~np.isnan(coords_n).any(axis=1)
    source_valid_ca = ~np.isnan(coords_ca).any(axis=1)
    source_valid_c = ~np.isnan(coords_c).any(axis=1)

    aligned_valid_n = np.zeros(target_len, dtype=np.float32)
    aligned_valid_ca = np.zeros(target_len, dtype=np.float32)
    aligned_valid_c = np.zeros(target_len, dtype=np.float32)

    for target_index, template_index in enumerate(mapping):
        if template_index < 0 or template_index >= len(template_sequence):
            continue

        aligned_sequence[target_index] = template_sequence[template_index]
        aligned_coords_n[target_index] = np.nan_to_num(coords_n[template_index], nan=0.0)
        aligned_coords_ca[target_index] = np.nan_to_num(coords_ca[template_index], nan=0.0)
        aligned_coords_c[target_index] = np.nan_to_num(coords_c[template_index], nan=0.0)
        aligned_valid_n[target_index] = float(source_valid_n[template_index])
        aligned_valid_ca[target_index] = float(source_valid_ca[template_index])
        aligned_valid_c[target_index] = float(source_valid_c[template_index])

    aligned_valid_backbone = (aligned_valid_n * aligned_valid_ca * aligned_valid_c).astype(np.float32)
    torsion_true, torsion_mask = backbone_torsions_from_coords(
        coords_n=aligned_coords_n,
        coords_ca=aligned_coords_ca,
        coords_c=aligned_coords_c,
        valid_backbone_mask=aligned_valid_backbone.astype(bool),
    )

    angle_feat = np.zeros((target_len, TEMPLATE_ANGLE_FEATURE_DIM), dtype=np.float32)
    angle_feat[:, : len(FEATURE_AA_STATES)] = _sequence_to_feature_one_hot("".join(aligned_sequence))
    angle_feat[:, 23:29] = torsion_true.reshape(target_len, 6).numpy()
    angle_feat[:, 29:32] = torsion_mask.numpy()
    angle_feat[:, 41] = aligned_valid_n
    angle_feat[:, 42] = aligned_valid_ca
    angle_feat[:, 43] = aligned_valid_c
    angle_feat[:, 44] = aligned_valid_backbone

    for residue_index in range(target_len):
        if aligned_valid_backbone[residue_index] == 0:
            continue

        n_local, c_local, n_unit, c_unit = _compute_backbone_local_geometry(
            aligned_coords_n[residue_index],
            aligned_coords_ca[residue_index],
            aligned_coords_c[residue_index],
        )
        angle_feat[residue_index, 32:41] = np.concatenate(
            [n_local, np.zeros(3, dtype=np.float32), c_local]
        )
        angle_feat[residue_index, 45:51] = np.concatenate([n_unit, c_unit])

    pair_feat = build_template_pair_features(aligned_coords_ca, aligned_valid_backbone)
    template_mask = aligned_valid_backbone

    return (
        torch.tensor(angle_feat, dtype=torch.float32),
        torch.tensor(pair_feat, dtype=torch.float32),
        torch.tensor(template_mask, dtype=torch.float32),
    )


def build_template_feature_tensors(
    *,
    query_name: str,
    msa_dir: str | Path,
    chain_data: dict[str, dict[str, Any]],
    matched_chain_id: str,
    target_sequence: str,
    identity_target_sequence: str | None = None,
    max_templates: int,
    min_template_identity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    target_len = len(target_sequence)
    scoring_target_sequence = identity_target_sequence or target_sequence
    if max_templates <= 0:
        empty_angle = torch.zeros((0, target_len, TEMPLATE_ANGLE_FEATURE_DIM), dtype=torch.float32)
        empty_pair = torch.zeros((0, target_len, target_len, TEMPLATE_PAIR_FEATURE_DIM), dtype=torch.float32)
        empty_mask = torch.zeros((0, target_len), dtype=torch.float32)
        return empty_angle, empty_pair, empty_mask, []

    hmm_output_path = Path(msa_dir).expanduser() / "hmm_output.sto"
    preferred_chain_ids = parse_same_structure_template_chain_ids(
        hmm_output_path,
        query_name=query_name,
    )
    if not preferred_chain_ids:
        preferred_chain_ids = list(chain_data.keys())

    scored_candidates: list[tuple[float, str]] = []
    seen: set[str] = set()
    for chain_id in preferred_chain_ids:
        if chain_id in seen or chain_id == matched_chain_id or chain_id not in chain_data:
            continue
        seen.add(chain_id)

        candidate_sequence = chain_data[chain_id]["sequence"]
        if not candidate_sequence:
            continue

        identity = sequence_identity(scoring_target_sequence, candidate_sequence)
        if identity < min_template_identity:
            continue

        scored_candidates.append((identity, chain_id))

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))

    template_angle_feats: list[torch.Tensor] = []
    template_pair_feats: list[torch.Tensor] = []
    template_masks: list[torch.Tensor] = []
    template_chain_ids: list[str] = []

    for _, chain_id in scored_candidates[:max_templates]:
        chain_info = chain_data[chain_id]
        angle_feat, pair_feat, template_mask = build_template_features_from_chain(
            target_sequence=target_sequence,
            template_sequence=chain_info["sequence"],
            coords_n=chain_info["coords_n"],
            coords_ca=chain_info["coords_ca"],
            coords_c=chain_info["coords_c"],
        )

        if float(template_mask.sum().item()) <= 0.0:
            continue

        template_angle_feats.append(angle_feat)
        template_pair_feats.append(pair_feat)
        template_masks.append(template_mask)
        template_chain_ids.append(chain_id)

    if not template_angle_feats:
        empty_angle = torch.zeros((0, target_len, TEMPLATE_ANGLE_FEATURE_DIM), dtype=torch.float32)
        empty_pair = torch.zeros((0, target_len, target_len, TEMPLATE_PAIR_FEATURE_DIM), dtype=torch.float32)
        empty_mask = torch.zeros((0, target_len), dtype=torch.float32)
        return empty_angle, empty_pair, empty_mask, []

    return (
        torch.stack(template_angle_feats, dim=0),
        torch.stack(template_pair_feats, dim=0),
        torch.stack(template_masks, dim=0),
        template_chain_ids,
    )


def _require_biopython():
    try:
        from Bio import Align
        from Bio.PDB.MMCIFParser import MMCIFParser
        from Bio.SeqUtils import seq1
    except ImportError as exc:
        raise ImportError(
            "Biopython is required for mmCIF parsing and sequence alignment. "
            "Install it with `pip install biopython`."
        ) from exc

    return Align, MMCIFParser, seq1


def safe_residue_to_aa(residue: Any, seq1_fn) -> str:
    resname = residue.get_resname().strip()
    try:
        aa = seq1_fn(resname)
        if len(aa) == 1:
            return aa
    except Exception:
        pass
    return "X"


@lru_cache(maxsize=128)
def _extract_chain_sequences_and_backbone_cached(cif_path: str) -> dict[str, dict[str, Any]]:
    _, mmcif_parser_cls, seq1_fn = _require_biopython()

    parser = mmcif_parser_cls(QUIET=True)
    structure = parser.get_structure(Path(cif_path).stem, cif_path)
    first_model = next(structure.get_models())

    out: dict[str, dict[str, Any]] = {}

    for chain in first_model:
        sequence_chars: list[str] = []
        coords_n: list[Any] = []
        coords_ca: list[Any] = []
        coords_c: list[Any] = []

        for residue in chain:
            if residue.id[0].strip() != "":
                continue

            sequence_chars.append(safe_residue_to_aa(residue, seq1_fn=seq1_fn))
            coords_n.append(residue["N"].coord if "N" in residue else [np.nan, np.nan, np.nan])
            coords_ca.append(residue["CA"].coord if "CA" in residue else [np.nan, np.nan, np.nan])
            coords_c.append(residue["C"].coord if "C" in residue else [np.nan, np.nan, np.nan])

        if not sequence_chars:
            continue

        out[chain.id] = {
            "sequence": "".join(sequence_chars),
            "coords_n": np.array(coords_n, dtype=np.float32),
            "coords_ca": np.array(coords_ca, dtype=np.float32),
            "coords_c": np.array(coords_c, dtype=np.float32),
        }

    return out


def extract_chain_sequences_and_backbone(cif_path: str | Path) -> dict[str, dict[str, Any]]:
    cached = _extract_chain_sequences_and_backbone_cached(str(Path(cif_path).expanduser()))
    copied: dict[str, dict[str, Any]] = {}

    for chain_id, info in cached.items():
        copied[chain_id] = {
            "sequence": info["sequence"],
            "coords_n": info["coords_n"].copy(),
            "coords_ca": info["coords_ca"].copy(),
            "coords_c": info["coords_c"].copy(),
        }

    return copied


def sequence_identity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_target_to_chain(
    target_seq: str,
    chain_data: dict[str, dict[str, Any]],
    min_identity: float = 0.85,
) -> tuple[str, float] | None:
    try:
        align_module, _, _ = _require_biopython()
        aligner = align_module.PairwiseAligner()
        aligner.mode = "local"
    except ImportError:
        aligner = None

    best_chain: str | None = None
    best_score = -1.0

    for chain_id, info in chain_data.items():
        chain_seq = info["sequence"]
        if not chain_seq:
            continue

        if aligner is not None:
            score = aligner.score(target_seq, chain_seq)
            normalized_score = score / max(1, len(chain_seq))
        else:
            normalized_score = sequence_identity(target_seq, chain_seq)

        if normalized_score > best_score:
            best_score = normalized_score
            best_chain = chain_id

    if best_chain is None or best_score < min_identity:
        return None

    return best_chain, float(best_score)


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    diffs = coords[:, None, :] - coords[None, :, :]
    return torch.sqrt(torch.sum(diffs**2, dim=-1) + 1e-8)


def dihedral_angle(p0, p1, p2, p3, eps: float = 1e-8):
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


def backbone_torsions_from_coords(
    coords_n,
    coords_ca,
    coords_c,
    valid_backbone_mask,
    eps: float = 1e-8,
):
    length = coords_n.shape[0]
    torsion_true = np.zeros((length, 3, 2), dtype=np.float32)
    torsion_mask = np.zeros((length, 3), dtype=np.float32)

    for index in range(length):
        if index > 0 and valid_backbone_mask[index - 1] and valid_backbone_mask[index]:
            angle = dihedral_angle(
                coords_c[index - 1],
                coords_n[index],
                coords_ca[index],
                coords_c[index],
                eps=eps,
            )
            if np.isfinite(angle):
                torsion_true[index, 0, 0] = np.sin(angle)
                torsion_true[index, 0, 1] = np.cos(angle)
                torsion_mask[index, 0] = 1.0

        if index < length - 1 and valid_backbone_mask[index] and valid_backbone_mask[index + 1]:
            angle = dihedral_angle(
                coords_n[index],
                coords_ca[index],
                coords_c[index],
                coords_n[index + 1],
                eps=eps,
            )
            if np.isfinite(angle):
                torsion_true[index, 1, 0] = np.sin(angle)
                torsion_true[index, 1, 1] = np.cos(angle)
                torsion_mask[index, 1] = 1.0

            angle = dihedral_angle(
                coords_ca[index],
                coords_c[index],
                coords_n[index + 1],
                coords_ca[index + 1],
                eps=eps,
            )
            if np.isfinite(angle):
                torsion_true[index, 2, 0] = np.sin(angle)
                torsion_true[index, 2, 1] = np.cos(angle)
                torsion_mask[index, 2] = 1.0

    return (
        torch.tensor(torsion_true, dtype=torch.float32),
        torch.tensor(torsion_mask, dtype=torch.float32),
    )


class FoldbenchProteinDataset(Dataset):
    def __init__(
        self,
        json_path: str | None = None,
        msa_root: str | None = None,
        cif_root: str | None = None,
        manifest_csv: str | None = None,
        max_msa_seqs: int = 128,
        use_a3m_name: str = "cfdb_hits.a3m",
        max_samples: int | None = None,
        min_identity: float = 0.90,
        single_sequence_mode: bool = False,
        max_extra_msa_seqs: int = 256,
        max_templates: int = 4,
        min_template_identity: float = 0.80,
        crop_size: int | None = None,
        random_crop: bool = True,
        masked_msa_replace_fraction: float = 0.15,
        masked_msa_profile_prob: float = 0.10,
        masked_msa_same_prob: float = 0.10,
        masked_msa_uniform_prob: float = 0.10,
        verbose: bool = True,
    ):
        self.json_path = Path(json_path).expanduser() if json_path is not None else None
        self.msa_root = Path(msa_root).expanduser() if msa_root is not None else None
        self.cif_root = Path(cif_root).expanduser() if cif_root is not None else None
        self.manifest_csv = Path(manifest_csv).expanduser() if manifest_csv is not None else None
        self.max_msa_seqs = max_msa_seqs
        self.max_extra_msa_seqs = max(0, int(max_extra_msa_seqs))
        self.max_templates = max(0, int(max_templates))
        self.use_a3m_name = use_a3m_name
        self.min_identity = min_identity
        self.min_template_identity = float(min_template_identity)
        self.single_sequence_mode = bool(single_sequence_mode)
        self.crop_size = None if crop_size is None else max(1, int(crop_size))
        self.random_crop = bool(random_crop)
        self.masked_msa_replace_fraction = float(masked_msa_replace_fraction)
        self.masked_msa_profile_prob = float(masked_msa_profile_prob)
        self.masked_msa_same_prob = float(masked_msa_same_prob)
        self.masked_msa_uniform_prob = float(masked_msa_uniform_prob)

        self.manifest_df = self._load_manifest()
        rows, dropped = self._build_index(self.manifest_df)

        if max_samples is not None:
            rows = rows[:max_samples]

        self.df = pd.DataFrame(rows).reset_index(drop=True)
        self.dropped = dropped

        if verbose:
            print(f"Dataset valid examples: {len(self.df)}")
            print(f"Dropped examples: {len(self.dropped)}")
            if not self.df.empty:
                print(
                    self.df[
                        ["query_name", "msa_chain_id", "matched_chain_id", "match_identity"]
                    ].head()
                )

    def _load_manifest(self) -> pd.DataFrame:
        if self.manifest_csv is not None:
            return load_manifest_dataframe(
                manifest_csv=self.manifest_csv,
                msa_root=self.msa_root,
                cif_root=self.cif_root,
            )

        if self.json_path is None or self.msa_root is None or self.cif_root is None:
            raise ValueError(
                "FoldbenchProteinDataset requires either manifest_csv or json_path + msa_root + cif_root."
            )

        return build_manifest_dataframe(
            json_path=self.json_path,
            msa_root=self.msa_root,
            cif_root=self.cif_root,
        )

    def _build_index(self, manifest_df: pd.DataFrame):
        rows: list[dict[str, Any]] = []
        dropped: list[tuple[str, str]] = []

        for row in manifest_df.to_dict(orient="records"):
            query_name = str(row.get("query_name", ""))
            target_sequence = str(row.get("sequence", "") or "")
            msa_chain_id = str(row.get("chain_id", "") or "")
            msa_dir = Path(str(row.get("msa_dir", ""))).expanduser()
            msa_file = msa_dir / self.use_a3m_name
            cif_value = row.get("cif_file")
            cif_file = Path(str(cif_value)).expanduser() if pd.notna(cif_value) and cif_value else None

            if not query_name:
                dropped.append((query_name, "missing_query_name"))
                continue
            if not target_sequence:
                dropped.append((query_name, "missing_target_sequence"))
                continue
            if cif_file is None or not cif_file.exists():
                dropped.append((query_name, "no_cif"))
                continue
            if not msa_file.exists():
                dropped.append((query_name, "no_msa"))
                continue

            try:
                chain_data = extract_chain_sequences_and_backbone(cif_file)
                match = match_target_to_chain(
                    target_seq=target_sequence,
                    chain_data=chain_data,
                    min_identity=self.min_identity,
                )
            except Exception as exc:
                dropped.append((query_name, f"parse_error:{exc}"))
                continue

            if match is None:
                dropped.append((query_name, "no_chain_match"))
                continue

            matched_chain_id, match_identity = match
            rows.append(
                {
                    "query_name": query_name,
                    "target_sequence": target_sequence,
                    "msa_chain_id": msa_chain_id,
                    "matched_chain_id": matched_chain_id,
                    "match_identity": match_identity,
                    "matched_chain_sequence": chain_data[matched_chain_id]["sequence"],
                    "msa_file": str(msa_file),
                    "cif_file": str(cif_file),
                }
            )

        return rows, dropped

    def __len__(self):
        return len(self.df)

    def _resolve_crop_bounds(self, length: int) -> tuple[int, int]:
        if self.crop_size is None or length <= self.crop_size:
            return 0, length

        crop_size = min(length, self.crop_size)
        max_start = length - crop_size
        if max_start <= 0 or not self.random_crop:
            return 0, crop_size

        start = int(np.random.randint(0, max_start + 1))
        return start, start + crop_size

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        query_name = row["query_name"]
        target_sequence = row["target_sequence"]
        matched_chain_id = row["matched_chain_id"]
        msa_file = Path(row["msa_file"])
        cif_file = Path(row["cif_file"])

        seq_tokens = tokenize_sequence(target_sequence)

        msa_seqs = read_a3m(msa_file, max_msa_seqs=self.max_msa_seqs)
        msa_seqs = select_msa_sequences(
            msa_seqs,
            target_sequence=target_sequence,
            target_len=len(target_sequence),
            max_msa_seqs=self.max_msa_seqs,
            single_sequence_mode=self.single_sequence_mode,
        )
        msa_tokens = tokenize_msa(msa_seqs)
        msa_mask = (msa_tokens != AA_VOCAB["-"]).float()

        chain_data = extract_chain_sequences_and_backbone(cif_file)
        chain_entry = chain_data[matched_chain_id]

        coords_n_np = chain_entry["coords_n"].copy()
        coords_ca_np = chain_entry["coords_ca"].copy()
        coords_c_np = chain_entry["coords_c"].copy()

        valid_n = ~np.isnan(coords_n_np).any(axis=1)
        valid_ca = ~np.isnan(coords_ca_np).any(axis=1)
        valid_c = ~np.isnan(coords_c_np).any(axis=1)

        valid_res_mask_np = valid_ca.astype(np.float32)
        valid_backbone_mask_np = (valid_n & valid_ca & valid_c).astype(np.float32)

        coords_n_np = np.nan_to_num(coords_n_np, nan=0.0)
        coords_ca_np = np.nan_to_num(coords_ca_np, nan=0.0)
        coords_c_np = np.nan_to_num(coords_c_np, nan=0.0)

        length = min(
            len(seq_tokens),
            coords_ca_np.shape[0],
            coords_n_np.shape[0],
            coords_c_np.shape[0],
            msa_tokens.shape[1],
        )

        seq_tokens = seq_tokens[:length]
        msa_tokens = msa_tokens[:, :length]
        msa_mask = msa_mask[:, :length]

        coords_n_np = coords_n_np[:length]
        coords_ca_np = coords_ca_np[:length]
        coords_c_np = coords_c_np[:length]

        valid_res_mask_np = valid_res_mask_np[:length]
        valid_backbone_mask_np = valid_backbone_mask_np[:length]
        full_target_sequence = target_sequence[:length]

        crop_start, crop_end = self._resolve_crop_bounds(length)
        if crop_start != 0 or crop_end != length:
            seq_tokens = seq_tokens[crop_start:crop_end]
            msa_tokens = msa_tokens[:, crop_start:crop_end]
            msa_mask = msa_mask[:, crop_start:crop_end]

            coords_n_np = coords_n_np[crop_start:crop_end]
            coords_ca_np = coords_ca_np[crop_start:crop_end]
            coords_c_np = coords_c_np[crop_start:crop_end]

            valid_res_mask_np = valid_res_mask_np[crop_start:crop_end]
            valid_backbone_mask_np = valid_backbone_mask_np[crop_start:crop_end]

        cropped_target_sequence = full_target_sequence[crop_start:crop_end]
        cropped_msa_seqs = [sequence[crop_start:crop_end] for sequence in msa_seqs]
        msa_tokens, masked_msa_true, masked_msa_mask = build_masked_msa_inputs(
            msa_tokens,
            msa_mask,
            replace_fraction=self.masked_msa_replace_fraction,
            profile_prob=self.masked_msa_profile_prob,
            same_prob=self.masked_msa_same_prob,
            uniform_prob=self.masked_msa_uniform_prob,
        )

        torsion_true, torsion_mask = backbone_torsions_from_coords(
            coords_n=coords_n_np,
            coords_ca=coords_ca_np,
            coords_c=coords_c_np,
            valid_backbone_mask=valid_backbone_mask_np.astype(bool),
        )

        extra_msa_records = build_extra_msa_records(
            msa_dir=msa_file.parent,
            target_sequence=cropped_target_sequence,
            main_msa_seqs=cropped_msa_seqs,
            max_extra_msa_seqs=self.max_extra_msa_seqs,
        )
        extra_msa_feat, extra_msa_mask = build_extra_msa_features(
            extra_msa_records,
            target_len=len(cropped_target_sequence),
        )

        template_angle_feat, template_pair_feat, template_mask, template_chain_ids = (
            build_template_feature_tensors(
                query_name=query_name,
                msa_dir=msa_file.parent,
                chain_data=chain_data,
                matched_chain_id=matched_chain_id,
                target_sequence=cropped_target_sequence,
                identity_target_sequence=full_target_sequence,
                max_templates=self.max_templates,
                min_template_identity=self.min_template_identity,
            )
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
            "template_chain_ids": template_chain_ids,
            "match_identity": torch.tensor(row["match_identity"], dtype=torch.float32),
            "sequence_str": cropped_target_sequence,
            "seq_tokens": seq_tokens,
            "msa_tokens": msa_tokens,
            "msa_mask": msa_mask,
            "masked_msa_true": masked_msa_true,
            "masked_msa_mask": masked_msa_mask,
            "extra_msa_feat": extra_msa_feat,
            "extra_msa_mask": extra_msa_mask,
            "template_angle_feat": template_angle_feat,
            "template_pair_feat": template_pair_feat,
            "template_mask": template_mask,
            "coords_n": coords_n,
            "coords_ca": coords_ca,
            "coords_c": coords_c,
            "dist_map": dist_map,
            "valid_res_mask": valid_res_mask,
            "valid_backbone_mask": valid_backbone_mask,
            "torsion_true": torsion_true,
            "torsion_mask": torsion_mask,
        }
