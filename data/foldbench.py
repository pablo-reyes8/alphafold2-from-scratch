from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml


DEFAULT_A3M_FILENAME = "cfdb_hits.a3m"


@dataclass(frozen=True)
class FoldbenchManifestRecord:
    query_name: str
    chain_id: str
    msa_dir_name: str
    msa_exists: bool
    msa_dir: str
    cif_exists: bool
    cif_file: str | None
    seq_len: int
    sequence: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def load_queries(json_path: str | Path) -> Mapping[str, Any]:
    with _as_path(json_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    queries = payload.get("queries")
    if not isinstance(queries, dict):
        raise ValueError(f"Invalid Foldbench JSON payload: missing 'queries' in {json_path}")

    return queries


def choose_primary_chain_id(chain_ids: Sequence[Any]) -> str | None:
    for chain_id in chain_ids:
        if isinstance(chain_id, str) and len(chain_id) == 1 and chain_id.isalpha():
            return chain_id

    return None


def build_msa_dir_name(query_name: str, chain_id: str) -> str:
    return f"{query_name.lower()}_{chain_id}"


def find_cif_file(cif_root: str | Path, query_name: str) -> Path | None:
    root = _as_path(cif_root)
    matches = sorted(root.glob(f"{query_name.lower()}-assembly1_*.cif"))
    return matches[0] if matches else None


def build_manifest_records(
    json_path: str | Path,
    msa_root: str | Path,
    cif_root: str | Path,
) -> list[FoldbenchManifestRecord]:
    queries = load_queries(json_path)
    msa_root_path = _as_path(msa_root)
    cif_root_path = _as_path(cif_root)

    records: list[FoldbenchManifestRecord] = []

    for query_name, query_payload in sorted(queries.items()):
        chains = query_payload.get("chains", [])
        if not chains:
            continue

        chain = chains[0]
        sequence = chain.get("sequence", "")
        chain_id = choose_primary_chain_id(chain.get("chain_ids", []))
        if chain_id is None:
            continue

        msa_dir_name = build_msa_dir_name(query_name, chain_id)
        msa_dir = msa_root_path / msa_dir_name
        cif_file = find_cif_file(cif_root_path, query_name)

        records.append(
            FoldbenchManifestRecord(
                query_name=query_name,
                chain_id=chain_id,
                msa_dir_name=msa_dir_name,
                msa_exists=msa_dir.exists(),
                msa_dir=str(msa_dir),
                cif_exists=cif_file is not None,
                cif_file=str(cif_file) if cif_file is not None else None,
                seq_len=len(sequence),
                sequence=sequence,
            )
        )

    return records


def manifest_dataframe_from_records(
    records: Iterable[FoldbenchManifestRecord],
) -> pd.DataFrame:
    rows = [record.to_row() for record in records]
    if not rows:
        return pd.DataFrame(
            columns=[
                "query_name",
                "chain_id",
                "msa_dir_name",
                "msa_exists",
                "msa_dir",
                "cif_exists",
                "cif_file",
                "seq_len",
                "sequence",
            ]
        )

    return pd.DataFrame(rows)


def build_manifest_dataframe(
    json_path: str | Path,
    msa_root: str | Path,
    cif_root: str | Path,
) -> pd.DataFrame:
    return manifest_dataframe_from_records(
        build_manifest_records(
            json_path=json_path,
            msa_root=msa_root,
            cif_root=cif_root,
        )
    )


def rewrite_manifest_paths(
    manifest_df: pd.DataFrame,
    msa_root: str | Path | None = None,
    cif_root: str | Path | None = None,
) -> pd.DataFrame:
    df = manifest_df.copy()

    if "seq_len" in df.columns:
        df["seq_len"] = pd.to_numeric(df["seq_len"], errors="coerce").fillna(0).astype(int)

    for flag_col in ("msa_exists", "cif_exists"):
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    if msa_root is not None and "msa_dir_name" in df.columns:
        msa_root_path = _as_path(msa_root)
        df["msa_dir"] = [str(msa_root_path / name) for name in df["msa_dir_name"]]
        df["msa_exists"] = [(msa_root_path / name).exists() for name in df["msa_dir_name"]]

    if cif_root is not None and "query_name" in df.columns:
        cif_root_path = _as_path(cif_root)
        resolved_cifs = [find_cif_file(cif_root_path, query_name) for query_name in df["query_name"]]
        df["cif_file"] = [str(path) if path is not None else None for path in resolved_cifs]
        df["cif_exists"] = [path is not None for path in resolved_cifs]

    return df


def load_manifest_dataframe(
    manifest_csv: str | Path,
    msa_root: str | Path | None = None,
    cif_root: str | Path | None = None,
) -> pd.DataFrame:
    manifest_path = _as_path(manifest_csv)
    manifest_df = pd.read_csv(manifest_path)
    return rewrite_manifest_paths(
        manifest_df=manifest_df,
        msa_root=msa_root,
        cif_root=cif_root,
    )


def filter_complete_records(manifest_df: pd.DataFrame) -> pd.DataFrame:
    if manifest_df.empty:
        return manifest_df.copy()

    return manifest_df.loc[
        manifest_df["msa_exists"].astype(bool) & manifest_df["cif_exists"].astype(bool)
    ].reset_index(drop=True)


def derive_targets(
    manifest_df: pd.DataFrame,
    limit: int | None = None,
) -> list[str]:
    targets = manifest_df["msa_dir_name"].dropna().astype(str).drop_duplicates().tolist()
    if limit is not None:
        targets = targets[:limit]
    return targets


def write_targets_file(
    manifest_df: pd.DataFrame,
    output_path: str | Path,
    limit: int | None = None,
) -> Path:
    path = _as_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = derive_targets(manifest_df, limit=limit)
    contents = "\n".join(targets)
    if contents:
        contents += "\n"
    path.write_text(contents, encoding="utf-8")
    return path


def summarize_manifest(manifest_df: pd.DataFrame) -> dict[str, Any]:
    lengths = manifest_df["seq_len"].astype(int).tolist() if not manifest_df.empty else []
    complete_df = filter_complete_records(manifest_df)

    summary: dict[str, Any] = {
        "records": int(len(manifest_df)),
        "complete_records": int(len(complete_df)),
        "msa_available": int(manifest_df["msa_exists"].sum()) if "msa_exists" in manifest_df else 0,
        "cif_available": int(manifest_df["cif_exists"].sum()) if "cif_exists" in manifest_df else 0,
        "unique_chain_ids": sorted(manifest_df["chain_id"].dropna().astype(str).unique().tolist())
        if "chain_id" in manifest_df
        else [],
        "targets_preview": manifest_df["query_name"].head(10).astype(str).tolist()
        if "query_name" in manifest_df
        else [],
    }

    if lengths:
        summary["sequence_length"] = {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": round(sum(lengths) / len(lengths), 2),
            "median": float(median(lengths)),
        }
    else:
        summary["sequence_length"] = {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}

    if not manifest_df.empty:
        longest = manifest_df.sort_values("seq_len", ascending=False).head(5)
        summary["longest_examples"] = longest[
            ["query_name", "chain_id", "seq_len"]
        ].to_dict(orient="records")
    else:
        summary["longest_examples"] = []

    return summary


def save_yaml(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = _as_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(payload),
            handle,
            sort_keys=False,
            allow_unicode=False,
        )
    return path
