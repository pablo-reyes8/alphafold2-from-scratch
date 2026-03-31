from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from data.foldbench import (
    build_manifest_dataframe,
    filter_complete_records,
    load_manifest_dataframe,
    save_yaml,
    summarize_manifest,
    write_targets_file,
)


def _read_yaml_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    with Path(config_path).expanduser().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return payload


def _nested_get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_setting(
    args_value: Any,
    config: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    if args_value is not None:
        return args_value
    return _nested_get(config, *keys, default=default)


def build_or_load_manifest(args: argparse.Namespace, config: dict[str, Any]):
    manifest_input = _resolve_setting(
        args.manifest_input,
        config,
        ("paths", "input_manifest_csv"),
    )
    msa_root = _resolve_setting(args.msa_root, config, ("paths", "msa_root"))
    cif_root = _resolve_setting(args.cif_root, config, ("paths", "cif_root"))

    if manifest_input is not None:
        return load_manifest_dataframe(
            manifest_csv=manifest_input,
            msa_root=msa_root,
            cif_root=cif_root,
        )

    json_path = _resolve_setting(args.json_path, config, ("paths", "json_path"))
    if json_path is None or msa_root is None or cif_root is None:
        raise ValueError(
            "You must provide either --manifest-input or the trio --json-path --msa-root --cif-root."
        )

    return build_manifest_dataframe(
        json_path=json_path,
        msa_root=msa_root,
        cif_root=cif_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or normalize a Foldbench manifest and export DataOps artifacts.",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file.")
    parser.add_argument("--json-path", type=str, default=None, help="Raw Foldbench JSON manifest.")
    parser.add_argument("--msa-root", type=str, default=None, help="Directory containing MSA folders.")
    parser.add_argument("--cif-root", type=str, default=None, help="Directory containing reference mmCIF files.")
    parser.add_argument(
        "--manifest-input",
        type=str,
        default=None,
        help="Existing CSV manifest to normalize instead of rebuilding from JSON.",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        default=None,
        help="Output CSV path. Defaults to config.outputs.manifest_csv or data/Proteinas_secuencias.csv.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Output YAML summary path. Defaults to config.outputs.summary_yaml or data/Proteinas_secuencias.yaml.",
    )
    parser.add_argument(
        "--targets-output",
        type=str,
        default=None,
        help="Output TXT path for target IDs. Defaults to config.outputs.targets_txt or data/fb_targets_50.txt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional target limit applied when writing the targets txt.",
    )
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Keep rows without both MSA and CIF assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_yaml_config(args.config)

    dataset_name = _nested_get(config, "metadata", "name", default="foldbench_subset")
    keep_only_complete = not args.keep_incomplete
    if args.keep_incomplete:
        keep_only_complete = False
    elif _nested_get(config, "dataset", "keep_only_complete_records", default=None) is not None:
        keep_only_complete = bool(_nested_get(config, "dataset", "keep_only_complete_records"))

    manifest_output = _resolve_setting(
        args.manifest_output,
        config,
        ("outputs", "manifest_csv"),
        default="data/Proteinas_secuencias.csv",
    )
    summary_output = _resolve_setting(
        args.summary_output,
        config,
        ("outputs", "summary_yaml"),
        default="data/Proteinas_secuencias.yaml",
    )
    targets_output = _resolve_setting(
        args.targets_output,
        config,
        ("outputs", "targets_txt"),
        default="data/fb_targets_50.txt",
    )
    target_limit = _resolve_setting(
        args.limit,
        config,
        ("dataset", "target_limit"),
    )

    manifest_df = build_or_load_manifest(args=args, config=config)
    if keep_only_complete:
        manifest_df = filter_complete_records(manifest_df)

    manifest_path = Path(manifest_output).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)

    summary = summarize_manifest(manifest_df)
    summary_payload = {
        "metadata": {
            "name": dataset_name,
            "generated_by": "python -m data.preproces_data",
            "keep_only_complete_records": keep_only_complete,
        },
        "paths": {
            "manifest_csv": str(manifest_path),
            "targets_txt": str(Path(targets_output).expanduser()),
        },
        "summary": summary,
    }
    save_yaml(summary_payload, summary_output)
    write_targets_file(manifest_df, targets_output, limit=target_limit)

    print(f"[data] manifest rows: {len(manifest_df)}")
    print(f"[data] manifest csv:  {manifest_path}")
    print(f"[data] summary yaml:  {Path(summary_output).expanduser()}")
    print(f"[data] targets txt:   {Path(targets_output).expanduser()}")
    print(f"[data] complete rows: {summary['complete_records']}")
    print(f"[data] length stats:  {summary['sequence_length']}")


if __name__ == "__main__":
    main()
