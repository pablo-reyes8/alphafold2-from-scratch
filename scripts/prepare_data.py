"""CLI for downloading, normalizing, and smoke-testing dataset artifacts.

This launcher wraps the repository's data tooling so a user can download the
Foldbench subset, rebuild the manifest, and verify the resulting dataloader
without remembering the lower-level commands.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    build_dataloader_from_config,
    build_dataset_from_config,
    build_train_eval_dataloaders_from_config,
    load_yaml_config,
    summarize_batch,
    summarize_dataset,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Foldbench data and dataloaders.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_cmd = subparsers.add_parser("download", help="Wrap data/download_data.sh.")
    download_cmd.add_argument("--output-root", type=str, default=None)
    download_cmd.add_argument("--targets-file", type=str, default=None)
    download_cmd.add_argument("--targets-csv", type=str, default="data/Proteinas_secuencias.csv")
    download_cmd.add_argument("--limit", type=int, default=None)
    download_cmd.add_argument("--list-targets", action="store_true")
    download_cmd.add_argument("--no-skip-existing", action="store_true")
    download_cmd.add_argument("--skip-json", action="store_true")
    download_cmd.add_argument("--skip-msas", action="store_true")
    download_cmd.add_argument("--skip-structures", action="store_true")
    download_cmd.add_argument("--aws-bin", type=str, default=None)

    manifest_cmd = subparsers.add_parser("manifest", help="Wrap python -m data.preprocess_data.")
    manifest_cmd.add_argument("--config", type=str, default="config/data/foldbench_subset.yaml")
    manifest_cmd.add_argument("--json-path", type=str, default=None)
    manifest_cmd.add_argument("--msa-root", type=str, default=None)
    manifest_cmd.add_argument("--cif-root", type=str, default=None)
    manifest_cmd.add_argument("--manifest-input", type=str, default=None)
    manifest_cmd.add_argument("--manifest-output", type=str, default=None)
    manifest_cmd.add_argument("--summary-output", type=str, default=None)
    manifest_cmd.add_argument("--targets-output", type=str, default=None)
    manifest_cmd.add_argument("--limit", type=int, default=None)
    manifest_cmd.add_argument("--keep-incomplete", action="store_true")

    smoke_cmd = subparsers.add_parser("loader-smoke", help="Build the dataset and print one batch summary.")
    smoke_cmd.add_argument("--config", type=str, default="config/experiments/af2_poc.yaml")
    smoke_cmd.add_argument("--manifest-csv", type=str, default=None)
    smoke_cmd.add_argument("--batch-size", type=int, default=None)
    smoke_cmd.add_argument("--max-samples", type=int, default=2)

    split_smoke_cmd = subparsers.add_parser(
        "train-eval-loader-smoke",
        help="Build deterministic train/eval dataloaders and print one batch summary from each split.",
    )
    split_smoke_cmd.add_argument("--config", type=str, default="config/experiments/af2_low_vram.yaml")
    split_smoke_cmd.add_argument("--manifest-csv", type=str, default=None)
    split_smoke_cmd.add_argument("--batch-size", type=int, default=None)
    split_smoke_cmd.add_argument("--max-samples", type=int, default=2)
    split_smoke_cmd.add_argument("--eval-size", type=int, default=None)

    bootstrap_cmd = subparsers.add_parser("bootstrap", help="Run download, manifest refresh, and loader smoke in sequence.")
    bootstrap_cmd.add_argument("--data-config", type=str, default="config/data/foldbench_subset.yaml")
    bootstrap_cmd.add_argument("--experiment-config", type=str, default="config/experiments/af2_poc.yaml")
    bootstrap_cmd.add_argument("--targets-csv", type=str, default="data/Proteinas_secuencias.csv")
    bootstrap_cmd.add_argument("--limit", type=int, default=None)
    bootstrap_cmd.add_argument("--output-root", type=str, default=None)
    bootstrap_cmd.add_argument("--skip-download", action="store_true")
    bootstrap_cmd.add_argument("--skip-manifest", action="store_true")
    bootstrap_cmd.add_argument("--skip-loader-smoke", action="store_true")
    bootstrap_cmd.add_argument("--max-samples", type=int, default=2)
    bootstrap_cmd.add_argument("--batch-size", type=int, default=None)

    return parser.parse_args(argv)


def _run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"[scripts.prepare_data] running: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT_DIR, check=True, env=env)


def run_download(args: argparse.Namespace) -> None:
    command = ["bash", "data/download_data.sh"]

    if args.output_root:
        command.extend(["--output-root", args.output_root])
    if args.targets_file:
        command.extend(["--targets-file", args.targets_file])
    if args.targets_csv:
        command.extend(["--targets-csv", args.targets_csv])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.list_targets:
        command.append("--list-targets")
    if args.no_skip_existing:
        command.append("--no-skip-existing")
    if args.skip_json:
        command.append("--skip-json")
    if args.skip_msas:
        command.append("--skip-msas")
    if args.skip_structures:
        command.append("--skip-structures")

    env = None
    if args.aws_bin:
        env = dict(os.environ, AWS_BIN=args.aws_bin)

    _run_command(command, env=env)


def run_manifest(args: argparse.Namespace) -> None:
    command = [sys.executable, "-m", "data.preprocess_data", "--config", args.config]

    optional_args = {
        "--json-path": args.json_path,
        "--msa-root": args.msa_root,
        "--cif-root": args.cif_root,
        "--manifest-input": args.manifest_input,
        "--manifest-output": args.manifest_output,
        "--summary-output": args.summary_output,
        "--targets-output": args.targets_output,
    }
    for flag, value in optional_args.items():
        if value is not None:
            command.extend([flag, value])

    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.keep_incomplete:
        command.append("--keep-incomplete")

    _run_command(command)


def run_loader_smoke(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    dataset = build_dataset_from_config(
        config,
        manifest_csv=args.manifest_csv,
        max_samples=args.max_samples,
        verbose=False,
    )
    loader = build_dataloader_from_config(
        dataset,
        config,
        batch_size=args.batch_size,
        shuffle=False,
    )

    dataset_summary = summarize_dataset(dataset)
    print(f"[scripts.prepare_data] dataset summary: {dataset_summary}")
    if len(dataset) == 0:
        raise ValueError("Loader smoke found zero valid examples. Check the manifest paths and drop reasons above.")

    batch = next(iter(loader))
    print(f"[scripts.prepare_data] batch summary: {summarize_batch(batch)}")


def run_train_eval_loader_smoke(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    dataset = build_dataset_from_config(
        config,
        manifest_csv=args.manifest_csv,
        max_samples=args.max_samples,
        verbose=False,
    )
    train_loader, eval_loader, split_info = build_train_eval_dataloaders_from_config(
        dataset,
        config,
        batch_size=args.batch_size,
        shuffle=False,
        eval_size=args.eval_size,
    )

    dataset_summary = summarize_dataset(dataset)
    print(f"[scripts.prepare_data] dataset summary: {dataset_summary}")
    print(
        "[scripts.prepare_data] split summary: "
        f"train_examples={len(split_info['train_indices'])}, eval_examples={len(split_info['eval_indices'])}"
    )

    if len(dataset) == 0:
        raise ValueError("Train/eval loader smoke found zero valid examples. Check the manifest paths and drop reasons above.")
    if eval_loader is None:
        raise ValueError("Train/eval loader smoke requires eval_size > 0 and at least one eval example.")

    train_batch = next(iter(train_loader))
    eval_batch = next(iter(eval_loader))
    print(f"[scripts.prepare_data] train batch summary: {summarize_batch(train_batch)}")
    print(f"[scripts.prepare_data] eval batch summary: {summarize_batch(eval_batch)}")


def run_bootstrap(args: argparse.Namespace) -> None:
    if not args.skip_download:
        run_download(
            argparse.Namespace(
                output_root=args.output_root,
                targets_file=None,
                targets_csv=args.targets_csv,
                limit=args.limit,
                list_targets=False,
                no_skip_existing=False,
                skip_json=False,
                skip_msas=False,
                skip_structures=False,
                aws_bin=None,
            )
        )

    if not args.skip_manifest:
        run_manifest(
            argparse.Namespace(
                config=args.data_config,
                json_path=None,
                msa_root=None,
                cif_root=None,
                manifest_input=None,
                manifest_output=None,
                summary_output=None,
                targets_output=None,
                limit=args.limit,
                keep_incomplete=False,
            )
        )

    if not args.skip_loader_smoke:
        run_loader_smoke(
            argparse.Namespace(
                config=args.experiment_config,
                manifest_csv=None,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
            )
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "download":
        run_download(args)
        return

    if args.command == "manifest":
        run_manifest(args)
        return

    if args.command == "loader-smoke":
        run_loader_smoke(args)
        return

    if args.command == "train-eval-loader-smoke":
        run_train_eval_loader_smoke(args)
        return

    if args.command == "bootstrap":
        run_bootstrap(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
