"""Run baseline and ablation jobs sequentially and export a comparison table.

This script delegates actual training to the dedicated ablation launchers,
captures their JSON summaries, and writes CSV/Markdown comparison tables from
the final training metrics.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline/ablation sweep and export a comparison table.")
    parser.add_argument("--all", action="store_true", help="Run the full predefined ablation suite.")
    parser.add_argument("--ablation", action="append", default=None, help="Specific ablation key. Repeatable.")
    parser.add_argument("--include-baseline", action="store_true", help="Include the non-ablated baseline run.")
    parser.add_argument("--config", type=str, default="config/experiments/af2_poc.yaml")
    parser.add_argument("--manifest-csv", type=str, default=None)
    parser.add_argument("--parallel-mode", choices=("single", "ddp", "model", "hybrid"), default="single")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-devices", type=str, default=None)
    parser.add_argument("--devices-per-replica", type=int, default=2)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default=None)
    parser.add_argument("--num-recycles", type=int, default=None)
    parser.add_argument("--stochastic-recycling", action="store_true")
    parser.add_argument("--max-recycles", type=int, default=None)
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--broadcast-buffers", action="store_true")
    parser.add_argument("--single-sequence-msa", action="store_true")
    parser.add_argument("--use-block-specific-params", action="store_true")
    parser.add_argument("--output-dir", type=str, default="artifacts/ablation_suite")
    return parser.parse_args(argv)


def _selected_variants(args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    if args.include_baseline:
        selected.append("BASELINE")
    if args.all:
        selected.extend(["AF2_1", "AF2_2", "AF2_3", "AF2_4", "AF2_5"])
    elif args.ablation:
        selected.extend(args.ablation)
    if not selected:
        raise ValueError("Select at least one run via --all, --ablation, or --include-baseline.")
    return selected


def _command_for_variant(args: argparse.Namespace, variant: str, results_json: Path) -> list[str]:
    if args.parallel_mode == "single":
        command = [sys.executable, "scripts/train_ablation.py"]
    elif args.parallel_mode == "model":
        command = [sys.executable, "scripts/train_ablation_parallel.py", "--parallel-mode", "model"]
    else:
        command = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            "scripts/train_ablation_parallel.py",
            "--parallel-mode",
            args.parallel_mode,
        ]

    command.extend(["--config", args.config, "--ablation", variant, "--results-json", str(results_json)])

    if args.manifest_csv:
        command.extend(["--manifest-csv", args.manifest_csv])
    if args.device:
        command.extend(["--device", args.device])
    if args.model_devices:
        command.extend(["--model-devices", args.model_devices])
    if args.devices_per_replica != 2:
        command.extend(["--devices-per-replica", str(args.devices_per_replica)])
    if args.backend:
        command.extend(["--backend", args.backend])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.max_batches is not None:
        command.extend(["--max-batches", str(args.max_batches)])
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.resume_path is not None:
        command.extend(["--resume-path", str(args.resume_path)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.deterministic:
        command.append("--deterministic")
    if args.dry_run:
        command.append("--dry-run")
    if args.no_ema:
        command.append("--no-ema")
    if args.no_amp:
        command.append("--no-amp")
    if args.amp_dtype is not None:
        command.extend(["--amp-dtype", str(args.amp_dtype)])
    if args.num_recycles is not None:
        command.extend(["--num-recycles", str(args.num_recycles)])
    if args.stochastic_recycling:
        command.append("--stochastic-recycling")
    if args.max_recycles is not None:
        command.extend(["--max-recycles", str(args.max_recycles)])
    if args.find_unused_parameters:
        command.append("--find-unused-parameters")
    if args.broadcast_buffers:
        command.append("--broadcast-buffers")
    if args.single_sequence_msa:
        command.append("--single-sequence-msa")
    if args.use_block_specific_params:
        command.append("--use-block-specific-params")

    return command


def _write_comparison_tables(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    md_path = output_dir / "comparison.md"

    fieldnames = [
        "variant",
        "title",
        "category",
        "loss",
        "fape_loss",
        "dist_loss",
        "msa_loss",
        "plddt_loss",
        "torsion_loss",
        "rmsd_logged",
        "tm_score_logged",
        "gdt_ts_logged",
        "num_recycles",
        "global_step",
        "best_metric",
        "run_name",
        "ckpt_dir",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(fieldnames[:14]) + " |"
    separator = "| " + " | ".join(["---"] * 14) + " |"
    lines = ["# Ablation Comparison", "", header, separator]
    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames[:14]]
        lines.append("| " + " | ".join(values) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    variants = _selected_variants(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for variant in variants:
        result_json = output_dir / f"{variant.lower()}_result.json"
        command = _command_for_variant(args, variant, result_json)
        print(f"[scripts.ablations.run_suite] running: {' '.join(command)}")
        subprocess.run(command, cwd=ROOT_DIR, check=True)

        payload = json.loads(result_json.read_text(encoding="utf-8"))
        stats = dict(payload.get("result", {}).get("last_train_stats", {}) or {})
        rows.append(
            {
                "variant": payload.get("ablation"),
                "title": payload.get("title"),
                "category": payload.get("category"),
                "loss": stats.get("loss"),
                "fape_loss": stats.get("fape_loss"),
                "dist_loss": stats.get("dist_loss"),
                "msa_loss": stats.get("msa_loss"),
                "plddt_loss": stats.get("plddt_loss"),
                "torsion_loss": stats.get("torsion_loss"),
                "rmsd_logged": stats.get("rmsd_logged"),
                "tm_score_logged": stats.get("tm_score_logged"),
                "gdt_ts_logged": stats.get("gdt_ts_logged"),
                "num_recycles": stats.get("num_recycles"),
                "global_step": payload.get("result", {}).get("global_step"),
                "best_metric": payload.get("result", {}).get("best_metric"),
                "run_name": payload.get("trainer", {}).get("run_name"),
                "ckpt_dir": payload.get("trainer", {}).get("ckpt_dir"),
            }
        )

    _write_comparison_tables(rows, output_dir)
    print(f"[scripts.ablations.run_suite] wrote comparison tables under {output_dir}")


if __name__ == "__main__":
    main()
