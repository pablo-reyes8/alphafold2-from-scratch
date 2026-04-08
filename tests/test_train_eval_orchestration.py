"""Validate train/eval orchestration without running the full AlphaFold2 model."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

import training.train_alphafold2 as train_loop_module


def test_train_alphafold2_prefers_eval_metrics_when_eval_loader_is_present(tmp_path, monkeypatch):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    captured = {}

    train_stats = {
        "loss": 2.0,
        "fape_loss": 1.0,
        "dist_loss": 0.5,
        "msa_loss": 0.25,
        "plddt_loss": 0.1,
        "torsion_loss": 0.1,
        "num_recycles": 0.0,
        "rmsd_logged": 3.0,
        "tm_score_logged": 0.2,
        "gdt_ts_logged": 0.3,
        "n_seen_batches": 1,
        "n_optimizer_steps": 1,
        "n_seen_samples": 1,
        "n_metric_logs": 1,
    }
    eval_stats = {
        "loss": 0.25,
        "fape_loss": 0.2,
        "dist_loss": 0.05,
        "msa_loss": 0.02,
        "plddt_loss": 0.01,
        "torsion_loss": 0.01,
        "num_recycles": 0.0,
        "rmsd_logged": 0.4,
        "tm_score_logged": 0.9,
        "gdt_ts_logged": 0.85,
        "n_seen_batches": 1,
        "n_seen_samples": 1,
        "n_metric_logs": 1,
    }

    def fake_train_one_epoch(**kwargs):
        return dict(train_stats), kwargs["global_step"] + 1

    def fake_eval_one_epoch(**kwargs):
        return dict(eval_stats)

    def fake_maybe_save_best_and_last(**kwargs):
        captured.update(kwargs)
        return kwargs["current_metric"], True

    monkeypatch.setattr(train_loop_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_loop_module, "eval_one_epoch", fake_eval_one_epoch)
    monkeypatch.setattr(train_loop_module, "maybe_save_best_and_last", fake_maybe_save_best_and_last)

    result = train_loop_module.train_alphafold2(
        model=model,
        train_loader=[{"dummy": 1}],
        eval_loader=[{"dummy": 2}],
        optimizer=optimizer,
        criterion=object(),
        scheduler=None,
        ema=None,
        scaler=None,
        device="cpu",
        epochs=1,
        amp_enabled=False,
        grad_clip=None,
        log_every=0,
        log_grad_norm=False,
        log_mem=False,
        max_batches=1,
        num_recycles=0,
        ckpt_dir=str(tmp_path),
        save_every=0,
        save_last=False,
        monitor_name="loss",
        monitor_mode="min",
        parallel_context=SimpleNamespace(is_main_process=True, distributed=False),
    )

    assert captured["current_metric"] == eval_stats["loss"]
    assert captured["train_metrics"]["loss"] == train_stats["loss"]
    assert captured["eval_metrics"]["loss"] == eval_stats["loss"]
    assert result["best_metric"] == eval_stats["loss"]
    assert result["last_eval_stats"]["tm_score_logged"] == eval_stats["tm_score_logged"]
