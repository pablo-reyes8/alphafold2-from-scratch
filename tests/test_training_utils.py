"""Test training-loop utilities, AMP helpers, EMA handling, and recycle flag propagation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import training.train_alphafold2 as train_loop_module
from training.autocast import get_effective_amp_dtype, normalize_device_type, resolve_amp_dtype
from training.checkpoints import get_resume_state, load_checkpoint, maybe_save_best_and_last, save_checkpoint
from training.ema import EMA, ema_health
from training.scheduler_warmup import WarmupCosineLR, build_alphafold_param_groups
from training.train_one_epoch import (
    compute_grad_norm,
    move_batch_to_device,
    resolve_batch_num_recycles,
    train_one_epoch,
)


def test_resolve_amp_dtype_aliases():
    assert resolve_amp_dtype("bf16") == torch.bfloat16
    assert resolve_amp_dtype("float16") == torch.float16
    assert resolve_amp_dtype("fp32") == torch.float32


def test_get_effective_amp_dtype_cpu_behavior():
    assert get_effective_amp_dtype(device="cpu", amp_dtype="bf16") == torch.bfloat16
    assert get_effective_amp_dtype(device="cpu", amp_dtype="fp16") is None
    assert get_effective_amp_dtype(device="cpu", amp_dtype="fp32") is None


def test_normalize_device_type_collapses_indexed_strings():
    assert normalize_device_type("cuda:1") == "cuda"
    assert normalize_device_type(torch.device("cpu")) == "cpu"


def test_move_batch_to_device_preserves_non_tensors():
    batch = {"x": torch.ones(2, 3), "meta": ["a", "b"], "name": "toy"}
    moved = move_batch_to_device(batch, "cpu")
    assert moved["x"].device.type == "cpu"
    assert moved["meta"] == ["a", "b"]
    assert moved["name"] == "toy"


def test_compute_grad_norm_is_positive_after_backward():
    model = nn.Linear(4, 2)
    inputs = torch.randn(3, 4)
    target = torch.randn(3, 2)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    assert compute_grad_norm(model) > 0.0


def test_resolve_batch_num_recycles_fixed_mode():
    assert resolve_batch_num_recycles(num_recycles=3, stochastic_recycling=False) == 3
    assert resolve_batch_num_recycles(num_recycles=-2, stochastic_recycling=False) == 0


def test_resolve_batch_num_recycles_stochastic_mode_is_seeded():
    torch.manual_seed(2024)
    first = [resolve_batch_num_recycles(stochastic_recycling=True, max_recycles=3) for _ in range(6)]
    torch.manual_seed(2024)
    second = [resolve_batch_num_recycles(stochastic_recycling=True, max_recycles=3) for _ in range(6)]

    assert first == second
    assert all(0 <= value <= 3 for value in first)


def test_build_alphafold_param_groups_separates_decay_and_no_decay():
    class TinyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_embedding = nn.Embedding(8, 4)
            self.proj = nn.Linear(4, 4)
            self.norm = nn.LayerNorm(4)

    module = TinyModule()
    param_groups = build_alphafold_param_groups(module, weight_decay=1e-4)

    assert len(param_groups) == 2
    decay_group, no_decay_group = param_groups
    assert decay_group["weight_decay"] == 1e-4
    assert no_decay_group["weight_decay"] == 0.0

    no_decay_ids = {id(param) for param in no_decay_group["params"]}
    assert id(module.input_embedding.weight) in no_decay_ids
    assert id(module.proj.bias) in no_decay_ids
    assert id(module.norm.weight) in no_decay_ids
    assert id(module.norm.bias) in no_decay_ids
    assert id(module.proj.weight) not in no_decay_ids


def test_ema_update_store_restore_and_health():
    model = nn.Linear(4, 2)
    ema = EMA(model, decay=0.9, device="cpu", use_num_updates=False)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    ema.update(model)
    ok, status, rel = ema_health(ema, model, rel_tol=10.0)
    assert ok
    assert status == "ok"
    assert rel >= 0.0

    ema.store(model)
    original_weight = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.zero_()
    ema.restore(model)
    assert torch.allclose(model.weight, original_weight)


def test_checkpoint_roundtrip_and_best_last_saving(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineLR(optimizer, total_steps=10, warmup_steps=2, min_lr=1e-5)
    ema = EMA(model, decay=0.999, device="cpu", use_num_updates=False)

    ckpt_path = tmp_path / "manual.pt"
    save_checkpoint(
        path=str(ckpt_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        epoch=3,
        global_step=17,
        best_metric=0.25,
        monitor_name="loss",
        metrics={"loss": 0.25},
        config={"run_name": "toy"},
    )

    cloned_model = nn.Linear(4, 2)
    cloned_optimizer = torch.optim.AdamW(cloned_model.parameters(), lr=1e-3)
    cloned_scheduler = WarmupCosineLR(cloned_optimizer, total_steps=10, warmup_steps=2, min_lr=1e-5)
    cloned_ema = EMA(cloned_model, decay=0.999, device="cpu", use_num_updates=False)

    checkpoint = load_checkpoint(
        path=str(ckpt_path),
        model=cloned_model,
        optimizer=cloned_optimizer,
        scheduler=cloned_scheduler,
        ema=cloned_ema,
        map_location="cpu",
    )
    resume = get_resume_state(checkpoint)

    assert resume["epoch"] == 3
    assert resume["global_step"] == 17
    assert resume["best_metric"] == 0.25

    save_dir = tmp_path / "managed"
    best_metric, improved = maybe_save_best_and_last(
        save_dir=str(save_dir),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        ema=ema,
        epoch=4,
        global_step=18,
        current_metric=0.2,
        best_metric=0.25,
        metric_name="loss",
        mode="min",
        val_metrics={"loss": 0.2},
        config={"run_name": "toy"},
    )

    assert improved
    assert best_metric == 0.2
    assert (save_dir / "last.pt").exists()
    assert (save_dir / "best.pt").exists()


def test_train_one_epoch_forwards_num_recycles(toy_model, toy_batch, toy_criterion):
    class RecordingModel(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.seen_num_recycles = []

        def forward(self, *args, num_recycles=0, **kwargs):
            self.seen_num_recycles.append(num_recycles)
            return self.inner(*args, num_recycles=num_recycles, **kwargs)

    model = RecordingModel(toy_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    stats, global_step = train_one_epoch(
        model=model,
        dataloader=[toy_batch],
        optimizer=optimizer,
        criterion=toy_criterion,
        device="cpu",
        amp_enabled=False,
        scaler=None,
        scheduler=None,
        ema=None,
        grad_clip=None,
        grad_accum_steps=1,
        max_batches=1,
        log_every=0,
        log_grad_norm=False,
        log_mem=False,
        on_oom="skip",
        global_step=0,
        ideal_backbone_local=toy_batch["ideal_backbone_local"],
        num_recycles=2,
    )

    assert model.seen_num_recycles == [2]
    assert stats["n_seen_batches"] == 1
    assert stats["num_recycles"] == 2.0
    assert global_step == 1


def test_train_one_epoch_stochastic_recycling_samples_per_batch(toy_batch):
    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.seen_num_recycles = []

        def forward(self, *args, num_recycles=0, **kwargs):
            self.seen_num_recycles.append(num_recycles)
            pred = self.weight * kwargs["seq_tokens"].float().mean()
            return {"pred": pred}

    class DummyCriterion:
        def __call__(self, outputs, batch):
            loss = outputs["pred"].pow(2)
            zero = loss.detach() * 0.0
            return {
                "loss": loss,
                "fape_loss": zero,
                "dist_loss": zero,
                "msa_loss": zero,
                "plddt_loss": zero,
                "torsion_loss": zero,
            }

    model = RecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = [toy_batch, toy_batch, toy_batch]

    torch.manual_seed(99)
    expected = [resolve_batch_num_recycles(stochastic_recycling=True, max_recycles=3) for _ in dataloader]

    torch.manual_seed(99)
    stats, global_step = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=DummyCriterion(),
        device="cpu",
        amp_enabled=False,
        scaler=None,
        scheduler=None,
        ema=None,
        grad_clip=None,
        grad_accum_steps=1,
        max_batches=None,
        log_every=0,
        log_grad_norm=False,
        log_mem=False,
        on_oom="skip",
        global_step=0,
        ideal_backbone_local=None,
        num_recycles=0,
        stochastic_recycling=True,
        max_recycles=3,
    )

    assert model.seen_num_recycles == expected
    assert stats["num_recycles"] == pytest.approx(sum(expected) / len(expected))
    assert global_step == len(dataloader)


def test_train_alphafold2_forwards_recycling_flags(monkeypatch, tmp_path):
    captured = {}

    def fake_train_one_epoch(**kwargs):
        captured["num_recycles"] = kwargs["num_recycles"]
        captured["stochastic_recycling"] = kwargs["stochastic_recycling"]
        captured["max_recycles"] = kwargs["max_recycles"]
        captured["global_step"] = kwargs["global_step"]
        return {
            "loss": 1.0,
            "fape_loss": 0.5,
            "dist_loss": 0.3,
            "msa_loss": 0.2,
            "plddt_loss": 0.1,
            "torsion_loss": 0.1,
            "num_recycles": 1.5,
            "rmsd_logged": float("nan"),
            "tm_score_logged": float("nan"),
            "gdt_ts_logged": float("nan"),
            "n_seen_batches": 1,
            "n_optimizer_steps": 1,
            "n_seen_samples": 1,
            "n_metric_logs": 0,
        }, kwargs["global_step"] + 1

    monkeypatch.setattr(train_loop_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        train_loop_module,
        "maybe_save_best_and_last",
        lambda **kwargs: (kwargs["best_metric"], False),
    )

    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = train_loop_module.train_alphafold2(
        model=model,
        train_loader=[],
        optimizer=optimizer,
        criterion=object(),
        scheduler=None,
        ema=None,
        scaler=None,
        device="cpu",
        epochs=1,
        amp_enabled=False,
        grad_clip=None,
        grad_accum_steps=1,
        log_every=0,
        log_grad_norm=False,
        log_mem=False,
        max_batches=1,
        on_oom="skip",
        ideal_backbone_local=None,
        num_recycles=0,
        stochastic_recycling=True,
        max_recycles=3,
        ckpt_dir=str(tmp_path / "checkpoints"),
        save_every=0,
        save_last=False,
        monitor_name="loss",
        monitor_mode="min",
        best_metric=None,
        config=None,
    )

    assert captured["num_recycles"] == 0
    assert captured["stochastic_recycling"] is True
    assert captured["max_recycles"] == 3
    assert captured["global_step"] == 0
    assert result["global_step"] == 1
