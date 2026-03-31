import inspect
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32}


def resolve_amp_dtype(
    amp_dtype: str = "bf16",
    device: str = "cuda") -> torch.dtype:

    """
    Resolve the requested AMP dtype to a torch.dtype.
    """
    amp_dtype = amp_dtype.lower()
    if amp_dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported amp dtype: {amp_dtype}")
    return DTYPE_MAP[amp_dtype]


def cuda_supports_bf16() -> bool:
    """
    Check whether current CUDA device supports bfloat16 autocast.
    """
    if not torch.cuda.is_available():
        return False

    # PyTorch exposes this helper in modern versions
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return torch.cuda.is_bf16_supported()
        except Exception:
            pass

    # Conservative fallback
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def get_effective_amp_dtype(
    amp_dtype: str = "bf16",
    device: str = "cuda") -> Optional[torch.dtype]:
    """
    Decide the actually usable AMP dtype on the current device.

    Returns:
        torch.dtype if AMP should be used
        None if AMP should be disabled / no-op
    """
    want = resolve_amp_dtype(amp_dtype, device=device)

    if device == "cuda":
        if not torch.cuda.is_available():
            return None

        if want == torch.bfloat16:
            return torch.bfloat16 if cuda_supports_bf16() else torch.float16

        if want == torch.float16:
            return torch.float16

        if want == torch.float32:
            return None

        return None

    if device == "cpu":
        # CPU autocast is mostly meaningful in bf16
        if want == torch.bfloat16:
            return torch.bfloat16
        return None

    return None


def should_use_grad_scaler(
    device: str = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16") -> bool:

    """
    GradScaler is useful for fp16, but usually not needed for bf16.
    """
    if not amp_enabled:
        return False

    effective_dtype = get_effective_amp_dtype(amp_dtype=amp_dtype, device=device)

    if device == "cuda" and effective_dtype == torch.float16:
        return True

    return False


def make_grad_scaler(
    device: str = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16"):
    """
    Build a GradScaler only when it is actually useful.
    """
    enabled = should_use_grad_scaler(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype)

    if not enabled:
        return None

    # Newer API
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)
            if len(sig.parameters) >= 1:
                return torch.amp.GradScaler(device_type=device)
            return torch.amp.GradScaler()
        except Exception:
            pass

    # Older CUDA API
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()

    return None


@contextmanager
def autocast_ctx(
    device: str = "cuda",
    enabled: bool = True,
    amp_dtype: str = "bf16",
    cache_enabled: bool = True):
    """
    Robust autocast context for AlphaFold2-like training.

    Behavior:
    - CUDA + bf16 requested:
        uses bf16 if supported, else falls back to fp16
    - CUDA + fp16 requested:
        uses fp16
    - CUDA + fp32 requested:
        disables autocast
    - CPU + bf16 requested:
        uses cpu autocast bf16 if available
    - otherwise:
        no-op
    """
    if not enabled:
        with nullcontext():
            yield
        return

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device)

    if effective_dtype is None:
        with nullcontext():
            yield
        return

    if device == "cuda":
        with torch.amp.autocast(
            device_type="cuda",
            dtype=effective_dtype,
            cache_enabled=cache_enabled):
          
            yield
        return

    if device == "cpu":
        try:
            with torch.amp.autocast(
                device_type="cpu",
                dtype=effective_dtype,
                cache_enabled=cache_enabled):
              
                yield
        except Exception:
            with nullcontext():
                yield
        return

    with nullcontext():
        yield

def build_amp_config(
    device: str = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16"):
  
    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device)

    scaler = make_grad_scaler(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype)

    return {
        "amp_enabled": amp_enabled and (effective_dtype is not None),
        "amp_dtype_requested": amp_dtype,
        "amp_dtype_effective": effective_dtype,
        "use_grad_scaler": scaler is not None,
        "scaler": scaler}

import os
import random
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _safe_state_dict(obj):
    return None if obj is None else obj.state_dict()


def get_rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate()}

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not isinstance(state, dict):
        return
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: Optional[float] = None,
    monitor_name: str = "val_loss",
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    save_optimizer_state: bool = True,
    save_rng_state: bool = True):
  
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = unwrap_model(model)

    ckpt = {
        "model": model_to_save.state_dict(),
        "ema": _safe_state_dict(ema),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": None if best_metric is None else float(best_metric),
        "monitor_name": monitor_name,
        "metrics": metrics or {},
        "config": config or {},
        "rng_state": get_rng_state() if save_rng_state else None}

    if save_optimizer_state:
        ckpt["optimizer"] = _safe_state_dict(optimizer)
        ckpt["scheduler"] = _safe_state_dict(scheduler)
        ckpt["scaler"] = _safe_state_dict(scaler)
    else:
        ckpt["optimizer"] = None
        ckpt["scheduler"] = None
        ckpt["scaler"] = None

    torch.save(ckpt, str(path))


def save_weights_only_checkpoint(
    path: str,
    model,
    ema=None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    monitor_name: str = "val_loss"):
  
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = unwrap_model(model)

    ckpt = {
        "model": model_to_save.state_dict(),
        "ema": _safe_state_dict(ema),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": metrics or {},
        "monitor_name": monitor_name}

    torch.save(ckpt, str(path))


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    map_location="cpu",
    strict: bool = True,
    load_optimizer_state: bool = True,
    restore_rng_state: bool = False):
  
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model_to_load = unwrap_model(model)
    model_to_load.load_state_dict(ckpt["model"], strict=strict)

    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])

    if load_optimizer_state:
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

    if restore_rng_state and ckpt.get("rng_state") is not None:
        set_rng_state(ckpt["rng_state"])

    return ckpt



def get_resume_state(ckpt: Dict[str, Any]):
    """
    Extract standard resume info.
    """
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "best_metric": ckpt.get("best_metric", None),
        "monitor_name": ckpt.get("monitor_name", "val_loss"),
        "metrics": ckpt.get("metrics", {}),
        "config": ckpt.get("config", {})}


def is_better_metric(current: float, best: Optional[float], mode: str = "min") -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    elif mode == "max":
        return current > best
    else:
        raise ValueError(f"mode must be 'min' or 'max', got {mode}")


def maybe_save_best_and_last(
    save_dir: str,
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    epoch: int,
    global_step: int,
    current_metric: float,
    best_metric: Optional[float],
    metric_name: str,
    mode: str,
    val_metrics: Dict[str, float],
    config: Optional[Dict[str, Any]] = None):
  
    os.makedirs(save_dir, exist_ok=True)

    improved = is_better_metric(current_metric, best_metric, mode=mode)
    new_best_metric = current_metric if improved else best_metric

    # save LAST with the UPDATED best metric
    save_checkpoint(
        path=os.path.join(save_dir, "last.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        epoch=epoch,
        global_step=global_step,
        best_metric=new_best_metric,
        monitor_name=metric_name,
        metrics=val_metrics,
        config=config,
        save_optimizer_state=True,
        save_rng_state=True)

    if improved:
        save_weights_only_checkpoint(
            path=os.path.join(save_dir, "best.pt"),
            model=model,
            ema=ema,
            epoch=epoch,
            global_step=global_step,
            metrics=val_metrics,
            monitor_name=metric_name)

    return new_best_metric, improved

import copy
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
 

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class EMA:
    """
    Exponential Moving Average over trainable model parameters.

    Features
    --------
    - stores shadow weights in fp32
    - maps by parameter name (robust across checkpointing / requires_grad changes)
    - supports optional offloading to cpu
    - can temporarily swap EMA weights into the model for evaluation

    Parameters
    ----------
    model : nn.Module
    decay : float
        Base EMA decay, e.g. 0.999 or 0.9999
    device : str | torch.device | None
        Where to store the shadow weights. Use "cpu" to save GPU memory.
    use_num_updates : bool
        If True, adapt effective decay at early steps.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[str | torch.device] = None,
        use_num_updates: bool = True):
      
        self.decay = float(decay)
        self.device = device
        self.use_num_updates = bool(use_num_updates)
        self.num_updates = 0

        model = unwrap_model(model)

        self.shadow = {}
        self.backup = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                s = p.detach().to(dtype=torch.float32).clone()
                if self.device is not None:
                    s = s.to(self.device)
                self.shadow[name] = s

    def _get_decay(self) -> float:
        """
        Optionally use a lower EMA decay early in training, then asymptote to self.decay.
        This is common and helps EMA start tracking sensibly.
        """
        if not self.use_num_updates:
            return self.decay

        # simple warmup-style schedule
        # starts lower, approaches target decay as updates grow
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        return float(d)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA shadow from current model parameters.
        """
        model = unwrap_model(model)
        decay = self._get_decay()

        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                s = self.shadow[name]
                p32 = p.detach().to(dtype=torch.float32)

                if s.device != p32.device:
                    p32 = p32.to(s.device)

                s.mul_(decay).add_(p32, alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        Copy EMA weights into model parameters.
        """
        model = unwrap_model(model)
        for name, p in model.named_parameters():
            if name in self.shadow:
                s = self.shadow[name]
                p.data.copy_(s.to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module):
        """
        Store current model params so we can later restore them after EMA eval.
        """
        model = unwrap_model(model)
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.detach().clone()

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """
        Restore model params previously saved by store().
        """
        model = unwrap_model(model)
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].to(device=p.device, dtype=p.dtype))
        self.backup = {}

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """
        Temporarily swap EMA weights into the model for evaluation.

        Usage:
            with ema.average_parameters(model):
                val_metrics = validate(...)
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    @torch.no_grad()
    def to(self, device: str | torch.device):
        """
        Move EMA shadow weights to a new device.
        """
        self.device = device
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)

    @torch.no_grad()
    def state_dict(self):
        """
        Safe checkpoint state.
        """
        return {
            "decay": self.decay,
            "device": str(self.device) if self.device is not None else None,
            "use_num_updates": self.use_num_updates,
            "num_updates": self.num_updates,
            "shadow": {name: s.detach().cpu() for name, s in self.shadow.items()}}

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        Restore EMA state from checkpoint.
        """
        self.decay = float(state_dict.get("decay", self.decay))
        self.use_num_updates = bool(state_dict.get("use_num_updates", self.use_num_updates))
        self.num_updates = int(state_dict.get("num_updates", self.num_updates))

        loaded_shadow = state_dict.get("shadow", {})

        for name, s in self.shadow.items():
            if name in loaded_shadow:
                s.data.copy_(loaded_shadow[name].to(device=s.device, dtype=s.dtype))
            else:
                print(f"[EMA] Warning: parameter '{name}' missing in checkpoint.")

    def __len__(self):
        return len(self.shadow)

@torch.no_grad()
def ema_health(ema: EMA, model: nn.Module, rel_tol: float = 5.0):
    """
    Basic sanity check comparing EMA weights against current model weights.

    Returns
    -------
    (ok: bool, status: str, rel_diff: float)
    """
    model = unwrap_model(model)

    def _flat(t):
        return t.detach().float().cpu().reshape(-1)

    m_params = []
    e_params = []

    for name, p in model.named_parameters():
        if name in ema.shadow:
            m_params.append(p)
            e_params.append(ema.shadow[name])

    if not m_params:
        return (False, "empty_ema", float("inf"))

    m_flat = torch.cat([_flat(p) for p in m_params], dim=0)
    e_flat = torch.cat([_flat(s) for s in e_params], dim=0)

    if not torch.isfinite(e_flat).all():
        return (False, "nan_or_inf_in_ema", float("inf"))

    m_norm = m_flat.norm().item()
    e_norm = e_flat.norm().item()

    if e_norm < 1e-12:
        return (False, "ema_zero_norm", float("inf"))
    if m_norm < 1e-12:
        return (False, "model_zero_norm", float("inf"))

    rel = (m_flat - e_flat).norm().item() / (m_norm + 1e-8)

    if rel > rel_tol:
        return (False, "large_rel_diff", rel)

    return (True, "ok", rel)


@torch.no_grad()
def ema_reinit_from_model(ema: EMA, model: nn.Module):
    """
    Hard reset EMA weights from current model weights.
    """
    model = unwrap_model(model)
    for name, p in model.named_parameters():
        if name in ema.shadow:
            s = ema.shadow[name]
            s.data.copy_(p.detach().to(dtype=torch.float32, device=s.device))


def ema_set_decay(ema: EMA, new_decay: float):
    ema.decay = float(new_decay)

import math
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn


def build_alphafold_param_groups(
    model: nn.Module,
    weight_decay: float = 1e-4):
  
    """
    Build optimizer parameter groups for AlphaFold2-like model.

    No weight decay for:
      - biases
      - normalization params
      - embeddings
      - 1D parameters

    Returns
    -------
    List[dict]
    """
    model = model.module if hasattr(model, "module") else model

    decay_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        name_l = name.lower()

        is_bias = name.endswith(".bias")
        is_norm = (
            "norm" in name_l or
            "layernorm" in name_l or
            ".ln" in name_l or
            "batchnorm" in name_l or
            ".bn" in name_l or
            "groupnorm" in name_l)
        
        is_embedding = "embedding" in name_l or "embed" in name_l
        is_1d = p.ndim <= 1

        if is_bias or is_norm or is_embedding or is_1d:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}]


class WarmupCosineLR:
    """
    Linear warmup, then cosine decay to min_lr. Step-based scheduler.

    Behavior
    --------
    - steps 1..warmup_steps:
        lr increases linearly from 0 to base_lr
    - after warmup:
        cosine decay from base_lr to min_lr
    - resume-safe through state_dict / load_state_dict
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 0.0):
      
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.step_num = 0

    def _compute_lr(self, base_lr: float, t: int) -> float:
        # Warmup
        if self.warmup_steps > 0 and t <= self.warmup_steps:
            return base_lr * (t / max(1, self.warmup_steps))

        # Cosine phase
        if self.total_steps <= self.warmup_steps:
            # Degenerate case: just stay at min_lr after warmup region
            return self.min_lr

        tt = min(max(t, self.warmup_steps), self.total_steps)
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = (tt - self.warmup_steps) / denom
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.min_lr + (base_lr - self.min_lr) * cosine
        return lr

    def _set_lr(self, t: int):
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            group["lr"] = self._compute_lr(base_lr, t)

    def step(self):
        self.step_num += 1
        self._set_lr(self.step_num)

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step_num": int(self.step_num),
            "base_lrs": list(self.base_lrs),
            "min_lr": float(self.min_lr),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps)}

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return

        self.step_num = int(state_dict.get("step_num", 0))

        loaded_base_lrs = state_dict.get("base_lrs", None)
        if isinstance(loaded_base_lrs, (list, tuple)) and len(loaded_base_lrs) == len(self.optimizer.param_groups):
            self.base_lrs = [float(x) for x in loaded_base_lrs]

        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))

        # restore LR exactly to the resumed step
        self._set_lr(self.step_num)

def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    total_steps: int = 100000,
    warmup_steps: int = 5000,
    min_lr: float = 1e-6):
  
    """
    Build AdamW optimizer + warmup cosine scheduler for AlphaFold2-like training.
    """
    param_groups = build_alphafold_param_groups(
        model=model,
        weight_decay=weight_decay)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps)

    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr)

    return optimizer, scheduler

import numpy as np
def seed_everything(seed: int = 0, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)



import math
import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False, eps: float = 1e-8):
    mask = mask.to(x.dtype)
    num = (x * mask).sum(dim=dim, keepdim=keepdim)
    den = mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return num / den


def center_coordinates(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    x:    [B, L, 3]
    mask: [B, L]

    returns:
      x_centered: [B, L, 3]
      centroid:   [B, 1, 3]
    """
    centroid = masked_mean(x, mask[..., None], dim=1, keepdim=True, eps=eps)
    x_centered = x - centroid
    x_centered = x_centered * mask[..., None]
    return x_centered, centroid


def kabsch_align(x_pred: torch.Tensor, x_true: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    Batched Kabsch alignment.
    
    Inputs
    ------
    x_pred : [B, L, 3]
    x_true : [B, L, 3]
    mask   : [B, L]

    Returns
    -------
    x_pred_aligned : [B, L, 3]
    R              : [B, 3, 3]
    t              : [B, 1, 3]
    """
    B, L, _ = x_pred.shape
    mask_f = mask.to(x_pred.dtype)

    x_pred_c, pred_centroid = center_coordinates(x_pred, mask, eps=eps)
    x_true_c, true_centroid = center_coordinates(x_true, mask, eps=eps)

    # Covariance: H = X_pred^T X_true
    H = torch.matmul((x_pred_c * mask_f[..., None]).transpose(1, 2), x_true_c)  # [B, 3, 3]

    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.transpose(-1, -2)

    # Reflection correction
    det = torch.det(torch.matmul(V, U.transpose(-1, -2)))
    D = torch.eye(3, device=x_pred.device, dtype=x_pred.dtype).unsqueeze(0).repeat(B, 1, 1)
    D[:, -1, -1] = torch.where(det < 0, -1.0, 1.0)

    R = torch.matmul(torch.matmul(V, D), U.transpose(-1, -2))  # [B,3,3]

    x_pred_aligned = torch.matmul(x_pred_c, R.transpose(-1, -2)) + true_centroid
    x_pred_aligned = x_pred_aligned * mask[..., None]

    t = true_centroid - torch.matmul(pred_centroid, R.transpose(-1, -2))
    return x_pred_aligned, R, t

def rmsd_metric(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True,
    eps: float = 1e-8):
  
    """
    RMSD per structure and mean RMSD over batch.

    Inputs
    ------
    x_pred : [B, L, 3]
    x_true : [B, L, 3]
    mask   : [B, L]

    Returns
    -------
    rmsd_per_sample : [B]
    rmsd_mean       : scalar tensor
    """
    if align:
        x_pred_use, _, _ = kabsch_align(x_pred, x_true, mask, eps=eps)
    else:
        x_pred_use = x_pred

    sq_err = ((x_pred_use - x_true) ** 2).sum(dim=-1)   # [B,L]
    mask_f = mask.to(sq_err.dtype)

    mse_per_sample = (sq_err * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)
    rmsd_per_sample = torch.sqrt(mse_per_sample + eps)
    rmsd_mean = rmsd_per_sample.mean()

    return rmsd_per_sample, rmsd_mean

def tm_score_metric(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True,
    eps: float = 1e-8):
  
    """
    TM-score per structure and batch mean.

    Inputs
    ------
    x_pred : [B, L, 3]
    x_true : [B, L, 3]
    mask   : [B, L]

    Returns
    -------
    tm_per_sample : [B]
    tm_mean       : scalar tensor
    """
    if align:
        x_pred_use, _, _ = kabsch_align(x_pred, x_true, mask, eps=eps)
    else:
        x_pred_use = x_pred

    dist = torch.sqrt(((x_pred_use - x_true) ** 2).sum(dim=-1) + eps)  # [B,L]
    mask_f = mask.to(dist.dtype)

    L_eff = mask_f.sum(dim=-1).clamp_min(1.0)  # [B]

    # Standard-ish d0 formula, clamped for small proteins
    d0 = 1.24 * torch.clamp(L_eff - 15.0, min=1.0) ** (1.0 / 3.0) - 1.8
    d0 = torch.clamp(d0, min=0.5)  # avoid pathological small values

    score = 1.0 / (1.0 + (dist / d0[:, None]) ** 2)  # [B,L]
    tm_per_sample = (score * mask_f).sum(dim=-1) / L_eff
    tm_mean = tm_per_sample.mean()

    return tm_per_sample, tm_mean

def gdt_ts_metric(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True,
    thresholds=(1.0, 2.0, 4.0, 8.0),
    eps: float = 1e-8):
  
    """
    GDT-TS per structure and batch mean.

    Inputs
    ------
    x_pred : [B, L, 3]
    x_true : [B, L, 3]
    mask   : [B, L]

    Returns
    -------
    gdt_per_sample : [B]
    gdt_mean       : scalar tensor
    """
    if align:
        x_pred_use, _, _ = kabsch_align(x_pred, x_true, mask, eps=eps)
    else:
        x_pred_use = x_pred

    dist = torch.sqrt(((x_pred_use - x_true) ** 2).sum(dim=-1) + eps)  # [B,L]
    mask_f = mask.to(dist.dtype)
    L_eff = mask_f.sum(dim=-1).clamp_min(1.0)

    scores = []
    for thr in thresholds:
        within = (dist <= thr).to(dist.dtype)
        frac = (within * mask_f).sum(dim=-1) / L_eff
        scores.append(frac)

    gdt_per_sample = torch.stack(scores, dim=-1).mean(dim=-1)  # [B]
    gdt_mean = gdt_per_sample.mean()

    return gdt_per_sample, gdt_mean

@torch.no_grad()
def compute_structure_metrics(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True):
  
    """
    Compute RMSD, TM-score, GDT-TS.
    
    Returns
    -------
    metrics : dict
    """
    rmsd_per_sample, rmsd_mean = rmsd_metric(x_pred, x_true, mask, align=align)
    tm_per_sample, tm_mean = tm_score_metric(x_pred, x_true, mask, align=align)
    gdt_per_sample, gdt_mean = gdt_ts_metric(x_pred, x_true, mask, align=align)

    return {
        "rmsd_per_sample": rmsd_per_sample,
        "tm_per_sample": tm_per_sample,
        "gdt_ts_per_sample": gdt_per_sample,
        "rmsd": rmsd_mean,
        "tm_score": tm_mean,
        "gdt_ts": gdt_mean}

import time
import torch
import torch.nn as nn


def move_batch_to_device(batch, device: str):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def compute_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5

def gpu_mem_mb(device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserv = torch.cuda.memory_reserved() / (1024**2)
        return alloc, reserv
    return 0.0, 0.0

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,                      # AlphaFoldLoss
    *,
    device: str = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    scaler=None,
    scheduler=None,
    ema=None,
    ema_target=None,
    grad_clip: float | None = 1.0,
    grad_accum_steps: int = 1,
    max_batches: int | None = None,
    log_every: int = 10,
    log_grad_norm: bool = True,
    log_mem: bool = False,
    on_oom: str = "skip",
    global_step: int = 0,
    ideal_backbone_local: torch.Tensor | None = None):
  
    """
    Train one epoch for AlphaFold2-like model.

    Returns
    -------
    epoch_stats : dict
    global_step : int
    """
    model.train()

    if ema_target is None:
        ema_target = model

    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    running = {
        "loss": 0.0,
        "fape_loss": 0.0,
        "dist_loss": 0.0,
        "plddt_loss": 0.0,
        "torsion_loss": 0.0,
        "rmsd": 0.0,
        "tm_score": 0.0,
        "gdt_ts": 0.0}

    n_seen_batches = 0
    n_optimizer_steps = 0
    n_seen_samples = 0

    # header bonito
    if log_every:
        print("┆ In-epoch statistics (AlphaFold2-like)")
        print("┆   {:>8} | {:>8} | {:>9} | {:>9} | {:>9} | {:>8} | {:>8} | {:>8}{}".format(
            "step", "batch", "loss", "fape", "dist", "rmsd", "tm", "gdt",
            (" | grad_norm | mem(MB)" if (log_grad_norm or log_mem) else "")))
        print("┆   " + "─" * 110)

    for i, batch in enumerate(dataloader):
        if (max_batches is not None) and (i >= max_batches):
            break

        try:
            t0 = time.perf_counter()

            batch = move_batch_to_device(batch, device)
            B = batch["seq_tokens"].shape[0]
            n_seen_samples += B

            with autocast_ctx( device=device, enabled=amp_enabled, amp_dtype=amp_dtype,):

                out = model(
                    seq_tokens=batch["seq_tokens"],
                    msa_tokens=batch["msa_tokens"],
                    seq_mask=batch["seq_mask"],
                    msa_mask=batch["msa_mask"],
                    ideal_backbone_local=ideal_backbone_local,)

                loss_dict = criterion(out, batch)
                loss = loss_dict["loss"] / grad_accum_steps

            # -------------------------
            # backward
            # -------------------------
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_now = ((i + 1) % grad_accum_steps) == 0
            grad_norm = None

            if step_now:
                did_unscale = False

                if log_grad_norm:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        did_unscale = True
                    grad_norm = compute_grad_norm(model)

                if grad_clip is not None:
                    if scaler is not None and (not did_unscale):
                        scaler.unscale_(optimizer)
                        did_unscale = True

                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                if ema is not None:
                    ema.update(ema_target)

                global_step += 1
                n_optimizer_steps += 1

            # -------------------------
            # métricas estructurales
            # -------------------------
            with torch.no_grad():
                x_true = batch["coords_ca"]
                mask = batch["valid_res_mask"]

                if out.get("backbone_coords", None) is not None:
                    # ideal_backbone_local = [N, CA, C, O], entonces CA índice 1
                    x_pred = out["backbone_coords"][:, :, 1, :]
                else:
                    x_pred = out["t"]

                metrics = compute_structure_metrics(
                    x_pred=x_pred,
                    x_true=x_true,
                    mask=mask,
                    align=True,
                )

            # -------------------------
            # acumular stats
            # -------------------------
            loss_val = float(loss_dict["loss"].detach().item())
            running["loss"] += loss_val
            running["fape_loss"] += float(loss_dict["fape_loss"].item())
            running["dist_loss"] += float(loss_dict["dist_loss"].item())
            running["plddt_loss"] += float(loss_dict["plddt_loss"].item())
            running["torsion_loss"] += float(loss_dict["torsion_loss"].item())
            running["rmsd"] += float(metrics["rmsd"].item())
            running["tm_score"] += float(metrics["tm_score"].item())
            running["gdt_ts"] += float(metrics["gdt_ts"].item())

            n_seen_batches += 1

            # -------------------------
            # logging
            # -------------------------
            if log_every and step_now and (global_step % log_every == 0):
                dt_ms = (time.perf_counter() - t0) * 1000.0

                mem_msg = ""
                if log_mem:
                    alloc, reserv = gpu_mem_mb(device)
                    mem_msg = f"{alloc:.0f}/{reserv:.0f}"
                else:
                    mem_msg = "—"

                gn_str = f"{grad_norm:.2e}" if grad_norm is not None else "—"

                print(
                    "┆   {:8d} | {:8d} | {:9.4f} | {:9.4f} | {:9.4f} | {:8.3f} | {:8.3f} | {:8.3f} | {:>9} | {:>9} | {:7.1f}ms".format(
                        global_step,
                        i + 1,
                        loss_val,
                        float(loss_dict["fape_loss"].item()),
                        float(loss_dict["dist_loss"].item()),
                        float(metrics["rmsd"].item()),
                        float(metrics["tm_score"].item()),
                        float(metrics["gdt_ts"].item()),
                        gn_str,
                        mem_msg,
                        dt_ms))

        except RuntimeError as e:
            if ("CUDA out of memory" in str(e)) and (on_oom == "skip"):
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[WARN][OOM] Batch {i} omitido. Limpié cache y sigo.")
                optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise

    # -------------------------
    # promedios de epoch
    # -------------------------
    denom = max(1, n_seen_batches)
    epoch_stats = {
        "loss": running["loss"] / denom,
        "fape_loss": running["fape_loss"] / denom,
        "dist_loss": running["dist_loss"] / denom,
        "plddt_loss": running["plddt_loss"] / denom,
        "torsion_loss": running["torsion_loss"] / denom,
        "rmsd": running["rmsd"] / denom,
        "tm_score": running["tm_score"] / denom,
        "gdt_ts": running["gdt_ts"] / denom,
        "n_seen_batches": n_seen_batches,
        "n_optimizer_steps": n_optimizer_steps,
        "n_seen_samples": n_seen_samples}

    return epoch_stats, global_step

import os
import sys
import shutil
import time


def _fmt_hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _rule(w=110, ch="─"):
    return ch * w


def _is_colab():
    return "google.colab" in sys.modules


def _ensure_drive_mounted():
    if _is_colab():
        drive_root = "/content/drive"
        if not os.path.isdir(drive_root):
            try:
                from google.colab import drive
                drive.mount(drive_root, force_remount=False)
            except Exception as e:
                print(f"[DRIVE] No se pudo montar automáticamente: {e}")


def _copy_ckpt_to_drive_fixed(src_path: str, drive_dir: str, fixed_name: str = "latest_alphafold2.pt"):
    try:
        if not drive_dir:
            return
        if drive_dir.startswith("/content/drive"):
            _ensure_drive_mounted()
        os.makedirs(drive_dir, exist_ok=True)
        dst_path = os.path.join(drive_dir, fixed_name)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copy2(src_path, dst_path)
        print(f"└─ [DRIVE] copiado → {dst_path}")
    except Exception as e:
        print(f"└─ [DRIVE] ERROR al copiar a Drive: {e}")

def train_alphafold2(
    *,
    model,
    train_loader,
    optimizer,
    criterion,
    scheduler=None,
    ema=None,
    scaler=None,
    device: str = "cuda",
    epochs: int = 10,
    start_epoch: int = 0,
    global_step: int = 0,
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    grad_clip: float | None = 1.0,
    grad_accum_steps: int = 1,
    log_every: int = 10,
    log_grad_norm: bool = True,
    log_mem: bool = False,
    max_batches: int | None = None,
    on_oom: str = "skip",
    ideal_backbone_local: torch.Tensor | None = None,

    # checkpoint / monitoring
    ckpt_dir: str = "checkpoints",
    run_name: str = "alphafold2",
    save_every: int = 1,
    save_last: bool = True,
    monitor_name: str = "loss",        # e.g. "loss", "rmsd", "tm_score", "gdt_ts"
    monitor_mode: str = "min",         # "min" for loss/rmsd, "max" for tm/gdt
    best_metric: float | None = None,
    config: dict | None = None,

    # optional resume
    resume_path: str | None = None,
    strict_resume: bool = True,
    restore_rng_state: bool = False,

    # drive mirror
    drive_ckpt_dir: str | None = None,
    copy_fixed_to_drive: bool = True,
    fixed_drive_name: str = "latest_alphafold2.pt"):
  
    """
    High-level training orchestrator for AlphaFold2-like model.

    Assumes model / optimizer / scheduler / ema / scaler / criterion are already created externally.
    """
    os.makedirs(ckpt_dir, exist_ok=True)


    # Optional resume
    if resume_path is not None and os.path.exists(resume_path):
        ckpt = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            map_location=device,
            strict=strict_resume,
            load_optimizer_state=True,
            restore_rng_state=restore_rng_state)
        
        resume_state = get_resume_state(ckpt)
        start_epoch = int(resume_state["epoch"]) + 1
        global_step = int(resume_state["global_step"])
        best_metric = resume_state["best_metric"]

        print(f"[RESUME] {resume_path}")
        print(f"[RESUME] start_epoch={start_epoch} | global_step={global_step} | best_metric={best_metric}")


    # Header
    ema_decay_val = getattr(ema, "decay", None)
    ema_str = f"{ema_decay_val:.6f}" if isinstance(ema_decay_val, (float, int)) else ("on" if ema is not None else "off")

    lr_now = optimizer.param_groups[0]["lr"]

    print(_rule())
    print(f"AlphaFold2-like run: {run_name}")
    print(
        f"Device: {device} | AMP: {amp_enabled}({amp_dtype}) | EMA: {ema_str} | "
        f"epochs: {epochs} | lr_now: {lr_now:.2e} | grad_clip: {grad_clip}")
    
    print(
        f"Monitor: {monitor_name} ({monitor_mode}) | "
        f"start_epoch: {start_epoch} | global_step: {global_step}")
    print(_rule())
    print(
        f"{'ep':>3} | {'step':>8} | {'loss':>10} | {'fape':>10} | {'dist':>10} | "
        f"{'plddt':>10} | {'tors':>10} | {'rmsd':>8} | {'tm':>8} | {'gdt':>8} | {'lr':>9} | {'time':>8}")
    print(_rule())


    # --------------------------------------------------
    # Train loop
    # --------------------------------------------------
    total_time = 0.0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_stats, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            scheduler=scheduler,
            ema=ema,
            ema_target=model,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            max_batches=max_batches,
            log_every=log_every,
            log_grad_norm=log_grad_norm,
            log_mem=log_mem,
            on_oom=on_oom,
            global_step=global_step,
            ideal_backbone_local=ideal_backbone_local)

        sec = time.time() - t0
        total_time += sec
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:3d} | {global_step:8d} | "
            f"{train_stats['loss']:10.5f} | {train_stats['fape_loss']:10.5f} | {train_stats['dist_loss']:10.5f} | "
            f"{train_stats['plddt_loss']:10.5f} | {train_stats['torsion_loss']:10.5f} | "
            f"{train_stats['rmsd']:8.3f} | {train_stats['tm_score']:8.3f} | {train_stats['gdt_ts']:8.3f} | "
            f"{lr_now:9.2e} | {_fmt_hms(sec):>8}")



        # Checkpointing
        current_metric = float(train_stats[monitor_name])

        best_metric, improved = maybe_save_best_and_last(
            save_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            epoch=epoch,
            global_step=global_step,
            current_metric=current_metric,
            best_metric=best_metric,
            metric_name=monitor_name,
            mode=monitor_mode,
            val_metrics=train_stats,   # aquí son train_stats; se puede pasar val_stats
            config=config,)

        if improved:
            print(f"└─ [BEST] improved {monitor_name} -> {best_metric:.6f}")

        # optional periodic named snapshots
        if (save_every is not None) and (save_every > 0) and ((epoch % save_every == 0) or (epoch == epochs - 1)):
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_e{epoch:03d}.pt")
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                monitor_name=monitor_name,
                metrics=train_stats,
                config=config,
                save_optimizer_state=True,
                save_rng_state=True)
            
            print(f"└─ [CKPT] saved → {ckpt_path}")

            if copy_fixed_to_drive and drive_ckpt_dir:
                _copy_ckpt_to_drive_fixed(
                    src_path=ckpt_path,
                    drive_dir=drive_ckpt_dir,
                    fixed_name=fixed_drive_name)


    # Final save_last
    if save_last:
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_last_manual.pt")
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            epoch=epochs - 1,
            global_step=global_step,
            best_metric=best_metric,
            monitor_name=monitor_name,
            metrics=train_stats,
            config=config,
            save_optimizer_state=True,
            save_rng_state=True,)
        print(f"└─ [CKPT] final saved → {ckpt_path}")

        if copy_fixed_to_drive and drive_ckpt_dir:
            _copy_ckpt_to_drive_fixed(
                src_path=ckpt_path,
                drive_dir=drive_ckpt_dir,
                fixed_name=fixed_drive_name)

    print(_rule())
    print(f"Entrenamiento finalizado en {_fmt_hms(total_time)}")
    print(_rule())

    return {
        "global_step": global_step,
        "best_metric": best_metric,
        "last_train_stats": train_stats}

