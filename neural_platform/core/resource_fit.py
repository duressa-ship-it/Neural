"""
NeuralForge — Resource Fit Pre-flight.

Estimates how much memory and disk a chosen model + dataset will consume,
then compares against what the user's machine actually has. The validator
calls this *before* downloading or loading anything, so an obviously
unfittable choice fails in milliseconds rather than after streaming
several GB to disk.

The estimates are intentionally conservative — when in doubt we round up
and warn rather than block. Real workloads vary wildly with batch size,
optimizer choice, gradient checkpointing, etc. so the goal here is "catch
the obvious cases" not "be a precise sizing tool."
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

@dataclass
class HostResources:
    """What the current machine has available."""
    cpu_count:    Optional[int] = None
    ram_total_b:  Optional[int] = None
    ram_free_b:   Optional[int] = None
    disk_free_b:  Optional[int] = None
    accelerator:  str = "cpu"            # "cuda" | "mps" | "cpu"
    gpu_name:     Optional[str] = None
    gpu_count:    int = 0
    vram_total_b: Optional[int] = None    # primary GPU
    vram_free_b:  Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitEstimate:
    """Predicted footprint for a particular run."""
    model_weight_b: int = 0      # disk + memory for params
    optimizer_b:    int = 0      # AdamW = 2× weights for moments
    gradients_b:    int = 0      # ~1× weights
    activations_b: int = 0       # rough; depends on batch size + arch
    dataset_disk_b: int = 0      # estimated download size
    runtime_total_b: int = 0     # what stays in (V)RAM during a step
    download_total_b: int = 0    # bytes that will land on disk

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitReport:
    purpose: str                       # "training" | "inference"
    host:    HostResources
    estimate: FitEstimate
    issues:  List[Dict[str, Any]] = field(default_factory=list)
    fits:    bool = True

    def add(self, severity: str, code: str, message: str, hint: Optional[str] = None):
        self.issues.append({"severity": severity, "code": code,
                            "message": message, "hint": hint})
        if severity == "error":
            self.fits = False

    def to_dict(self) -> dict:
        return {
            "purpose":  self.purpose,
            "fits":     self.fits,
            "host":     self.host.to_dict(),
            "estimate": self.estimate.to_dict(),
            "issues":   self.issues,
        }


# ---------------------------------------------------------------------------
# Host introspection
# ---------------------------------------------------------------------------

def snapshot_host() -> HostResources:
    """Best-effort host introspection. Never raises."""
    h = HostResources(cpu_count=os.cpu_count())
    # RAM
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        h.ram_total_b = int(vm.total)
        h.ram_free_b  = int(vm.available)
    except Exception:
        pass
    # Disk free in cwd
    try:
        h.disk_free_b = int(shutil.disk_usage(os.getcwd()).free)
    except Exception:
        pass
    # Accelerator
    try:
        import torch
        if torch.cuda.is_available():
            h.accelerator = "cuda"
            h.gpu_count = torch.cuda.device_count()
            if h.gpu_count:
                props = torch.cuda.get_device_properties(0)
                h.gpu_name = props.name
                h.vram_total_b = int(props.total_memory)
                # `mem_get_info` exists since PyTorch 1.11
                if hasattr(torch.cuda, "mem_get_info"):
                    free, total = torch.cuda.mem_get_info(0)
                    h.vram_free_b  = int(free)
                    h.vram_total_b = int(total)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            h.accelerator = "mps"
            h.gpu_name    = "Apple Silicon (MPS)"
            h.gpu_count   = 1
            # MPS shares system memory but has its own watermark ceiling
            # (PYTORCH_MPS_HIGH_WATERMARK_RATIO defaults to ~0.5×RAM on
            # smaller Macs, capped per device generation). Newer PyTorch
            # exposes `recommended_max_memory()` which gives us the actual
            # ceiling Metal will allocate up to before refusing.
            try:
                if hasattr(torch.mps, "recommended_max_memory"):
                    h.vram_total_b = int(torch.mps.recommended_max_memory())
                if hasattr(torch.mps, "current_allocated_memory") and h.vram_total_b:
                    used = int(torch.mps.current_allocated_memory())
                    h.vram_free_b = max(0, h.vram_total_b - used)
            except Exception:
                pass
            # Fallback: assume ~50% of RAM is the MPS budget. Better to be
            # conservative — this is the policy CLI/Builder will warn off of.
            if not h.vram_total_b and h.ram_total_b:
                h.vram_total_b = int(h.ram_total_b * 0.5)
                h.vram_free_b = h.vram_total_b
    except Exception:
        pass
    return h


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

# Rough per-parameter overhead (bytes) for a training step using AdamW.
# Weights (4) + grads (4) + Adam m (4) + Adam v (4) = 16 B for fp32.
_TRAIN_OVERHEAD_PER_PARAM_B = 16
# fp32 inference is just the weights themselves.
_INFERENCE_PER_PARAM_B      = 4


def estimate_model_footprint(parameters: Optional[int],
                              size_bytes: Optional[int],
                              purpose: str = "training",
                              optimizer: str = "adamw",
                              batch_size: int = 1,
                              sequence_length: Optional[int] = None,
                              hidden_size: Optional[int] = None,
                              num_layers: Optional[int] = None,
                              num_heads: Optional[int] = None,
                              dtype_bytes: int = 4,
                              quantization_bits: Optional[int] = None) -> FitEstimate:
    """Predict bytes for a particular model + purpose.

    `parameters` / `size_bytes` come from the model source's `ModelInfo`.
    The optional `batch_size`, `sequence_length`, and architecture hints
    let us produce a transformer-aware activation estimate — the previous
    flat ½×weights heuristic was wrong for the regimes that actually OOM
    (large model, modest batch but long sequence, attention-heavy).

    `dtype_bytes` defaults to fp32 (4). Pass 2 for fp16 / bf16.

    ``quantization_bits`` (one of 4 or 8) lets the caller request a
    quantized-weights estimate. The runtime weight size scales with the
    requested bits, **but activations stay at the original dtype** —
    bitsandbytes dequantizes on the fly during matmuls. This matches the
    actual VRAM curve users observe: a 7B model with load_in_4bit fits
    on an 8GB card mainly because the weight buffer shrunk, not because
    activations got smaller.
    """
    est = FitEstimate()
    if not parameters and size_bytes:
        parameters = max(1, size_bytes // 4)
    if parameters:
        # Quantized weights: bits/8 bytes per parameter for the weight
        # buffer; activations and gradients still use dtype_bytes (4
        # for fp32, 2 for fp16/bf16). Quantization is inference-only —
        # bitsandbytes doesn't support backprop through 4-bit weights
        # in any first-class way — so we force purpose='inference'
        # silently when quantization is on.
        if quantization_bits in (4, 8):
            est.model_weight_b = max(1, (parameters * quantization_bits) // 8)
            if purpose == "training":
                purpose = "inference"
        else:
            est.model_weight_b = parameters * dtype_bytes
        if purpose == "training":
            mult = {
                "adamw": _TRAIN_OVERHEAD_PER_PARAM_B,
                "adam":  _TRAIN_OVERHEAD_PER_PARAM_B,
                "sgd":    8,   # weights + grads
            }.get((optimizer or "adamw").lower(), _TRAIN_OVERHEAD_PER_PARAM_B)
            est.gradients_b   = parameters * 4
            # Optimizer state (m + v for Adam-family). 4 B each in fp32.
            est.optimizer_b   = max(0, parameters * (mult - 8))
            est.activations_b = _estimate_activations(
                parameters=parameters,
                weight_bytes=est.model_weight_b,
                batch_size=batch_size,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dtype_bytes=dtype_bytes,
                purpose=purpose,
            )
            est.runtime_total_b = (est.model_weight_b + est.gradients_b
                                    + est.optimizer_b + est.activations_b)
        else:
            # Inference: weights + a smaller activation budget (no grads, no optim).
            est.activations_b = _estimate_activations(
                parameters=parameters,
                weight_bytes=est.model_weight_b,
                batch_size=batch_size,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dtype_bytes=dtype_bytes,
                purpose=purpose,
            )
            est.runtime_total_b = est.model_weight_b + est.activations_b
    if size_bytes:
        est.download_total_b = int(size_bytes)
    return est


def _estimate_activations(parameters: int,
                          weight_bytes: int,
                          batch_size: int,
                          sequence_length: Optional[int],
                          hidden_size: Optional[int],
                          num_layers: Optional[int],
                          num_heads: Optional[int],
                          dtype_bytes: int,
                          purpose: str) -> int:
    """Activation memory estimate.

    For transformers we use:
        activations ≈ B × T × D × L × c  +  B × H × T² × dtype_bytes × L
                       ──────────────────     ──────────────────────────
                       hidden states           attention scores
    where c ≈ 8 (LayerNorm + MLP intermediates + residuals, very rough),
    L = num_layers, B = batch, T = sequence, D = hidden, H = num_heads.

    For training, multiply by 2 (forward + backward retains intermediates).

    When we don't have arch hints, fall back to a weight-relative heuristic
    that scales with batch×seq so big-context runs aren't underestimated.
    """
    # Rich path — we know the arch.
    if hidden_size and num_layers and sequence_length:
        B, T, D, L = batch_size, sequence_length, hidden_size, num_layers
        H = num_heads or max(1, D // 64)
        hidden = B * T * D * L * 8 * dtype_bytes
        attn   = B * H * T * T * dtype_bytes * L
        total  = hidden + attn
        if purpose == "training":
            total *= 2     # backward pass keeps activations
        return int(total)

    # Lean path — scale a fraction of the weights by batch×seq vs. an arbitrary
    # baseline of 1×128 to keep the order-of-magnitude reasonable.
    seq_factor   = (sequence_length or 128) / 128.0
    batch_factor = max(1, batch_size) / 1.0
    base_fraction = 0.5 if purpose == "training" else 0.25
    return int(weight_bytes * base_fraction * seq_factor * batch_factor)


def add_dataset_footprint(est: FitEstimate, dataset_size_bytes: Optional[int]) -> FitEstimate:
    if dataset_size_bytes:
        est.dataset_disk_b = int(dataset_size_bytes)
        est.download_total_b += int(dataset_size_bytes)
    return est


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def check_fit(est: FitEstimate,
              host: HostResources,
              purpose: str = "training",
              device: str = "auto") -> FitReport:
    """Compare an estimate against a host snapshot. Returns a FitReport
    where `fits=False` if any hard constraint is violated."""
    report = FitReport(purpose=purpose, host=host, estimate=est)

    # Disk
    if est.download_total_b and host.disk_free_b is not None:
        if est.download_total_b > host.disk_free_b:
            report.add(
                "error", "disk_too_small",
                f"Download needs ~{_h(est.download_total_b)} but only "
                f"{_h(host.disk_free_b)} is free in the working directory.",
                "Free disk, point HF_HOME / TRANSFORMERS_CACHE at a roomier "
                "volume, or pick a smaller model/dataset.",
            )
        elif est.download_total_b > 0.8 * host.disk_free_b:
            report.add(
                "warning", "disk_tight",
                f"Download will use >80% of free disk "
                f"({_h(est.download_total_b)} of {_h(host.disk_free_b)}).",
            )

    # Accelerator memory: CUDA and MPS are both gated against vram_total_b.
    # MPS shares system memory but the Metal driver enforces a watermark,
    # so the effective "VRAM" is much smaller than total RAM. Treating MPS
    # like CUDA here (instead of falling through to CPU/RAM) is what would
    # have caught the Qwen 1.47B-on-9GB-MPS OOM the user just hit.
    using_accelerator = (
        host.accelerator in ("cuda", "mps")
        and device in ("auto", "cuda", "mps", host.accelerator)
    )
    if using_accelerator and est.runtime_total_b and host.vram_total_b:
        accel_label = "GPU" if host.accelerator == "cuda" else "MPS"
        # Compare against total VRAM, not free VRAM — other processes may
        # come and go, so total is the policy-relevant ceiling.
        if est.runtime_total_b > host.vram_total_b:
            report.add(
                "error", "vram_too_small",
                f"{purpose} needs ~{_h(est.runtime_total_b)} of {accel_label} memory "
                f"but {host.gpu_name or accel_label} only has {_h(host.vram_total_b)}.",
                "Reduce batch_size, shorten sequence_length, freeze the backbone, "
                "enable mixed_precision, or pick a smaller model. On MPS you can "
                "raise the ceiling with PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 "
                "(may cause system instability).",
            )
        elif est.runtime_total_b > 0.85 * host.vram_total_b:
            report.add(
                "warning", "vram_tight",
                f"{purpose} estimate ({_h(est.runtime_total_b)}) is within 15% "
                f"of the {accel_label}'s {_h(host.vram_total_b)}.",
                "Consider lower batch_size or fp16 to leave headroom.",
            )

    # CPU fallback — compare against system RAM. (MPS is handled above; we
    # don't double-count it by also gating against RAM.)
    cpu_only = device == "cpu" or host.accelerator == "cpu"
    if cpu_only and est.runtime_total_b and host.ram_total_b:
        if est.runtime_total_b > host.ram_total_b:
            report.add(
                "error", "ram_too_small",
                f"{purpose} needs ~{_h(est.runtime_total_b)} of memory but the "
                f"system has {_h(host.ram_total_b)} of RAM.",
                "Pick a smaller model or run on a GPU box.",
            )
        elif est.runtime_total_b > 0.75 * host.ram_total_b:
            report.add(
                "warning", "ram_tight",
                f"{purpose} estimate ({_h(est.runtime_total_b)}) will use "
                f"≥75% of the system's {_h(host.ram_total_b)} RAM.",
            )

    return report


# ---------------------------------------------------------------------------
# Pretty bytes
# ---------------------------------------------------------------------------

def _h(n: int) -> str:
    if n is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.1f} {units[i]}"
