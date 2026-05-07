"""
NeuralForge — Pluggable Model Source Layer.

A *model source* is anywhere we can pull a pretrained model from: the
HuggingFace Hub, a local checkpoint directory, an ONNX model zoo, an
internal model registry, etc. Each source knows how to:

  * `search` — list candidate models matching a query / task filter.
  * `get_info` — return per-model metadata (pipeline_tag, parameter count,
                 declared modality, file size, supported tasks, license).
  * `inspect_compat` — given an *intended* task and (optionally) dataset
                       modality, return a `CompatReport` flagging mismatches
                       *before* any weights are downloaded.

This module replaces the previous "type 'openai/whisper-tiny' into a YAML
field and pray" flow. The Whisper-vs-IMDB error that motivated the rewrite
is now a hard error at config validation time.

Concrete sources:
  * `HFModelSource`      — HuggingFace Hub (this is the first implementation)
  * `LocalCheckpointSource` — directory of .pt files trained inside NeuralForge
  * (future) ONNX hub, internal registry, etc.

The sources self-register so the rest of the platform can iterate over
`registered_sources()` without hardcoding the list.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """Lightweight summary of a model in a source. The shape is uniform so
    different sources can populate it without UI churn."""
    id:            str                  # e.g. "openai/whisper-tiny" or "local:my_model"
    source:       str                   # e.g. "huggingface", "local"
    pipeline_tag: Optional[str] = None  # e.g. "automatic-speech-recognition"
    modality:     Optional[str] = None  # detected via tags
    library:      Optional[str] = None  # "transformers", "diffusers", etc.
    downloads:    int = 0
    likes:        int = 0
    tags:         List[str] = field(default_factory=list)
    description:  Optional[str] = None
    private:      bool = False
    gated:        bool = False
    last_modified: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelInfo(ModelCard):
    """Detailed metadata about a single model (more expensive to fetch)."""
    parameters:           Optional[int] = None       # total params (best-effort)
    size_bytes:           Optional[int] = None       # weight file size
    supported_tasks:      List[str] = field(default_factory=list)
    config:               Optional[dict] = None      # raw config.json (subset)
    architectures:        List[str] = field(default_factory=list)  # e.g. ["WhisperForConditionalGeneration"]
    license:              Optional[str] = None
    siblings:             List[str] = field(default_factory=list)  # weight files
    # How the model is packaged on disk — drives which loader path we take.
    # Values: "safetensors", "pytorch_bin", "sharded_safetensors",
    # "sharded_pytorch", "peft_adapter", "gguf", "onnx", "tf", "flax",
    # "diffusers", "unknown".
    loading_pattern:      str = "unknown"
    # When the pattern is "peft_adapter", the base model the adapter was
    # trained on top of (parsed from adapter_config.json). Required to
    # actually load the adapter.
    base_model:           Optional[str] = None
    # Whether the standard transformers AutoModel.from_pretrained path will
    # work without auxiliary libraries (peft / llama.cpp / etc.).
    standard_loadable:    bool = True


@dataclass
class CompatIssue:
    severity: str            # "error" | "warning" | "info"
    code:     str            # short stable id, e.g. "task_mismatch"
    message:  str
    hint:     Optional[str] = None


@dataclass
class CompatReport:
    """Result of `ModelSource.inspect_compat(...)`. The validator and the UI
    both consume this — the UI greys out the train button on errors and shows
    the issues inline."""
    model_id:           str
    source:             str
    intended_task:      Optional[str]
    detected_pipeline:  Optional[str]
    detected_modality:  Optional[str]
    fits_resources:     Optional[bool] = None    # set when resource_fit is computed
    issues:             List[CompatIssue] = field(default_factory=list)
    info:               Optional[ModelInfo] = None
    resource_fit:       Optional[dict] = None    # populated by resource_fit module

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def add_error(self, code: str, msg: str, hint: Optional[str] = None) -> None:
        self.issues.append(CompatIssue("error", code, msg, hint))

    def add_warning(self, code: str, msg: str, hint: Optional[str] = None) -> None:
        self.issues.append(CompatIssue("warning", code, msg, hint))

    def add_info(self, code: str, msg: str, hint: Optional[str] = None) -> None:
        self.issues.append(CompatIssue("info", code, msg, hint))

    def to_dict(self) -> dict:
        return {
            "model_id":          self.model_id,
            "source":            self.source,
            "intended_task":     self.intended_task,
            "detected_pipeline": self.detected_pipeline,
            "detected_modality": self.detected_modality,
            "fits_resources":    self.fits_resources,
            "ok":                self.ok,
            "issues":            [asdict(i) for i in self.issues],
            "info":              asdict(self.info) if self.info else None,
            "resource_fit":      self.resource_fit,
        }


# ---------------------------------------------------------------------------
# Source ABC + registry
# ---------------------------------------------------------------------------

class ModelSource(ABC):
    """Abstract base — implement this to add a new model source."""

    name: str = "base"

    @abstractmethod
    def search(self,
               query: Optional[str] = None,
               task: Optional[str] = None,
               modality: Optional[str] = None,
               sort: str = "downloads",
               limit: int = 24) -> List[ModelCard]:
        """List candidate models matching the filters."""

    @abstractmethod
    def get_info(self, model_id: str) -> ModelInfo:
        """Fetch detailed metadata for one model."""

    def inspect_compat(self,
                       model_id: str,
                       intended_task: Optional[str] = None,
                       dataset_modality: Optional[str] = None) -> CompatReport:
        """
        Default compatibility inspection. Subclasses override when they have
        richer signals (e.g. HF reads `pipeline_tag` directly).
        """
        info = self.get_info(model_id)
        report = CompatReport(
            model_id=model_id,
            source=self.name,
            intended_task=intended_task,
            detected_pipeline=info.pipeline_tag,
            detected_modality=info.modality,
            info=info,
        )
        _check_task_compat(report, intended_task, info)
        _check_modality_compat(report, dataset_modality, info)
        return report


_REGISTRY: Dict[str, ModelSource] = {}


def register_source(source: ModelSource) -> None:
    """Register a `ModelSource` instance under its `name`. Idempotent."""
    if not source.name:
        raise ValueError("ModelSource.name must be set on the subclass.")
    _REGISTRY[source.name] = source


def get_source(name: str) -> ModelSource:
    if name not in _REGISTRY:
        raise KeyError(
            f"No model source registered under '{name}'. "
            f"Registered: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def registered_sources() -> List[ModelSource]:
    return list(_REGISTRY.values())


# ---------------------------------------------------------------------------
# Compatibility checks shared across sources
# ---------------------------------------------------------------------------

# Maps an HF `pipeline_tag` to the broad modality that *consumes* its inputs.
# This is the source of truth for "is this model and that dataset talking
# about the same kind of data?"
_PIPELINE_MODALITY: Dict[str, str] = {
    # Text
    "text-classification":          "text",
    "token-classification":         "text",
    "question-answering":           "text",
    "summarization":                "text",
    "translation":                  "text",
    "text-generation":              "text",
    "fill-mask":                    "text",
    "sentence-similarity":          "text",
    "feature-extraction":           "text",
    "zero-shot-classification":     "text",
    "text-ranking":                 "text",
    "table-question-answering":     "text",
    # Vision
    "image-classification":         "image",
    "image-segmentation":           "image",
    "object-detection":             "image",
    "depth-estimation":             "image",
    "image-to-image":               "image",
    "image-to-text":                "image",
    "text-to-image":                "text",      # consumes text
    "unconditional-image-generation": "image",
    "zero-shot-image-classification": "image",
    "keypoint-detection":           "image",
    # Video
    "video-classification":         "video",
    "text-to-video":                "text",
    "video-to-video":               "video",
    "image-to-video":               "image",
    # Audio
    "audio-classification":         "audio",
    "automatic-speech-recognition": "audio",
    "text-to-speech":               "text",
    "text-to-audio":                "text",
    "audio-to-audio":               "audio",
    "voice-activity-detection":     "audio",
    # Tabular / TS
    "tabular-classification":       "tabular",
    "tabular-regression":           "tabular",
    "time-series-forecasting":      "time_series",
}


def pipeline_to_modality(pipeline_tag: Optional[str]) -> Optional[str]:
    """Translate an HF pipeline_tag to a modality string. None for unknowns."""
    if not pipeline_tag:
        return None
    return _PIPELINE_MODALITY.get(pipeline_tag.lower())


# ---------------------------------------------------------------------------
# Loading-pattern detection
# ---------------------------------------------------------------------------

# Recognized patterns, ordered most-specific-first. The first hit wins.
def detect_loading_pattern(siblings: List[str]) -> str:
    """Decide how to load a model based on the file list in its repo.

    HF model repos expose a *bunch* of different on-disk layouts:
      * `model.safetensors` / `pytorch_model.bin`           — the standard
        single-file weights — `AutoModel.from_pretrained` works directly.
      * `model.safetensors.index.json` + shards             — sharded weights,
        also handled by `AutoModel.from_pretrained` transparently.
      * `adapter_model.safetensors` + `adapter_config.json` — PEFT/LoRA
        adapters. The actual weights live in the BASE model named in
        `adapter_config.json["base_model_name_or_path"]`. Loading these
        without `peft` and the base model gives the cryptic "does not
        appear to have a file named pytorch_model.bin or model.safetensors"
        error.
      * `*.gguf`                                            — llama.cpp /
        text-generation-inference format. Not loadable via
        `transformers.AutoModel`; needs llama-cpp-python or vllm.
      * `*.onnx`                                            — ONNX export.
        Loadable via optimum.onnxruntime, not vanilla transformers.
      * Diffusion model repos — have `model_index.json`. Need `diffusers`.

    Detecting the pattern up front lets the inspector tell users *exactly*
    why a model won't load with the default path, instead of letting them
    download multi-GB files and crash in the loader.
    """
    if not siblings:
        return "unknown"
    files = {f for f in siblings if f}
    has = lambda needle: any(needle in f for f in files)
    has_exact = lambda name: name in files

    # PEFT / LoRA adapter — most common 'unloadable' pattern in 2025.
    if has_exact("adapter_config.json") or has("adapter_model"):
        return "peft_adapter"

    # GGUF (llama.cpp)
    if any(f.lower().endswith(".gguf") for f in files):
        return "gguf"

    # Diffusion repos (have model_index.json + nested unet/, vae/, etc.)
    if has_exact("model_index.json"):
        return "diffusers"

    # Sharded transformers
    if has_exact("model.safetensors.index.json"):
        return "sharded_safetensors"
    if has_exact("pytorch_model.bin.index.json"):
        return "sharded_pytorch"

    # Single-file transformers
    if has_exact("model.safetensors"):
        return "safetensors"
    if has_exact("pytorch_model.bin"):
        return "pytorch_bin"

    # ONNX — exported variants
    if any(f.lower().endswith(".onnx") for f in files):
        return "onnx"

    # TF / Flax (rare but real)
    if has_exact("tf_model.h5"):
        return "tf"
    if has_exact("flax_model.msgpack"):
        return "flax"

    return "unknown"


# Patterns the standard `transformers.AutoModel.from_pretrained` path can
# load on its own. Anything outside this set needs an auxiliary library
# (peft, diffusers, llama.cpp, optimum, etc.) and probably also a base
# model identifier.
_STANDARD_LOADABLE_PATTERNS = {
    "safetensors",
    "pytorch_bin",
    "sharded_safetensors",
    "sharded_pytorch",
}


def is_standard_loadable(pattern: str) -> bool:
    return pattern in _STANDARD_LOADABLE_PATTERNS


def _check_task_compat(report: CompatReport,
                       intended_task: Optional[str],
                       info: ModelInfo) -> None:
    """Compare intended task against what the model declares it does.

    The most common bug this catches: user grabs an audio model
    (`pipeline_tag=automatic-speech-recognition`) and tries to use it for
    text classification on IMDB. The HF Auto-class loader fails 30s into
    training; we'd rather fail in 30ms here.
    """
    if not intended_task:
        return
    intended = intended_task.lower()

    # Rich comparison: model has explicit pipeline_tag.
    if info.pipeline_tag:
        declared = info.pipeline_tag.lower()
        if declared == intended:
            return

        intended_modality = pipeline_to_modality(intended)
        declared_modality = pipeline_to_modality(declared)

        if intended_modality and declared_modality and intended_modality != declared_modality:
            report.add_error(
                "task_modality_mismatch",
                f"Model '{report.model_id}' is built for '{declared}' "
                f"({declared_modality}); you asked for '{intended}' ({intended_modality}). "
                f"The model expects {declared_modality} inputs and won't accept "
                f"{intended_modality} batches.",
                f"Pick a model whose pipeline_tag is '{intended}' — search with "
                f"`source=huggingface, task={intended}` — or change "
                f"`training.pipeline_task` to '{declared}'.",
            )
            return

        # Same modality, different task — usually still trainable (e.g. swap
        # text-classification for token-classification on the same encoder).
        report.add_warning(
            "task_mismatch_same_modality",
            f"Model '{report.model_id}' declares '{declared}'; you asked for '{intended}'. "
            "Same modality, but the head will be reinitialized.",
            "If you intend to fine-tune, this is fine. "
            "Otherwise pick a model whose pipeline_tag matches your task.",
        )
        return

    # No pipeline_tag declared — fall back to architectures.
    if info.architectures and intended in _PIPELINE_MODALITY:
        # Try to see if any architecture name hints at the intended modality.
        arch_text = " ".join(info.architectures).lower()
        if intended_modality := pipeline_to_modality(intended):
            audio_arch = any(t in arch_text for t in ("whisper", "wav2vec", "hubert", "audio"))
            text_arch  = any(t in arch_text for t in ("bert", "roberta", "gpt", "llama", "t5", "bart"))
            image_arch = any(t in arch_text for t in ("vit", "resnet", "convnext", "swin", "detr", "segformer"))
            if intended_modality == "text" and audio_arch and not text_arch:
                report.add_error(
                    "task_modality_mismatch",
                    f"Model architectures {info.architectures} look like an audio "
                    f"backbone; you asked for '{intended}' (text task).",
                    "Pick a text encoder (bert, distilbert, roberta, etc.) instead.",
                )
            elif intended_modality == "audio" and text_arch and not audio_arch:
                report.add_error(
                    "task_modality_mismatch",
                    f"Model architectures {info.architectures} look like a text "
                    f"backbone; you asked for '{intended}' (audio task).",
                    "Pick an audio encoder (wav2vec2, hubert, whisper, etc.) instead.",
                )
            elif intended_modality == "image" and (audio_arch or text_arch) and not image_arch:
                report.add_warning(
                    "arch_modality_unclear",
                    f"Model architectures {info.architectures} don't look image-like.",
                    "Verify this model can accept image inputs.",
                )


def _check_modality_compat(report: CompatReport,
                           dataset_modality: Optional[str],
                           info: ModelInfo) -> None:
    """If we know the dataset's modality, make sure the model can consume it."""
    if not dataset_modality:
        return
    model_modality = info.modality or pipeline_to_modality(info.pipeline_tag or "")
    if not model_modality:
        return
    if model_modality != dataset_modality:
        report.add_error(
            "data_model_modality_mismatch",
            f"Dataset is {dataset_modality}; model handles {model_modality}.",
            f"Pick a {dataset_modality} model from {report.source}.",
        )


def _check_loading_pattern(report: CompatReport, info: ModelInfo) -> None:
    """Flag models that won't load via the standard path.

    The 'felixwangg/Qwen2.5-Coder-...' case: only `adapter_model.safetensors`
    is in the repo, so `transformers.AutoModel.from_pretrained` errors out
    with "does not appear to have a file named pytorch_model.bin or
    model.safetensors". By detecting the file pattern up front we tell the
    user *what kind of model this is* and *what they need to load it*,
    instead of letting them download multi-GB checkpoints to fail.
    """
    pattern = info.loading_pattern or "unknown"
    if is_standard_loadable(pattern):
        return

    if pattern == "peft_adapter":
        base = info.base_model or "(unknown — adapter_config.json was not readable)"
        report.add_error(
            "peft_adapter_required",
            f"Model '{report.model_id}' is a PEFT/LoRA adapter, not a full model. "
            f"It only ships `adapter_*` weights and depends on a base model "
            f"({base}).",
            "Install `peft` (`pip install peft`) and either: "
            "(1) load this adapter on top of its base model — see "
            "https://huggingface.co/docs/peft — or "
            "(2) point `model.hf_pipeline.pretrained` at the base model directly "
            "if you don't need the adapter weights.",
        )
        return
    if pattern == "gguf":
        report.add_error(
            "gguf_unsupported",
            f"Model '{report.model_id}' is a GGUF (llama.cpp) file. NeuralForge "
            "trains via PyTorch + transformers, which don't load GGUF.",
            "Use the original PyTorch checkpoint (typically a sibling repo with "
            "the same model id minus a `-GGUF` suffix), or switch to a runtime "
            "that supports GGUF (llama-cpp-python, vllm) for inference only.",
        )
        return
    if pattern == "diffusers":
        report.add_error(
            "diffusers_required",
            f"Model '{report.model_id}' is a diffusion pipeline (has "
            "`model_index.json`). It loads via `diffusers`, not `transformers`.",
            "Install `diffusers` (`pip install diffusers`) and load with "
            "`DiffusionPipeline.from_pretrained`. NeuralForge doesn't drive a "
            "diffusion training loop today — track the request in DESIGN.md.",
        )
        return
    if pattern == "onnx":
        report.add_warning(
            "onnx_format",
            f"Model '{report.model_id}' is an ONNX export. The standard loader "
            "won't pick this up.",
            "Install `optimum` (`pip install optimum[onnxruntime]`) and use "
            "`ORTModelForXxx.from_pretrained` for inference, or fetch the "
            "original PyTorch checkpoint for training.",
        )
        return
    if pattern in ("tf", "flax"):
        report.add_warning(
            "non_pytorch_weights",
            f"Model '{report.model_id}' ships {pattern.upper()} weights. "
            "AutoModel.from_pretrained will try to convert on the fly, which "
            "occasionally fails for custom heads.",
            "If loading errors, look for a PyTorch (`pytorch_model.bin` or "
            "`model.safetensors`) variant of this model.",
        )
        return
    if pattern == "unknown":
        # Only warn — there are legitimate edge cases (custom format,
        # private repo with auth quirks, etc.) and we don't want to be
        # noisy.
        report.add_warning(
            "unknown_loading_pattern",
            f"Couldn't detect a recognized weight format in '{report.model_id}'. "
            f"Files seen: {info.siblings[:6]}{'…' if len(info.siblings) > 6 else ''}.",
            "If loading fails, check the repo's README for the right loader.",
        )


# ---------------------------------------------------------------------------
# HuggingFace source
# ---------------------------------------------------------------------------

_HF_API_BASE = "https://huggingface.co/api"
_HF_TIMEOUT_S = 10.0

# HuggingFace model id format: `<owner>/<repo>` or `<canonical>` (single
# segment for legacy models like `bert-base-uncased`). Owner/repo each are
# 1–96 chars from [a-zA-Z0-9_.-]. We also reject leading dashes/dots and
# any path traversal characters. This is what the HF Hub itself enforces;
# matching client-side prevents `\`, `..`, `?token=...` etc. from being
# turned into open-ended HTTP redirects.
_HF_MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}"           # canonical (single-segment)
    r"(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$"      # optional /repo
)


class InvalidModelIdError(ValueError):
    """Raised when a string clearly isn't a HuggingFace model id.

    Distinct from a runtime 404: this fires before any HTTP request, so
    callers can short-circuit without exposing the API to drive-by inputs
    like `\\`, `..`, `?token=…`, or query-string injection.
    """


def validate_hf_model_id(model_id: Optional[str]) -> str:
    """Validate the shape of an HF model id and return it cleaned.

    Raises `InvalidModelIdError` for empty input, whitespace-only, control
    chars, path traversal, schemes (`http://`), query strings, or anything
    that doesn't match the documented `<owner>/<repo>` shape.
    """
    if not model_id or not str(model_id).strip():
        raise InvalidModelIdError("Model id is empty.")
    s = str(model_id).strip()
    # Strip a leading https://huggingface.co/ if present so the user can
    # paste a Hub URL directly. Anything else with a scheme is rejected.
    for prefix in ("https://huggingface.co/", "http://huggingface.co/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if "://" in s:
        raise InvalidModelIdError(
            f"Model id '{model_id}' looks like a URL. Use the "
            "`<owner>/<repo>` form (e.g. `openai/whisper-tiny`)."
        )
    if any(ch in s for ch in "\t\n\r"):
        raise InvalidModelIdError("Model id contains whitespace / control characters.")
    if "?" in s or "#" in s or "&" in s:
        raise InvalidModelIdError("Model id must not include query strings or fragments.")
    if ".." in s or s.startswith(("/", "\\", ".", "-")) or s.endswith(("/", "\\")):
        raise InvalidModelIdError(
            f"'{model_id}' isn't a valid HuggingFace model id (path traversal "
            "or stray separator)."
        )
    if not _HF_MODEL_ID_RE.match(s):
        raise InvalidModelIdError(
            f"'{model_id}' isn't a valid HuggingFace model id. Expected "
            "`<owner>/<repo>` with [A-Za-z0-9._-] characters."
        )
    return s


class HFModelSource(ModelSource):
    """HuggingFace Hub-backed model source.

    All network calls are guarded with short timeouts and any HTTP error
    surfaces to the caller — the validator wraps this and degrades to a
    "couldn't reach the Hub" warning rather than a hard error so users
    aren't blocked offline.
    """

    name = "huggingface"

    def __init__(self, token: Optional[str] = None,
                 cache_ttl_s: int = 600) -> None:
        # Token resolution lives in `core.hf_auth` so we honor every
        # location HF supports (env vars + ~/.cache/huggingface/token) and
        # so we get the redaction helpers for any error path.
        if token is not None:
            self._explicit_token: Optional[str] = token
        else:
            self._explicit_token = None
        # Tiny in-process cache so the inspector + validator + UI don't
        # hammer HF when the user is editing a config.
        self._cache: Dict[str, tuple] = {}   # key → (timestamp, value)
        self._cache_ttl = cache_ttl_s

    @property
    def token(self) -> Optional[str]:
        """Resolve the token at *call time* — the user may run
        `huggingface-cli login` between an inspector run and a download.
        Never logged; never returned from any public API."""
        if self._explicit_token is not None:
            return self._explicit_token
        try:
            from neural_platform.core.hf_auth import get_token
            return get_token()
        except Exception:
            return None

    # ------------------------------------------------------------------
    def search(self,
               query: Optional[str] = None,
               task: Optional[str] = None,
               modality: Optional[str] = None,
               sort: str = "downloads",
               limit: int = 24) -> List[ModelCard]:
        params: Dict[str, Any] = {
            "limit": min(max(int(limit), 1), 100),
            "sort":  sort,
            "full":  "true",     # gives us pipeline_tag + modelId in the listing
        }
        if query:
            params["search"] = query
        # HF's `pipeline_tag` filter expects the literal pipeline tag string.
        if task:
            params["pipeline_tag"] = task
        # `filter` accepts comma-separated tag filters. Modality is the only
        # one we wire today; users can append more via free-text query.
        filters: List[str] = []
        if modality:
            mod = modality.strip().lower()
            if mod == "time_series":
                mod = "time-series"
            filters.append(f"modality:{mod}")
        if filters:
            params["filter"] = ",".join(filters)

        rows = self._get_json(f"{_HF_API_BASE}/models", params=params)
        if not isinstance(rows, list):
            return []
        return [self._row_to_card(r) for r in rows if isinstance(r, dict)]

    # ------------------------------------------------------------------
    def get_info(self, model_id: str) -> ModelInfo:
        # Validate the id shape BEFORE issuing the HTTP request — `\`,
        # `..`, schemes, and query strings would otherwise hit the Hub and
        # come back as 302s (which httpx turns into surprising errors) or
        # worse, leak control characters into the URL path.
        clean_id = validate_hf_model_id(model_id)
        cache_key = f"info:{clean_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        row = self._get_json(f"{_HF_API_BASE}/models/{clean_id}")
        if not isinstance(row, dict):
            raise RuntimeError(f"HF returned no info for '{model_id}'.")

        card = self._row_to_card(row)
        siblings = [s.get("rfilename") for s in (row.get("siblings") or []) if isinstance(s, dict)]
        siblings = [s for s in siblings if s]

        # Total parameter count — surfaced under either `safetensors.total` or
        # the legacy `params` field on some older models.
        params = None
        st = row.get("safetensors") or {}
        if isinstance(st, dict):
            params = st.get("total")
        if not params:
            params = row.get("parameters")

        # Total weight size — sum of safetensors shards if available.
        size_bytes = None
        if isinstance(st, dict) and isinstance(st.get("parameters"), dict):
            # When safetensors metadata is exposed, each dtype maps to a count.
            # Mass over fp32 (4B) is a defensible upper bound.
            try:
                total_params = sum(int(v) for v in st["parameters"].values())
                size_bytes = total_params * 4
            except (TypeError, ValueError):
                pass

        config_block = row.get("config") or {}
        architectures: List[str] = []
        if isinstance(config_block, dict):
            archs = config_block.get("architectures")
            if isinstance(archs, list):
                architectures = [str(a) for a in archs if a]

        # Some models declare *which* tasks they're useful for via tags
        # like `task:text-classification`.
        supported_tasks: List[str] = []
        for tag in card.tags:
            if isinstance(tag, str) and tag.startswith("task:"):
                supported_tasks.append(tag.split(":", 1)[1])
        if card.pipeline_tag and card.pipeline_tag not in supported_tasks:
            supported_tasks.insert(0, card.pipeline_tag)

        license_field = None
        for tag in card.tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                license_field = tag.split(":", 1)[1]
                break

        loading_pattern = detect_loading_pattern(siblings)
        base_model = None
        if loading_pattern == "peft_adapter":
            # Pull the base model id out of adapter_config.json. We try
            # the raw file fetch — if it fails we just leave base_model
            # unset; the inspector will still flag the adapter and the
            # user can fix it.
            base_model = self._fetch_adapter_base_model(model_id) or \
                          (config_block.get("base_model_name_or_path")
                           if isinstance(config_block, dict) else None)

        info = ModelInfo(
            id=card.id,
            source=self.name,
            pipeline_tag=card.pipeline_tag,
            modality=card.modality,
            library=card.library,
            downloads=card.downloads,
            likes=card.likes,
            tags=card.tags,
            description=card.description,
            private=card.private,
            gated=card.gated,
            last_modified=card.last_modified,
            parameters=int(params) if params else None,
            size_bytes=int(size_bytes) if size_bytes else None,
            supported_tasks=supported_tasks,
            config=config_block if isinstance(config_block, dict) else None,
            architectures=architectures,
            license=license_field,
            siblings=siblings,
            loading_pattern=loading_pattern,
            base_model=base_model,
            standard_loadable=is_standard_loadable(loading_pattern),
        )
        self._cache_put(cache_key, info)
        return info

    def _fetch_adapter_base_model(self, model_id: str) -> Optional[str]:
        """Read `adapter_config.json` from a PEFT repo to discover the base
        model. Returns None on any error so callers can degrade gracefully.

        The file is small (<1 KB) and fetching it costs less than a single
        round-trip, which is much cheaper than downloading the full model
        only to fail at load time.
        """
        try:
            import httpx
            url = f"https://huggingface.co/{model_id}/raw/main/adapter_config.json"
            with httpx.Client(timeout=_HF_TIMEOUT_S, headers=self._headers()) as client:
                r = client.get(url)
                if r.status_code != 200:
                    return None
                payload = r.json()
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload.get("base_model_name_or_path") or payload.get("base_model")

    def inspect_compat(self,
                       model_id: str,
                       intended_task: Optional[str] = None,
                       dataset_modality: Optional[str] = None) -> CompatReport:
        # Reject obviously-bad model ids up front. Without this an input
        # like `\` is passed straight to HF, which 302-redirects to the
        # general /api/models listing — the response has no shape we
        # recognize, so the previous code marked the model "compatible by
        # default" with a soft warning. Now we hard-block.
        try:
            clean_id = validate_hf_model_id(model_id)
        except InvalidModelIdError as exc:
            r = CompatReport(
                model_id=model_id or "",
                source=self.name,
                intended_task=intended_task,
                detected_pipeline=None,
                detected_modality=None,
            )
            r.add_error("invalid_model_id", str(exc),
                         "Use the `<owner>/<repo>` form, e.g. `openai/whisper-tiny`.")
            return r
        try:
            info = self.get_info(clean_id)
        except InvalidModelIdError as exc:
            r = CompatReport(
                model_id=model_id, source=self.name,
                intended_task=intended_task,
                detected_pipeline=None, detected_modality=None,
            )
            r.add_error("invalid_model_id", str(exc), None)
            return r
        except Exception as exc:
            r = CompatReport(
                model_id=clean_id,
                source=self.name,
                intended_task=intended_task,
                detected_pipeline=None,
                detected_modality=None,
            )
            r.add_warning(
                "info_unreachable",
                f"Couldn't fetch HF Hub metadata for '{clean_id}': {exc}",
                "Network/auth issue — training will proceed without the up-front "
                "compatibility check, and may surface a runtime error instead.",
            )
            return r

        report = CompatReport(
            model_id=model_id,
            source=self.name,
            intended_task=intended_task,
            detected_pipeline=info.pipeline_tag,
            detected_modality=info.modality,
            info=info,
        )
        # Auth gating: if the model is gated/private, we need a valid token
        # AND that token's user must have been granted access. Surface the
        # most useful error we can without ever leaking the token itself.
        if info.gated or info.private:
            try:
                from neural_platform.core.hf_auth import is_authenticated
            except Exception:
                _has_token = False
            else:
                _has_token = is_authenticated()
            kind = "gated" if info.gated else "private"
            if not _has_token:
                report.add_error(
                    "auth_required",
                    f"'{model_id}' is {kind} and no HF token is configured.",
                    "Run `huggingface-cli login`, or set HF_TOKEN in your environment "
                    f"AND request access at https://huggingface.co/{model_id} "
                    "before training.",
                )
            else:
                # Token present, but we still don't *know* it has access until
                # the actual download tries. Warn so the UI can surface it.
                report.add_warning(
                    f"{kind}_model",
                    f"'{model_id}' is {kind}. Your HF token is configured but "
                    "access may still be denied if you haven't been granted it.",
                    f"Verify access at https://huggingface.co/{model_id}.",
                )

        _check_task_compat(report, intended_task, info)
        _check_modality_compat(report, dataset_modality, info)
        _check_loading_pattern(report, info)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _row_to_card(self, row: dict) -> ModelCard:
        tags = list(row.get("tags") or [])
        modality = None
        for t in tags:
            tl = (t or "").lower()
            if tl.startswith("modality:"):
                modality = tl.split(":", 1)[1].replace("-", "_")
                break
        if not modality:
            # Infer from pipeline_tag.
            modality = pipeline_to_modality(row.get("pipeline_tag")) or None
        desc = (row.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 240:
            desc = desc[:237] + "…"
        return ModelCard(
            id=row.get("id") or row.get("modelId") or "?",
            source=self.name,
            pipeline_tag=row.get("pipeline_tag"),
            modality=modality,
            library=row.get("library_name"),
            downloads=row.get("downloads") or 0,
            likes=row.get("likes") or 0,
            tags=tags,
            description=desc or None,
            private=bool(row.get("private")),
            gated=bool(row.get("gated")),
            last_modified=row.get("lastModified") or row.get("last_modified"),
        )

    def _headers(self) -> Dict[str, str]:
        h = {"User-Agent": "neuralforge/0.3"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get_json(self, url: str, params: Optional[dict] = None,
                  *, max_redirects: int = 3):
        """GET a JSON endpoint on the Hub with controlled redirect handling.

        We can't blindly `follow_redirects=True` — the Hub returns 302s for
        malformed ids that point at `/api/models` (a successful listing
        with no error fields), which would silently make the inspector
        think any id is "compatible". But we can't blindly *reject* every
        3xx either: canonical short names like
        ``distilbert-base-uncased-finetuned-sst-2-english`` legitimately
        redirect (HTTP 307) to the same path on the same host as part of
        HF's URL canonicalization.

        Compromise:
          * Stay on `follow_redirects=False`.
          * If the response is 3xx, inspect the ``Location`` header.
              - Same scheme + host (i.e. still ``huggingface.co``) → follow,
                up to ``max_redirects`` hops.
              - Cross-origin or scheme change → reject as before. That's the
                actual attack surface — open-ended redirects to a search
                page or a third-party host.
          * 4xx/5xx still raise via ``raise_for_status``.
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "HFModelSource requires `httpx`. Install with: pip install httpx"
            ) from exc

        from urllib.parse import urlsplit, urljoin
        with httpx.Client(timeout=_HF_TIMEOUT_S, headers=self._headers(),
                           follow_redirects=False) as client:
            current_url = url
            current_params = params
            hops = 0
            while True:
                r = client.get(current_url, params=current_params)
                if not (300 <= r.status_code < 400):
                    r.raise_for_status()
                    return r.json()
                # Got a redirect — decide whether it's safe to follow.
                location = r.headers.get("Location") or r.headers.get("location")
                if not location:
                    raise RuntimeError(
                        f"HF Hub returned HTTP {r.status_code} for {current_url} "
                        "without a Location header — refusing to chase blind."
                    )
                # Resolve relative redirects against the current URL.
                next_url = urljoin(current_url, location)
                src = urlsplit(current_url)
                dst = urlsplit(next_url)
                same_origin = (dst.scheme == src.scheme and
                               (dst.hostname or "").lower() ==
                               (src.hostname or "").lower())
                if not same_origin:
                    # Cross-origin redirect = the attack surface this guard
                    # was added for. Reject with the same message users
                    # already learned to recognize.
                    raise RuntimeError(
                        f"HF Hub redirected for {url} (HTTP {r.status_code}) to a "
                        f"different origin ({dst.scheme}://{dst.hostname}). "
                        "This usually means the model id is malformed."
                    )
                hops += 1
                if hops > max_redirects:
                    raise RuntimeError(
                        f"HF Hub redirect loop for {url} (>{max_redirects} hops)."
                    )
                # Same-host redirect — follow. Drop the original query
                # params after the first hop because the new URL already
                # bakes them in (HF canonicalization includes the path).
                current_url = next_url
                current_params = None

    def _cache_get(self, key: str):
        hit = self._cache.get(key)
        if not hit:
            return None
        ts, val = hit
        if time.time() - ts > self._cache_ttl:
            self._cache.pop(key, None)
            return None
        return val

    def _cache_put(self, key: str, value) -> None:
        self._cache[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# Local checkpoint source
# ---------------------------------------------------------------------------

class LocalCheckpointSource(ModelSource):
    """Discovers .pt checkpoints under one or more search roots.

    The training pipeline stores checkpoints with a serialized `model_config`
    dict; we read that to surface the model_type, parameter count, and the
    task the model was trained for. This unlocks the "browse + reuse" flow
    where a user can pick an old checkpoint as the starting point for a new
    fine-tune.
    """

    name = "local"

    def __init__(self, roots: Optional[List[Path]] = None) -> None:
        self.roots = [Path(r) for r in (roots or [Path("runs")])]

    def search(self,
               query: Optional[str] = None,
               task: Optional[str] = None,
               modality: Optional[str] = None,
               sort: str = "modified",
               limit: int = 24) -> List[ModelCard]:
        cards: List[ModelCard] = []
        for root in self.roots:
            if not root.exists():
                continue
            for pt_path in root.rglob("*.pt"):
                try:
                    card = self._card_from_checkpoint(pt_path)
                except Exception:
                    continue
                if query and query.lower() not in card.id.lower():
                    continue
                if task and (card.pipeline_tag or "").lower() != task.lower():
                    continue
                if modality and (card.modality or "").lower() != modality.lower():
                    continue
                cards.append(card)
        if sort == "modified":
            cards.sort(key=lambda c: c.last_modified or "", reverse=True)
        return cards[:limit]

    def get_info(self, model_id: str) -> ModelInfo:
        path = Path(model_id)
        if not path.exists():
            raise FileNotFoundError(f"Local checkpoint not found: {model_id}")
        card = self._card_from_checkpoint(path)
        return ModelInfo(
            id=card.id,
            source=self.name,
            pipeline_tag=card.pipeline_tag,
            modality=card.modality,
            library=card.library,
            tags=card.tags,
            description=card.description,
            last_modified=card.last_modified,
            parameters=None,
            size_bytes=path.stat().st_size,
            supported_tasks=[card.pipeline_tag] if card.pipeline_tag else [],
            architectures=[],
        )

    def _card_from_checkpoint(self, path: Path) -> ModelCard:
        # Don't trigger heavy torch.load for a listing — pull the JSON header
        # if present (PyTorch's `torch.save` writes a zip with `data.pkl`),
        # otherwise fall back to filename-based heuristics. Reading the full
        # pickle is too expensive for a search and risks executing arbitrary
        # objects.
        modified = time.strftime("%Y-%m-%dT%H:%M:%S",
                                  time.gmtime(path.stat().st_mtime))
        return ModelCard(
            id=str(path),
            source=self.name,
            pipeline_tag=None,
            modality=None,
            library="neuralforge",
            description=f"Local checkpoint at {path}",
            last_modified=modified,
        )


# ---------------------------------------------------------------------------
# Default-source bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_default_sources() -> None:
    if "huggingface" not in _REGISTRY:
        register_source(HFModelSource())
    if "local" not in _REGISTRY:
        register_source(LocalCheckpointSource())


_bootstrap_default_sources()
