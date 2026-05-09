"""
NeuralForge — HFPipelineModel

A universal wrapper around any HuggingFace pretrained model. Pairs a
`pipeline_task` (e.g. ``audio-classification``) with a `pretrained` model id
(e.g. ``facebook/wav2vec2-base``) and dispatches to the right
``transformers.Auto*`` class so any HF-supported task becomes trainable
through NeuralForge without hand-coding an architecture per task.

What this unlocks:
  * **Audio classification / ASR** — wav2vec2, hubert, whisper, etc.
  * **Image classification / segmentation / detection** — ViT, DETR, SegFormer.
  * **Text classification / token-classification / QA / summarization /
    translation / generation** — BERT, RoBERTa, T5, BART, GPT-2, …
  * **Multi-modal** — image-text-to-text via BLIP/LLaVA-style models, ASR,
    visual-question-answering.

What this *doesn't* do (yet):
  * Custom multi-tower architectures beyond what HF ships.
  * Reinforcement learning or robotics.

The wrapper accepts the same batch shape the corresponding HF Auto-class
expects — so `audio-classification` gets `(B, samples)`, `image-classification`
gets `(B, C, H, W)`, etc. NeuralForge's data loaders already produce these
shapes for the matching modality.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig
from neural_platform.core.registry import registry, MODEL
from neural_platform.models.base import BaseModel


def _resolve_auto_class_name(task: Optional[str]) -> str:
    """Pick a `transformers.Auto*` class **name** for the given task.

    Returns the *preferred* class name from the spec's fallback chain —
    not the actual class object, and not necessarily one that exists in
    the installed transformers. For runtime resolution use
    :func:`_resolve_auto_class` below, which probes the chain.
    """
    from neural_platform.core.pipeline_specs import resolve
    return resolve(task).auto_class


def _resolve_auto_class(transformers_module, task: Optional[str]):
    """Probe transformers for the first Auto* class in the task's spec
    chain that actually exists in the installed library.

    Returns ``(class, name)``. Raises a RuntimeError listing the chain
    when nothing resolves — that message replaces the older
    ``module transformers has no attribute AutoModelForVision2Seq``
    message and is what the user sees when they install a transformers
    version that's drifted past our spec.
    """
    from neural_platform.core.pipeline_specs import resolve, resolve_auto_class
    spec = resolve(task)
    return resolve_auto_class(transformers_module, spec)


# Backward-compat: a few external callers / tests reference this name. We
# build it lazily from the spec table so it stays in sync.
def _build_legacy_auto_class_map() -> Dict[str, str]:
    from neural_platform.core.pipeline_specs import PIPELINE_SPECS
    return {task: spec.auto_class for task, spec in PIPELINE_SPECS.items()}


_TASK_TO_AUTO_CLASS = _build_legacy_auto_class_map()


@registry.register(MODEL, "hf_pipeline")
class HFPipelineModel(BaseModel):
    """
    Loads a HuggingFace pretrained model based on a configured task.

    The forward pass is a passthrough — `inputs` is whatever the underlying
    HF model expects (a 1D waveform tensor for audio classification, a 4D
    image tensor for image classification, a tokenized dict for text, etc.).
    The loss / metrics are then computed downstream by the framework
    adapter using the model's logits output.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        arch = config.hf_pipeline
        # Pull the pipeline_task off the training config when available;
        # we don't have the full ExperimentConfig here, only the model
        # half, so callers may attach `_resolved_task` for us via the
        # adapter (see PyTorchAdapter.build_model).
        self._task: Optional[str] = getattr(config, "_resolved_task", None)
        # Lazy-cached set of forward() parameter names (filled on first forward
        # call). Initializing here avoids the AttributeError that bit us when
        # `_filter_kwargs` referenced this before it existed.
        self._fwd_param_names: Optional[set] = None

        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "model.type='hf_pipeline' requires the `transformers` package.\n"
                "Install with: pip install transformers"
            ) from exc

        # Probe the spec's fallback chain — transformers v4 → v5 renames
        # mean a single class name (e.g. AutoModelForVision2Seq) won't
        # always exist. resolve_auto_class returns the first Auto* class
        # that the installed transformers actually exposes; if none do, it
        # raises a clear error listing every name we tried.
        try:
            auto_cls, auto_name = _resolve_auto_class(transformers, self._task)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not resolve an Auto* class for task '{self._task}': {exc}"
            ) from exc

        # `output_size` only makes sense for classification heads. For
        # generative / non-classification tasks, passing `num_labels` either
        # gets silently dropped or — worse — overrides the model's vocab head.
        # `_resolve_auto_class_name` returns one of these classification
        # AutoModel*ForSequenceClassification / *ForImageClassification etc.
        # variants when the task is a classification task; in any other case,
        # `num_labels` is meaningless and we drop it.
        load_kwargs: Dict[str, Any] = {}
        if arch.revision:           load_kwargs["revision"]          = arch.revision
        if arch.trust_remote_code:  load_kwargs["trust_remote_code"] = True
        if arch.output_size and arch.output_size > 0 and "Classification" in auto_name:
            load_kwargs["num_labels"] = arch.output_size
        # Pass the discovered HF token through `transformers` so gated repos
        # work without the user having to plumb auth themselves. We resolve
        # it from `core.hf_auth` (env or huggingface-cli cache) and only at
        # call time — the token never lives on the model instance.
        try:
            from neural_platform.core.hf_auth import get_token as _hf_token
            _t = _hf_token()
            if _t:
                load_kwargs["token"] = _t
        except Exception:
            pass

        # Pre-flight: detect the loading pattern (PEFT / GGUF / diffusers /
        # standard transformers) BEFORE we try to download the weights. This
        # turns the cryptic "does not appear to have a file named
        # pytorch_model.bin or model.safetensors" into a clear "this is a
        # PEFT adapter, here's how to load it" error.
        loading_pattern, base_model = _detect_repo_pattern(arch.pretrained)

        if loading_pattern == "peft_adapter":
            self.encoder = _load_peft_adapter(
                adapter_id=arch.pretrained,
                base_model=base_model,
                auto_cls=auto_cls,
                auto_name=auto_name,
                load_kwargs=load_kwargs,
            )
        elif loading_pattern in ("gguf", "diffusers"):
            raise RuntimeError(
                f"'{arch.pretrained}' is packaged as {loading_pattern}, which "
                f"NeuralForge's HF wrapper can't load. "
                + ("Use the original PyTorch checkpoint (the non-GGUF sibling repo) "
                   "instead." if loading_pattern == "gguf"
                   else "Diffusion pipelines load via the `diffusers` library, not transformers.")
            )
        else:
            try:
                self.encoder = auto_cls.from_pretrained(arch.pretrained, **load_kwargs)
            except OSError as exc:
                msg = _redact_msg(str(exc))
                # Specific recovery for the "no recognized weight file"
                # error: tell the user this is probably a PEFT adapter.
                if "adapter_model" in msg or "adapter_config.json" in msg:
                    raise RuntimeError(
                        f"'{arch.pretrained}' looks like a PEFT/LoRA adapter "
                        f"(only `adapter_*` files in the repo). Install `peft` "
                        f"(`pip install peft`) and either point `pretrained` at "
                        f"the base model, or wait for adapter-aware loading.",
                    ) from exc
                # Auth-specific failures: 401/403/gated. Give the user a
                # clear, redacted hint instead of the raw HTTP soup.
                if any(s in msg for s in ("401", "gated", "Restricted", "must have access")):
                    raise RuntimeError(
                        f"Access denied for '{arch.pretrained}'. The repo is gated "
                        f"or restricted; your HF token doesn't have access. "
                        f"Request access at https://huggingface.co/{arch.pretrained} "
                        f"and re-run after the request is approved.\n"
                        f"Hub said: {msg.splitlines()[0][:200]}"
                    ) from exc
                raise RuntimeError(
                    f"Could not load HF model '{arch.pretrained}' as {auto_name}: {msg}\n"
                    f"Hint: check that the model id is correct and the task matches "
                    f"the model's pipeline_tag on the Hub."
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load HF model '{arch.pretrained}' as {auto_name}: "
                    f"{_redact_msg(str(exc))}\n"
                    f"Hint: check that the model id is correct and the task matches "
                    f"the model's pipeline_tag on the Hub."
                ) from exc

        if arch.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Try to leave the classification head trainable when present
            for head_name in ("classifier", "score", "lm_head", "qa_outputs"):
                head = getattr(self.encoder, head_name, None)
                if isinstance(head, nn.Module):
                    for p in head.parameters():
                        p.requires_grad = True

    def forward(self, *args, **kwargs):
        """Passthrough to the underlying HF model.

        Accepts both shapes the framework adapter can yield:
          * Positional Tensor (e.g. ``model(waveform)`` for audio classification)
          * Keyword dict (e.g. ``model(input_ids=…, attention_mask=…)`` for
            text after the loader unpacks the tokenizer dict via ``**inputs``).

        Returns:
          * ``outputs.logits`` for the classification family (single
            tensor, simple to argmax + render).
          * ``outputs.last_hidden_state`` for feature-extraction.
          * The **full structured output** for everything else — QA
            (``QuestionAnsweringModelOutput.start_logits/.end_logits``),
            object detection, segmentation, depth, etc. The server's
            postproc path branches on ``spec.output_kind`` and reads
            whatever fields it needs.

        Tolerates the full set of fields HF tokenizers / image processors
        emit — anything the underlying model doesn't recognize is silently
        dropped, so e.g. ``token_type_ids`` getting passed to a Whisper
        backbone won't blow up.
        """
        # Filter kwargs against the HF model's own forward signature.
        if kwargs:
            kwargs = self._filter_kwargs(kwargs)
            outputs = self.encoder(*args, **kwargs)
        elif len(args) == 1 and isinstance(args[0], dict):
            # Some callers pass a single dict positionally
            outputs = self.encoder(**self._filter_kwargs(args[0]))
        else:
            outputs = self.encoder(*args)

        # Structured outputs (QA, object detection, segmentation, depth)
        # carry several tensors. If we collapse them to one we lose the
        # ones the postprocessor needs (start_logits + end_logits for QA,
        # pred_boxes + scores for detection). Detect by attribute presence
        # and pass the whole object through.
        if (hasattr(outputs, "start_logits") and hasattr(outputs, "end_logits")):
            return outputs   # QA: postproc reads .start_logits / .end_logits
        if hasattr(outputs, "pred_boxes"):
            return outputs   # object detection
        if hasattr(outputs, "predicted_depth"):
            return outputs   # depth estimation
        if hasattr(outputs, "logits") and hasattr(outputs, "pred_masks"):
            return outputs   # segmentation (logits + masks)

        if hasattr(outputs, "logits"):
            return outputs.logits
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    def _filter_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Drop kwargs the underlying HF model's `forward` doesn't accept.

        HF models have wildly different forward signatures (audio models
        take ``input_features``, text models take ``input_ids``, vision
        models take ``pixel_values``). Filtering against the actual sig
        means our generic dict-unpacking dataloader path works against any
        backbone without hand-coding a shim per model.

        Note: we only do safe *name-only* renames (input_ids→input_features
        is dropped — the data shapes are different). The renames here are
        for fields like ``labels`` / ``label_ids`` that some HF models
        accept under either name with the same shape.
        """
        if self._fwd_param_names is None:
            import inspect as _inspect
            try:
                sig = _inspect.signature(self.encoder.forward)
                self._fwd_param_names = set(sig.parameters.keys())
            except (TypeError, ValueError):
                self._fwd_param_names = set()
        if not self._fwd_param_names:
            return kwargs

        # Renames are *only* safe when the source and target hold tensors of
        # the same shape (just a different attribute name). Cross-modality
        # renames like input_ids→input_features are NOT safe because token
        # IDs and mel-spectrogram features have completely different shapes,
        # so we deliberately don't include them here. Modality mismatch is
        # caught earlier by the validator + model inspector.
        renames = {
            "label_ids": "labels",   # token-classification & some seq2seq models
        }
        filtered: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in self._fwd_param_names:
                filtered[k] = v
            elif k in renames and renames[k] in self._fwd_param_names and renames[k] not in filtered:
                filtered[renames[k]] = v
            # else: silently drop — the model doesn't take this kwarg
        return filtered


# ---------------------------------------------------------------------------
# Loading-pattern probe + PEFT loader
# ---------------------------------------------------------------------------

def _redact_msg(msg: str) -> str:
    """Scrub anything that looks like an HF token from an error string
    before it surfaces to the user / logs / API. Falls back to the raw
    string if the auth module can't be imported.
    """
    try:
        from neural_platform.core.hf_auth import redact
        return redact(msg)
    except Exception:
        return msg


def _detect_repo_pattern(model_id: str):
    """Cheap pre-flight pattern detection. Returns (pattern, base_model).

    Uses the model source layer when available (which talks to the HF Hub
    API). Falls back to ('unknown', None) on any error so the caller can
    still attempt the standard load path.
    """
    try:
        from neural_platform.core.model_source import get_source
        info = get_source("huggingface").get_info(model_id)
        return info.loading_pattern, info.base_model
    except Exception:
        return "unknown", None


def _load_peft_adapter(adapter_id: str,
                        base_model: Optional[str],
                        auto_cls,
                        auto_name: str,
                        load_kwargs: Dict[str, Any]):
    """Load a PEFT/LoRA adapter on top of its base model.

    NeuralForge's HF wrapper hands us back the *merged* model so the rest of
    the training/inference pipeline doesn't have to know it's an adapter.
    The encoder is the base model with the adapter applied.

    This requires both the base model id (parsed from `adapter_config.json`
    by the model source layer) and the `peft` library installed.
    """
    if not base_model:
        raise RuntimeError(
            f"'{adapter_id}' is a PEFT/LoRA adapter but its base model couldn't "
            "be discovered (no readable `adapter_config.json`). Either set "
            "`pretrained` to the base model directly, or — if you have a local "
            "copy of the adapter — point `pretrained` at the local path so the "
            "loader can read `adapter_config.json` from disk."
        )
    try:
        from peft import PeftModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"'{adapter_id}' is a PEFT/LoRA adapter on top of '{base_model}'. "
            "Loading requires the `peft` package: pip install peft"
        ) from exc

    # Load the base model first via the same Auto class the user requested,
    # so heads / num_labels / classification configuration apply correctly.
    try:
        base = auto_cls.from_pretrained(base_model, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load adapter base model '{base_model}' as {auto_name}: "
            f"{exc}. The adapter '{adapter_id}' depends on this model loading "
            "successfully."
        ) from exc

    try:
        merged = PeftModel.from_pretrained(base, adapter_id)
    except Exception as exc:
        raise RuntimeError(
            f"Loaded base '{base_model}' but couldn't apply adapter "
            f"'{adapter_id}': {exc}."
        ) from exc

    return merged
