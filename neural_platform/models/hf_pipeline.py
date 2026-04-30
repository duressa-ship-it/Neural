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


# Map our task strings → the right transformers.Auto* class name.
# Kept as a string-table so we don't import transformers at module load.
_TASK_TO_AUTO_CLASS: Dict[str, str] = {
    # Text
    "text-classification":         "AutoModelForSequenceClassification",
    "token-classification":        "AutoModelForTokenClassification",
    "question-answering":          "AutoModelForQuestionAnswering",
    "summarization":               "AutoModelForSeq2SeqLM",
    "translation":                 "AutoModelForSeq2SeqLM",
    "text-generation":             "AutoModelForCausalLM",
    "fill-mask":                   "AutoModelForMaskedLM",
    "feature-extraction":          "AutoModel",
    "sentence-similarity":         "AutoModel",

    # Vision
    "image-classification":        "AutoModelForImageClassification",
    "image-segmentation":          "AutoModelForSemanticSegmentation",
    "object-detection":            "AutoModelForObjectDetection",
    "depth-estimation":            "AutoModelForDepthEstimation",
    "zero-shot-image-classification": "AutoModelForZeroShotImageClassification",

    # Video
    "video-classification":        "AutoModelForVideoClassification",

    # Audio
    "audio-classification":        "AutoModelForAudioClassification",
    "automatic-speech-recognition": "AutoModelForSpeechSeq2Seq",
    "voice-activity-detection":    "AutoModelForAudioClassification",

    # Multi-modal (where HF has a dedicated Auto class — others fall through to AutoModel)
    "visual-question-answering":   "AutoModelForVisualQuestionAnswering",
    "document-question-answering": "AutoModelForDocumentQuestionAnswering",
    "image-to-text":               "AutoModelForVision2Seq",
    "image-text-to-text":          "AutoModelForImageTextToText",
}


def _resolve_auto_class_name(task: Optional[str]) -> str:
    """Pick a `transformers.Auto*` class name for the given task.

    Falls back to ``AutoModel`` (the encoder-only base) when the task is
    None or unrecognized — that's still useful for feature extraction and
    most encoder-only fine-tuning workflows.
    """
    if task and task in _TASK_TO_AUTO_CLASS:
        return _TASK_TO_AUTO_CLASS[task]
    return "AutoModel"


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

        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "model.type='hf_pipeline' requires the `transformers` package.\n"
                "Install with: pip install transformers"
            ) from exc

        auto_name = _resolve_auto_class_name(self._task)
        try:
            auto_cls = getattr(transformers, auto_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"transformers does not expose `{auto_name}` for task "
                f"'{self._task}'. Upgrade `transformers` or pick a different task."
            ) from exc

        load_kwargs: Dict[str, Any] = {}
        if arch.revision:           load_kwargs["revision"]          = arch.revision
        if arch.trust_remote_code:  load_kwargs["trust_remote_code"] = True
        if arch.output_size:        load_kwargs["num_labels"]        = arch.output_size

        try:
            self.encoder = auto_cls.from_pretrained(arch.pretrained, **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load HF model '{arch.pretrained}' as {auto_name}: {exc}\n"
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

        Returns the model's logits (or last_hidden_state for encoder-only
        feature-extraction tasks). Tolerates the full set of fields HF
        tokenizers / image processors emit — anything the underlying model
        doesn't recognize is silently dropped, so e.g. ``token_type_ids``
        getting passed to a Whisper backbone won't blow up.
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
        """
        if self._fwd_param_names is None:
            import inspect as _inspect
            try:
                sig = _inspect.signature(self.encoder.forward)
                self._fwd_param_names = set(sig.parameters.keys())
            except (TypeError, ValueError):
                self._fwd_param_names = None
        if not self._fwd_param_names:
            return kwargs

        filtered: Dict[str, Any] = {}
        renames = {
            # Map common loader-side names to whatever the HF model expects.
            "input_ids": "input_features",   # ASR/audio models prefer features
        }
        for k, v in kwargs.items():
            if k in self._fwd_param_names:
                filtered[k] = v
            elif k in renames and renames[k] in self._fwd_param_names and renames[k] not in filtered:
                filtered[renames[k]] = v
            # else: silently drop — the model doesn't take this kwarg
        return filtered
