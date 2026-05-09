"""
NeuralForge — HF Pipeline Specs

Single source of truth that maps every supported HuggingFace
``pipeline_tag`` to the **everything else** the platform needs to wire a
real inference server (or a from-scratch trainer) around it:

  * which ``transformers.Auto*`` classes can load it (a fallback chain so
    we keep working across HF v4 → v5 API drift)
  * which preprocessor to load (tokenizer / feature extractor / image
    processor / unified processor)
  * what shape of input the server endpoint should accept
  * which post-processing path turns the model's outputs into a
    ``Prediction`` list (argmax over logits vs. ``.generate()`` decoded
    text vs. boxes vs. mask vs. QA span vs. depth)
  * the coarse :class:`core.tasks.Task` enum the validator should pin
    (so synthesized configs validate the same way the Builder does)
  * a default loss + a hint flag describing whether ``output_size`` is
    even meaningful

**Why fallback chains for the Auto class.** The transformers library has
shifted classes between v4 and v5: ``AutoModelForVision2Seq`` was folded
into ``AutoModelForImageTextToText`` in v5, ``AutoModelForSemanticSegmentation``
was renamed ``AutoModelForImageSegmentation``, and so on. Picking a single
class name leaves us one ``transformers`` upgrade away from a launch-time
``AttributeError``. The chain lets us prefer the modern name, fall back to
older names, and finally use a generic ``AutoModel`` so the server boots
even when the table is slightly behind the installed library.

Until this module existed the spec was scattered:

  * ``models/hf_pipeline.py::_TASK_TO_AUTO_CLASS`` knew the auto class
  * ``deploy/server.py::_try_load_tokenizer`` only ever loaded a
    tokenizer — wrong for image / audio / multimodal models
  * ``deploy/server.py::_build_hf_pipeline_input`` had its own task→shape
    dispatch that didn't agree with the auto-class table
  * ``web/inference_manager.py::_HF_TASK_TO_COARSE`` had a third version
    of the mapping for the synthesized config's ``Task`` field

That drift is why launching ``openai/whisper-tiny`` lit up:

  1. ``/info`` 500'd because the server didn't know it was a no-checkpoint
     launch and Pydantic rejected ``checkpoint_path=None``.
  2. The synthesized config defaulted ``training.task=classification``
     even though ASR returns text.
  3. The processor loader returned a tokenizer when Whisper needs the
     ``WhisperProcessor`` (feature extractor + tokenizer) to convert raw
     waveform into log-mel spectrograms.

Every consumer now reads from ``PIPELINE_SPECS`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from neural_platform.core.tasks import Task


# ---------------------------------------------------------------------------
# Enums (string constants — keeps the table dense and JSON-serializable)
# ---------------------------------------------------------------------------

# What kind of input the inference server accepts on /predict for this task.
# Drives the input adapter in deploy/server.py::_build_hf_pipeline_input.
INPUT_TEXT          = "text"             # str  → tokenizer
INPUT_IMAGE         = "image"            # base64 PNG/JPG → image processor
INPUT_AUDIO         = "audio"            # list[float] waveform → feature extractor
INPUT_VIDEO         = "video"            # 4D nested list → video processor
INPUT_IMAGE_TEXT    = "image+text"       # multimodal: VQA, image-text-to-text
INPUT_AUDIO_TEXT    = "audio+text"       # multimodal: audio captioning + question
INPUT_VIDEO_TEXT    = "video+text"
INPUT_TEXT_PAIR     = "text_pair"        # QA, sentence similarity
INPUT_TEXT_LABELS   = "text+labels"      # zero-shot classification
INPUT_IMAGE_LABELS  = "image+labels"     # zero-shot image classification
INPUT_ANY           = "any"              # any-to-any: route by what's in the request
INPUT_TENSOR        = "tensor"           # raw float vector (fallback)
INPUT_NONE          = "none"             # generative-only (text-to-image, etc.)

# What kind of preprocessor to load. Maps to a transformers.Auto* class.
PROCESSOR_NONE              = "none"
PROCESSOR_TOKENIZER         = "tokenizer"          # AutoTokenizer
PROCESSOR_IMAGE             = "image"              # AutoImageProcessor
PROCESSOR_FEATURE_EXTRACTOR = "feature_extractor"  # AutoFeatureExtractor (audio)
PROCESSOR_PROCESSOR         = "processor"          # AutoProcessor (multimodal: Whisper, BLIP, …)

# Post-processing path. The server's predict route uses this to know
# whether to argmax logits, decode generated tokens, or surface boxes.
POSTPROC_LOGITS            = "logits"            # standard classifier surface
POSTPROC_TOKEN_LOGITS      = "token_logits"      # per-token classification
POSTPROC_GENERATED_TEXT    = "generated_text"    # call .generate() then decode
POSTPROC_QA_SPANS          = "qa_spans"          # start_logits + end_logits
POSTPROC_BOXES             = "boxes"             # object detection
POSTPROC_MASKS             = "masks"             # segmentation
POSTPROC_DEPTH             = "depth"
POSTPROC_KEYPOINTS         = "keypoints"
POSTPROC_EMBEDDINGS        = "embeddings"        # last_hidden_state
POSTPROC_GENERATED_IMAGE   = "generated_image"   # diffusion / GAN — not really us
POSTPROC_GENERATED_AUDIO   = "generated_audio"
POSTPROC_GENERATED_VIDEO   = "generated_video"


@dataclass(frozen=True)
class PipelineSpec:
    """Everything the server / synthesizer / trainer need for one HF task.

    ``auto_classes`` is a fallback chain — the wrapper probes
    ``transformers`` at load time and picks the first name that exists in
    the installed library. This is the surface that survives v4↔v5 API
    renames (``AutoModelForVision2Seq`` → ``AutoModelForImageTextToText``,
    ``AutoModelForSemanticSegmentation`` → ``AutoModelForImageSegmentation``)
    without us having to pin a transformers version.
    """
    task:              str
    coarse_task:       Task
    auto_classes:      Tuple[str, ...]   # primary first, then v4/v5 alternates, then AutoModel
    processor_kind:    str
    input_kind:        str
    output_kind:       str
    needs_generation:  bool = False
    has_class_head:    bool = False
    default_loss:      str = "cross_entropy"
    modality:          str = "text"
    notes:             str = ""

    @property
    def auto_class(self) -> str:
        """Backward-compat alias — returns the preferred (first) Auto class
        name. Anything that needs the actual loaded class object should
        call ``resolve_auto_class(transformers, spec)`` instead."""
        return self.auto_classes[0] if self.auto_classes else "AutoModel"


# ---------------------------------------------------------------------------
# The table — single source of truth
# ---------------------------------------------------------------------------

# Add new HF tasks here. Anything missing falls through to a sensible
# AutoModel + tokenizer default in resolve(); the validator already warns.
#
# Auto-class chains: prefer the v5 name first (since that's where HF is
# moving), then v4 alternates, then a generic fallback. The wrapper picks
# the first one that the installed transformers actually exposes.
PIPELINE_SPECS: Dict[str, PipelineSpec] = {

    # ===== Text =====
    "text-classification": PipelineSpec(
        task="text-classification",
        coarse_task=Task.TEXT_CLASSIFICATION,
        auto_classes=("AutoModelForSequenceClassification",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_LOGITS,
        has_class_head=True,
        default_loss="cross_entropy",
        modality="text",
    ),
    "token-classification": PipelineSpec(
        task="token-classification",
        coarse_task=Task.TOKEN_CLASSIFICATION,
        auto_classes=("AutoModelForTokenClassification",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_TOKEN_LOGITS,
        has_class_head=True,
        default_loss="cross_entropy",
        modality="text",
        notes="Output is per-token; use the tokenizer's offsets to align to spans.",
    ),
    "question-answering": PipelineSpec(
        task="question-answering",
        coarse_task=Task.QUESTION_ANSWERING,
        auto_classes=("AutoModelForQuestionAnswering",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT_PAIR,
        output_kind=POSTPROC_QA_SPANS,
        default_loss="cross_entropy",
        modality="text",
        notes="Send {text: <question>, context: <passage>} via the proxy.",
    ),
    "summarization": PipelineSpec(
        task="summarization",
        coarse_task=Task.SUMMARIZATION,
        auto_classes=("AutoModelForSeq2SeqLM",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="text",
    ),
    "translation": PipelineSpec(
        task="translation",
        coarse_task=Task.TRANSLATION,
        auto_classes=("AutoModelForSeq2SeqLM",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="text",
    ),
    "text-generation": PipelineSpec(
        task="text-generation",
        coarse_task=Task.TEXT_GENERATION,
        auto_classes=("AutoModelForCausalLM",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="text",
    ),
    "text2text-generation": PipelineSpec(
        task="text2text-generation",
        coarse_task=Task.TEXT_GENERATION,
        auto_classes=("AutoModelForSeq2SeqLM",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="text",
    ),
    "fill-mask": PipelineSpec(
        task="fill-mask",
        coarse_task=Task.FILL_MASK,
        auto_classes=("AutoModelForMaskedLM",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_LOGITS,
        modality="text",
    ),
    "feature-extraction": PipelineSpec(
        task="feature-extraction",
        coarse_task=Task.FEATURE_EXTRACTION,
        auto_classes=("AutoModel",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_EMBEDDINGS,
        modality="text",
    ),
    "sentence-similarity": PipelineSpec(
        task="sentence-similarity",
        coarse_task=Task.SENTENCE_SIMILARITY,
        auto_classes=("AutoModel",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT_PAIR,
        output_kind=POSTPROC_EMBEDDINGS,
        modality="text",
    ),
    "zero-shot-classification": PipelineSpec(
        task="zero-shot-classification",
        coarse_task=Task.ZERO_SHOT_CLASSIFICATION,
        auto_classes=("AutoModelForSequenceClassification",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT_LABELS,
        output_kind=POSTPROC_LOGITS,
        modality="text",
        notes="Pass candidate labels in the request body — server runs NLI per label.",
    ),

    # ===== Vision =====
    "image-classification": PipelineSpec(
        task="image-classification",
        coarse_task=Task.IMAGE_CLASSIFICATION,
        auto_classes=("AutoModelForImageClassification",),
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_LOGITS,
        has_class_head=True,
        modality="image",
    ),
    "image-segmentation": PipelineSpec(
        task="image-segmentation",
        coarse_task=Task.IMAGE_SEGMENTATION,
        # v5 renamed `Semantic` → `Image`. Keep both so older transformers
        # versions still resolve.
        auto_classes=("AutoModelForImageSegmentation",
                       "AutoModelForSemanticSegmentation"),
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_MASKS,
        modality="image",
    ),
    "object-detection": PipelineSpec(
        task="object-detection",
        coarse_task=Task.OBJECT_DETECTION,
        auto_classes=("AutoModelForObjectDetection",),
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_BOXES,
        modality="image",
    ),
    "depth-estimation": PipelineSpec(
        task="depth-estimation",
        coarse_task=Task.DEPTH_ESTIMATION,
        auto_classes=("AutoModelForDepthEstimation",),
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_DEPTH,
        default_loss="mse",
        modality="image",
    ),
    "zero-shot-image-classification": PipelineSpec(
        task="zero-shot-image-classification",
        coarse_task=Task.ZERO_SHOT_IMAGE_CLASSIF,
        auto_classes=("AutoModelForZeroShotImageClassification",),
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_IMAGE_LABELS,
        output_kind=POSTPROC_LOGITS,
        modality="image",
    ),
    "image-to-text": PipelineSpec(
        task="image-to-text",
        coarse_task=Task.IMAGE_TO_TEXT,
        # v5: AutoModelForVision2Seq was removed — folded into
        # AutoModelForImageTextToText. TrOCR-style classic image-captioning
        # models can also be loaded as VisionEncoderDecoderModel directly.
        # AutoModelForTextRecognition is the v5 OCR-specific class.
        auto_classes=("AutoModelForImageTextToText",
                       "AutoModelForVision2Seq",
                       "AutoModelForTextRecognition",
                       "VisionEncoderDecoderModel"),
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="image",
        notes=(
            "Vision-to-text path: TrOCR / BLIP-style captioners. The chain "
            "tries the v5 unified class first and falls back to the legacy "
            "Vision2Seq / VisionEncoderDecoder names so the server boots "
            "regardless of installed transformers version."
        ),
    ),
    "image-to-image": PipelineSpec(
        task="image-to-image",
        coarse_task=Task.IMAGE_TO_IMAGE,
        auto_classes=("AutoModelForImageToImage",),
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_GENERATED_IMAGE,
        modality="image",
    ),
    "keypoint-detection": PipelineSpec(
        task="keypoint-detection",
        coarse_task=Task.KEYPOINT_DETECTION,
        auto_classes=("AutoModelForKeypointDetection",),
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_KEYPOINTS,
        modality="image",
    ),

    # ===== Video =====
    "video-classification": PipelineSpec(
        task="video-classification",
        coarse_task=Task.VIDEO_CLASSIFICATION,
        auto_classes=("AutoModelForVideoClassification",),
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_VIDEO,
        output_kind=POSTPROC_LOGITS,
        has_class_head=True,
        modality="video",
    ),

    # ===== Audio =====
    "audio-classification": PipelineSpec(
        task="audio-classification",
        coarse_task=Task.AUDIO_CLASSIFICATION,
        auto_classes=("AutoModelForAudioClassification",),
        processor_kind=PROCESSOR_FEATURE_EXTRACTOR,
        input_kind=INPUT_AUDIO,
        output_kind=POSTPROC_LOGITS,
        has_class_head=True,
        modality="audio",
    ),
    "automatic-speech-recognition": PipelineSpec(
        task="automatic-speech-recognition",
        coarse_task=Task.AUTOMATIC_SPEECH_RECOGNITION,
        auto_classes=("AutoModelForSpeechSeq2Seq",),
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_AUDIO,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="audio",
        notes=(
            "Send `inputs` as a flat list of float32 waveform samples at the "
            "model's expected sample rate (Whisper: 16kHz). The processor "
            "converts to log-mel features automatically; the server then "
            "runs `.generate()` and decodes back to text."
        ),
    ),
    "voice-activity-detection": PipelineSpec(
        task="voice-activity-detection",
        coarse_task=Task.VOICE_ACTIVITY_DETECTION,
        auto_classes=("AutoModelForAudioClassification",),
        processor_kind=PROCESSOR_FEATURE_EXTRACTOR,
        input_kind=INPUT_AUDIO,
        output_kind=POSTPROC_LOGITS,
        modality="audio",
    ),

    # ===== Multi-modal =====
    "visual-question-answering": PipelineSpec(
        task="visual-question-answering",
        coarse_task=Task.VISUAL_QUESTION_ANSWERING,
        # AutoModelForVisualQuestionAnswering exists in v4 + v5; modern VQA
        # models also surface as ImageTextToText (BLIP, LLaVA, …).
        auto_classes=("AutoModelForVisualQuestionAnswering",
                       "AutoModelForImageTextToText"),
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_IMAGE_TEXT,
        output_kind=POSTPROC_LOGITS,
        modality="image",
        notes="Send {image_b64: …, text: <question>}.",
    ),
    "document-question-answering": PipelineSpec(
        task="document-question-answering",
        coarse_task=Task.DOCUMENT_QUESTION_ANSWERING,
        auto_classes=("AutoModelForDocumentQuestionAnswering",
                       "AutoModelForImageTextToText"),
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_IMAGE_TEXT,
        output_kind=POSTPROC_QA_SPANS,
        modality="image",
    ),
    "image-text-to-text": PipelineSpec(
        task="image-text-to-text",
        coarse_task=Task.IMAGE_TEXT_TO_TEXT,
        # Modern HF unified-multimodal: LLaVA, BLIP-2, Idefics, Qwen-VL,
        # Gemma-3 vision, etc. v5 dropped Vision2Seq; we keep it as a v4
        # fallback. CausalLM is a last resort for plain decoder-only
        # multimodal LMs that don't expose the unified Auto class.
        auto_classes=("AutoModelForImageTextToText",
                       "AutoModelForVision2Seq",
                       "AutoModelForCausalLM"),
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_IMAGE_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="image",
    ),
    "any-to-any": PipelineSpec(
        task="any-to-any",
        coarse_task=Task.ANY_TO_ANY,
        # Unified multimodal LMs (Gemma-3, Phi-3.5-vision, Qwen2-VL, etc.)
        # The chain prefers ImageTextToText (most common), falls back to
        # CausalLM (decoder-only LMs), and finally AutoModel for the
        # ungainly cases. Routing on the input side happens via
        # input_kind=INPUT_ANY: whatever the request carries (image / text /
        # audio / mix) gets dispatched to the processor's unified call.
        auto_classes=("AutoModelForImageTextToText",
                       "AutoModelForCausalLM",
                       "AutoModelForSeq2SeqLM",
                       "AutoModel"),
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_ANY,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="any",
        notes=(
            "Any-to-any wraps unified multimodal LMs (Gemma-3, Qwen2-VL, "
            "Phi-3.5-vision). The processor accepts whichever combination "
            "of `text` / `image_b64` / `inputs` (audio) is present in the "
            "request; the server runs `.generate()` and decodes the output."
        ),
    ),
    "table-question-answering": PipelineSpec(
        task="table-question-answering",
        coarse_task=Task.TABLE_QUESTION_ANSWERING,
        auto_classes=("AutoModelForTableQuestionAnswering",),
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT_PAIR,
        output_kind=POSTPROC_LOGITS,
        modality="text",
    ),
}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

# Used when the user picked a task we don't have a richer spec for — keep
# the server functional with sensible defaults rather than crashing.
_DEFAULT_SPEC = PipelineSpec(
    task="custom",
    coarse_task=Task.CUSTOM,
    auto_classes=("AutoModel",),
    processor_kind=PROCESSOR_TOKENIZER,
    input_kind=INPUT_TENSOR,
    output_kind=POSTPROC_EMBEDDINGS,
    modality="unknown",
    notes="Unrecognized pipeline_tag — falling back to AutoModel + tokenizer.",
)


def resolve(pipeline_task: Optional[str]) -> PipelineSpec:
    """Look up the spec for an HF pipeline_tag.

    Returns ``_DEFAULT_SPEC`` for unknown / empty tasks. The default keeps
    the server functional (tokenizer + AutoModel + raw embeddings) and the
    validator is responsible for surfacing the unknown-task warning so the
    user knows they're on the fallback path.
    """
    if not pipeline_task:
        return _DEFAULT_SPEC
    return PIPELINE_SPECS.get(pipeline_task.strip().lower(), _DEFAULT_SPEC)


def supported_tasks() -> list[str]:
    """All pipeline_tag strings the table knows. UI uses this for dropdowns."""
    return sorted(PIPELINE_SPECS.keys())


def resolve_auto_class(transformers_module: Any, spec: PipelineSpec
                        ) -> Tuple[Any, str]:
    """Probe ``transformers`` for the first auto-class in the spec's chain
    that actually exists in the installed library.

    Returns ``(class, name)``. Raises ``RuntimeError`` listing every name
    we tried when none resolve — keeps the failure message actionable
    instead of leaving the user staring at the v4 class name and wondering
    whether to upgrade or downgrade.
    """
    tried: list[str] = []
    for name in spec.auto_classes:
        tried.append(name)
        try:
            cls = getattr(transformers_module, name)
        except (AttributeError, ImportError):
            continue
        if cls is not None:
            return cls, name
    # Last-ditch: AutoModel is in every transformers release. Surface it as
    # a soft fallback so the server doesn't crash at startup; predict-time
    # output handling will degrade to embeddings (which is at least
    # debuggable) instead of a 500 on the lifespan hook.
    fallback = getattr(transformers_module, "AutoModel", None)
    if fallback is not None:
        tried.append("AutoModel")
        return fallback, "AutoModel"
    raise RuntimeError(
        f"None of {spec.auto_classes!r} are exposed by the installed "
        f"transformers library, and AutoModel itself is missing. "
        f"Tried: {tried}. Upgrade transformers or pick a different task."
    )


# Translates a PROCESSOR_* constant to the transformers Auto* class name.
# Centralized here so deploy/server.py doesn't have its own table.
PROCESSOR_AUTO_CLASS: Dict[str, str] = {
    PROCESSOR_TOKENIZER:         "AutoTokenizer",
    PROCESSOR_IMAGE:             "AutoImageProcessor",
    PROCESSOR_FEATURE_EXTRACTOR: "AutoFeatureExtractor",
    PROCESSOR_PROCESSOR:         "AutoProcessor",
}
