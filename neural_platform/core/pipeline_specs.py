"""
NeuralForge — HF Pipeline Specs

Single source of truth that maps every supported HuggingFace
``pipeline_tag`` to the **everything else** the platform needs to wire a
real inference server (or a from-scratch trainer) around it:

  * which ``transformers.Auto*`` class to instantiate
  * which preprocessor to load (tokenizer / feature extractor / image
    processor / unified processor)
  * what shape of input the server endpoint should accept
  * which post-processing path turns the model's outputs into a
    ``Prediction`` list (argmax over logits vs. ``.generate()`` decoded
    text vs. boxes vs. mask vs. depth)
  * the coarse :class:`core.tasks.Task` enum the validator should pin
    (so synthesized configs validate the same way the Builder does)
  * a default loss + a hint flag describing whether ``output_size`` is
    even meaningful

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

from dataclasses import dataclass
from typing import Dict, Optional

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
    """Everything the server / synthesizer / trainer need for one HF task."""
    task:              str             # HF pipeline_tag, e.g. "text-classification"
    coarse_task:       Task            # NeuralForge Task enum value (drives validator + loss)
    auto_class:        str             # transformers.Auto* class name
    processor_kind:    str             # which preprocessor to load (PROCESSOR_*)
    input_kind:        str             # what input shape /predict accepts (INPUT_*)
    output_kind:       str             # how to render the model's output (POSTPROC_*)
    needs_generation:  bool = False    # call model.generate() instead of forward()
    has_class_head:    bool = False    # output_size / num_labels is meaningful
    default_loss:      str = "cross_entropy"   # for synthesized configs
    modality:          str = "text"    # primary modality (text / image / audio / video)
    notes:             str = ""        # human hints (rendered in /info)


# ---------------------------------------------------------------------------
# The table — single source of truth
# ---------------------------------------------------------------------------

# Add new HF tasks here. Anything missing falls through to a sensible
# AutoModel + tokenizer default in resolve(); the validator already warns.
PIPELINE_SPECS: Dict[str, PipelineSpec] = {

    # ===== Text =====
    "text-classification": PipelineSpec(
        task="text-classification",
        coarse_task=Task.TEXT_CLASSIFICATION,
        auto_class="AutoModelForSequenceClassification",
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
        auto_class="AutoModelForTokenClassification",
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
        auto_class="AutoModelForQuestionAnswering",
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
        auto_class="AutoModelForSeq2SeqLM",
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="text",
    ),
    "translation": PipelineSpec(
        task="translation",
        coarse_task=Task.TRANSLATION,
        auto_class="AutoModelForSeq2SeqLM",
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="text",
    ),
    "text-generation": PipelineSpec(
        task="text-generation",
        coarse_task=Task.TEXT_GENERATION,
        auto_class="AutoModelForCausalLM",
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="text",
    ),
    "text2text-generation": PipelineSpec(
        task="text2text-generation",
        coarse_task=Task.TEXT_GENERATION,
        auto_class="AutoModelForSeq2SeqLM",
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="text",
    ),
    "fill-mask": PipelineSpec(
        task="fill-mask",
        coarse_task=Task.FILL_MASK,
        auto_class="AutoModelForMaskedLM",
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_LOGITS,
        modality="text",
    ),
    "feature-extraction": PipelineSpec(
        task="feature-extraction",
        coarse_task=Task.FEATURE_EXTRACTION,
        auto_class="AutoModel",
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT,
        output_kind=POSTPROC_EMBEDDINGS,
        modality="text",
    ),
    "sentence-similarity": PipelineSpec(
        task="sentence-similarity",
        coarse_task=Task.SENTENCE_SIMILARITY,
        auto_class="AutoModel",
        processor_kind=PROCESSOR_TOKENIZER,
        input_kind=INPUT_TEXT_PAIR,
        output_kind=POSTPROC_EMBEDDINGS,
        modality="text",
    ),
    "zero-shot-classification": PipelineSpec(
        task="zero-shot-classification",
        coarse_task=Task.ZERO_SHOT_CLASSIFICATION,
        auto_class="AutoModelForSequenceClassification",
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
        auto_class="AutoModelForImageClassification",
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_LOGITS,
        has_class_head=True,
        modality="image",
    ),
    "image-segmentation": PipelineSpec(
        task="image-segmentation",
        coarse_task=Task.IMAGE_SEGMENTATION,
        auto_class="AutoModelForSemanticSegmentation",
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_MASKS,
        modality="image",
    ),
    "object-detection": PipelineSpec(
        task="object-detection",
        coarse_task=Task.OBJECT_DETECTION,
        auto_class="AutoModelForObjectDetection",
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_BOXES,
        modality="image",
    ),
    "depth-estimation": PipelineSpec(
        task="depth-estimation",
        coarse_task=Task.DEPTH_ESTIMATION,
        auto_class="AutoModelForDepthEstimation",
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_DEPTH,
        default_loss="mse",
        modality="image",
    ),
    "zero-shot-image-classification": PipelineSpec(
        task="zero-shot-image-classification",
        coarse_task=Task.ZERO_SHOT_IMAGE_CLASSIF,
        auto_class="AutoModelForZeroShotImageClassification",
        processor_kind=PROCESSOR_PROCESSOR,        # CLIP-style — image + text together
        input_kind=INPUT_IMAGE_LABELS,
        output_kind=POSTPROC_LOGITS,
        modality="image",
    ),
    "image-to-text": PipelineSpec(
        task="image-to-text",
        coarse_task=Task.IMAGE_TO_TEXT,
        auto_class="AutoModelForVision2Seq",
        processor_kind=PROCESSOR_PROCESSOR,        # image processor + tokenizer
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="image",
    ),
    "keypoint-detection": PipelineSpec(
        task="keypoint-detection",
        coarse_task=Task.KEYPOINT_DETECTION,
        auto_class="AutoModelForKeypointDetection",
        processor_kind=PROCESSOR_IMAGE,
        input_kind=INPUT_IMAGE,
        output_kind=POSTPROC_KEYPOINTS,
        modality="image",
    ),

    # ===== Video =====
    "video-classification": PipelineSpec(
        task="video-classification",
        coarse_task=Task.VIDEO_CLASSIFICATION,
        auto_class="AutoModelForVideoClassification",
        processor_kind=PROCESSOR_IMAGE,           # HF video models reuse image processors
        input_kind=INPUT_VIDEO,
        output_kind=POSTPROC_LOGITS,
        has_class_head=True,
        modality="video",
    ),

    # ===== Audio =====
    "audio-classification": PipelineSpec(
        task="audio-classification",
        coarse_task=Task.AUDIO_CLASSIFICATION,
        auto_class="AutoModelForAudioClassification",
        processor_kind=PROCESSOR_FEATURE_EXTRACTOR,
        input_kind=INPUT_AUDIO,
        output_kind=POSTPROC_LOGITS,
        has_class_head=True,
        modality="audio",
    ),
    "automatic-speech-recognition": PipelineSpec(
        task="automatic-speech-recognition",
        coarse_task=Task.AUTOMATIC_SPEECH_RECOGNITION,
        auto_class="AutoModelForSpeechSeq2Seq",
        processor_kind=PROCESSOR_PROCESSOR,        # WhisperProcessor / Wav2Vec2Processor
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
        auto_class="AutoModelForAudioClassification",
        processor_kind=PROCESSOR_FEATURE_EXTRACTOR,
        input_kind=INPUT_AUDIO,
        output_kind=POSTPROC_LOGITS,
        modality="audio",
    ),

    # ===== Multi-modal =====
    "visual-question-answering": PipelineSpec(
        task="visual-question-answering",
        coarse_task=Task.VISUAL_QUESTION_ANSWERING,
        auto_class="AutoModelForVisualQuestionAnswering",
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_IMAGE_TEXT,
        output_kind=POSTPROC_LOGITS,
        modality="image",
        notes="Send {image_b64: …, text: <question>}.",
    ),
    "document-question-answering": PipelineSpec(
        task="document-question-answering",
        coarse_task=Task.DOCUMENT_QUESTION_ANSWERING,
        auto_class="AutoModelForDocumentQuestionAnswering",
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_IMAGE_TEXT,
        output_kind=POSTPROC_QA_SPANS,
        modality="image",
    ),
    "image-text-to-text": PipelineSpec(
        task="image-text-to-text",
        coarse_task=Task.IMAGE_TEXT_TO_TEXT,
        auto_class="AutoModelForImageTextToText",
        processor_kind=PROCESSOR_PROCESSOR,
        input_kind=INPUT_IMAGE_TEXT,
        output_kind=POSTPROC_GENERATED_TEXT,
        needs_generation=True,
        modality="image",
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
    auto_class="AutoModel",
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


# Translates a PROCESSOR_* constant to the transformers Auto* class name.
# Centralized here so deploy/server.py doesn't have its own table.
PROCESSOR_AUTO_CLASS: Dict[str, str] = {
    PROCESSOR_TOKENIZER:         "AutoTokenizer",
    PROCESSOR_IMAGE:             "AutoImageProcessor",
    PROCESSOR_FEATURE_EXTRACTOR: "AutoFeatureExtractor",
    PROCESSOR_PROCESSOR:         "AutoProcessor",
}
