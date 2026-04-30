"""
NeuralForge Task Taxonomy

A "task" is *what the model does* (image-classification, ASR, summarization,
visual-question-answering) — distinct from the "model_type" which is *how
it's built* (cnn, transformer, audio_cnn, hf_pipeline). One task can be
served by several model types; one model type can serve several tasks.

The taxonomy intentionally mirrors the HuggingFace `pipeline_tag` values
so that:

  * `/api/hf/search?modality=audio` can also filter by `task=audio-classification`
  * The Builder asks "what task?" first, then suggests an architecture
  * The validator checks task ↔ model_type compatibility
  * `HFPipelineModel` (the universal HF wrapper) routes to the right
    `transformers.Auto*` class based on the task

Reference: https://huggingface.co/tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Task(str, Enum):
    """Mirrors HuggingFace pipeline_tag values, with hyphens."""

    # Natural language processing
    TEXT_CLASSIFICATION         = "text-classification"
    TOKEN_CLASSIFICATION        = "token-classification"
    QUESTION_ANSWERING          = "question-answering"
    ZERO_SHOT_CLASSIFICATION    = "zero-shot-classification"
    SUMMARIZATION               = "summarization"
    TRANSLATION                 = "translation"
    TEXT_GENERATION             = "text-generation"
    FILL_MASK                   = "fill-mask"
    SENTENCE_SIMILARITY         = "sentence-similarity"
    FEATURE_EXTRACTION          = "feature-extraction"
    TABLE_QUESTION_ANSWERING    = "table-question-answering"
    TEXT_RANKING                = "text-ranking"

    # Computer vision
    IMAGE_CLASSIFICATION        = "image-classification"
    IMAGE_SEGMENTATION          = "image-segmentation"
    OBJECT_DETECTION            = "object-detection"
    DEPTH_ESTIMATION            = "depth-estimation"
    IMAGE_TO_IMAGE              = "image-to-image"
    IMAGE_TO_TEXT               = "image-to-text"
    TEXT_TO_IMAGE               = "text-to-image"
    UNCONDITIONAL_IMAGE_GEN     = "unconditional-image-generation"
    ZERO_SHOT_IMAGE_CLASSIF     = "zero-shot-image-classification"
    KEYPOINT_DETECTION          = "keypoint-detection"

    # Video
    VIDEO_CLASSIFICATION        = "video-classification"
    TEXT_TO_VIDEO               = "text-to-video"
    VIDEO_TO_VIDEO              = "video-to-video"
    IMAGE_TO_VIDEO              = "image-to-video"

    # Audio
    AUDIO_CLASSIFICATION        = "audio-classification"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    TEXT_TO_SPEECH              = "text-to-speech"
    TEXT_TO_AUDIO               = "text-to-audio"
    AUDIO_TO_AUDIO              = "audio-to-audio"
    VOICE_ACTIVITY_DETECTION    = "voice-activity-detection"

    # Tabular
    TABULAR_CLASSIFICATION      = "tabular-classification"
    TABULAR_REGRESSION          = "tabular-regression"
    TIME_SERIES_FORECASTING     = "time-series-forecasting"

    # Multi-modal
    VISUAL_QUESTION_ANSWERING   = "visual-question-answering"
    DOCUMENT_QUESTION_ANSWERING = "document-question-answering"
    IMAGE_TEXT_TO_TEXT          = "image-text-to-text"
    VIDEO_TEXT_TO_TEXT          = "video-text-to-text"
    AUDIO_TEXT_TO_TEXT          = "audio-text-to-text"
    ANY_TO_ANY                  = "any-to-any"

    # Reinforcement / Robotics
    REINFORCEMENT_LEARNING      = "reinforcement-learning"
    ROBOTICS                    = "robotics"

    # Generic / unknown
    CLASSIFICATION              = "classification"        # generic, modality-agnostic
    REGRESSION                  = "regression"            # generic
    CUSTOM                      = "custom"


@dataclass(frozen=True)
class TaskMeta:
    """Metadata describing one task — what input/output, which architectures fit."""
    task:           Task
    inputs:         List[str]                  # e.g. ["text"], ["image"], ["image", "text"]
    outputs:        List[str]                  # e.g. ["class"], ["text"], ["image"]
    modality:       str                        # primary modality (matches core.modality.Modality)
    suggested_models: List[str]                # ordered list of model_type strings
    multimodal:     bool = False
    generative:     bool = False
    requires_pretrained: bool = False          # tasks like ASR/translation need an HF backbone

    @property
    def is_classification(self) -> bool:
        return "class" in self.outputs


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# `hf_pipeline` is the universal HF wrapper (added later this round).
# When `requires_pretrained=True`, the validator forces a `pretrained` field
# on the model since these tasks aren't realistically trained from scratch.
TASK_CATALOG: dict[Task, TaskMeta] = {

    # Text
    Task.TEXT_CLASSIFICATION:   TaskMeta(Task.TEXT_CLASSIFICATION,   ["text"],         ["class"],  "text",  ["transformer", "hf_pipeline", "rnn"]),
    Task.TOKEN_CLASSIFICATION:  TaskMeta(Task.TOKEN_CLASSIFICATION,  ["text"],         ["class[]"],"text",  ["hf_pipeline", "transformer"], requires_pretrained=True),
    Task.QUESTION_ANSWERING:    TaskMeta(Task.QUESTION_ANSWERING,    ["text", "text"], ["text"],   "text",  ["hf_pipeline"], requires_pretrained=True, generative=True),
    Task.SUMMARIZATION:         TaskMeta(Task.SUMMARIZATION,         ["text"],         ["text"],   "text",  ["hf_pipeline"], requires_pretrained=True, generative=True),
    Task.TRANSLATION:           TaskMeta(Task.TRANSLATION,           ["text"],         ["text"],   "text",  ["hf_pipeline"], requires_pretrained=True, generative=True),
    Task.TEXT_GENERATION:       TaskMeta(Task.TEXT_GENERATION,       ["text"],         ["text"],   "text",  ["hf_pipeline"], requires_pretrained=True, generative=True),
    Task.FILL_MASK:             TaskMeta(Task.FILL_MASK,             ["text"],         ["text"],   "text",  ["hf_pipeline", "transformer"], requires_pretrained=True),
    Task.ZERO_SHOT_CLASSIFICATION: TaskMeta(Task.ZERO_SHOT_CLASSIFICATION, ["text", "labels"], ["class"], "text", ["hf_pipeline"], requires_pretrained=True),
    Task.SENTENCE_SIMILARITY:   TaskMeta(Task.SENTENCE_SIMILARITY,   ["text", "text"], ["score"],  "text",  ["hf_pipeline"], requires_pretrained=True),
    Task.FEATURE_EXTRACTION:    TaskMeta(Task.FEATURE_EXTRACTION,    ["text"],         ["embedding"], "text", ["hf_pipeline", "transformer"]),
    Task.TABLE_QUESTION_ANSWERING: TaskMeta(Task.TABLE_QUESTION_ANSWERING, ["table", "text"], ["text"], "text", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.TEXT_RANKING:          TaskMeta(Task.TEXT_RANKING,          ["text", "text[]"], ["score[]"], "text", ["hf_pipeline"], requires_pretrained=True),

    # Vision
    Task.IMAGE_CLASSIFICATION:  TaskMeta(Task.IMAGE_CLASSIFICATION,  ["image"],        ["class"],  "image", ["cnn", "hf_pipeline"]),
    Task.IMAGE_SEGMENTATION:    TaskMeta(Task.IMAGE_SEGMENTATION,    ["image"],        ["mask"],   "image", ["hf_pipeline"], requires_pretrained=True),
    Task.OBJECT_DETECTION:      TaskMeta(Task.OBJECT_DETECTION,      ["image"],        ["boxes"],  "image", ["hf_pipeline"], requires_pretrained=True),
    Task.DEPTH_ESTIMATION:      TaskMeta(Task.DEPTH_ESTIMATION,      ["image"],        ["depth"],  "image", ["hf_pipeline"], requires_pretrained=True),
    Task.IMAGE_TO_IMAGE:        TaskMeta(Task.IMAGE_TO_IMAGE,        ["image"],        ["image"],  "image", ["hf_pipeline"], requires_pretrained=True, generative=True),
    Task.IMAGE_TO_TEXT:         TaskMeta(Task.IMAGE_TO_TEXT,         ["image"],        ["text"],   "image", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.TEXT_TO_IMAGE:         TaskMeta(Task.TEXT_TO_IMAGE,         ["text"],         ["image"],  "image", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.UNCONDITIONAL_IMAGE_GEN: TaskMeta(Task.UNCONDITIONAL_IMAGE_GEN, [],          ["image"],  "image", ["hf_pipeline"], requires_pretrained=True, generative=True),
    Task.ZERO_SHOT_IMAGE_CLASSIF: TaskMeta(Task.ZERO_SHOT_IMAGE_CLASSIF, ["image", "labels"], ["class"], "image", ["hf_pipeline"], multimodal=True, requires_pretrained=True),
    Task.KEYPOINT_DETECTION:    TaskMeta(Task.KEYPOINT_DETECTION,    ["image"],        ["keypoints"], "image", ["hf_pipeline"], requires_pretrained=True),

    # Video
    Task.VIDEO_CLASSIFICATION:  TaskMeta(Task.VIDEO_CLASSIFICATION,  ["video"],        ["class"],  "video", ["video_cnn", "hf_pipeline"]),
    Task.TEXT_TO_VIDEO:         TaskMeta(Task.TEXT_TO_VIDEO,         ["text"],         ["video"],  "video", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.VIDEO_TO_VIDEO:        TaskMeta(Task.VIDEO_TO_VIDEO,        ["video"],        ["video"],  "video", ["hf_pipeline"], requires_pretrained=True, generative=True),
    Task.IMAGE_TO_VIDEO:        TaskMeta(Task.IMAGE_TO_VIDEO,        ["image"],        ["video"],  "video", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),

    # Audio
    Task.AUDIO_CLASSIFICATION:  TaskMeta(Task.AUDIO_CLASSIFICATION,  ["audio"],        ["class"],  "audio", ["audio_cnn", "hf_pipeline"]),
    Task.AUTOMATIC_SPEECH_RECOGNITION: TaskMeta(Task.AUTOMATIC_SPEECH_RECOGNITION, ["audio"], ["text"], "audio", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.TEXT_TO_SPEECH:        TaskMeta(Task.TEXT_TO_SPEECH,        ["text"],         ["audio"],  "audio", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.TEXT_TO_AUDIO:         TaskMeta(Task.TEXT_TO_AUDIO,         ["text"],         ["audio"],  "audio", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.AUDIO_TO_AUDIO:        TaskMeta(Task.AUDIO_TO_AUDIO,        ["audio"],        ["audio"],  "audio", ["hf_pipeline"], requires_pretrained=True, generative=True),
    Task.VOICE_ACTIVITY_DETECTION: TaskMeta(Task.VOICE_ACTIVITY_DETECTION, ["audio"], ["mask"], "audio", ["hf_pipeline"], requires_pretrained=True),

    # Tabular
    Task.TABULAR_CLASSIFICATION: TaskMeta(Task.TABULAR_CLASSIFICATION, ["features"],   ["class"],  "tabular", ["tabular", "mlp"]),
    Task.TABULAR_REGRESSION:    TaskMeta(Task.TABULAR_REGRESSION,    ["features"],     ["scalar"], "tabular", ["tabular", "mlp"]),
    Task.TIME_SERIES_FORECASTING: TaskMeta(Task.TIME_SERIES_FORECASTING, ["sequence"], ["sequence"], "time_series", ["tcn", "rnn", "hf_pipeline"]),

    # Multi-modal
    Task.VISUAL_QUESTION_ANSWERING:   TaskMeta(Task.VISUAL_QUESTION_ANSWERING,   ["image", "text"], ["text"], "image", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.DOCUMENT_QUESTION_ANSWERING: TaskMeta(Task.DOCUMENT_QUESTION_ANSWERING, ["image", "text"], ["text"], "document", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.IMAGE_TEXT_TO_TEXT:    TaskMeta(Task.IMAGE_TEXT_TO_TEXT,    ["image", "text"], ["text"], "image", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.VIDEO_TEXT_TO_TEXT:    TaskMeta(Task.VIDEO_TEXT_TO_TEXT,    ["video", "text"], ["text"], "video", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.AUDIO_TEXT_TO_TEXT:    TaskMeta(Task.AUDIO_TEXT_TO_TEXT,    ["audio", "text"], ["text"], "audio", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),
    Task.ANY_TO_ANY:            TaskMeta(Task.ANY_TO_ANY,            ["any"],          ["any"],   "unknown", ["hf_pipeline"], multimodal=True, requires_pretrained=True, generative=True),

    # RL / Robotics
    Task.REINFORCEMENT_LEARNING: TaskMeta(Task.REINFORCEMENT_LEARNING, ["state"],     ["action"], "unknown", ["custom"]),
    Task.ROBOTICS:              TaskMeta(Task.ROBOTICS,              ["state"],        ["action"], "unknown", ["custom"]),

    # Generic
    Task.CLASSIFICATION:        TaskMeta(Task.CLASSIFICATION,        ["features"],     ["class"],  "tabular", ["mlp", "tabular", "transformer", "cnn", "rnn"]),
    Task.REGRESSION:            TaskMeta(Task.REGRESSION,            ["features"],     ["scalar"], "tabular", ["mlp", "tabular"]),
    Task.CUSTOM:                TaskMeta(Task.CUSTOM,                ["any"],          ["any"],    "unknown", []),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_meta(task: Task | str) -> TaskMeta:
    """Look up a task's metadata. Accepts the enum or its string value."""
    if isinstance(task, str):
        try:
            task = Task(task)
        except ValueError:
            return TASK_CATALOG[Task.CUSTOM]
    return TASK_CATALOG.get(task, TASK_CATALOG[Task.CUSTOM])


def tasks_by_modality(modality: str) -> List[Task]:
    """All tasks that primarily consume the given modality."""
    return [t for t, m in TASK_CATALOG.items() if m.modality == modality]


def tasks_by_model(model_type: str) -> List[Task]:
    """All tasks for which `model_type` is a sane choice."""
    return [t for t, m in TASK_CATALOG.items() if model_type in m.suggested_models]


def is_compatible(task: Task | str, model_type: str) -> bool:
    """Does this model_type work for this task?"""
    return model_type in get_meta(task).suggested_models


def grouped_for_ui() -> List[dict]:
    """Group tasks by family for the Builder picker."""
    return [
        {"label": "Text",         "tasks": [t.value for t in [
            Task.TEXT_CLASSIFICATION, Task.TOKEN_CLASSIFICATION, Task.QUESTION_ANSWERING,
            Task.SUMMARIZATION, Task.TRANSLATION, Task.TEXT_GENERATION,
            Task.FILL_MASK, Task.ZERO_SHOT_CLASSIFICATION, Task.SENTENCE_SIMILARITY,
            Task.FEATURE_EXTRACTION, Task.TEXT_RANKING,
        ]]},
        {"label": "Vision",       "tasks": [t.value for t in [
            Task.IMAGE_CLASSIFICATION, Task.IMAGE_SEGMENTATION, Task.OBJECT_DETECTION,
            Task.DEPTH_ESTIMATION, Task.IMAGE_TO_IMAGE, Task.IMAGE_TO_TEXT,
            Task.TEXT_TO_IMAGE, Task.UNCONDITIONAL_IMAGE_GEN,
            Task.ZERO_SHOT_IMAGE_CLASSIF, Task.KEYPOINT_DETECTION,
        ]]},
        {"label": "Video",        "tasks": [t.value for t in [
            Task.VIDEO_CLASSIFICATION, Task.TEXT_TO_VIDEO, Task.VIDEO_TO_VIDEO, Task.IMAGE_TO_VIDEO,
        ]]},
        {"label": "Audio",        "tasks": [t.value for t in [
            Task.AUDIO_CLASSIFICATION, Task.AUTOMATIC_SPEECH_RECOGNITION,
            Task.TEXT_TO_SPEECH, Task.TEXT_TO_AUDIO, Task.AUDIO_TO_AUDIO,
            Task.VOICE_ACTIVITY_DETECTION,
        ]]},
        {"label": "Tabular",      "tasks": [t.value for t in [
            Task.TABULAR_CLASSIFICATION, Task.TABULAR_REGRESSION, Task.TIME_SERIES_FORECASTING,
        ]]},
        {"label": "Multi-modal",  "tasks": [t.value for t in [
            Task.VISUAL_QUESTION_ANSWERING, Task.DOCUMENT_QUESTION_ANSWERING,
            Task.IMAGE_TEXT_TO_TEXT, Task.VIDEO_TEXT_TO_TEXT, Task.AUDIO_TEXT_TO_TEXT,
            Task.ANY_TO_ANY, Task.TABLE_QUESTION_ANSWERING,
        ]]},
        {"label": "Other",        "tasks": [t.value for t in [
            Task.CLASSIFICATION, Task.REGRESSION, Task.REINFORCEMENT_LEARNING,
            Task.ROBOTICS, Task.CUSTOM,
        ]]},
    ]
