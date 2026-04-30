"""
NeuralForge Modality System

A "modality" is what kind of data a model consumes. Mapping HuggingFace
dataset features → modalities → recommended model architectures is the
backbone of the Builder UI, the validator, and the data loader's
dispatch logic.

The 9 modalities here mirror the HuggingFace Hub's `modality=…` filter:
https://huggingface.co/datasets

Modality is *separate* from model type. A single modality can be served
by several model types (e.g. TIME_SERIES → RNN or TCN), and a single
model type can consume several modalities (e.g. CNN → IMAGE or DOCUMENT).
"""

from __future__ import annotations

from enum import Enum
from typing import List


class Modality(str, Enum):
    """High-level data modality. The string values match HF Hub conventions."""
    TABULAR        = "tabular"
    IMAGE          = "image"
    TEXT           = "text"
    AUDIO          = "audio"
    VIDEO          = "video"
    TIME_SERIES    = "time_series"
    DOCUMENT       = "document"        # OCR'd or layout-aware (often image+text)
    GEOSPATIAL     = "geospatial"      # raster + vector — usually multi-band imagery
    POINT_CLOUD_3D = "point_cloud_3d"  # LiDAR / 3D scans / mesh
    UNKNOWN        = "unknown"


# ---------------------------------------------------------------------------
# Modality → recommended model types
# ---------------------------------------------------------------------------

# Each modality lists model_type strings (matching ModelType enum values)
# that work with that modality, ordered from most-recommended to alternatives.
MODALITY_MODELS: dict[Modality, List[str]] = {
    Modality.TABULAR:        ["tabular", "mlp", "hf_pipeline"],
    Modality.IMAGE:          ["cnn", "hf_pipeline"],
    Modality.TEXT:           ["transformer", "hf_pipeline", "rnn"],
    Modality.AUDIO:          ["audio_cnn", "hf_pipeline", "rnn"],
    Modality.VIDEO:          ["video_cnn", "hf_pipeline"],
    Modality.TIME_SERIES:    ["tcn", "rnn", "hf_pipeline"],
    Modality.DOCUMENT:       ["hf_pipeline", "transformer", "cnn"],
    Modality.GEOSPATIAL:     ["cnn", "hf_pipeline"],
    Modality.POINT_CLOUD_3D: ["hf_pipeline"],
    Modality.UNKNOWN:        ["hf_pipeline", "mlp"],
}


# Which model types are fully implemented vs. experimental.
# The Builder UI uses this to show a yellow "experimental" badge.
EXPERIMENTAL_MODELS = {"video_cnn"}
UNIMPLEMENTED_MODELS = set()  # filled at validator time when a model class isn't registered


def detect_from_features(schema: dict) -> Modality:
    """
    Map a `core.hf_introspect.inspect_features(...)` schema dict to a Modality.

    The dict keys we care about (schema is the rich version, after the
    modality detection patch in hf_introspect):
        image_columns, text_columns, audio_columns, video_columns,
        sequence_columns, label_columns, numeric_columns
    """
    # Order matters — we resolve the most specific first so a dataset with
    # both audio + label is AUDIO, not TABULAR.
    if schema.get("video_columns"):
        return Modality.VIDEO
    if schema.get("audio_columns"):
        return Modality.AUDIO
    if schema.get("image_columns") and not schema.get("text_columns"):
        return Modality.IMAGE
    if schema.get("image_columns") and schema.get("text_columns"):
        # Image + text in one row strongly implies document-AI or VQA-ish data.
        return Modality.DOCUMENT
    if schema.get("text_columns"):
        return Modality.TEXT
    # 3D / geospatial features aren't first-class HF feature types yet — fall
    # through to heuristics in the inspector.
    if schema.get("sequence_columns"):
        return Modality.TIME_SERIES
    if schema.get("numeric_columns") or schema.get("label_columns"):
        return Modality.TABULAR
    return Modality.UNKNOWN


def recommend_model(modality: Modality) -> str:
    """First recommended model_type string for a modality."""
    options = MODALITY_MODELS.get(modality, [])
    return options[0] if options else "mlp"


def model_supports(model_type: str, modality: Modality) -> bool:
    """Is `model_type` a sane choice for `modality`?"""
    return model_type in MODALITY_MODELS.get(modality, [])
