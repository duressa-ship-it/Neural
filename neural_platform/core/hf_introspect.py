"""
Lightweight HuggingFace dataset introspection.

Pure-Python — no torch, no datasets-package required (the schema-summary
helper duck-types its way through the features mapping). Used by:

  * `core/validator.py`   — to flag bad model/dataset modality combos at
                             config time, before any subprocess starts.
  * `data/loader.py`      — to pick the right Dataset wrapper (image vs text)
                             at dataloader-build time.
  * `cli/commands.py`     — `neural inspect <name>` formats this for humans.
  * `web/app.py`          — `/api/hf/inspect` exposes this to the dashboard.

Keeping this in `core/` (instead of `data/`) means the validator can call
into it without dragging torch into the import graph.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def inspect_features(features) -> Dict[str, Any]:
    """
    Categorize a HuggingFace `Features` mapping by inferred modality.

    Tolerates the `datasets` package being absent — it duck-types every
    field, only relying on:
      - `type(feat).__name__`           — to spot Image/ClassLabel features
      - `feat.names`                     — for ClassLabel
      - `feat.dtype`                     — for Value(string), Value(int*), etc.

    Returns a dict with these keys:
      columns          list[str]      — every column name, in order
      image_columns    list[str]      — columns of HF Image type
      text_columns     list[str]      — columns of Value(string) type
      label_columns    list[str]      — ClassLabel columns + numeric columns
                                         literally named "label" / "labels"
                                         / "class" / "target"
      numeric_columns  list[str]      — non-label numeric columns
      other_columns    list[str]      — everything else
      class_names      list[str]|None — ClassLabel.names from the first
                                         label column we found
      has_images       bool
      has_text         bool
    """
    if features is None:
        return _empty()

    out: Dict[str, Any] = {
        "columns":          list(features.keys()),
        "image_columns":    [],
        "text_columns":     [],
        "audio_columns":    [],
        "video_columns":    [],
        "sequence_columns": [],   # numeric Sequence(Value(int/float)) — time-series
        "label_columns":    [],
        "numeric_columns":  [],
        "other_columns":    [],
        "class_names":      None,
        "has_images":       False,
        "has_text":         False,
        "has_audio":        False,
        "has_video":        False,
        "has_sequence":     False,
    }

    for col, feat in features.items():
        repr_name = type(feat).__name__

        # Audio features — datasets.features.Audio
        if repr_name in ("Audio", "AudioFeature"):
            out["audio_columns"].append(col); out["has_audio"] = True
            continue

        # Video features — datasets.features.Video (newer)
        if repr_name in ("Video", "VideoFeature"):
            out["video_columns"].append(col); out["has_video"] = True
            continue

        # Image features
        if repr_name in ("Image", "ImageFeature"):
            out["image_columns"].append(col); out["has_images"] = True
            continue

        # ClassLabel features
        names = getattr(feat, "names", None)
        if names:
            out["label_columns"].append(col)
            if out["class_names"] is None:
                out["class_names"] = list(names)
            continue

        dtype = getattr(feat, "dtype", None)

        # Value(string) → text
        if dtype == "string":
            out["text_columns"].append(col); out["has_text"] = True
            continue

        # Value(int*/float*) → numeric
        if dtype and (dtype.startswith("int") or dtype.startswith("float")):
            out["numeric_columns"].append(col)
            continue

        # Sequence — could be text-of-tokens, numeric time-series, or list-of-strings
        if repr_name == "Sequence":
            inner = getattr(feat, "feature", None)
            inner_dtype = getattr(inner, "dtype", None) if inner else None
            if inner_dtype == "string":
                out["text_columns"].append(col); out["has_text"] = True
                continue
            if inner_dtype and (inner_dtype.startswith("int") or inner_dtype.startswith("float")):
                out["sequence_columns"].append(col); out["has_sequence"] = True
                continue
            # Sequence of images → frame stack ≈ video
            if type(inner).__name__ in ("Image", "ImageFeature"):
                out["video_columns"].append(col); out["has_video"] = True
                continue

        # Array2D / Array3D / Array4D / Array5D — NumPy-shaped tensors
        if repr_name in ("Array2D", "Array3D", "Array4D", "Array5D"):
            shape = getattr(feat, "shape", None)
            # Heuristic: 3+ dims with a small-channel-like first/last dim ≈ image-ish
            if shape and len(shape) >= 2:
                # 4D arrays often represent volumetric / 3D point clouds
                if len(shape) == 4:
                    out["video_columns"].append(col); out["has_video"] = True
                else:
                    out["other_columns"].append(col)
                continue

        out["other_columns"].append(col)

    # Numeric columns whose *name* implies "this is a label" should be
    # promoted out of generic numeric_columns. We accept exact matches
    # and a few common prefix patterns ("label_*", "target_*", "class_*").
    _label_promotion(out)

    # Detect "sequence-shaped tabular" via column-name patterns. Datasets
    # like recommendation logs or session features often expose 100+ columns
    # named `domain_a_seq_0..N`, `feat_seq_*`, `*_history`, etc. Group
    # those into `pattern_sequence_groups` so the loader can stack them
    # into a (T, F) tensor without the user spelling each one out.
    out["pattern_sequence_groups"] = _group_sequence_patterns(out["numeric_columns"])

    out["label_columns"] = list(dict.fromkeys(out["label_columns"]))
    return out


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

# Tokens that, when present anywhere in a column name, mean "this is a label".
_LABEL_TOKENS = ("label", "labels", "class", "target", "y_true", "ground_truth")
# Prefixes / suffixes that indicate a label.
_LABEL_PREFIXES = ("label_", "target_", "class_", "y_")
_LABEL_SUFFIXES = ("_label", "_target", "_class")


def _label_promotion(out: Dict[str, Any]) -> None:
    promoted = []
    for col in list(out["numeric_columns"]):
        lower = col.lower()
        if (
            lower in _LABEL_TOKENS
            or lower in ("y", "label")
            or any(lower.startswith(p) for p in _LABEL_PREFIXES)
            or any(lower.endswith(s) for s in _LABEL_SUFFIXES)
        ):
            promoted.append(col)
    for col in promoted:
        out["numeric_columns"].remove(col)
        if col not in out["label_columns"]:
            out["label_columns"].append(col)


_SEQ_PATTERN = re.compile(r"^(?P<prefix>.+?)[_\-]?(?P<idx>\d+)$")

def _group_sequence_patterns(columns: List[str]) -> List[Dict[str, Any]]:
    """
    Cluster columns sharing a common prefix + numeric suffix.

    For each cluster of size >= 4, return:
      {"prefix": "domain_a_seq", "columns": ["domain_a_seq_38", ...], "length": 9}

    These clusters are excellent candidates for "treat me as a sequence" —
    the loader stacks the values into a (timesteps,) vector per row.
    """
    groups: Dict[str, List[str]] = {}
    for col in columns:
        m = _SEQ_PATTERN.match(col)
        if not m:
            continue
        prefix = m.group("prefix").rstrip("_-")
        groups.setdefault(prefix, []).append(col)
    out: List[Dict[str, Any]] = []
    for prefix, cols in groups.items():
        if len(cols) >= 4:
            cols_sorted = sorted(cols, key=lambda c: int(_SEQ_PATTERN.match(c).group("idx")))
            out.append({"prefix": prefix, "columns": cols_sorted, "length": len(cols_sorted)})
    return out


def inspect_dataset(hf_dataset) -> Dict[str, Any]:
    """Inspect a Dataset instance (any object with `.features`)."""
    return inspect_features(getattr(hf_dataset, "features", None))


def _empty() -> Dict[str, Any]:
    return {
        "columns": [], "image_columns": [], "text_columns": [], "audio_columns": [],
        "video_columns": [], "sequence_columns": [], "label_columns": [],
        "numeric_columns": [], "other_columns": [], "class_names": None,
        "has_images": False, "has_text": False, "has_audio": False,
        "has_video": False, "has_sequence": False,
    }


def detect_modality(schema: Dict[str, Any]) -> str:
    """Resolve a schema dict to a Modality string (see core.modality.Modality)."""
    from neural_platform.core.modality import detect_from_features
    return detect_from_features(schema).value


def parse_available_configs(err_text: str) -> List[str]:
    """
    Pull the available config names out of a `datasets` ValueError that
    looks like::

        Config name is missing. Please pick one among the available
        configs: ['asr', 'er', 'ic', 'ks', 'sd', 'si']

    Returns the list of configs, or [] if the error wasn't this shape.
    """
    if "Config" not in err_text or "missing" not in err_text:
        return []
    m = re.search(r"configs?:\s*\[([^\]]+)\]", err_text, flags=re.IGNORECASE)
    if not m:
        return []
    return re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))


def class_names_for(hf_dataset, label_col: Optional[str]) -> Optional[List[str]]:
    """Pull `ClassLabel.names` out of a dataset's features if available."""
    if not label_col:
        return None
    try:
        features = getattr(hf_dataset, "features", None) or {}
        feat = features.get(label_col)
        names = getattr(feat, "names", None)
        if names:
            return list(names)
    except Exception:
        pass
    return None
