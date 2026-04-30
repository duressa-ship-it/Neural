"""
Dataset × model compatibility harness.

What this catches:
  * Misconfigurations the validator should flag *before* a training subprocess
    is spawned — e.g. CNN paired with a text-only HF dataset.
  * Modality auto-detection regressions in `inspect_hf_features`.
  * Pre-flight `validate_config` regressions.

Run:
    pytest tests/test_dataset_compat.py -v

Most cases below use fixture-style fake HF datasets so the suite never has to
hit the network. There's a single optional `@pytest.mark.online` block at the
bottom for anyone who wants to verify against the real HF Hub.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import pytest

# We don't want this suite to require torch — keep imports lazy and use stubs
# for anything outside the validator/inspector logic.

from neural_platform.core.config import ExperimentConfig
from neural_platform.core.validator import validate_config
from neural_platform.core.hf_introspect import inspect_dataset as inspect_hf_features


def _validate(cfg):
    """Validator wrapper used by tests. Skips the package-availability audit
    so the suite stays deterministic in minimal environments (CI, sandbox)
    that may lack optional deps like `transformers` or `torchaudio`."""
    return validate_config(cfg, check_deps=False)


# ---------------------------------------------------------------------------
# Tiny fake `datasets.features` types — duck-typed enough for our inspector
# ---------------------------------------------------------------------------

class Image:                          # mimics datasets.features.Image
    pass


class Value:                          # mimics datasets.features.Value
    def __init__(self, dtype: str): self.dtype = dtype


class ClassLabel:                     # mimics datasets.features.ClassLabel
    def __init__(self, names: List[str]): self.names = names


# Aliases the existing tests reference
_Image = Image
_Value = Value
_ClassLabel = ClassLabel


class _FakeFeatures(dict):
    """Behaves like datasets.Features (a regular dict mapping name → feature)."""
    pass


def make_fake_dataset(features: Dict[str, Any]) -> Any:
    """Return an object that quacks like a HuggingFace Dataset for inspection."""
    class _DS:
        def __init__(self, features):
            self.features = _FakeFeatures(features)
            self.column_names = list(features.keys())
        def __len__(self): return 0
    return _DS(features)


# ---------------------------------------------------------------------------
# Inspector tests — pure, no network
# ---------------------------------------------------------------------------

class TestFeatureInspection:
    def test_pure_image_dataset(self):
        ds = make_fake_dataset({"image": _Image()})
        info = inspect_hf_features(ds)
        assert info["image_columns"] == ["image"]
        assert info["text_columns"] == []
        assert info["label_columns"] == []

    def test_image_classification_dataset(self):
        ds = make_fake_dataset({
            "image": _Image(),
            "label": _ClassLabel(names=["cat", "dog", "bird"]),
        })
        info = inspect_hf_features(ds)
        assert info["image_columns"] == ["image"]
        assert info["label_columns"] == ["label"]
        assert info["class_names"] == ["cat", "dog", "bird"]

    def test_text_classification_dataset(self):
        ds = make_fake_dataset({
            "text": _Value("string"),
            "label": _ClassLabel(names=["neg", "pos"]),
        })
        info = inspect_hf_features(ds)
        assert info["text_columns"] == ["text"]
        assert info["label_columns"] == ["label"]
        assert info["class_names"] == ["neg", "pos"]

    def test_numeric_label_named_label_is_promoted(self):
        """Datasets with `label: Value(int64)` (no ClassLabel) still get categorized
        as a label column based on the column name."""
        ds = make_fake_dataset({
            "image": _Image(),
            "label": _Value("int64"),
        })
        info = inspect_hf_features(ds)
        assert "label" in info["label_columns"]
        assert "label" not in info["numeric_columns"]

    def test_unlabeled_image_dataset(self):
        ds = make_fake_dataset({"image": _Image(), "caption": _Value("string")})
        info = inspect_hf_features(ds)
        assert info["image_columns"] == ["image"]
        assert info["text_columns"] == ["caption"]
        assert info["label_columns"] == []

    def test_label_promotion_with_prefix(self):
        """Numeric columns named `label_type`, `label_time`, `target_x` should be
        promoted to label_columns even without ClassLabel features."""
        ds = make_fake_dataset({
            "feat_1":     _Value("float32"),
            "feat_2":     _Value("float32"),
            "label_type": _Value("int32"),     # prefix match
            "label_time": _Value("int64"),     # prefix match
            "target_y":   _Value("float32"),   # prefix match
            "user_class": _Value("int32"),     # not promoted (only suffix _class)
            "score_class": _Value("int32"),    # promoted (suffix _class)
        })
        info = inspect_hf_features(ds)
        assert "label_type" in info["label_columns"]
        assert "label_time" in info["label_columns"]
        assert "target_y" in info["label_columns"]
        assert "score_class" in info["label_columns"]
        # Plain numeric features stay in numeric_columns
        assert "feat_1" in info["numeric_columns"]
        assert "feat_2" in info["numeric_columns"]

    def test_sequence_pattern_grouping(self):
        """Datasets with `prefix_0..N` numeric columns should be clustered into
        pattern_sequence_groups for RNN/TCN consumption."""
        cols = {f"domain_a_seq_{i}": _Value("int32") for i in range(38, 47)}  # 9 cols
        cols.update({f"feat_{i}": _Value("float32") for i in range(3)})
        cols["label"] = _Value("int32")
        ds = make_fake_dataset(cols)
        info = inspect_hf_features(ds)
        groups = info["pattern_sequence_groups"]
        assert any(g["prefix"] == "domain_a_seq" and g["length"] == 9 for g in groups), groups
        # short groups (< 4) shouldn't be flagged
        assert all(g["length"] >= 4 for g in groups)


# ---------------------------------------------------------------------------
# Validator tests — checks that the right errors fire for known bad combos
# ---------------------------------------------------------------------------

def make_cfg(model_type: str, **data_overrides) -> ExperimentConfig:
    arch_block: Dict[str, Any] = {
        "mlp":         {"mlp": {"input_size": 16, "hidden_layers": [{"size": 32}], "output_size": 3}},
        "cnn":         {"cnn": {"input_channels": 3, "input_height": 32, "input_width": 32, "output_size": 10}},
        "rnn":         {"rnn": {"input_size": 8, "hidden_size": 32, "output_size": 3}},
        "transformer": {"transformer": {"output_size": 2}},
        "audio_cnn":   {"audio_cnn": {"output_size": 35}},
        "tcn":         {"tcn": {"input_size": 1, "output_size": 4}},
        "tabular":     {"tabular": {"output_size": 2,
                                     "numeric_features": ["a", "b"],
                                     "categorical_features": []}},
        "video_cnn":   {"video_cnn": {"output_size": 10}},
    }[model_type]
    base = {
        "name": "test",
        "model": {"type": model_type, "name": "m", "framework": "pytorch", **arch_block},
        "data": {"source": "synthetic", "synthetic_n_features": 16, "synthetic_n_classes": 3},
    }
    base["data"].update(data_overrides)
    return ExperimentConfig.model_validate(base)


class TestValidatorCoreCases:
    """The minimum set of errors the validator MUST catch before training."""

    def test_well_formed_mlp_synthetic(self):
        report = _validate(make_cfg("mlp"))
        assert report.ok, report.fmt()

    def test_huggingface_no_dataset_name(self):
        report = _validate(make_cfg("transformer", source="huggingface"))
        assert not report.ok
        assert any("dataset_name" in i.field for i in report.errors)

    def test_csv_no_path(self):
        report = _validate(make_cfg("mlp", source="csv"))
        assert not report.ok
        assert any(i.field == "data.path" for i in report.errors)

    def test_csv_no_target(self, tmp_path):
        f = tmp_path / "x.csv"; f.write_text("a,b\n1,2\n")
        report = _validate(make_cfg("mlp", source="csv", path=str(f)))
        assert any(i.field == "data.target_column" for i in report.errors)

    def test_image_folder_missing_path(self):
        report = _validate(make_cfg("cnn", source="image_folder"))
        assert any(i.field == "data.path" for i in report.errors)

    def test_regression_with_cross_entropy(self):
        cfg = make_cfg("mlp")
        cfg = ExperimentConfig.model_validate({
            **cfg.model_dump(),
            "training": {**cfg.training.model_dump(), "task": "regression", "loss": "cross_entropy"},
        })
        report = _validate(cfg)
        assert any(i.field == "training.loss" for i in report.errors)

    def test_from_scratch_transformer_with_synthetic_data(self):
        report = _validate(make_cfg("transformer"))
        assert any("transformer" in (i.message or "").lower() for i in report.errors)

    def test_tcn_validates(self):
        """TCN with valid block validates clean."""
        report = _validate(make_cfg("tcn"))
        assert report.ok, report.fmt()

    def test_tcn_rejects_no_channels(self):
        cfg = make_cfg("tcn")
        # Inject empty channels
        d = cfg.model_dump()
        d["model"]["tcn"]["channels"] = []
        cfg = ExperimentConfig.model_validate(d)
        report = _validate(cfg)
        assert any("channels" in i.field for i in report.errors), report.fmt()

    def test_tabular_rejects_no_features(self):
        cfg = make_cfg("tabular")
        d = cfg.model_dump()
        d["model"]["tabular"]["numeric_features"] = []
        d["model"]["tabular"]["categorical_features"] = []
        cfg = ExperimentConfig.model_validate(d)
        report = _validate(cfg)
        assert any("tabular" in i.field for i in report.errors), report.fmt()

    def test_audio_cnn_validates(self):
        """audio_cnn with default config validates."""
        report = _validate(make_cfg("audio_cnn"))
        # Some warnings ok (e.g. torchaudio not installed in test env), but no errors
        assert all(i.severity != "error" or "torchaudio" in i.message
                   for i in report.issues), report.fmt()

    def test_video_cnn_warns_experimental(self):
        report = _validate(make_cfg("video_cnn"))
        assert any("experimental" in i.message.lower() for i in report.warnings), report.fmt()


class TestDependencies:
    """Probe the new core.deps module."""

    def test_requirements_dedup(self):
        """torchvision is referenced by both `cnn` model and `image_folder`
        source — it shouldn't appear twice in the combined requirements list."""
        from neural_platform.core.deps import requirements_for
        deps = requirements_for("cnn", "image_folder")
        names = [d.package for d in deps]
        assert names.count("torchvision") == 1, names

    def test_audio_cnn_required_deps(self):
        from neural_platform.core.deps import requirements_for
        deps = requirements_for("audio_cnn", "huggingface")
        names = [d.package for d in deps]
        assert "torchaudio" in names
        assert "datasets" in names
        # torchcodec and soundfile are optional, so only required when checked
        torchcodec = next((d for d in deps if d.package == "torchcodec"), None)
        assert torchcodec is None or not torchcodec.required

    def test_check_dependencies_smoke(self):
        from neural_platform.core.deps import check_dependencies
        rep = check_dependencies("mlp", "synthetic")
        # Smoke: must return a DepReport-like with `ok` boolean
        assert hasattr(rep, "ok")
        assert hasattr(rep, "missing_required")

    def test_install_command_when_clean(self):
        from neural_platform.core.deps import DepReport, install_command
        empty = DepReport()
        assert install_command(empty) is None

    def test_format_report(self):
        from neural_platform.core.deps import check_dependencies, format_report
        out = format_report(check_dependencies("mlp"))
        assert isinstance(out, str)


_torch_available = True
try:
    import torch  # noqa: F401
except Exception:
    _torch_available = False


@pytest.mark.skipif(not _torch_available, reason="torch + numpy required for audio coerce tests")
class TestAudioWaveformCoercion:
    """Regression test for the music_genres ndarray-of-object bug."""

    def test_coerce_list_of_floats(self):
        from neural_platform.data.loader import _array_to_float_tensor
        import torch
        out = _array_to_float_tensor([0.1, 0.2, 0.3])
        assert out is not None
        assert out.dtype == torch.float32
        assert list(out.shape) == [3]

    def test_coerce_object_dtype_fallback(self):
        """numpy object array of plain floats — should still coerce."""
        from neural_platform.data.loader import _array_to_float_tensor
        import numpy as np
        import torch
        arr = np.array([0.1, 0.2, 0.3], dtype=object)
        out = _array_to_float_tensor(arr)
        assert out is not None
        assert out.dtype == torch.float32
        assert list(out.shape) == [3]

    def test_coerce_nested_object_arrays(self):
        """Object array containing per-channel arrays — flattens via per-elem cast."""
        from neural_platform.data.loader import _array_to_float_tensor
        import numpy as np
        arr = np.array([np.array([0.1, 0.2], dtype=np.float32),
                        np.array([0.3, 0.4], dtype=np.float32)], dtype=object)
        out = _array_to_float_tensor(arr)
        assert out is not None  # may be flat or 2D depending on path; we only assert tensor-ness
        assert out.numel() == 4

    def test_coerce_truly_non_numeric_returns_none(self):
        """When the items can't be cast at all, return None so the caller can
        fall back to the bytes/path decode path."""
        from neural_platform.data.loader import _array_to_float_tensor
        import numpy as np
        arr = np.array(["audio", "tag"], dtype=object)
        out = _array_to_float_tensor(arr)
        assert out is None


class TestKnownDatasetHeuristics:
    """The validator's offline fallback: name-based detection.

    These are run with `datasets.load_dataset_builder` patched to fail, so the
    validator falls back to the name list. That's the path used when the user
    doesn't have `datasets` installed or is offline.
    """

    @pytest.fixture(autouse=True)
    def disable_load_dataset_builder(self, monkeypatch):
        # Force the validator to fall back to its name-based heuristic
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("datasets disabled for this test")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", fake_import)

    def test_cnn_with_imdb_text_dataset(self):
        report = _validate(make_cfg(
            "cnn", source="huggingface", dataset_name="imdb", text_column="text", label_column="label",
        ))
        assert not report.ok
        # Validator should report a modality mismatch — exact wording: "Model
        # 'cnn' doesn't match dataset modality 'text'".
        assert any(
            "modality" in i.message.lower() and "cnn" in i.message
            for i in report.errors
        ), report.fmt()

    def test_transformer_with_cifar10(self):
        report = _validate(make_cfg(
            "transformer", source="huggingface", dataset_name="cifar10",
        ))
        # transformer + image dataset → modality mismatch error
        assert any(
            "modality" in i.message.lower() and "transformer" in i.message
            for i in report.errors
        ), report.fmt()

    def test_cnn_with_cifar10_is_clean(self):
        report = _validate(make_cfg(
            "cnn", source="huggingface", dataset_name="cifar10",
        ))
        # No modality mismatch errors should fire for the matched pair.
        modality_errors = [i for i in report.errors if "modality" in i.message.lower()]
        assert not modality_errors, report.fmt()


# ---------------------------------------------------------------------------
# Online sanity (skipped by default — only run when explicitly requested)
# ---------------------------------------------------------------------------

@pytest.mark.skipif("not config.getoption('--online', default=False)",
                    reason="needs --online to hit the HF Hub")
class TestOnlineHF:
    def test_imdb_features(self):
        from datasets import load_dataset_builder
        builder = load_dataset_builder("imdb")
        from neural_platform.core.validator import _features_summary
        summary = _features_summary(builder.info.features)
        assert summary["has_text"]
        assert "text" in summary["text_columns"]


def pytest_addoption(parser):  # pytest hook so --online flag is available
    parser.addoption("--online", action="store_true", default=False,
                     help="Run online tests that hit the HuggingFace Hub")
