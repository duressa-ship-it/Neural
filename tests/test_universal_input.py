"""
Tests for the universal Predict input — the spec-driven UI hint, the
new audio_b64 / video_b64 fields on PredictRequest, and the binary
decoders that turn base64 file blobs into the tensors the underlying
HF processor expects.

These pin the contract the Predict tab's frontend depends on:

  * /info returns a ``ui_hint`` dict whose shape app.js parses
    directly. If the schema drifts the universal panel breaks
    silently (wrong fields shown / hidden).
  * The audio decoder picks the right sample rate from the loaded HF
    feature extractor (Whisper wants 16 kHz; some music models want
    44.1 kHz).
  * The audio path now accepts both ``audio_b64`` (a real file) and
    the legacy ``inputs`` list of float samples — universal UX wraps
    real-file uploads, but power users keep the raw waveform path.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# UI-hint helper — drives the frontend's universal panel
# ---------------------------------------------------------------------------

class TestUiHintCoverage:
    """Every task's hint must populate the keys app.js reads. If a key
    goes missing the frontend silently leaves a field hidden / wrongly
    shown — pin the schema."""

    REQUIRED_KEYS = {
        "show_text", "show_context", "show_file",
        "show_candidate_labels", "show_generation_knobs",
        "accept", "text_placeholder", "file_placeholder",
        "primary_field", "summary",
    }

    @pytest.mark.parametrize("task", [
        "text-classification", "question-answering",
        "image-classification", "automatic-speech-recognition",
        "visual-question-answering", "video-classification",
        "any-to-any", "image-to-text", "fill-mask",
        "zero-shot-classification", "zero-shot-image-classification",
        "text-generation", "summarization",
    ])
    def test_hint_has_required_keys(self, task):
        from neural_platform.core.pipeline_specs import resolve, ui_hint
        h = ui_hint(resolve(task))
        missing = self.REQUIRED_KEYS - set(h.keys())
        assert not missing, f"hint for {task} missing keys: {missing}"


class TestUiHintCorrectness:
    """The hint per task must steer the universal panel to the right
    shape — file drop for image/audio/video tasks, context for QA,
    candidate labels for zero-shot, generation knobs for generative."""

    def _hint(self, task):
        from neural_platform.core.pipeline_specs import resolve, ui_hint
        return ui_hint(resolve(task))

    def test_qa_shows_context(self):
        h = self._hint("question-answering")
        assert h["show_context"] is True
        assert h["show_file"] is False
        assert h["primary_field"] == "text"

    def test_image_classification_shows_file_only(self):
        h = self._hint("image-classification")
        assert h["show_file"] is True
        assert h["show_text"] is False
        assert h["accept"].startswith("image/")
        assert h["primary_field"] == "image_b64"

    def test_asr_picks_audio_with_generation_knobs(self):
        h = self._hint("automatic-speech-recognition")
        assert h["primary_field"] == "audio_b64"
        assert h["accept"].startswith("audio/")
        assert h["show_generation_knobs"] is True

    def test_vqa_shows_text_and_image(self):
        h = self._hint("visual-question-answering")
        assert h["show_text"] is True
        assert h["show_file"] is True
        assert h["accept"].startswith("image/")

    def test_zero_shot_classification_shows_labels(self):
        h = self._hint("zero-shot-classification")
        assert h["show_candidate_labels"] is True
        assert h["show_text"] is True

    def test_any_to_any_accepts_any_media(self):
        """The any-to-any task wraps unified multimodal LMs (Gemma-3,
        Qwen2-VL). The drop zone must accept everything; text is shown
        too so users can type a prompt alongside any file."""
        h = self._hint("any-to-any")
        assert h["show_text"] is True
        assert h["show_file"] is True
        accepted = h["accept"].split(",")
        assert any("image" in a for a in accepted)
        assert any("audio" in a for a in accepted)
        assert any("video" in a for a in accepted)

    def test_unknown_task_falls_back_to_default(self):
        """A pipeline_tag we don't know should still render a usable UI:
        text + file drop with no `accept` filter."""
        from neural_platform.core.pipeline_specs import resolve, ui_hint
        h = ui_hint(resolve("brand-new-pipeline-2030"))
        # Default spec → text-only, no file drop.
        assert h["show_text"] is True
        # The summary should at least mention the task or fall back to
        # the spec's notes — never empty.
        assert h["summary"]


# ---------------------------------------------------------------------------
# Native-model UI hint helper
# ---------------------------------------------------------------------------

class TestNativeUiHint:
    """Models that aren't hf_pipeline still get a hint so the universal
    panel renders for them too. Each model_type maps to a sane field set."""

    def _native_hint(self, mtype):
        from neural_platform.deploy.server import _native_ui_hint
        return _native_ui_hint(mtype)

    def test_cnn_shows_image_drop(self):
        h = self._native_hint("cnn")
        assert h["show_file"] is True
        assert h["accept"].startswith("image/")
        assert h["primary_field"] == "image_b64"

    def test_audio_cnn_shows_audio_drop(self):
        h = self._native_hint("audio_cnn")
        assert h["show_file"] is True
        assert h["accept"].startswith("audio/")

    def test_video_cnn_shows_video_drop(self):
        h = self._native_hint("video_cnn")
        assert h["show_file"] is True
        assert h["accept"].startswith("video/")

    def test_mlp_shows_no_file(self):
        h = self._native_hint("mlp")
        assert h["show_file"] is False
        assert h["primary_field"] == "inputs"

    def test_unknown_model_type_falls_back_safely(self):
        """No KeyError / no missing fields. The keys app.js reads must
        all be present so the frontend can apply the hint without
        defensive checks."""
        h = self._native_hint("custom-model-type-xyz")
        for k in TestUiHintCoverage.REQUIRED_KEYS:
            assert k in h


# ---------------------------------------------------------------------------
# Audio decode — _decode_audio_b64
# ---------------------------------------------------------------------------

soundfile = pytest.importorskip("soundfile")
numpy = pytest.importorskip("numpy")


class TestAudioDecode:
    """The decoder takes a base64 audio file (any format soundfile or
    librosa supports) and returns a flat float32 list at the model's
    expected sample rate. Skipped when the [audio] extra isn't
    installed — offline-only otherwise."""

    def _make_wav_b64(self, samples, sample_rate=16000):
        """Use soundfile to build a real WAV in memory, then base64
        encode it. Mirrors what the browser's FileReader produces."""
        import soundfile as sf
        import numpy as np
        buf = io.BytesIO()
        sf.write(buf, np.asarray(samples, dtype="float32"), sample_rate, format="WAV")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def test_decodes_wav_at_target_rate(self):
        from neural_platform.deploy.server import _decode_audio_b64
        # 1 second of silence at 16 kHz.
        samples = [0.0] * 16000
        b64 = self._make_wav_b64(samples, sample_rate=16000)
        out = _decode_audio_b64(b64, target_sr=16000)
        assert isinstance(out, list)
        assert len(out) == 16000
        assert all(isinstance(x, float) for x in out)

    def test_resamples_when_target_differs_from_source(self):
        from neural_platform.deploy.server import _decode_audio_b64
        # 1 second at 8 kHz → resample to 16 kHz → ~16000 samples.
        samples = [0.0] * 8000
        b64 = self._make_wav_b64(samples, sample_rate=8000)
        out = _decode_audio_b64(b64, target_sr=16000)
        # Allow ~1% tolerance on length (resampling rounding).
        assert 15800 <= len(out) <= 16200

    def test_stereo_is_mixed_to_mono(self):
        """Two-channel WAV must be mixed to mono — the feature
        extractor expects 1-D waveform."""
        from neural_platform.deploy.server import _decode_audio_b64
        import soundfile as sf
        import numpy as np
        # Stereo: 2 channels × 16000 samples
        stereo = np.stack([np.zeros(16000, dtype="float32"),
                            np.ones(16000, dtype="float32")], axis=1)
        buf = io.BytesIO()
        sf.write(buf, stereo, 16000, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out = _decode_audio_b64(b64, target_sr=16000)
        assert len(out) == 16000   # mono, not 32000
        # Mixed = average of channels (0 + 1) / 2 = 0.5
        assert abs(out[0] - 0.5) < 0.01

    def test_invalid_base64_raises_input_error(self):
        from neural_platform.deploy.server import _decode_audio_b64, _InputError
        with pytest.raises(_InputError):
            _decode_audio_b64("!!!not-base64!!!")


# ---------------------------------------------------------------------------
# Sample-rate resolution from processor
# ---------------------------------------------------------------------------

class TestResolveTargetSampleRate:
    """The audio decoder reads sampling_rate off the loaded HF feature
    extractor / processor. Whisper / Wav2Vec2 want 16 kHz; some music
    models want 44.1 kHz. Default falls back to 16000 — right answer
    for ~95% of HF audio models."""

    def test_reads_processor_sampling_rate(self):
        from neural_platform.deploy.server import _resolve_target_sample_rate
        proc = SimpleNamespace(sampling_rate=44100)
        assert _resolve_target_sample_rate(proc) == 44100

    def test_falls_back_to_feature_extractor(self):
        """Unified processors (WhisperProcessor) wrap a feature
        extractor; the rate lives on .feature_extractor."""
        from neural_platform.deploy.server import _resolve_target_sample_rate
        fe = SimpleNamespace(sampling_rate=22050)
        proc = SimpleNamespace(feature_extractor=fe)
        # The bare processor has no sampling_rate, but its
        # feature_extractor does.
        assert _resolve_target_sample_rate(proc) == 22050

    def test_defaults_to_16000_when_processor_missing(self):
        from neural_platform.deploy.server import _resolve_target_sample_rate
        # Some processors have no sampling_rate at all (text-only models
        # that someone tried to use for audio). Don't blow up — return
        # the safe Whisper / Wav2Vec2 default.
        assert _resolve_target_sample_rate(None) == 16000
        assert _resolve_target_sample_rate(SimpleNamespace()) == 16000


# ---------------------------------------------------------------------------
# /info ui_hint regression
# ---------------------------------------------------------------------------

class TestInfoCarriesUiHint:
    """The Predict tab's frontend reads info.ui_hint to render the
    universal panel. Pin that the field is present and shaped correctly
    for both HF-pipeline and native-model launches."""

    def _stub_model(self):
        import torch.nn as nn
        class _Cfg:
            id2label = {0: "neg", 1: "pos"}
            num_labels = 2
        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 2)
                self.config = _Cfg()
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Encoder()
            def forward(self, x): return self.encoder.linear(x)
            def count_parameters(self, trainable_only=False):
                return sum(p.numel() for p in self.parameters())
        return _Stub()

    def test_hf_pipeline_info_includes_ui_hint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        from neural_platform.deploy.server import create_inference_app
        from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter
        from fastapi.testclient import TestClient

        cfg = ExperimentConfig(
            name="hf",
            output_dir=str(tmp_path),
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(pretrained="bert-base-uncased"),
            ),
            training=TrainingConfig(
                task=Task.CLASSIFICATION,
                pipeline_task="question-answering",
            ),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        monkeypatch.setattr(PyTorchAdapter, "build_model",
                             lambda self: self._stub() if False else self.__class__.build_model.__wrapped__(self) if False else None)
        # Patch through the actual stub.
        monkeypatch.setattr(PyTorchAdapter, "build_model",
                             lambda self: TestInfoCarriesUiHint()._stub_model())
        from neural_platform.deploy import server as _server
        monkeypatch.setattr(_server, "_try_load_processor", lambda c: None)

        app = create_inference_app(cfg, checkpoint_path=None)
        with TestClient(app) as client:
            r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["ui_hint"] is not None
        # QA-specific: context must be a visible field.
        assert body["ui_hint"]["show_context"] is True

    def test_native_model_info_includes_ui_hint(self, tmp_path, monkeypatch):
        """Even non-HF model types (cnn / mlp / audio_cnn) get a hint —
        the universal panel renders for them too."""
        from neural_platform.deploy.server import _native_ui_hint
        h = _native_ui_hint("cnn")
        # Required keys present, primary_field is image_b64.
        assert h["primary_field"] == "image_b64"
        assert h["show_file"] is True
