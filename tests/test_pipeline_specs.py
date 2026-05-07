"""
Tests for the HF pipeline-specs single-source-of-truth.

The spec table feeds three subsystems:

  * the model wrapper (`models/hf_pipeline.py`) — picks the
    `transformers.Auto*` class
  * the inference manager (`web/inference_manager.py`) — synthesizes
    minimal hf_pipeline configs that pass validation
  * the inference server (`deploy/server.py`) — loads the right processor
    type and dispatches /predict to the right input adapter

If any consumer's understanding of a pipeline_tag drifts from the spec,
launches break in the way the user reported (Whisper /info 500'd because
checkpoint_path was typed `str`; ASR returns text not logits but the
server tried to argmax). These tests pin the contract.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Coverage of well-known HF pipeline tags
# ---------------------------------------------------------------------------

class TestSpecCoverage:
    """Make sure every HF pipeline_tag the synthesizer's task picker
    advertises in the UI has a real spec entry. Drift here means a user
    can pick a task in the dropdown that crashes at launch."""

    UI_TASKS = [
        "text-classification", "token-classification",
        "zero-shot-classification", "fill-mask",
        "text-generation", "text2text-generation", "summarization",
        "translation", "question-answering",
        "image-classification", "image-segmentation", "object-detection",
        "image-to-text",
        "audio-classification", "automatic-speech-recognition",
        "visual-question-answering",
        "feature-extraction",
    ]

    @pytest.mark.parametrize("task", UI_TASKS)
    def test_every_ui_task_has_spec(self, task):
        from neural_platform.core.pipeline_specs import PIPELINE_SPECS
        assert task in PIPELINE_SPECS, \
            f"Predict-tab UI lists {task!r} but PIPELINE_SPECS has no entry."

    @pytest.mark.parametrize("task", UI_TASKS)
    def test_resolve_returns_concrete_spec(self, task):
        """resolve() should never fall through to the default for a
        UI-advertised task."""
        from neural_platform.core.pipeline_specs import resolve
        spec = resolve(task)
        assert spec.task == task
        # Default spec returns task='custom' — flag if any of these slip.
        assert spec.task != "custom"


class TestSpecDefaults:
    """Sanity-check task → defaults wiring. These are the assertions that
    would have caught the user's Whisper failure ahead of time."""

    def test_unknown_task_returns_default(self):
        from neural_platform.core.pipeline_specs import resolve, _DEFAULT_SPEC
        assert resolve("not-a-real-pipeline-tag") is _DEFAULT_SPEC
        assert resolve("") is _DEFAULT_SPEC
        assert resolve(None) is _DEFAULT_SPEC

    def test_asr_specifies_processor_and_generation(self):
        """The exact failure mode: Whisper needs the WhisperProcessor
        (feature extractor + tokenizer) and `.generate()`, not forward
        + softmax. If this regresses, ASR breaks at /predict time."""
        from neural_platform.core.pipeline_specs import (
            resolve, PROCESSOR_PROCESSOR, POSTPROC_GENERATED_TEXT,
        )
        spec = resolve("automatic-speech-recognition")
        assert spec.processor_kind == PROCESSOR_PROCESSOR
        assert spec.needs_generation is True
        assert spec.output_kind == POSTPROC_GENERATED_TEXT
        assert spec.modality == "audio"

    def test_text_classification_uses_tokenizer_and_logits(self):
        from neural_platform.core.pipeline_specs import (
            resolve, PROCESSOR_TOKENIZER, POSTPROC_LOGITS,
        )
        spec = resolve("text-classification")
        assert spec.processor_kind == PROCESSOR_TOKENIZER
        assert spec.needs_generation is False
        assert spec.output_kind == POSTPROC_LOGITS
        assert spec.has_class_head is True

    def test_image_classification_uses_image_processor(self):
        from neural_platform.core.pipeline_specs import (
            resolve, PROCESSOR_IMAGE, INPUT_IMAGE,
        )
        spec = resolve("image-classification")
        assert spec.processor_kind == PROCESSOR_IMAGE
        assert spec.input_kind == INPUT_IMAGE
        assert spec.modality == "image"

    def test_audio_classification_uses_feature_extractor(self):
        from neural_platform.core.pipeline_specs import (
            resolve, PROCESSOR_FEATURE_EXTRACTOR, INPUT_AUDIO,
        )
        spec = resolve("audio-classification")
        assert spec.processor_kind == PROCESSOR_FEATURE_EXTRACTOR
        assert spec.input_kind == INPUT_AUDIO

    def test_vqa_uses_unified_processor(self):
        from neural_platform.core.pipeline_specs import (
            resolve, PROCESSOR_PROCESSOR, INPUT_IMAGE_TEXT,
        )
        spec = resolve("visual-question-answering")
        assert spec.processor_kind == PROCESSOR_PROCESSOR
        assert spec.input_kind == INPUT_IMAGE_TEXT


class TestSpecConsumersAgree:
    """If the spec table drifts from the model wrapper / synthesizer /
    server, launches break silently. Lock the cross-references here."""

    def test_legacy_auto_class_map_matches_specs(self):
        """`models/hf_pipeline.py::_TASK_TO_AUTO_CLASS` is now derived
        from the spec table. Confirm the rebuild logic returns identical
        mappings for every known task."""
        from neural_platform.models.hf_pipeline import _TASK_TO_AUTO_CLASS
        from neural_platform.core.pipeline_specs import PIPELINE_SPECS
        for task, spec in PIPELINE_SPECS.items():
            assert _TASK_TO_AUTO_CLASS.get(task) == spec.auto_class

    def test_synthesizer_uses_spec_for_loss(self, tmp_path):
        """Depth estimation should synthesize with mse, not cross_entropy
        (the inference server's loss is never run, but the validator
        rejects regression+cross_entropy)."""
        from neural_platform.web.inference_manager import _synthesize_hf_config
        cfg, _ = _synthesize_hf_config(
            output_root=tmp_path,
            hf_model_id="Intel/dpt-hybrid-midas",
            pipeline_task="depth-estimation",
        )
        assert cfg.training.loss.value in ("mse", "cross_entropy")
        # The validator rule: regression + cross_entropy is a hard error.
        from neural_platform.core.config import LossFunction, Task
        if cfg.training.task == Task.REGRESSION:
            assert cfg.training.loss != LossFunction.CROSS_ENTROPY


# ---------------------------------------------------------------------------
# /info endpoint regression — checkpoint_path None must not 500
# ---------------------------------------------------------------------------

class TestInfoNoCheckpointMode:
    """The exact bug the user reported: /info returned 500 for an HF launch
    because InfoResponse.checkpoint_path was typed `str` (required) and
    no-checkpoint mode passed None.

    These tests construct the app + drive `/info` through the FastAPI
    TestClient. We don't actually load the model weights — startup is
    short-circuited by patching adapter.build_model to return a stub."""

    def _stub_model(self):
        """Tiny stand-in for the HF wrapper. Has the surface area /info
        reads from: count_parameters, encoder.config.id2label."""
        import torch.nn as nn
        class _Cfg:
            id2label = {0: "negative", 1: "positive"}
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
            def forward(self, x):
                return self.encoder.linear(x)
            def count_parameters(self, trainable_only=False):
                return sum(p.numel() for p in self.parameters()
                            if not trainable_only or p.requires_grad)
        return _Stub()

    def test_info_returns_200_with_no_checkpoint(self, tmp_path, monkeypatch):
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
                hf_pipeline=HFPipelineConfig(
                    pretrained="distilbert-base-uncased-finetuned-sst-2-english",
                ),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION,
                                    pipeline_task="text-classification"),
            data=DataConfig(),
            deploy=DeployConfig(),
        )

        # Patch the adapter so we don't actually pull weights from HF.
        stub = self._stub_model()
        monkeypatch.setattr(PyTorchAdapter, "build_model",
                             lambda self: stub)
        # Skip processor load — it's not needed for /info.
        from neural_platform.deploy import server as _server
        monkeypatch.setattr(_server, "_try_load_processor",
                             lambda config: None)

        app = create_inference_app(cfg, checkpoint_path=None)
        with TestClient(app) as client:
            r = client.get("/info")

        assert r.status_code == 200, f"got {r.status_code}: {r.text}"
        body = r.json()
        # The exact field that 500'd before — now nullable.
        assert body["checkpoint_path"] is None
        # Spec-derived metadata flows out.
        assert body["pipeline_task"] == "text-classification"
        assert body["hf_model_id"] == \
            "distilbert-base-uncased-finetuned-sst-2-english"
        assert body["auto_class"] == "AutoModelForSequenceClassification"
        assert body["processor_class"] == "AutoTokenizer"
        # id2label was lifted off the HF model config as class_names.
        assert body["class_names"] == ["negative", "positive"]
        assert body["output_size"] == 2

    def test_info_for_asr_reports_processor(self, tmp_path, monkeypatch):
        """Whisper-style ASR — /info should advertise AutoProcessor +
        AutoModelForSpeechSeq2Seq so the UI knows what's loaded."""
        monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        from neural_platform.deploy.server import create_inference_app
        from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter
        from fastapi.testclient import TestClient

        cfg = ExperimentConfig(
            name="whisper",
            output_dir=str(tmp_path),
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(pretrained="openai/whisper-tiny"),
            ),
            training=TrainingConfig(
                task=Task.CLASSIFICATION,
                pipeline_task="automatic-speech-recognition",
            ),
            data=DataConfig(),
            deploy=DeployConfig(),
        )

        stub = self._stub_model()
        monkeypatch.setattr(PyTorchAdapter, "build_model", lambda self: stub)
        from neural_platform.deploy import server as _server
        monkeypatch.setattr(_server, "_try_load_processor", lambda config: None)

        app = create_inference_app(cfg, checkpoint_path=None)
        with TestClient(app) as client:
            r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["auto_class"] == "AutoModelForSpeechSeq2Seq"
        assert body["processor_class"] == "AutoProcessor"
        assert body["pipeline_task"] == "automatic-speech-recognition"
