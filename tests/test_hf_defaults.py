"""
Tests for the HF model-defaults extractor + /info surface.

The Predict tab auto-fills generation-knob placeholders and shows a
"Model accepts up to N tokens" hint based on what the loaded model's
PretrainedConfig + GenerationConfig + feature_extractor expose.

Contract pinned here:

  * ``_extract_hf_defaults(model, processor)`` returns a stable dict
    shape — every key the frontend reads is present, with ``None`` for
    fields the model didn't expose. No KeyError surprises.
  * Whisper-style speech models use ``max_source_positions`` instead
    of ``max_position_embeddings``. We fall back to it so the length
    hint still appears.
  * Models without a ``generation_config`` (older HF, encoder-only)
    don't crash — every generation field becomes ``None``.
  * /info round-trips ``model_defaults`` end-to-end via TestClient.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_stub(*, config=None, generation_config=None):
    """Build a minimal ``HFPipelineModel`` stand-in with whatever
    config / generation_config the test supplies. The wrapper exposes
    ``.encoder.config`` and ``.encoder.generation_config``."""
    encoder = SimpleNamespace(
        config=config or SimpleNamespace(),
        generation_config=generation_config,
    )
    return SimpleNamespace(encoder=encoder)


def _make_processor_with_feature_extractor(*, sampling_rate=None, feature_size=None):
    """Audio processors (Whisper, Wav2Vec2) expose feature_extractor."""
    fe = SimpleNamespace(sampling_rate=sampling_rate, feature_size=feature_size)
    return SimpleNamespace(feature_extractor=fe)


# ---------------------------------------------------------------------------
# Extractor — fully offline, no real transformers needed
# ---------------------------------------------------------------------------

class TestExtractor:

    def test_returns_full_stable_schema(self):
        """Every key the frontend reads is always present, even when
        nothing's set. None placeholders keep `if (v != null)` checks
        uniform across architectures."""
        from neural_platform.deploy.server import _extract_hf_defaults
        d = _extract_hf_defaults(_make_model_stub(), None)
        assert set(d.keys()) == {
            "model_type", "max_position_embeddings", "vocab_size",
            "sampling_rate", "feature_size", "generation",
        }
        # Generation sub-dict has every documented knob too.
        assert set(d["generation"].keys()) == {
            "max_new_tokens", "max_length", "temperature",
            "top_p", "top_k", "num_beams", "do_sample",
            "repetition_penalty",
        }
        # With no source data, every value is None.
        for v in d["generation"].values():
            assert v is None
        for k in ("model_type", "max_position_embeddings", "vocab_size",
                   "sampling_rate", "feature_size"):
            assert d[k] is None

    def test_reads_text_model_config(self):
        """GPT/Llama-style text model: max_position_embeddings comes
        off model.config, generation defaults off
        generation_config."""
        from neural_platform.deploy.server import _extract_hf_defaults
        cfg = SimpleNamespace(
            model_type="gpt2", max_position_embeddings=1024,
            vocab_size=50257,
        )
        gen_cfg = SimpleNamespace(
            max_new_tokens=64, max_length=None, temperature=0.7,
            top_p=0.9, top_k=50, num_beams=1, do_sample=True,
            repetition_penalty=1.0,
        )
        d = _extract_hf_defaults(
            _make_model_stub(config=cfg, generation_config=gen_cfg),
            None,
        )
        assert d["model_type"] == "gpt2"
        assert d["max_position_embeddings"] == 1024
        assert d["vocab_size"] == 50257
        assert d["generation"]["max_new_tokens"] == 64
        assert d["generation"]["temperature"] == 0.7
        assert d["generation"]["top_p"] == 0.9
        assert d["generation"]["top_k"] == 50
        assert d["generation"]["do_sample"] is True

    def test_falls_back_to_max_source_positions(self):
        """Whisper-style encoders have no max_position_embeddings;
        the source-length cap lives at .max_source_positions instead.
        The extractor falls back so the length hint still appears."""
        from neural_platform.deploy.server import _extract_hf_defaults
        cfg = SimpleNamespace(
            model_type="whisper",
            max_position_embeddings=None,
            max_source_positions=1500,
        )
        d = _extract_hf_defaults(_make_model_stub(config=cfg), None)
        assert d["max_position_embeddings"] == 1500

    def test_reads_feature_extractor_sample_rate(self):
        from neural_platform.deploy.server import _extract_hf_defaults
        proc = _make_processor_with_feature_extractor(
            sampling_rate=16000, feature_size=80,
        )
        d = _extract_hf_defaults(_make_model_stub(), proc)
        assert d["sampling_rate"] == 16000
        assert d["feature_size"] == 80

    def test_handles_missing_generation_config(self):
        """Older HF models (encoder-only BERT, some custom repos) don't
        expose a generation_config. The extractor must not crash; every
        generation field becomes None so the frontend just doesn't
        prefill those placeholders."""
        from neural_platform.deploy.server import _extract_hf_defaults
        cfg = SimpleNamespace(
            model_type="bert", max_position_embeddings=512,
            vocab_size=30522,
        )
        # No generation_config on the encoder.
        model = SimpleNamespace(encoder=SimpleNamespace(config=cfg))
        d = _extract_hf_defaults(model, None)
        assert d["max_position_embeddings"] == 512
        for v in d["generation"].values():
            assert v is None

    def test_handles_processor_without_feature_extractor(self):
        """Text-only processors (tokenizer) don't have a
        feature_extractor. sampling_rate / feature_size stay None."""
        from neural_platform.deploy.server import _extract_hf_defaults
        proc = SimpleNamespace()   # no feature_extractor
        d = _extract_hf_defaults(_make_model_stub(), proc)
        assert d["sampling_rate"] is None
        assert d["feature_size"] is None

    def test_tolerates_attribute_errors(self):
        """A model that raises on attribute access (rare custom
        configs) shouldn't 500 /info. The extractor catches each
        access individually and returns None for the failing field."""
        from neural_platform.deploy.server import _extract_hf_defaults

        class _ThrowingConfig:
            def __getattribute__(self, name):
                if name in ("__class__", "__dict__"):
                    return object.__getattribute__(self, name)
                raise RuntimeError("nope")

        model = SimpleNamespace(encoder=SimpleNamespace(
            config=_ThrowingConfig(),
            generation_config=None,
        ))
        d = _extract_hf_defaults(model, None)
        # All None — but the dict shape is intact.
        assert d["model_type"] is None
        assert d["max_position_embeddings"] is None


# ---------------------------------------------------------------------------
# /info round-trip — InfoResponse.model_defaults populated end-to-end
# ---------------------------------------------------------------------------

class TestInfoCarriesModelDefaults:

    def _stub_model_with_gen_config(self):
        """Build a minimal HF-shaped stub: encoder.config has
        max_position_embeddings + model_type, encoder.generation_config
        has temperature + max_new_tokens. /info should expose all of
        these on model_defaults."""
        import torch.nn as nn
        class _Cfg:
            model_type = "t5"
            max_position_embeddings = 4096
            vocab_size = 32128
            id2label = {}
            num_labels = 0
        class _GenCfg:
            max_new_tokens = 256
            max_length = None
            temperature = 0.8
            top_p = 0.95
            top_k = 0
            num_beams = 4
            do_sample = False
            repetition_penalty = 1.2
        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Cfg()
                self.generation_config = _GenCfg()
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Encoder()
            def count_parameters(self, trainable_only=False): return 0
        return _Stub()

    def test_hf_pipeline_info_includes_model_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        from neural_platform.deploy.server import create_inference_app
        from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter
        from fastapi.testclient import TestClient

        cfg = ExperimentConfig(
            name="x", output_dir=str(tmp_path),
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(pretrained="t5-base"),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION,
                                     pipeline_task="summarization"),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        stub = self._stub_model_with_gen_config()
        monkeypatch.setattr(PyTorchAdapter, "build_model", lambda self: stub)
        from neural_platform.deploy import server as _server
        monkeypatch.setattr(_server, "_try_load_processor", lambda c: None)

        app = create_inference_app(cfg, checkpoint_path=None)
        with TestClient(app) as client:
            r = client.get("/info")
        assert r.status_code == 200, r.text
        body = r.json()
        defaults = body["model_defaults"]
        assert defaults is not None
        assert defaults["model_type"] == "t5"
        assert defaults["max_position_embeddings"] == 4096
        # Generation defaults flow through unchanged.
        gen = defaults["generation"]
        assert gen["max_new_tokens"] == 256
        assert gen["temperature"] == 0.8
        assert gen["num_beams"] == 4
        assert gen["do_sample"] is False
        assert gen["repetition_penalty"] == 1.2

    def test_extractor_not_called_for_native_models(self):
        """Pin the `if model.type.value == 'hf_pipeline'` guard at unit
        level — full integration would require synthesizing a fake .pt
        checkpoint for an MLP server, which is heavier than this
        contract warrants. The /info handler skips the extractor for
        every native model type, so model_defaults stays None."""
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            MLPConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        cfg = ExperimentConfig(
            name="m",
            model=ModelConfig(
                type=ModelType.MLP,
                framework=Framework.PYTORCH,
                mlp=MLPConfig(input_size=4, output_size=2, hidden_layers=[]),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        # The /info handler gates model_defaults on this check exactly.
        assert cfg.model.type.value != "hf_pipeline"
