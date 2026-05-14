"""
Tests for the quantization (bitsandbytes) plumbing.

The launch path:

  UI checkboxes
    → POST /api/inference/start_hf {load_in_4bit, load_in_8bit, bnb_compute_dtype}
    → InferenceServerManager.start_from_hf(...)
    → _synthesize_hf_config(...)
    → HFPipelineConfig(...)
    → HFPipelineModel.__init__ builds BitsAndBytesConfig
    → AutoClass.from_pretrained(..., quantization_config=…)

Tests pin each handoff so a refactor doesn't silently drop the flags:

  * HFPipelineConfig schema rejects 4+8 together, validates dtype names
  * _synthesize_hf_config carries the flags into the persisted config
  * resource_fit estimates scale with quantization_bits
  * the wrapper's from_pretrained call gets the right BnB kwargs
    (mocked transformers; no real model download)
  * a clear ImportError fires when bitsandbytes is missing
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Schema — mutual exclusion + dtype validation
# ---------------------------------------------------------------------------

class TestQuantizationSchema:

    def test_4bit_and_8bit_together_rejected(self):
        from neural_platform.core.config import HFPipelineConfig
        with pytest.raises(ValueError, match="mutually exclusive"):
            HFPipelineConfig(
                pretrained="org/model",
                load_in_4bit=True,
                load_in_8bit=True,
            )

    def test_each_quantization_mode_accepted_alone(self):
        from neural_platform.core.config import HFPipelineConfig
        a = HFPipelineConfig(pretrained="org/model", load_in_4bit=True)
        assert a.load_in_4bit is True and a.load_in_8bit is False
        b = HFPipelineConfig(pretrained="org/model", load_in_8bit=True)
        assert b.load_in_8bit is True and b.load_in_4bit is False

    def test_dtype_validation(self):
        from neural_platform.core.config import HFPipelineConfig
        # Accepted dtypes.
        for ok in ("float16", "bfloat16", "float32"):
            HFPipelineConfig(pretrained="x", load_in_4bit=True,
                              bnb_compute_dtype=ok)
        # Rejected.
        with pytest.raises(ValueError, match="bnb_compute_dtype"):
            HFPipelineConfig(pretrained="x", load_in_4bit=True,
                              bnb_compute_dtype="fp8")

    def test_dtype_optional_when_neither_flag_set(self):
        """The validator only checks the dtype enum; setting the dtype
        without enabling quantization is harmless (the wrapper ignores
        it)."""
        from neural_platform.core.config import HFPipelineConfig
        cfg = HFPipelineConfig(pretrained="x", bnb_compute_dtype="float16")
        assert cfg.bnb_compute_dtype == "float16"


# ---------------------------------------------------------------------------
# Synthesizer carries flags through
# ---------------------------------------------------------------------------

class TestSynthesizerFlags:

    def test_synth_carries_4bit_flag(self, tmp_path):
        from neural_platform.web.inference_manager import _synthesize_hf_config
        cfg, _ = _synthesize_hf_config(
            output_root=tmp_path,
            hf_model_id="org/big-model",
            pipeline_task="text-generation",
            load_in_4bit=True,
            bnb_compute_dtype="bfloat16",
        )
        assert cfg.model.hf_pipeline.load_in_4bit is True
        assert cfg.model.hf_pipeline.load_in_8bit is False
        assert cfg.model.hf_pipeline.bnb_compute_dtype == "bfloat16"

    def test_synth_carries_8bit_flag(self, tmp_path):
        from neural_platform.web.inference_manager import _synthesize_hf_config
        cfg, _ = _synthesize_hf_config(
            output_root=tmp_path,
            hf_model_id="org/big-model",
            pipeline_task="text-generation",
            load_in_8bit=True,
        )
        assert cfg.model.hf_pipeline.load_in_8bit is True
        assert cfg.model.hf_pipeline.load_in_4bit is False

    def test_synth_defaults_when_omitted(self, tmp_path):
        """Most launches don't quantize — defaults stay False so we don't
        accidentally try to load BnB when the user didn't ask for it."""
        from neural_platform.web.inference_manager import _synthesize_hf_config
        cfg, _ = _synthesize_hf_config(
            output_root=tmp_path,
            hf_model_id="org/small-model",
            pipeline_task="text-classification",
        )
        assert cfg.model.hf_pipeline.load_in_4bit is False
        assert cfg.model.hf_pipeline.load_in_8bit is False
        assert cfg.model.hf_pipeline.bnb_compute_dtype is None


# ---------------------------------------------------------------------------
# Endpoint — POST /api/inference/start_hf accepts quantization fields
# ---------------------------------------------------------------------------

class TestStartHFEndpoint:

    def test_endpoint_forwards_quantization_kwargs(self, tmp_path):
        """The InferenceServerManager is created via `from … import`
        inside create_dashboard_app, so we patch it at the source
        module — that's where the name resolves at construction time."""
        from fastapi.testclient import TestClient
        from neural_platform.web.app import create_dashboard_app
        from neural_platform.web import inference_manager as _imgr

        instance = MagicMock()
        instance.start_from_hf.return_value = SimpleNamespace(
            to_dict=lambda: {"id": "x", "name": "x", "port": 8090,
                              "status": "starting", "source": "huggingface"},
        )
        with patch.object(_imgr, "InferenceServerManager",
                            return_value=instance):
            app = create_dashboard_app(str(tmp_path))
            with TestClient(app) as client:
                r = client.post("/api/inference/start_hf", json={
                    "hf_model_id":     "google/gemma-3-4b-it",
                    "pipeline_task":   "text-generation",
                    "load_in_4bit":    True,
                    "bnb_compute_dtype": "bfloat16",
                })
            assert r.status_code == 200, r.text
            instance.start_from_hf.assert_called_once()
            kwargs = instance.start_from_hf.call_args.kwargs
            assert kwargs["load_in_4bit"] is True
            assert kwargs["load_in_8bit"] is False
            assert kwargs["bnb_compute_dtype"] == "bfloat16"


# ---------------------------------------------------------------------------
# Resource-fit — quantized weight size scales with bits
# ---------------------------------------------------------------------------

class TestResourceFitWithQuantization:

    def test_4bit_weights_are_one_eighth_of_fp32(self):
        from neural_platform.core.resource_fit import estimate_model_footprint
        params = 1_000_000_000   # 1B
        # fp32 baseline — 4 bytes per param.
        full = estimate_model_footprint(params, size_bytes=None,
                                          purpose="inference")
        # 4-bit — 0.5 bytes per param.
        q4 = estimate_model_footprint(params, size_bytes=None,
                                        purpose="inference",
                                        quantization_bits=4)
        # 4-bit weight buffer is exactly 1/8 the fp32 buffer
        # (4 bytes → 0.5 bytes per param).
        ratio = q4.model_weight_b / full.model_weight_b
        assert 0.124 <= ratio <= 0.126, f"ratio={ratio}"

    def test_8bit_weights_are_one_quarter_of_fp32(self):
        from neural_platform.core.resource_fit import estimate_model_footprint
        params = 500_000_000
        full = estimate_model_footprint(params, size_bytes=None,
                                          purpose="inference")
        q8 = estimate_model_footprint(params, size_bytes=None,
                                        purpose="inference",
                                        quantization_bits=8)
        ratio = q8.model_weight_b / full.model_weight_b
        assert 0.249 <= ratio <= 0.251, f"ratio={ratio}"

    def test_quantization_forces_inference_purpose(self):
        """bitsandbytes doesn't support backprop through 4-bit weights
        as a first-class workflow. When the caller passes
        quantization_bits + purpose='training', the estimator silently
        switches to inference (no gradient / optimizer state) so the
        VRAM estimate isn't artificially inflated."""
        from neural_platform.core.resource_fit import estimate_model_footprint
        params = 100_000_000
        est = estimate_model_footprint(params, size_bytes=None,
                                         purpose="training",
                                         quantization_bits=4)
        # If 'training' had been honored, gradients_b would be
        # params * 4 = 400 MB. With the forced switch it's 0.
        assert est.gradients_b == 0
        assert est.optimizer_b == 0


# ---------------------------------------------------------------------------
# Wrapper — BitsAndBytesConfig threaded through from_pretrained
# ---------------------------------------------------------------------------

class TestWrapperBitsAndBytesPlumbing:
    """Pins the wrapper's BnB plumbing.

    We can't replace ``sys.modules['transformers']`` wholesale — that
    breaks the wrapper's own ``import torch`` (torch is sensitive to
    sibling module reloads). Instead we let the real transformers
    module load and patch only the specific attributes the wrapper
    touches (the Auto* class + ``BitsAndBytesConfig``)."""

    @pytest.fixture
    def patched_tx(self):
        """Patch transformers.AutoModelForCausalLM.from_pretrained AND
        transformers.BitsAndBytesConfig. Yields (captured_load_kwargs,
        captured_bnb_kwargs) so each test can assert on what the
        wrapper would have passed downstream."""
        import transformers
        captured: dict = {}
        bnb_kwargs: dict = {}

        def _from_pretrained(repo, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return MagicMock(config=SimpleNamespace(num_labels=0))

        def _bnb_ctor(**kwargs):
            bnb_kwargs.clear()
            bnb_kwargs.update(kwargs)
            return SimpleNamespace(**kwargs)

        with patch.object(transformers.AutoModelForCausalLM,
                            "from_pretrained", _from_pretrained), \
             patch.object(transformers, "BitsAndBytesConfig", _bnb_ctor):
            yield captured, bnb_kwargs

    def _build_model(self, *, has_bitsandbytes=True, **arch_kwargs):
        """Construct the wrapper. Manages bitsandbytes availability via
        sys.modules without touching transformers / torch."""
        from neural_platform.core.config import (
            HFPipelineConfig, ModelConfig, ModelType, Framework,
        )
        from neural_platform.models.hf_pipeline import HFPipelineModel
        # The wrapper takes a ModelConfig — wrap our HFPipelineConfig in
        # one so .hf_pipeline / .type resolve as the wrapper expects.
        hf_cfg = HFPipelineConfig(pretrained="google/gemma-3-4b-it",
                                    **arch_kwargs)
        model_cfg = ModelConfig(
            type=ModelType.HF_PIPELINE,
            framework=Framework.PYTORCH,
            hf_pipeline=hf_cfg,
        )
        # The adapter normally stamps this on at build time; the wrapper
        # reads it directly via getattr.
        model_cfg._resolved_task = "text-generation"   # type: ignore[attr-defined]
        import sys
        saved_bnb = sys.modules.get("bitsandbytes")
        if has_bitsandbytes:
            sys.modules["bitsandbytes"] = MagicMock()
        else:
            sys.modules["bitsandbytes"] = None   # makes `import bitsandbytes` raise ImportError
        try:
            HFPipelineModel(model_cfg)
        finally:
            if saved_bnb is not None:
                sys.modules["bitsandbytes"] = saved_bnb
            else:
                sys.modules.pop("bitsandbytes", None)

    def test_4bit_load_passes_bnb_config_to_from_pretrained(self, patched_tx):
        captured, bnb_kwargs = patched_tx
        self._build_model(load_in_4bit=True, bnb_compute_dtype="bfloat16")
        # quantization_config was set on from_pretrained.
        assert "quantization_config" in captured
        # device_map='auto' is required alongside BnB.
        assert captured.get("device_map") == "auto"
        # BnB ctor saw the right kwargs.
        assert bnb_kwargs.get("load_in_4bit") is True
        assert bnb_kwargs.get("bnb_4bit_quant_type") == "nf4"
        assert bnb_kwargs.get("bnb_4bit_use_double_quant") is True
        # compute_dtype mapped to torch.bfloat16.
        import torch
        assert bnb_kwargs.get("bnb_4bit_compute_dtype") is torch.bfloat16

    def test_8bit_load_passes_8bit_flag(self, patched_tx):
        captured, bnb_kwargs = patched_tx
        self._build_model(load_in_8bit=True)
        assert "quantization_config" in captured
        assert bnb_kwargs.get("load_in_8bit") is True
        # The 4-bit specific kwargs should NOT be set for an 8-bit load.
        assert "bnb_4bit_quant_type" not in bnb_kwargs
        assert "bnb_4bit_compute_dtype" not in bnb_kwargs

    def test_no_quantization_no_bnb_config(self, patched_tx):
        """When the user didn't ask for quantization, the wrapper must
        not silently inject a BitsAndBytesConfig — that would change
        the loaded weights' dtype without consent."""
        captured, _ = patched_tx
        self._build_model()   # no flags set
        assert "quantization_config" not in captured
        assert "device_map" not in captured

    def test_missing_bitsandbytes_raises_clear_error(self, patched_tx):
        """The wrapper refuses to silently fall back to full precision
        when the user explicitly asked for 4-bit — that would mask an
        OOM with a config the user thought was applied. Surface a
        clear ImportError pointing at the [quantization] extra."""
        with pytest.raises(ImportError, match="bitsandbytes"):
            self._build_model(has_bitsandbytes=False, load_in_4bit=True)
