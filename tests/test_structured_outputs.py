"""
Tests for the structured-output postprocessors — boxes / depth / masks /
QA spans / token spans. Each task's response carries

  * a stable ``result_kind`` so the frontend can dispatch to the right renderer
  * structured details in ``Prediction.metadata`` (bbox / start_idx /
    image_b64 thumbnails / token offsets) so the frontend can do
    something better than print "<label> @ x,y,x,y" as a flat string.

Without these tests, dropping back to the flat-string output is a
silent regression that's only visible by clicking through the UI.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Object detection — bbox + score in metadata
# ---------------------------------------------------------------------------

class TestBoxesResponse:

    def _fake_det_output(self, num_queries=5, num_classes=4):
        """Mimic the HF object-detection output shape:
        logits (1, Q, num_classes+1) with the last class being 'no object',
        pred_boxes (1, Q, 4) in cxcywh normalized [0, 1]."""
        torch.manual_seed(0)
        # Strong predictions for class 1 on the first two queries.
        logits = torch.full((1, num_queries, num_classes + 1), -10.0)
        logits[0, 0, 1] = 9.0   # confident "class 1"
        logits[0, 1, 2] = 7.0   # less confident "class 2"
        logits[0, 2, num_classes] = 9.0   # "no object" — should be skipped
        # All others stay near uniform; their max class scores will be tiny.
        boxes = torch.tensor([
            [[0.1, 0.2, 0.3, 0.4],
             [0.5, 0.5, 0.2, 0.2],
             [0.0, 0.0, 1.0, 1.0],
             [0.4, 0.6, 0.1, 0.1],
             [0.7, 0.8, 0.05, 0.05]],
        ])
        return SimpleNamespace(logits=logits, pred_boxes=boxes)

    def test_result_kind_is_boxes(self):
        from neural_platform.deploy.server import _boxes_response
        out = _boxes_response(self._fake_det_output(),
                                model_type="hf_pipeline", t0=0.0,
                                class_names_fn=lambda i: f"thing_{i}")
        assert out.result_kind == "boxes"

    def test_metadata_contains_bbox_in_xyxy_norm(self):
        """The frontend overlays boxes on the previewed image. We commit
        to xyxy_norm coordinates (top-left origin, [0, 1] of image dims)."""
        from neural_platform.deploy.server import _boxes_response
        out = _boxes_response(self._fake_det_output(),
                                model_type="hf_pipeline", t0=0.0,
                                class_names_fn=lambda i: f"thing_{i}")
        preds = out.predictions[0]
        assert preds, "Expected at least one detection"
        first = preds[0].metadata
        assert first["format"] == "xyxy_norm"
        assert isinstance(first["bbox"], list)
        assert len(first["bbox"]) == 4
        x1, y1, x2, y2 = first["bbox"]
        # cxcywh = (0.1, 0.2, 0.3, 0.4) → (x1=-0.05, y1=0.0, x2=0.25, y2=0.4)
        assert pytest.approx(x1, abs=1e-6) == -0.05
        assert pytest.approx(x2, abs=1e-6) == 0.25

    def test_class_name_is_human_label_not_coordinate_string(self):
        """Regression: previously the class_name carried "<label> @ x,y,x,y"
        which made the UI render unreadable strings as bar labels.
        class_name must now be the bare label."""
        from neural_platform.deploy.server import _boxes_response
        out = _boxes_response(self._fake_det_output(),
                                model_type="hf_pipeline", t0=0.0,
                                class_names_fn=lambda i: f"thing_{i}")
        for p in out.predictions[0]:
            assert "@" not in (p.class_name or "")
            assert "," not in (p.class_name or "")

    def test_no_pred_boxes_returns_safe_empty_response(self):
        """If the HF output is missing pred_boxes (custom model), surface
        a clean placeholder rather than 500 the request."""
        from neural_platform.deploy.server import _boxes_response
        out = _boxes_response(SimpleNamespace(logits=None, pred_boxes=None),
                                model_type="hf_pipeline", t0=0.0,
                                class_names_fn=lambda i: None)
        assert out.result_kind == "boxes"
        assert out.predictions[0][0].class_name == "(no detections)"


# ---------------------------------------------------------------------------
# Depth estimation — colormapped PNG thumbnail in metadata
# ---------------------------------------------------------------------------

class TestDepthResponse:

    def _fake_depth_output(self, h=64, w=64):
        # A simple gradient depth map [0, 1].
        depth = torch.linspace(0, 1, h * w).view(1, h, w)
        return SimpleNamespace(predicted_depth=depth)

    def test_result_kind_is_depth(self):
        from neural_platform.deploy.server import _depth_response
        out = _depth_response(self._fake_depth_output(), "hf_pipeline", 0.0)
        assert out.result_kind == "depth"

    def test_metadata_contains_png_thumbnail(self):
        """The depth tensor is too big for the wire — server colormaps
        and thumbnails it. Verify the response carries a valid base64
        PNG that the frontend can render via <img src="data:...">."""
        from neural_platform.deploy.server import _depth_response
        out = _depth_response(self._fake_depth_output(), "hf_pipeline", 0.0)
        md = out.predictions[0][0].metadata
        assert md["image_mime"] == "image/png"
        assert isinstance(md["image_b64"], str)
        # Decoded bytes should start with the PNG magic number.
        png_bytes = base64.b64decode(md["image_b64"])
        assert png_bytes[:4] == b"\x89PNG"
        # Thumbnail caps at 256 on the longer side.
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))
        assert max(img.size) <= 256

    def test_metadata_carries_depth_stats(self):
        """min/max/mean help users interpret the colormap (closer =
        brighter). Pin them as proper floats."""
        from neural_platform.deploy.server import _depth_response
        out = _depth_response(self._fake_depth_output(), "hf_pipeline", 0.0)
        md = out.predictions[0][0].metadata
        assert md["min"] == pytest.approx(0.0, abs=1e-3)
        assert md["max"] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Segmentation — per-class mask thumbnails ranked by area
# ---------------------------------------------------------------------------

class TestMasksResponse:

    def _fake_seg_logits(self, num_classes=3, h=32, w=32):
        """Fake segmentation logits. Class 0 dominates ~80% of the
        image; class 1 takes a corner; class 2 a few stray pixels."""
        logits = torch.full((1, num_classes, h, w), -10.0)
        logits[0, 0, :, :] = 1.0      # class 0 background
        logits[0, 1, :8, :8] = 5.0    # class 1 corner
        logits[0, 2, 0, 0] = 8.0      # class 2 single pixel
        return SimpleNamespace(logits=logits, pred_masks=None)

    def test_result_kind_is_masks(self):
        from neural_platform.deploy.server import _masks_response
        out = _masks_response(self._fake_seg_logits(), "hf_pipeline", 0.0)
        assert out.result_kind == "masks"

    def test_returns_per_class_thumbnails_sorted_by_area(self):
        from neural_platform.deploy.server import _masks_response
        out = _masks_response(self._fake_seg_logits(), "hf_pipeline", 0.0)
        preds = out.predictions[0]
        assert len(preds) >= 1
        # First entry should be the largest class — 0 (background, ~88%
        # of the image after argmax: 32*32=1024 total; class 1 covers
        # 8*8=64, class 2 covers 1, class 0 covers the rest = 959).
        assert preds[0].label == 0
        # All entries carry a PNG thumbnail.
        for p in preds:
            md = p.metadata
            assert md["image_mime"] == "image/png"
            assert base64.b64decode(md["image_b64"])[:4] == b"\x89PNG"
            assert 0.0 <= md["coverage"] <= 1.0

    def test_no_mask_output_returns_placeholder(self):
        from neural_platform.deploy.server import _masks_response
        out = _masks_response(SimpleNamespace(logits=None, pred_masks=None),
                                "hf_pipeline", 0.0)
        assert out.result_kind == "masks"
        assert out.predictions[0][0].class_name == "(no mask output)"


# ---------------------------------------------------------------------------
# QA spans — start/end indices in metadata + decoded answer
# ---------------------------------------------------------------------------

class TestQAMetadata:

    def test_qa_response_carries_indices_and_kind(self):
        """QA postproc was rewritten to populate metadata.start_idx /
        end_idx / start_prob / end_prob so the UI can highlight the span
        in the user's context. Pin those fields."""
        from neural_platform.deploy.server import _qa_response
        s = torch.full((1, 12), -10.0); s[0, 3] = 10.0
        e = torch.full((1, 12), -10.0); e[0, 5] = 10.0
        outputs = SimpleNamespace(start_logits=s, end_logits=e)
        tok = MagicMock()
        tok.decode.return_value = "answer span"
        proc = SimpleNamespace(tokenizer=tok)
        resp = _qa_response(outputs, {"input_ids": torch.arange(12).unsqueeze(0)},
                             proc, "hf_pipeline", 0.0)
        assert resp.result_kind == "qa_spans"
        md = resp.predictions[0][0].metadata
        assert md["start_idx"] == 3
        assert md["end_idx"] == 5
        assert 0.0 <= md["start_prob"] <= 1.0


# ---------------------------------------------------------------------------
# Token classification — per-token labels + offsets
# ---------------------------------------------------------------------------

class TestTokenSpansResponse:

    def _fake_token_logits(self, seq_len=8, num_classes=3):
        """Token classification logits. Tokens 2 and 5 are tagged class 1."""
        logits = torch.full((1, seq_len, num_classes), -10.0)
        logits[0, :, 0] = 5.0   # background by default
        logits[0, 2, 1] = 9.0
        logits[0, 5, 1] = 9.0
        return logits

    def test_returns_token_spans_kind(self):
        from neural_platform.deploy.server import _token_spans_response
        # Fake tokenizer that returns token strings.
        tok = MagicMock()
        tok.convert_ids_to_tokens.return_value = [f"tok_{i}" for i in range(8)]
        proc = SimpleNamespace(tokenizer=tok)
        out = _token_spans_response(
            self._fake_token_logits(),
            tensor_input={"input_ids": torch.arange(8).unsqueeze(0)},
            processor=proc, model_type="hf_pipeline", t0=0.0,
            class_names=["O", "ENTITY", "OTHER"], top_k=10,
        )
        assert out.result_kind == "token_spans"
        # Two non-background spans expected (tokens 2 and 5).
        assert len(out.predictions[0]) == 2
        for p in out.predictions[0]:
            assert p.class_name == "ENTITY"
            assert p.metadata["token"].startswith("tok_")

    def test_falls_back_to_logits_kind_when_no_tagged_spans(self):
        """If every token is background (class 0), there's nothing to
        highlight — fall through to the regular top-K bars so the UI
        still has something to show."""
        from neural_platform.deploy.server import _token_spans_response
        seq_len, num_classes = 8, 3
        all_zero = torch.full((1, seq_len, num_classes), -10.0)
        all_zero[0, :, 0] = 5.0   # everything class 0
        out = _token_spans_response(
            all_zero,
            tensor_input={"input_ids": torch.arange(seq_len).unsqueeze(0)},
            processor=None, model_type="hf_pipeline", t0=0.0,
            class_names=["O", "ENT", "OTHER"], top_k=3,
        )
        assert out.result_kind == "logits"


# ---------------------------------------------------------------------------
# Tensor → PNG helper
# ---------------------------------------------------------------------------

class TestTensorToPng:

    def test_2d_tensor_renders_to_png(self):
        from neural_platform.deploy.server import _tensor_to_png_b64
        t = torch.linspace(0, 1, 64*64).view(64, 64)
        b64 = _tensor_to_png_b64(t)
        assert isinstance(b64, str)
        assert base64.b64decode(b64)[:4] == b"\x89PNG"

    def test_3d_tensor_argmaxes_over_channels(self):
        """For (C, H, W) tensors (raw segmentation logits) the helper
        argmaxes to a label map and renders that. Without the argmax the
        PIL conversion would fail on 3-D float input."""
        from neural_platform.deploy.server import _tensor_to_png_b64
        t = torch.zeros(3, 16, 16)
        t[1] = 1.0   # class 1 wins everywhere
        b64 = _tensor_to_png_b64(t)
        assert isinstance(b64, str)

    def test_constant_tensor_doesnt_crash(self):
        """A flat min == max tensor used to divide by zero. Now we
        return a uniform black image instead."""
        from neural_platform.deploy.server import _tensor_to_png_b64
        t = torch.zeros(8, 8)
        b64 = _tensor_to_png_b64(t)
        assert isinstance(b64, str)


# ---------------------------------------------------------------------------
# Generated text result_kind
# ---------------------------------------------------------------------------

class TestGeneratedTextKind:
    """The generative branch used to return the response without a
    result_kind. The frontend now dispatches on it, so it must always
    be set to 'generated_text' for ASR / summarization / image-to-text /
    text-generation / image-text-to-text / any-to-any."""

    def test_generative_path_sets_result_kind(self, tmp_path, monkeypatch):
        """Drive _do_predict end-to-end with a stub model that has a
        .generate() method, and verify the response carries
        result_kind='generated_text'."""
        monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        from neural_platform.deploy.server import create_inference_app
        from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter
        from fastapi.testclient import TestClient

        # Stub model whose encoder.generate returns a short token sequence.
        import torch.nn as nn
        class _Encoder(nn.Module):
            def __init__(self): super().__init__()
            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3]])
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Encoder()
            def count_parameters(self, trainable_only=False): return 0

        # Stub processor: tokenizes text on call, decodes to a fixed string.
        proc = MagicMock()
        proc.return_value = {"input_ids": torch.tensor([[101, 7592, 102]])}
        proc.batch_decode = MagicMock(return_value=["hello there"])

        cfg = ExperimentConfig(
            name="gen", output_dir=str(tmp_path),
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(pretrained="t5-small"),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION,
                                     pipeline_task="summarization"),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        monkeypatch.setattr(PyTorchAdapter, "build_model",
                             lambda self: _Stub())
        from neural_platform.deploy import server as _server
        monkeypatch.setattr(_server, "_try_load_processor", lambda c: proc)

        app = create_inference_app(cfg, checkpoint_path=None)
        with TestClient(app) as client:
            r = client.post("/predict", json={"text": "summarize this"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["result_kind"] == "generated_text"
        assert body["predictions"][0][0]["class_name"] == "hello there"
