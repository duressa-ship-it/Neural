"""
Tests for the multi-turn chat surface.

The chat path is gated by ``request.messages`` being non-empty AND the
loaded processor / tokenizer exposing a ``chat_template``. When both
hold, the input adapter calls ``processor.apply_chat_template(messages,
add_generation_prompt=True)`` and uses the result as the model's input;
the universal-input fields are bypassed.

These tests pin:

  * ``ChatMessage`` / ``ChatContentPart`` schemas accept text + image +
    audio parts
  * ``_build_chat_input`` calls ``apply_chat_template`` with the right
    argument shape
  * Multimodal content parts (text + image) make it through with the
    image decoded to a PIL.Image
  * Without a chat template, the helper falls back to a flat
    role-prefixed concatenation (text-only) and rejects multimodal
  * /info reports ``has_chat_template`` and surfaces it on
    ``ui_hint.show_chat``
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Schema acceptance
# ---------------------------------------------------------------------------

class TestChatSchemas:

    def test_chat_message_accepts_string_content(self):
        from neural_platform.deploy.server import ChatMessage
        m = ChatMessage(role="user", content="hello")
        assert isinstance(m.content, str)

    def test_chat_message_accepts_typed_parts(self):
        from neural_platform.deploy.server import ChatMessage, ChatContentPart
        parts = [
            ChatContentPart(type="text", text="describe this"),
            ChatContentPart(type="image", image_b64="abc=="),
        ]
        m = ChatMessage(role="user", content=parts)
        assert len(m.content) == 2
        assert m.content[1].type == "image"

    def test_predict_request_accepts_messages(self):
        from neural_platform.deploy.server import PredictRequest, ChatMessage
        req = PredictRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert req.messages is not None
        assert req.messages[0].role == "user"


# ---------------------------------------------------------------------------
# _build_chat_input — drives the chat template
# ---------------------------------------------------------------------------

class TestBuildChatInput:

    def _make_processor(self, *, with_template=True, decoded_text="<input>"):
        """Stub processor that mimics the HF chat-template surface.

        - apply_chat_template returns a fake encoding dict
        - tokenizer.chat_template is the gating attribute; set to None
          to test the no-template fallback path
        """
        import torch
        proc = MagicMock()
        tok = MagicMock()
        tok.chat_template = "{{ messages }}" if with_template else None
        proc.tokenizer = tok
        # Mimic the modern API: apply_chat_template returns a tensor dict.
        proc.apply_chat_template.return_value = {
            "input_ids":      torch.tensor([[101, 999, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        # Plain tokenizer call for the fallback path.
        tok.return_value = {"input_ids": torch.tensor([[101, 102]])}
        return proc

    def _build_request(self, messages):
        from neural_platform.deploy.server import PredictRequest
        return PredictRequest(messages=messages)

    def test_apply_chat_template_called_with_messages(self):
        """The HF chat surface is the right path — confirm we hand it
        the dict-shape it expects, with add_generation_prompt=True."""
        from neural_platform.deploy.server import _build_chat_input
        from neural_platform.deploy.server import ChatMessage
        proc = self._make_processor(with_template=True)
        req = self._build_request([
            ChatMessage(role="user", content="hello"),
        ])
        out = _build_chat_input(req, proc, device="cpu")
        # Was called once, with messages + add_generation_prompt flag.
        proc.apply_chat_template.assert_called_once()
        args = proc.apply_chat_template.call_args
        msgs = args.args[0] if args.args else args.kwargs.get("messages")
        kwargs = args.kwargs
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        assert kwargs.get("add_generation_prompt") is True
        # Returns the encoded dict, with tensors moved (cpu device just
        # passes through the .to() shim).
        assert "input_ids" in out

    def test_typed_parts_with_image_decoded_to_pil(self):
        """When a ChatContentPart carries image_b64, _build_chat_input
        decodes it to a PIL.Image before handing the parts list to the
        chat template — that's the shape Qwen2-VL / LLaVA expect."""
        from PIL import Image
        from neural_platform.deploy.server import (
            _build_chat_input, ChatMessage, ChatContentPart,
        )
        # Make a tiny valid PNG to encode.
        img = Image.new("RGB", (4, 4), color=(0, 128, 255))
        buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        proc = self._make_processor(with_template=True)
        req = self._build_request([
            ChatMessage(role="user", content=[
                ChatContentPart(type="text", text="what is this?"),
                ChatContentPart(type="image", image_b64=b64),
            ]),
        ])
        _build_chat_input(req, proc, device="cpu")
        msgs = proc.apply_chat_template.call_args.args[0]
        # The text part is preserved verbatim.
        parts = msgs[0]["content"]
        assert parts[0] == {"type": "text", "text": "what is this?"}
        # The image part carries a PIL.Image object — *not* the raw b64.
        assert parts[1]["type"] == "image"
        assert isinstance(parts[1]["image"], Image.Image)

    def test_fallback_concatenates_when_no_chat_template(self):
        """Older tokenizers without a chat template — fall back to
        plain role-prefixed concatenation. Loses role / multimodal
        info but keeps simple text chat working."""
        from neural_platform.deploy.server import (
            _build_chat_input, ChatMessage,
        )
        proc = self._make_processor(with_template=False)
        req = self._build_request([
            ChatMessage(role="system", content="you are helpful"),
            ChatMessage(role="user", content="hi"),
        ])
        _build_chat_input(req, proc, device="cpu")
        # apply_chat_template MUST NOT have been called.
        proc.apply_chat_template.assert_not_called()
        # The tokenizer was called once with the concatenated string.
        proc.tokenizer.assert_called()
        flat_arg = proc.tokenizer.call_args.args[0]
        assert "SYSTEM:" in flat_arg
        assert "USER:" in flat_arg
        assert "hi" in flat_arg
        # The fallback also appends the assistant prompt so generation
        # can pick up where the user left off.
        assert "ASSISTANT:" in flat_arg

    def test_fallback_rejects_multimodal_messages(self):
        """No chat template + multimodal content is unsupported. We
        surface a clear error rather than silently dropping the image."""
        from neural_platform.deploy.server import (
            _build_chat_input, ChatMessage, ChatContentPart, _InputError,
        )
        proc = self._make_processor(with_template=False)
        req = self._build_request([
            ChatMessage(role="user", content=[
                ChatContentPart(type="text", text="describe"),
                ChatContentPart(type="image", image_b64="abc=="),
            ]),
        ])
        with pytest.raises(_InputError):
            _build_chat_input(req, proc, device="cpu")


# ---------------------------------------------------------------------------
# /info exposes has_chat_template
# ---------------------------------------------------------------------------

class TestInfoHasChatTemplate:
    """The Predict UI gates the chat transcript pane on
    info.has_chat_template, so the field must be present in the
    InfoResponse and reflect what the loaded tokenizer actually
    supports."""

    def _stub(self, with_template):
        import torch.nn as nn
        class _Cfg:
            id2label = {}
            num_labels = 0
        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Cfg()
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Encoder()
            def count_parameters(self, trainable_only=False): return 0
        return _Stub()

    def _processor(self, with_template):
        proc = MagicMock()
        tok = MagicMock()
        tok.chat_template = "fake-template" if with_template else None
        proc.tokenizer = tok
        return proc

    def _info(self, monkeypatch, *, with_template):
        monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        from neural_platform.deploy.server import create_inference_app
        from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter
        from fastapi.testclient import TestClient

        cfg = ExperimentConfig(
            name="x", output_dir="/tmp",
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(pretrained="meta-llama/whatever"),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION,
                                     pipeline_task="text-generation"),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        monkeypatch.setattr(PyTorchAdapter, "build_model",
                             lambda self: self._stub_for_test()
                                          if hasattr(self, "_stub_for_test")
                                          else None)
        # Inject our stub by patching the adapter directly.
        stub = self._stub(with_template)
        monkeypatch.setattr(PyTorchAdapter, "build_model", lambda self: stub)
        proc = self._processor(with_template)
        from neural_platform.deploy import server as _server
        monkeypatch.setattr(_server, "_try_load_processor", lambda c: proc)

        app = create_inference_app(cfg, checkpoint_path=None)
        with TestClient(app) as client:
            r = client.get("/info")
        assert r.status_code == 200, r.text
        return r.json()

    def test_info_reports_chat_template_when_present(self, monkeypatch):
        body = self._info(monkeypatch, with_template=True)
        assert body["has_chat_template"] is True
        assert body["ui_hint"]["show_chat"] is True

    def test_info_reports_no_chat_template_when_absent(self, monkeypatch):
        body = self._info(monkeypatch, with_template=False)
        assert body["has_chat_template"] is False
        assert body["ui_hint"]["show_chat"] is False


# ---------------------------------------------------------------------------
# End-to-end: messages flow through to apply_chat_template via /predict
# ---------------------------------------------------------------------------

class TestChatRoutingEndToEnd:
    """One integration test that confirms the routing actually fires
    when a request carries `messages` — guards against the chat path
    being short-circuited by some other branch landing first."""

    def test_predict_routes_messages_to_chat_template(self, monkeypatch):
        monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        from neural_platform.deploy.server import create_inference_app
        from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter
        from fastapi.testclient import TestClient
        import torch
        import torch.nn as nn

        # Encoder.generate returns a fixed token sequence.
        class _Enc(nn.Module):
            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3]])
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Enc()
            def count_parameters(self, trainable_only=False): return 0

        proc = MagicMock()
        tok = MagicMock()
        tok.chat_template = "ok"
        proc.tokenizer = tok
        proc.apply_chat_template.return_value = {
            "input_ids":      torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        proc.batch_decode = MagicMock(return_value=["hi back"])

        cfg = ExperimentConfig(
            name="g", output_dir="/tmp",
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(pretrained="org/model"),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION,
                                     pipeline_task="text-generation"),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        monkeypatch.setattr(PyTorchAdapter, "build_model", lambda self: _Stub())
        from neural_platform.deploy import server as _server
        monkeypatch.setattr(_server, "_try_load_processor", lambda c: proc)

        app = create_inference_app(cfg, checkpoint_path=None)
        with TestClient(app) as client:
            r = client.post("/predict", json={
                "messages": [
                    {"role": "user", "content": "hi there"},
                ],
            })
        assert r.status_code == 200, r.text
        # The chat template must have been called, NOT the regular
        # tokenizer call (which the universal text path would use).
        proc.apply_chat_template.assert_called_once()
        body = r.json()
        assert body["result_kind"] == "generated_text"
