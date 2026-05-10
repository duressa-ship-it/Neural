"""
Tests for the /predict/stream SSE endpoint and the dashboard's
streaming proxy.

The streaming path runs ``model.encoder.generate(streamer=…)`` on a
background thread and pumps tokens through ``TextIteratorStreamer``.
We replace the streamer with a tiny iterator stub so the tests are
strictly offline + deterministic — no real generation, no live HF
model. All we're verifying is the wire shape:

  * non-generative tasks 400 *before* opening the stream
  * each token from the streamer becomes one ``event: token\\ndata: …``
    SSE record
  * the final ``event: done`` carries the assembled text + latency
  * the dashboard's ``/api/inference/{id}/predict/stream`` proxy
    forwards the SSE chunks unchanged with the bearer attached
    server-side
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — fake streamer + stub model
# ---------------------------------------------------------------------------

class _FakeStreamer:
    """Stand-in for transformers.TextIteratorStreamer.

    Pre-loads the pieces the test wants emitted, plus a flag for
    early termination via ``end()``.
    """

    def __init__(self, pieces):
        self._pieces = list(pieces)
        self._idx = 0
        self.ended = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.ended or self._idx >= len(self._pieces):
            raise StopIteration
        out = self._pieces[self._idx]
        self._idx += 1
        return out

    def end(self):
        self.ended = True


def _make_app(monkeypatch, *, pipeline_task: str,
               pieces=("Hello", " world", "!")):
    """Build an inference app pointed at a stub HF-pipeline model.

    The model has only what the streaming path touches:
      * .encoder.generate(streamer=…, **kwargs) — the test wires it to
        feed `pieces` into the streamer when called.
      * .count_parameters / .config — surfaces enough for /info.
    """
    monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
    from neural_platform.core.config import (
        ExperimentConfig, ModelConfig, ModelType, Framework,
        HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig, Task,
    )
    from neural_platform.deploy.server import create_inference_app
    from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter

    import torch.nn as nn
    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
        def generate(self, **kwargs):
            streamer = kwargs.get("streamer")
            if streamer is not None:
                # Feed the fake streamer so the SSE generator can read
                # tokens off it. The test passes us the pre-loaded
                # streamer directly via the patched constructor below;
                # this ``generate()`` is a no-op other than ending it.
                pass
            return None
    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()
        def count_parameters(self, trainable_only=False): return 0

    cfg = ExperimentConfig(
        name="stream", output_dir="/tmp",
        model=ModelConfig(
            type=ModelType.HF_PIPELINE,
            framework=Framework.PYTORCH,
            hf_pipeline=HFPipelineConfig(pretrained="t5-small"),
        ),
        training=TrainingConfig(task=Task.CLASSIFICATION,
                                 pipeline_task=pipeline_task),
        data=DataConfig(),
        deploy=DeployConfig(),
    )
    monkeypatch.setattr(PyTorchAdapter, "build_model",
                         lambda self: _Stub())

    # Stub processor: tokenizes text input into a tensor dict, exposes
    # a tokenizer attribute so the streamer constructor accepts it.
    proc = MagicMock()
    proc.tokenizer = MagicMock()
    import torch
    proc.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    from neural_platform.deploy import server as _server
    monkeypatch.setattr(_server, "_try_load_processor", lambda c: proc)

    # Patch TextIteratorStreamer to yield a deterministic sequence.
    fake = _FakeStreamer(pieces)
    monkeypatch.setattr(
        "transformers.TextIteratorStreamer",
        lambda *a, **kw: fake,
    )
    return create_inference_app(cfg, checkpoint_path=None), fake


def _consume_sse(client, request_body=None):
    """POST /predict/stream and parse the SSE body into a list of
    ``(event, data)`` tuples. Synchronous — the TestClient already
    drives the async event loop for us."""
    request_body = request_body or {"text": "stream this please"}
    with client.stream("POST", "/predict/stream",
                        json=request_body) as r:
        body = b""
        for chunk in r.iter_bytes():
            body += chunk
        status = r.status_code
    text = body.decode("utf-8", "replace")
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        ev_type, data_lines = "", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        events.append((ev_type, "\n".join(data_lines)))
    return status, events


# ---------------------------------------------------------------------------
# /predict/stream endpoint
# ---------------------------------------------------------------------------

class TestStreamEndpoint:

    def test_yields_one_event_per_token_then_done(self, monkeypatch):
        from fastapi.testclient import TestClient
        app, _ = _make_app(monkeypatch, pipeline_task="summarization",
                            pieces=("Hello", " world", "!"))
        with TestClient(app) as client:
            status, events = _consume_sse(client)
        assert status == 200
        # Last event is "done"; everything before should be "token"s.
        token_events = [(e, d) for (e, d) in events if e == "token"]
        done_events  = [(e, d) for (e, d) in events if e == "done"]
        assert len(token_events) == 3, f"got {events}"
        # Tokens come back as JSON-encoded strings.
        import json as _json
        decoded = [_json.loads(d) for (_, d) in token_events]
        assert decoded == ["Hello", " world", "!"]
        # "done" carries the joined text and a latency_ms number.
        assert len(done_events) == 1
        done = _json.loads(done_events[0][1])
        assert done["text"] == "Hello world!"
        assert done["result_kind"] == "generated_text"
        assert isinstance(done["latency_ms"], (int, float))

    def test_400s_for_non_generative_task(self, monkeypatch):
        """Calling /predict/stream against e.g. text-classification
        must reject upfront — streaming a softmax distribution doesn't
        make sense and would pollute the response."""
        from fastapi.testclient import TestClient
        app, _ = _make_app(monkeypatch,
                            pipeline_task="text-classification")
        with TestClient(app) as client:
            r = client.post("/predict/stream", json={"text": "hi"})
        assert r.status_code == 400
        assert "isn't generative" in r.text or "isn\\u2019t generative" in r.text

    def test_input_validation_errors_surface_before_stream_opens(self, monkeypatch):
        """Missing input → 422 instead of opening an SSE connection
        and yielding nothing useful."""
        from fastapi.testclient import TestClient
        app, _ = _make_app(monkeypatch,
                            pipeline_task="automatic-speech-recognition")
        with TestClient(app) as client:
            # ASR needs `inputs` / `audio_b64` — sending neither.
            r = client.post("/predict/stream", json={})
        assert r.status_code == 422
        # The detail mentions the audio requirement.
        assert "audio" in r.text.lower()


# ---------------------------------------------------------------------------
# Inference manager — proxy_stream forwards bytes with bearer attached
# ---------------------------------------------------------------------------

class TestProxyStream:
    """proxy_stream returns an async generator. Run it via
    asyncio.run() so the test suite doesn't need pytest-asyncio
    configured — keeps CI's [dev] dep set minimal."""

    def test_proxy_stream_forwards_chunks(self, tmp_path):
        import asyncio
        asyncio.run(self._test_proxy_stream_forwards_chunks(tmp_path))

    async def _test_proxy_stream_forwards_chunks(self, tmp_path):
        """The dashboard proxy must forward SSE chunks unchanged AND
        attach the per-server bearer token on the outbound request.
        We mock httpx.AsyncClient.stream to verify both."""
        from neural_platform.web.inference_manager import (
            InferenceServerManager, _ManagedServer, ServerInfo,
        )
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        # Hand-build a managed entry so we don't need a real subprocess.
        info = ServerInfo(id="abc", name="x", port=9000, pid=1,
                            status="running")
        proc = MagicMock()
        proc.poll.return_value = None
        managed = _ManagedServer(info=info, proc=proc, token="t-secret-xyz")
        mgr._servers["abc"] = managed
        # Skip healthcheck — the test forces "running" status.
        mgr._healthcheck_ok = lambda m: True

        # Fake httpx.AsyncClient.stream: returns an async context manager
        # whose body is an async iterator of bytes. Capture the request
        # so we can assert on the Authorization header.
        captured = {}

        class _FakeResp:
            status_code = 200
            async def aiter_bytes(self):
                for b in [b"event: token\ndata: \"a\"\n\n",
                           b"event: token\ndata: \"b\"\n\n",
                           b"event: done\ndata: {\"text\":\"ab\"}\n\n"]:
                    yield b

        class _FakeStreamCtx:
            def __init__(self, resp): self._resp = resp
            async def __aenter__(self): return self._resp
            async def __aexit__(self, *a): return False

        class _FakeAsyncClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            def stream(self, method, url, headers=None, json=None):
                captured["method"]  = method
                captured["url"]     = url
                captured["headers"] = headers or {}
                captured["json"]    = json
                return _FakeStreamCtx(_FakeResp())

        with patch("httpx.AsyncClient", _FakeAsyncClient):
            agen = mgr.proxy_stream(
                "abc", "/predict/stream", json_body={"text": "hi"},
            )
            collected = b""
            async for chunk in agen:
                collected += chunk

        # All three SSE events forwarded unchanged.
        assert b"event: token" in collected
        assert b"event: done" in collected
        # Bearer token attached on the outbound request — never appears
        # in the bytes we yielded back.
        assert captured["headers"].get("Authorization") == "Bearer t-secret-xyz"
        assert b"t-secret-xyz" not in collected
        # JSON body forwarded.
        assert captured["json"] == {"text": "hi"}
        # Method + URL composed correctly.
        assert captured["method"] == "POST"
        assert captured["url"].endswith(":9000/predict/stream")

    def test_proxy_stream_emits_error_on_upstream_4xx(self, tmp_path):
        import asyncio
        asyncio.run(self._test_proxy_stream_emits_error_on_upstream_4xx(tmp_path))

    async def _test_proxy_stream_emits_error_on_upstream_4xx(self, tmp_path):
        """If the inference subprocess 400s (e.g. non-generative task),
        the proxy should yield a single SSE error event instead of
        leaking the raw HTTP error to the browser."""
        from neural_platform.web.inference_manager import (
            InferenceServerManager, _ManagedServer, ServerInfo,
        )
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        info = ServerInfo(id="abc", name="x", port=9000, pid=1,
                            status="running")
        proc = MagicMock(); proc.poll.return_value = None
        mgr._servers["abc"] = _ManagedServer(
            info=info, proc=proc, token="t-secret-xyz",
        )
        mgr._healthcheck_ok = lambda m: True

        class _FakeResp:
            status_code = 400
            async def aread(self): return b"task isn't generative"

        class _FakeStreamCtx:
            def __init__(self, resp): self._resp = resp
            async def __aenter__(self): return self._resp
            async def __aexit__(self, *a): return False

        class _FakeAsyncClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            def stream(self, *a, **kw):
                return _FakeStreamCtx(_FakeResp())

        with patch("httpx.AsyncClient", _FakeAsyncClient):
            agen = mgr.proxy_stream("abc", "/predict/stream",
                                            json_body={})
            collected = b""
            async for chunk in agen:
                collected += chunk

        assert b"event: error" in collected
        assert b"isn" in collected   # detail forwarded
