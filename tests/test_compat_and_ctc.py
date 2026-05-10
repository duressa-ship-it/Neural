"""
Tests for the round of fixes that landed alongside structured outputs
+ streaming + chat:

  * **proxy_stream** is a *regular function* that returns an async
    generator — not an ``async def``. Confirms `async for` over the
    result works without awaiting the call.
  * **CTC ASR support** — the auto-class chain for
    ``automatic-speech-recognition`` now includes
    ``AutoModelForCTC`` for models like google/medasr / Wav2Vec2.
    The predict path detects the absence of ``.generate()`` and
    runs the forward + argmax + decode flow instead.
  * **HF load-report capture** — when transformers prints a
    LOAD REPORT to stderr (UNEXPECTED / MISSING keys), the wrapper
    captures it and stashes a parsed summary on the model. /info
    surfaces this so users can see when their model loaded with
    randomly-initialized weights.
  * **transformers compat shim** — older ``trust_remote_code`` repos
    import names that moved between transformers releases. The shim
    re-exports those names at their old paths so legacy modeling
    code keeps loading.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# proxy_stream — the async-for fix
# ---------------------------------------------------------------------------

class TestProxyStreamIsRegularFunction:
    """The bug: proxy_stream was `async def` so calling it without
    `await` produced a coroutine, which ASGI tried to `async for`
    over and crashed with TypeError. The fix makes proxy_stream a
    regular function that returns the async generator directly."""

    def test_returns_async_iterable_synchronously(self, tmp_path):
        """Calling proxy_stream(...) without `await` returns an object
        you can `async for` over. Pin this — regressing it would
        re-break the entire streaming proxy."""
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

        # Without `await` — the regression test.
        result = mgr.proxy_stream("abc", "/predict/stream", json_body={})
        # An async generator has __aiter__ AND __anext__.
        assert hasattr(result, "__aiter__")
        assert hasattr(result, "__anext__")


# ---------------------------------------------------------------------------
# CTC ASR — AutoModelForCTC fallback
# ---------------------------------------------------------------------------

class TestASRCtcFallback:
    """The ASR spec used to chain only AutoModelForSpeechSeq2Seq, which
    blew up on CTC ASR models (google/medasr, wav2vec2-base-960h, MMS,
    etc.). The chain now also includes AutoModelForCTC."""

    def test_asr_chain_includes_ctc(self):
        from neural_platform.core.pipeline_specs import resolve
        spec = resolve("automatic-speech-recognition")
        # Whisper (Seq2Seq) stays the preferred class.
        assert spec.auto_classes[0] == "AutoModelForSpeechSeq2Seq"
        # CTC families resolve via the fallback.
        assert "AutoModelForCTC" in spec.auto_classes

    def test_resolver_picks_ctc_when_seq2seq_missing(self):
        """Simulates a transformers install where AutoModelForCTC is
        present but AutoModelForSpeechSeq2Seq isn't (artificial, but
        confirms the chain probes both)."""
        from neural_platform.core.pipeline_specs import resolve, resolve_auto_class
        spec = resolve("automatic-speech-recognition")
        fake = SimpleNamespace(
            AutoModelForCTC=MagicMock(name="AutoModelForCTC"),
            AutoModel=MagicMock(name="AutoModel"),
        )
        cls, name = resolve_auto_class(fake, spec)
        assert name == "AutoModelForCTC"


# ---------------------------------------------------------------------------
# Load-report capture
# ---------------------------------------------------------------------------

class TestLoadReportCapture:
    """transformers prints a LOAD REPORT to stderr when checkpoint
    keys don't match the loaded auto class. We tee that into a buffer
    and parse the UNEXPECTED / MISSING lines so /info can show users
    when their model loaded with random head weights."""

    def test_captures_load_report(self):
        from neural_platform.models.hf_pipeline import _capture_hf_load_report
        import sys
        with _capture_hf_load_report() as cap:
            # Simulate transformers' output. The capture must detect
            # the LOAD REPORT banner AND extract the key statuses.
            sys.stderr.write("[transformers] FooModel LOAD REPORT from foo/bar\n")
            sys.stderr.write("model.layers.0.attn.qkv.weight | UNEXPECTED | extra\n")
            sys.stderr.write("model.layers.0.mlp.gate.weight | MISSING    | new\n")
        assert "UNEXPECTED" in cap.get("text", "")
        assert cap["unexpected"] == ["model.layers.0.attn.qkv.weight"]
        assert cap["missing"]    == ["model.layers.0.mlp.gate.weight"]

    def test_no_capture_when_load_was_clean(self):
        """If transformers doesn't print a LOAD REPORT, the captured
        dict stays empty — /info won't surface a misleading 'all
        clear' banner."""
        from neural_platform.models.hf_pipeline import _capture_hf_load_report
        import sys
        with _capture_hf_load_report() as cap:
            sys.stderr.write("Loading weights: 100% 5/5\n")  # tqdm-ish
        assert cap == {}

    def test_capture_does_not_swallow_stderr(self, capsys):
        """The capture tees rather than redirects — the real stderr
        still sees the output so operator-facing tqdm/progress bars
        remain visible."""
        from neural_platform.models.hf_pipeline import _capture_hf_load_report
        import sys
        with _capture_hf_load_report():
            sys.stderr.write("[transformers] FooModel LOAD REPORT\n")
        captured = capsys.readouterr()
        # The text leaks through to the real stderr (capsys captures it).
        assert "LOAD REPORT" in captured.err


# ---------------------------------------------------------------------------
# transformers compat shim
# ---------------------------------------------------------------------------

class TestHFCompatShim:

    def test_install_is_idempotent(self):
        from neural_platform.core.hf_compat import (
            install_compat_shims, reset_for_testing,
        )
        reset_for_testing()
        first = install_compat_shims()
        # Second call no-ops.
        second = install_compat_shims()
        assert second == {}, "Second call should return empty (idempotent)"
        reset_for_testing()

    def test_restores_missing_symbol_when_new_location_exposes_it(self):
        """The actual shim behavior: when transformers.generation
        doesn't expose DisjunctiveConstraint but
        transformers.generation.beam_constraints does, we re-bind it."""
        from neural_platform.core.hf_compat import (
            install_compat_shims, reset_for_testing,
        )
        try:
            import transformers.generation as gen_mod
            import transformers.generation.beam_constraints as bc_mod
        except ImportError:
            pytest.skip("transformers not installed")

        # Force the "missing" state: delete DisjunctiveConstraint from
        # the old location if it's there.
        had = hasattr(gen_mod, "DisjunctiveConstraint")
        if had:
            del gen_mod.DisjunctiveConstraint
        reset_for_testing()
        try:
            statuses = install_compat_shims()
            # If the symbol exists at the new location, the shim
            # restored it; if not, we mark it unavailable.
            key = "transformers.generation.DisjunctiveConstraint"
            if hasattr(bc_mod, "DisjunctiveConstraint"):
                assert statuses[key] == "shimmed"
                assert hasattr(gen_mod, "DisjunctiveConstraint")
            else:
                assert statuses[key] in ("unavailable", "present")
        finally:
            reset_for_testing()
            # Restore the symbol if we removed it, so other tests
            # see the same library shape as before.
            if had and hasattr(bc_mod, "DisjunctiveConstraint"):
                gen_mod.DisjunctiveConstraint = bc_mod.DisjunctiveConstraint


# ---------------------------------------------------------------------------
# /info exposes load_warnings + CTC path end-to-end
# ---------------------------------------------------------------------------

class TestInfoLoadWarnings:
    """When the wrapper captured a non-empty LOAD REPORT, /info should
    surface a structured summary so the UI can warn users that weights
    were partially random-initialized."""

    def test_info_includes_load_warnings_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        from neural_platform.deploy.server import create_inference_app
        from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter
        from fastapi.testclient import TestClient

        # Stub model whose encoder carries a pre-baked _nf_load_report.
        import torch.nn as nn
        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(id2label={}, num_labels=0)
                self._nf_load_report = {
                    "text":       "FAKE LOAD REPORT",
                    "missing":    ["m.layer.0.weight", "m.layer.1.weight"],
                    "unexpected": ["m.legacy.weight"],
                }
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Enc()
            def count_parameters(self, trainable_only=False): return 0

        cfg = ExperimentConfig(
            name="x", output_dir="/tmp",
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(pretrained="org/odd-model"),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION,
                                     pipeline_task="text-classification"),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        monkeypatch.setattr(PyTorchAdapter, "build_model", lambda self: _Stub())
        from neural_platform.deploy import server as _server
        monkeypatch.setattr(_server, "_try_load_processor", lambda c: None)

        app = create_inference_app(cfg, checkpoint_path=None)
        with TestClient(app) as client:
            r = client.get("/info")
        assert r.status_code == 200, r.text
        body = r.json()
        warns = body["load_warnings"]
        assert warns is not None
        assert warns["missing_count"] == 2
        assert warns["unexpected_count"] == 1
        # Examples truncated to a small list — JSON-safe.
        assert warns["missing_examples"] == [
            "m.layer.0.weight", "m.layer.1.weight",
        ]


class TestCtcPredictRouting:
    """End-to-end: when the loaded ASR model has no .generate() (CTC
    family), /predict runs forward + argmax + decode instead of
    raising 'task needs .generate()'. Verifies the predict path
    branches correctly."""

    def test_ctc_predict_runs_forward_and_decodes(self, tmp_path, monkeypatch):
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

        # CTC encoder: forward returns per-frame logits, NO generate.
        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, **kwargs):
                # (batch=1, time=4, vocab=8) — fake logits that argmax
                # to ids [2, 2, 3, 0].
                logits = torch.full((1, 4, 8), -10.0)
                logits[0, 0, 2] = 5
                logits[0, 1, 2] = 5
                logits[0, 2, 3] = 5
                logits[0, 3, 0] = 5
                return SimpleNamespace(logits=logits)
            # Crucially NO .generate
        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Enc()
            def count_parameters(self, trainable_only=False): return 0

        # Processor: tokenizes raw audio + has batch_decode. We force
        # the input adapter to accept the audio via `inputs` so we
        # don't need the audio_b64 decoder path here.
        proc = MagicMock()
        proc.return_value = {"input_values": torch.tensor([[0.0] * 1024])}
        proc.batch_decode = MagicMock(return_value=["hello"])
        proc.feature_extractor = SimpleNamespace(sampling_rate=16000)

        cfg = ExperimentConfig(
            name="ctc", output_dir="/tmp",
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(pretrained="google/medasr"),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION,
                                     pipeline_task="automatic-speech-recognition"),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        monkeypatch.setattr(PyTorchAdapter, "build_model", lambda self: _Stub())
        from neural_platform.deploy import server as _server
        monkeypatch.setattr(_server, "_try_load_processor", lambda c: proc)

        app = create_inference_app(cfg, checkpoint_path=None)
        with TestClient(app) as client:
            # Send a raw waveform via `inputs`.
            r = client.post("/predict", json={"inputs": [0.0] * 1024})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["result_kind"] == "generated_text"
        assert body["predictions"][0][0]["class_name"] == "hello"
        # batch_decode was called with the argmax tensor (shape (1, 4)).
        proc.batch_decode.assert_called_once()
