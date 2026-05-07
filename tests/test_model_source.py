"""
Unit tests for the pluggable model source layer + inspector + resource fit.

Designed to be deterministic and offline — no real HF Hub calls. Each test
constructs a `ModelInfo` by hand and feeds it to `_check_task_compat` /
`_check_modality_compat` so the inspector behaviour is verified without
network or auth.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from neural_platform.core.model_source import (
    CompatReport,
    HFModelSource,
    LocalCheckpointSource,
    ModelCard,
    ModelInfo,
    _check_loading_pattern,
    _check_modality_compat,
    _check_task_compat,
    detect_loading_pattern,
    is_standard_loadable,
    pipeline_to_modality,
    register_source,
    registered_sources,
)
from neural_platform.core.resource_fit import (
    HostResources,
    add_dataset_footprint,
    check_fit,
    estimate_model_footprint,
)


# ---------------------------------------------------------------------------
# pipeline_to_modality
# ---------------------------------------------------------------------------

class TestPipelineToModality:

    def test_audio_pipelines_map_to_audio(self):
        assert pipeline_to_modality("automatic-speech-recognition") == "audio"
        assert pipeline_to_modality("audio-classification") == "audio"

    def test_text_pipelines_map_to_text(self):
        assert pipeline_to_modality("text-classification") == "text"
        assert pipeline_to_modality("token-classification") == "text"

    def test_image_pipelines_map_to_image(self):
        assert pipeline_to_modality("image-classification") == "image"
        assert pipeline_to_modality("object-detection") == "image"

    def test_unknown_returns_none(self):
        assert pipeline_to_modality("not-a-real-task") is None

    def test_none_returns_none(self):
        assert pipeline_to_modality(None) is None

    def test_case_insensitive(self):
        assert pipeline_to_modality("Text-Classification") == "text"


# ---------------------------------------------------------------------------
# _check_task_compat — the Whisper-vs-IMDB style mismatch
# ---------------------------------------------------------------------------

class TestTaskCompat:

    def _whisper_info(self) -> ModelInfo:
        return ModelInfo(
            id="openai/whisper-tiny",
            source="huggingface",
            pipeline_tag="automatic-speech-recognition",
            modality="audio",
            architectures=["WhisperForConditionalGeneration"],
        )

    def _bert_info(self) -> ModelInfo:
        return ModelInfo(
            id="bert-base-uncased",
            source="huggingface",
            pipeline_tag="fill-mask",
            modality="text",
            architectures=["BertForMaskedLM"],
        )

    def _make_report(self, info, intended):
        return CompatReport(
            model_id=info.id,
            source=info.source,
            intended_task=intended,
            detected_pipeline=info.pipeline_tag,
            detected_modality=info.modality,
            info=info,
        )

    def test_whisper_with_text_classification_is_error(self):
        info = self._whisper_info()
        report = self._make_report(info, "text-classification")
        _check_task_compat(report, "text-classification", info)
        assert not report.ok
        codes = [i.code for i in report.issues]
        assert "task_modality_mismatch" in codes

    def test_whisper_with_asr_is_clean(self):
        info = self._whisper_info()
        report = self._make_report(info, "automatic-speech-recognition")
        _check_task_compat(report, "automatic-speech-recognition", info)
        assert report.ok
        assert not report.issues

    def test_bert_with_token_classification_is_warning(self):
        # Same modality (text) but different task — should warn, not error.
        info = self._bert_info()
        report = self._make_report(info, "token-classification")
        _check_task_compat(report, "token-classification", info)
        assert report.ok  # warnings, not errors
        codes = [i.code for i in report.issues]
        assert "task_mismatch_same_modality" in codes

    def test_no_intended_task_is_silent(self):
        info = self._whisper_info()
        report = self._make_report(info, None)
        _check_task_compat(report, None, info)
        assert report.ok
        assert not report.issues

    def test_arch_heuristic_catches_audio_arch_with_text_task(self):
        """When pipeline_tag is missing, fall back to architecture sniffing."""
        info = ModelInfo(
            id="custom/whisper-fork",
            source="huggingface",
            pipeline_tag=None,
            architectures=["WhisperForConditionalGeneration"],
        )
        report = self._make_report(info, "text-classification")
        _check_task_compat(report, "text-classification", info)
        assert not report.ok
        assert any(i.code == "task_modality_mismatch" for i in report.issues)


# ---------------------------------------------------------------------------
# _check_modality_compat
# ---------------------------------------------------------------------------

class TestModalityCompat:

    def test_text_dataset_with_audio_model_errors(self):
        info = ModelInfo(
            id="openai/whisper-tiny", source="huggingface",
            pipeline_tag="automatic-speech-recognition",
            modality="audio",
        )
        report = CompatReport(
            model_id=info.id, source=info.source,
            intended_task=None, detected_pipeline=info.pipeline_tag,
            detected_modality=info.modality, info=info,
        )
        _check_modality_compat(report, "text", info)
        assert not report.ok
        assert any(i.code == "data_model_modality_mismatch" for i in report.issues)

    def test_matching_modalities_clean(self):
        info = ModelInfo(
            id="bert-base-uncased", source="huggingface",
            pipeline_tag="fill-mask", modality="text",
        )
        report = CompatReport(
            model_id=info.id, source=info.source,
            intended_task=None, detected_pipeline="fill-mask",
            detected_modality="text", info=info,
        )
        _check_modality_compat(report, "text", info)
        assert report.ok


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

class TestSourceRegistry:

    def test_default_sources_registered(self):
        names = [s.name for s in registered_sources()]
        assert "huggingface" in names
        assert "local" in names

    def test_register_source_idempotent(self):
        # Re-registering the default doesn't crash and replaces the entry.
        before = [s.name for s in registered_sources()]
        register_source(HFModelSource())
        after = [s.name for s in registered_sources()]
        assert sorted(before) == sorted(after)


# ---------------------------------------------------------------------------
# HFModelSource — without hitting the network
# ---------------------------------------------------------------------------

class TestHFModelSourceMocked:

    def test_row_to_card_extracts_modality_from_pipeline_tag(self):
        src = HFModelSource()
        card = src._row_to_card({
            "id": "openai/whisper-tiny",
            "pipeline_tag": "automatic-speech-recognition",
            "tags": ["audio", "speech"],
            "downloads": 12345,
        })
        assert card.id == "openai/whisper-tiny"
        assert card.pipeline_tag == "automatic-speech-recognition"
        assert card.modality == "audio"
        assert card.downloads == 12345

    def test_inspect_compat_handles_unreachable_hub(self):
        src = HFModelSource()
        with patch.object(src, "get_info", side_effect=RuntimeError("network down")):
            report = src.inspect_compat("any/model", intended_task="text-classification")
        # Don't error — degrade to a warning so offline runs aren't blocked.
        assert report.ok
        assert any(i.code == "info_unreachable" for i in report.issues)


# ---------------------------------------------------------------------------
# Resource fit
# ---------------------------------------------------------------------------

class TestResourceFit:

    def test_estimate_size_from_parameters(self):
        est = estimate_model_footprint(parameters=1_000_000, size_bytes=None,
                                        purpose="training", optimizer="adamw")
        # 1M params * 4 = 4MB weights, 4MB grads, 8MB Adam moments ≈ 16MB before activations
        assert est.model_weight_b == 4_000_000
        assert est.gradients_b == 4_000_000
        assert est.optimizer_b == 8_000_000
        assert est.runtime_total_b > est.model_weight_b

    def test_estimate_inference_is_smaller_than_training(self):
        train = estimate_model_footprint(parameters=1_000_000, size_bytes=None,
                                          purpose="training")
        infer = estimate_model_footprint(parameters=1_000_000, size_bytes=None,
                                          purpose="inference")
        assert infer.runtime_total_b < train.runtime_total_b

    def test_size_bytes_used_when_parameters_missing(self):
        est = estimate_model_footprint(parameters=None, size_bytes=4_000_000)
        assert est.model_weight_b > 0
        assert est.download_total_b == 4_000_000

    def test_dataset_size_added_to_download(self):
        est = estimate_model_footprint(parameters=1_000_000, size_bytes=4_000_000)
        before_total = est.download_total_b
        est = add_dataset_footprint(est, 10_000_000)
        assert est.dataset_disk_b == 10_000_000
        assert est.download_total_b == before_total + 10_000_000

    def test_fit_blocks_when_disk_too_small(self):
        host = HostResources(disk_free_b=1_000)  # tiny
        est = estimate_model_footprint(parameters=1_000_000, size_bytes=4_000_000)
        report = check_fit(est, host, purpose="training", device="cpu")
        assert not report.fits
        assert any(i["code"] == "disk_too_small" for i in report.issues)

    def test_fit_blocks_when_vram_too_small_on_gpu(self):
        host = HostResources(
            accelerator="cuda", gpu_count=1, gpu_name="GeForce GT 1010",
            vram_total_b=1_000_000_000,  # 1 GB
        )
        # 100M params training ≈ 1.6 GB — too big for 1 GB
        est = estimate_model_footprint(parameters=100_000_000, size_bytes=None)
        report = check_fit(est, host, purpose="training", device="cuda")
        assert not report.fits
        assert any(i["code"] == "vram_too_small" for i in report.issues)

    def test_fit_warns_when_vram_tight(self):
        host = HostResources(
            accelerator="cuda", gpu_count=1,
            vram_total_b=2_000_000_000,
        )
        # 100M params ≈ 1.6 GB → 80% of 2 GB
        est = estimate_model_footprint(parameters=100_000_000, size_bytes=None)
        report = check_fit(est, host, purpose="training", device="cuda")
        # Should still fit (1.6 < 2.0), but warn.
        assert report.fits
        assert any(i["code"] == "vram_tight" for i in report.issues)

    def test_fit_passes_when_resources_ample(self):
        host = HostResources(
            accelerator="cuda", gpu_count=1, vram_total_b=24_000_000_000,
            ram_total_b=64_000_000_000, disk_free_b=500_000_000_000,
        )
        est = estimate_model_footprint(parameters=10_000_000, size_bytes=40_000_000)
        report = check_fit(est, host, purpose="training", device="cuda")
        assert report.fits
        assert not [i for i in report.issues if i["severity"] == "error"]


# ---------------------------------------------------------------------------
# Loading pattern detection — the felixwangg/Qwen scenario
# ---------------------------------------------------------------------------

class TestLoadingPatternDetection:

    def test_peft_adapter_detected(self):
        siblings = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json"]
        assert detect_loading_pattern(siblings) == "peft_adapter"
        assert not is_standard_loadable("peft_adapter")

    def test_safetensors_standard(self):
        siblings = ["config.json", "model.safetensors", "tokenizer.json"]
        assert detect_loading_pattern(siblings) == "safetensors"
        assert is_standard_loadable("safetensors")

    def test_pytorch_bin_standard(self):
        siblings = ["config.json", "pytorch_model.bin", "vocab.txt"]
        assert detect_loading_pattern(siblings) == "pytorch_bin"
        assert is_standard_loadable("pytorch_bin")

    def test_sharded_safetensors(self):
        siblings = [
            "config.json", "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        assert detect_loading_pattern(siblings) == "sharded_safetensors"
        assert is_standard_loadable("sharded_safetensors")

    def test_gguf_detected(self):
        siblings = ["model-q4_K_M.gguf", "config.json"]
        assert detect_loading_pattern(siblings) == "gguf"
        assert not is_standard_loadable("gguf")

    def test_diffusers_detected(self):
        siblings = ["model_index.json", "unet/diffusion_pytorch_model.safetensors"]
        assert detect_loading_pattern(siblings) == "diffusers"
        assert not is_standard_loadable("diffusers")

    def test_onnx_detected(self):
        siblings = ["config.json", "model.onnx"]
        assert detect_loading_pattern(siblings) == "onnx"

    def test_empty_siblings_unknown(self):
        assert detect_loading_pattern([]) == "unknown"
        assert detect_loading_pattern(None) == "unknown"


class TestLoadingPatternInspectorIssues:

    def _report(self, info):
        return CompatReport(
            model_id=info.id, source=info.source,
            intended_task=None, detected_pipeline=info.pipeline_tag,
            detected_modality=info.modality, info=info,
        )

    def test_peft_adapter_blocks_with_clear_error(self):
        info = ModelInfo(
            id="felixwangg/Qwen2.5-Coder-7B-sft",
            source="huggingface",
            siblings=["adapter_config.json", "adapter_model.safetensors"],
            loading_pattern="peft_adapter",
            base_model="Qwen/Qwen2.5-Coder-7B",
            standard_loadable=False,
        )
        report = self._report(info)
        _check_loading_pattern(report, info)
        assert not report.ok
        codes = [i.code for i in report.issues]
        assert "peft_adapter_required" in codes
        # The hint should mention peft installation and the base model
        msg = next(i for i in report.issues if i.code == "peft_adapter_required")
        assert "peft" in (msg.hint or "").lower()
        assert "Qwen/Qwen2.5-Coder-7B" in msg.message

    def test_gguf_blocks(self):
        info = ModelInfo(id="some/model-gguf", source="huggingface",
                          loading_pattern="gguf", standard_loadable=False)
        report = self._report(info)
        _check_loading_pattern(report, info)
        assert not report.ok
        assert any(i.code == "gguf_unsupported" for i in report.issues)

    def test_diffusers_blocks(self):
        info = ModelInfo(id="runwayml/stable-diffusion-v1-5", source="huggingface",
                          loading_pattern="diffusers", standard_loadable=False)
        report = self._report(info)
        _check_loading_pattern(report, info)
        assert not report.ok
        assert any(i.code == "diffusers_required" for i in report.issues)

    def test_standard_safetensors_clean(self):
        info = ModelInfo(id="bert-base-uncased", source="huggingface",
                          loading_pattern="safetensors", standard_loadable=True)
        report = self._report(info)
        _check_loading_pattern(report, info)
        assert report.ok
        assert not report.issues


# ---------------------------------------------------------------------------
# LocalCheckpointSource
# ---------------------------------------------------------------------------

class TestLocalCheckpointSource:

    def test_search_with_no_roots_returns_empty(self, tmp_path):
        src = LocalCheckpointSource(roots=[tmp_path / "nonexistent"])
        assert src.search() == []

    def test_search_finds_pt_files(self, tmp_path):
        (tmp_path / "checkpoint_best.pt").write_bytes(b"\x80\x04")
        (tmp_path / "checkpoint_epoch_5.pt").write_bytes(b"\x80\x04")
        src = LocalCheckpointSource(roots=[tmp_path])
        results = src.search()
        assert len(results) == 2
        ids = {r.id for r in results}
        assert any("checkpoint_best.pt" in i for i in ids)


# ---------------------------------------------------------------------------
# Model id validation — security + UX
# ---------------------------------------------------------------------------

class TestModelIdValidation:
    """The HF API path used to forward arbitrary strings (`\\`, `..`,
    `?token=...`) to the Hub, which 302-redirected and made the
    inspector misclassify junk as 'compatible'. validate_hf_model_id
    rejects these up front; HFModelSource also turns Hub redirects into
    explicit errors."""

    def test_accepts_canonical_owner_repo(self):
        from neural_platform.core.model_source import validate_hf_model_id
        assert validate_hf_model_id("openai/whisper-tiny") == "openai/whisper-tiny"
        assert validate_hf_model_id("bert-base-uncased") == "bert-base-uncased"
        assert validate_hf_model_id("Qwen/Qwen2.5-Coder-7B") == "Qwen/Qwen2.5-Coder-7B"

    def test_strips_huggingface_url_prefix(self):
        from neural_platform.core.model_source import validate_hf_model_id
        out = validate_hf_model_id("https://huggingface.co/openai/whisper-tiny")
        assert out == "openai/whisper-tiny"

    def test_rejects_backslash(self):
        from neural_platform.core.model_source import (
            validate_hf_model_id, InvalidModelIdError,
        )
        with pytest.raises(InvalidModelIdError):
            validate_hf_model_id("\\")

    def test_rejects_path_traversal(self):
        from neural_platform.core.model_source import (
            validate_hf_model_id, InvalidModelIdError,
        )
        for bad in ("..", "../etc/passwd", "owner/..", "/etc/passwd"):
            with pytest.raises(InvalidModelIdError):
                validate_hf_model_id(bad)

    def test_rejects_query_strings(self):
        from neural_platform.core.model_source import (
            validate_hf_model_id, InvalidModelIdError,
        )
        with pytest.raises(InvalidModelIdError):
            validate_hf_model_id("openai/whisper?token=hf_xxx")
        with pytest.raises(InvalidModelIdError):
            validate_hf_model_id("openai/whisper#frag")

    def test_rejects_empty_or_whitespace(self):
        from neural_platform.core.model_source import (
            validate_hf_model_id, InvalidModelIdError,
        )
        for bad in ("", "  ", "\t", "\n"):
            with pytest.raises(InvalidModelIdError):
                validate_hf_model_id(bad)

    def test_rejects_other_schemes(self):
        from neural_platform.core.model_source import (
            validate_hf_model_id, InvalidModelIdError,
        )
        with pytest.raises(InvalidModelIdError):
            validate_hf_model_id("file:///etc/passwd")
        with pytest.raises(InvalidModelIdError):
            validate_hf_model_id("ftp://huggingface.co/owner/repo")

    def test_inspector_returns_invalid_id_error(self):
        """Public inspector surface — what the dashboard /api/models/inspect
        endpoint actually calls. Bad ids must NOT be silently treated as
        compatible (the previous bug)."""
        from unittest.mock import patch
        src = HFModelSource()
        with patch.object(src, "get_info") as get_info:
            r = src.inspect_compat("\\", intended_task=None)
            # Critical: get_info MUST NOT have been called — we should reject
            # the id before any HTTP request goes out.
            get_info.assert_not_called()
        assert not r.ok
        assert any(i.code == "invalid_model_id" for i in r.issues)


class TestHFRedirectHardening:
    """The Hub uses 307s to canonicalize legitimate short names (e.g.
    `distilbert-base-uncased-finetuned-sst-2-english` → same path on
    `huggingface.co`). Same-host redirects are safe to follow; cross-
    origin redirects are the actual attack surface and stay rejected."""

    class _Resp:
        """Tiny stand-in for httpx.Response. Defined at the class level
        so tests can build response scripts before the fake client wraps
        them."""
        def __init__(self, status_code, headers=None, json_body=None):
            self.status_code = status_code
            self.headers = headers or {}
            self._json = json_body
        def json(self): return self._json
        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    def _client_factory(self, scripted_responses):
        """Build a fake httpx.Client class that yields one scripted
        response per GET, in order. Returns (ClientClass, seen_urls_list)
        where seen_urls is appended to on each GET so tests can assert
        on the redirect chain."""
        responses = list(scripted_responses)
        seen_urls = []

        class _Client:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url, params=None):
                seen_urls.append(url)
                if not responses:
                    return TestHFRedirectHardening._Resp(500)
                return responses.pop(0)
        return _Client, seen_urls

    def test_same_host_redirect_followed(self):
        """A 307 from `/api/models/foo` → `/api/models/foo/` (HF's
        canonicalization) must be followed, not rejected — this is the
        regression that broke `distilbert-base-uncased-finetuned-...`."""
        from neural_platform.core.model_source import HFModelSource
        good = {"id": "distilbert-base-uncased-finetuned-sst-2-english",
                "modelId": "distilbert-base-uncased-finetuned-sst-2-english"}
        Client, seen = self._client_factory([
            self._Resp(307, headers={"Location": "https://huggingface.co/api/models/distilbert-base-uncased-finetuned-sst-2-english/"}),
            self._Resp(200, json_body=good),
        ])

        from unittest.mock import patch
        import httpx
        src = HFModelSource()
        with patch.object(httpx, "Client", Client):
            out = src._get_json(
                "https://huggingface.co/api/models/distilbert-base-uncased-finetuned-sst-2-english"
            )
        assert out == good
        # Two GETs went out: original + the canonical URL the 307 pointed at.
        assert len(seen) == 2

    def test_cross_origin_redirect_rejected(self):
        """A 302 to a different host (or scheme) is the original attack
        surface — open-ended redirect to a search page or third-party
        URL. Still rejected with a clear error."""
        from neural_platform.core.model_source import HFModelSource
        Client, _ = self._client_factory([
            self._Resp(302, headers={"Location": "https://attacker.example.com/leak"}),
        ])
        from unittest.mock import patch
        import httpx
        src = HFModelSource()
        with patch.object(httpx, "Client", Client):
            with pytest.raises(RuntimeError, match="different origin"):
                src._get_json("https://huggingface.co/api/models/bad")

    def test_relative_redirect_resolved_against_base(self):
        """HF often returns a relative `Location` header (e.g.
        `/api/models/canonical-id`). We must resolve it against the
        original URL so the same-host check fires correctly."""
        from neural_platform.core.model_source import HFModelSource
        Client, seen = self._client_factory([
            self._Resp(307, headers={"Location": "/api/models/canonical-id"}),
            self._Resp(200, json_body={"id": "canonical-id"}),
        ])
        from unittest.mock import patch
        import httpx
        src = HFModelSource()
        with patch.object(httpx, "Client", Client):
            out = src._get_json("https://huggingface.co/api/models/short-name")
        assert out == {"id": "canonical-id"}
        # Second GET should have been against the absolute resolved URL.
        assert seen[1] == "https://huggingface.co/api/models/canonical-id"

    def test_scheme_change_rejected(self):
        """https → http downgrade is suspicious — reject."""
        from neural_platform.core.model_source import HFModelSource
        Client, _ = self._client_factory([
            self._Resp(302, headers={"Location": "http://huggingface.co/api/models/x"}),
        ])
        from unittest.mock import patch
        import httpx
        src = HFModelSource()
        with patch.object(httpx, "Client", Client):
            with pytest.raises(RuntimeError, match="different origin"):
                src._get_json("https://huggingface.co/api/models/x")

    def test_redirect_loop_terminates(self):
        """If the Hub somehow redirects forever, give up after a few
        hops with a clear error rather than hanging."""
        from neural_platform.core.model_source import HFModelSource
        # Always 307 to itself (same host) — would loop forever without the cap.
        Client, _ = self._client_factory([
            self._Resp(307, headers={"Location": "https://huggingface.co/api/models/loop"})
            for _ in range(10)
        ])
        from unittest.mock import patch
        import httpx
        src = HFModelSource()
        with patch.object(httpx, "Client", Client):
            with pytest.raises(RuntimeError, match="redirect loop"):
                src._get_json("https://huggingface.co/api/models/loop")

    def test_missing_location_header_rejected(self):
        """A 3xx with no Location header is a server-side bug we shouldn't
        try to chase — surface a clear error instead of looping."""
        from neural_platform.core.model_source import HFModelSource
        Client, _ = self._client_factory([self._Resp(307, headers={})])
        from unittest.mock import patch
        import httpx
        src = HFModelSource()
        with patch.object(httpx, "Client", Client):
            with pytest.raises(RuntimeError, match="without a Location"):
                src._get_json("https://huggingface.co/api/models/x")


# ---------------------------------------------------------------------------
# HF auth — token discovery + redaction (fully offline)
# ---------------------------------------------------------------------------

class TestHfAuth:

    def test_redact_scrubs_hf_token_pattern(self):
        from neural_platform.core.hf_auth import redact
        msg = "401 from https://hf.co/...?token=hf_AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRr"
        assert "hf_" not in redact(msg)
        assert "REDACTED" in redact(msg)

    def test_redact_scrubs_bearer_header(self):
        from neural_platform.core.hf_auth import redact
        msg = "Authorization: Bearer hf_AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRr"
        out = redact(msg)
        assert "hf_AaBbCc" not in out

    def test_set_token_for_session_overrides_env(self, monkeypatch):
        # Clear any real env so the test is deterministic
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        from neural_platform.core import hf_auth
        try:
            hf_auth.set_token_for_session("hf_test_session_token_12345678901234567890")
            assert hf_auth.is_authenticated()
            assert hf_auth.get_token().startswith("hf_test_session")
        finally:
            hf_auth.set_token_for_session(None)
        assert not hf_auth.is_authenticated()

    def test_auth_status_returns_no_token_material(self, monkeypatch):
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        from neural_platform.core import hf_auth
        try:
            hf_auth.set_token_for_session("hf_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
            # Stub httpx so we don't actually call HF — exercise the path
            # where Hub is unreachable, which should still mark
            # authenticated=True (token was discovered locally) and not
            # leak the token in the dataclass.
            from unittest.mock import patch
            class _Boom(Exception):
                pass
            with patch("httpx.Client", side_effect=_Boom("no network")):
                status = hf_auth.auth_status(timeout_s=1.0)
            assert status.authenticated is True
            d = status.to_dict()
            # The dict must not contain the actual token anywhere.
            assert "hf_zzzzz" not in str(d)
        finally:
            hf_auth.set_token_for_session(None)

    def test_inspector_blocks_gated_without_auth(self, monkeypatch):
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        from neural_platform.core import hf_auth
        hf_auth.set_token_for_session(None)
        info = ModelInfo(
            id="z-lab/Qwen3.6-27B-DFlash", source="huggingface",
            gated=True, loading_pattern="safetensors", standard_loadable=True,
        )
        report = CompatReport(
            model_id=info.id, source=info.source,
            intended_task=None, detected_pipeline=None,
            detected_modality=None, info=info,
        )
        # Run the gating path the inspector executes
        from unittest.mock import patch
        src = HFModelSource()
        with patch.object(src, "get_info", return_value=info):
            r = src.inspect_compat("z-lab/Qwen3.6-27B-DFlash")
        assert not r.ok
        assert any(i.code == "auth_required" for i in r.issues)


# ---------------------------------------------------------------------------
# Resource fit — MPS-aware estimate
# ---------------------------------------------------------------------------

class TestMpsFit:

    def test_mps_blocks_when_budget_too_small(self):
        from neural_platform.core.resource_fit import (
            HostResources, estimate_model_footprint, check_fit,
        )
        # Apple Silicon w/ 8 GB recommended_max_memory ceiling
        host = HostResources(
            accelerator="mps", gpu_count=1, gpu_name="Apple Silicon (MPS)",
            vram_total_b=int(8e9), ram_total_b=int(16e9),
        )
        # Qwen 1.47B params trained with AdamW ≈ 23.5 GB — won't fit
        est = estimate_model_footprint(parameters=1_470_000_000,
                                        size_bytes=None,
                                        purpose="training",
                                        batch_size=32,
                                        sequence_length=128)
        report = check_fit(est, host, purpose="training", device="mps")
        assert not report.fits
        assert any(i["code"] == "vram_too_small" for i in report.issues)

    def test_mps_passes_for_small_model(self):
        from neural_platform.core.resource_fit import (
            HostResources, estimate_model_footprint, check_fit,
        )
        host = HostResources(
            accelerator="mps", gpu_count=1, vram_total_b=int(8e9),
            ram_total_b=int(16e9),
        )
        est = estimate_model_footprint(parameters=110_000_000,  # bert-base
                                        size_bytes=None,
                                        purpose="training",
                                        batch_size=8,
                                        sequence_length=128)
        report = check_fit(est, host, purpose="training", device="mps")
        assert report.fits

    def test_activation_estimate_scales_with_batch_and_sequence(self):
        from neural_platform.core.resource_fit import estimate_model_footprint
        small = estimate_model_footprint(
            parameters=110_000_000, size_bytes=None, purpose="training",
            batch_size=4, sequence_length=128,
            hidden_size=768, num_layers=12, num_heads=12,
        )
        big = estimate_model_footprint(
            parameters=110_000_000, size_bytes=None, purpose="training",
            batch_size=64, sequence_length=2048,
            hidden_size=768, num_layers=12, num_heads=12,
        )
        # Both have the same parameter count, but big-batch+long-seq has
        # dramatically more activations.
        assert big.activations_b > small.activations_b * 100
