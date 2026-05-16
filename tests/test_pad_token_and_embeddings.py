"""
Tests for:
  * Bug-1 — wav2vec2 CTC encoder access regression (wrapper now
    surfaces a clear error if self.encoder is missing).
  * Bug-2 — GPT-2 pad_token missing on decoder-only LMs.
  * Embedding rendering — feature-extraction / sentence-similarity
    results return dim + L2 norm + preview, not bare logits.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Bug-1 — wrapper surfaces clear error when self.encoder is missing
# ---------------------------------------------------------------------------

class TestWrapperEncoderGuard:

    def test_forward_raises_clear_error_when_encoder_missing(self):
        """When ``self.encoder`` somehow isn't attached, the wrapper's
        forward used to leak ``AttributeError: 'HFPipelineModel' object
        has no attribute 'encoder'``. The defensive check now produces
        a message that mentions the auto-class mismatch (the most
        common root cause)."""
        from neural_platform.models.hf_pipeline import HFPipelineModel
        # __new__ bypasses __init__ so we can construct an instance
        # without self.encoder being set — exactly the pathological
        # state the bug-1 user hit.
        wrapper = HFPipelineModel.__new__(HFPipelineModel)
        wrapper._task = "automatic-speech-recognition"
        wrapper._fwd_param_names = set()
        import torch
        with pytest.raises(RuntimeError, match="not attached"):
            wrapper.forward(torch.zeros(1, 1024))


# ---------------------------------------------------------------------------
# Bug-2 — GPT-2 pad_token alias to eos_token
# ---------------------------------------------------------------------------

class TestEnsurePadToken:

    def test_aliases_pad_to_eos_when_missing(self):
        """Decoder-only LMs ship with pad_token=None. The loader
        applies the HF-canonical pad_token = eos_token alias so the
        universal text-input adapter's padding="max_length" path
        doesn't 422 with 'Asking to pad but the tokenizer does not
        have a padding token'."""
        from neural_platform.deploy.server import _ensure_pad_token
        tok = SimpleNamespace(pad_token=None, eos_token="</s>")
        _ensure_pad_token(tok)
        assert tok.pad_token == "</s>"

    def test_idempotent_when_pad_already_set(self):
        """A tokenizer that already has a pad_token isn't touched."""
        from neural_platform.deploy.server import _ensure_pad_token
        tok = SimpleNamespace(pad_token="[PAD]", eos_token="</s>")
        _ensure_pad_token(tok)
        assert tok.pad_token == "[PAD]"

    def test_no_op_when_eos_token_also_missing(self):
        """A tokenizer with neither pad_token nor eos_token is
        unusual — bail rather than picking an arbitrary string."""
        from neural_platform.deploy.server import _ensure_pad_token
        tok = SimpleNamespace(pad_token=None, eos_token=None)
        _ensure_pad_token(tok)
        assert tok.pad_token is None

    def test_walks_processor_tokenizer_attr(self):
        """Multimodal processors expose the tokenizer at
        ``processor.tokenizer`` — the helper walks both the processor
        itself and the wrapped tokenizer."""
        from neural_platform.deploy.server import _ensure_pad_token
        inner_tok = SimpleNamespace(pad_token=None, eos_token="<|endoftext|>")
        proc = SimpleNamespace(tokenizer=inner_tok)
        _ensure_pad_token(proc)
        assert inner_tok.pad_token == "<|endoftext|>"

    def test_tolerates_setattr_failure(self):
        """Some processors raise on attribute set when wrapping
        immutable state. The helper must swallow that so the load
        path isn't broken by an exotic tokenizer."""
        from neural_platform.deploy.server import _ensure_pad_token

        class _Stubborn:
            eos_token = "</s>"
            @property
            def pad_token(self): return None
            # Setting pad_token would raise from the property.

        _ensure_pad_token(_Stubborn())   # no exception expected


# ---------------------------------------------------------------------------
# Embedding rendering
# ---------------------------------------------------------------------------

class TestEmbeddingResponse:

    def test_pools_per_token_embeddings_to_doc_vector(self):
        """Models that return last_hidden_state have shape
        (batch, seq, dim). The renderer mean-pools across the seq
        axis to produce one doc-level embedding."""
        from neural_platform.deploy.server import _embeddings_response
        import torch
        # (batch=1, seq=4, dim=8) — a constant per-dim so the mean
        # equals the constant; easy to assert on.
        hidden = torch.tensor([[[0.5] * 8] * 4])
        out = SimpleNamespace(last_hidden_state=hidden)
        resp = _embeddings_response(out, "hf_pipeline", 0.0)
        assert resp.result_kind == "embedding"
        p = resp.predictions[0][0]
        assert p.metadata["dim"] == 8
        # All elements equal 0.5 → L2 norm = sqrt(8 * 0.5^2) = sqrt(2)
        assert p.metadata["l2_norm"] == pytest.approx(1.414, abs=1e-2)
        # Preview keeps the first 16 (or fewer) dims for the sparkline.
        assert p.metadata["preview"] == [0.5] * 8

    def test_handles_bare_tensor_output(self):
        """Some HF models return a tensor directly rather than a
        BaseModelOutput-shaped dataclass. The helper detects this and
        treats the tensor as already-pooled when 1-D."""
        from neural_platform.deploy.server import _embeddings_response
        import torch
        vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
        resp = _embeddings_response(vec, "hf_pipeline", 0.0)
        assert resp.result_kind == "embedding"
        p = resp.predictions[0][0]
        assert p.metadata["dim"] == 4
        assert p.metadata["l2_norm"] == pytest.approx(1.0, abs=1e-3)

    def test_class_name_summary_for_ui(self):
        """The Predict UI's fallback bar-chart renderer reads
        ``class_name`` when it doesn't recognize the result_kind.
        We include dim + ‖v‖ in there so older clients still see
        something useful."""
        from neural_platform.deploy.server import _embeddings_response
        import torch
        resp = _embeddings_response(torch.zeros(1, 4, 8), "hf_pipeline", 0.0)
        cn = resp.predictions[0][0].class_name
        assert "embedding" in cn
        assert "8-d" in cn

    def test_no_embedding_in_output_returns_safe_placeholder(self):
        """If the model's output object has neither
        last_hidden_state nor logits, surface a placeholder rather
        than 500ing the request."""
        from neural_platform.deploy.server import _embeddings_response
        resp = _embeddings_response(SimpleNamespace(), "hf_pipeline", 0.0)
        assert resp.result_kind == "embedding"
        assert "no embedding" in resp.predictions[0][0].class_name.lower()


# ---------------------------------------------------------------------------
# Spec — feature-extraction routes through embedding postproc
# ---------------------------------------------------------------------------

class TestSpecRoutesToEmbeddings:

    def test_feature_extraction_uses_embeddings_postproc(self):
        from neural_platform.core.pipeline_specs import (
            resolve, POSTPROC_EMBEDDINGS,
        )
        spec = resolve("feature-extraction")
        assert spec.output_kind == POSTPROC_EMBEDDINGS

    def test_sentence_similarity_uses_embeddings_postproc(self):
        from neural_platform.core.pipeline_specs import (
            resolve, POSTPROC_EMBEDDINGS,
        )
        spec = resolve("sentence-similarity")
        assert spec.output_kind == POSTPROC_EMBEDDINGS

    def test_default_unknown_task_uses_embeddings_postproc(self):
        """Unknown pipeline_tags fall back to AutoModel + embeddings
        rendering — at least the user sees a valid response shape."""
        from neural_platform.core.pipeline_specs import (
            resolve, POSTPROC_EMBEDDINGS,
        )
        spec = resolve("brand-new-pipeline-2030")
        assert spec.output_kind == POSTPROC_EMBEDDINGS
