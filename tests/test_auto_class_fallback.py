"""
Tests for the transformers v4 → v5 auto-class fallback chain.

Covers the user's exact crash:
  ``AttributeError: module transformers has no attribute AutoModelForVision2Seq``

In transformers v5 ``AutoModelForVision2Seq`` was folded into
``AutoModelForImageTextToText``. The spec table now carries a fallback
chain per task; ``resolve_auto_class()`` probes the installed
transformers and returns the first class that exists. These tests pin
the chain order, the runtime resolver, and the new structured-output
postprocessors (QA / boxes / depth / masks) that depend on the wrapper
returning the raw HF output object instead of just ``.logits``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Spec — auto_classes is a fallback chain
# ---------------------------------------------------------------------------

class TestAutoClassChain:

    def test_image_to_text_chain_prefers_v5_then_falls_back(self):
        """The v5 unified class is preferred. v4's Vision2Seq + the bare
        VisionEncoderDecoderModel fallback follow."""
        from neural_platform.core.pipeline_specs import resolve
        spec = resolve("image-to-text")
        # Preferred is the v5 name.
        assert spec.auto_classes[0] == "AutoModelForImageTextToText"
        # The v4 name we used to hard-code is in the chain.
        assert "AutoModelForVision2Seq" in spec.auto_classes
        # Last-resort generic loader for older HF.
        assert "VisionEncoderDecoderModel" in spec.auto_classes

    def test_any_to_any_spec_exists_with_chain(self):
        """User asked for explicit any-to-any support — verify the spec
        exists and chains through ImageTextToText / CausalLM / AutoModel."""
        from neural_platform.core.pipeline_specs import resolve, INPUT_ANY
        spec = resolve("any-to-any")
        assert spec.task == "any-to-any"
        assert spec.input_kind == INPUT_ANY
        assert spec.needs_generation is True
        assert "AutoModelForImageTextToText" in spec.auto_classes
        assert "AutoModelForCausalLM" in spec.auto_classes
        # AutoModel is the universal fallback for unrecognized model heads.
        assert "AutoModel" in spec.auto_classes

    def test_image_text_to_text_chain(self):
        from neural_platform.core.pipeline_specs import resolve
        spec = resolve("image-text-to-text")
        assert spec.auto_classes[0] == "AutoModelForImageTextToText"
        # CausalLM is the fallback for decoder-only multimodal models that
        # don't expose the unified Auto class.
        assert "AutoModelForCausalLM" in spec.auto_classes

    def test_image_segmentation_handles_v4_v5_rename(self):
        """v5 renamed AutoModelForSemanticSegmentation → AutoModelForImageSegmentation.
        Both names should be in the chain so we work either way."""
        from neural_platform.core.pipeline_specs import resolve
        spec = resolve("image-segmentation")
        assert "AutoModelForImageSegmentation" in spec.auto_classes
        assert "AutoModelForSemanticSegmentation" in spec.auto_classes

    def test_auto_class_property_returns_first_for_back_compat(self):
        """Older tests / consumers reference `spec.auto_class` (singular).
        Keep that property pointing at the first chain entry."""
        from neural_platform.core.pipeline_specs import resolve
        spec = resolve("image-to-text")
        assert spec.auto_class == spec.auto_classes[0]


# ---------------------------------------------------------------------------
# resolve_auto_class — runtime probe
# ---------------------------------------------------------------------------

class TestResolveAutoClass:

    def _fake_transformers(self, present: set[str]):
        """Build a SimpleNamespace exposing ONLY the named Auto classes."""
        ns = SimpleNamespace()
        for name in present:
            setattr(ns, name, MagicMock(name=name))
        return ns

    def test_picks_first_available_class(self):
        from neural_platform.core.pipeline_specs import resolve, resolve_auto_class
        spec = resolve("image-to-text")
        # v5: only AutoModelForImageTextToText exists. The chain should
        # skip past Vision2Seq (absent) and pick the v5 class.
        fake = self._fake_transformers({"AutoModelForImageTextToText"})
        cls, name = resolve_auto_class(fake, spec)
        assert name == "AutoModelForImageTextToText"
        assert cls is fake.AutoModelForImageTextToText

    def test_falls_back_to_v4_class_when_v5_missing(self):
        """If only the v4 name exists (e.g. transformers 4.x install),
        the resolver should still find a usable class — it's the same
        behavior in the other direction."""
        from neural_platform.core.pipeline_specs import resolve, resolve_auto_class
        spec = resolve("image-to-text")
        fake = self._fake_transformers({
            "AutoModelForVision2Seq",
            "AutoModel",
        })
        cls, name = resolve_auto_class(fake, spec)
        assert name == "AutoModelForVision2Seq"

    def test_falls_back_to_automodel_when_chain_empty(self):
        """If NONE of the spec's preferred classes exist but AutoModel does,
        we degrade to AutoModel — server boots, gives back embeddings,
        debuggable."""
        from neural_platform.core.pipeline_specs import resolve, resolve_auto_class
        spec = resolve("image-to-text")
        fake = self._fake_transformers({"AutoModel"})
        cls, name = resolve_auto_class(fake, spec)
        assert name == "AutoModel"

    def test_raises_with_clear_message_when_nothing_resolves(self):
        """When even AutoModel is absent we raise a RuntimeError listing
        every name we tried — replaces the old `module transformers has
        no attribute AutoModelForVision2Seq` confusion."""
        from neural_platform.core.pipeline_specs import resolve, resolve_auto_class
        spec = resolve("image-to-text")
        fake = self._fake_transformers(set())   # totally empty
        with pytest.raises(RuntimeError) as exc_info:
            resolve_auto_class(fake, spec)
        msg = str(exc_info.value)
        # The error mentions every name we tried — actionable.
        for name in spec.auto_classes:
            assert name in msg
        assert "AutoModel" in msg


# ---------------------------------------------------------------------------
# Wrapper integration — the user's specific crash regression-tests here
# ---------------------------------------------------------------------------

class TestWrapperUsesChain:

    def test_resolve_auto_class_helper_in_wrapper(self):
        """The wrapper module exposes a thin helper that just delegates to
        pipeline_specs.resolve_auto_class. Pin it so refactors don't
        accidentally inline a single class lookup again."""
        from neural_platform.models import hf_pipeline as hp
        assert hasattr(hp, "_resolve_auto_class")
        # The helper accepts a transformers-like module + task name.
        fake = SimpleNamespace(
            AutoModelForImageTextToText=MagicMock(),
            AutoModel=MagicMock(),
        )
        cls, name = hp._resolve_auto_class(fake, "image-to-text")
        assert name == "AutoModelForImageTextToText"


# ---------------------------------------------------------------------------
# _build_predictions shape robustness
# ---------------------------------------------------------------------------

class TestBuildPredictionsShapes:
    """The user hit ``a Tensor with 33 elements cannot be converted to
    Scalar`` because the dispatcher tested ``size(0) == 1`` instead of
    ``numel() == 1``. These tests pin the new behavior across every shape
    the HF wrapper might surface."""

    def _call(self, logits, top_k=5, return_probs=True, class_names=None):
        from neural_platform.deploy.server import _build_predictions
        return _build_predictions(logits, top_k, return_probs, class_names)

    def test_multiclass_with_leading_singleton(self):
        """Privacy-filter style: shape (1, 33) used to crash the binary
        branch. Now we squeeze, softmax, and top-k normally."""
        logits = torch.zeros(1, 33)
        logits[0, 7] = 5.0   # peak at class 7
        out = self._call(logits, top_k=3)
        assert len(out) == 3
        assert out[0].label == 7

    def test_multiclass_with_double_leading_singleton(self):
        """(1, 1, num_classes) — shows up when the wrapper passes a
        token-classification single-token output through. Same fix:
        squeeze leading singletons, then softmax."""
        logits = torch.zeros(1, 1, 5)
        logits[0, 0, 2] = 9.0
        out = self._call(logits, top_k=2)
        assert out[0].label == 2

    def test_token_classification_averages_to_topk(self):
        """(seq_len, num_classes) — token-level model output. The new
        path averages across the sequence axis to produce a document-
        level top-K. Not perfect for NER (use the dedicated
        POSTPROC_TOKEN_LOGITS path for per-token output) but it pins the
        previously-crashing case down to a deterministic top-1."""
        logits = torch.zeros(8, 4)
        # Class 1 wins the unweighted vote: 8 votes vs. 1 outlier.
        logits[:, 1] = 5.0   # strong support for class 1 across all tokens
        logits[3, 2] = 9.0   # one outlier token favoring class 2
        out = self._call(logits, top_k=2)
        # 8 * 5 / 8 = 5.0 for class 1, 9.0 / 8 ≈ 1.125 for class 2.
        # Class 1 wins after averaging.
        assert out[0].label == 1

    def test_true_scalar_triggers_binary_branch(self):
        """A model with a single output logit (regression / binary) goes
        through the sigmoid branch. The fix uses numel()==1 — works for
        shape (), (1,), (1,1), …"""
        logits = torch.tensor(2.0)   # 0-d scalar
        out = self._call(logits)
        assert len(out) == 1
        assert out[0].probability is not None
        assert 0.0 <= out[0].probability <= 1.0
        # And the (1,1) shape:
        out2 = self._call(torch.tensor([[2.0]]))
        assert len(out2) == 1

    def test_plain_1d_classification_still_works(self):
        """Make sure we didn't regress the simplest (and most common)
        shape — a flat (num_classes,) vector."""
        logits = torch.zeros(10)
        logits[3] = 7.0
        out = self._call(logits, top_k=2)
        assert out[0].label == 3


# ---------------------------------------------------------------------------
# QA postproc — the "QuestionAnsweringModelOutput has no attribute 'dim'"
# regression
# ---------------------------------------------------------------------------

class TestQAPostproc:
    """Roberta-base-squad2 + similar QA models return a structured output
    with start_logits / end_logits, NOT plain logits. The wrapper now
    passes that structured object through; the server's _qa_response
    decodes the answer span via the tokenizer."""

    def _fake_qa_outputs(self, start_idx: int, end_idx: int, num_tokens: int = 16):
        s = torch.full((1, num_tokens), -10.0)
        e = torch.full((1, num_tokens), -10.0)
        s[0, start_idx] = 10.0
        e[0, end_idx] = 10.0
        return SimpleNamespace(start_logits=s, end_logits=e)

    def _fake_tokenizer(self, decoded: str = "the answer"):
        tok = MagicMock()
        tok.decode.return_value = decoded
        return tok

    def test_qa_response_decodes_span(self):
        """End-to-end: structured QA output + an input_ids tensor → a
        decoded answer string in Prediction.class_name."""
        from neural_platform.deploy.server import _qa_response
        outputs = self._fake_qa_outputs(start_idx=3, end_idx=5)
        tensor_input = {"input_ids": torch.arange(16).unsqueeze(0)}
        proc = self._fake_tokenizer(decoded="the answer span")
        # _qa_response uses processor.tokenizer if present, else processor itself.
        proc_with_tokenizer = SimpleNamespace(tokenizer=proc)
        resp = _qa_response(outputs, tensor_input, proc_with_tokenizer,
                             "hf_pipeline", t0=0.0)
        assert len(resp.predictions) == 1
        pred = resp.predictions[0][0]
        assert pred.class_name == "the answer span"
        # decode was called with the slice [3:6] (inclusive end).
        called_with = proc.decode.call_args.args[0]
        assert list(called_with.tolist()) == [3, 4, 5]

    def test_qa_response_handles_reversed_spans(self):
        """Models occasionally argmax end < start. We swap rather than
        return an empty span — keeps the response actionable."""
        from neural_platform.deploy.server import _qa_response
        outputs = self._fake_qa_outputs(start_idx=8, end_idx=3)
        tensor_input = {"input_ids": torch.arange(16).unsqueeze(0)}
        proc = self._fake_tokenizer(decoded="recovered span")
        resp = _qa_response(outputs, tensor_input,
                             SimpleNamespace(tokenizer=proc),
                             "hf_pipeline", t0=0.0)
        assert resp.predictions[0][0].class_name == "recovered span"

    def test_qa_postproc_dispatched_via_spec(self):
        """The spec for question-answering routes through QA postproc.
        Pin the spec values so a refactor doesn't accidentally take the
        plain logits path again."""
        from neural_platform.core.pipeline_specs import (
            resolve, POSTPROC_QA_SPANS, INPUT_TEXT_PAIR,
        )
        spec = resolve("question-answering")
        assert spec.output_kind == POSTPROC_QA_SPANS
        assert spec.input_kind == INPUT_TEXT_PAIR


# ---------------------------------------------------------------------------
# Wrapper passes structured outputs through unchanged
# ---------------------------------------------------------------------------

class TestWrapperPassthrough:
    """The wrapper's forward needs to recognize structured HF outputs
    (QA, object detection, depth, segmentation) and return them whole
    so the server's postproc path can extract the right fields."""

    def test_qa_output_passes_through(self):
        """forward sees QuestionAnsweringModelOutput → returns it as-is."""
        from neural_platform.models.hf_pipeline import HFPipelineModel
        # Build a stub with the encoder we want, bypassing __init__.
        wrapper = HFPipelineModel.__new__(HFPipelineModel)
        wrapper._task = "question-answering"
        wrapper._fwd_param_names = {"input_ids", "attention_mask"}
        # Encoder returns a QA-shaped output object.
        qa_out = SimpleNamespace(
            start_logits=torch.zeros(1, 4),
            end_logits=torch.zeros(1, 4),
        )
        encoder = MagicMock(return_value=qa_out)
        wrapper.encoder = encoder
        out = wrapper.forward(input_ids=torch.zeros(1, 4, dtype=torch.long))
        assert out is qa_out
        assert hasattr(out, "start_logits")

    def test_object_detection_passes_through(self):
        from neural_platform.models.hf_pipeline import HFPipelineModel
        wrapper = HFPipelineModel.__new__(HFPipelineModel)
        wrapper._task = "object-detection"
        wrapper._fwd_param_names = {"pixel_values"}
        det_out = SimpleNamespace(
            logits=torch.zeros(1, 100, 91),
            pred_boxes=torch.zeros(1, 100, 4),
        )
        wrapper.encoder = MagicMock(return_value=det_out)
        out = wrapper.forward(pixel_values=torch.zeros(1, 3, 224, 224))
        assert out is det_out

    def test_logits_only_output_collapses_to_tensor(self):
        """For plain classifiers we still return the bare tensor — the
        common-case path stays unchanged."""
        from neural_platform.models.hf_pipeline import HFPipelineModel
        wrapper = HFPipelineModel.__new__(HFPipelineModel)
        wrapper._task = "text-classification"
        wrapper._fwd_param_names = {"input_ids"}
        out_obj = SimpleNamespace(logits=torch.zeros(1, 5))
        wrapper.encoder = MagicMock(return_value=out_obj)
        out = wrapper.forward(input_ids=torch.zeros(1, 4, dtype=torch.long))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 5)
