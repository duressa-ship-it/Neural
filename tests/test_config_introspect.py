"""
Tests for the schema introspector that drives the Builder tab.

The contract pinned here:

  * Every model family (mlp / cnn / rnn / transformer / audio_cnn /
    tcn / tabular / video_cnn / hf_pipeline) gets its own group.
  * Each arch group is gated by ``visible_when={"model.type": <family>}``
    so the Builder shows exactly one sub-form per model_type.
  * Field ``kind`` is correct for each Pydantic type (int / number /
    string / bool / enum / list[primitive] / list[object] / object).
  * Nested ``$ref`` resolves — ``training.optimizer.lr`` shows up as
    its own field, not buried in the optimizer object.
  * Numeric constraints (``gt`` / ``ge`` / ``le``) from
    ``Field(gt=0)`` round-trip into ``min`` / ``exclusive_min`` so the
    frontend can put them on the HTML input.
  * Adding a Pydantic field surfaces in the flattened output
    automatically — this is the whole point of the refactor.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest


def _flatten():
    """Convenience: run flatten_for_ui against the real
    ExperimentConfig.model_json_schema()."""
    from neural_platform.core.config import ExperimentConfig
    from neural_platform.core.config_introspect import flatten_for_ui
    return flatten_for_ui(ExperimentConfig.model_json_schema())


def _group(flat: Dict[str, Any], gid: str) -> Dict[str, Any]:
    """Pull one group by id, asserting it exists."""
    for g in flat["groups"]:
        if g["id"] == gid:
            return g
    raise AssertionError(f"group {gid!r} not found; have {[g['id'] for g in flat['groups']]}")


def _field(group: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Pull one field within a group by full dotted path."""
    for f in group["fields"]:
        if f["path"] == path:
            return f
    raise AssertionError(f"field {path!r} not found in {group['id']!r}")


# ---------------------------------------------------------------------------
# Coverage — every UI-advertised model family has a group
# ---------------------------------------------------------------------------

class TestModelFamilyCoverage:

    @pytest.mark.parametrize("family", [
        "mlp", "cnn", "rnn", "transformer", "audio_cnn",
        "tcn", "tabular", "video_cnn", "hf_pipeline",
    ])
    def test_each_family_has_a_group(self, family):
        flat = _flatten()
        g = _group(flat, f"model.{family}")
        # The discriminator on ModelConfig.type drives the visible_when
        # condition — value matches the field name.
        assert g.get("visible_when") == {"model.type": family}

    def test_no_arch_group_without_visible_when(self):
        """Every model.<family> group must be gated by model.type or
        the Builder would render multiple arch forms at once."""
        flat = _flatten()
        for g in flat["groups"]:
            if g["id"].startswith("model.") and g["id"] != "model":
                assert "visible_when" in g, f"{g['id']} missing visible_when"

    def test_top_level_groups_present(self):
        """The four top-level sections (model / training / data /
        deploy) plus the experiment metadata group all exist."""
        flat = _flatten()
        ids = {g["id"] for g in flat["groups"]}
        for required in ("experiment", "model", "training",
                          "training.optimizer", "training.scheduler",
                          "data", "deploy"):
            assert required in ids, f"missing top-level group {required}"


# ---------------------------------------------------------------------------
# Field kind classification
# ---------------------------------------------------------------------------

class TestFieldKinds:

    def test_int_field_classified(self):
        flat = _flatten()
        f = _field(_group(flat, "model.mlp"), "model.mlp.input_size")
        assert f["kind"] == "int"
        assert f["required"] is True

    def test_optional_int_gets_question_suffix(self):
        flat = _flatten()
        # training.val_batch_size is Optional[int].
        f = _field(_group(flat, "training"), "training.val_batch_size")
        assert f["kind"] == "int?"

    def test_number_field_with_exclusive_min(self):
        """Field(gt=0) on training.optimizer.lr becomes
        exclusiveMinimum=0 in JSON-Schema. The introspector surfaces
        that as exclusive_min so the frontend renders min=0 + step=any."""
        flat = _flatten()
        lr = _field(_group(flat, "training.optimizer"), "training.optimizer.lr")
        assert lr["kind"] == "number"
        # Pydantic emits exclusiveMinimum for gt; we round-trip it.
        assert lr.get("exclusive_min") == 0 or lr.get("min") == 0
        assert lr["default"] == pytest.approx(0.001)

    def test_bool_field(self):
        flat = _flatten()
        f = _field(_group(flat, "training"), "training.mixed_precision")
        assert f["kind"] == "bool"
        assert f["default"] is False

    def test_enum_field_carries_choices(self):
        flat = _flatten()
        f = _field(_group(flat, "training.optimizer"), "training.optimizer.type")
        assert f["kind"] == "enum"
        # Pydantic enum values flow into choices.
        assert "adamw" in f["choices"]
        assert "sgd" in f["choices"]

    def test_list_of_object_carries_item_def(self):
        """MLPConfig.hidden_layers is List[LayerConfig]; the frontend
        needs item_def='LayerConfig' to look up the sub-form shape."""
        flat = _flatten()
        f = _field(_group(flat, "model.mlp"), "model.mlp.hidden_layers")
        assert f["kind"] == "list[object]"
        assert f["item_def"] == "LayerConfig"
        # The def itself is included in the response so the frontend
        # doesn't need a second round-trip.
        assert "LayerConfig" in flat["defs"]

    def test_list_of_primitive_carries_item_kind(self):
        """OptimizerConfig.betas is List[float]; item_kind tells the
        renderer to use number inputs rather than text."""
        flat = _flatten()
        f = _field(_group(flat, "training.optimizer"),
                    "training.optimizer.betas")
        assert f["kind"] == "list[primitive]"
        assert f["item_kind"] == "number"

    def test_string_field_with_default(self):
        flat = _flatten()
        f = _field(_group(flat, "experiment"), "name")
        assert f["kind"] == "string"
        assert f["default"] == "experiment"


# ---------------------------------------------------------------------------
# Discriminator detection — the magic that makes 9 arch groups work
# ---------------------------------------------------------------------------

class TestDiscriminator:

    def test_model_type_is_the_discriminator(self):
        """The introspector detects ModelConfig.type (an enum whose
        values match the names of the optional arch sub-blocks) as
        the discriminator. Every arch group's visible_when references
        it."""
        flat = _flatten()
        for family in ("mlp", "cnn", "transformer", "hf_pipeline"):
            g = _group(flat, f"model.{family}")
            assert g["visible_when"]["model.type"] == family

    def test_training_has_no_discriminator(self):
        """TrainingConfig has no enum whose values match its
        sub-blocks (optimizer, scheduler), so nothing gets a
        visible_when — both are always shown."""
        flat = _flatten()
        opt = _group(flat, "training.optimizer")
        sched = _group(flat, "training.scheduler")
        assert "visible_when" not in opt
        assert "visible_when" not in sched


# ---------------------------------------------------------------------------
# Help text round-trip — Field(description=…) → help
# ---------------------------------------------------------------------------

class TestDescriptionRoundTrip:

    def test_field_descriptions_become_help_text(self):
        """The introspector preserves Field(description=…) as the
        UI's `help` text so users see what each control does."""
        flat = _flatten()
        lr = _field(_group(flat, "training.optimizer"), "training.optimizer.lr")
        assert "rate" in lr["help"].lower()


# ---------------------------------------------------------------------------
# Adding a Pydantic field surfaces automatically — the whole point
# ---------------------------------------------------------------------------

class TestSchemaFutureProof:
    """The point of this refactor is that adding a new field to a
    Pydantic model surfaces in the Builder without UI work. Build a
    tiny throw-away model + run the introspector against it, confirm
    the new field appears."""

    def test_new_field_on_arbitrary_model_appears_in_groups(self):
        from pydantic import BaseModel, Field
        from neural_platform.core.config_introspect import flatten_for_ui

        class _ChildA(BaseModel):
            x: int = Field(1, description="An x")
            y: float = Field(2.0, description="A y", gt=0)

        class _Wrapper(BaseModel):
            child: _ChildA = Field(default_factory=_ChildA)
            name: str = "default"

        flat = flatten_for_ui(_Wrapper.model_json_schema())
        # `name` is a top-level primitive → goes in the root group.
        root = next(g for g in flat["groups"] if "name" in
                    {f["path"] for f in g["fields"]})
        name_field = next(f for f in root["fields"] if f["path"] == "name")
        assert name_field["kind"] == "string"
        # `child.x` and `child.y` become their own group's fields.
        child_g = next(g for g in flat["groups"]
                        if any(f["path"].endswith(".x") for f in g["fields"]))
        x = next(f for f in child_g["fields"] if f["path"].endswith(".x"))
        y = next(f for f in child_g["fields"] if f["path"].endswith(".y"))
        assert x["kind"] == "int"
        assert y["kind"] == "number"
        assert y.get("exclusive_min") == 0
        # No UI code was touched to make this work — the whole point.


# ---------------------------------------------------------------------------
# Endpoint round-trip
# ---------------------------------------------------------------------------

class TestSchemaEndpoint:

    def test_endpoint_returns_full_descriptor(self, tmp_path):
        """/api/configs/schema returns the {groups, defs} shape the
        frontend reads. One call, full tree — saves the Builder from
        re-fetching when the model_type flips."""
        from fastapi.testclient import TestClient
        from neural_platform.web.app import create_dashboard_app
        app = create_dashboard_app(str(tmp_path))
        with TestClient(app) as client:
            r = client.get("/api/configs/schema")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "groups" in body and "defs" in body
        # The 9 arch groups + 4 top-level groups + optimizer/scheduler.
        ids = {g["id"] for g in body["groups"]}
        for required in ("experiment", "model", "training", "data", "deploy",
                          "model.mlp", "model.cnn", "model.transformer",
                          "model.hf_pipeline"):
            assert required in ids
