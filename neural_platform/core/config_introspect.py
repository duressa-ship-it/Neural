"""
NeuralForge — schema introspector for the Builder tab.

Walks a Pydantic JSON Schema (as emitted by ``Model.model_json_schema()``)
and flattens it into a list of UI groups + fields that the frontend can
render without per-model-family hardcoding. Adding a new model family
becomes a Pydantic-only change once the Builder is wired through this.

Schema → UI mapping
-------------------

Pydantic v2 emits something like::

    {
      "$defs": {
        "MLPConfig":      {"type": "object", "properties": {...}, "required": [...]},
        "OptimizerConfig":{...},
        "ModelType":      {"enum": ["mlp","cnn",...], "type": "string"},
        ...
      },
      "properties": {
        "name":    {"type": "string", "default": "experiment", ...},
        "model":   {"$ref": "#/$defs/ModelConfig"},
        "training":{"$ref": "#/$defs/TrainingConfig"},
        ...
      },
    }

We turn that into a stable, frontend-friendly shape::

    {
      "groups": [
        {"id": "experiment", "title": "Experiment",
         "fields": [
           {"path": "name", "kind": "string", "default": "experiment", ...},
           {"path": "description", "kind": "string?", ...},
           ...
         ]},
        {"id": "model", "title": "Model",
         "fields": [{"path": "model.type", "kind": "enum",
                     "choices": ["mlp","cnn",...]}, ...]},
        {"id": "model.mlp", "title": "MLP architecture",
         "visible_when": {"model.type": "mlp"},
         "fields": [...]},
        {"id": "model.cnn", "title": "CNN architecture",
         "visible_when": {"model.type": "cnn"},
         "fields": [...]},
        ...
        {"id": "training", "title": "Training", "fields": [...]},
        {"id": "training.optimizer", "title": "Optimizer", "fields": [...]},
        {"id": "training.scheduler", "title": "Scheduler", "fields": [...]},
        {"id": "data", "title": "Data", "fields": [...]},
        {"id": "deploy", "title": "Deploy", "fields": [...]},
      ],
      "defs": {
        "LayerConfig": {... raw JSON-Schema kept for list-of-object editors ...},
        ...
      },
    }

The frontend then has one renderer that loops over ``groups`` and emits
controls based on ``kind``. Heterogeneous-arch unions (the 9 optional
``model.<family>`` sub-blocks) are surfaced as separate groups gated by
``visible_when`` rules so exactly one shows for the picked
``model.type``.

Field kinds (stable enum the frontend switches on)
--------------------------------------------------

  ``int`` / ``int?``           — bare or Optional integer
  ``number`` / ``number?``     — float
  ``string`` / ``string?``     — text
  ``bool``                     — checkbox
  ``enum`` / ``enum?``         — dropdown with ``choices``
  ``list[primitive]``          — repeatable text/number inputs
  ``list[object]``             — repeatable sub-form (item shape via ``item_def``)
  ``list[any]``                — opaque JSON textarea (heterogeneous lists)
  ``object``                   — nested form (item shape via ``object_def``)
  ``any``                      — raw JSON textarea (fallback)

``?`` suffix flags Optional. The frontend uses it to render a clear/null
button alongside the input.

Visibility rules
----------------

The ``ModelConfig.check_arch_config_present`` validator requires exactly
one of {mlp, cnn, rnn, transformer, audio_cnn, tcn, tabular, video_cnn,
hf_pipeline} to be populated based on ``model.type``. We pre-bake that
discriminated-union behavior into ``visible_when={"model.type": "mlp"}``
on each arch group so the Builder hides the eight other arch forms
automatically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flatten_for_ui(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a Pydantic JSON Schema into the Builder-friendly shape.

    The ``schema`` argument is the dict that
    ``ExperimentConfig.model_json_schema()`` returns. Output is the
    ``{groups, defs}`` shape documented in the module docstring.

    Stable: keys never disappear once added. Frontend can rely on every
    documented ``kind`` value being one of the enumerated strings.
    """
    defs = schema.get("$defs", {}) or {}
    top_props = schema.get("properties", {}) or {}
    top_required = set(schema.get("required", []) or [])

    groups: List[Dict[str, Any]] = []

    # ----- Top-level Experiment metadata (name / description / tags / output_dir)
    # Everything that's a primitive at the top level lives in one group.
    # The nested $ref entries (model / training / data / deploy) become
    # their own groups below.
    exp_fields: List[Dict[str, Any]] = []
    for name, prop in top_props.items():
        if _is_ref(prop):
            continue
        exp_fields.append(_field(name, prop, defs, required=name in top_required))
    if exp_fields:
        groups.append({
            "id":     "experiment",
            "title":  "Experiment",
            "fields": exp_fields,
        })

    # ----- Nested top-level objects (any $ref at the top level)
    # Each one becomes its own group; the nested sub-models inside
    # (training.optimizer, model.mlp, …) become their own groups too,
    # with paths like "training.optimizer.lr". We walk in declaration
    # order so the Builder renders identity → model → training →
    # data → deploy the same way Pydantic emits the fields.
    for top_name, prop in top_props.items():
        def_name = _ref_target(prop, defs)
        if not def_name or def_name not in defs:
            continue
        if not _is_object_def(defs[def_name]):
            continue
        _emit_groups_for_def(
            defs=defs,
            def_name=def_name,
            base_path=top_name,
            base_title=top_name.replace("_", " ").capitalize(),
            groups=groups,
        )

    return {
        "groups": groups,
        "defs":   defs,
    }


# ---------------------------------------------------------------------------
# Group emission — walks one nested BaseModel
# ---------------------------------------------------------------------------

def _emit_groups_for_def(*, defs: Dict[str, Any], def_name: str,
                          base_path: str, base_title: str,
                          groups: List[Dict[str, Any]]) -> None:
    """Walk one def (e.g. ``ModelConfig``) and emit one group for its own
    primitive fields plus separate groups for every nested BaseModel
    referenced by ``$ref`` / ``anyOf``.

    The ``ModelConfig`` walk is the interesting case: every architecture
    sub-block (``mlp``, ``cnn``, …) is an ``Optional[XConfig]`` whose
    ``anyOf`` carries a ``$ref``. We pick the discriminator field
    (``type``) and use it to derive ``visible_when={"model.type": "mlp"}``
    so the Builder shows exactly one arch sub-form at a time.
    """
    block = defs.get(def_name) or {}
    props = block.get("properties", {}) or {}
    required = set(block.get("required", []) or [])

    primitive_fields: List[Dict[str, Any]] = []
    nested: List[Tuple[str, str]] = []   # (field_name, sub_def_name)
    discriminator_field = _discriminator_field(def_name, props, defs)

    for field_name, prop in props.items():
        sub_def = _ref_target(prop, defs)
        if sub_def is not None and _is_object_def(defs.get(sub_def, {})):
            # Nested BaseModel — emits its own group below.
            nested.append((field_name, sub_def))
            continue
        # Primitive (including enum, list, optional-of-primitive).
        primitive_fields.append(_field(
            field_name, prop, defs,
            path_prefix=base_path,
            required=field_name in required,
        ))

    # Emit the primitive group first (e.g. ModelConfig's `type`, `name`,
    # `framework` live here — the nested arch sub-forms come after).
    if primitive_fields:
        groups.append({
            "id":     base_path,
            "title":  base_title,
            "fields": primitive_fields,
        })

    # Emit each nested sub-model as its own group.
    for field_name, sub_def in nested:
        sub_path = f"{base_path}.{field_name}"
        sub_title = f"{base_title} · {field_name.replace('_', ' ').title()}"

        # Heterogeneous-arch case: when the parent has a discriminator
        # (ModelConfig.type), every Optional[XConfig] sub-form is gated
        # by the discriminator's enum value. The mapping
        # 'mlp' field → 'mlp' discriminator value follows naturally from
        # the field name matching the enum value (this is how the
        # existing ModelConfig is structured).
        visible_when: Optional[Dict[str, str]] = None
        if discriminator_field is not None:
            disc_path = f"{base_path}.{discriminator_field}"
            # The field name on the model (`mlp`, `cnn`, …) IS the
            # discriminator value the user picks for `model.type`.
            visible_when = {disc_path: field_name}

        # Inline the sub-def's primitives + recurse for any deeper nests.
        sub_props = defs.get(sub_def, {}).get("properties", {}) or {}
        sub_required = set(defs.get(sub_def, {}).get("required", []) or [])
        inline_fields: List[Dict[str, Any]] = []
        deeper_nested: List[Tuple[str, str]] = []
        for inner_name, inner_prop in sub_props.items():
            inner_def = _ref_target(inner_prop, defs)
            if inner_def is not None and _is_object_def(defs.get(inner_def, {})):
                deeper_nested.append((inner_name, inner_def))
                continue
            inline_fields.append(_field(
                inner_name, inner_prop, defs,
                path_prefix=sub_path,
                required=inner_name in sub_required,
            ))
        group: Dict[str, Any] = {
            "id":     sub_path,
            "title":  sub_title,
            "fields": inline_fields,
        }
        if visible_when:
            group["visible_when"] = visible_when
        groups.append(group)

        # Recurse for sub-sub-models (e.g. training.scheduler within
        # training; this is rare but the validator-rich configs occasionally
        # nest two levels deep).
        for inner_name, inner_def in deeper_nested:
            _emit_groups_for_def(
                defs=defs,
                def_name=inner_def,
                base_path=f"{sub_path}.{inner_name}",
                base_title=f"{sub_title} · {inner_name.replace('_', ' ').title()}",
                groups=groups,
            )


# ---------------------------------------------------------------------------
# Field classification — schema fragment → {kind, default, min, max, ...}
# ---------------------------------------------------------------------------

def _field(name: str, prop: Dict[str, Any], defs: Dict[str, Any],
            *, path_prefix: str = "", required: bool = False) -> Dict[str, Any]:
    """Turn one JSON-Schema property into a UI field record.

    The output keys the frontend reads:

      ``path``    — full dotted path (``model.mlp.input_size``)
      ``kind``    — one of the documented kinds above
      ``label``   — human title (derived from ``title`` / field name)
      ``help``    — description text from ``Field(description=…)``
      ``default`` — when set
      ``min`` / ``max`` — from minimum / maximum / exclusiveMinimum
      ``choices`` — for enums
      ``item_def`` — name in ``defs`` for ``list[object]`` field
      ``object_def`` — name in ``defs`` for nested ``object`` field
      ``required`` — bool
    """
    path = f"{path_prefix}.{name}" if path_prefix else name
    kind, extra = _classify(prop, defs)
    field: Dict[str, Any] = {
        "path":     path,
        "kind":     kind,
        "label":    prop.get("title") or _humanize(name),
        "help":     prop.get("description") or "",
        "required": required,
    }
    if "default" in prop:
        field["default"] = prop["default"]
    # Numeric bounds — both inclusive and exclusive are surfaced. The
    # frontend uses them to set the ``min`` / ``max`` HTML attributes.
    for src, dst in (("minimum", "min"),
                      ("maximum", "max"),
                      ("exclusiveMinimum", "exclusive_min"),
                      ("exclusiveMaximum", "exclusive_max")):
        if src in prop:
            field[dst] = prop[src]
    # Splice in extras from the classifier (choices / item_def / object_def).
    field.update(extra)
    return field


def _classify(prop: Dict[str, Any],
               defs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return ``(kind, extra)`` for one schema property.

    ``extra`` carries kind-specific metadata (choices for enums,
    item_def for list-of-object, etc.) that gets merged into the
    field record.
    """
    # ----- Optional[T] surfaces as anyOf [T, null]
    if "anyOf" in prop:
        non_null = [v for v in prop["anyOf"]
                    if v.get("type") != "null"]
        if len(non_null) == 1:
            inner = non_null[0]
            kind, extra = _classify(inner, defs)
            if not kind.endswith("?"):
                kind = kind + "?"
            return kind, extra
        # Heterogeneous Union — too complex for the form generator; the
        # frontend renders a raw JSON textarea.
        return "any", {}

    # ----- $ref to a def
    target = _ref_target(prop, defs)
    if target is not None:
        target_def = defs.get(target, {})
        # Enum class (typically Pydantic StrEnum / IntEnum).
        if "enum" in target_def:
            return "enum", {"choices": list(target_def["enum"])}
        # Object def — surface as nested object field. (Sub-form rendering
        # happens at the group level, not the field level, so the
        # frontend rarely hits this; kept for completeness.)
        if _is_object_def(target_def):
            return "object", {"object_def": target}

    # ----- Inline enum
    if "enum" in prop:
        return "enum", {"choices": list(prop["enum"])}

    t = prop.get("type")

    # ----- Array
    if t == "array":
        items = prop.get("items") or {}
        item_ref = _ref_target(items, defs)
        if item_ref is not None and _is_object_def(defs.get(item_ref, {})):
            return "list[object]", {"item_def": item_ref}
        # Primitive item type (string / number / integer / boolean).
        item_t = items.get("type")
        if item_t in ("string", "number", "integer", "boolean"):
            return "list[primitive]", {"item_kind": _type_to_kind(item_t)}
        # Untyped or heterogeneous list — opaque JSON.
        return "list[any]", {}

    # ----- Primitive scalar
    return _type_to_kind(t), {}


def _type_to_kind(t: Optional[str]) -> str:
    """Map a JSON-Schema ``type`` keyword to our UI kind."""
    return {
        "integer": "int",
        "number":  "number",
        "string":  "string",
        "boolean": "bool",
        "object":  "object",
    }.get(t, "any")


# ---------------------------------------------------------------------------
# $ref helpers
# ---------------------------------------------------------------------------

def _is_ref(prop: Dict[str, Any]) -> bool:
    return isinstance(prop, dict) and "$ref" in prop

def _ref_name(prop: Dict[str, Any]) -> Optional[str]:
    """Pull the def name out of a ``{"$ref": "#/$defs/Foo"}`` fragment."""
    ref = prop.get("$ref")
    if not isinstance(ref, str):
        return None
    if not ref.startswith("#/$defs/"):
        return None
    return ref[len("#/$defs/"):]

def _ref_target(prop: Dict[str, Any],
                 defs: Dict[str, Any]) -> Optional[str]:
    """Resolve direct ``$ref`` AND ``anyOf: [{$ref: ...}, {type: 'null'}]``.

    The Optional[BaseModel] case is the common one — every
    ``ModelConfig.<family>`` sub-block is Optional in the schema.
    """
    if not isinstance(prop, dict):
        return None
    if "$ref" in prop:
        return _ref_name(prop)
    # Optional sub-model: anyOf [$ref, null]
    if "anyOf" in prop:
        for v in prop["anyOf"]:
            if isinstance(v, dict) and "$ref" in v:
                return _ref_name(v)
    return None

def _is_object_def(def_block: Dict[str, Any]) -> bool:
    """True when the def is a BaseModel-shaped object (not an enum)."""
    if not isinstance(def_block, dict):
        return False
    if def_block.get("type") != "object":
        return False
    if "enum" in def_block:
        return False
    return True


def _discriminator_field(def_name: str,
                          props: Dict[str, Any],
                          defs: Dict[str, Any]) -> Optional[str]:
    """Pick a field whose enum values match sibling Optional sub-block
    names. This is how we detect ``ModelConfig.type`` as the
    discriminator for the 9 ``model.<family>`` arch sub-blocks.

    Heuristic: a discriminator is a field whose value-set (an enum)
    equals (or is a superset of) the names of the parent's Optional
    sub-model fields. For ``ModelConfig``, ``type`` is an ``enum``
    with values ``["mlp","cnn",...]`` matching the
    ``mlp / cnn / rnn / ...`` sub-block field names exactly.
    """
    # Collect the set of sub-model field names on this def.
    sub_names: List[str] = []
    for fname, prop in props.items():
        sub_def = _ref_target(prop, defs)
        if sub_def is not None and _is_object_def(defs.get(sub_def, {})):
            sub_names.append(fname)
    if not sub_names:
        return None
    sub_set = set(sub_names)
    # Look for an enum field whose values overlap with the sub_names.
    for fname, prop in props.items():
        target = _ref_target(prop, defs)
        enum_vals: Optional[List[Any]] = None
        if target is not None:
            tgt = defs.get(target, {})
            if "enum" in tgt:
                enum_vals = list(tgt["enum"])
        elif "enum" in prop:
            enum_vals = list(prop["enum"])
        if enum_vals is None:
            continue
        if sub_set.issubset(set(map(str, enum_vals))):
            return fname
    return None


# ---------------------------------------------------------------------------
# Cosmetics
# ---------------------------------------------------------------------------

def _humanize(name: str) -> str:
    """Convert ``num_attention_heads`` → ``Num attention heads``."""
    return name.replace("_", " ").capitalize()
