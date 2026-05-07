"""
NeuralForge Pre-flight Configuration Validator
Catches the kinds of errors that *should* fail at config time, not 30 seconds
into a training run when a worker subprocess explodes.

The CLI runs this before spawning a training subprocess; the dashboard runs
this before POST /api/train/start; users can run it on demand via
`neural validate -c config.yaml`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from neural_platform.core.config import (
    DataSource, ExperimentConfig, LossFunction, ModelType, Task,
)


@dataclass
class ValidationIssue:
    severity: str           # "error" | "warning"
    field: str              # dotted path, e.g. "training.optimizer.lr"
    message: str
    hint: Optional[str] = None

    def fmt(self) -> str:
        prefix = "✗" if self.severity == "error" else "⚠"
        out = f"{prefix} {self.field}: {self.message}"
        if self.hint:
            out += f"\n   → {self.hint}"
        return out


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, field: str, message: str, hint: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue("error", field, message, hint))

    def add_warning(self, field: str, message: str, hint: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue("warning", field, message, hint))

    def fmt(self) -> str:
        if not self.issues:
            return "✓ Config is valid."
        lines = []
        for i in self.issues:
            lines.append(i.fmt())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {"severity": i.severity, "field": i.field, "message": i.message, "hint": i.hint}
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_config(cfg: ExperimentConfig,
                    check_deps: bool = True,
                    check_remote: bool = True,
                    check_resources: bool = True) -> ValidationReport:
    """
    Run a battery of cross-cutting checks that Pydantic alone can't catch.
    Returns a ValidationReport; callers decide whether to abort.

    `check_deps` (default True) runs the modality-specific package-availability
    audit (`core.deps`). Tests typically pass `check_deps=False` to keep
    configs validating cleanly even in environments missing optional packages.

    `check_remote` (default True) reaches out to the HuggingFace Hub to ask
    the model inspector whether the chosen model id agrees with the chosen
    pipeline_task. This is the check that catches "openai/whisper-tiny +
    imdb (text)" combos in milliseconds. Tests pass False to stay offline.

    `check_resources` (default True) compares estimated model + dataset
    footprint against host CPU/GPU/RAM/disk. Surfaces tight-fit warnings
    and hard-blocks runs that obviously won't fit.
    """
    r = ValidationReport()
    _validate_identity(cfg, r)
    _validate_model(cfg, r)
    _validate_training(cfg, r)
    _validate_data(cfg, r)
    _validate_data_model_compat(cfg, r)
    if check_deps:
        _validate_dependencies(cfg, r)
    if check_remote:
        _validate_model_source(cfg, r)
    if check_resources:
        _validate_resource_fit(cfg, r)
    return r


def _validate_dependencies(cfg: ExperimentConfig, r: ValidationReport) -> None:
    """Use core.deps to flag missing packages with a copy-pastable install line."""
    from neural_platform.core.deps import check_dependencies, install_command
    report = check_dependencies(cfg.model.type.value, cfg.data.source.value)
    if report.missing_required:
        cmd = install_command(report, only_required=True)
        names = ", ".join(d.package for d in report.missing_required)
        r.add_error(
            "dependencies",
            f"Required package(s) not installed: {names}.",
            f"Install with: `{cmd}`",
        )
    for d in report.missing_optional:
        # Only surface the optional miss as a warning when the user explicitly
        # turned the related feature on (e.g. transformer.use_pretrained needs
        # `transformers`). Silence otherwise — too noisy.
        if d.package == "transformers" and (
            (cfg.model.type.value == "transformer" and cfg.model.transformer and cfg.model.transformer.use_pretrained)
            or (cfg.model.type.value == "audio_cnn" and cfg.model.audio_cnn and cfg.model.audio_cnn.pretrained)
        ):
            r.add_error(
                "dependencies",
                f"`{d.package}` is required because `pretrained` is set.",
                f"Install with: pip install {d.package}",
            )
        elif d.package == "torchcodec" and cfg.data.source.value == "huggingface" \
             and cfg.model.type.value == "audio_cnn":
            r.add_warning(
                "dependencies",
                f"`{d.package}` not installed — HF Audio decoding may fail on some datasets.",
                f"Install with: pip install {d.package} (or pip install soundfile as a fallback).",
            )


# ---------------------------------------------------------------------------
# Sub-validators
# ---------------------------------------------------------------------------

def _validate_identity(cfg: ExperimentConfig, r: ValidationReport) -> None:
    if not cfg.name or not cfg.name.strip():
        r.add_error("name", "Experiment name is empty.",
                    "Pick a short slug — it becomes the directory name.")
    elif "/" in cfg.name or " " in cfg.name:
        r.add_warning("name", f"'{cfg.name}' contains a space or slash.",
                      "Stick to [a-z0-9_-] for clean directory names.")


def _validate_model(cfg: ExperimentConfig, r: ValidationReport) -> None:
    arch = cfg.model.get_arch_config()

    if cfg.model.type == ModelType.MLP:
        if arch.input_size <= 0:
            r.add_error("model.mlp.input_size", "Must be positive.")
        if arch.output_size <= 0:
            r.add_error("model.mlp.output_size", "Must be positive.")
        if not arch.hidden_layers:
            r.add_warning("model.mlp.hidden_layers", "No hidden layers — MLP is just a linear model.",
                          "Add at least one hidden layer for any non-trivial task.")

    elif cfg.model.type == ModelType.CNN:
        if arch.input_height <= 0 or arch.input_width <= 0:
            r.add_error("model.cnn", "input_height and input_width must be positive.")
        if arch.input_channels not in (1, 3, 4):
            r.add_warning("model.cnn.input_channels",
                          f"Unusual channel count {arch.input_channels}.",
                          "Typical values: 1 (grayscale), 3 (RGB), 4 (RGBA).")
        if arch.backbone:
            allowed = ("resnet18", "resnet34", "resnet50", "resnet101",
                       "vgg16", "efficientnet_b0", "efficientnet_b1")
            if arch.backbone not in allowed:
                r.add_warning("model.cnn.backbone",
                              f"'{arch.backbone}' is not a built-in backbone.",
                              f"Built-ins: {', '.join(allowed)}")

    elif cfg.model.type == ModelType.RNN:
        if arch.input_size <= 0 or arch.hidden_size <= 0 or arch.output_size <= 0:
            r.add_error("model.rnn", "input/hidden/output sizes must be positive.")
        if arch.num_layers > 1 and arch.dropout == 0:
            r.add_warning("model.rnn.dropout",
                          "dropout=0 with multiple stacked layers — easy to overfit.",
                          "Try 0.1–0.3 for a regularization boost.")

    elif cfg.model.type == ModelType.TRANSFORMER:
        if arch.d_model % arch.num_heads != 0:
            r.add_error("model.transformer.num_heads",
                        f"num_heads ({arch.num_heads}) must divide d_model ({arch.d_model}).",
                        "Pydantic should catch this — re-validate the config.")
        # `transformers` package availability is checked centrally in
        # `_validate_dependencies` with proper required/optional handling.

    elif cfg.model.type == ModelType.AUDIO_CNN:
        if arch.output_size <= 0:
            r.add_error("model.audio_cnn.output_size", "Must be positive.")
        if arch.sample_rate <= 0:
            r.add_error("model.audio_cnn.sample_rate", "Must be positive.")
        if arch.duration_secs <= 0:
            r.add_error("model.audio_cnn.duration_secs", "Must be positive.")
        # Detailed package-availability messaging is handled centrally in
        # `_validate_dependencies`. We only do shape/value checks here.

    elif cfg.model.type == ModelType.TCN:
        if arch.input_size <= 0 or arch.output_size <= 0:
            r.add_error("model.tcn", "input_size and output_size must be positive.")
        if not arch.channels:
            r.add_error("model.tcn.channels", "Need at least one channel block.",
                        "Try [64, 64, 64, 64] for a 4-block TCN.")
        if arch.kernel_size < 2:
            r.add_warning("model.tcn.kernel_size",
                          f"kernel_size={arch.kernel_size} is degenerate.",
                          "Common values: 2, 3, 5.")

    elif cfg.model.type == ModelType.TABULAR:
        if arch.output_size <= 0:
            r.add_error("model.tabular.output_size", "Must be positive.")
        if not arch.numeric_features and not arch.categorical_features:
            r.add_error(
                "model.tabular",
                "TabularNet has no input features.",
                "Set numeric_features and/or categorical_features.",
            )
        for spec in arch.categorical_features:
            if "name" not in spec or "cardinality" not in spec:
                r.add_error("model.tabular.categorical_features",
                            f"Each entry needs `name` and `cardinality`. Got: {spec}",
                            "Format: {name: country, cardinality: 200, embed_dim: 16 (optional)}")
            elif int(spec["cardinality"]) <= 0:
                r.add_error("model.tabular.categorical_features",
                            f"cardinality for '{spec['name']}' must be positive.")

    elif cfg.model.type == ModelType.HF_PIPELINE:
        if not arch.pretrained:
            r.add_error("model.hf_pipeline.pretrained",
                        "An HF model id is required (e.g. 'openai/whisper-tiny').",
                        "Set model.hf_pipeline.pretrained.")
        # Sanity-check the task → make sure it's a known pipeline_task.
        ptask = (cfg.training.pipeline_task or "").strip()
        if ptask:
            try:
                from neural_platform.core.tasks import get_meta, Task
                meta = get_meta(ptask)
                if meta.task == Task.CUSTOM and ptask != "custom":
                    r.add_warning(
                        "training.pipeline_task",
                        f"'{ptask}' isn't in the known HF pipeline-tag list.",
                        "Will fall back to AutoModel — fine for feature extraction.",
                    )
            except Exception:
                pass
        else:
            r.add_warning(
                "training.pipeline_task",
                "model.type is 'hf_pipeline' but no pipeline_task is set.",
                "Set training.pipeline_task to one of HF's pipeline_tag values "
                "(e.g. 'audio-classification', 'automatic-speech-recognition').",
            )

    elif cfg.model.type == ModelType.VIDEO_CNN:
        if arch.output_size <= 0:
            r.add_error("model.video_cnn.output_size", "Must be positive.")
        if arch.num_frames < 2:
            r.add_warning("model.video_cnn.num_frames",
                          f"num_frames={arch.num_frames} — temporal convs need at least 2 frames.",
                          "Try 8, 16, or 32 frames per clip.")
        if not arch.conv_layers:
            r.add_error("model.video_cnn.conv_layers", "Need at least one conv block.")
        r.add_warning("model.type",
                      "video_cnn is experimental — basic 3D CNN baseline only.",
                      "Real video tasks need purpose-built architectures (I3D, SlowFast). "
                      "Track production-quality video support in DESIGN.md roadmap.")


def _validate_training(cfg: ExperimentConfig, r: ValidationReport) -> None:
    t = cfg.training
    if t.num_epochs <= 0:
        r.add_error("training.num_epochs", "Must be at least 1.")
    if t.batch_size <= 0:
        r.add_error("training.batch_size", "Must be at least 1.")
    if t.batch_size > 4096:
        r.add_warning("training.batch_size",
                      f"batch_size={t.batch_size} is unusually large.",
                      "Common range: 16–512 depending on task and memory.")
    if t.optimizer.lr <= 0:
        r.add_error("training.optimizer.lr", "Learning rate must be positive.")
    if t.optimizer.lr > 1.0:
        r.add_warning("training.optimizer.lr",
                      f"lr={t.optimizer.lr} is unusually high.",
                      "Typical Adam ranges: 1e-5 to 1e-2.")
    if t.scheduler.type.value == "warmup_cosine" and t.scheduler.warmup_steps <= 0:
        r.add_warning("training.scheduler.warmup_steps",
                      "warmup_cosine with warmup_steps=0 collapses to plain cosine.",
                      "Set warmup_steps to ~1–10% of total steps.")
    if t.mixed_precision:
        try:
            import torch
            if not torch.cuda.is_available():
                r.add_warning("training.mixed_precision",
                              "AMP requested but no CUDA device available.",
                              "AMP only helps on CUDA — set mixed_precision=false on CPU/MPS.")
        except ImportError:
            pass


def _validate_data(cfg: ExperimentConfig, r: ValidationReport) -> None:
    d = cfg.data

    if d.val_split + d.test_split >= 1.0:
        r.add_error("data.val_split",
                    f"val_split ({d.val_split}) + test_split ({d.test_split}) "
                    f"leaves nothing for training.",
                    "Sum must be < 1.0.")

    if d.source == DataSource.CSV:
        if not d.path:
            r.add_error("data.path",
                        "CSV source requires `data.path`.",
                        "Point it at a .csv file or switch source to 'synthetic' for testing.")
        elif not Path(d.path).exists():
            r.add_error("data.path", f"File not found: {d.path}",
                        "Check the path is relative to the directory you'll run `neural train` from.")
        elif not d.target_column:
            r.add_error("data.target_column",
                        "CSV source requires `data.target_column`.",
                        "Set this to the column you're trying to predict.")

    elif d.source == DataSource.IMAGE_FOLDER:
        if not d.path:
            r.add_error("data.path",
                        "image_folder source requires `data.path`.",
                        "Point it at a directory of class subfolders, or a parent with train/ and val/.")
        elif not Path(d.path).exists():
            r.add_error("data.path", f"Directory not found: {d.path}")
        elif not Path(d.path).is_dir():
            r.add_error("data.path", f"`{d.path}` is not a directory.")

    elif d.source == DataSource.HUGGINGFACE:
        if not d.dataset_name:
            r.add_error("data.dataset_name",
                        "huggingface source requires `data.dataset_name`.",
                        "e.g. 'imdb', 'mnist', 'cifar10', or any HuggingFace dataset id.")
        # Built-in torchvision short-circuits don't need transformers
        builtin = {"mnist", "fashionmnist", "cifar10", "cifar100", "svhn"}
        normalized = (d.dataset_name or "").lower().replace("-", "").replace("_", "")
        if normalized and normalized not in builtin:
            try:
                import datasets  # noqa: F401
            except ImportError:
                r.add_error("data.source",
                            f"HuggingFace dataset '{d.dataset_name}' requires the `datasets` package.",
                            "Install with: pip install datasets")
            # Transformer text training also needs a tokenizer
            if cfg.model.type == ModelType.TRANSFORMER:
                if not d.text_column:
                    r.add_warning("data.text_column",
                                  "No text_column set — defaulting to 'text'.",
                                  "Set this if your dataset uses a different column name.")
                if not d.label_column:
                    r.add_warning("data.label_column",
                                  "No label_column set — defaulting to 'label'.")

    elif d.source == DataSource.NUMPY:
        if not d.path:
            r.add_error("data.path",
                        "numpy source requires `data.path`.",
                        "Point at a .npz with X and y arrays, or use build_dataloaders_from_arrays directly.")

    elif d.source == DataSource.SYNTHETIC:
        if d.synthetic_n_samples < 50:
            r.add_warning("data.synthetic_n_samples",
                          f"Only {d.synthetic_n_samples} synthetic samples — not enough to learn anything.",
                          "Try 1000+ for a meaningful test.")


def _validate_data_model_compat(cfg: ExperimentConfig, r: ValidationReport) -> None:
    """Cross-checks where the data shape must match the model arch."""
    d, m = cfg.data, cfg.model

    # Synthetic data with MLP: feature count must line up
    if d.source == DataSource.SYNTHETIC and m.type == ModelType.MLP:
        if d.synthetic_n_features != m.mlp.input_size:
            r.add_warning(
                "model.mlp.input_size",
                f"input_size={m.mlp.input_size} but synthetic_n_features={d.synthetic_n_features}.",
                "These should match, otherwise the linear layer will fail at the first batch.",
            )
        if d.synthetic_n_classes != m.mlp.output_size and d.synthetic_n_classes > 1:
            r.add_warning(
                "model.mlp.output_size",
                f"output_size={m.mlp.output_size} but synthetic_n_classes={d.synthetic_n_classes}.",
                "For classification, these should match.",
            )

    # Loss-task compat
    cls_tasks = {Task.CLASSIFICATION, Task.IMAGE_CLASSIFICATION, Task.TEXT_CLASSIFICATION}
    if cfg.training.task in cls_tasks and cfg.training.loss == LossFunction.MSE:
        r.add_warning(
            "training.loss",
            "MSE loss with a classification task is unusual.",
            "Try cross_entropy (multi-class) or bce (binary).",
        )
    if cfg.training.task == Task.REGRESSION and cfg.training.loss == LossFunction.CROSS_ENTROPY:
        r.add_error(
            "training.loss",
            "cross_entropy doesn't fit regression.",
            "Use mse, mae, or huber.",
        )

    # Transformer with a from-scratch encoder + non-tokenized data is doomed
    if (
        m.type == ModelType.TRANSFORMER
        and not m.transformer.use_pretrained
        and d.source not in (DataSource.HUGGINGFACE, DataSource.NUMPY)
    ):
        r.add_error(
            "data.source",
            f"From-scratch transformer needs token IDs; '{d.source.value}' source provides "
            "raw features instead.",
            "Either set model.transformer.use_pretrained='bert-base-uncased' (requires "
            "`transformers`) or switch data.source to 'huggingface'.",
        )

    # hf_pipeline + synthetic / random-feature sources don't make sense.
    # Synthesized configs from the Predict tab's "Launch from HF" button
    # always set data.source=synthetic (it's never read — the inference
    # server owns the model). If a user picks one of those configs in the
    # Train tab, they hit a runtime crash deep inside the embedding layer
    # ("got MPSFloatType instead" / "Long, Int" expected). Catch it here
    # so the error surfaces at validate time with an actionable hint.
    if (
        m.type == ModelType.HF_PIPELINE
        and d.source in (DataSource.SYNTHETIC, DataSource.NUMPY, DataSource.CSV)
    ):
        ptask = (cfg.training.pipeline_task or "").lower()
        text_pipelines = (
            "text-classification", "token-classification",
            "zero-shot-classification", "fill-mask", "text-generation",
            "text2text-generation", "summarization", "translation",
            "question-answering", "feature-extraction",
        )
        is_text_task = (
            ptask in text_pipelines
            or "text" in ptask
            or "fill-mask" in ptask
            or "translation" in ptask
            or "summarization" in ptask
        )
        if is_text_task:
            hint = (
                "Switch data.source to 'huggingface' with a real text dataset "
                "(e.g. dataset_name='imdb'), or — if you only meant to **serve** "
                "this model, not train it — launch it from the Predict tab's "
                "'Launch from HF' button instead. Synthesized server-only configs "
                "live under runs/_hf_servers/ and aren't trainable."
            )
            r.add_error(
                "data.source",
                f"model.type='hf_pipeline' with pipeline_task='{ptask or 'text'}' "
                f"expects tokenized text input, but data.source='{d.source.value}' "
                "produces random feature tensors. The forward pass will crash "
                "inside the embedding layer.",
                hint,
            )
        else:
            # Image / audio / multimodal pipelines could conceivably accept a
            # numeric tensor of the right shape, but a synthetic source
            # with default classification settings is still almost always
            # wrong — surface as a warning, not an error.
            r.add_warning(
                "data.source",
                f"model.type='hf_pipeline' with data.source='{d.source.value}' "
                f"is unusual — most HF pipelines expect dataset-shaped inputs.",
                "Switch to data.source='huggingface', or confirm your synthetic "
                "tensors match the model's expected input shape exactly.",
            )

    # CNN with non-image data
    if m.type == ModelType.CNN and d.source in (DataSource.CSV, DataSource.NUMPY, DataSource.SYNTHETIC):
        r.add_warning(
            "data.source",
            f"CNN with '{d.source.value}' source — make sure your features reshape to "
            f"({m.cnn.input_channels}, {m.cnn.input_height}, {m.cnn.input_width}).",
            "CNNs typically use image_folder or huggingface (mnist/cifar).",
        )

    # Modality compatibility for HuggingFace datasets. We try to introspect
    # the dataset's `features` directly (cheap, only fetches metadata) — but
    # if the `datasets` package isn't available or the dataset is gated, we
    # fall back to the name-based heuristic.
    if d.source == DataSource.HUGGINGFACE and d.dataset_name:
        _validate_hf_modality(cfg, r)


def _validate_hf_modality(cfg: ExperimentConfig, r: ValidationReport) -> None:
    """
    Best-effort schema inspection for HF datasets.

    Strategy:
      1. Try datasets.load_dataset_builder(...).info.features — fast, no download.
      2. If unavailable / fails, fall back to a small hardcoded list of well-known
         text/image dataset names. This still catches 'imdb', 'cifar10', etc.
    """
    d, m = cfg.data, cfg.model
    name = d.dataset_name
    schema = None

    # Step 1: ask the HF Hub what columns this dataset has, without downloading
    # the actual data.
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(name)
        info_features = getattr(builder.info, "features", None)
        if info_features:
            schema = _features_summary(info_features)
    except Exception:
        schema = None

    # Step 2: fallback hardcoded heuristic. Populate the same shape that
    # `inspect_features` would produce, so `detect_from_features` works the
    # same way on either path.
    if schema is None:
        normalized = name.lower().replace("-", "").replace("_", "")
        image_datasets = {
            "mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "imagenet",
            "stl10", "flowers", "flowers102", "celeba", "lsun", "oxfordflowers",
        }
        text_datasets = {
            "imdb", "sst2", "ag_news", "agnews", "amazon_polarity", "yelp_polarity",
            "trec", "snli", "mnli", "rotten_tomatoes", "rottentomatoes", "wikitext",
            "squad", "cnn_dailymail", "tweet_eval", "tweeteval", "emotion",
        }
        empty = {
            "columns": [], "image_columns": [], "text_columns": [],
            "audio_columns": [], "video_columns": [], "sequence_columns": [],
            "label_columns": [], "numeric_columns": [], "other_columns": [],
            "class_names": None,
            "has_images": False, "has_text": False,
            "has_audio": False, "has_video": False, "has_sequence": False,
        }
        if normalized in image_datasets:
            schema = {**empty, "image_columns": ["image"], "has_images": True}
        elif normalized in text_datasets or any(t in normalized for t in ("text", "qa", "review", "sentence")):
            schema = {**empty, "text_columns": ["text"], "has_text": True}
        else:
            return  # genuinely unknown — can't help, let runtime decide

    # Step 3: compare model type vs detected modality. Use the central
    # `core.modality.MODALITY_MODELS` mapping so this stays in sync with the
    # loader and the Builder UI's recommendations.
    from neural_platform.core.modality import (
        Modality, MODALITY_MODELS, detect_from_features, recommend_model,
    )
    cols     = schema.get("columns", []) or []
    cols_str = f" Columns: {cols}." if cols else ""

    detected = detect_from_features(schema)
    valid_models = MODALITY_MODELS.get(detected, [])
    if detected != Modality.UNKNOWN and valid_models and m.type.value not in valid_models:
        r.add_error(
            "data.dataset_name",
            f"Model '{m.type.value}' doesn't match dataset modality '{detected.value}'. "
            f"Forward pass will crash on the first batch.{cols_str}",
            f"For a {detected.value} dataset, use one of: {', '.join(valid_models)}. "
            f"Suggested: '{recommend_model(detected)}'.",
        )

    # Step 4: column-level checks. Specifically, if the user set text_column /
    # label_column to something that doesn't exist, flag it now (much earlier
    # than the runtime KeyError).
    if cols:
        if d.text_column and d.text_column not in cols:
            r.add_error(
                "data.text_column",
                f"text_column '{d.text_column}' isn't in dataset '{name}'.",
                f"Available columns: {cols}. Leave text_column empty to let NeuralForge "
                f"auto-detect, or pick one of: {schema.get('text_columns') or '(no string columns)'}.",
            )
        if d.label_column and d.label_column not in cols:
            r.add_error(
                "data.label_column",
                f"label_column '{d.label_column}' isn't in dataset '{name}'.",
                f"Available columns: {cols}. Likely label candidates: "
                f"{schema.get('label_columns') or '(no obvious label column)'}.",
            )


def _features_summary(features) -> dict:
    """Lightweight schema summary from a `datasets.Features` mapping.

    Thin wrapper around `core.hf_introspect.inspect_features` — kept here so
    callers that imported `_features_summary` from this module still work.
    """
    from neural_platform.core.hf_introspect import inspect_features
    return inspect_features(features)


# ---------------------------------------------------------------------------
# Pluggable model-source compatibility check
# ---------------------------------------------------------------------------

def _validate_model_source(cfg: ExperimentConfig, r: ValidationReport) -> None:
    """When the model is an `hf_pipeline`, ask the HF inspector whether
    the chosen `pretrained` id is compatible with the chosen `pipeline_task`
    and the dataset's modality.

    Catches the canonical Whisper-vs-IMDB failure mode: user picks an
    audio model and points it at a text task. The inspector reads the
    model's `pipeline_tag` from the Hub and rejects the mismatch with a
    pointer to the right model search.

    Network problems degrade to silence: we don't want offline users to
    be blocked from training a known-good config.
    """
    if cfg.model.type != ModelType.HF_PIPELINE:
        return
    pretrained = (cfg.model.hf_pipeline.pretrained or "").strip()
    if not pretrained:
        return
    intended_task = (cfg.training.pipeline_task or "").strip() or None

    # Detect dataset modality cheaply — reuse the introspect path.
    dataset_modality = None
    try:
        if cfg.data.source == DataSource.HUGGINGFACE and cfg.data.dataset_name:
            from datasets import load_dataset_builder
            from neural_platform.core.hf_introspect import inspect_features
            from neural_platform.core.modality import detect_from_features
            builder = load_dataset_builder(cfg.data.dataset_name)
            schema = inspect_features(getattr(builder.info, "features", None))
            dataset_modality = detect_from_features(schema).value
    except Exception:
        dataset_modality = None

    try:
        from neural_platform.core.model_source import get_source
        source = get_source("huggingface")
        report = source.inspect_compat(
            pretrained,
            intended_task=intended_task,
            dataset_modality=dataset_modality,
        )
    except Exception as exc:
        r.add_warning(
            "model.hf_pipeline.pretrained",
            f"Could not query HuggingFace Hub for '{pretrained}': {exc}",
            "Network/auth issue — proceeding without compatibility check.",
        )
        return

    for issue in report.issues:
        field_name = "model.hf_pipeline.pretrained"
        if issue.code in ("task_modality_mismatch", "task_mismatch_same_modality"):
            field_name = "training.pipeline_task"
        elif issue.code == "auth_required":
            field_name = "auth.HF_TOKEN"
        if issue.severity == "error":
            r.add_error(field_name, issue.message, issue.hint)
        elif issue.severity == "warning":
            r.add_warning(field_name, issue.message, issue.hint)


# ---------------------------------------------------------------------------
# Resource-fit pre-flight
# ---------------------------------------------------------------------------

def _validate_resource_fit(cfg: ExperimentConfig, r: ValidationReport) -> None:
    """Estimate model + dataset footprint and compare against host resources.

    Best-effort: we only have parameter counts for HF Pipeline and HF
    Transformer models. For from-scratch models the trainer would need to
    materialize the architecture to count params — that's a heavier check
    we run later in `Trainer._sanity_check_first_batch`.
    """
    try:
        from neural_platform.core.resource_fit import (
            snapshot_host, estimate_model_footprint, add_dataset_footprint, check_fit,
        )
    except Exception:
        return  # resource_fit module missing — silent skip

    parameters = None
    size_bytes = None
    arch_hints: dict = {}    # hidden_size, num_layers, num_heads, sequence_length

    def _harvest_arch_hints(info) -> None:
        """Pull architecture sizes out of a HF config.json so the activation
        estimate uses the rich (B×T×D×L) path instead of the lean fallback."""
        cfg_block = info.config or {}
        if not isinstance(cfg_block, dict):
            return
        for src_key, dst_key in [
            ("hidden_size", "hidden_size"),
            ("d_model", "hidden_size"),
            ("num_hidden_layers", "num_layers"),
            ("n_layer", "num_layers"),
            ("num_layers", "num_layers"),
            ("num_attention_heads", "num_heads"),
            ("n_head", "num_heads"),
            ("max_position_embeddings", "sequence_length"),
        ]:
            val = cfg_block.get(src_key)
            if val and dst_key not in arch_hints:
                try:
                    arch_hints[dst_key] = int(val)
                except (TypeError, ValueError):
                    pass

    # HF pipeline → ask the model source for the param count
    if cfg.model.type == ModelType.HF_PIPELINE and cfg.model.hf_pipeline.pretrained:
        try:
            from neural_platform.core.model_source import get_source
            info = get_source("huggingface").get_info(cfg.model.hf_pipeline.pretrained)
            parameters = info.parameters
            size_bytes = info.size_bytes
            _harvest_arch_hints(info)
        except Exception:
            return  # offline / unauthorized — skip resource check silently

    # Transformer with use_pretrained — same path
    if cfg.model.type == ModelType.TRANSFORMER and cfg.model.transformer \
       and cfg.model.transformer.use_pretrained:
        try:
            from neural_platform.core.model_source import get_source
            info = get_source("huggingface").get_info(cfg.model.transformer.use_pretrained)
            parameters = info.parameters
            size_bytes = info.size_bytes
            _harvest_arch_hints(info)
        except Exception:
            return

    if not parameters and not size_bytes:
        return  # nothing to estimate

    # Cap sequence_length at what the user actually configured. HF models
    # often advertise huge context windows (128K) that we won't actually use.
    user_seq_len = None
    if cfg.model.type == ModelType.TRANSFORMER and cfg.model.transformer:
        user_seq_len = getattr(cfg.model.transformer, "max_seq_len", None) \
            or getattr(cfg.model.transformer, "sequence_length", None)
    transforms = cfg.data.transforms
    if isinstance(transforms, dict) and transforms.get("text", {}).get("max_length"):
        user_seq_len = transforms["text"]["max_length"]
    if user_seq_len:
        arch_hints["sequence_length"] = min(int(user_seq_len),
                                              arch_hints.get("sequence_length", 1 << 30))
    else:
        # Default to 128 — the loader's default — so we don't size against 128K.
        arch_hints.setdefault("sequence_length", 128)

    # Dataset size from the HF builder (if available)
    dataset_size = None
    try:
        if cfg.data.source == DataSource.HUGGINGFACE and cfg.data.dataset_name:
            from datasets import load_dataset_builder
            builder = load_dataset_builder(cfg.data.dataset_name)
            dataset_size = getattr(builder.info, "dataset_size", None) \
                or getattr(builder.info, "download_size", None)
    except Exception:
        dataset_size = None

    host = snapshot_host()
    est = estimate_model_footprint(
        parameters=parameters, size_bytes=size_bytes,
        purpose="training",
        optimizer=cfg.training.optimizer.type.value if hasattr(cfg.training.optimizer.type, "value") else "adamw",
        batch_size=cfg.training.batch_size,
        sequence_length=arch_hints.get("sequence_length"),
        hidden_size=arch_hints.get("hidden_size"),
        num_layers=arch_hints.get("num_layers"),
        num_heads=arch_hints.get("num_heads"),
        dtype_bytes=2 if cfg.training.mixed_precision else 4,
    )
    est = add_dataset_footprint(est, dataset_size)
    fit = check_fit(est, host, purpose="training", device=cfg.training.device)
    for issue in fit.issues:
        sev = issue.get("severity") or "warning"
        msg = issue.get("message") or ""
        hint = issue.get("hint")
        field_name = (
            "training.batch_size" if issue.get("code") in ("vram_tight", "vram_too_small")
            else "data.dataset_name" if issue.get("code", "").startswith("disk")
            else "model.hf_pipeline.pretrained"
        )
        if sev == "error":
            r.add_error(field_name, msg, hint)
        else:
            r.add_warning(field_name, msg, hint)
