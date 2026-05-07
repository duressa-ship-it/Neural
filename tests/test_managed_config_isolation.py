"""
Tests for the fixes that keep the Predict tab's HF-launcher configs from
breaking the Train tab.

The "Launch from HF" button synthesizes a config under
``runs/_hf_servers/<id>/config.yaml`` so the inference subprocess has
something to ``--config`` against. Those configs use ``data.source:
synthetic`` (it's never read — the inference server owns the model) and
``num_epochs: 1`` for schema reasons. They were leaking into the Train
tab's config picker; trying to train one crashed deep inside the
embedding layer with an MPSFloatType vs Long mismatch.

Two layers of defense, two test groups:

  * The dashboard's ``/api/configs`` listing filters out the
    ``_hf_servers/`` subtree, so the configs don't show up in the Train
    tab to begin with.
  * The pre-flight validator catches ``hf_pipeline`` + ``synthetic``
    (or any other random-feature source) with a hard error before the
    trainer subprocess spawns. Belt-and-suspenders for the case where
    a user types a path manually.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# /api/configs filter — _hf_servers/ subtree hidden by default
# ---------------------------------------------------------------------------

class TestScanConfigsFilter:

    def _seed(self, tmp_path: Path) -> tuple[Path, Path]:
        """Lay out one normal training config and one synthesized HF
        managed-server config, both under the same output_dir."""
        normal = tmp_path / "real_run" / "config.yaml"
        normal.parent.mkdir(parents=True)
        normal.write_text(
            "name: real_run\n"
            "model:\n  type: mlp\n  framework: pytorch\n"
            "  mlp: {input_size: 4, output_size: 2, hidden_layers: []}\n"
        )
        managed = tmp_path / "_hf_servers" / "hf_distilbert_xyz" / "config.yaml"
        managed.parent.mkdir(parents=True)
        managed.write_text(
            "name: hf_distilbert_xyz\n"
            "model:\n  type: hf_pipeline\n  framework: pytorch\n"
            "  hf_pipeline: {pretrained: distilbert-base-uncased}\n"
        )
        return normal, managed

    def test_default_excludes_hf_servers_subtree(self, tmp_path):
        from neural_platform.web.app import _scan_configs
        normal, managed = self._seed(tmp_path)
        listed = _scan_configs(str(tmp_path))
        paths = {entry["path"] for entry in listed}
        assert str(normal) in paths
        assert str(managed) not in paths, \
            "Managed _hf_servers/ config leaked into the Train tab listing."

    def test_include_managed_returns_everything(self, tmp_path):
        """The escape hatch — debug endpoints / future Predict tab UI may
        want to enumerate the managed configs explicitly."""
        from neural_platform.web.app import _scan_configs
        normal, managed = self._seed(tmp_path)
        listed = _scan_configs(str(tmp_path), include_managed=True)
        paths = {entry["path"] for entry in listed}
        assert str(normal) in paths
        assert str(managed) in paths

    def test_filter_doesnt_match_substring_in_other_dirs(self, tmp_path):
        """A run literally named '_hf_servers_demo' (different leaf) must
        NOT be filtered out — the rule is exact subtree, not substring."""
        from neural_platform.web.app import _scan_configs
        sneaky = tmp_path / "_hf_servers_demo" / "config.yaml"
        sneaky.parent.mkdir(parents=True)
        sneaky.write_text("name: sneaky\nmodel:\n  type: mlp\n  framework: pytorch\n"
                          "  mlp: {input_size: 4, output_size: 2, hidden_layers: []}\n")
        listed = _scan_configs(str(tmp_path))
        assert str(sneaky) in {e["path"] for e in listed}


# ---------------------------------------------------------------------------
# Validator rule: hf_pipeline + synthetic data is a hard error
# ---------------------------------------------------------------------------

class TestSyntheticHFPipelineGuard:

    def _make_cfg(self, *, source: str, pipeline_task: str = "text-classification"):
        """Build a minimal ExperimentConfig that exercises the new rule."""
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            HFPipelineConfig, TrainingConfig, DataConfig, DataSource,
            DeployConfig, Task,
        )
        return ExperimentConfig(
            name="hf_smoke",
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(pretrained="bert-base-uncased"),
            ),
            training=TrainingConfig(
                task=Task.CLASSIFICATION,
                pipeline_task=pipeline_task,
                num_epochs=1,
            ),
            data=DataConfig(source=DataSource(source)),
            deploy=DeployConfig(),
        )

    def test_synthetic_text_pipeline_is_error(self):
        """The exact failure mode the user hit: synthesized HF server
        config got picked up by the trainer and crashed in the embedding
        layer. The validator now catches it before training starts."""
        from neural_platform.core.validator import validate_config
        cfg = self._make_cfg(source="synthetic", pipeline_task="text-classification")
        report = validate_config(cfg, check_deps=False, check_remote=False,
                                  check_resources=False)
        assert not report.ok, "Validator should reject synthetic + hf_pipeline (text)"
        text_errors = [i for i in report.errors if i.field == "data.source"]
        assert text_errors, "No data.source error raised"
        # Hint should mention the workaround paths so users aren't stuck.
        joined = " ".join((i.hint or "") for i in text_errors)
        assert "huggingface" in joined.lower()
        assert "_hf_servers" in joined or "Predict" in joined

    @pytest.mark.parametrize("task", [
        "text-classification",
        "fill-mask",
        "summarization",
        "translation",
        "question-answering",
        "text-generation",
    ])
    def test_all_text_pipelines_caught(self, task):
        from neural_platform.core.validator import validate_config
        cfg = self._make_cfg(source="synthetic", pipeline_task=task)
        report = validate_config(cfg, check_deps=False, check_remote=False,
                                  check_resources=False)
        assert not report.ok, f"task={task} should error"

    def test_image_pipeline_only_warns(self):
        """Image / audio HF pipelines could conceivably accept a numeric
        tensor of the right shape, so we soften to a warning rather than
        a hard error."""
        from neural_platform.core.validator import validate_config
        cfg = self._make_cfg(source="synthetic", pipeline_task="image-classification")
        report = validate_config(cfg, check_deps=False, check_remote=False,
                                  check_resources=False)
        # Should still be `ok` (no errors), but a warning fired.
        assert report.ok
        warnings = [i for i in report.warnings if i.field == "data.source"]
        assert warnings, "Expected a synthetic+hf_pipeline warning for image task"

    def test_hf_pipeline_with_huggingface_data_source_passes(self):
        """The legitimate config — hf_pipeline + huggingface dataset —
        must still validate cleanly. Sanity check that we didn't tighten
        the rule too far."""
        from neural_platform.core.validator import validate_config
        cfg = self._make_cfg(source="huggingface", pipeline_task="text-classification")
        # Will likely produce a "dataset_name required" error elsewhere,
        # but NOT the synthetic-mismatch error we just added.
        report = validate_config(cfg, check_deps=False, check_remote=False,
                                  check_resources=False)
        synthetic_errors = [
            i for i in report.errors
            if i.field == "data.source" and "synthetic" in (i.message or "")
        ]
        assert not synthetic_errors

    def test_csv_and_numpy_also_blocked_for_text(self):
        """CSV/numpy don't produce token IDs either — same crash mode."""
        from neural_platform.core.validator import validate_config
        for source in ("csv", "numpy"):
            cfg = self._make_cfg(source=source, pipeline_task="text-classification")
            report = validate_config(cfg, check_deps=False, check_remote=False,
                                      check_resources=False)
            assert not report.ok, f"source={source} should be rejected for text pipelines"
