"""
Tests for the dashboard's "Launch from HuggingFace" feature.

The launcher synthesizes a minimal hf_pipeline ExperimentConfig, writes
it to a private staging dir, and spawns ``neural serve --no-checkpoint``
on a free port. These tests exercise:

  * `_synthesize_hf_config` — does the synthesized config validate? does
    it pin pretrained / pipeline_task correctly? does it land under the
    manager's `_hf_servers/` subtree so the experiments list stays clean?
  * `start_from_hf` — does it pre-validate the HF id, pass `--no-checkpoint`
    to the subprocess, and wire the bearer token through the env (and
    NOT through ServerInfo)?
  * `create_inference_app(checkpoint_path=None)` — does it accept None for
    hf_pipeline models and refuse it for everything else? This is the
    server-side change that makes the launch path possible.

All HTTP / HF / subprocess interactions are mocked. Strictly offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_platform.web.inference_manager import (
    InferenceServerManager, _synthesize_hf_config,
)


# ---------------------------------------------------------------------------
# Config synthesis
# ---------------------------------------------------------------------------

class TestSynthesizeHFConfig:

    def test_produces_valid_config(self, tmp_path):
        cfg, run_dir = _synthesize_hf_config(
            output_root=tmp_path,
            hf_model_id="distilbert-base-uncased-finetuned-sst-2-english",
            pipeline_task="text-classification",
        )
        # Pydantic validation already happened inside the constructor —
        # any error would have raised. Spot-check the wired values.
        assert cfg.model.type.value == "hf_pipeline"
        assert cfg.model.hf_pipeline.pretrained == \
            "distilbert-base-uncased-finetuned-sst-2-english"
        assert cfg.training.pipeline_task == "text-classification"

    def test_writes_config_file_to_run_dir(self, tmp_path):
        cfg, run_dir = _synthesize_hf_config(
            output_root=tmp_path,
            hf_model_id="openai/whisper-tiny",
            pipeline_task="automatic-speech-recognition",
        )
        cfg_path = run_dir / "config.yaml"
        assert cfg_path.exists()
        # YAML is parseable + keeps the model id.
        import yaml
        data = yaml.safe_load(cfg_path.read_text())
        assert data["model"]["type"] == "hf_pipeline"
        assert data["model"]["hf_pipeline"]["pretrained"] == "openai/whisper-tiny"

    def test_run_dir_is_under_hf_servers_subtree(self, tmp_path):
        """The synthesized configs sit under `_hf_servers/` so they don't
        get mistaken for real training runs in the experiments list."""
        cfg, run_dir = _synthesize_hf_config(
            output_root=tmp_path,
            hf_model_id="microsoft/resnet-50",
            pipeline_task="image-classification",
        )
        assert "_hf_servers" in run_dir.parts
        assert run_dir.parent.name == "_hf_servers"
        assert run_dir.parent.parent == tmp_path.resolve()

    def test_run_dir_unique_per_call(self, tmp_path):
        """Two launches of the same model don't clash — the random suffix
        keeps the staging dirs distinct."""
        _, dir_a = _synthesize_hf_config(
            output_root=tmp_path, hf_model_id="bert-base-uncased",
            pipeline_task="fill-mask",
        )
        _, dir_b = _synthesize_hf_config(
            output_root=tmp_path, hf_model_id="bert-base-uncased",
            pipeline_task="fill-mask",
        )
        assert dir_a != dir_b

    def test_unknown_pipeline_task_falls_back_to_classification(self, tmp_path):
        """The validator needs the coarse Task field set to *something*.
        Any unmapped HF task should still produce a valid config."""
        cfg, _ = _synthesize_hf_config(
            output_root=tmp_path,
            hf_model_id="some-org/some-model",
            pipeline_task="brand-new-pipeline-tag-not-in-our-map",
        )
        # Coarse task always populates — fall back path.
        assert cfg.training.task is not None
        assert cfg.training.pipeline_task == "brand-new-pipeline-tag-not-in-our-map"

    def test_propagates_revision_and_trust_remote_code(self, tmp_path):
        cfg, _ = _synthesize_hf_config(
            output_root=tmp_path,
            hf_model_id="microsoft/Phi-3-mini-4k-instruct",
            pipeline_task="text-generation",
            revision="abc123",
            trust_remote_code=True,
        )
        assert cfg.model.hf_pipeline.revision == "abc123"
        assert cfg.model.hf_pipeline.trust_remote_code is True


# ---------------------------------------------------------------------------
# start_from_hf — subprocess command + token wiring
# ---------------------------------------------------------------------------

class TestStartFromHF:

    def test_rejects_malformed_id(self, tmp_path):
        """Path-traversal / scheme inputs MUST not flow through to the Hub.
        Single-segment names like 'bert-base-uncased' are legitimate so we
        deliberately use a clearly-bad form here."""
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        for bad in ("../../etc/passwd", "file:///etc/passwd",
                    "owner/repo?token=abc", ""):
            with pytest.raises(ValueError):
                mgr.start_from_hf(bad, pipeline_task="text-classification")

    def test_rejects_empty_pipeline_task(self, tmp_path):
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="pipeline_task"):
            mgr.start_from_hf("openai/whisper-tiny", pipeline_task="")

    def test_strips_huggingface_url_prefix(self, tmp_path):
        """The id validator accepts full Hub URLs and strips the host —
        users can paste either form."""
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = MagicMock(); proc.poll.return_value = None; proc.pid = 1
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc):
            info = mgr.start_from_hf(
                "https://huggingface.co/openai/whisper-tiny",
                pipeline_task="automatic-speech-recognition",
            )
        assert info.model_id == "openai/whisper-tiny"

    def test_command_includes_no_checkpoint_flag(self, tmp_path):
        """The subprocess MUST receive --no-checkpoint, otherwise the
        inference app aborts at startup looking for a NeuralForge .pt."""
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = MagicMock(); proc.poll.return_value = None; proc.pid = 1
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc) as popen:
            mgr.start_from_hf("openai/whisper-tiny",
                              pipeline_task="automatic-speech-recognition")
        cmd = popen.call_args.args[0]
        assert "serve" in cmd
        assert "--no-checkpoint" in cmd
        # And specifically NOT --checkpoint — that would 422 the CLI's
        # mutual-exclusion check.
        assert "--checkpoint" not in cmd

    def test_token_passed_via_env_not_in_server_info(self, tmp_path):
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = MagicMock(); proc.poll.return_value = None; proc.pid = 1
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc) as popen:
            info = mgr.start_from_hf("openai/whisper-tiny",
                                      pipeline_task="automatic-speech-recognition")
        env = popen.call_args.kwargs["env"]
        token = env["NEURAL_INFERENCE_TOKEN"]
        # ServerInfo must not leak the token.
        assert token not in str(info.to_dict())

    def test_server_info_marks_source_huggingface(self, tmp_path):
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = MagicMock(); proc.poll.return_value = None; proc.pid = 1
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc):
            info = mgr.start_from_hf("openai/whisper-tiny",
                                      pipeline_task="automatic-speech-recognition")
        assert info.source == "huggingface"
        assert info.model_id == "openai/whisper-tiny"
        assert info.model_type == "hf_pipeline"
        assert info.checkpoint_path is None

    def test_two_hf_servers_get_distinct_run_dirs(self, tmp_path):
        """Each launch lands in its own staging dir under `_hf_servers/`,
        even when the same model id is launched twice — the random suffix
        keeps the configs (and thus the inference logs) from colliding."""
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = MagicMock(); proc.poll.return_value = None; proc.pid = 1
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc):
            a = mgr.start_from_hf("openai/whisper-tiny",
                                   pipeline_task="automatic-speech-recognition")
            b = mgr.start_from_hf("openai/whisper-tiny",
                                   pipeline_task="automatic-speech-recognition")
        assert a.config_path != b.config_path


# ---------------------------------------------------------------------------
# create_inference_app(checkpoint_path=None)
# ---------------------------------------------------------------------------

class TestServerNoCheckpointMode:
    """The launch path depends on the inference app accepting a None
    checkpoint, but ONLY for hf_pipeline. Anything else would mean
    serving random weights — refuse it."""

    def test_app_constructible_without_checkpoint_for_hf(self, tmp_path,
                                                          monkeypatch):
        # Don't auto-generate a token at module import time — keeps test stderr quiet.
        monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        from neural_platform.deploy.server import create_inference_app
        cfg = ExperimentConfig(
            name="hf",
            output_dir=str(tmp_path),
            model=ModelConfig(
                type=ModelType.HF_PIPELINE,
                framework=Framework.PYTORCH,
                hf_pipeline=HFPipelineConfig(
                    pretrained="distilbert-base-uncased-finetuned-sst-2-english",
                ),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION,
                                    pipeline_task="text-classification"),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        # Just constructing the app must succeed — the actual model load
        # happens on the startup hook, which we don't trigger here. (We
        # don't want to pull a real HF model from the Hub during tests.)
        app = create_inference_app(cfg, checkpoint_path=None)
        assert app is not None

    def test_startup_hook_refuses_no_checkpoint_for_non_hf(self, tmp_path,
                                                            monkeypatch):
        """Catching this early at the startup hook (rather than 50 epochs
        in) is the whole point of the guard."""
        monkeypatch.setenv("NEURAL_INFERENCE_AUTH", "off")
        from neural_platform.core.config import (
            ExperimentConfig, ModelConfig, ModelType, Framework,
            MLPConfig, TrainingConfig, DataConfig, DeployConfig, Task,
        )
        from neural_platform.deploy.server import create_inference_app
        cfg = ExperimentConfig(
            name="mlp",
            output_dir=str(tmp_path),
            model=ModelConfig(
                type=ModelType.MLP,
                framework=Framework.PYTORCH,
                mlp=MLPConfig(input_size=4, output_size=2, hidden_layers=[]),
            ),
            training=TrainingConfig(task=Task.CLASSIFICATION),
            data=DataConfig(),
            deploy=DeployConfig(),
        )
        app = create_inference_app(cfg, checkpoint_path=None)
        # Drive the startup hook ourselves so we can assert the error
        # without standing up a real ASGI server.
        from fastapi.testclient import TestClient
        with pytest.raises(RuntimeError, match="hf_pipeline"):
            with TestClient(app):
                pass
