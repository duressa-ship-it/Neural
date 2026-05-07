"""
Tests for the trainer's --resume / --finetune flow.

Covers the two resume modes the user asked for:

  * **full** — restores model weights, optimizer state, scheduler state,
    epoch counter and best-so-far metrics. The next training run picks up
    where the last one left off.
  * **weights_only** — restores model weights only. Optimizer / scheduler
    start fresh, the epoch counter resets to 1. This is the fine-tuning
    code path used when adapting a model to a new dataset or LR.

Strictly offline — no real epochs are run. We patch the Trainer's loop
machinery (`_set_seed`, `Progress`) where needed, and exercise the
checkpoint round-trip + the ``_apply_resume`` private method directly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from neural_platform.core.config import (
    DataConfig, DataSource, DeployConfig, ExperimentConfig, Framework,
    LossFunction, MLPConfig, ModelConfig, ModelType, Optimizer, OptimizerConfig,
    Scheduler, SchedulerConfig, Task, TrainingConfig,
)
from neural_platform.core.trainer import Trainer
from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path, *, lr: float = 1e-3,
              num_epochs: int = 5,
              scheduler: Scheduler = Scheduler.STEP) -> ExperimentConfig:
    """A tiny, fully-offline config we can build a real PyTorch model from."""
    return ExperimentConfig(
        name="resume_smoke",
        output_dir=str(tmp_path),
        model=ModelConfig(
            type=ModelType.MLP,
            framework=Framework.PYTORCH,
            mlp=MLPConfig(input_size=4, output_size=3,
                          hidden_layers=[]),
        ),
        training=TrainingConfig(
            task=Task.CLASSIFICATION,
            loss=LossFunction.CROSS_ENTROPY,
            num_epochs=num_epochs,
            batch_size=2,
            optimizer=OptimizerConfig(type=Optimizer.ADAMW, lr=lr),
            scheduler=SchedulerConfig(type=scheduler, step_size=1, gamma=0.5),
            device="cpu",
            num_workers=0,
        ),
        data=DataConfig(source=DataSource.SYNTHETIC,
                        synthetic_n_samples=8, synthetic_n_features=4,
                        synthetic_n_classes=3),
        deploy=DeployConfig(),
    )


def _save_fake_checkpoint(adapter: PyTorchAdapter, model, optimizer,
                          path: Path, *, epoch: int = 4,
                          val_loss: float = 0.42,
                          best_val_loss: float | None = 0.30,
                          best_epoch: int = 3,
                          scheduler=None,
                          scaler=None):
    """Write a checkpoint that mimics what the trainer's checkpoint hook
    persists — including scheduler/scaler state in the `extra` payload.
    """
    extra = {
        "epoch": epoch,
        "val_loss": val_loss,
        "train_metrics": {"loss": val_loss + 0.1},
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        extra["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        extra["scaler_state"] = scaler.state_dict()
    adapter.save_checkpoint(model, optimizer, str(path), extra)


def _build_runtime(cfg):
    """Build the (model, optimizer, scheduler) trio the Trainer would build."""
    adapter = PyTorchAdapter(cfg)
    model = adapter.build_model()
    optimizer = adapter.build_optimizer(model)
    scheduler = adapter.build_scheduler(optimizer)
    return adapter, model, optimizer, scheduler


# ---------------------------------------------------------------------------
# Checkpoint round-trip carries enough state for a full resume
# ---------------------------------------------------------------------------

class TestCheckpointPersistence:
    """Verifies the trainer's `save_checkpoint` extra dict captures
    everything `_apply_resume(full)` later needs. If these fields stop
    being saved, --resume regresses silently — we'd lose only the
    optimizer state, which is hard to spot from the metrics alone."""

    def test_extra_carries_scheduler_and_best_metrics(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        adapter, model, optimizer, scheduler = _build_runtime(cfg)
        # Step the scheduler once so its state is non-default.
        scheduler.step()

        ckpt_path = tmp_path / "ckpt.pt"
        _save_fake_checkpoint(adapter, model, optimizer, ckpt_path,
                              epoch=4, scheduler=scheduler)

        payload = torch.load(str(ckpt_path), weights_only=False)
        assert "state_dict" in payload
        assert "optimizer_state" in payload
        assert "scheduler_state" in payload
        assert payload["epoch"] == 4
        assert payload["best_epoch"] == 3
        # We round-tripped the scheduler — _last_lr should match.
        assert payload["scheduler_state"]["_last_lr"] == scheduler.state_dict()["_last_lr"]


# ---------------------------------------------------------------------------
# _apply_resume — the heart of the feature
# ---------------------------------------------------------------------------

class TestApplyResumeFull:
    """`--resume`: restore everything, continue counting epochs."""

    def test_returns_correct_start_epoch(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        adapter, model, optimizer, scheduler = _build_runtime(cfg)
        ckpt = tmp_path / "ckpt.pt"
        _save_fake_checkpoint(adapter, model, optimizer, ckpt,
                              epoch=7, best_val_loss=0.12, best_epoch=6,
                              scheduler=scheduler)

        # Fresh trainer state for the resume target.
        cfg2 = _make_cfg(tmp_path, num_epochs=20)
        trainer = Trainer(cfg2)
        _, model2, opt2, sched2 = _build_runtime(cfg2)

        start_epoch, best_val_loss, best_epoch = trainer._apply_resume(
            model2, opt2, sched2, None, str(ckpt), "full",
        )
        assert start_epoch == 8           # last epoch + 1
        assert best_val_loss == pytest.approx(0.12)
        assert best_epoch == 6

    def test_optimizer_state_restored(self, tmp_path):
        cfg = _make_cfg(tmp_path, lr=0.123)   # distinctive LR
        adapter, model, optimizer, scheduler = _build_runtime(cfg)
        # Step the optimizer with a real grad so its state has Adam moments.
        x = torch.randn(2, 4)
        y = torch.tensor([0, 1])
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()

        ckpt = tmp_path / "ckpt.pt"
        _save_fake_checkpoint(adapter, model, optimizer, ckpt,
                              scheduler=scheduler)

        # New optimizer at a different LR — resume should overwrite it.
        cfg2 = _make_cfg(tmp_path, lr=999.0)
        trainer = Trainer(cfg2)
        _, model2, opt2, sched2 = _build_runtime(cfg2)
        trainer._apply_resume(model2, opt2, sched2, None,
                              str(ckpt), "full")

        # The restored optimizer's `state` dict should have entries
        # (Adam exp_avg / exp_avg_sq) for our parameters.
        assert any(opt2.state.values()), \
            "Optimizer state was not restored — Adam moments are empty after --resume."

    def test_weight_load_is_strict_warn_not_fail(self, tmp_path, capsys):
        """If the resumed checkpoint has extra/missing keys (architecture
        drift) we warn and continue rather than aborting the run."""
        cfg = _make_cfg(tmp_path)
        adapter, model, optimizer, scheduler = _build_runtime(cfg)
        ckpt = tmp_path / "ckpt.pt"
        _save_fake_checkpoint(adapter, model, optimizer, ckpt,
                              scheduler=scheduler)
        # Corrupt the checkpoint: add a key the new model won't have.
        payload = torch.load(str(ckpt), weights_only=False)
        payload["state_dict"]["some_phantom_layer.weight"] = torch.zeros(1)
        torch.save(payload, str(ckpt))

        cfg2 = _make_cfg(tmp_path)
        trainer = Trainer(cfg2)
        _, model2, opt2, sched2 = _build_runtime(cfg2)
        # Should NOT raise — strict=False handling tolerates the drift.
        start_epoch, _, _ = trainer._apply_resume(
            model2, opt2, sched2, None, str(ckpt), "full",
        )
        assert start_epoch >= 1


class TestApplyResumeWeightsOnly:
    """`--finetune`: model weights only, fresh optimizer + scheduler."""

    def test_resets_epoch_counter(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        adapter, model, optimizer, scheduler = _build_runtime(cfg)
        ckpt = tmp_path / "ckpt.pt"
        _save_fake_checkpoint(adapter, model, optimizer, ckpt,
                              epoch=42, best_val_loss=0.05, best_epoch=40,
                              scheduler=scheduler)

        cfg2 = _make_cfg(tmp_path)
        trainer = Trainer(cfg2)
        _, model2, opt2, sched2 = _build_runtime(cfg2)
        start_epoch, best_val_loss, best_epoch = trainer._apply_resume(
            model2, opt2, sched2, None, str(ckpt), "weights_only",
        )
        # Fine-tune semantics: no resume of metric history.
        assert start_epoch == 1
        assert best_val_loss == float("inf")
        assert best_epoch == 0

    def test_does_not_restore_optimizer_state(self, tmp_path):
        """The whole point of --finetune is a fresh optimizer. If we
        accidentally restore Adam moments, fine-tuning at a new LR will
        diverge."""
        cfg = _make_cfg(tmp_path)
        adapter, model, optimizer, scheduler = _build_runtime(cfg)
        # Populate optimizer state with a real grad step.
        x = torch.randn(2, 4)
        y = torch.tensor([0, 1])
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()

        ckpt = tmp_path / "ckpt.pt"
        _save_fake_checkpoint(adapter, model, optimizer, ckpt,
                              scheduler=scheduler)

        cfg2 = _make_cfg(tmp_path)
        trainer = Trainer(cfg2)
        _, model2, opt2, sched2 = _build_runtime(cfg2)
        trainer._apply_resume(model2, opt2, sched2, None,
                              str(ckpt), "weights_only")
        # opt2 should be a fresh Adam — no per-param state.
        assert not any(opt2.state.values()), \
            "Optimizer state leaked into --finetune; should be empty."

    def test_weights_actually_match_checkpoint(self, tmp_path):
        """Sanity: after fine-tune, the loaded model's weights equal the
        checkpoint's, even though everything else was reset."""
        cfg = _make_cfg(tmp_path)
        adapter, model, optimizer, scheduler = _build_runtime(cfg)
        # Stamp an obvious value into the model's first layer so we can
        # verify it's restored after the round-trip.
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.5)

        ckpt = tmp_path / "ckpt.pt"
        _save_fake_checkpoint(adapter, model, optimizer, ckpt,
                              scheduler=scheduler)

        cfg2 = _make_cfg(tmp_path)
        trainer = Trainer(cfg2)
        _, model2, opt2, sched2 = _build_runtime(cfg2)
        trainer._apply_resume(model2, opt2, sched2, None,
                              str(ckpt), "weights_only")

        for p in model2.parameters():
            assert torch.all(p == 0.5).item()


class TestApplyResumeErrors:

    def test_missing_file_raises_filenotfound(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        trainer = Trainer(cfg)
        _, model, opt, sched = _build_runtime(cfg)
        with pytest.raises(FileNotFoundError):
            trainer._apply_resume(model, opt, sched, None,
                                   str(tmp_path / "ghost.pt"), "full")

    def test_unknown_mode_raises(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        adapter, model, optimizer, scheduler = _build_runtime(cfg)
        ckpt = tmp_path / "ckpt.pt"
        _save_fake_checkpoint(adapter, model, optimizer, ckpt,
                              scheduler=scheduler)
        trainer = Trainer(cfg)
        _, model2, opt2, sched2 = _build_runtime(cfg)
        with pytest.raises(ValueError, match="resume_mode"):
            trainer._apply_resume(model2, opt2, sched2, None,
                                   str(ckpt), "bogus_mode")

    def test_non_neuralforge_checkpoint_rejected(self, tmp_path):
        """A bare torch.save({...}) without our 'state_dict' key surfaces
        a clear error rather than crashing later in load_state_dict."""
        ckpt = tmp_path / "ckpt.pt"
        torch.save({"some_unrelated_key": 1}, str(ckpt))

        cfg = _make_cfg(tmp_path)
        trainer = Trainer(cfg)
        _, model, opt, sched = _build_runtime(cfg)
        with pytest.raises(RuntimeError, match="not a NeuralForge checkpoint"):
            trainer._apply_resume(model, opt, sched, None, str(ckpt), "full")


# ---------------------------------------------------------------------------
# CLI helper: --resume-from-best resolution
# ---------------------------------------------------------------------------

class TestResumeFromBestResolver:

    def test_resolves_run_directory(self, tmp_path, monkeypatch):
        from neural_platform.cli.commands import _resolve_best_for_run
        run_dir = tmp_path / "my_run"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        best = ckpts / "checkpoint_best.pt"
        best.write_bytes(b"\x80")
        out = _resolve_best_for_run(str(run_dir))
        assert out == str(best.resolve())

    def test_resolves_direct_pt_path(self, tmp_path):
        from neural_platform.cli.commands import _resolve_best_for_run
        ckpt = tmp_path / "x.pt"
        ckpt.write_bytes(b"\x80")
        out = _resolve_best_for_run(str(ckpt))
        assert out == str(ckpt.resolve())

    def test_falls_back_to_newest_pt(self, tmp_path):
        from neural_platform.cli.commands import _resolve_best_for_run
        run_dir = tmp_path / "my_run"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        ckpt = ckpts / "checkpoint_epoch_0050.pt"
        ckpt.write_bytes(b"\x80")
        out = _resolve_best_for_run(str(run_dir))
        assert out == str(ckpt.resolve())

    def test_returns_none_when_nothing_found(self, tmp_path):
        from neural_platform.cli.commands import _resolve_best_for_run
        out = _resolve_best_for_run(str(tmp_path / "no_such_run"))
        assert out is None
