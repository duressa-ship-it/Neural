"""
NeuralForge Trainer
Unified training loop with early stopping, LR scheduling, checkpointing,
live logging via Rich, experiment tracking, and real-time SSE event streaming.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table

from neural_platform.core.config import ExperimentConfig, Scheduler
from neural_platform.core.evaluator import Evaluator, MetricAccumulator
from neural_platform.core.experiment import ExperimentTracker
from neural_platform.core.event_bus import TrainingEventWriter
from neural_platform.frameworks.factory import get_adapter

console = Console()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


class _BatchModalityError(RuntimeError):
    """Raised when the first batch's shape doesn't fit the configured model."""


def _describe_batch(batch) -> str:
    """Human-readable shape summary for error messages."""
    if isinstance(batch, torch.Tensor):
        return f"Tensor{tuple(batch.shape)} dtype={batch.dtype}"
    if isinstance(batch, dict):
        return "{" + ", ".join(f"{k}: {_describe_batch(v)}" for k, v in batch.items()) + "}"
    if isinstance(batch, (list, tuple)):
        n = len(batch)
        if n > 4:
            sample = ", ".join(_describe_batch(b) for b in list(batch)[:3]) + f", … ({n} items)"
        else:
            sample = ", ".join(_describe_batch(b) for b in batch)
        return f"{type(batch).__name__}({sample})"
    return type(batch).__name__


class EarlyStopping:
    """Stops training when monitored metric stops improving."""

    def __init__(self, patience: int, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0
        self.stopped = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
            return False
        self._counter += 1
        if self._counter >= self.patience:
            self.stopped = True
            return True
        return False


class Trainer:
    """
    High-level trainer. Orchestrates:
      - Framework adapter (PyTorch / TF / JAX)
      - Training loop with gradient accumulation
      - Validation, LR scheduling, checkpointing, early stopping
      - Experiment & metric tracking (SQLite)
      - Real-time event streaming (JSONL → SSE dashboard)
      - Rich progress display
    """

    def __init__(self, config: ExperimentConfig, db_path: Optional[str] = None):
        self.config = config
        self.adapter = get_adapter(config)
        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        db_path = db_path or str(Path(config.output_dir) / "neuralforge.db")
        self.tracker = ExperimentTracker(db_path)

        # Live event stream — one file PER RUN, written next to the config.
        # Previously this was `<output_dir>/live_events.jsonl` shared across
        # all experiments, which meant a second `neural train` would clobber
        # the first run's stream. Per-run files let the dashboard track
        # multiple concurrent runs and stream each independently.
        events_path = self.run_dir / "live_events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events = TrainingEventWriter(events_path)
        self.events_path = events_path

    def fit(
        self,
        train_loader,
        val_loader=None,
        *,
        resume_from: Optional[str] = None,
        resume_mode: Literal["full", "weights_only"] = "full",
    ) -> Dict[str, Any]:
        """Run the training loop.

        ``resume_from`` is a path to a NeuralForge checkpoint (.pt) saved by
        :meth:`PyTorchAdapter.save_checkpoint`. When set, the model's
        ``state_dict`` is restored before training begins.

        ``resume_mode`` controls how much state is restored:

        * ``"full"``    — restores optimizer state, scheduler state, GradScaler
                          state if present, last completed epoch, and best
                          val loss / best epoch counters. Use this to **continue**
                          a run that was interrupted (or to extend its
                          ``num_epochs`` for more training).
        * ``"weights_only"`` — restores only the model weights. Optimizer +
                          scheduler start fresh, epoch counter resets to 1, best
                          metrics reset. Use this for **fine-tuning**: a new
                          loss landscape or a smaller LR makes a new optimizer
                          state more useful than the saved one.

        The two modes correspond to the ``--resume`` and ``--finetune`` flags
        on ``neural train`` respectively.
        """
        cfg = self.config
        train_cfg = cfg.training
        _set_seed(train_cfg.seed)

        # Sanity-check the first batch so modality mismatches surface in seconds,
        # not 50 epochs in. We peek at one batch, validate the input shape vs.
        # the model's expectation, and let the DataLoader keep its iterator.
        try:
            self._sanity_check_first_batch(train_loader)
        except _BatchModalityError as exc:
            console.print(f"[red]Batch sanity check failed:[/] {exc}")
            raise

        console.rule(f"[bold blue]NeuralForge — {cfg.name}")
        console.print(f"[dim]Output dir:[/] {self.run_dir}")
        console.print(f"[dim]Framework:[/]  {cfg.model.framework.value}")
        console.print(f"[dim]Model:[/]       {cfg.model.type.value} ({cfg.model.name})")

        # Build components
        model = self.adapter.build_model()
        optimizer = self.adapter.build_optimizer(model)
        scheduler = self.adapter.build_scheduler(optimizer)
        loss_fn = self.adapter.build_loss()
        scaler = self.adapter.make_scaler() if hasattr(self.adapter, "make_scaler") else None
        evaluator = Evaluator(self.adapter, loss_fn)

        if hasattr(model, "count_parameters"):
            n_params = model.count_parameters()
            console.print(f"[dim]Parameters:[/]  {n_params:,}")
        device = self.adapter.get_device()
        console.print(f"[dim]Device:[/]      {device}")

        # ── Resume / fine-tune from a prior checkpoint ───────────────────────
        # Done after the optimizer/scheduler/scaler are built so we can
        # restore their states in-place. start_epoch defaults to 1 for fresh
        # runs; resume("full") sets it to (last_completed + 1). Best metrics
        # carry over so we don't overwrite a better checkpoint with a worse
        # one written in the first resumed epoch.
        start_epoch = 1
        resumed_best_val_loss = float("inf")
        resumed_best_epoch = 0
        if resume_from:
            start_epoch, resumed_best_val_loss, resumed_best_epoch = (
                self._apply_resume(model, optimizer, scheduler, scaler,
                                   resume_from, resume_mode)
            )

        console.print()

        total_batches = len(train_loader)

        # Emit training_start so the dashboard knows a run began
        self.events.training_start(
            experiment=cfg.name,
            model_type=cfg.model.type.value,
            framework=cfg.model.framework.value,
            total_epochs=train_cfg.num_epochs,
            total_batches=total_batches,
            device=str(device),
            resume_from=str(Path(resume_from).resolve()) if resume_from else None,
            resume_mode=resume_mode if resume_from else None,
            resume_start_epoch=start_epoch if resume_from else None,
        )

        # SQLite experiment tracking
        exp_id = self.tracker.create_experiment(
            name=cfg.name,
            config=cfg,
            description=cfg.description or "",
            tags=cfg.tags,
        )
        run_id = self.tracker.start_run(
            exp_id,
            framework=cfg.model.framework.value,
            device=str(device),
        )

        early_stopper = (
            EarlyStopping(train_cfg.early_stopping_patience)
            if train_cfg.early_stopping_patience
            else None
        )

        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "epochs": []}
        best_val_loss = resumed_best_val_loss
        best_epoch = resumed_best_epoch
        best_ckpt_path: Optional[str] = None
        start_time = time.time()
        status = "completed"

        # Guard: if the user resumed from a checkpoint at epoch N but the
        # config's num_epochs is <= N, there's nothing to do. Return early
        # with the already-completed state so we don't write a synthetic
        # "training_end" with zero epochs and confuse the dashboard.
        if start_epoch > train_cfg.num_epochs:
            console.print(
                f"[yellow]Nothing to do:[/] resumed at epoch {start_epoch} but "
                f"training.num_epochs is only {train_cfg.num_epochs}. Bump "
                f"`training.num_epochs` to extend the run."
            )
            self.events.training_end(
                experiment=cfg.name,
                status="completed",
                best_epoch=best_epoch,
                best_val_loss=best_val_loss if best_val_loss < float("inf") else None,
                total_epochs=0,
                duration=0.0,
            )
            return {
                "history": history,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_checkpoint": resume_from,
                "experiment_id": None,
                "run_id": None,
                "resumed": True,
                "resume_mode": resume_mode,
                "no_op": True,
            }

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                epoch_task = progress.add_task("[cyan]Training", total=train_cfg.num_epochs)
                if start_epoch > 1:
                    progress.advance(epoch_task, advance=start_epoch - 1)

                for epoch in range(start_epoch, train_cfg.num_epochs + 1):
                    # ── Training phase ──────────────────────────────────
                    model.train()
                    train_acc = MetricAccumulator()
                    optimizer.zero_grad()

                    batch_task = progress.add_task(
                        f"  Epoch {epoch}/{train_cfg.num_epochs}", total=total_batches
                    )

                    for step, batch in enumerate(train_loader, 1):
                        loss_val, metrics = self.adapter.train_step(
                            model, batch, optimizer, loss_fn, scaler
                        )
                        train_acc.update(metrics, n=1)

                        if step % train_cfg.accumulation_steps == 0:
                            if hasattr(self.adapter, "optimizer_step"):
                                self.adapter.optimizer_step(model, optimizer, scaler)
                            else:
                                optimizer.step()
                                optimizer.zero_grad()

                        # ── Emit batch event every log_every steps ──
                        if step % train_cfg.log_every == 0 or step == total_batches:
                            self.events.batch(
                                experiment=cfg.name,
                                epoch=epoch,
                                total_epochs=train_cfg.num_epochs,
                                batch=step,
                                total_batches=total_batches,
                                loss=loss_val,
                                metrics={k: v for k, v in metrics.items() if k != "loss"},
                                lr=_current_lr(optimizer),
                            )

                        progress.advance(batch_task)

                    # Flush remaining accumulation
                    if total_batches % train_cfg.accumulation_steps != 0:
                        if hasattr(self.adapter, "optimizer_step"):
                            self.adapter.optimizer_step(model, optimizer, scaler)

                    progress.remove_task(batch_task)
                    train_metrics = train_acc.compute()
                    history["train_loss"].append(train_metrics.get("loss", 0.0))

                    # ── Validation phase ─────────────────────────────────
                    val_metrics: Dict[str, float] = {}
                    if val_loader is not None:
                        val_metrics = evaluator.evaluate(model, val_loader, phase="val")
                        history["val_loss"].append(val_metrics.get("loss", float("inf")))
                    history["epochs"].append(epoch)

                    # SQLite metrics
                    self.tracker.log_metrics(run_id, epoch, "train", train_metrics)
                    if val_metrics:
                        self.tracker.log_metrics(run_id, epoch, "val", val_metrics)

                    # LR scheduling
                    if scheduler is not None:
                        sched_val = val_metrics.get("loss", train_metrics.get("loss", 0.0))
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(sched_val)
                        else:
                            scheduler.step()

                    # ── Emit epoch event ──────────────────────────────────
                    self.events.epoch(
                        experiment=cfg.name,
                        epoch=epoch,
                        total_epochs=train_cfg.num_epochs,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        lr=_current_lr(optimizer),
                        elapsed=time.time() - start_time,
                    )

                    # ── Checkpointing ──────────────────────────────────────
                    val_loss = val_metrics.get("loss", float("inf"))
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        best_epoch = epoch

                    ckpt_dir = cfg.checkpoint_dir
                    ckpt_dir.mkdir(parents=True, exist_ok=True)

                    if is_best or (epoch % train_cfg.checkpoint_every == 0):
                        ckpt_suffix = "best" if is_best else f"epoch_{epoch:04d}"
                        ckpt_path = str(ckpt_dir / f"checkpoint_{ckpt_suffix}.pt")
                        extra = {
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "train_metrics": train_metrics,
                            # Track best-so-far inside the checkpoint so a
                            # later `--resume` doesn't overwrite a real best
                            # checkpoint with the first resumed epoch's loss.
                            "best_val_loss": (best_val_loss
                                              if best_val_loss < float("inf") else None),
                            "best_epoch": best_epoch,
                        }
                        # Capture optimizer hyperparams + scheduler/scaler
                        # state alongside the model weights so a future
                        # `--resume` can pick up exactly where we left off.
                        # Optimizer state lives on its own key inside the
                        # adapter's payload; everything else rides in extra.
                        if scheduler is not None and hasattr(scheduler, "state_dict"):
                            try:
                                extra["scheduler_state"] = scheduler.state_dict()
                            except Exception:
                                pass
                        if scaler is not None and hasattr(scaler, "state_dict"):
                            try:
                                extra["scaler_state"] = scaler.state_dict()
                            except Exception:
                                pass
                        # Persist class names if discovered upstream
                        class_names = getattr(self.config, "_class_names", None)
                        if class_names:
                            extra["class_names"] = class_names
                        self.adapter.save_checkpoint(
                            model, optimizer, ckpt_path, extra,
                        )
                        if is_best:
                            best_ckpt_path = ckpt_path
                        self.events.checkpoint(cfg.name, epoch, ckpt_path, is_best)

                    self._log_epoch(epoch, train_cfg.num_epochs, train_metrics, val_metrics)
                    progress.advance(epoch_task)

                    # ── Early stopping ────────────────────────────────────
                    if early_stopper and val_metrics:
                        if early_stopper.step(val_metrics.get("loss", float("inf"))):
                            console.print(f"[yellow]Early stopping at epoch {epoch}[/]")
                            self.events.early_stop(cfg.name, epoch, best_epoch, best_val_loss)
                            break

        except KeyboardInterrupt:
            console.print("\n[yellow]Training interrupted by user.[/]")
            status = "interrupted"

        # ── Emit training_end ─────────────────────────────────────────────
        self.events.training_end(
            experiment=cfg.name,
            status=status,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss if best_val_loss < float("inf") else None,
            total_epochs=len(history["epochs"]),
            duration=time.time() - start_time,
        )

        # SQLite finish
        self.tracker.finish_run(
            run_id,
            status=status,
            best_val_loss=best_val_loss if best_val_loss < float("inf") else None,
            best_epoch=best_epoch,
            total_epochs=len(history["epochs"]),
            checkpoint_path=best_ckpt_path,
            started_at=start_time,
        )
        self.tracker.update_experiment_status(exp_id, status)

        self._print_summary(history, best_val_loss, best_epoch, best_ckpt_path, start_time)

        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_checkpoint": best_ckpt_path,
            "experiment_id": exp_id,
            "run_id": run_id,
        }

    def _apply_resume(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        resume_from: str,
        resume_mode: str,
    ):
        """Restore training state from a NeuralForge checkpoint, in-place.

        Returns ``(start_epoch, best_val_loss, best_epoch)`` so the caller's
        loop can pick up at the right epoch and avoid clobbering a real
        best checkpoint on the first resumed epoch's metrics.

        Behavior:

        * **weights_only** — restore ``state_dict`` only. Optimizer / scheduler
          / scaler / epoch counter / best metrics all start fresh. Use for
          fine-tuning, where the loss landscape changes and a stale Adam
          moment estimate would be more harmful than useful.
        * **full** — also restore optimizer, scheduler, scaler, epoch counter,
          and best metrics. Use to **continue** an interrupted run.

        Both modes do a shape-tolerant load: weights with mismatched shapes
        are skipped with a warning instead of failing the whole resume.
        """
        path = Path(resume_from)
        if not path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_from}. "
                "Pass --resume / --finetune with a path to an existing .pt file."
            )

        try:
            payload = torch.load(str(path), map_location=self.adapter.get_device(),
                                 weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint {resume_from}: {exc}"
            )

        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise RuntimeError(
                f"Checkpoint {resume_from} is not a NeuralForge checkpoint "
                "(missing 'state_dict' key). Was this produced by `neural train`?"
            )

        # ----- model weights (always) -----
        sd = payload["state_dict"]
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not apply state_dict from {resume_from}: {exc}. "
                "The checkpoint's model architecture probably doesn't match "
                "the current config — confirm `model.type` and arch params line up."
            )
        if missing:
            console.print(f"[yellow]⚠[/] {len(missing)} param(s) not in checkpoint, "
                          f"randomly initialized: e.g. {missing[:3]}")
        if unexpected:
            console.print(f"[yellow]⚠[/] {len(unexpected)} extra param(s) in "
                          f"checkpoint, ignored: e.g. {unexpected[:3]}")

        if resume_mode == "weights_only":
            console.print(
                f"[green]✓[/] Fine-tune mode: loaded weights from "
                f"[bold]{path.name}[/]; optimizer/scheduler start fresh."
            )
            return 1, float("inf"), 0

        if resume_mode != "full":
            raise ValueError(
                f"Unknown resume_mode {resume_mode!r}; expected 'full' or 'weights_only'."
            )

        # ----- full state restore -----
        try:
            if "optimizer_state" in payload and optimizer is not None:
                optimizer.load_state_dict(payload["optimizer_state"])
        except Exception as exc:
            console.print(f"[yellow]⚠ Could not restore optimizer state:[/] {exc}. "
                          "Continuing with a fresh optimizer.")

        try:
            if "scheduler_state" in payload and scheduler is not None and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(payload["scheduler_state"])
        except Exception as exc:
            console.print(f"[yellow]⚠ Could not restore scheduler state:[/] {exc}.")

        try:
            if "scaler_state" in payload and scaler is not None and hasattr(scaler, "load_state_dict"):
                scaler.load_state_dict(payload["scaler_state"])
        except Exception:
            pass

        last_epoch = int(payload.get("epoch", 0) or 0)
        best_val_loss = payload.get("best_val_loss")
        best_epoch = int(payload.get("best_epoch", last_epoch) or 0)
        if best_val_loss is None:
            # Older checkpoints didn't track best separately — fall back to
            # this epoch's val loss so we don't downgrade good checkpoints.
            bvl = payload.get("val_loss")
            best_val_loss = float(bvl) if bvl is not None else float("inf")

        start_epoch = last_epoch + 1
        console.print(
            f"[green]✓[/] Resume mode: continuing from epoch {start_epoch} "
            f"(best so far: val_loss={best_val_loss:.6f} @ epoch {best_epoch}, "
            f"checkpoint={path.name})."
            if best_val_loss < float("inf") else
            f"[green]✓[/] Resume mode: continuing from epoch {start_epoch} "
            f"(checkpoint={path.name})."
        )
        return start_epoch, float(best_val_loss), best_epoch

    def _sanity_check_first_batch(self, train_loader) -> None:
        """
        Pull one batch from the loader and verify its shape/type matches what
        the configured model expects. Fails fast with an actionable error
        instead of letting an opaque 'takes 2 positional arguments but N were
        given' surface 30 epochs into the run.

        We do *not* consume the iterator the trainer will use — we just peek
        via `next(iter(train_loader))`, which yields a brand-new iterator.
        """
        try:
            batch = next(iter(train_loader))
        except Exception as exc:
            raise _BatchModalityError(
                f"Could not fetch a batch from the train loader: {exc}"
            )

        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise _BatchModalityError(
                f"Batch is shaped as {_describe_batch(batch)} — expected a "
                "(inputs, targets) tuple."
            )

        inputs = batch[0]
        mtype = self.config.model.type.value

        # Inputs is supposed to be Tensor (most models) or dict (transformer)
        # Both `transformer` and `hf_pipeline` legitimately accept dict batches
        # (text tasks → tokenized {input_ids, attention_mask, ...}). The
        # universal HF wrapper also accepts plain tensors when the task is
        # image/audio/video, so we treat both as identical for the sanity check.
        if mtype in ("transformer", "hf_pipeline"):
            if not isinstance(inputs, dict) and not isinstance(inputs, torch.Tensor):
                raise _BatchModalityError(
                    f"{mtype} expects a tokenized dict batch (with 'input_ids') or a "
                    f"Tensor; got {_describe_batch(inputs)}. "
                    "Hint: for text tasks, set data.transforms.text.tokenizer or "
                    "model.transformer.use_pretrained='bert-base-uncased'. "
                    "For image/audio HF models, the dataloader should yield a Tensor."
                )
            return

        # CNN / MLP / RNN all expect a single Tensor for the input
        if not isinstance(inputs, torch.Tensor):
            sample_types = []
            if isinstance(inputs, (list, tuple)):
                sample_types = [type(x).__name__ for x in list(inputs)[:3]]
            raise _BatchModalityError(
                f"{mtype.upper()} model expects a Tensor input but the dataloader "
                f"yielded {_describe_batch(inputs)}"
                + (f" (first items: {sample_types})" if sample_types else "")
                + ".\nThis usually means the dataset's modality doesn't match the model. "
                f"Common causes:\n"
                f"  • CNN/MLP pointed at a text dataset (e.g. data.dataset_name='imdb' "
                f"with model.type='cnn') — the dataloader gave you raw strings.\n"
                f"  • Custom dataset returning PIL images instead of tensors — add a "
                f"`data.transforms.image` block.\n"
                f"Run `neural validate -c <your-config>` for a structured diagnosis."
            )

        # Sanity-check spatial dims for CNN
        if mtype == "cnn" and inputs.dim() == 4:
            arch = self.config.model.cnn
            _, c, h, w = inputs.shape
            expected = (arch.input_channels, arch.input_height, arch.input_width)
            if (c, h, w) != expected:
                # Soft warn — many image datasets are auto-resized via the
                # transforms config, but a mismatch here will still trip up
                # the CNN's first conv layer.
                console.print(
                    f"[yellow]⚠[/] Batch shape ({c},{h},{w}) doesn't match model "
                    f"input ({expected[0]},{expected[1]},{expected[2]}). "
                    f"Add a resize transform or update model.cnn.input_height/width."
                )

        # MLP feature count check
        if mtype == "mlp" and inputs.dim() == 2:
            arch = self.config.model.mlp
            if inputs.shape[1] != arch.input_size:
                raise _BatchModalityError(
                    f"MLP expects {arch.input_size} features per sample but the "
                    f"dataloader yielded shape {tuple(inputs.shape)}. "
                    f"Either retrain with model.mlp.input_size={inputs.shape[1]} or "
                    f"adjust the dataset."
                )

    def _log_epoch(self, epoch, total, train_metrics, val_metrics):
        parts = [f"Epoch {epoch:>4}/{total}"]
        parts.append(f"train_loss={train_metrics.get('loss', 0):.4f}")
        if "accuracy" in train_metrics:
            parts.append(f"train_acc={train_metrics['accuracy']:.4f}")
        if val_metrics:
            parts.append(f"val_loss={val_metrics.get('loss', 0):.4f}")
            if "accuracy" in val_metrics:
                parts.append(f"val_acc={val_metrics['accuracy']:.4f}")
        console.print("  " + " | ".join(parts))

    def _print_summary(self, history, best_val_loss, best_epoch, ckpt_path, start_time):
        elapsed = time.time() - start_time
        console.print()
        console.rule("[bold green]Training Complete")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Epochs trained", str(len(history["epochs"])))
        table.add_row("Best val loss", f"{best_val_loss:.6f}" if best_val_loss < 1e9 else "N/A")
        table.add_row("Best epoch", str(best_epoch))
        table.add_row("Duration", f"{elapsed:.1f}s")
        table.add_row("Best checkpoint", str(ckpt_path or "none"))
        console.print(table)
