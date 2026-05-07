"""
NeuralForge Event Bus
File-based inter-process event streaming between the CLI trainer and the web dashboard.

The trainer writes JSONL events to a file; the dashboard SSE endpoint tails it
and pushes events to connected browser clients via Server-Sent Events.

Event types:
  training_start  — emitted once when training begins
  batch           — emitted every log_every batches (live loss/acc)
  epoch           — emitted at end of each epoch (train + val metrics)
  checkpoint      — emitted when a checkpoint is saved
  early_stop      — emitted when early stopping triggers
  training_end    — emitted when training finishes or is interrupted
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator, Optional


# ---------------------------------------------------------------------------
# Event schema helpers
# ---------------------------------------------------------------------------

def _event(type_: str, **kwargs) -> dict:
    return {"type": type_, "ts": time.time(), **kwargs}


# ---------------------------------------------------------------------------
# Writer  (used by the trainer — synchronous, file-append)
# ---------------------------------------------------------------------------

class TrainingEventWriter:
    """
    Appends JSON-lines events to a file so the dashboard can tail it.
    Safe to call from a synchronous training loop.
    """

    def __init__(self, events_path: str | Path):
        self.path = Path(events_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate/create fresh on each training run
        self.path.write_text("")

    def _write(self, event: dict):
        line = json.dumps(event, default=str) + "\n"
        with open(self.path, "a") as f:
            f.write(line)

    def training_start(self, experiment: str, model_type: str, framework: str,
                       total_epochs: int, total_batches: int, device: str,
                       resume_from: Optional[str] = None,
                       resume_mode: Optional[str] = None,
                       resume_start_epoch: Optional[int] = None):
        self._write(_event("training_start",
            experiment=experiment,
            model_type=model_type,
            framework=framework,
            total_epochs=total_epochs,
            total_batches=total_batches,
            device=device,
            resume_from=resume_from,
            resume_mode=resume_mode,
            resume_start_epoch=resume_start_epoch,
        ))

    def batch(self, experiment: str, epoch: int, total_epochs: int,
              batch: int, total_batches: int, loss: float,
              metrics: dict, lr: float):
        self._write(_event("batch",
            experiment=experiment,
            epoch=epoch,
            total_epochs=total_epochs,
            batch=batch,
            total_batches=total_batches,
            loss=round(loss, 6),
            metrics={k: round(v, 6) for k, v in metrics.items()},
            lr=round(lr, 8),
        ))

    def epoch(self, experiment: str, epoch: int, total_epochs: int,
              train_metrics: dict, val_metrics: dict, lr: float, elapsed: float):
        self._write(_event("epoch",
            experiment=experiment,
            epoch=epoch,
            total_epochs=total_epochs,
            train_metrics={k: round(v, 6) for k, v in train_metrics.items()},
            val_metrics={k: round(v, 6) for k, v in val_metrics.items()},
            lr=round(lr, 8),
            elapsed=round(elapsed, 1),
        ))

    def checkpoint(self, experiment: str, epoch: int, path: str, is_best: bool):
        self._write(_event("checkpoint",
            experiment=experiment,
            epoch=epoch,
            path=path,
            is_best=is_best,
        ))

    def early_stop(self, experiment: str, epoch: int, best_epoch: int, best_val_loss: float):
        self._write(_event("early_stop",
            experiment=experiment,
            epoch=epoch,
            best_epoch=best_epoch,
            best_val_loss=round(best_val_loss, 6),
        ))

    def training_end(self, experiment: str, status: str, best_epoch: int,
                     best_val_loss: Optional[float], total_epochs: int, duration: float):
        self._write(_event("training_end",
            experiment=experiment,
            status=status,
            best_epoch=best_epoch,
            best_val_loss=round(best_val_loss, 6) if best_val_loss is not None else None,
            total_epochs=total_epochs,
            duration=round(duration, 1),
        ))


# ---------------------------------------------------------------------------
# Reader  (used by the dashboard SSE endpoint — async)
# ---------------------------------------------------------------------------

class TrainingEventReader:
    """
    Async tail of a JSONL events file.
    Yields parsed event dicts as they are written by the trainer.

    Robust against:
      - File being truncated/recreated between training runs
        (a new `neural train` rewrites live_events.jsonl, the reader
        detects the shrink and seeks back to 0).
      - File being missing initially.
      - Partial line writes — incomplete trailing JSON is held back
        until the next poll completes the line.
    """

    def __init__(self, events_path: str | Path, poll_interval: float = 0.3):
        self.path = Path(events_path)
        self.poll_interval = poll_interval

    async def tail(self, from_start: bool = True) -> AsyncIterator[dict]:
        """
        Async generator that yields events.
        If from_start=True, replays all existing events first (good for
        clients connecting mid-training to catch up).
        """
        position = 0 if from_start else self._safe_size()
        last_inode: Optional[int] = self._safe_inode()

        while True:
            if not self.path.exists():
                # File may not have been created yet — wait for trainer to start.
                position = 0
                last_inode = None
                await asyncio.sleep(self.poll_interval)
                continue

            # Detect file rotation (new inode) or truncation (size < position)
            current_inode = self._safe_inode()
            current_size = self._safe_size()
            if current_inode != last_inode or current_size < position:
                position = 0
                last_inode = current_inode

            try:
                with open(self.path, "r") as f:
                    f.seek(position)
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        # Skip partial lines (no trailing newline) — they will be
                        # picked up on the next iteration once flushed.
                        if not line.endswith("\n"):
                            break
                        position = f.tell()
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                pass
            except FileNotFoundError:
                position = 0
                last_inode = None

            await asyncio.sleep(self.poll_interval)

    def _safe_size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def _safe_inode(self) -> Optional[int]:
        try:
            return self.path.stat().st_ino
        except FileNotFoundError:
            return None

    async def snapshot(self) -> list[dict]:
        """Return all events currently in the file (non-streaming)."""
        if not self.path.exists():
            return []
        events = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return events
