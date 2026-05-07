"""
NeuralForge — Training Run Lifecycle Manager.

Tracks zero or more concurrent `neural train` subprocesses spawned from
the dashboard. Lets the user start, stop, and monitor each one
independently — previous versions only kept a single `train_proc` slot,
so a second start would 409 and the Training/Live tabs only ever showed
the last run.

Each run gets:
  * a stable `run_id` (used in API URLs)
  * an absolute config path
  * a per-run `train.log` (so logs don't clobber across runs)
  * a per-run `live_events.jsonl` (the trainer already writes one per
    run dir as of v0.3.6 — we just read it back per-id)

The manager itself is thread-safe; the registry and PTY bridges are
guarded by a lock.

Tokens / secrets concern: training subprocesses don't accept network
input, so no bearer token is needed here — auth is enforced by the
dashboard endpoint that owns the manager.
"""

from __future__ import annotations

import os
import secrets
import subprocess
# import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    """Public-facing summary of a managed training run.

    Anything the UI / API consumes goes here. PTY file descriptors and
    bridge threads stay on the internal `_ManagedRun` companion.
    """
    id:          str
    name:        str                        # experiment name
    config_path: str                        # absolute path
    log_path:    str                        # absolute path to per-run train.log
    events_path: str                        # absolute path to per-run live_events.jsonl
    pid:         Optional[int] = None
    started_at:  float = 0.0
    status:      str = "starting"           # starting | running | exited | failed | stopped
    exit_code:   Optional[int] = None
    last_event:  Optional[Dict[str, Any]] = None  # newest event for at-a-glance progress

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _ManagedRun:
    """Internal companion to RunInfo carrying file handles + bridge state."""
    info:           RunInfo
    proc:           subprocess.Popen
    log_fh:         Optional[Any] = None
    bridge_thread:  Optional[threading.Thread] = None
    bridge_stop:    Optional[threading.Event] = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class TrainingRunManager:
    """Tracks training subprocesses by run id.

    `start()` spawns a `neural train` subprocess. The caller (web/app.py)
    is responsible for resolving the config path and validating the
    config — we just take an already-good config_path and a list of
    overrides.

    `pty_spawner` is the existing `_spawn_with_pty_log` helper from the
    web module; we receive it as a callable so we don't drag PTY logic
    into a new module.
    """

    def __init__(self, output_dir: str = "runs",
                 pty_spawner: Optional[Callable] = None,
                 neural_cmd: Optional[Callable] = None) -> None:
        self.output_dir = Path(output_dir).resolve()
        self._runs: Dict[str, _ManagedRun] = {}
        self._lock = threading.Lock()
        self._pty_spawner = pty_spawner
        self._neural_cmd = neural_cmd or (lambda: ["neural"])

    # ------------------------------------------------------------------
    def start(self,
              config_path: str,
              overrides: Optional[List[str]] = None,
              experiment_name: Optional[str] = None) -> RunInfo:
        cfg_path = Path(config_path).resolve()
        if not cfg_path.exists() or not cfg_path.is_file():
            raise ValueError(f"Config not found: {config_path}")

        run_dir = cfg_path.parent
        # Per-run log file. Including a short suffix prevents two starts of
        # the same config (re-runs) from overwriting each other.
        rid = _generate_id()
        log_path = run_dir / f"train_{rid[:6]}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Trainer (core/trainer.py) writes events to <run_dir>/live_events.jsonl
        # but a re-run truncates that file. Our per-run id makes the log path
        # unique; for the events path we accept the trainer's choice and rely
        # on the dashboard reading the latest events from a tracked
        # `started_at` timestamp.
        events_path = run_dir / "live_events.jsonl"

        cmd = self._neural_cmd() + ["train", "--config", str(cfg_path)]
        for ov in overrides or []:
            cmd += ["--override", ov]

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        if self._pty_spawner is not None:
            proc, log_fh, bridge_thread, bridge_stop = self._pty_spawner(cmd, env, log_path)
        else:
            log_fh = open(log_path, "ab", buffering=0)
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, env=env)
            bridge_thread = None
            bridge_stop = None

        info = RunInfo(
            id=rid,
            name=experiment_name or run_dir.name,
            config_path=str(cfg_path),
            log_path=str(log_path),
            events_path=str(events_path),
            pid=proc.pid,
            started_at=time.time(),
            status="running",
        )
        managed = _ManagedRun(
            info=info, proc=proc, log_fh=log_fh,
            bridge_thread=bridge_thread, bridge_stop=bridge_stop,
        )
        with self._lock:
            self._runs[rid] = managed
        return info

    def stop(self, run_id: str) -> bool:
        """Terminate one run by id. Returns True if it was managed."""
        with self._lock:
            managed = self._runs.get(run_id)
        if not managed:
            return False
        proc = managed.proc
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)
        except Exception:
            pass
        managed.info.status = "stopped"
        managed.info.exit_code = proc.returncode
        self._teardown_bridge(managed)
        return True

    def list(self) -> List[RunInfo]:
        with self._lock:
            runs = list(self._runs.values())
        for managed in runs:
            self._refresh(managed)
        # Newest first — most useful default for the UI
        runs.sort(key=lambda m: m.info.started_at, reverse=True)
        return [m.info for m in runs]

    def get(self, run_id: str) -> Optional[RunInfo]:
        with self._lock:
            managed = self._runs.get(run_id)
        if not managed:
            return None
        self._refresh(managed)
        return managed.info

    def remove(self, run_id: str) -> bool:
        """Stop (if running) and forget a run. Frees file descriptors."""
        self.stop(run_id)
        with self._lock:
            managed = self._runs.pop(run_id, None)
        if managed:
            self._teardown_bridge(managed)
            return True
        return False

    def active_run_id(self) -> Optional[str]:
        """Return the id of the most recently started still-running run, or
        None. Used by backwards-compat single-run endpoints."""
        with self._lock:
            runs = list(self._runs.values())
        for managed in sorted(runs, key=lambda m: m.info.started_at, reverse=True):
            self._refresh(managed)
            if managed.info.status == "running":
                return managed.info.id
        # No running run — fall back to the most recent one (so logs/status
        # for a just-finished run still answer).
        if runs:
            runs.sort(key=lambda m: m.info.started_at, reverse=True)
            return runs[0].info.id
        return None

    # ------------------------------------------------------------------
    def _refresh(self, managed: _ManagedRun) -> None:
        rc = managed.proc.poll()
        if rc is None:
            if managed.info.status == "starting":
                managed.info.status = "running"
        else:
            if managed.info.status not in ("stopped",):
                managed.info.status = "exited" if rc == 0 else "failed"
            managed.info.exit_code = rc
            self._teardown_bridge(managed)

    def _teardown_bridge(self, managed: _ManagedRun) -> None:
        try:
            if managed.bridge_stop is not None:
                managed.bridge_stop.set()
        except Exception:
            pass
        try:
            if managed.bridge_thread is not None:
                managed.bridge_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if managed.log_fh is not None:
                managed.log_fh.close()
        except Exception:
            pass
        managed.bridge_stop = None
        managed.bridge_thread = None
        managed.log_fh = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id() -> str:
    return secrets.token_hex(6)
