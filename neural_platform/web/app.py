"""
NeuralForge Web Dashboard
FastAPI backend + single-page HTML frontend.

Endpoints
─────────
REST          /api/experiments[/{id}]       experiment CRUD (GET/DELETE)
              /api/experiments/{id}/metrics
              /api/experiments/search?q=&status=
              /api/runs/{id}/metrics
              /api/checkpoints
              /api/checkpoints/recent
              /api/stats
              /api/configs                   list experiment YAML configs
              /api/configs/save  (POST)      write a new config.yaml
              /api/configs/load?path=        load a config as JSON
              /api/system                    CPU/RAM/GPU snapshot
              /api/health                    dashboard + tracker liveness

Train mgmt    /api/train/status              current subprocess status
              /api/train/start   (POST)      launch `neural train` subprocess
              /api/train/stop    (POST)      terminate running train process
              /api/train/cleanup (POST)      mark stale 'running' rows as 'interrupted'
              /api/train/logs                last N lines of subprocess stdout

Proxy         /api/proxy/health
              /api/proxy/info
              /api/proxy/predict (POST)      forward to inference server (CORS-safe)

Streaming     /api/training/live             snapshot of all events so far
              /api/events/stream             SSE stream (real-time training events)

SPA           /                              serves static/index.html
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import signal as _signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ProxyPredictRequest(BaseModel):
    server_url: str
    inputs: Optional[Any] = None
    tokens: Optional[Any] = None
    attention_mask: Optional[Any] = None
    image_b64: Optional[str] = None
    text: Optional[str] = None
    top_k: int = 5
    return_probabilities: bool = True


class TrainStartRequest(BaseModel):
    config_path: str
    overrides: List[str] = []   # e.g. ["training.lr=0.001", "training.num_epochs=50"]


class SaveConfigRequest(BaseModel):
    name: str                      # experiment name → directory name
    config: Dict[str, Any]         # full ExperimentConfig as a dict
    output_dir: Optional[str] = None
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_neural_cmd() -> list[str]:
    """
    Return the command to invoke the `neural` CLI.
    Prefers the script installed alongside the current Python interpreter
    (works inside a virtualenv); falls back to running as a module.
    """
    neural_script = Path(sys.executable).parent / "neural"
    if neural_script.exists():
        return [str(neural_script)]
    return [sys.executable, "-m", "neural_platform.cli.commands"]


def _append_training_end_event(events_path: Path, experiment: str, last_progress: dict) -> None:
    """
    Append a synthetic training_end event so the SSE tail loop and the live
    training dashboard know the run ended (even when killed mid-way).
    Pulls best-known progress from `last_progress` so the UI doesn't reset
    its counters to zero.
    """
    event = {
        "type": "training_end",
        "ts": time.time(),
        "experiment": experiment,
        "status": "interrupted",
        "best_epoch": last_progress.get("best_epoch", 0) or 0,
        "best_val_loss": last_progress.get("best_val_loss"),
        "total_epochs": last_progress.get("total_epochs", 0) or 0,
        "duration": last_progress.get("duration", 0.0) or 0.0,
    }
    try:
        with open(events_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass


def _scan_last_progress(events_path: Path) -> dict:
    """Read live_events.jsonl and pull out the best-known progress so far."""
    out = {"best_epoch": 0, "best_val_loss": None, "total_epochs": 0, "duration": 0.0}
    if not events_path.exists():
        return out
    try:
        start_ts = None
        last_ts = None
        for line in events_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "training_start":
                start_ts = ev.get("ts")
            last_ts = ev.get("ts", last_ts)
            if ev.get("type") == "epoch":
                out["total_epochs"] = max(out["total_epochs"], int(ev.get("epoch", 0)))
                vl = ev.get("val_metrics", {}).get("loss")
                if vl is not None:
                    if out["best_val_loss"] is None or vl < out["best_val_loss"]:
                        out["best_val_loss"] = vl
                        out["best_epoch"] = int(ev.get("epoch", 0))
        if start_ts and last_ts:
            out["duration"] = max(0.0, last_ts - start_ts)
    except Exception:
        pass
    return out


def _kill_process_group(proc: subprocess.Popen) -> None:
    """
    Three-stage shutdown of the training subprocess and its entire process
    group (DataLoader workers, HF dataset downloaders, tqdm threads):

      1. **SIGINT** — graceful. The HuggingFace dataset downloader honors
         this and aborts mid-shard cleanly; PyTorch DataLoader workers wind
         down their queues; Python's resource tracker gets a chance to
         unregister the leaked semaphores you'd otherwise see in stderr.
      2. **SIGTERM** — escalation if the process is still alive after 4 s.
      3. **SIGKILL** — absolute last resort.

    Handles missing PIDs gracefully (a process that already exited).
    """
    try:
        pgid = os.getpgid(proc.pid)

        # 1) SIGINT: lets HF abort the download and Python clean up resource
        #    tracker semaphores. This is the single biggest win for the
        #    "leaked semaphore objects to clean up" warning the user saw.
        try:
            os.killpg(pgid, _signal.SIGINT)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=4)
            return
        except subprocess.TimeoutExpired:
            pass

        # 2) SIGTERM
        try:
            os.killpg(pgid, _signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=4)
            return
        except subprocess.TimeoutExpired:
            pass

        # 3) SIGKILL — last resort
        try:
            os.killpg(pgid, _signal.SIGKILL)
            proc.wait(timeout=3)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass

    except (ProcessLookupError, OSError):
        pass   # process already gone


def _spawn_with_pty_log(cmd, env, log_path: Path):
    """
    Spawn a subprocess attached to a pseudo-terminal so progress writers
    (tqdm, HF dataset downloaders, rich progress bars) believe stdout is a
    real TTY and emit live updates instead of suppressing themselves.

    Output is bridged into `log_path` by a background thread that reads from
    the master PTY and writes to disk. Carriage-return progress updates land
    in the file as separate "lines" so the dashboard's log endpoint can show
    the latest progress instead of a single ever-growing line.

    Returns `(Popen, log_fh, bridge_thread, stop_event)`. The caller is
    responsible for `stop_event.set()` + `thread.join()` on shutdown.
    """
    import pty as _pty
    import threading as _threading

    master_fd, slave_fd = _pty.openpty()
    log_fh = open(log_path, "w", buffering=1)

    proc = subprocess.Popen(
        cmd,
        stdout=slave_fd,
        stderr=slave_fd,
        stdin=slave_fd,
        env=env,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave_fd)  # parent doesn't need its end; child holds it

    stop_event = _threading.Event()

    def _bridge():
        # Buffer up to a chunk and flush to log_fh, normalising '\r' so each
        # progress update becomes its own line and the dashboard's tail sees
        # newest-first instead of one huge wrapped line.
        buf = b""
        try:
            while not stop_event.is_set():
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                # Normalise: convert solo '\r' into '\n' (so tqdm progress
                # updates flush to disk as discrete lines).
                normalised = buf.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                # Keep only complete lines for atomic writes
                if b"\n" in normalised:
                    last_nl = normalised.rfind(b"\n")
                    head, buf = normalised[:last_nl + 1], normalised[last_nl + 1:]
                    try:
                        log_fh.write(head.decode("utf-8", errors="replace"))
                        log_fh.flush()
                    except Exception:
                        break
                else:
                    buf = normalised
        finally:
            # Flush whatever's left (e.g. final no-newline line) on close
            if buf:
                try:
                    log_fh.write(buf.decode("utf-8", errors="replace"))
                    log_fh.flush()
                except Exception:
                    pass
            try: os.close(master_fd)
            except Exception: pass

    thread = _threading.Thread(target=_bridge, daemon=True, name="forge-pty-bridge")
    thread.start()
    return proc, log_fh, thread, stop_event


_ANSI_RE = None  # lazily compiled below

def _strip_ansi(text: str) -> str:
    """Remove ANSI color/cursor escape sequences. The PTY makes Rich/click
    emit color codes the browser can't render — they look like ``[1m...[0m``
    in the UI. This produces clean human-readable text instead."""
    global _ANSI_RE
    if _ANSI_RE is None:
        import re as _re
        # Matches CSI-style escapes (color + cursor) plus bare control chars.
        _ANSI_RE = _re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return _ANSI_RE.sub("", text)


def _collapse_progress(lines: List[str]) -> List[str]:
    """
    Collapse consecutive tqdm-style progress lines that share a common
    prefix (everything before the percentage), keeping only the latest.

    A real terminal collapses these via `\\r`, but we already split each
    overwrite into its own line on disk. This restores the same visual.
    """
    import re as _re
    pct_re = _re.compile(r"\s+\d{1,3}%")    # the "  19%" or " 100%" marker
    out: List[str] = []
    for ln in lines:
        m = pct_re.search(ln)
        if not m:
            out.append(ln); continue
        prefix = ln[:m.start()]
        if out and out[-1].startswith(prefix) and pct_re.search(out[-1]):
            out[-1] = ln              # replace previous progress for same task
        else:
            out.append(ln)
    return out


def _shutdown_pty_bridge(state: dict) -> None:
    """Stop the PTY-bridge thread and close the log file. Idempotent."""
    stop_evt = state.get("train_bridge_stop")
    if stop_evt is not None:
        try: stop_evt.set()
        except Exception: pass
    thread = state.get("train_bridge_thread")
    if thread is not None:
        try: thread.join(timeout=1.5)
        except Exception: pass
    fh = state.get("train_log_fh")
    if fh is not None:
        try: fh.close()
        except Exception: pass
    state["train_bridge_stop"] = None
    state["train_bridge_thread"] = None
    state["train_log_fh"] = None


def _events_recently_active(events_path: Path, threshold_secs: float = 60.0) -> bool:
    """
    Return True if the live_events.jsonl file has been written to recently
    (i.e. the trainer is probably still alive elsewhere).  Used at startup
    to avoid falsely marking an actively-training CLI run as interrupted.
    """
    if not events_path.exists():
        return False
    try:
        last_line: Optional[str] = None
        for line in events_path.read_text().splitlines():
            if line.strip():
                last_line = line
        if last_line:
            ev = json.loads(last_line)
            if ev.get("type") in ("training_end", "early_stop"):
                return False  # explicitly ended
            ts = ev.get("ts")
            if ts and (time.time() - ts) < threshold_secs:
                return True
    except Exception:
        pass
    # Fallback: mtime
    try:
        return (time.time() - events_path.stat().st_mtime) < threshold_secs
    except Exception:
        return False


def _scan_configs(output_dir: str) -> list[dict]:
    """Return metadata for every config.yaml found under output_dir."""
    configs = []
    for cfg_path in sorted(Path(output_dir).rglob("config.yaml")):
        entry: dict = {
            "path": str(cfg_path),
            "name": cfg_path.parent.name,
            "experiment_name": cfg_path.parent.name,
            "model_type": "?",
            "framework": "?",
            "num_epochs": "?",
            "batch_size": "?",
            "lr": None,
            "data_source": "?",
        }
        try:
            import yaml
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            entry["experiment_name"] = data.get("name", cfg_path.parent.name)
            entry["model_type"]   = data.get("model", {}).get("type", "?")
            entry["framework"]    = data.get("model", {}).get("framework", "?")
            entry["num_epochs"]   = data.get("training", {}).get("num_epochs", "?")
            entry["batch_size"]   = data.get("training", {}).get("batch_size", "?")
            entry["lr"]           = data.get("training", {}).get("optimizer", {}).get("lr")
            entry["data_source"]  = data.get("data", {}).get("source", "?")
            entry["tags"]         = data.get("tags", []) or []
            entry["modified"]     = cfg_path.stat().st_mtime
        except Exception:
            pass
        configs.append(entry)
    return configs


def _parse_available_configs(err_text: str) -> List[str]:
    """Compatibility shim — delegates to core.hf_introspect.parse_available_configs."""
    from neural_platform.core.hf_introspect import parse_available_configs
    return parse_available_configs(err_text)


def _summarize_hf_search_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce a HuggingFace Hub search-result row to the fields the Builder
    actually needs. Infers a coarse modality from the row's `tags` list
    (HF tags include things like `modality:image`, `task_categories:audio-classification`).
    """
    tags = row.get("tags", []) or []
    modality = "unknown"
    # HF tags commonly look like `modality:image`, `modality:audio`, etc.
    for t in tags:
        t = (t or "").lower()
        if t.startswith("modality:"):
            modality = t.split(":", 1)[1].replace("-", "_")
            break
    if modality == "unknown":
        # Fall back to task_categories
        for t in tags:
            t = (t or "").lower()
            if "image" in t:    modality = "image";    break
            if "audio" in t:    modality = "audio";    break
            if "video" in t:    modality = "video";    break
            if "text" in t or "nlp" in t: modality = "text"; break
            if "tabular" in t:  modality = "tabular";  break
            if "time" in t and "series" in t: modality = "time_series"; break

    # Description can be enormous — truncate hard
    desc = (row.get("description") or "").strip().replace("\n", " ")
    if len(desc) > 240:
        desc = desc[:237] + "…"

    return {
        "id":            row.get("id") or row.get("modelId"),
        "downloads":     row.get("downloads") or 0,
        "likes":         row.get("likes") or 0,
        "modality":      modality,
        "tags":          [t for t in tags if not t.startswith("modality:")][:8],
        "description":   desc or None,
        "private":       bool(row.get("private")),
        "gated":         bool(row.get("gated")),
        "lastModified":  row.get("lastModified") or row.get("last_modified"),
    }


def _summarize_preview_row(row: Any) -> Dict[str, Any]:
    """Squash a HF-decoded row into something JSON-safe and small."""
    if not isinstance(row, dict):
        return {"value": str(row)[:120]}
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
            continue
        if isinstance(v, (int, float, bool)):
            out[k] = v
            continue
        if isinstance(v, str):
            out[k] = v[:120] + ("…" if len(v) > 120 else "")
            continue
        # Common HF audio dict: {array: ndarray, sampling_rate: int, path: str}
        if isinstance(v, dict) and "sampling_rate" in v:
            arr = v.get("array")
            n = getattr(arr, "size", None) or (len(arr) if arr is not None else 0)
            out[k] = f"<Audio {v.get('sampling_rate')}Hz, {n} samples>"
            continue
        # PIL-like image
        if hasattr(v, "size") and hasattr(v, "mode"):
            out[k] = f"<Image {v.mode} {v.size[0]}×{v.size[1]}>"
            continue
        # numpy arrays
        if hasattr(v, "shape"):
            out[k] = f"<Array shape={tuple(v.shape)} dtype={getattr(v, 'dtype', '?')}>"
            continue
        # Lists — show length and first few items
        if isinstance(v, list):
            head = v[:3]
            try:
                head_repr = ", ".join(str(x)[:30] for x in head)
            except Exception:
                head_repr = "?"
            out[k] = f"[{len(v)} items: {head_repr}{'…' if len(v) > 3 else ''}]"
            continue
        out[k] = str(type(v).__name__)
    return out


def _nvidia_smi_utilization() -> Dict[int, Dict[str, Any]]:
    """
    Best-effort: shell out to `nvidia-smi` for real GPU utilization%.
    Returns `{gpu_index: {utilization, temperature, power}}`. Empty dict
    if `nvidia-smi` is missing (CPU-only or non-NVIDIA host).
    """
    out: Dict[int, Dict[str, Any]] = {}
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if proc.returncode != 0:
            return out
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0])
                out[idx] = {
                    "utilization": float(parts[1]) if parts[1] != "[N/A]" else None,
                    "temperature": float(parts[2]) if parts[2] != "[N/A]" else None,
                    "power":       float(parts[3]) if parts[3] != "[N/A]" else None,
                }
            except (ValueError, IndexError):
                continue
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return out


def _system_snapshot() -> dict:
    """
    Best-effort system info — never raises.  Includes CPU/RAM/disk via psutil
    if available, GPU info via torch.cuda if a CUDA build is installed.
    """
    info: Dict[str, Any] = {
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()}",
        "python":   platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore
        cpu = psutil.cpu_percent(interval=None)
        vm  = psutil.virtual_memory()
        try:
            disk = psutil.disk_usage(str(Path.cwd()))
            info["disk"] = {
                "total_gb": round(disk.total / 1e9, 2),
                "used_gb":  round(disk.used  / 1e9, 2),
                "percent":  disk.percent,
            }
        except Exception:
            info["disk"] = None
        info["cpu_percent"] = cpu
        info["memory"] = {
            "total_gb": round(vm.total / 1e9, 2),
            "used_gb":  round(vm.used  / 1e9, 2),
            "percent":  vm.percent,
        }
    except Exception:
        info["cpu_percent"] = None
        info["memory"] = None
        info["disk"] = None

    info["gpus"] = []
    info["accelerator"] = None
    try:
        import torch
        if torch.cuda.is_available():
            info["accelerator"] = "cuda"
            # nvidia-smi gives true utilization; fall back to memory if unavailable
            util_map = _nvidia_smi_utilization()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_alloc = torch.cuda.memory_allocated(i)
                mem_total = props.total_memory
                util = util_map.get(i, {}).get("utilization")
                info["gpus"].append({
                    "index":          i,
                    "name":           props.name,
                    "mem_total_gb":   round(mem_total / 1e9, 2),
                    "mem_used_gb":    round(mem_alloc / 1e9, 2),
                    "mem_percent":    round(100.0 * mem_alloc / max(mem_total, 1), 1),
                    "util_percent":   util,
                    "temperature_c":  util_map.get(i, {}).get("temperature"),
                    "power_watts":    util_map.get(i, {}).get("power"),
                    "kind":           "cuda",
                })
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["accelerator"] = "mps"
            # Apple silicon: no per-device memory API, but we can show a clear
            # "available" state instead of an empty cell.
            mps_info = {
                "index": 0,
                "name": "Apple Silicon (MPS)",
                "mem_total_gb": None,
                "mem_used_gb":  None,
                "mem_percent":  None,
                "util_percent": None,
                "kind":         "mps",
                "available":    True,
            }
            # On macOS we can call `ioreg` or `powermetrics` for real GPU util,
            # but those need root. Skip gracefully — UI shows "MPS available".
            try:
                if hasattr(torch.mps, "current_allocated_memory"):
                    used = torch.mps.current_allocated_memory()
                    mps_info["mem_used_gb"] = round(used / 1e9, 2)
            except Exception:
                pass
            info["gpus"].append(mps_info)
        else:
            info["accelerator"] = "cpu"
    except Exception:
        info["accelerator"] = info["accelerator"] or "cpu"

    info["torch_available"] = False
    try:
        import torch  # noqa
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
    except Exception:
        pass

    return info


def _normalize_predict_response(raw: Any) -> Dict[str, Any]:
    """
    Reshape NeuralForge inference-server responses (and a few common
    third-party shapes) into a single canonical form the frontend can render:

        {
          "predictions": [ { "label": "...", "probability": 0.42 }, ... ],
          "model_type":  "...",
          "latency_ms":  1.2,
          "raw":         <original JSON>
        }

    For batched responses (list-of-lists) only the first sample is surfaced
    in `predictions`; the full nested array is preserved under `raw`.
    """
    out: Dict[str, Any] = {
        "predictions": [],
        "model_type": None,
        "latency_ms": None,
        "raw": raw,
    }
    if not isinstance(raw, dict):
        return out

    out["model_type"] = raw.get("model_type")
    out["latency_ms"] = raw.get("latency_ms")

    preds = raw.get("predictions")
    if isinstance(preds, list) and preds:
        first = preds[0]
        # NeuralForge: predictions is List[List[Prediction]], pick first sample
        if not isinstance(first, list):
            first = preds
        out["predictions"] = [_normalize_one_prediction(p) for p in first if isinstance(p, dict)]
        return out

    # Fallback: top_k / class fields
    if "top_k" in raw and isinstance(raw["top_k"], list):
        out["predictions"] = [
            _normalize_one_prediction(p) for p in raw["top_k"] if isinstance(p, dict)
        ]
    elif "class" in raw or "class_id" in raw or "label" in raw:
        out["predictions"] = [_normalize_one_prediction(raw)]
    return out


def _normalize_one_prediction(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull a single prediction dict into the canonical shape, preserving the
    fields the dashboard's Predict UI uses:
      - label       (int|str)  : numeric class id when the server returns one
      - class_name  (str|None) : human-readable label, when the checkpoint had names
      - probability (float|None)
      - score       (float|None)  : raw logit
    """
    label = p.get("label")
    if label is None:
        label = p.get("class")
    if label is None:
        label = p.get("class_id")
    prob = p.get("probability")
    if prob is None:
        prob = p.get("score")
    return {
        "label":       label if label is not None else "?",
        "class_name":  p.get("class_name"),
        "probability": float(prob) if prob is not None else None,
        "score":       float(p["score"]) if p.get("score") is not None else None,
    }


def _normalize_proxy_base_url(raw_url: str) -> str:
    """
    Normalize and validate a proxy base URL (scheme + host + optional port).
    Rejects credentials, paths, queries, and fragments to reduce SSRF surface.
    """
    raw_url = (raw_url or "").strip()
    if not raw_url:
        raise ValueError("server_url is required")

    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("server_url must use http or https")
    if not parsed.hostname:
        raise ValueError("server_url must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("server_url must not include credentials")
    if parsed.path not in ("", "/") or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("server_url must be a bare base URL without path/query/fragment")

    host = parsed.hostname
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port
    return f"{parsed.scheme}://{host}:{port}"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_dashboard_app(output_dir: str = "runs") -> FastAPI:
    from neural_platform.core.experiment import ExperimentTracker
    from neural_platform.core.event_bus import TrainingEventReader

    db_path     = Path(output_dir) / "neuralforge.db"
    events_path = Path(output_dir) / "live_events.jsonl"
    train_log   = Path(output_dir) / "train_subprocess.log"

    app = FastAPI(
        title="NeuralForge Dashboard API",
        version="0.2.0",
        description=(
            "REST API powering the NeuralForge web dashboard. Manages experiments, "
            "training subprocesses, configs, checkpoints, system telemetry, and a "
            "CORS-friendly proxy to running inference servers.\n\n"
            "**Endpoints group by tag:**\n"
            "- **Experiments** — list / inspect / search / delete tracked runs\n"
            "- **Metrics** — per-run and per-experiment training history\n"
            "- **Training** — launch / stop / monitor the training subprocess\n"
            "- **Configs** — discover, load, save, and validate experiment configs\n"
            "- **Checkpoints** — saved `.pt` files with metadata\n"
            "- **System** — host CPU/RAM/GPU/disk telemetry\n"
            "- **Inference** — CORS proxy to a `neural serve` instance\n"
            "- **Live** — server-sent event stream of in-flight training progress"
        ),
        contact={"name": "NeuralForge"},
        openapi_tags=[
            {"name": "System",       "description": "Dashboard health & host telemetry."},
            {"name": "Experiments",  "description": "Manage experiments stored in the SQLite tracker."},
            {"name": "Metrics",      "description": "Per-epoch metrics for runs and experiments."},
            {"name": "Configs",      "description": "Discover, load, save, and validate config YAMLs."},
            {"name": "Training",     "description": "Launch, stop, and monitor the training subprocess."},
            {"name": "Checkpoints",  "description": "Saved model checkpoints (.pt files)."},
            {"name": "Inference",    "description": "CORS-safe proxy to a running inference server."},
            {"name": "Live",         "description": "Real-time training events via Server-Sent Events."},
        ],
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    state: Dict[str, Any] = {
        "train_proc":    None,   # subprocess.Popen | None
        "train_log_fh":  None,   # open log file handle
    }
    configured_proxy_targets = (
        os.environ.get("NEURAL_PROXY_ALLOWED_SERVER_URLS")
        or "http://localhost:8080,http://127.0.0.1:8080,http://[::1]:8080"
    )
    allowed_proxy_targets = {
        _normalize_proxy_base_url(url)
        for url in configured_proxy_targets.split(",")
        if url.strip()
    }

    def _validated_proxy_base_url(server_url: str) -> str:
        try:
            base = _normalize_proxy_base_url(server_url)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if base not in allowed_proxy_targets:
            raise HTTPException(
                400,
                "server_url is not allowed. Set NEURAL_PROXY_ALLOWED_SERVER_URLS to permit additional targets.",
            )
        return base

    @app.on_event("startup")
    async def startup() -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_inst = ExperimentTracker(db_path)
        state["tracker"]             = tracker_inst
        state["events_path"]         = events_path
        state["output_dir"]          = output_dir
        state["train_log_path"]      = train_log
        state["current_experiment"]  = "unknown"
        state["dashboard_started_at"] = time.time()

        # Only mark stale 'running' rows as interrupted if there's no sign of
        # an active CLI training session (live_events.jsonl recently written).
        # Otherwise we'd kill the bookkeeping for a perfectly healthy run that
        # happened to start before the dashboard.
        if not _events_recently_active(events_path):
            try:
                tracker_inst.interrupt_stale_runs()
            except Exception:
                pass

    @app.on_event("shutdown")
    async def shutdown() -> None:
        proc = state.get("train_proc")
        if proc is not None and proc.poll() is None:
            await asyncio.to_thread(_kill_process_group, proc)
        # Stop the PTY bridge thread + close log
        stop_evt = state.get("train_bridge_stop")
        if stop_evt is not None:
            try: stop_evt.set()
            except Exception: pass
        bridge = state.get("train_bridge_thread")
        if bridge is not None:
            try: bridge.join(timeout=1.0)
            except Exception: pass
        fh = state.get("train_log_fh")
        if fh is not None:
            try: fh.close()
            except Exception: pass

    def tracker() -> ExperimentTracker:
        return state["tracker"]

    # ------------------------------------------------------------------
    # Experiments REST API
    # ------------------------------------------------------------------

    @app.get("/api/experiments", tags=["Experiments"], summary="List all experiments")
    async def list_experiments():
        """All experiments, newest first. Includes derived `best_val_loss` and `best_epoch`."""
        return tracker().list_experiments()

    @app.get("/api/experiments/search", tags=["Experiments"],
             summary="Search experiments by name/description and status")
    async def search_experiments(q: Optional[str] = None, status: Optional[str] = None):
        """Filtered list. `q` matches name or description (case-insensitive)."""
        return tracker().search_experiments(q, status)

    @app.get("/api/experiments/{exp_id}", tags=["Experiments"],
             summary="Get one experiment with all its runs")
    async def get_experiment(exp_id: int):
        exp = tracker().get_experiment(exp_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")
        return {"experiment": exp, "runs": tracker().list_runs(exp_id)}

    @app.delete("/api/experiments/{exp_id}", tags=["Experiments"],
                summary="Permanently delete an experiment and its runs/metrics")
    async def delete_experiment(exp_id: int):
        if not tracker().get_experiment(exp_id):
            raise HTTPException(404, "Experiment not found")
        ok = tracker().delete_experiment(exp_id)
        return {"deleted": ok, "id": exp_id}

    @app.get("/api/experiments/{exp_id}/metrics", tags=["Metrics"],
             summary="All per-epoch metrics for an experiment")
    async def get_experiment_metrics(exp_id: int):
        return tracker().get_experiment_metrics(exp_id)

    @app.get("/api/runs/{run_id}/metrics", tags=["Metrics"],
             summary="All per-epoch metrics for a single run")
    async def get_run_metrics(run_id: int):
        return tracker().get_metrics(run_id)

    @app.get("/api/checkpoints", tags=["Checkpoints"], summary="All saved checkpoints")
    async def list_checkpoints():
        import torch
        checkpoints = []
        for pt in sorted(Path(output_dir).rglob("*.pt")):
            entry = {
                "path": str(pt),
                "name": pt.name,
                "size_mb": round(pt.stat().st_size / 1e6, 2),
                "experiment": pt.parent.parent.name,
                "config_path": str(pt.parent.parent / "config.yaml"),
                "epoch": None, "val_loss": None,
                "model_type": None, "model_name": None,
                "modified": pt.stat().st_mtime,
            }
            try:
                # Never allow full pickle deserialization from user-provided checkpoints.
                payload = torch.load(pt, map_location="cpu", weights_only=True)
                if isinstance(payload, dict):
                    entry["epoch"] = payload.get("epoch")
                    entry["val_loss"] = payload.get("val_loss")
                    entry["class_names"] = payload.get("class_names")
                    mc = payload.get("model_config", {})
                    if isinstance(mc, dict):
                        entry["model_type"] = mc.get("type")
                        entry["model_name"] = mc.get("name")
            except Exception:
                pass
            checkpoints.append(entry)
        checkpoints.sort(
            key=lambda c: (c["experiment"], 0 if "best" in c["name"] else 1, c["name"])
        )
        return checkpoints

    @app.get("/api/checkpoints/recent", tags=["Checkpoints"],
             summary="Most recently modified checkpoints")
    async def recent_checkpoints(limit: int = 6):
        cks = await list_checkpoints()
        cks = sorted(cks, key=lambda c: c.get("modified", 0), reverse=True)
        return cks[:limit]

    @app.get("/api/stats", tags=["System"], summary="Dashboard summary counts")
    async def dashboard_stats():
        exps = tracker().list_experiments()
        ckpts = list(Path(output_dir).rglob("*.pt"))
        total_size_mb = sum(p.stat().st_size for p in ckpts) / 1e6
        proc_running = state["train_proc"] is not None and state["train_proc"].poll() is None
        return {
            "total_experiments":  len(exps),
            "completed":          sum(1 for e in exps if e["status"] == "completed"),
            "running":            sum(1 for e in exps if e["status"] == "running"),
            "interrupted":        sum(1 for e in exps if e["status"] == "interrupted"),
            "failed":             sum(1 for e in exps if e["status"] == "failed"),
            "total_checkpoints":  len(ckpts),
            "checkpoints_size_mb": round(total_size_mb, 2),
            "active_subprocess":  proc_running,
        }

    @app.get("/api/hf/search", tags=["Configs"],
             summary="Search the HuggingFace Hub for datasets")
    async def hf_search(
        q: Optional[str] = None,
        modality: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 24,
    ):
        """
        Proxy a HuggingFace Hub dataset search.

        - **q**: free-text query (matches dataset id and description)
        - **modality**: image / text / audio / video / time_series / tabular / etc.
                        Maps to HF Hub's `modality:<x>` filter.
        - **sort**: downloads (default) | likes | trending | updated
        - **limit**: 1–100 results

        Returns a list of dataset summaries: `id`, `downloads`, `likes`,
        `tags`, `description`, plus a derived `modality` we infer from tags.
        """
        import httpx
        params: Dict[str, Any] = {"limit": min(max(int(limit), 1), 100), "sort": sort}
        if q: params["search"] = q
        # Combine HF tag filters using their & convention
        filters: List[str] = []
        if modality:
            # HF Hub uses 'modality:' prefix and 'image-classification' style
            mod = modality.strip().lower()
            if mod == "time_series":
                mod = "time-series"
            filters.append(f"modality:{mod}")
        if filters:
            params["filter"] = ",".join(filters)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get("https://huggingface.co/api/datasets", params=params)
                r.raise_for_status()
                rows = r.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, f"HuggingFace Hub error: {e}")
        except Exception as e:
            raise HTTPException(503, f"Could not reach HuggingFace Hub: {e}")

        return [_summarize_hf_search_row(row) for row in rows if isinstance(row, dict)]

    @app.get("/api/hf/featured", tags=["Configs"],
             summary="Curated 'getting started' dataset list")
    async def hf_featured(modality: Optional[str] = None):
        """A short, hand-picked list of well-behaved datasets for each modality —
        the same shortcuts the Builder shows when the user opens the browser
        for the first time. No network calls."""
        catalog = {
            "image": [
                {"id": "cifar10",         "description": "60K 32×32 color images, 10 classes."},
                {"id": "mnist",           "description": "Hand-written digits, 28×28 grayscale, 10 classes."},
                {"id": "fashion_mnist",   "description": "Fashion item classification — drop-in replacement for MNIST."},
                {"id": "cifar100",        "description": "60K 32×32 color images, 100 classes."},
                {"id": "huggan/flowers-102-categories", "description": "Oxford Flowers, 102 species."},
            ],
            "text": [
                {"id": "imdb",            "description": "Binary movie sentiment (50K reviews)."},
                {"id": "ag_news",         "description": "News topic classification, 4 classes."},
                {"id": "sst2",            "description": "Stanford Sentiment Treebank, binary."},
                {"id": "rotten_tomatoes", "description": "Movie review sentiment, binary."},
                {"id": "tweet_eval",      "description": "Twitter classification, multiple sub-tasks."},
            ],
            "audio": [
                {"id": "speech_commands", "description": "1-second spoken keyword clips, 35 classes."},
                {"id": "common_voice",    "description": "Mozilla Common Voice — speech recognition."},
                {"id": "superb",          "description": "Speech understanding benchmark."},
            ],
            "video": [
                {"id": "ucf101",          "description": "Action recognition, 101 classes."},
                {"id": "kinetics-700-2020", "description": "Larger action recognition benchmark."},
            ],
            "time_series": [
                {"id": "monash/m3",       "description": "M3 forecasting competition."},
                {"id": "monash_tsf",      "description": "Monash time-series forecasting archive."},
            ],
            "tabular": [
                {"id": "adult",           "description": "Census income — classic tabular benchmark."},
                {"id": "covertype",       "description": "Forest cover type, 7-class."},
            ],
        }
        if modality:
            return catalog.get(modality.lower(), [])
        # Flatten with modality tag
        flat = []
        for mod, rows in catalog.items():
            for row in rows:
                flat.append({**row, "modality": mod})
        return flat

    @app.get("/api/hf/preview", tags=["Configs"],
             summary="First N rows of a HuggingFace dataset",
             responses={404: {"description": "Dataset not found / gated / failed to load."}})
    async def hf_preview(name: str, split: str = "train", n: int = 5,
                         config: Optional[str] = None):
        """
        Cheap preview: load just the first `n` rows of a dataset's split.
        Pass `?config=` for datasets that require a sub-configuration.
        Big binary fields (Image/Audio/Video) are summarized as their type
        rather than encoded — enough to show shape, not enough to ship MB.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise HTTPException(503, "The `datasets` package is not installed on the server.")
        try:
            kwargs: Dict[str, Any] = {"split": split, "streaming": True}
            args = (name,) if not config else (name, config)
            ds = load_dataset(*args, **kwargs)
            rows = []
            for i, row in enumerate(ds):
                if i >= max(1, min(int(n), 10)):
                    break
                rows.append(_summarize_preview_row(row))
        except Exception as exc:
            available = _parse_available_configs(str(exc))
            if available:
                return {"name": name, "needs_config": True,
                        "available_configs": available, "rows": []}
            raise HTTPException(404, f"Could not preview '{name}' split={split}: {exc}")
        return {"name": name, "config": config, "split": split, "rows": rows}

    @app.get("/api/hf/inspect", tags=["Configs"],
             summary="Inspect a HuggingFace dataset's features without downloading data",
             responses={404: {"description": "Dataset not found or private/gated."}})
    async def hf_inspect(name: str, config: Optional[str] = None):
        """
        Surface a HuggingFace dataset's columns and detected modality so the
        Builder UI can pre-fill the right text/label/image columns without the
        user guessing or downloading 5 GB of images.

        Some datasets (e.g. `superb`, `glue`, `super_glue`) require picking a
        sub-configuration. Pass `?config=<name>` to drill in. When config is
        required and not provided, this endpoint returns HTTP 200 with
        `needs_config=true` and the list of `available_configs` instead of
        throwing — that way the UI can render a picker.
        """
        try:
            from datasets import load_dataset_builder
        except ImportError:
            raise HTTPException(503, "The `datasets` package is not installed on the server.")
        try:
            builder = load_dataset_builder(name, config) if config else load_dataset_builder(name)
        except Exception as exc:
            # `datasets` raises ValueError("Config name is missing. … available
            # configs: ['asr', 'er', …]") when a multi-config dataset is queried
            # without a config. Detect that and return a structured response
            # so the UI can render config chips instead of an opaque 404.
            available = _parse_available_configs(str(exc))
            if available:
                return {
                    "name": name,
                    "needs_config": True,
                    "available_configs": available,
                    "description": None,
                    "splits": [],
                    "configs": available,
                    "schema": None,
                    "modality": "unknown",
                    "suggested_model": None,
                    "compatible_models": [],
                }
            raise HTTPException(404, f"Could not fetch '{name}': {exc}")
        info = builder.info
        from neural_platform.core.hf_introspect import inspect_features
        from neural_platform.core.modality import (
            detect_from_features, recommend_model, MODALITY_MODELS, EXPERIMENTAL_MODELS,
        )
        summary = inspect_features(info.features) if info.features else {
            "columns": [], "image_columns": [], "text_columns": [], "audio_columns": [],
            "video_columns": [], "sequence_columns": [], "label_columns": [],
            "numeric_columns": [], "other_columns": [], "class_names": None,
            "has_images": False, "has_text": False, "has_audio": False,
            "has_video": False, "has_sequence": False,
        }
        modality = detect_from_features(summary)
        suggested = recommend_model(modality)
        return {
            "name": name,
            "config": config,
            "needs_config": False,
            "description": (info.description or "").strip()[:400] or None,
            "splits": list((info.splits or {}).keys()),
            "configs": [getattr(c, "name", None) for c in (getattr(builder, "BUILDER_CONFIGS", []) or [])] or None,
            "schema": summary,
            "modality": modality.value,
            "suggested_model": suggested,
            "compatible_models": MODALITY_MODELS.get(modality, []),
            "experimental_warning": suggested in EXPERIMENTAL_MODELS,
            "size_bytes": getattr(info, "dataset_size", None),
        }

    @app.get("/api/tasks", tags=["Configs"],
             summary="HF-aligned task taxonomy with suggested architectures")
    async def list_tasks():
        """
        Return the full task catalog grouped by family (Text / Vision / Audio
        / Video / Tabular / Multi-modal / Other) plus per-task metadata
        (inputs, outputs, suggested model_types, requires_pretrained flag).

        The Builder UI uses this to render its first-step Task picker.
        """
        from neural_platform.core.tasks import TASK_CATALOG, grouped_for_ui
        meta = {}
        for task, m in TASK_CATALOG.items():
            meta[task.value] = {
                "task": task.value,
                "inputs": m.inputs,
                "outputs": m.outputs,
                "modality": m.modality,
                "suggested_models": m.suggested_models,
                "multimodal": m.multimodal,
                "generative": m.generative,
                "requires_pretrained": m.requires_pretrained,
            }
        return {"groups": grouped_for_ui(), "meta": meta}

    @app.get("/api/deps", tags=["System"],
             summary="Modality dependency audit — what's installed vs missing")
    async def deps_audit(model: Optional[str] = None, source: Optional[str] = None):
        """
        Probe Python package availability for each model type and data source
        the platform supports. The dashboard's Settings page renders this so
        users can spot a missing torchaudio/torchvision/transformers without
        running `neural deps` in a terminal.
        """
        from neural_platform.core.deps import check_dependencies, check_all, install_command
        if model or source:
            rep = check_dependencies(model or "mlp", source)
            return {
                "scope": {"model": model, "source": source},
                "ok": rep.ok,
                "statuses": [vars(s) for s in rep.statuses],
                "missing_required": [d.package for d in rep.missing_required],
                "missing_optional": [d.package for d in rep.missing_optional],
                "install_command": install_command(rep),
            }
        out: Dict[str, Any] = {}
        for key, rep in check_all().items():
            out[key] = {
                "ok": rep.ok,
                "missing_required": [d.package for d in rep.missing_required],
                "missing_optional": [d.package for d in rep.missing_optional],
                "statuses": [vars(s) for s in rep.statuses],
            }
        return out

    @app.get("/api/system", tags=["System"], summary="Host CPU/RAM/GPU/disk telemetry")
    async def system():
        return _system_snapshot()

    @app.get("/api/health", tags=["System"], summary="Dashboard liveness")
    async def health():
        return {
            "ok": True,
            "version": "0.2.0",
            "uptime": round(time.time() - state.get("dashboard_started_at", time.time()), 1),
            "output_dir": output_dir,
            "db_exists": db_path.exists(),
        }

    # ------------------------------------------------------------------
    # Config discovery + persistence
    # ------------------------------------------------------------------

    @app.get("/api/configs", tags=["Configs"],
             summary="List all config.yaml files under the output dir")
    async def list_configs():
        """Scan output_dir and return metadata for every config.yaml found."""
        return _scan_configs(output_dir)

    @app.get("/api/configs/load", tags=["Configs"], summary="Load a config as JSON")
    async def load_config(path: str):
        p = Path(path)
        if not p.exists():
            raise HTTPException(404, "Config not found")
        try:
            import yaml
            return yaml.safe_load(p.read_text()) or {}
        except Exception as exc:
            raise HTTPException(500, f"Failed to parse config: {exc}")

    @app.post("/api/configs/validate", tags=["Configs"], summary="Pre-flight validate a config")
    async def validate_config_endpoint(req: SaveConfigRequest):
        """
        Run the same validator the CLI uses on a posted config dict — without
        writing anything to disk. Returns a list of error/warning issues.
        """
        try:
            from neural_platform.core.config import ExperimentConfig
            from neural_platform.core.validator import validate_config
            cfg = ExperimentConfig.model_validate(req.config)
        except Exception as exc:
            return {"ok": False, "schema_error": str(exc),
                    "issues": [{"severity": "error", "field": "schema", "message": str(exc)}]}
        report = validate_config(cfg)
        return report.to_dict()

    @app.post("/api/configs/save", tags=["Configs"], summary="Persist a config to disk")
    async def save_config(req: SaveConfigRequest):
        """Persist a config dict to <output_dir>/<name>/config.yaml.

        The dashboard's visual builder posts here — the resulting file is the
        same shape as `neural init` would produce, so it shows up immediately
        in the launcher dropdown and works with `neural train`.
        """
        try:
            from neural_platform.core.config import ExperimentConfig
            ExperimentConfig.model_validate(req.config)
        except Exception as exc:
            raise HTTPException(422, f"Invalid config: {exc}")

        run_dir = Path(req.output_dir or output_dir) / req.name
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists() and not req.overwrite:
            raise HTTPException(409, f"Config already exists at {cfg_path}. Pass overwrite=true to replace.")

        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
            with open(cfg_path, "w") as f:
                yaml.dump(req.config, f, default_flow_style=False, sort_keys=False)
        except Exception as exc:
            raise HTTPException(500, f"Failed to write config: {exc}")

        return {"saved": True, "path": str(cfg_path)}

    # ------------------------------------------------------------------
    # Training subprocess management
    # ------------------------------------------------------------------

    def _detect_crash_and_emit() -> None:
        """
        If the subprocess died but never wrote a `training_end` event, append a
        synthetic one so the SSE stream + Live UI stop spinning. Also clears
        the stored Popen reference.
        """
        proc: Optional[subprocess.Popen] = state.get("train_proc")
        if proc is None:
            return
        rc = proc.poll()
        if rc is None:
            return  # still running

        # Check whether a training_end already exists in the events file
        ev_path: Path = state["events_path"]
        ended = False
        if ev_path.exists():
            try:
                last_line = None
                for line in ev_path.read_text().splitlines():
                    if line.strip():
                        last_line = line
                if last_line:
                    last = json.loads(last_line)
                    ended = last.get("type") in ("training_end", "early_stop")
            except Exception:
                pass

        if not ended:
            progress = _scan_last_progress(ev_path)
            event = {
                "type": "training_end",
                "ts": time.time(),
                "experiment": state.get("current_experiment", "unknown"),
                "status": "completed" if rc == 0 else "failed",
                "best_epoch": progress.get("best_epoch", 0) or 0,
                "best_val_loss": progress.get("best_val_loss"),
                "total_epochs": progress.get("total_epochs", 0) or 0,
                "duration": progress.get("duration", 0.0) or 0.0,
                "exit_code": rc,
            }
            try:
                with open(ev_path, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception:
                pass
            # Mirror into SQLite so the Experiments table reflects reality
            try:
                tracker().interrupt_stale_runs()
            except Exception:
                pass

        # Clear the Popen — subprocess is gone — and let the PTY bridge
        # drain so we don't leak a fd / orphan the worker thread.
        state["train_proc"] = None
        _shutdown_pty_bridge(state)

    def _proc_status() -> dict:
        proc: Optional[subprocess.Popen] = state["train_proc"]
        if proc is None:
            return {"running": False, "pid": None, "returncode": None,
                    "experiment": None}
        rc = proc.poll()
        if rc is not None:
            # Side-effect: turn the dead process into a synthetic event so
            # downstream watchers (SSE, Live UI) know to stop.
            _detect_crash_and_emit()
            return {"running": False, "pid": proc.pid, "returncode": rc,
                    "experiment": state.get("current_experiment"),
                    "exit_code": rc}
        return {"running": True, "pid": proc.pid, "returncode": None,
                "experiment": state.get("current_experiment")}

    @app.get("/api/train/status", tags=["Training"],
             summary="Status of the training subprocess")
    async def train_status():
        return _proc_status()

    @app.post("/api/train/start", tags=["Training"],
              summary="Launch a training subprocess",
              responses={
                  409: {"description": "Another training is already in progress."},
                  404: {"description": "Config file not found."},
                  422: {"description": "Config failed pre-flight validation."},
              })
    async def train_start(req: TrainStartRequest):
        proc: Optional[subprocess.Popen] = state["train_proc"]
        if proc is not None and proc.poll() is None:
            raise HTTPException(409, "A training run is already in progress — stop it first.")

        cfg_path = Path(req.config_path)
        if not cfg_path.exists():
            raise HTTPException(404, f"Config not found: {req.config_path}")

        # Pre-flight validate so we never spawn a subprocess that will explode
        # on the first batch with a confusing traceback.
        try:
            from neural_platform.core.config import ExperimentConfig
            from neural_platform.core.validator import validate_config
            import yaml as _yaml
            with open(cfg_path) as f:
                cfg_data = _yaml.safe_load(f) or {}
            cfg_obj = ExperimentConfig.model_validate(cfg_data)
            report = validate_config(cfg_obj)
            if not report.ok:
                raise HTTPException(
                    422,
                    {
                        "message": "Config failed validation — fix errors and retry.",
                        "issues": report.to_dict()["issues"],
                    },
                )
            state["current_experiment"] = cfg_data.get("name", cfg_path.parent.name)
        except HTTPException:
            raise
        except Exception:
            state["current_experiment"] = cfg_path.parent.name

        # Stop the previous bridge thread and close the log file (if any).
        bridge_stop = state.get("train_bridge_stop")
        if bridge_stop is not None:
            try: bridge_stop.set()
            except Exception: pass
        bridge_thread = state.get("train_bridge_thread")
        if bridge_thread is not None:
            try: bridge_thread.join(timeout=1.0)
            except Exception: pass
        if state["train_log_fh"] is not None:
            try: state["train_log_fh"].close()
            except Exception: pass

        log_path: Path = state["train_log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = _find_neural_cmd() + ["train", "--config", str(cfg_path)]
        for ov in req.overrides:
            cmd += ["--override", ov]

        # Force unbuffered stdio so tqdm/print show up live, and disable HF
        # tqdm's "leave only final" non-TTY behaviour by giving it a real PTY.
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        try:
            new_proc, log_fh, bridge_thread, bridge_stop = _spawn_with_pty_log(
                cmd, env, log_path,
            )
        except Exception as exc:
            raise HTTPException(500, f"Failed to start training: {exc}") from exc

        state["train_log_fh"]      = log_fh
        state["train_bridge_thread"] = bridge_thread
        state["train_bridge_stop"]   = bridge_stop

        state["train_proc"] = new_proc
        return {
            "started": True,
            "pid": new_proc.pid,
            "cmd": " ".join(cmd),
            "log": str(log_path),
            "experiment": state["current_experiment"],
        }

    @app.post("/api/train/stop", tags=["Training"], summary="Stop the active training subprocess")
    async def train_stop():
        proc: Optional[subprocess.Popen] = state["train_proc"]
        if proc is None or proc.poll() is not None:
            state["train_proc"] = None
            _shutdown_pty_bridge(state)
            return {"stopped": False, "reason": "No active training process"}

        pid = proc.pid
        await asyncio.to_thread(_kill_process_group, proc)
        state["train_proc"] = None  # actually clear

        # Tear down the PTY bridge so the master fd is closed and the worker
        # thread exits cleanly. This is what eliminates the "leaked semaphore"
        # warning the user saw — the bridge thread holds an open fd to the
        # master PTY and the resource tracker complains if it isn't released.
        _shutdown_pty_bridge(state)

        # Compose a synthetic training_end with whatever progress we observed
        last_progress = _scan_last_progress(state["events_path"])
        _append_training_end_event(
            state["events_path"],
            state.get("current_experiment", "unknown"),
            last_progress,
        )

        try:
            tracker().interrupt_stale_runs()
        except Exception:
            pass

        return {"stopped": True, "pid": pid, "progress": last_progress}

    @app.post("/api/train/cleanup", tags=["Training"],
              summary="Mark stale 'running' rows as 'interrupted'")
    async def train_cleanup():
        try:
            n = tracker().interrupt_stale_runs()
            return {"cleaned": True, "rows_updated": n}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.post("/api/experiments/{exp_id}/interrupt", tags=["Experiments"],
              summary="Mark a single experiment + its running runs as interrupted")
    async def interrupt_experiment(exp_id: int):
        exp = tracker().get_experiment(exp_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")
        tracker().update_experiment_status(exp_id, "interrupted")
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        runs = tracker().list_runs(exp_id)
        for run in runs:
            if run["status"] == "running":
                tracker().finish_run(
                    run["id"],
                    status="interrupted",
                    best_val_loss=run.get("best_val_loss"),
                    best_epoch=run.get("best_epoch"),
                    total_epochs=run.get("total_epochs") or 0,
                    checkpoint_path=run.get("checkpoint_path"),
                    started_at=time.time(),
                )
        return {"interrupted": True, "experiment_id": exp_id}

    @app.get("/api/train/logs", tags=["Training"],
             summary="Tail of subprocess stdout/stderr")
    async def train_logs(lines: int = 200, raw: bool = False):
        """Tail of the subprocess stdout/stderr.

        By default the response strips ANSI color sequences and collapses
        consecutive tqdm-style progress lines (sharing an identifying
        prefix) so a 5-minute download isn't 4000 wrapped lines in the UI.
        Pass `?raw=true` to opt out and see exactly what was written.
        """
        log_path: Path = state["train_log_path"]
        if not log_path.exists():
            return {"lines": []}
        with open(log_path) as f:
            all_lines = f.readlines()
        out_lines = [ln.rstrip("\n") for ln in all_lines[-lines:]]
        if not raw:
            out_lines = [_strip_ansi(ln) for ln in out_lines]
            out_lines = _collapse_progress(out_lines)
        return {"lines": out_lines}

    # ------------------------------------------------------------------
    # Inference proxy
    # ------------------------------------------------------------------

    @app.get("/api/proxy/health", tags=["Inference"],
             summary="Proxy health check to a remote inference server")
    async def proxy_health(server_url: str = "http://localhost:8080"):
        import httpx
        base = _validated_proxy_base_url(server_url)
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{base}/health")
                return r.json()
        except Exception as e:
            raise HTTPException(503, f"Inference server unreachable: {e}")

    @app.get("/api/proxy/info", tags=["Inference"],
             summary="Proxy /info from a remote inference server")
    async def proxy_info(server_url: str = "http://localhost:8080"):
        import httpx
        base = _validated_proxy_base_url(server_url)
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{base}/info")
                r.raise_for_status()
                data = r.json()
                # Normalize key names so the frontend can rely on a stable shape
                return {
                    "model_name":      data.get("model_name") or data.get("name"),
                    "model_type":      data.get("model_type"),
                    "framework":       data.get("framework"),
                    "parameter_count": data.get("parameter_count") or data.get("parameters"),
                    "trainable_parameters": data.get("trainable_parameters"),
                    "checkpoint_path": data.get("checkpoint_path"),
                    "device":          data.get("device"),
                    "class_names":     data.get("class_names"),
                    "output_size":     data.get("output_size"),
                    "epoch":           data.get("epoch"),
                    "val_loss":        data.get("val_loss"),
                    "raw":             data,
                }
        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, str(e))
        except Exception as e:
            raise HTTPException(503, f"Inference server unreachable: {e}")

    @app.post("/api/proxy/predict", tags=["Inference"],
              summary="Run a prediction via a remote inference server (CORS-safe)")
    async def proxy_predict(req: ProxyPredictRequest):
        import httpx
        base = _validated_proxy_base_url(req.server_url)
        payload: Dict[str, Any] = {
            "top_k": req.top_k,
            "return_probabilities": req.return_probabilities,
        }
        if req.inputs         is not None: payload["inputs"]         = req.inputs
        if req.tokens         is not None: payload["tokens"]         = req.tokens
        if req.attention_mask is not None: payload["attention_mask"] = req.attention_mask
        if req.image_b64      is not None: payload["image_b64"]      = req.image_b64
        if req.text           is not None: payload["text"]           = req.text

        url = f"{base}/predict"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                t0 = time.time()
                r = await client.post(url, json=payload)
                wall_ms = (time.time() - t0) * 1000
                r.raise_for_status()
                raw = r.json()
                normalized = _normalize_predict_response(raw)
                normalized["wall_latency_ms"] = round(wall_ms, 1)
                return normalized
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            raise HTTPException(e.response.status_code, detail)
        except Exception as e:
            raise HTTPException(503, f"Inference server error: {e}")

    # ------------------------------------------------------------------
    # Live training SSE
    # ------------------------------------------------------------------

    @app.get("/api/training/live", tags=["Live"],
             summary="Snapshot of all events in the current live_events.jsonl")
    async def live_snapshot():
        reader = TrainingEventReader(state["events_path"])
        events = await reader.snapshot()
        is_running = bool(events) and events[-1]["type"] not in ("training_end", "early_stop")
        # Cross-check with the actual subprocess: if the subprocess is gone but
        # the last event isn't training_end (crash), report not-running so the
        # UI doesn't spin forever.
        proc: Optional[subprocess.Popen] = state["train_proc"]
        if is_running and (proc is None or proc.poll() is not None):
            is_running = False
        return {"events": events, "is_running": is_running}

    @app.get("/api/events/stream", tags=["Live"],
             summary="Server-sent event stream of training events",
             responses={200: {"description": "text/event-stream of training_start/batch/epoch/checkpoint/early_stop/training_end"}})
    async def events_stream(request: Request):
        async def generator():
            reader = TrainingEventReader(state["events_path"], poll_interval=0.25)
            yield "event: connected\ndata: {}\n\n"
            last_heartbeat = time.time()
            last_event_ts = time.time()
            async for event in reader.tail(from_start=True):
                payload = json.dumps(event)
                yield f"event: {event['type']}\ndata: {payload}\n\n"
                last_event_ts = time.time()
                if event["type"] == "training_end":
                    await asyncio.sleep(0.5)
                    break
                if await request.is_disconnected():
                    break
                # Crash detection: if our managed subprocess died and we
                # haven't seen an event in a while, synthesize training_end.
                proc: Optional[subprocess.Popen] = state.get("train_proc")
                if proc is not None and proc.poll() is not None and (time.time() - last_event_ts) > 1.0:
                    _detect_crash_and_emit()
                    # The next reader.tail iteration will pick up the synthetic event.
                if time.time() - last_heartbeat > 15:
                    last_heartbeat = time.time()
                    yield ": heartbeat\n\n"

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # SPA
    # ------------------------------------------------------------------

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_ui():
        html_file = static_dir / "index.html"
        if html_file.exists():
            return HTMLResponse(html_file.read_text())
        return HTMLResponse("<h1>NeuralForge Dashboard</h1><p>Static files not found.</p>")

    return app
