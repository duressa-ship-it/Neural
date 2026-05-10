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
import codecs
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

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


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
    # Optional bearer token to forward to the inference server. Kept in the
    # request body (not the URL) so it isn't logged in access logs. The
    # dashboard strips it from any payload echoed back and never persists it.
    bearer_token: Optional[str] = Field(
        None, description="Optional bearer for the inference server's /predict endpoint."
    )


class TrainStartRequest(BaseModel):
    config_path: str
    overrides: List[str] = []   # e.g. ["training.lr=0.001", "training.num_epochs=50"]


class StartTrainRunRequest(BaseModel):
    """Body for `POST /api/train/runs/start`.

    Module-scoped (not inside `create_dashboard_app`) so Pydantic / FastAPI
    can resolve the forward ref into the OpenAPI schema. Local classes
    surface as 422 'missing query param req' and 500s on /openapi.json,
    which is the same trap StartInferenceRequest hit earlier.
    """
    config_path: str = Field(..., description="Path to the experiment config YAML")
    overrides:   List[str] = Field(default_factory=list,
                                    description="Override expressions, e.g. ['training.lr=0.001']")
    name:        Optional[str] = Field(None, description="Display name; defaults to the run dir name")


class StartInferenceRequest(BaseModel):
    """Body for `POST /api/inference/start`.

    Defined at module scope (not inside `create_dashboard_app`) so FastAPI's
    OpenAPI generator can resolve the forward ref. Anything declared inside
    the factory function is local-scoped and Pydantic refuses to build the
    schema for it, which manifested as a 422 "missing query param req"
    error and a 500 from `/openapi.json`.
    """
    config_path: str = Field(..., description="Path to the experiment config YAML")
    checkpoint:  Optional[str] = Field(None, description="Optional checkpoint .pt path; defaults to best")
    name:        Optional[str] = Field(None, description="Display name; defaults to the run dir name")


class StartHFInferenceRequest(BaseModel):
    """Body for `POST /api/inference/start_hf`.

    Launches a managed inference server straight from a HuggingFace model id —
    no prior `neural train` run, no checkpoint on disk. The dashboard
    synthesizes a minimal hf_pipeline config and the manager spawns
    ``neural serve --no-checkpoint`` against it.
    """
    hf_model_id:    str = Field(..., description="HuggingFace model id, e.g. 'distilbert-base-uncased-finetuned-sst-2-english'")
    pipeline_task:  str = Field(..., description="HuggingFace pipeline_tag, e.g. 'text-classification', 'image-classification'")
    name:           Optional[str] = Field(None, description="Display name; defaults to hf:<id>")
    revision:       Optional[str] = Field(None, description="Optional model revision / git ref")
    trust_remote_code: bool = Field(False, description="Pass trust_remote_code=True to from_pretrained — required for some custom HF repos")


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
    the master PTY and writes to disk without rewriting control characters.
    This preserves raw terminal semantics (`\r`, ANSI controls, etc.) so the
    renderer can correctly model in-place updates.

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
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while not stop_event.is_set():
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                try:
                    log_fh.write(decoder.decode(chunk, final=False))
                    log_fh.flush()
                except Exception:
                    break
        finally:
            try:
                tail = decoder.decode(b"", final=True)
                if tail:
                    log_fh.write(tail)
                    log_fh.flush()
            except Exception:
                pass
            try: os.close(master_fd)
            except Exception: pass

    thread = _threading.Thread(target=_bridge, daemon=True, name="forge-pty-bridge")
    thread.start()
    return proc, log_fh, thread, stop_event


def _tail_text(path: Path, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(size, max_chars * 4)
            f.seek(max(0, size - read_size), os.SEEK_SET)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text
    except Exception:
        return ""


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


def _scan_configs(output_dir: str, *, include_managed: bool = False) -> list[dict]:
    """Return metadata for every config.yaml found under output_dir.

    The ``_hf_servers/`` subtree holds **synthesized** configs that exist
    only so a managed inference server can spawn ``neural serve
    --no-checkpoint``. They use ``data.source: synthetic`` and have
    ``num_epochs: 1`` for schema reasons, and would fail any real training
    run (e.g. random floats can't satisfy a tokenizer's embedding layer).
    By default we filter them out so they don't appear in the Train tab's
    config picker. Call with ``include_managed=True`` if you actually need
    the full list (e.g. a debug endpoint).
    """
    out_root = Path(output_dir).resolve()
    managed_root = (out_root / "_hf_servers").resolve()
    configs = []
    for cfg_path in sorted(out_root.rglob("config.yaml")):
        if not include_managed:
            try:
                cfg_path.resolve().relative_to(managed_root)
                continue   # under _hf_servers/ — skip
            except ValueError:
                pass       # not under managed root — keep
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
    # Forward result_kind so the Predict UI can pick the right renderer
    # (boxes vs depth vs masks vs token_spans vs default top-K bars).
    # Defaults to "logits" for back-compat with older inference servers.
    out["result_kind"] = raw.get("result_kind", "logits")

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
        # Pass structured-output details (bbox / qa span indices / depth
        # PNG / token offsets) through unchanged so the renderer can
        # dispatch on result_kind.
        "metadata":    p.get("metadata"),
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
        version="0.3.0",
        # Disable the default /docs/redoc; we serve themed versions below
        # at /api/docs and /api/redoc so they live under the same prefix as
        # the rest of the dashboard's API routes.
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
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
    # CORS: restrict to localhost origins by default. The dashboard is a
    # local-first app — wide-open `*` was useful in early dev but is overly
    # permissive in production. Users who actually need cross-origin access
    # can opt in via NEURAL_DASHBOARD_CORS_ORIGINS (comma-separated list).
    cors_origins_env = (os.environ.get("NEURAL_DASHBOARD_CORS_ORIGINS") or "").strip()
    if cors_origins_env:
        cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    else:
        cors_origins = [
            "http://localhost", "http://127.0.0.1", "http://[::1]",
            "http://localhost:8000", "http://127.0.0.1:8000",
            "http://localhost:8080", "http://127.0.0.1:8080",
            "http://localhost:8765", "http://127.0.0.1:8765",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    # Token-redaction exception handler. Any unhandled error path that
    # would otherwise echo the request URL (which may carry `?token=...`)
    # or the raw HTTP error from huggingface_hub gets scrubbed first.
    from fastapi import Request
    from fastapi.responses import JSONResponse
    try:
        from neural_platform.core.hf_auth import redact as _redact_secret
    except Exception:
        _redact_secret = lambda s: s

    @app.exception_handler(Exception)
    async def _safe_error_handler(request: Request, exc: Exception):
        msg = _redact_secret(str(exc) or exc.__class__.__name__)
        # Keep the same status code shape for HTTPException; everything else
        # becomes a 500 with a redacted detail.
        from fastapi import HTTPException as _HTTPException
        if isinstance(exc, _HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": _redact_secret(str(exc.detail))},
            )
        return JSONResponse(status_code=500, content={"detail": msg})

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
    output_root = Path(output_dir).resolve()

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

    def _validated_config_path(raw_path: str) -> Path:
        """Resolve and validate an incoming config path.

        Accepts three shapes the UI / API can hand us:
          1. an absolute path (already resolved)
          2. a path relative to the current working directory — this is what
             `_scan_configs` emits today (e.g. "runs/exp/config.yaml" when
             the dashboard was launched from the project root)
          3. a path relative to `output_root` — happens when callers pass
             just "exp/config.yaml" without the runs/ prefix

        Previous bug: only path-shape (3) was handled, so a (2)-shaped path
        like "runs/exp/config.yaml" got `output_root` prepended, producing
        "<cwd>/runs/runs/exp/config.yaml" which doesn't exist → 404. The
        UI dropdown sends (2)-shaped paths, hence the inference-launch 404.
        """
        if not raw_path or not raw_path.strip():
            raise HTTPException(400, "path is required")
        raw = raw_path.strip()
        candidate = Path(raw)
        if candidate.is_absolute():
            candidate = candidate.resolve()
        else:
            # Try cwd-relative first (what _scan_configs emits today);
            # fall back to output_root-relative.
            cwd_path = (Path.cwd() / candidate).resolve()
            out_path = (output_root / candidate).resolve()
            if cwd_path.exists() and cwd_path.is_file():
                candidate = cwd_path
            elif out_path.exists() and out_path.is_file():
                candidate = out_path
            else:
                candidate = cwd_path  # use cwd_path for the 404 message
        try:
            candidate.relative_to(output_root)
        except ValueError:
            raise HTTPException(
                400,
                f"path must be inside the configured output directory ({output_root})",
            )
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(404, f"Config not found: {candidate}")
        return candidate

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

    # ------------------------------------------------------------------
    # Pluggable model source (HF first, local checkpoints, future ONNX hub)
    # ------------------------------------------------------------------

    @app.get("/api/models/sources", tags=["Models"],
             summary="List registered model sources")
    async def list_model_sources():
        """Surface every registered `ModelSource` so the Builder can render
        a source tab strip (HF / Local / ONNX / etc.). The list is dynamic —
        installing a plugin that registers a new source makes it appear here
        without code changes elsewhere."""
        from neural_platform.core.model_source import registered_sources
        return [{"name": s.name} for s in registered_sources()]

    @app.get("/api/models/search", tags=["Models"],
             summary="Search a model source for compatible models")
    async def models_search(
        source: str = "huggingface",
        q: Optional[str] = None,
        task: Optional[str] = None,
        modality: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 24,
    ):
        """Search the chosen model source (HF Hub by default).

        - **source**: which `ModelSource` to query (e.g. `huggingface`, `local`)
        - **q**: free-text query
        - **task**: HF pipeline_tag, e.g. `audio-classification`
        - **modality**: image / text / audio / video / time_series / tabular
        - **sort**: `downloads` (default) | `likes` | `trending` | `updated`
        - **limit**: 1–100

        Returns `[{id, source, pipeline_tag, modality, downloads, likes, tags,
        description, library, gated, private}]`. The Builder uses this to
        populate the model dropdown; each entry is small (~200 B) so a
        page of 24 is < 5 KB."""
        try:
            from neural_platform.core.model_source import get_source
            src = get_source(source)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        try:
            cards = src.search(query=q, task=task, modality=modality,
                                sort=sort, limit=limit)
        except Exception as exc:
            raise HTTPException(503, f"Source '{source}' search failed: {exc}")
        return [c.to_dict() for c in cards]

    @app.get("/api/models/inspect", tags=["Models"],
             summary="Inspect compatibility of a model with a task and dataset")
    async def models_inspect(
        source: str = "huggingface",
        id: str = "",
        task: Optional[str] = None,
        dataset_modality: Optional[str] = None,
        dataset: Optional[str] = None,
    ):
        """Run the model inspector. This is the endpoint the Builder calls
        every time the user changes the model id or task — surfacing the
        Whisper-vs-IMDB error in the UI without a training run.

        Pass `dataset=<hf_id>` (with `source=huggingface`) and we'll
        auto-detect the dataset's modality. Or pass `dataset_modality`
        directly to skip the network call.
        """
        if not id:
            raise HTTPException(422, "Model id is required.")
        try:
            from neural_platform.core.model_source import get_source
            src = get_source(source)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

        # Auto-detect dataset modality if a dataset name was passed
        if dataset and not dataset_modality:
            try:
                from datasets import load_dataset_builder
                from neural_platform.core.hf_introspect import inspect_features
                from neural_platform.core.modality import detect_from_features
                builder = load_dataset_builder(dataset)
                schema = inspect_features(getattr(builder.info, "features", None))
                dataset_modality = detect_from_features(schema).value
            except Exception:
                dataset_modality = None

        try:
            report = src.inspect_compat(
                id, intended_task=task, dataset_modality=dataset_modality
            )
        except Exception as exc:
            raise HTTPException(503, f"Inspect failed: {exc}")
        return report.to_dict()

    @app.get("/api/models/fit", tags=["Models"],
             summary="Predict resource fit (RAM/VRAM/disk) for a model + dataset")
    async def models_fit(
        source: str = "huggingface",
        id: str = "",
        dataset: Optional[str] = None,
        purpose: str = "training",
        device: str = "auto",
        optimizer: str = "adamw",
    ):
        """Estimate the RAM/VRAM/disk a model + dataset will consume and
        compare against the host's available resources.

        Returns `{purpose, fits, host: {ram, vram, disk, gpu}, estimate:
        {bytes by category}, issues: [{severity, code, message}]}`. Use
        `purpose=inference` for serving, `training` (default) for training.
        """
        if not id:
            raise HTTPException(422, "Model id is required.")
        try:
            from neural_platform.core.model_source import (
                get_source, validate_hf_model_id, InvalidModelIdError,
                is_standard_loadable,
            )
            from neural_platform.core.resource_fit import (
                snapshot_host, estimate_model_footprint,
                add_dataset_footprint, check_fit,
            )
            if source == "huggingface":
                # Reject obviously-malformed ids before hitting the Hub.
                try:
                    id = validate_hf_model_id(id)
                except InvalidModelIdError as exc:
                    raise HTTPException(422, str(exc))
            info = get_source(source).get_info(id)
        except HTTPException:
            raise
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except Exception as exc:
            raise HTTPException(503, f"Could not fetch model info: {exc}")

        dataset_size = None
        if dataset:
            try:
                from datasets import load_dataset_builder
                builder = load_dataset_builder(dataset)
                dataset_size = (
                    getattr(builder.info, "dataset_size", None)
                    or getattr(builder.info, "download_size", None)
                )
            except Exception:
                dataset_size = None

        host = snapshot_host()
        est = estimate_model_footprint(
            parameters=info.parameters,
            size_bytes=info.size_bytes,
            purpose=purpose,
            optimizer=optimizer,
        )
        est = add_dataset_footprint(est, dataset_size)
        report = check_fit(est, host, purpose=purpose, device=device)
        result = report.to_dict()

        # Don't claim "fits" when the loader can't even consume this format
        # (GGUF, PEFT-only, diffusers, ONNX, TF/Flax). The inspector flags
        # these as errors but the resource fit alone would happily say
        # "0 bytes ≤ host budget = fits". Override here so the UI's "fits"
        # banner reflects reality.
        pattern = info.loading_pattern or "unknown"
        if not is_standard_loadable(pattern) and pattern != "unknown":
            result["fits"] = False
            result.setdefault("issues", []).insert(0, {
                "severity": "error",
                "code": "format_unsupported",
                "message": (
                    f"Model is packaged as `{pattern}`, which the "
                    "transformers loader doesn't support — resource estimate "
                    "is meaningless until the format is changed."
                ),
                "hint": "Pick a sibling repo with `model.safetensors` / "
                         "`pytorch_model.bin`, or use a runtime that supports "
                         f"{pattern} for inference only.",
            })
        # When the source returned NO parameter / size info at all, mark
        # the fit as unknown rather than 'fits=true'. GGUF + a few private
        # repos hit this — the previous behavior gave a misleading green
        # check that vanished at training time.
        if not info.parameters and not info.size_bytes:
            result["fits"] = False
            result["estimate_known"] = False
            result.setdefault("issues", []).insert(0, {
                "severity": "warning",
                "code": "estimate_unknown",
                "message": (
                    "Couldn't determine the model's parameter count or weight "
                    "size from the Hub metadata — resource fit can't be "
                    "checked. The training subprocess may run out of memory."
                ),
                "hint": "Open the model card on the Hub and verify it lists "
                         "`safetensors` total parameters or a model size.",
            })
        else:
            result["estimate_known"] = True
        return result

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

    # ------------------------------------------------------------------
    # Inference server lifecycle (start / list / stop / proxy)
    # ------------------------------------------------------------------

    from neural_platform.web.inference_manager import InferenceServerManager
    inference_mgr = InferenceServerManager(output_dir=output_dir)
    state["inference_mgr"] = inference_mgr

    # ------------------------------------------------------------------
    # Training run manager (multiple concurrent training subprocesses)
    # ------------------------------------------------------------------
    from neural_platform.web.training_manager import TrainingRunManager
    training_mgr = TrainingRunManager(
        output_dir=output_dir,
        pty_spawner=_spawn_with_pty_log,
        neural_cmd=_find_neural_cmd,
    )
    state["training_mgr"] = training_mgr

    @app.get("/api/train/runs", tags=["Training"],
             summary="List managed training runs (running + recent)")
    async def list_training_runs():
        return [r.to_dict() for r in training_mgr.list()]

    @app.post("/api/train/runs/start", tags=["Training"],
              summary="Spawn a new training run (concurrent — does NOT stop existing)")
    async def start_training_run(req: StartTrainRunRequest):
        cfg_path = _validated_config_path(req.config_path)
        # Pre-flight validate so we never spawn a subprocess that will explode.
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
        except HTTPException:
            raise
        except Exception:
            cfg_data = {}
        exp_name = req.name or cfg_data.get("name") or cfg_path.parent.name
        try:
            info = training_mgr.start(
                config_path=str(cfg_path),
                overrides=list(req.overrides or []),
                experiment_name=exp_name,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(500, f"Failed to start training: {exc}")
        return info.to_dict()

    @app.get("/api/train/runs/{run_id}", tags=["Training"],
             summary="Status of one training run")
    async def get_training_run(run_id: str):
        info = training_mgr.get(run_id)
        if not info:
            raise HTTPException(404, f"No training run '{run_id}'.")
        return info.to_dict()

    @app.post("/api/train/runs/{run_id}/stop", tags=["Training"],
              summary="Terminate one training run by id")
    async def stop_training_run(run_id: str):
        if not training_mgr.stop(run_id):
            raise HTTPException(404, f"No training run '{run_id}'.")
        return {"stopped": True, "id": run_id}

    @app.post("/api/train/runs/{run_id}/forget", tags=["Training"],
              summary="Stop and remove a run from the registry")
    async def forget_training_run(run_id: str):
        if not training_mgr.remove(run_id):
            raise HTTPException(404, f"No training run '{run_id}'.")
        return {"forgotten": True, "id": run_id}

    @app.get("/api/train/runs/{run_id}/logs", tags=["Training"],
             summary="Tail of one run's stdout/stderr log")
    async def training_run_logs(run_id: str, chars: int = 200000):
        info = training_mgr.get(run_id)
        if not info:
            raise HTTPException(404, f"No training run '{run_id}'.")
        log_path = Path(info.log_path)
        if not log_path.exists():
            return {"text": "", "log_path": str(log_path)}
        return {
            "text": _tail_text(log_path, max_chars=min(max(int(chars), 1), 1_000_000)),
            "log_path": str(log_path),
        }

    @app.get("/api/train/runs/{run_id}/logs/stream", tags=["Training"],
             summary="SSE stream of one run's raw log chunks")
    async def training_run_logs_stream(run_id: str, request: Request):
        info = training_mgr.get(run_id)
        if not info:
            raise HTTPException(404, f"No training run '{run_id}'.")
        log_path = Path(info.log_path)

        async def generator():
            pos = 0
            last_heartbeat = time.time()
            yield "event: connected\ndata: {}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    if log_path.exists():
                        size = log_path.stat().st_size
                        if size < pos:
                            pos = 0
                        if size > pos:
                            with open(log_path, "rb") as f:
                                f.seek(pos, os.SEEK_SET)
                                raw = f.read(size - pos)
                            pos = size
                            chunk = raw.decode("utf-8", errors="replace")
                            if chunk:
                                payload = json.dumps({"chunk": chunk, "run_id": run_id})
                                yield f"event: chunk\ndata: {payload}\n\n"
                except Exception:
                    pass
                if time.time() - last_heartbeat > 15:
                    last_heartbeat = time.time()
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.2)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/train/runs/{run_id}/events", tags=["Training"],
             summary="Snapshot of one run's training events (live_events.jsonl)")
    async def training_run_events(run_id: str):
        info = training_mgr.get(run_id)
        if not info:
            raise HTTPException(404, f"No training run '{run_id}'.")
        ev_path = Path(info.events_path)
        if not ev_path.exists():
            return {"events": [], "is_running": info.status == "running"}
        events = []
        try:
            for line in ev_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            pass
        is_running = info.status == "running"
        return {"events": events, "is_running": is_running, "run": info.to_dict()}

    @app.get("/api/train/runs/{run_id}/events/stream", tags=["Training"],
             summary="Per-run SSE stream of training events")
    async def training_run_events_stream(run_id: str, request: Request):
        info = training_mgr.get(run_id)
        if not info:
            raise HTTPException(404, f"No training run '{run_id}'.")
        ev_path = Path(info.events_path)

        async def generator():
            from neural_platform.core.event_bus import TrainingEventReader
            reader = TrainingEventReader(ev_path, poll_interval=0.25)
            yield "event: connected\ndata: {}\n\n"
            async for event in reader.tail(from_start=True):
                payload = json.dumps({**event, "run_id": run_id})
                yield f"event: {event['type']}\ndata: {payload}\n\n"
                if event["type"] == "training_end":
                    await asyncio.sleep(0.5)
                    break
                if await request.is_disconnected():
                    break

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/inference/start", tags=["Inference"],
              summary="Spawn a new inference server from a checkpoint")
    async def inference_start(req: StartInferenceRequest):
        """Launch a `neural serve` subprocess on a free localhost port.

        The manager generates a per-server bearer token, passes it to the
        subprocess via `NEURAL_INFERENCE_TOKEN`, and **never returns it** —
        clients call `/api/inference/{id}/predict` to talk to the server,
        and the manager attaches the bearer header internally."""
        try:
            cfg_path = _validated_config_path(req.config_path)
            info = inference_mgr.start_from_config(
                config_path=str(cfg_path),
                checkpoint=req.checkpoint,
                name=req.name,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(500, f"Could not start inference server: {exc}")
        return info.to_dict()

    @app.post("/api/inference/start_hf", tags=["Inference"],
              summary="Spawn a managed inference server from a HuggingFace model id")
    async def inference_start_hf(req: StartHFInferenceRequest):
        """Launch a `neural serve --no-checkpoint` subprocess that wraps a
        HuggingFace pipeline model.

        The dashboard:
          1. Validates the HF model id (rejects malformed shapes / URLs).
          2. Synthesizes a minimal hf_pipeline config under
             ``<output_dir>/_hf_servers/<rand>/config.yaml``.
          3. Spawns the inference subprocess on a free localhost port with
             a freshly generated bearer token (held in-process).
          4. Returns a ``ServerInfo`` whose id flows back into the existing
             managed-server registry — clients use the same
             ``/api/inference/{id}/predict`` proxy as for checkpoint-backed
             servers, so token material never reaches the browser.

        For finer-grained validation (does the HF id agree with the chosen
        task? is the model family standard-loadable? does it fit on the
        host?), run ``GET /api/models/inspect`` and ``GET /api/models/fit``
        first — those are the same checks the Train builder uses, and they
        report actionable issues without spending a subprocess slot.
        """
        try:
            info = inference_mgr.start_from_hf(
                hf_model_id=req.hf_model_id,
                pipeline_task=req.pipeline_task,
                name=req.name,
                revision=req.revision,
                trust_remote_code=req.trust_remote_code,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(500, f"Could not start HF inference server: {exc}")
        return info.to_dict()

    @app.get("/api/inference/list", tags=["Inference"],
             summary="List all managed inference servers")
    async def inference_list():
        return [s.to_dict() for s in inference_mgr.list()]

    @app.get("/api/inference/{server_id}", tags=["Inference"],
             summary="Server status (no token material returned)")
    async def inference_get(server_id: str):
        info = inference_mgr.get(server_id)
        if not info:
            raise HTTPException(404, f"No inference server '{server_id}'.")
        return info.to_dict()

    @app.post("/api/inference/{server_id}/stop", tags=["Inference"],
              summary="Stop and forget a managed inference server")
    async def inference_stop(server_id: str):
        ok = inference_mgr.remove(server_id)
        if not ok:
            raise HTTPException(404, f"No inference server '{server_id}'.")
        return {"stopped": True, "id": server_id}

    @app.post("/api/inference/{server_id}/predict", tags=["Inference"],
              summary="Predict via a managed inference server (token auto-attached)")
    async def inference_predict(server_id: str, body: Dict[str, Any]):
        """Proxy a `/predict` call to the held server, attaching the bearer
        token server-side so it never reaches the browser. The body is the
        same `PredictRequest` shape the inference server accepts.

        The response goes through `_normalize_predict_response` for parity
        with `/api/proxy/predict`, so the frontend sees a flat
        `predictions: [{label, probability, ...}]` list. Without this the
        Predict UI rendered empty bars because the raw NeuralForge
        response is `predictions: [[ ... ]]` (per-sample × top-k).
        """
        try:
            t0 = time.time()
            raw = inference_mgr.proxy(server_id, "/predict",
                                       method="POST", json_body=body)
            wall_ms = (time.time() - t0) * 1000
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except RuntimeError as exc:
            raise HTTPException(502, str(exc))

        normalized = _normalize_predict_response(raw)
        normalized["wall_latency_ms"] = round(wall_ms, 1)
        return normalized

    @app.post("/api/inference/{server_id}/predict/stream", tags=["Inference"],
              summary="Stream tokens from a managed generative server (SSE)")
    async def inference_predict_stream(server_id: str, body: Dict[str, Any], request: Request):
        """Proxy an SSE token stream from a managed inference server.

        The browser POSTs the same PredictRequest body it would send to
        ``/predict``. We open a streaming HTTP connection to the
        inference subprocess (passing its bearer token via header so it
        stays out of the browser), then forward the SSE chunks back.

        Disconnect handling: when the client closes the connection,
        FastAPI closes the response generator → ``httpx`` cancels its
        underlying request → the inference server's ``StreamingResponse``
        unwinds and the generator's ``finally`` clause stops the
        ``TextIteratorStreamer``.
        """
        from fastapi.responses import StreamingResponse as _SSE
        try:
            # proxy_stream returns an async generator directly (not a
            # coroutine) — calling it with `await` would yield a
            # coroutine that ASGI can't iterate.
            agen = inference_mgr.proxy_stream(
                server_id, "/predict/stream", json_body=body,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except RuntimeError as exc:
            raise HTTPException(502, str(exc))

        async def relay():
            async for chunk in agen:
                if await request.is_disconnected():
                    break
                yield chunk

        return _SSE(relay(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })

    @app.get("/api/inference/{server_id}/info", tags=["Inference"],
             summary="Proxy /info from a managed inference server")
    async def inference_info(server_id: str):
        try:
            return inference_mgr.proxy(server_id, "/info", method="GET")
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except RuntimeError as exc:
            raise HTTPException(502, str(exc))

    # ------------------------------------------------------------------
    # Themed Swagger / ReDoc docs at /api/docs (and /api/redoc)
    # ------------------------------------------------------------------

    from fastapi.responses import HTMLResponse

    _DOCS_THEME_CSS = """
      <style>
        :root { color-scheme: dark; }
        html, body { background: #0d0e10; color: #e6e8ea; }
        .swagger-ui, .swagger-ui .topbar { background: #0d0e10 !important; }
        .swagger-ui .info, .swagger-ui .scheme-container { background: #131418 !important; }
        .swagger-ui .info .title, .swagger-ui h1, .swagger-ui h2,
        .swagger-ui h3, .swagger-ui h4, .swagger-ui h5, .swagger-ui .info p,
        .swagger-ui .opblock-tag { color: #e6e8ea !important; }
        .swagger-ui .opblock { background: #131418 !important; border-color: #23262d !important; }
        .swagger-ui .opblock .opblock-summary { background: #1a1d22 !important; }
        .swagger-ui .opblock-description-wrapper p,
        .swagger-ui .opblock-external-docs-wrapper p,
        .swagger-ui .opblock-title_normal p,
        .swagger-ui .response-col_description__inner p,
        .swagger-ui table thead tr th, .swagger-ui table thead tr td,
        .swagger-ui .parameter__name, .swagger-ui .parameter__type,
        .swagger-ui .parameter__in, .swagger-ui .markdown p { color: #c0c4c9 !important; }
        .swagger-ui .btn { background: #232733 !important; color: #e6e8ea !important; border-color: #2c3140 !important; }
        .swagger-ui input[type=text], .swagger-ui textarea, .swagger-ui select,
        .swagger-ui .parameters-col_description input,
        .swagger-ui .body-param__text { background: #0d0e10 !important; color: #e6e8ea !important; border-color: #2c3140 !important; }
        .swagger-ui .topbar-wrapper img { display: none; }
        .swagger-ui .topbar-wrapper::before {
          content: "NeuralForge Dashboard API";
          color: #e6e8ea; font-weight: 600; font-size: 14px; padding: 10px 14px;
        }
      </style>
    """

    @app.get("/api/docs", include_in_schema=False)
    @app.get("/docs", include_in_schema=False)        # backward-compat alias
    async def themed_docs() -> HTMLResponse:
        """Themed Swagger UI for the dashboard API. Mirrors the dark theme."""
        html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><title>NeuralForge — Dashboard API</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"/>
{_DOCS_THEME_CSS}
</head><body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
  window.ui = SwaggerUIBundle({{
    url: '/api/openapi.json',
    dom_id: '#swagger-ui',
    deepLinking: true,
    layout: 'BaseLayout',
    docExpansion: 'list',
    defaultModelsExpandDepth: 0,
  }});
</script>
</body></html>"""
        return HTMLResponse(html)

    @app.get("/api/redoc", include_in_schema=False)
    async def themed_redoc() -> HTMLResponse:
        html = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><title>NeuralForge — Dashboard API (ReDoc)</title>
<style>body { background: #0d0e10; }</style>
</head><body>
<redoc spec-url="/api/openapi.json" theme='{"colors":{"primary":{"main":"#7c87ff"}}}'></redoc>
<script src="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js"></script>
</body></html>"""
        return HTMLResponse(html)

    @app.get("/api/auth/status", tags=["System"],
             summary="HF authentication status (no token material returned)")
    async def auth_status_endpoint():
        """Probe whether the server has a HuggingFace token configured and
        whether it's accepted by the Hub. The response intentionally never
        contains the token itself — only `authenticated`, `source` (e.g.
        `env:HF_TOKEN`), `name`, and `org`.

        The dashboard renders this in Settings so users can see at a glance
        why a gated model didn't load. Tokens are sourced from env vars
        (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`) and the
        `huggingface-cli login` cache (`~/.cache/huggingface/token`)."""
        try:
            from neural_platform.core.hf_auth import auth_status
            status = auth_status()
            return status.to_dict()
        except Exception as exc:
            from neural_platform.core.hf_auth import redact
            return {"authenticated": False, "error": redact(str(exc))}

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
        p = _validated_config_path(path)
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
              summary="Launch a training subprocess (concurrent — does NOT stop existing)",
              responses={
                  404: {"description": "Config file not found."},
                  422: {"description": "Config failed pre-flight validation."},
              })
    async def train_start(req: TrainStartRequest):
        # Multi-run support: a previously-running process is left alone.
        # Use POST /api/train/runs/{id}/stop to kill a specific run, or the
        # legacy /api/train/stop to kill the most-recent one. The 409 from
        # earlier versions has been removed.

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
    async def train_logs(chars: int = 200000):
        """Tail of subprocess stdout/stderr as raw terminal text."""
        log_path: Path = state["train_log_path"]
        if not log_path.exists():
            return {"text": ""}
        return {"text": _tail_text(log_path, max_chars=min(max(int(chars), 1), 1_000_000))}

    @app.get("/api/train/logs/stream", tags=["Training"], summary="SSE stream of raw terminal log chunks")
    async def train_logs_stream(request: Request):
        async def generator():
            log_path: Path = state["train_log_path"]
            pos = 0
            last_heartbeat = time.time()
            yield "event: connected\ndata: {}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    if log_path.exists():
                        size = log_path.stat().st_size
                        if size < pos:
                            pos = 0
                        if size > pos:
                            with open(log_path, "rb") as f:
                                f.seek(pos, os.SEEK_SET)
                                raw = f.read(size - pos)
                            pos = size
                            chunk = raw.decode("utf-8", errors="replace")
                            if chunk:
                                payload = json.dumps({"chunk": chunk})
                                yield f"event: chunk\ndata: {payload}\n\n"
                except Exception:
                    pass
                if time.time() - last_heartbeat > 15:
                    last_heartbeat = time.time()
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.2)

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
    # Inference proxy
    # ------------------------------------------------------------------

    def _proxy_bearer_headers(authorization: Optional[str]) -> Dict[str, str]:
        """Build the outbound header dict for proxy calls.

        We forward the browser's `Authorization` header verbatim so the
        bearer token only travels inside the header (never as a URL param,
        never in our access logs). When no header was sent, return an empty
        dict — the inference server will 401 if it requires auth and the UI
        surfaces a "needs token" hint.
        """
        if authorization and authorization.lower().startswith("bearer "):
            return {"Authorization": authorization}
        return {}

    @app.get("/api/proxy/health", tags=["Inference"],
             summary="Proxy health check to a remote inference server")
    async def proxy_health(server_url: str = "http://localhost:8080",
                            authorization: Optional[str] = Header(default=None)):
        import httpx
        base = _validated_proxy_base_url(server_url)
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{base}/health",
                                      headers=_proxy_bearer_headers(authorization))
                return r.json()
        except Exception as e:
            raise HTTPException(503, f"Inference server unreachable: {e}")

    @app.get("/api/proxy/info", tags=["Inference"],
             summary="Proxy /info from a remote inference server")
    async def proxy_info(server_url: str = "http://localhost:8080",
                          authorization: Optional[str] = Header(default=None)):
        import httpx
        base = _validated_proxy_base_url(server_url)
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{base}/info",
                                      headers=_proxy_bearer_headers(authorization))
                if r.status_code == 401:
                    # Distinct error so the UI can prompt for a token.
                    raise HTTPException(401, "Inference server requires a bearer token. "
                                              "Paste it in the Token field on the Connect card.")
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
    async def proxy_predict(req: ProxyPredictRequest,
                             authorization: Optional[str] = Header(default=None)):
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

        # Prefer the Authorization header (most secure — never in logs/URLs).
        # Fall back to `bearer_token` in the request body for clients that
        # can't set custom headers.
        outbound_headers = _proxy_bearer_headers(authorization)
        if not outbound_headers and req.bearer_token:
            outbound_headers = {"Authorization": f"Bearer {req.bearer_token}"}

        url = f"{base}/predict"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                t0 = time.time()
                r = await client.post(url, json=payload, headers=outbound_headers)
                wall_ms = (time.time() - t0) * 1000
                if r.status_code == 401:
                    raise HTTPException(401, "Inference server requires a bearer token. "
                                              "Paste it in the Token field on the Connect card.")
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
