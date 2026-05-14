"""
NeuralForge — Inference Server Lifecycle Manager.

The dashboard needs to spin up `neural serve` instances on demand:

  * From a local checkpoint (the existing flow, but without the user
    running a terminal command).
  * From an HF model id, with no prior training (zero-shot inference on
    a published model).
  * From other registered model sources (planned).

Each launched server runs as a separate Python subprocess on a free
localhost port, so it can crash, restart, or be torn down without
touching the dashboard.

**Security model.** Every launched server gets a freshly generated
bearer token (32 bytes, base64) passed through `NEURAL_INFERENCE_TOKEN`
env var. The inference app's middleware enforces it. The dashboard
manager holds the token in process memory keyed by `server_id` and
**never returns it** from any public API — clients call the manager's
proxy endpoints (`/api/inference/{id}/predict` etc.) and the manager
attaches the bearer header internally. This keeps tokens out of browser
storage, network traces, and logs.

The manager exposes:

  * `start(ServerLaunchRequest) -> ServerInfo`
  * `list() -> List[ServerInfo]`
  * `stop(server_id) -> bool`
  * `proxy(server_id, path, method, body) -> dict` — forward to the held
    server using the bearer token.
"""

from __future__ import annotations

# import base64
import os
import secrets
import socket
import subprocess
import sys
# import tempfile
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ServerInfo:
    """Public-facing summary of a managed inference server.

    Tokens never appear here. The dashboard / UI only sees these fields.
    """
    id:           str
    name:         str
    port:         int
    pid:          Optional[int] = None
    started_at:   float = 0.0
    status:       str = "starting"   # starting | running | exited | failed
    model_type:   Optional[str] = None
    source:       Optional[str] = None    # "checkpoint" | "huggingface" | "local"
    model_id:     Optional[str] = None    # HF id or checkpoint path
    config_path:  Optional[str] = None
    checkpoint_path: Optional[str] = None  # absolute path the subprocess loaded
    last_error:   Optional[str] = None
    exit_code:    Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal: the bag we keep per running server
# ---------------------------------------------------------------------------

@dataclass
class _ManagedServer:
    info:  ServerInfo
    proc:  subprocess.Popen
    token: str                         # NEVER in ServerInfo
    log_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class InferenceServerManager:
    """Tracks one or more `neural serve` subprocesses started from the UI.

    Thread-safe — registry mutations are guarded by `_lock`.
    """

    def __init__(self, output_dir: str = "runs",
                 host: str = "127.0.0.1",
                 port_range: tuple[int, int] = (8090, 8190)) -> None:
        self.output_dir = Path(output_dir)
        self.host = host
        self.port_low, self.port_high = port_range
        self._servers: Dict[str, _ManagedServer] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_from_config(self, config_path: str,
                          checkpoint: Optional[str] = None,
                          name: Optional[str] = None) -> ServerInfo:
        """Spawn a `neural serve --config <config_path>` subprocess.

        Two pre-flight checks happen here so we fail fast in the API
        response instead of leaving a dead subprocess and a vague log
        line for the user:

          1. The config file exists.
          2. A checkpoint exists — either the one the caller named, or
             `<run_dir>/checkpoints/checkpoint_best.pt`. We resolve the
             *absolute* path here and pass it via `--checkpoint`. This
             also avoids the path-doubling bug where the subprocess runs
             with `cwd=run_dir` and `cfg.checkpoint_dir` (which is
             `output_dir/name/checkpoints`) resolves under that cwd to
             `<run_dir>/<output_dir>/<name>/checkpoints` — doesn't exist.
        """
        cfg_path = Path(config_path).resolve()
        if not cfg_path.exists() or not cfg_path.is_file():
            raise ValueError(f"Config not found: {config_path}")

        # Resolve the checkpoint to an absolute path BEFORE launching, so
        # the subprocess gets `--checkpoint /abs/path/.pt` and never has
        # to do its own resolution against a `cwd` that's been moved into
        # the run dir.
        run_dir = cfg_path.parent
        resolved_ckpt = self._resolve_checkpoint(cfg_path, checkpoint, run_dir)
        if not resolved_ckpt:
            raise ValueError(
                f"No checkpoint found for {run_dir.name}. Train this config "
                f"first (`neural train -c <config>`), or pass an explicit "
                f"checkpoint path. Looked under {run_dir}/checkpoints/ and "
                f"the runs root."
            )

        # Pick a port and reserve it (close the socket immediately so the
        # subprocess can bind).
        port = self._reserve_port()

        # Generate a fresh bearer token and stash it ONLY in our own state.
        token = _generate_token()

        # Build the command. We invoke the same Python interpreter that's
        # running the dashboard so we're guaranteed the same package set.
        # Using `-c` instead of `-m` because the CLI module is a Click group
        # without a `__main__` block.
        bootstrap = "from neural_platform.cli.commands import cli; cli()"
        cmd = [
            sys.executable, "-c", bootstrap,
            "serve",
            "--config", str(cfg_path),
            "--host", self.host,
            "--port", str(port),
            "--checkpoint", str(resolved_ckpt),
        ]

        # Each server logs to its own file under runs/<name>/inference.log.
        log_path = run_dir / "inference.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["NEURAL_INFERENCE_TOKEN"] = token
        # Make sure stdout/stderr are line-buffered so the log file is useful.
        env["PYTHONUNBUFFERED"] = "1"

        log_fh = open(log_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                cmd, stdout=log_fh, stderr=log_fh, env=env,
                cwd=str(run_dir),
            )
        except Exception:
            log_fh.close()
            raise

        sid = _generate_id()
        info = ServerInfo(
            id=sid,
            name=name or cfg_path.parent.name,
            port=port,
            pid=proc.pid,
            started_at=time.time(),
            status="starting",
            source="checkpoint",
            checkpoint_path=str(resolved_ckpt),
            config_path=str(cfg_path),
        )
        managed = _ManagedServer(info=info, proc=proc, token=token, log_path=log_path)
        with self._lock:
            self._servers[sid] = managed
        return info

    def start_from_hf(self,
                      hf_model_id: str,
                      pipeline_task: str,
                      *,
                      name: Optional[str] = None,
                      revision: Optional[str] = None,
                      trust_remote_code: bool = False,
                      load_in_4bit: bool = False,
                      load_in_8bit: bool = False,
                      bnb_compute_dtype: Optional[str] = None) -> ServerInfo:
        """Spawn an inference server backed by a HuggingFace model id.

        Synthesizes a minimal ``ExperimentConfig`` (model.type=hf_pipeline,
        the user's pipeline_task + pretrained id), writes it to a private
        staging dir under the manager's ``output_dir``, and spawns
        ``neural serve --config <cfg> --no-checkpoint``. The ``--no-checkpoint``
        flag tells the inference app to skip the checkpoint load and rely on
        ``HFPipelineModel.from_pretrained`` for weights.

        Pre-flight checks:
          * The HF id matches ``<owner>/<repo>``.
          * ``pipeline_task`` is non-empty.
          * The synthesized config passes Pydantic validation (offline only —
            the deeper inspector check is the UI's job before launching, so we
            don't double-pay the network round-trip here).
        """
        # ---- validate inputs -------------------------------------------------
        # Reuse the existing HF id validator so we reject malformed ids before
        # writing anything to disk or spawning a subprocess. Avoids the trap
        # of `\` or `?token=...` flowing through to the Hub.
        try:
            from neural_platform.core.model_source import validate_hf_model_id
        except Exception:
            validate_hf_model_id = None  # type: ignore
        if validate_hf_model_id:
            try:
                hf_model_id = validate_hf_model_id(hf_model_id)
            except Exception as exc:
                raise ValueError(str(exc))

        task = (pipeline_task or "").strip()
        if not task:
            raise ValueError("pipeline_task is required (e.g. 'text-classification', "
                             "'image-classification', 'automatic-speech-recognition').")

        # ---- synthesize a config + write it to a per-server staging dir ----
        cfg, run_dir = _synthesize_hf_config(
            output_root=self.output_dir,
            hf_model_id=hf_model_id,
            pipeline_task=task,
            revision=revision,
            trust_remote_code=trust_remote_code,
            display_name=name,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_compute_dtype=bnb_compute_dtype,
        )
        cfg_path = run_dir / "config.yaml"

        # ---- spawn the subprocess ------------------------------------------
        port = self._reserve_port()
        token = _generate_token()

        bootstrap = "from neural_platform.cli.commands import cli; cli()"
        cmd = [
            sys.executable, "-c", bootstrap,
            "serve",
            "--config", str(cfg_path),
            "--host", self.host,
            "--port", str(port),
            "--no-checkpoint",
        ]

        log_path = run_dir / "inference.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["NEURAL_INFERENCE_TOKEN"] = token
        env["PYTHONUNBUFFERED"] = "1"

        log_fh = open(log_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                cmd, stdout=log_fh, stderr=log_fh, env=env, cwd=str(run_dir),
            )
        except Exception:
            log_fh.close()
            raise

        sid = _generate_id()
        info = ServerInfo(
            id=sid,
            name=name or f"hf:{hf_model_id}",
            port=port,
            pid=proc.pid,
            started_at=time.time(),
            status="starting",
            model_type="hf_pipeline",
            source="huggingface",
            model_id=hf_model_id,
            checkpoint_path=None,
            config_path=str(cfg_path),
        )
        managed = _ManagedServer(info=info, proc=proc, token=token, log_path=log_path)
        with self._lock:
            self._servers[sid] = managed
        return info

    def stop(self, server_id: str, *, timeout_s: float = 5.0) -> bool:
        """Terminate a managed server. Returns True if it stopped cleanly."""
        with self._lock:
            managed = self._servers.get(server_id)
        if not managed:
            return False
        proc = managed.proc
        try:
            proc.terminate()
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=timeout_s)
        except Exception:
            pass
        managed.info.status = "exited"
        managed.info.exit_code = proc.returncode
        return True

    def list(self) -> List[ServerInfo]:
        """Snapshot of all managed servers, with `status` refreshed."""
        out: List[ServerInfo] = []
        with self._lock:
            servers = list(self._servers.values())
        for managed in servers:
            self._refresh(managed)
            out.append(managed.info)
        return out

    def get(self, server_id: str) -> Optional[ServerInfo]:
        with self._lock:
            managed = self._servers.get(server_id)
        if not managed:
            return None
        self._refresh(managed)
        return managed.info

    def remove(self, server_id: str) -> bool:
        """Stop and forget a server."""
        self.stop(server_id)
        with self._lock:
            return self._servers.pop(server_id, None) is not None

    # ------------------------------------------------------------------
    # Token-attached proxy
    # ------------------------------------------------------------------

    def proxy(self, server_id: str, path: str,
              method: str = "GET", json_body: Optional[dict] = None,
              timeout_s: float = 30.0) -> Dict[str, Any]:
        """Forward a request to the held server with its bearer token attached.

        Returns the parsed JSON body on success. Raises on transport / HTTP
        errors. The token never leaves this process.
        """
        with self._lock:
            managed = self._servers.get(server_id)
        if not managed:
            raise KeyError(f"Unknown inference server '{server_id}'.")
        self._refresh(managed)
        if managed.info.status != "running":
            raise RuntimeError(
                f"Inference server '{server_id}' is {managed.info.status}; "
                f"can't proxy requests."
            )

        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required to proxy to inference servers") from exc

        url = f"http://{self.host}:{managed.info.port}{path if path.startswith('/') else '/' + path}"
        headers = {"Authorization": f"Bearer {managed.token}"}
        with httpx.Client(timeout=timeout_s) as client:
            if method.upper() == "GET":
                r = client.get(url, headers=headers)
            else:
                r = client.request(method.upper(), url, json=json_body, headers=headers)
        # Don't include the token in any error surfaces.
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text[:300]
            raise RuntimeError(f"Inference server returned {r.status_code}: {detail}")
        try:
            return r.json()
        except Exception:
            return {"text": r.text}

    def proxy_stream(self, server_id: str, path: str,
                      json_body: Optional[dict] = None,
                      timeout_s: float = 600.0):
        """Async-generator proxy that forwards an SSE stream from the
        held server with its bearer token attached.

        Returns an **async generator** directly (not a coroutine). The
        caller does ``async for chunk in mgr.proxy_stream(...)``; the
        old ``async def`` form returned a coroutine that ASGI couldn't
        iterate, surfacing as ``'async for' requires __aiter__, got
        coroutine`` from inside Starlette's StreamingResponse.

        Yields raw bytes chunks suitable for re-emission via
        ``StreamingResponse``. The token never appears in the yielded
        bytes — it's only added to the outbound request header. If the
        upstream returns a non-200 the generator yields a single SSE
        ``error`` event so the browser can render something useful.
        """
        with self._lock:
            managed = self._servers.get(server_id)
        if not managed:
            raise KeyError(f"Unknown inference server '{server_id}'.")
        self._refresh(managed)
        if managed.info.status != "running":
            raise RuntimeError(
                f"Inference server '{server_id}' is {managed.info.status}; "
                f"can't stream."
            )
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required to proxy streams") from exc

        url = f"http://{self.host}:{managed.info.port}{path}"
        headers = {
            "Authorization": f"Bearer {managed.token}",
            "Accept":        "text/event-stream",
        }

        async def gen():
            import json as _json
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                try:
                    async with client.stream(
                        "POST", url, headers=headers, json=json_body or {},
                    ) as resp:
                        if resp.status_code >= 400:
                            try:
                                detail = (await resp.aread()).decode("utf-8", "replace")
                            except Exception:
                                detail = f"HTTP {resp.status_code}"
                            payload = _json.dumps({"detail": detail[:300]})
                            yield f"event: error\ndata: {payload}\n\n".encode()
                            return
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                yield chunk
                except Exception as exc:
                    payload = _json.dumps({"detail": _redact(str(exc))[:300]})
                    yield f"event: error\ndata: {payload}\n\n".encode()
        return gen()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh(self, managed: _ManagedServer) -> None:
        """Update `info.status` based on the current subprocess state."""
        rc = managed.proc.poll()
        if rc is None:
            # Still alive — verify the HTTP server has come up via /health.
            if managed.info.status in ("starting",):
                if self._healthcheck_ok(managed):
                    managed.info.status = "running"
        else:
            managed.info.status = "exited" if rc == 0 else "failed"
            managed.info.exit_code = rc
            # Try to surface the last few log lines as `last_error` for the UI.
            if managed.log_path and managed.log_path.exists():
                try:
                    tail = _tail(managed.log_path, max_bytes=2048)
                    if tail:
                        managed.info.last_error = _redact(tail.strip().splitlines()[-1])[:240]
                except Exception:
                    pass

    def _healthcheck_ok(self, managed: _ManagedServer) -> bool:
        """Hit /health (no auth required) to confirm the server is up.

        Sub-second timeout — failure just means "not ready yet"; the UI will
        poll again.
        """
        try:
            import httpx
        except ImportError:
            return False
        try:
            url = f"http://{self.host}:{managed.info.port}/health"
            with httpx.Client(timeout=0.6) as client:
                r = client.get(url)
            return r.status_code == 200
        except Exception:
            return False

    def _resolve_checkpoint(self,
                             cfg_path: Path,
                             explicit: Optional[str],
                             run_dir: Path) -> Optional[Path]:
        """Find the checkpoint path to load, *as an absolute path*.

        Search order:
          1. If `explicit` was provided, resolve it (relative to cwd, then
             relative to run_dir, then relative to the manager's
             output_dir). Reject if it doesn't point at a file.
          2. `<run_dir>/checkpoints/checkpoint_best.pt`
          3. The newest `<run_dir>/checkpoints/*.pt` if best isn't there.
          4. `<output_dir>/<run_dir.name>/checkpoints/...` — same patterns,
             in case the run_dir we got from cfg_path.parent isn't the
             canonical one.

        Returns None if nothing is found — caller turns this into a clear
        4xx error before launching the subprocess.
        """
        # Path 1: explicit checkpoint
        if explicit:
            for base in (Path.cwd(), run_dir, self.output_dir.resolve()):
                candidate = (base / explicit).resolve()
                if candidate.exists() and candidate.is_file():
                    return candidate
            return None

        # Path 2 + 3: checkpoints under run_dir
        ckpt_dir = run_dir / "checkpoints"
        best = ckpt_dir / "checkpoint_best.pt"
        if best.exists():
            return best.resolve()
        if ckpt_dir.exists():
            pts = sorted(ckpt_dir.glob("*.pt"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                return pts[0].resolve()

        # Path 4: fallback under the manager's configured output_dir, in
        # case the user moved things around (or the run dir was renamed).
        alt = (self.output_dir / run_dir.name / "checkpoints").resolve()
        if alt.exists():
            best = alt / "checkpoint_best.pt"
            if best.exists():
                return best
            pts = sorted(alt.glob("*.pt"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                return pts[0]

        return None

    def _reserve_port(self) -> int:
        """Find a free port in the configured range. We try the canonical
        IANA `bind to 0` trick first, then walk the range."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((self.host, 0))
                _, port = s.getsockname()
                if self.port_low <= port <= self.port_high:
                    return port
            except OSError:
                pass
        for port in range(self.port_low, self.port_high + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((self.host, port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(
            f"No free ports in {self.port_low}-{self.port_high} for new inference server."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HF-launch config synthesis
# ---------------------------------------------------------------------------

# Map a NeuralForge core.tasks.Task to the closest match in
# core.config.Task. The two enums overlap but aren't 1:1 — config.Task is a
# narrower set tied to the trainer's loss + metrics dispatch.
_COARSE_TASK_FALLBACKS: Dict[str, str] = {
    # Text generation / seq2seq don't have a first-class slot in config.Task,
    # so we map them to CLASSIFICATION and let the loss default ride. The
    # trainer never runs for these synthesized configs anyway — this only
    # exists to satisfy the validator's schema check.
    "summarization":            "classification",
    "translation":              "classification",
    "text-generation":          "classification",
    "fill-mask":                "classification",
    "feature-extraction":       "classification",
    "automatic-speech-recognition": "classification",
    "image-to-text":            "classification",
    "image-text-to-text":       "classification",
    "visual-question-answering":"classification",
    "document-question-answering": "classification",
    "question-answering":       "classification",
    "depth-estimation":         "regression",
    # Direct passthroughs — both enums have these strings.
    "text-classification":      "text_classification",
    "token-classification":     "classification",
    "image-classification":     "image_classification",
    "audio-classification":     "classification",
    "video-classification":     "classification",
    "zero-shot-classification": "classification",
    "image-segmentation":       "classification",
    "object-detection":         "classification",
    "voice-activity-detection": "classification",
}


def _coarse_task_for_config(spec) -> str:
    """Pick a value for ``training.task`` (an instance of
    ``core.config.Task``) from a pipeline spec. Different enum than the
    fine-grained ``core.tasks.Task`` — see the fallback table above.
    """
    return _COARSE_TASK_FALLBACKS.get(spec.task, "classification")


def _synthesize_hf_config(
    output_root: Path,
    hf_model_id: str,
    pipeline_task: str,
    *,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    display_name: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_compute_dtype: Optional[str] = None,
):
    """Build a minimal ExperimentConfig + write it to a private run dir.

    Returns ``(ExperimentConfig, run_dir)``. The run dir is the dashboard's
    output_dir + ``_hf_servers/<rand>/`` so the synthesized configs sit
    next to other runs but in their own subtree (easy to clean up; doesn't
    pollute the experiments list).

    The synthesized config is intentionally minimal — just enough for the
    inference server to build the model and dispatch requests:

      * model.type        = hf_pipeline
      * model.hf_pipeline.pretrained = the user's HF id
      * training.pipeline_task        = the user's task (used by the input adapter)
      * training.task / training.loss = derived from the pipeline spec so the
        validator picks the right code path (e.g. depth-estimation → MSE,
        not cross-entropy)
      * data.source       = synthetic (won't be used; required by schema)

    All of the above defaults come from
    :mod:`core.pipeline_specs` so server-side and config-side stay in sync.

    We **don't** call the HF Hub here. Inspector checks happen in the UI
    before launch; doing them again would double the latency on every
    Launch click for no win.
    """
    from neural_platform.core.config import (
        ExperimentConfig, ModelConfig, ModelType, Framework,
        HFPipelineConfig, TrainingConfig, DataConfig, DeployConfig,
        LossFunction, Task,
    )
    from neural_platform.core.pipeline_specs import resolve
    import yaml as _yaml

    spec = resolve(pipeline_task)
    coarse = _coarse_task_for_config(spec)

    # Loss derived from the spec. Fall back to cross_entropy for anything
    # not explicitly mapped — the synthesized config is for serving, not
    # training, so the loss is never actually exercised.
    loss_value = (spec.default_loss or "cross_entropy").lower()
    try:
        loss = LossFunction(loss_value)
    except ValueError:
        loss = LossFunction.CROSS_ENTROPY

    # Slugify the model id for the run dir name. We keep the random suffix
    # so re-launches of the same model don't collide and we can tell which
    # subprocess wrote which log file.
    slug = hf_model_id.replace("/", "__").replace("-", "_")[:48]
    rand = secrets.token_hex(4)
    run_name = f"hf_{slug}_{rand}"
    run_dir = (output_root / "_hf_servers" / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        name=run_name,
        description=display_name or f"Managed HF server for {hf_model_id}",
        output_dir=str(output_root),
        model=ModelConfig(
            type=ModelType.HF_PIPELINE,
            name=display_name or hf_model_id,
            framework=Framework.PYTORCH,
            hf_pipeline=HFPipelineConfig(
                pretrained=hf_model_id,
                revision=revision,
                trust_remote_code=trust_remote_code,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_compute_dtype=bnb_compute_dtype,
            ),
        ),
        training=TrainingConfig(
            task=Task(coarse),
            pipeline_task=pipeline_task,
            loss=loss,
            num_epochs=1,    # never trained; satisfy schema
        ),
        data=DataConfig(),   # synthetic defaults — never used
        deploy=DeployConfig(),
    )

    # The inference subprocess runs with cwd=run_dir, so the config is
    # written next to it; that mirrors the runs/<exp>/config.yaml layout.
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(_yaml.safe_dump(cfg.model_dump(mode="json"),
                                        sort_keys=False, default_flow_style=False))
    return cfg, run_dir


def _generate_token() -> str:
    """32-byte URL-safe random token. Used as the bearer secret per server."""
    return secrets.token_urlsafe(32)


def _generate_id() -> str:
    return secrets.token_hex(8)


def _tail(path: Path, max_bytes: int = 4096) -> str:
    """Read the last `max_bytes` bytes of a log file, decoded as utf-8 best-effort."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _redact(text: str) -> str:
    """Scrub anything that looks like a bearer/HF token from log tails."""
    try:
        from neural_platform.core.hf_auth import redact
        return redact(text)
    except Exception:
        return text
