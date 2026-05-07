"""
Tests for the inference-server lifecycle manager + bearer-token gate.

Strictly offline / local — no real `neural serve` subprocess is spawned.
Where the manager *would* spawn one, we patch `subprocess.Popen` and
verify the env var, command line, and proxy headers without launching
anything heavy.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from neural_platform.web.inference_manager import (
    InferenceServerManager,
    ServerInfo,
    _generate_token,
    _redact,
)


# ---------------------------------------------------------------------------
# Token generation + redaction
# ---------------------------------------------------------------------------

class TestTokenSafety:

    def test_generated_token_has_useful_entropy(self):
        a, b = _generate_token(), _generate_token()
        assert a != b
        assert len(a) >= 30
        assert all(c.isalnum() or c in "-_" for c in a)

    def test_redact_strips_hf_token(self):
        out = _redact("token=hf_AAAAAAAAAAAAAAAAAAAAAAAA")
        assert "hf_AAAAA" not in out

    def test_server_info_to_dict_excludes_token(self):
        info = ServerInfo(id="abc", name="x", port=8090)
        d = info.to_dict()
        # Token is intentionally not on ServerInfo. Verify no leak.
        assert "token" not in d
        assert "bearer" not in {k.lower() for k in d.keys()}


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

class TestCheckpointResolution:
    """The manager should pre-resolve the checkpoint to an absolute path
    BEFORE spawning, so cwd-relative resolution inside the subprocess
    can't double-prefix and 404."""

    def test_picks_checkpoint_best_when_present(self, tmp_path):
        run_dir = tmp_path / "runs" / "exp"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        cfg = run_dir / "config.yaml"
        cfg.write_text("name: exp")
        best = ckpts / "checkpoint_best.pt"
        best.write_bytes(b"\x80")
        mgr = InferenceServerManager(output_dir=str(tmp_path / "runs"))
        out = mgr._resolve_checkpoint(cfg, None, run_dir)
        assert out == best.resolve()

    def test_falls_back_to_newest_pt(self, tmp_path):
        run_dir = tmp_path / "runs" / "exp"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        cfg = run_dir / "config.yaml"
        cfg.write_text("name: exp")
        # No checkpoint_best.pt — only an epoch ckpt.
        epoch_ckpt = ckpts / "checkpoint_epoch_0010.pt"
        epoch_ckpt.write_bytes(b"\x80")
        mgr = InferenceServerManager(output_dir=str(tmp_path / "runs"))
        out = mgr._resolve_checkpoint(cfg, None, run_dir)
        assert out == epoch_ckpt.resolve()

    def test_returns_none_when_no_checkpoints(self, tmp_path):
        run_dir = tmp_path / "runs" / "exp"
        run_dir.mkdir(parents=True)
        cfg = run_dir / "config.yaml"
        cfg.write_text("name: exp")
        mgr = InferenceServerManager(output_dir=str(tmp_path / "runs"))
        assert mgr._resolve_checkpoint(cfg, None, run_dir) is None

    def test_explicit_checkpoint_resolved_against_run_dir(self, tmp_path):
        run_dir = tmp_path / "runs" / "exp"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        cfg = run_dir / "config.yaml"
        cfg.write_text("name: exp")
        ckpt = ckpts / "my_ckpt.pt"
        ckpt.write_bytes(b"\x80")
        mgr = InferenceServerManager(output_dir=str(tmp_path / "runs"))
        # Pass a relative path; resolver should find it under run_dir.
        out = mgr._resolve_checkpoint(cfg, "checkpoints/my_ckpt.pt", run_dir)
        assert out == ckpt.resolve()

    def test_start_raises_clear_error_when_no_checkpoint(self, tmp_path):
        """Replaces the cryptic 'No checkpoint found. Run neural train first.'
        message buried in the subprocess log with a 400 from the API."""
        run_dir = tmp_path / "runs" / "exp"
        run_dir.mkdir(parents=True)
        cfg = run_dir / "config.yaml"
        cfg.write_text("name: exp")
        mgr = InferenceServerManager(output_dir=str(tmp_path / "runs"))
        with pytest.raises(ValueError) as exc_info:
            mgr.start_from_config(str(cfg))
        msg = str(exc_info.value)
        assert "No checkpoint" in msg
        assert "exp" in msg  # mentions which run

    def test_start_passes_absolute_checkpoint_path_to_subprocess(self, tmp_path):
        """Critical: the subprocess must receive --checkpoint <ABSOLUTE
        PATH> so its own cwd-relative resolution can't double-prefix."""
        run_dir = tmp_path / "runs" / "exp"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        cfg = run_dir / "config.yaml"
        cfg.write_text("name: exp")
        best = ckpts / "checkpoint_best.pt"
        best.write_bytes(b"\x80")
        mgr = InferenceServerManager(output_dir=str(tmp_path / "runs"))

        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 1
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc) as popen:
            info = mgr.start_from_config(str(cfg))

        cmd = popen.call_args.args[0]
        assert "--checkpoint" in cmd
        ckpt_arg = cmd[cmd.index("--checkpoint") + 1]
        assert Path(ckpt_arg).is_absolute()
        assert Path(ckpt_arg) == best.resolve()
        # ServerInfo also exposes it (so the UI can show what's loaded)
        assert info.checkpoint_path == str(best.resolve())


# ---------------------------------------------------------------------------
# Lifecycle (mocked Popen)
# ---------------------------------------------------------------------------

class TestLifecycle:

    def _make_proc(self, alive: bool = True, returncode: int = 0):
        proc = MagicMock()
        proc.poll.return_value = None if alive else returncode
        proc.returncode = None if alive else returncode
        proc.terminate = MagicMock()
        proc.wait = MagicMock()
        proc.kill = MagicMock()
        proc.pid = 12345
        return proc

    def _seed_run(self, tmp_path):
        """Make a `runs/exp/{config.yaml,checkpoints/checkpoint_best.pt}` so
        start_from_config has both a config and a checkpoint to point at."""
        run_dir = tmp_path / "exp"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        cfg = run_dir / "config.yaml"
        cfg.write_text("name: exp")
        (ckpts / "checkpoint_best.pt").write_bytes(b"\x80")
        return cfg

    def test_start_validates_config_path(self, tmp_path):
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        with pytest.raises(ValueError):
            mgr.start_from_config(str(tmp_path / "missing.yaml"))

    def test_start_passes_token_via_env(self, tmp_path):
        cfg_path = self._seed_run(tmp_path)
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = self._make_proc(alive=True)

        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc) as popen:
            info = mgr.start_from_config(str(cfg_path), name="testserver")

        # Verify the call: env contains a token, command is a -c bootstrap.
        kwargs = popen.call_args.kwargs
        env = kwargs["env"]
        assert "NEURAL_INFERENCE_TOKEN" in env
        token = env["NEURAL_INFERENCE_TOKEN"]
        assert len(token) >= 30
        # The token must NOT appear in the public ServerInfo.
        d = info.to_dict()
        assert token not in str(d)
        # Command should invoke the CLI's serve subcommand.
        cmd = popen.call_args.args[0]
        assert "serve" in cmd
        assert "--config" in cmd
        assert str(cfg_path) in cmd

    def test_list_reports_status(self, tmp_path):
        cfg_path = self._seed_run(tmp_path)
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = self._make_proc(alive=True)
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc):
            info = mgr.start_from_config(str(cfg_path))
        assert info.status == "starting"
        # Patch healthcheck to say "yes it's up"
        with patch.object(mgr, "_healthcheck_ok", return_value=True):
            listed = mgr.list()
        assert len(listed) == 1
        assert listed[0].status == "running"

    def test_stop_terminates_subprocess(self, tmp_path):
        cfg_path = self._seed_run(tmp_path)
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = self._make_proc(alive=True)
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc):
            info = mgr.start_from_config(str(cfg_path))
        ok = mgr.stop(info.id)
        assert ok
        proc.terminate.assert_called_once()

    def test_remove_clears_registry(self, tmp_path):
        cfg_path = self._seed_run(tmp_path)
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = self._make_proc(alive=True)
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc):
            info = mgr.start_from_config(str(cfg_path))
        assert mgr.remove(info.id)
        assert mgr.get(info.id) is None

    def test_dead_subprocess_marked_failed(self, tmp_path):
        cfg_path = self._seed_run(tmp_path)
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = self._make_proc(alive=True)
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc):
            info = mgr.start_from_config(str(cfg_path))
        # Kill it: make poll return a non-zero exit code
        proc.poll.return_value = 1
        proc.returncode = 1
        listed = mgr.list()
        assert listed[0].status == "failed"
        assert listed[0].exit_code == 1


# ---------------------------------------------------------------------------
# Proxy: bearer header attached, token never in error messages
# ---------------------------------------------------------------------------

class TestProxy:

    def _seed_run(self, tmp_path):
        run_dir = tmp_path / "exp"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        cfg = run_dir / "config.yaml"
        cfg.write_text("name: exp")
        (ckpts / "checkpoint_best.pt").write_bytes(b"\x80")
        return cfg

    def _make_running_mgr(self, tmp_path):
        cfg_path = self._seed_run(tmp_path)
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 1
        with patch("neural_platform.web.inference_manager.subprocess.Popen",
                    return_value=proc):
            info = mgr.start_from_config(str(cfg_path))
        # Force status to running so proxy is allowed
        with patch.object(mgr, "_healthcheck_ok", return_value=True):
            mgr.list()
        return mgr, info

    def test_proxy_attaches_bearer_token(self, tmp_path):
        mgr, info = self._make_running_mgr(tmp_path)

        # Mock httpx.Client to capture headers
        import httpx
        captured = {}

        class _FakeResponse:
            status_code = 200
            text = "{}"
            def json(self): return {"ok": True}

        class _FakeClient:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url, headers=None):
                captured["url"] = url
                captured["headers"] = headers or {}
                return _FakeResponse()
            def request(self, method, url, json=None, headers=None):
                captured["method"] = method
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers or {}
                return _FakeResponse()

        with patch.object(httpx, "Client", _FakeClient):
            mgr.proxy(info.id, "/info", method="GET")

        assert captured["headers"].get("Authorization", "").startswith("Bearer ")
        # The url must be local
        assert captured["url"].startswith("http://127.0.0.1:")

    def test_proxy_unknown_server_raises(self, tmp_path):
        mgr = InferenceServerManager(output_dir=str(tmp_path))
        with pytest.raises(KeyError):
            mgr.proxy("nope", "/info")
