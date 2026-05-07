"""
Regression tests for `_validated_config_path` / `_scan_configs` path-shape
handling. Specifically: the UI dropdown emits cwd-relative paths like
`runs/exp/config.yaml`, which used to produce a 404 because the validator
prepended output_root and produced `<cwd>/runs/runs/...`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neural_platform.web.app import create_dashboard_app


def _make_config(tmp_path: Path, name: str = "exp") -> Path:
    """Create a minimal valid `runs/<name>/config.yaml` under tmp_path."""
    cfg_dir = tmp_path / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(
        "name: " + name + "\n"
        "model:\n  type: mlp\n  framework: pytorch\n"
        "  mlp:\n    input_size: 4\n    output_size: 2\n"
        "training:\n  task: classification\n  loss: cross_entropy\n"
        "  num_epochs: 1\n  batch_size: 8\n"
        "  optimizer: {type: adam, lr: 0.001}\n"
        "  scheduler: {type: none}\n"
        "data:\n  source: synthetic\n"
    )
    return cfg_dir / "config.yaml"


class TestConfigPathShapes:

    def test_absolute_path_resolves(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        monkeypatch.chdir(tmp_path.parent)
        app = create_dashboard_app(output_dir=str(tmp_path))
        client = TestClient(app)
        r = client.get(f"/api/configs/load?path={cfg}")
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "exp"

    def test_cwd_relative_path_resolves(self, tmp_path, monkeypatch):
        """Most important case — this is what /api/configs returns to the UI.

        The dashboard was launched from a parent directory, so the dropdown
        sends back paths like 'runs/exp/config.yaml' relative to cwd. The
        previous validator double-prefixed and 404'd.
        """
        runs_dir = tmp_path / "runs"
        cfg = _make_config(runs_dir)
        monkeypatch.chdir(tmp_path)
        app = create_dashboard_app(output_dir="runs")
        client = TestClient(app)
        # Path as the dropdown would emit it
        r = client.get("/api/configs/load?path=runs/exp/config.yaml")
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "exp"

    def test_output_root_relative_path_resolves(self, tmp_path, monkeypatch):
        """A bare 'exp/config.yaml' (no runs/ prefix) still works."""
        runs_dir = tmp_path / "runs"
        _make_config(runs_dir)
        monkeypatch.chdir(tmp_path)
        app = create_dashboard_app(output_dir="runs")
        client = TestClient(app)
        r = client.get("/api/configs/load?path=exp/config.yaml")
        assert r.status_code == 200, r.text

    def test_nonexistent_path_404s_with_clear_message(self, tmp_path, monkeypatch):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        app = create_dashboard_app(output_dir="runs")
        client = TestClient(app)
        r = client.get("/api/configs/load?path=runs/missing/config.yaml")
        assert r.status_code == 404
        # The error mentions the resolved path so users can debug
        assert "missing" in r.json().get("detail", "").lower() \
            or "not found" in r.json().get("detail", "").lower()

    def test_path_outside_output_dir_400s(self, tmp_path, monkeypatch):
        runs_dir = tmp_path / "runs"
        _make_config(runs_dir)
        # Plant a config OUTSIDE runs/ — the validator must reject it
        # even though it exists.
        outside = tmp_path / "secrets" / "config.yaml"
        outside.parent.mkdir()
        outside.write_text("name: secret")
        monkeypatch.chdir(tmp_path)
        app = create_dashboard_app(output_dir="runs")
        client = TestClient(app)
        r = client.get(f"/api/configs/load?path={outside}")
        assert r.status_code == 400

    def test_inference_launch_with_dropdown_path(self, tmp_path, monkeypatch):
        """End-to-end: /api/configs returns a path, /api/inference/start
        accepts it. This is exactly the user-reported failure path."""
        runs_dir = tmp_path / "runs"
        cfg = _make_config(runs_dir)
        monkeypatch.chdir(tmp_path)
        app = create_dashboard_app(output_dir="runs")
        client = TestClient(app)

        listed = client.get("/api/configs").json()
        assert listed, "Expected /api/configs to find the seeded config"
        path_from_dropdown = listed[0]["path"]

        # /api/inference/start should accept whatever shape /api/configs emits.
        # It will fail later (subprocess) but must NOT 404 on path validation.
        r = client.post(
            "/api/inference/start",
            json={"config_path": path_from_dropdown},
        )
        assert r.status_code != 404, f"Got 404 'Config not found' on dropdown path: {r.json()}"
        # Cleanup any stray subprocess
        try:
            sid = r.json().get("id")
            if sid:
                client.post(f"/api/inference/{sid}/stop")
        except Exception:
            pass
