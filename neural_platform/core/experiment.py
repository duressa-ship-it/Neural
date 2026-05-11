"""
NeuralForge Experiment Tracker
SQLite-backed persistent logging of all runs, epochs, and metrics.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_FILE = "neuralforge.db"


class ExperimentTracker:
    """
    Tracks experiments, runs, and per-epoch metrics in a SQLite database.
    One database per output_dir. Designed for concurrent read, single-writer access.
    """

    def __init__(self, db_path: str | Path = DB_FILE):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    description TEXT,
                    tags        TEXT,
                    config_json TEXT,
                    status      TEXT DEFAULT 'pending',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
                    run_number      INTEGER NOT NULL,
                    framework       TEXT,
                    device          TEXT,
                    status          TEXT DEFAULT 'running',
                    best_val_loss   REAL,
                    best_epoch      INTEGER,
                    total_epochs    INTEGER,
                    duration_secs   REAL,
                    checkpoint_path TEXT,
                    started_at      TEXT NOT NULL,
                    finished_at     TEXT
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      INTEGER NOT NULL REFERENCES runs(id),
                    epoch       INTEGER NOT NULL,
                    phase       TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    logged_at   TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, epoch);

                -- Predict history: per managed-inference-server call log.
                -- One row per successful /predict (or /predict/stream "done"
                -- event). Drives the Predict tab's "Recent" sidebar.
                -- request_json:   the body sent to the inference server,
                --                 with large b64 fields replaced by placeholders
                --                 (see _shrink_predict_request).
                -- response_json:  predictions array + result_kind + the
                --                 metadata the renderer needs to replay.
                --                 Large embedded PNGs are dropped.
                -- latency_ms:     wall-clock latency measured at proxy time.
                -- result_kind:    duplicated from response_json for cheap WHERE.
                CREATE TABLE IF NOT EXISTS predict_history (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    server_id     TEXT NOT NULL,
                    server_url    TEXT,
                    server_name   TEXT,
                    pipeline_task TEXT,
                    request_json  TEXT NOT NULL,
                    response_json TEXT,
                    result_kind   TEXT,
                    latency_ms    REAL,
                    created_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_predict_server
                    ON predict_history(server_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_predict_created
                    ON predict_history(created_at DESC);
            """)

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def create_experiment(self, name: str, config: Any, description: str = "", tags: list = []) -> int:
        now = datetime.utcnow().isoformat()
        config_json = json.dumps(config.model_dump() if hasattr(config, "model_dump") else config)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO experiments (name, description, tags, config_json, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'running', ?, ?)",
                (name, description, json.dumps(tags), config_json, now, now),
            )
            return cur.lastrowid

    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_experiments(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.*,
                       MIN(r.best_val_loss) AS best_val_loss,
                       MAX(r.best_epoch)    AS best_epoch
                FROM experiments e
                LEFT JOIN runs r ON r.experiment_id = e.id
                GROUP BY e.id
                ORDER BY e.created_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def update_experiment_status(self, experiment_id: int, status: str):
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, experiment_id),
            )

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def start_run(self, experiment_id: int, framework: str, device: str) -> int:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            # Get next run number for this experiment
            row = conn.execute(
                "SELECT COALESCE(MAX(run_number), 0) + 1 FROM runs WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            run_number = row[0]
            cur = conn.execute(
                "INSERT INTO runs (experiment_id, run_number, framework, device, status, started_at) "
                "VALUES (?, ?, ?, ?, 'running', ?)",
                (experiment_id, run_number, framework, str(device), now),
            )
            return cur.lastrowid

    def finish_run(
        self,
        run_id: int,
        status: str,
        best_val_loss: Optional[float],
        best_epoch: Optional[int],
        total_epochs: int,
        checkpoint_path: Optional[str],
        started_at: float,
    ):
        now = datetime.utcnow().isoformat()
        duration = time.time() - started_at
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status=?, best_val_loss=?, best_epoch=?, total_epochs=?, "
                "duration_secs=?, checkpoint_path=?, finished_at=? WHERE id=?",
                (status, best_val_loss, best_epoch, total_epochs,
                 duration, checkpoint_path, now, run_id),
            )

    def interrupt_stale_runs(self) -> int:
        """
        Mark every experiment/run that is still status='running' as 'interrupted'.
        Called when a training subprocess is forcibly stopped from the dashboard,
        or on startup when no subprocess is alive.

        Computes a duration based on started_at so the row isn't left
        with a NULL duration. Pulls best metrics if any epochs were logged
        before the interruption.

        Returns the number of runs updated.
        """
        now_iso = datetime.utcnow().isoformat()
        with self._connect() as conn:
            stale = conn.execute(
                "SELECT id, started_at, best_val_loss, best_epoch, total_epochs "
                "FROM runs WHERE status='running'"
            ).fetchall()

            for row in stale:
                run_id = row["id"]
                # Best metric and epoch count from logged epochs, if any
                metric_row = conn.execute(
                    "SELECT MIN(json_extract(metrics_json, '$.loss')) AS best_loss, "
                    "       MAX(epoch) AS max_epoch, "
                    "       COUNT(DISTINCT epoch) AS n_epochs "
                    "FROM metrics WHERE run_id = ? AND phase = 'val'",
                    (run_id,),
                ).fetchone()

                best_loss = row["best_val_loss"]
                if best_loss is None and metric_row and metric_row["best_loss"] is not None:
                    best_loss = metric_row["best_loss"]
                total_epochs = row["total_epochs"] or (metric_row["n_epochs"] if metric_row else 0) or 0

                # Compute duration from started_at
                duration: Optional[float] = None
                try:
                    started = datetime.fromisoformat(row["started_at"])
                    duration = max(0.0, (datetime.utcnow() - started).total_seconds())
                except Exception:
                    pass

                conn.execute(
                    "UPDATE runs SET status='interrupted', finished_at=?, "
                    "best_val_loss=COALESCE(best_val_loss, ?), total_epochs=?, "
                    "duration_secs=COALESCE(duration_secs, ?) "
                    "WHERE id=?",
                    (now_iso, best_loss, total_epochs, duration, run_id),
                )

            conn.execute(
                "UPDATE experiments SET status='interrupted', updated_at=? WHERE status='running'",
                (now_iso,),
            )
            return len(stale)

    def delete_experiment(self, experiment_id: int) -> bool:
        """Cascade delete an experiment, all its runs, and all logged metrics."""
        with self._connect() as conn:
            run_ids = [r["id"] for r in conn.execute(
                "SELECT id FROM runs WHERE experiment_id=?", (experiment_id,)
            ).fetchall()]
            if run_ids:
                placeholders = ",".join("?" * len(run_ids))
                conn.execute(f"DELETE FROM metrics WHERE run_id IN ({placeholders})", run_ids)
            conn.execute("DELETE FROM runs WHERE experiment_id=?", (experiment_id,))
            cur = conn.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
            return cur.rowcount > 0

    def search_experiments(self, query: Optional[str] = None,
                           status: Optional[str] = None) -> List[Dict]:
        """Return experiments filtered by name (LIKE) and/or status."""
        sql = """
            SELECT e.*,
                   MIN(r.best_val_loss) AS best_val_loss,
                   MAX(r.best_epoch)    AS best_epoch,
                   COUNT(r.id)          AS run_count
            FROM experiments e
            LEFT JOIN runs r ON r.experiment_id = e.id
            WHERE 1=1
        """
        params: List[Any] = []
        if query:
            sql += " AND (e.name LIKE ? OR COALESCE(e.description,'') LIKE ?)"
            like = f"%{query}%"
            params.extend([like, like])
        if status:
            sql += " AND e.status = ?"
            params.append(status)
        sql += " GROUP BY e.id ORDER BY e.created_at DESC"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_run(self, run_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(self, experiment_id: int) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE experiment_id = ? ORDER BY run_number DESC",
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metrics(self, run_id: int, epoch: int, phase: str, metrics: Dict[str, float]):
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO metrics (run_id, epoch, phase, metrics_json, logged_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, epoch, phase, json.dumps(metrics), now),
            )

    def get_metrics(self, run_id: int) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY epoch, phase",
                (run_id,),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["metrics"] = json.loads(d.pop("metrics_json"))
                result.append(d)
            return result

    def get_experiment_metrics(self, experiment_id: int) -> List[Dict]:
        """Get all metrics across all runs for an experiment."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*, r.run_number FROM metrics m
                JOIN runs r ON r.id = m.run_id
                WHERE r.experiment_id = ?
                ORDER BY r.run_number, m.epoch, m.phase
                """,
                (experiment_id,),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["metrics"] = json.loads(d.pop("metrics_json"))
                result.append(d)
            return result

    # ------------------------------------------------------------------
    # Predict history
    # ------------------------------------------------------------------
    # The dashboard's predict proxy calls record_prediction() after every
    # successful round-trip so users can browse + replay recent calls per
    # managed server. Heavy fields (audio_b64, video_b64, depth/mask PNG
    # thumbnails) are stripped before storage so the DB doesn't bloat at
    # a few MB per call — we keep enough to render a sidebar row and
    # repopulate the universal input fields.

    # Cap on string size for any request/response field we persist. 32 KB
    # is comfortable for text + tokens + small image previews, but rejects
    # multi-MB audio / video / depth-map blobs.
    _PREDICT_BLOB_CAP = 32 * 1024

    @staticmethod
    def _shrink_predict_request(req: Dict[str, Any]) -> Dict[str, Any]:
        """Strip / truncate fields that would bloat the DB.

        Audio / video b64 blobs and embedded PNG thumbnails are replaced
        with sentinel ``"<stripped:audio_b64 N bytes>"`` strings — the
        sidebar shows that an audio file was sent without storing
        megabytes per row. Small image previews (<32 KB) are kept so
        image-classification replays still surface a thumbnail.
        """
        if not isinstance(req, dict):
            return {}
        out: Dict[str, Any] = {}
        cap = ExperimentTracker._PREDICT_BLOB_CAP
        for k, v in req.items():
            if k in ("audio_b64", "video_b64"):
                if isinstance(v, str) and v:
                    out[k] = f"<stripped:{k} {len(v)} bytes>"
                else:
                    out[k] = None
                continue
            if k == "image_b64":
                if isinstance(v, str) and len(v) > cap:
                    out[k] = f"<stripped:{k} {len(v)} bytes>"
                else:
                    out[k] = v
                continue
            # `messages` may carry image / audio parts — recurse one level.
            if k == "messages" and isinstance(v, list):
                out[k] = [
                    ExperimentTracker._shrink_message(m) for m in v
                ]
                continue
            # Anything else: keep the value verbatim. Generic large
            # strings get a length-only sentinel.
            if isinstance(v, str) and len(v) > cap:
                out[k] = f"<stripped:{k} {len(v)} bytes>"
            else:
                out[k] = v
        return out

    @staticmethod
    def _shrink_message(m: Any) -> Any:
        """Strip image_b64 / audio_b64 from a ChatMessage dict."""
        if not isinstance(m, dict):
            return m
        out = dict(m)
        content = out.get("content")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if not isinstance(part, dict):
                    new_parts.append(part)
                    continue
                p2 = dict(part)
                for key in ("image_b64", "audio_b64"):
                    if isinstance(p2.get(key), str) and p2[key]:
                        p2[key] = f"<stripped:{key} {len(p2[key])} bytes>"
                new_parts.append(p2)
            out["content"] = new_parts
        return out

    @staticmethod
    def _shrink_predict_response(resp: Dict[str, Any]) -> Dict[str, Any]:
        """Strip the heavy parts of a response — embedded depth/mask
        PNGs in metadata blow past 100 KB / row. We keep the label,
        probability, and shape info so the sidebar can render a summary
        and replay the request."""
        if not isinstance(resp, dict):
            return {}
        cap = ExperimentTracker._PREDICT_BLOB_CAP
        out = {
            "predictions": [],
            "result_kind": resp.get("result_kind", "logits"),
            "latency_ms": resp.get("latency_ms"),
            "wall_latency_ms": resp.get("wall_latency_ms"),
            "model_type": resp.get("model_type"),
        }
        preds = resp.get("predictions") or []
        # Flatten the optional nested-list shape, take up to 10 entries.
        flat: List[Dict[str, Any]] = []
        if preds and isinstance(preds[0], list):
            flat = preds[0]
        elif isinstance(preds, list):
            flat = preds
        for p in flat[:10]:
            if not isinstance(p, dict):
                continue
            md = (p.get("metadata") or {}).copy() if isinstance(p.get("metadata"), dict) else None
            if md is not None:
                for key in ("image_b64",):   # depth/mask thumbnails
                    if isinstance(md.get(key), str) and len(md[key]) > cap:
                        md[key] = f"<stripped:{key} {len(md[key])} bytes>"
            out["predictions"].append({
                "label": p.get("label"),
                "class_name": p.get("class_name"),
                "probability": p.get("probability"),
                "score": p.get("score"),
                "metadata": md,
            })
        return out

    def record_prediction(
        self,
        server_id: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        *,
        server_url: Optional[str] = None,
        server_name: Optional[str] = None,
        pipeline_task: Optional[str] = None,
    ) -> int:
        """Persist a single predict round-trip to the history table.

        Idempotent in that it always inserts a new row — call sites
        run after the proxy has confirmed a successful response, so
        we never log 4xx/5xx as a "successful" prediction. The shrink
        helpers cap stored size so a session with 50 audio predicts
        doesn't bloat the DB past a few hundred KB.
        """
        now = datetime.utcnow().isoformat()
        req_min = self._shrink_predict_request(request or {})
        resp_min = self._shrink_predict_response(response or {})
        result_kind = resp_min.get("result_kind") or "logits"
        latency_ms = resp_min.get("latency_ms") or resp_min.get("wall_latency_ms")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO predict_history "
                "(server_id, server_url, server_name, pipeline_task, "
                " request_json, response_json, result_kind, latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (server_id, server_url, server_name, pipeline_task,
                 json.dumps(req_min, default=str), json.dumps(resp_min, default=str),
                 result_kind, latency_ms, now),
            )
            return cur.lastrowid

    def list_predictions(self,
                          server_id: Optional[str] = None,
                          *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent predictions, newest first.

        Optionally filter by ``server_id`` so the Predict tab only sees
        the history for the currently-connected managed server. ``limit``
        caps the response — the sidebar shows the latest 50 by default,
        which is plenty for a session and keeps the JSON payload small.
        """
        sql = "SELECT * FROM predict_history"
        params: List[Any] = []
        if server_id:
            sql += " WHERE server_id = ?"
            params.append(server_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["request"] = json.loads(d.pop("request_json") or "{}")
            except Exception:
                d["request"] = {}
            try:
                d["response"] = json.loads(d.pop("response_json") or "{}")
            except Exception:
                d["response"] = {}
            out.append(d)
        return out

    def delete_prediction(self, prediction_id: int) -> bool:
        """Remove one entry — supports the per-row ✕ button in the
        sidebar."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM predict_history WHERE id = ?", (prediction_id,),
            )
            return cur.rowcount > 0

    def clear_predictions(self, server_id: Optional[str] = None) -> int:
        """Clear history. With no ``server_id``, drops everything (the
        "Clear all" button); with a server_id, drops just that server's
        rows (the per-server "Clear history" button)."""
        with self._connect() as conn:
            if server_id:
                cur = conn.execute(
                    "DELETE FROM predict_history WHERE server_id = ?",
                    (server_id,),
                )
            else:
                cur = conn.execute("DELETE FROM predict_history")
            return cur.rowcount
