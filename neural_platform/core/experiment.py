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
