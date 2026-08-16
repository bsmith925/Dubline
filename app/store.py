from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from fastapi import HTTPException


class JobStore:
    """Small durable JSON document store backed by SQLite/WAL."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL NOT NULL, updated REAL NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS job_cues (job_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            # One-time in-place migration from the original monolithic document.
            for row in db.execute("SELECT id, payload FROM jobs").fetchall():
                payload = json.loads(row["payload"])
                if "cues" not in payload:
                    continue
                cues = payload.pop("cues") or []
                payload.setdefault("cue_count", len(cues))
                payload.setdefault("cue_revision", 0)
                db.execute("INSERT OR REPLACE INTO job_cues(job_id, payload) VALUES(?, ?)",
                           (row["id"], json.dumps(cues, ensure_ascii=False)))
                db.execute("UPDATE jobs SET payload = ? WHERE id = ?",
                           (json.dumps(payload, ensure_ascii=False), row["id"]))

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, payload: dict) -> dict:
        now = time.time()
        cues = list(payload.pop("cues", []))
        payload = {
            **payload, "created_at": now, "updated_at": now,
            "cue_count": len(cues), "cue_revision": 0, "log_revision": 0,
        }
        with self.lock, self._connect() as db:
            db.execute("INSERT INTO jobs(id, created, updated, payload) VALUES(?, ?, ?, ?)",
                       (payload["id"], now, now, json.dumps(payload, ensure_ascii=False)))
            db.execute("INSERT INTO job_cues(job_id, payload) VALUES(?, ?)",
                       (payload["id"], json.dumps(cues, ensure_ascii=False)))
        return {**payload, "cues": cues}

    def get(self, job_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT jobs.payload, job_cues.payload AS cues FROM jobs "
                "LEFT JOIN job_cues ON job_cues.job_id = jobs.id WHERE jobs.id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload"])
        payload["cues"] = json.loads(row["cues"]) if row["cues"] else []
        return payload

    def get_or_404(self, job_id: str) -> dict:
        job = self.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    def update(self, job_id: str, **values) -> dict:
        with self.lock, self._connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            payload = json.loads(row["payload"])
            cues = values.pop("cues", None)
            if cues is not None:
                values["cue_count"] = len(cues)
                values["cue_revision"] = int(payload.get("cue_revision", 0)) + 1
            if "logs" in values:
                values["log_revision"] = int(payload.get("log_revision", 0)) + 1
            payload.update(values)
            payload["updated_at"] = time.time()
            db.execute("UPDATE jobs SET updated = ?, payload = ? WHERE id = ?",
                       (payload["updated_at"], json.dumps(payload, ensure_ascii=False), job_id))
            if cues is not None:
                db.execute("INSERT OR REPLACE INTO job_cues(job_id, payload) VALUES(?, ?)",
                           (job_id, json.dumps(cues, ensure_ascii=False)))
        return {**payload, **({"cues": cues} if cues is not None else {})}

    def append_log(self, job_id: str, message: str) -> dict:
        with self.lock, self._connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            payload = json.loads(row["payload"])
            logs = list(payload.get("logs", []))[-79:] + [message]
            payload["logs"] = logs
            payload["log_revision"] = int(payload.get("log_revision", 0)) + 1
            payload["updated_at"] = time.time()
            db.execute("UPDATE jobs SET updated = ?, payload = ? WHERE id = ?",
                       (payload["updated_at"], json.dumps(payload, ensure_ascii=False), job_id))
        return payload

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT jobs.payload, job_cues.payload AS cues FROM jobs "
                "LEFT JOIN job_cues ON job_cues.job_id = jobs.id ORDER BY created DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{**json.loads(row["payload"]),
                 "cues": json.loads(row["cues"]) if row["cues"] else []} for row in rows]

    def list_summaries(self, limit: int = 50) -> list[dict]:
        """Return lightweight polling records without loading cue and log documents."""
        fields = (
            "id", "filename", "status", "stage", "progress", "created_at", "updated_at",
            "processing_started_at", "active_run_started_at", "active_processing_seconds",
            "cue_count", "cue_revision", "log_revision", "current_cue", "output_size",
            "error", "control", "options", "media", "qc", "eta", "throughput",
            "media_selection",
        )
        select = ", ".join(f"json_extract(payload, '$.{field}') AS {field}" for field in fields)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT {select} FROM jobs ORDER BY created DESC LIMIT ?", (limit,)
            ).fetchall()
        summaries = []
        for row in rows:
            item = dict(row)
            for field in ("options", "media", "qc", "media_selection", "eta", "throughput"):
                if isinstance(item.get(field), str):
                    item[field] = json.loads(item[field])
            summaries.append(item)
        return summaries

    def delete(self, job_id: str) -> None:
        with self.lock, self._connect() as db:
            db.execute("DELETE FROM job_cues WHERE job_id = ?", (job_id,))
            db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def recover_interrupted(self) -> list[str]:
        pending = []
        for job in self.list_summaries(1000):
            if job.get("status") in {"queued", "processing"}:
                if job.get("control") == "cancel":
                    self.update(job["id"], status="cancelled", stage="Cancelled", control=None)
                elif job.get("control") == "pause":
                    self.update(job["id"], status="paused", stage="Paused before processing", control=None)
                else:
                    self.update(job["id"], status="queued", stage="Recovered after restart", control=None)
                    pending.append(job["id"])
        return pending
