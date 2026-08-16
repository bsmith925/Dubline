from __future__ import annotations

"""Crash-evident supervision for every CUDA phase in the film pipeline."""

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


_lock = threading.RLock()


def parse_nvidia_status(value: str) -> dict:
    fields = [item.strip() for item in value.strip().split(",")]
    if len(fields) < 5:
        raise RuntimeError("NVIDIA health query returned incomplete data")
    return {"free_mb": int(fields[0]), "total_mb": int(fields[1]),
            "temperature_c": int(fields[2]), "utilization": int(fields[3]),
            "power_w": float(fields[4])}


def query_nvidia() -> dict:
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=memory.free,memory.total,temperature.gpu,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8)
    if result.returncode or not result.stdout.strip():
        detail = (result.stdout + result.stderr).strip()[-600:]
        raise RuntimeError("The NVIDIA driver is unavailable" + (f": {detail}" if detail else ""))
    return parse_nvidia_status(result.stdout.splitlines()[0])


def _data_root(folder: Path) -> Path:
    folder = folder.resolve()
    for candidate in (folder, *folder.parents):
        if candidate.name.lower() == "jobs":
            return candidate.parent
    return Path(os.getenv("DUB_WORKDIR", "data")).resolve()


def _state_path(folder: Path) -> Path:
    return _data_root(folder) / "gpu-safety.json"


def gpu_safety_summary(data_root: Path | None = None) -> dict:
    root = (data_root or Path(os.getenv("DUB_WORKDIR", "data"))).resolve()
    state = _read_state(root / "gpu-safety.json")
    return {key: state.get(key) for key in (
        "status", "stage", "last_stage", "started_at", "interrupted_at",
        "last_completed_at", "last_canary_at", "health",
    ) if state.get(key) is not None}


def _boot_token() -> int:
    return int((time.time() - time.monotonic()) // 30)


def _read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_state(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _canary(path: Path, checkpoint: Callable[[], None]) -> None:
    checkpoint()
    _write_state(path, {"status": "canary", "stage": "CUDA safety probe",
                        "boot_token": _boot_token(), "pid": os.getpid(),
                        "started_at": time.time()})
    result = subprocess.run(
        [sys.executable, "-m", "app.services.gpu_canary_worker", "--seconds", "2"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    if result.returncode:
        _write_state(path, {"status": "unsafe", "stage": "CUDA safety probe",
                            "boot_token": _boot_token(), "failed_at": time.time(),
                            "error": (result.stdout + result.stderr)[-1200:]})
        raise RuntimeError("CUDA safety probe failed; the job was stopped before loading a production model")
    health = query_nvidia()
    _write_state(path, {"status": "idle", "boot_token": _boot_token(),
                        "last_canary_at": time.time(), "health": health})


def _cooldown(checkpoint: Callable[[], None], seconds: float = 8.0) -> dict:
    deadline = time.monotonic() + max(0.0, seconds)
    latest = query_nvidia()
    while time.monotonic() < deadline:
        checkpoint()
        # A cool, idle, mostly released device is a safe hand-off boundary.
        if (latest["temperature_c"] <= 72 and latest["utilization"] <= 10
                and latest["free_mb"] >= latest["total_mb"] - 900):
            return latest
        time.sleep(1.0)
        latest = query_nvidia()
    return latest


@contextmanager
def gpu_stage(folder: Path, stage: str, checkpoint: Callable[[], None],
              *, minimum_free_mb: int = 5600) -> Iterator[None]:
    """Record an in-flight stage, validate CUDA, and enforce a cool hand-off.

    If Windows reboots while the body is active, the durable marker remains.
    The next CUDA phase must pass an isolated canary before production resumes.
    """
    path = _state_path(folder)
    with _lock:
        checkpoint()
        state = _read_state(path)
        # The process-wide lock makes any pre-existing active marker stale: no
        # legitimate second GPU stage can be entering concurrently.
        stale = state.get("status") in {"active", "canary", "unsafe"}
        if stale or not state.get("last_canary_at"):
            _canary(path, checkpoint)
            state = _read_state(path)
        health = query_nvidia()
        if health["free_mb"] < minimum_free_mb:
            raise RuntimeError(
                f"CUDA stage '{stage}' needs {minimum_free_mb} MB free, but only "
                f"{health['free_mb']} MB is available. Close other GPU applications and resume."
            )
        started = time.time()
        _write_state(path, {"status": "active", "stage": stage,
                            "boot_token": _boot_token(), "pid": os.getpid(),
                            "started_at": started, "health_before": health,
                            "last_canary_at": state.get("last_canary_at")})
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            if succeeded:
                try:
                    health_after = _cooldown(checkpoint)
                except BaseException:
                    _write_state(path, {"status": "interrupted", "boot_token": _boot_token(),
                                        "stage": stage, "interrupted_at": time.time(),
                                        "last_canary_at": state.get("last_canary_at")})
                    raise
                else:
                    _write_state(path, {"status": "idle", "boot_token": _boot_token(),
                                        "last_stage": stage, "last_completed_at": time.time(),
                                        "last_canary_at": state.get("last_canary_at"),
                                        "health": health_after})
            else:
                # A normal Python exception reaches here; a kernel crash does not.
                _write_state(path, {"status": "interrupted", "boot_token": _boot_token(),
                                    "stage": stage, "interrupted_at": time.time(),
                                    "last_canary_at": state.get("last_canary_at")})
