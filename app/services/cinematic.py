from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _worker(command: list[str], outputs: list[Path], label: str,
            progress: Callable[[float], None], checkpoint: Callable[[], None]) -> None:
    """Run a model worker with a drained stream and unconditional child cleanup."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        assert process.stdout is not None
        lines: queue.Queue[str | None] = queue.Queue()

        def drain() -> None:
            assert process.stdout is not None
            for value in process.stdout:
                lines.put(value)
            lines.put(None)

        reader = threading.Thread(target=drain, name=f"{label}-output", daemon=True)
        reader.start()
        while True:
            checkpoint()
            try:
                line = lines.get(timeout=.25)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if line is None:
                break
            tail.append(line.rstrip()); tail = tail[-20:]
            try:
                event = json.loads(line)
                if "progress" in event:
                    progress(float(event["progress"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        code = process.wait()
        reader.join(timeout=2)
        if code != 0:
            raise RuntimeError(f"{label} failed: " + ("\n".join(tail[-12:]) or f"exit code {code}"))
        missing = [path.name for path in outputs if not path.is_file()]
        if missing:
            raise RuntimeError(f"{label} did not produce: {', '.join(missing)}")
    except BaseException:
        _terminate(process)
        for path in outputs:
            path.unlink(missing_ok=True)
        raise
    finally:
        _terminate(process)
        if process.stdout:
            process.stdout.close()


def separate_cinematic_audio(film_mix: Path, dialogue: Path, background: Path,
                             progress: Callable[[float], None], checkpoint: Callable[[], None]) -> None:
    repo = Path(os.getenv("BANDIT_REPO", "vendor/bandit-v2")).resolve()
    weights = Path(os.getenv("BANDIT_CHECKPOINT", repo / "checkpoints" / "checkpoint-multi.ckpt")).resolve()
    if not weights.is_file():
        raise RuntimeError("The multilingual Bandit v2 cinematic separation checkpoint is missing")
    _worker(
        [sys.executable, "-m", "app.services.separator_worker", "--input", str(film_mix),
         "--dialogue", str(dialogue), "--background", str(background),
         "--checkpoint", str(weights), "--repo", str(repo), "--batch-size", "2"],
        [dialogue, background], "Cinematic separation", progress, checkpoint,
    )


def recover_vocals(film_mix: Path, vocals: Path, progress: Callable[[float], None],
                   checkpoint: Callable[[], None]) -> None:
    """Run music-trained HTDemucs as an independent safety net for missed film dialogue."""
    _worker(
        [sys.executable, "-m", "app.services.vocal_worker", "--input", str(film_mix),
         "--output", str(vocals), "--model", "htdemucs"],
        [vocals], "Vocal recovery", progress, checkpoint,
    )


def recover_roformer(film_mix: Path, vocals: Path, progress: Callable[[float], None],
                     checkpoint: Callable[[], None]) -> None:
    model_dir = Path(os.getenv(
        "ROFORMER_MODEL_DIR", "vendor/melband-roformer/melband-roformer-kim-vocals")).resolve()
    weights = model_dir / "MelBandRoformer.ckpt"
    config = model_dir / "config_vocals_mel_band_roformer.yaml"
    if not weights.is_file() or not config.is_file():
        raise RuntimeError("The local MelBand-RoFormer recovery model is missing")
    _worker(
        [sys.executable, "-m", "app.services.roformer_worker", "--input", str(film_mix),
         "--output", str(vocals), "--checkpoint", str(weights), "--config", str(config)],
        [vocals], "MelBand-RoFormer recovery", progress, checkpoint,
    )
