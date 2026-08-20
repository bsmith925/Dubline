from __future__ import annotations

"""Pre-flight spoken-language identification for the working soundtrack.

Runs before any heavy model so a job can report "Detected: Spanish (99%)"
within seconds, force the ASR language when the evidence is strong, and stop
early when the soundtrack is already in the target language.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

from app.config import settings
from app.services.subprocess_control import controlled_lines, terminate_process

SAMPLE_SECONDS = 30.0
FORCE_LANGUAGE_CONFIDENCE = 0.85
SAME_LANGUAGE_CONFIDENCE = 0.80


def sample_offsets(duration: float, maximum: int = 4) -> list[float]:
    """Spread short probes across the programme, skipping the very start and end."""
    if duration <= SAMPLE_SECONDS * 1.2:
        return [0.0]
    count = max(1, min(maximum, int(duration // 120) + 1))
    usable = duration - SAMPLE_SECONDS
    return [round(usable * (index + 0.5) / count, 3) for index in range(count)]


def detect_language(source: Path, audio_index: int, duration: float, folder: Path,
                    checkpoint: Callable[[], None], model_name: str | None = None) -> dict:
    result_path = folder / "language-id.json"
    if result_path.is_file():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("code"):
                return cached
        except (OSError, ValueError):
            pass
    sample_dir = folder / "language-id"
    sample_dir.mkdir(exist_ok=True)
    samples = []
    for number, offset in enumerate(sample_offsets(duration)):
        checkpoint()
        target = sample_dir / f"sample-{number:02d}.wav"
        if not target.is_file():
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.3f}", "-t", f"{SAMPLE_SECONDS:.0f}",
                "-i", str(source), "-map", f"0:{audio_index}", "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(target),
            ], check=True, capture_output=True)
        samples.append(str(target))
    manifest = folder / "language-id-manifest.json"
    manifest.write_text(json.dumps({
        "samples": samples, "model": model_name or settings.whisper_model,
        "cache": str(settings.whisper_cache_dir),
    }, indent=2), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.language_id_worker", "--manifest", str(manifest),
         "--output", str(result_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-20:]
        code = process.wait()
    except BaseException:
        terminate_process(process); raise
    if code != 0 or not result_path.is_file():
        raise RuntimeError("Language identification worker failed: " + "\n".join(tail[-8:]))
    return json.loads(result_path.read_text(encoding="utf-8"))


def same_language(detection: dict, target_language: str) -> bool:
    return (str(detection.get("language", "")).strip().lower() == str(target_language).strip().lower()
            and float(detection.get("confidence") or 0.0) >= SAME_LANGUAGE_CONFIDENCE)
