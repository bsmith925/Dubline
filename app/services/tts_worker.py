from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

from app.config import settings
from app.services.audio_fit import active_seconds, fit_audio
from app.services.tts import IndexTTSEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest["items"]
    engine = IndexTTSEngine(manifest["engine"])
    for index, item in enumerate(items):
        raw = Path(item["raw"])
        fitted = Path(item["fitted"])
        if not raw.exists():
            vector = item.get("emotion_vector")
            emotion_audio = Path(item["emotion_audio"]) if item.get("emotion_audio") else None
            engine.synthesize(
                item["text"], Path(item["reference"]), raw, float(item["target"]),
                vector, float(item["emotion_strength"]), emotion_audio, item.get("language", "EN"),
                bool(item.get("use_random", False)),
            )
        target = float(item["target"])
        actual = duration(raw)
        active = active_seconds(raw)
        attempts = 1
        initial_error = max(0.0, active / target - 1.0)
        if initial_error > 0.08:
            retry = raw.with_name(raw.stem + ".retry.wav")
            corrected_target = target * target / max(active, 0.1)
            engine.synthesize(
                item["text"], Path(item["reference"]), retry, corrected_target,
                item.get("emotion_vector"), float(item["emotion_strength"]),
                Path(item["emotion_audio"]) if item.get("emotion_audio") else None,
                item.get("language", "EN"), bool(item.get("use_random", False)),
            )
            attempts = 2
            retry_duration = duration(retry); retry_active = active_seconds(retry)
            if max(0.0, retry_active / target - 1.0) < initial_error:
                os.replace(retry, raw)
                actual, active = retry_duration, retry_active
            else:
                retry.unlink(missing_ok=True)
        fitted_valid = False
        if fitted.exists():
            info = sf.info(fitted)
            fitted_valid = info.samplerate == 24_000 and abs(info.frames / info.samplerate - target) <= .002
            if not fitted_valid:
                fitted.unlink(missing_ok=True)
        metrics = fit_audio(raw, fitted, target) if not fitted_valid else {
            "active_duration": round(active, 4), "active_fill_percent": round(active / target * 100, 2),
            "padding_ms": 0.0, "phrase_count": 1, "stretch_percent": round(max(0, active / target - 1) * 100, 2),
        }
        print(json.dumps({"progress": (index + 1) / max(1, len(items)), "index": index,
                          "cue_index": int(item.get("cue_index", index)),
                          "raw_duration": round(actual, 4), **metrics, "attempts": attempts}), flush=True)
        if engine.mode != "preview":
            time.sleep(settings.dub_gpu_line_cooldown_seconds)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return max(0.01, float(result.stdout.strip()))


if __name__ == "__main__":
    main()
