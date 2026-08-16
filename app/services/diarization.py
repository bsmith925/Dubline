from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from app.services.subprocess_control import controlled_lines, terminate_process


def diarization_runtime_settings(environment: dict[str, str] | None = None) -> tuple[str, int, int]:
    environment = environment or os.environ
    device = environment.get("DUB_DIARIZATION_DEVICE", "cuda").strip().lower()
    if device not in {"cpu", "cuda"}:
        device = "cuda"
    try:
        batch_size = max(1, min(8, int(environment.get("DUB_DIARIZATION_BATCH_SIZE", "4"))))
    except ValueError:
        batch_size = 4
    try:
        cpu_threads = max(1, min(12, int(environment.get("DUB_DIARIZATION_CPU_THREADS", "8"))))
    except ValueError:
        cpu_threads = 8
    return device, batch_size, cpu_threads


def diarize(audio: Path, folder: Path, progress: Callable[[float], None], checkpoint: Callable[[], None]) -> dict | None:
    model = Path(os.getenv("PYANNOTE_MODEL", "vendor/pyannote-community-1")).resolve()
    if not (model / "config.yaml").is_file():
        return None
    runtime = Path(os.getenv("PYANNOTE_RUNTIME", "vendor/pyannote-env/Scripts/python.exe")).resolve()
    if not runtime.is_file():
        raise RuntimeError("The isolated pyannote runtime is missing")
    output = folder / "speaker-diarization.json"
    if output.is_file():
        try:
            cached = json.loads(output.read_text(encoding="utf-8"))
            if int(cached.get("version", -1)) == 2:
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        output.unlink(missing_ok=True)
    device, batch_size, cpu_threads = diarization_runtime_settings()
    process = subprocess.Popen(
        [str(runtime), "-m", "app.services.diarization_worker", "--audio", str(audio),
         "--model", str(model), "--output", str(output), "--device", device,
         "--batch-size", str(batch_size), "--cpu-threads", str(cpu_threads)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-20:]
            try:
                progress(float(json.loads(line)["progress"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        code = process.wait()
    except BaseException:
        terminate_process(process); raise
    if code != 0 or not output.is_file():
        raise RuntimeError("Speaker diarization worker failed: " + "\n".join(tail[-10:]))
    return json.loads(output.read_text(encoding="utf-8"))


def assign_diarized_speakers(cues: list[dict], result: dict) -> None:
    exclusive = result.get("exclusive_diarization") or result.get("diarization") or []
    regular = result.get("diarization") or exclusive
    # Keep IDs stable across selected sections by numbering every full-context
    # cluster at its first occurrence, not merely those present in this cue list.
    label_order: dict[str, int] = {}
    for turn in sorted(exclusive, key=lambda item: (float(item["start"]), float(item["end"]))):
        label = str(turn["speaker"])
        label_order.setdefault(label, len(label_order) + 1)
    for cue in cues:
        start, end = float(cue["start"]), float(cue["end"])
        duration = max(0.05, end - start)
        scores: dict[str, float] = {}
        score_turns = regular if cue.get("simultaneous_card") else exclusive
        for turn in score_turns:
            amount = max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"])))
            scores[turn["speaker"]] = scores.get(turn["speaker"], 0.0) + amount
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        overlaps = {turn["speaker"] for turn in regular
                    if max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"]))) > 0.08}
        cue["overlapping_speech"] = len(overlaps) > 1
        if not ranked or ranked[0][1] / duration < 0.3:
            cue["speaker_id"] = 0; cue["speaker"] = "Uncertain voice"; cue["speaker_confidence"] = 0.0
            cue["speaker_assignment"] = "uncertain"
            continue
        requested_rank = int(cue.get("card_speaker_index", 0)) if cue.get("simultaneous_card") else 0
        selected_rank = min(requested_rank, len(ranked) - 1)
        label, amount = ranked[selected_rank]
        cue["speaker_id"] = label_order[label]
        cue["speaker"] = f"Voice {label_order[label]}"
        cue["speaker_confidence"] = round(min(0.98, amount / duration) *
                                           (0.62 if cue["overlapping_speech"] else 1.0), 3)
        cue["diarization_label"] = label
        cue["speaker_assignment"] = "confident" if cue["speaker_confidence"] >= 0.62 else "tentative"
