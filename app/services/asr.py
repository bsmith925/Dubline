from __future__ import annotations

import json
import os
import subprocess
import difflib
import re
from pathlib import Path
from typing import Callable

from app.services.subprocess_control import controlled_lines, terminate_process


ALIGNER_LANGUAGES = {"Chinese", "English", "Cantonese", "French", "German", "Italian",
                     "Japanese", "Korean", "Portuguese", "Russian", "Spanish"}


def _checkpoint_ready(folder: Path) -> bool:
    return ((folder / "model.safetensors").is_file()
            or (folder / "model.safetensors.index.json").is_file())


def _run_aligned_worker(windows: list[dict], folder: Path, language: str | None,
                        asr_model: Path, aligner_model: Path, name: str,
                        progress: Callable[[float, int], None], checkpoint: Callable[[], None]) -> list[dict]:
    runtime = Path(os.getenv("QWEN_ASR_RUNTIME", "vendor/pyannote-env/Scripts/python.exe")).resolve()
    manifest = folder / f"qwen-asr-{name}-manifest.json"
    output = folder / f"asr-dialogue-map-{name}.json"
    manifest.write_text(json.dumps({
        "windows": [{**window, "language": language,
                     "window_id": int(window.get("window_id", index))}
                    for index, window in enumerate(windows)],
        "asr_model": str(asr_model), "aligner_model": str(aligner_model),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    process = subprocess.Popen(
        [str(runtime), "-m", "app.services.qwen_asr_worker", "--manifest", str(manifest),
         "--output", str(output)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-20:]
            try:
                event = json.loads(line)
                progress(float(event["progress"]), int(event["window"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        code = process.wait()
    except BaseException:
        terminate_process(process); raise
    if code != 0 or not output.is_file():
        raise RuntimeError("Qwen3 aligned ASR worker failed: " + "\n".join(tail[-10:]))
    return json.loads(output.read_text(encoding="utf-8"))


def _agreement(first: list[dict], second: list[dict]) -> float:
    def text(cues: list[dict]) -> str:
        return re.sub(r"\W+", "", "".join(str(c.get("source", "")) for c in cues).lower())
    left, right = text(first), text(second)
    return difflib.SequenceMatcher(None, left, right).ratio() if left and right else 0.0


def transcribe_aligned(windows: list[dict], folder: Path, language: str | None,
                       progress: Callable[[float, int], None], checkpoint: Callable[[], None]) -> list[dict] | None:
    runtime = Path(os.getenv("QWEN_ASR_RUNTIME", "vendor/pyannote-env/Scripts/python.exe")).resolve()
    asr_model = Path(os.getenv("QWEN_ASR_MODEL", "vendor/qwen3-asr-0.6b-qwen")).resolve()
    escalation_model = Path(os.getenv(
        "QWEN_ASR_ESCALATION_MODEL", "vendor/qwen3-asr-1.7b-qwen")).resolve()
    aligner_model = Path(os.getenv("QWEN_ALIGNER_MODEL", "vendor/qwen3-forced-aligner-0.6b-qwen")).resolve()
    if not runtime.is_file() or not (asr_model / "model.safetensors").is_file() \
            or not (aligner_model / "model.safetensors").is_file():
        return None
    if language and language not in ALIGNER_LANGUAGES:
        return None
    indexed = [{**window, "window_id": index} for index, window in enumerate(windows)]
    primary = _run_aligned_worker(indexed, folder, language, asr_model, aligner_model,
                                  "primary", progress, checkpoint)
    by_window: dict[int, list[dict]] = {}
    for cue in primary:
        by_window.setdefault(int(cue.get("asr_window", -1)), []).append(cue)
    low_ids = {index for index in range(len(indexed)) if not by_window.get(index) or
               min(float(c.get("alignment_confidence", 0)) for c in by_window[index]) < .82}
    if not low_ids or not _checkpoint_ready(escalation_model):
        (folder / "asr-dialogue-map.json").write_text(
            json.dumps(primary, ensure_ascii=False, indent=2), encoding="utf-8")
        return primary
    try:
        escalated = _run_aligned_worker([indexed[index] for index in sorted(low_ids)], folder, language,
                                        escalation_model, aligner_model, "escalated", progress, checkpoint)
    except RuntimeError as exc:
        for cue in primary:
            if int(cue.get("asr_window", -1)) in low_ids:
                cue["asr_escalation_error"] = str(exc)[-500:]
        (folder / "asr-dialogue-map.json").write_text(
            json.dumps(primary, ensure_ascii=False, indent=2), encoding="utf-8")
        return primary
    high_by_window: dict[int, list[dict]] = {}
    for cue in escalated:
        high_by_window.setdefault(int(cue.get("asr_window", -1)), []).append(cue)
    merged = []
    for index in range(len(indexed)):
        base = by_window.get(index, [])
        higher = high_by_window.get(index, [])
        if index not in low_ids or not higher:
            merged.extend(base); continue
        agreement = _agreement(base, higher)
        base_alignment = sum(float(c.get("alignment_confidence", 0)) for c in base) / max(1, len(base))
        higher_alignment = sum(float(c.get("alignment_confidence", 0)) for c in higher) / max(1, len(higher))
        use_higher = (higher_alignment >= base_alignment + .03
                      or (agreement >= .72 and higher_alignment >= base_alignment - .05))
        selected = higher if use_higher else base
        for cue in selected:
            alignment = float(cue.get("alignment_confidence", 0))
            cue["model_agreement"] = round(agreement, 3)
            cue["transcription_confidence"] = round(min(1.0,
                .46 * alignment + .34 * agreement + .20 * .92), 3)
            cue["confidence_source"] = "alignment validity + 0.6B/1.7B agreement"
            cue["asr_escalation_consulted"] = True
            cue["asr_escalated"] = use_higher
        merged.extend(selected)
    merged.sort(key=lambda cue: (float(cue["start"]), float(cue["end"])))
    (folder / "asr-dialogue-map.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged
