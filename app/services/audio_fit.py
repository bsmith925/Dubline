from __future__ import annotations

"""Timing-fit helpers shared by every TTS worker.

Kept free of ``app.config`` and other pydantic-dependent imports on purpose:
``qwen_tts_worker`` runs inside the isolated Qwen-TTS virtual environment,
which only has numpy, soundfile, torch and ffmpeg available.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def activity(audio: np.ndarray, rate: int) -> tuple[int, int, list[tuple[int, int]]]:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    frame = max(1, round(rate * .01)); count = len(mono) // frame
    if count == 0:
        return 0, len(mono), [(0, len(mono))]
    levels = np.sqrt(np.mean(mono[:count * frame].reshape(count, frame) ** 2, axis=1) + 1e-12)
    threshold = max(10 ** (-48 / 20), float(np.percentile(levels, 90)) * .09)
    active = levels >= threshold
    indices = np.flatnonzero(active)
    if not len(indices):
        return 0, len(mono), [(0, len(mono))]
    start = max(0, int(indices[0] * frame - rate * .02))
    end = min(len(mono), int((indices[-1] + 1) * frame + rate * .035))
    runs: list[tuple[int, int]] = []
    run_start = int(indices[0])
    previous = int(indices[0])
    max_gap = round(.18 / .01)
    for value in map(int, indices[1:]):
        if value - previous > max_gap:
            runs.append((max(start, run_start * frame), min(end, (previous + 1) * frame)))
            run_start = value
        previous = value
    runs.append((max(start, run_start * frame), min(end, (previous + 1) * frame)))
    return start, end, runs


def active_seconds(path: Path) -> float:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    _, _, runs = activity(audio, rate)
    return sum(end - start for start, end in runs) / rate


def stretch_phrase(values: np.ndarray, rate: int, tempo: float, folder: Path, index: int) -> np.ndarray:
    if abs(tempo - 1.0) <= 0.005 or len(values) < round(rate * .08):
        return values.copy()
    source = folder / f"phrase-{index:03d}.wav"; output = folder / f"phrase-{index:03d}-fit.wav"
    sf.write(source, values, rate, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-af",
                    f"rubberband=tempo={tempo:.8f}", "-ar", str(rate), "-ac", "1",
                    "-c:a", "pcm_s16le", str(output)], check=True, capture_output=True)
    result, _ = sf.read(output, dtype="float32", always_2d=True)
    return result.mean(axis=1)


MAX_SLOWDOWN = 1.08  # gentle; beyond this a slowed voice sounds drugged


def max_slowdown() -> float:
    # EXP-TIMING-008: the UTMOS dose-response (2026-08-25) measured ~-1 MOS at 8 %
    # slowdown (and ~-0.9 already at 5 %) while padding costs nothing, so "gentle" was
    # wrong. 1.0 disables slowdown entirely: the take keeps its natural rate and the
    # spare span becomes padding. Env override so experiments can flip it per run;
    # kept off app.config on purpose (this module runs in the isolated TTS venvs).
    try:
        return float(os.environ.get("DUB_MAX_SLOWDOWN", "") or MAX_SLOWDOWN)
    except ValueError:
        return MAX_SLOWDOWN


def fit_audio(source: Path, output: Path, target: float, anchors: list[list[float]] | None = None) -> dict:
    """Fit a take to ``target`` seconds.

    ``anchors`` (optional): the source's own speech runs, in seconds relative to the
    take start. When given and the take is shorter than the span, phrases are placed
    onto successive runs so the dub speaks where the actor's mouth was moving,
    rather than being front-loaded with evenly widened pauses.
    """
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1); start, end, runs = activity(audio, rate)
    target_frames = max(1, round(target * rate))
    active_frames = sum(run_end - run_start for run_start, run_end in runs)
    active_duration = active_frames / rate
    trimmed = mono[start:end]
    padding_ms = 0.0; tempo_correction = 0.0; slowdown = 0.0
    if len(trimmed) <= target_frames:
        # Short delivery: first slow the voice very gently (inaudible below ~8%),
        # then spread the phrases across the span by widening the pauses between
        # them so speech follows the actor's mouth activity through the line.
        relative_runs = [(a - start, b - start) for a, b in runs]
        slow = min(max_slowdown(), target_frames / max(1, len(trimmed)))
        if slow > 1.01:
            with tempfile.TemporaryDirectory(prefix="dubline-slow-") as temp:
                trimmed = stretch_phrase(trimmed, rate, 1.0 / slow, Path(temp), 0)
            scale = len(trimmed) / max(1, (relative_runs[-1][1] if relative_runs else len(trimmed)))
            relative_runs = [(int(a * scale), int(b * scale)) for a, b in relative_runs]
            slowdown = (slow - 1.0) * 100
        result = np.zeros(target_frames, dtype=np.float32)
        spare = max(0, target_frames - len(trimmed))
        lead = min(round(rate * .025), spare)
        placed = False
        if anchors and relative_runs and spare > lead:
            # Greedy placement: each phrase starts at the next source run that begins
            # at or after the cursor; phrases never overlap and never run past the span.
            anchor_starts = sorted(max(0.0, float(a[0])) for a in anchors if len(a) == 2)
            cursor = lead
            for run_index, (run_start, run_end) in enumerate(relative_runs):
                piece = trimmed[run_start:run_end]
                natural_gap = (run_start - relative_runs[run_index - 1][1]) if run_index else 0
                candidates = [round(a * rate) for a in anchor_starts if round(a * rate) >= cursor + natural_gap]
                start_at = candidates[0] if candidates else cursor + natural_gap
                if start_at + len(piece) > target_frames:           # keep everything: fall back to tight packing
                    start_at = max(cursor + natural_gap, target_frames - len(piece))
                    if start_at + len(piece) > target_frames:
                        start_at = cursor
                result[start_at:start_at + len(piece)] = piece[: max(0, target_frames - start_at)]
                cursor = start_at + len(piece)
            placed = True
        if not placed and len(relative_runs) >= 2 and spare > lead:
            # Cap how far any single pause may grow so phrasing stays natural.
            max_extra_per_gap = round(rate * 1.2)
            gaps = len(relative_runs) - 1
            extra_per_gap = min(max_extra_per_gap, (spare - lead) // gaps)
            cursor = lead
            for run_index, (run_start, run_end) in enumerate(relative_runs):
                if run_index:
                    cursor += (run_start - relative_runs[run_index - 1][1]) + extra_per_gap
                piece = trimmed[run_start:run_end]
                result[cursor:cursor + len(piece)] = piece
                cursor += len(piece)
        elif not placed:
            result[lead:lead + len(trimmed)] = trimmed
        padding_ms = spare / rate * 1000
    else:
        relative_runs = [(a - start, b - start) for a, b in runs]
        internal_gaps = sum(relative_runs[i][0] - relative_runs[i - 1][1]
                            for i in range(1, len(relative_runs)))
        allowed_gaps = min(internal_gaps, round(target_frames * .35))
        available_voice = max(round(rate * .15), target_frames - allowed_gaps)
        tempo = max(1.0, active_frames / available_voice)
        tempo_correction = (tempo - 1.0) * 100
        pieces: list[np.ndarray] = []
        with tempfile.TemporaryDirectory(prefix="dubline-phrases-") as temp:
            scratch = Path(temp)
            for index, (run_start, run_end) in enumerate(relative_runs):
                if index:
                    original_gap = run_start - relative_runs[index - 1][1]
                    gap = min(original_gap, max(0, allowed_gaps - sum(len(x) for x in pieces if np.max(np.abs(x)) == 0)))
                    if gap:
                        pieces.append(np.zeros(gap, dtype=np.float32))
                pieces.append(stretch_phrase(trimmed[run_start:run_end], rate, tempo, scratch, index))
        result = np.concatenate(pieces) if pieces else trimmed
        result = result[:target_frames]
        if len(result) < target_frames:
            result = np.pad(result, (0, target_frames - len(result)))
    fade = min(round(rate * .015), len(result) // 4)
    if fade:
        result[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        result[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    output_rate = 24_000
    if rate != output_rate:
        import torch
        import torchaudio
        result = torchaudio.functional.resample(torch.from_numpy(result.copy()), rate, output_rate).numpy()
        wanted = max(1, round(target * output_rate))
        result = result[:wanted] if len(result) >= wanted else np.pad(result, (0, wanted - len(result)))
    sf.write(output, np.clip(result * .96, -.97, .97), output_rate, subtype="PCM_16")
    return {"active_duration": round(active_duration, 4), "active_fill_percent": round(active_duration / target * 100, 2),
            "padding_ms": round(padding_ms, 1), "phrase_count": len(runs),
            "stretch_percent": round(tempo_correction, 2), "slowdown_percent": round(slowdown, 2),
            "placed_on_anchors": bool(anchors) and padding_ms > 0}
