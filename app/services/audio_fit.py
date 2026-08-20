from __future__ import annotations

"""Timing-fit helpers shared by every TTS worker.

Kept free of ``app.config`` and other pydantic-dependent imports on purpose:
``qwen_tts_worker`` runs inside the isolated Qwen-TTS virtual environment,
which only has numpy, soundfile, torch and ffmpeg available.
"""

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
    if tempo <= 1.005 or len(values) < round(rate * .08):
        return values.copy()
    source = folder / f"phrase-{index:03d}.wav"; output = folder / f"phrase-{index:03d}-fit.wav"
    sf.write(source, values, rate, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-af",
                    f"rubberband=tempo={tempo:.8f}", "-ar", str(rate), "-ac", "1",
                    "-c:a", "pcm_s16le", str(output)], check=True, capture_output=True)
    result, _ = sf.read(output, dtype="float32", always_2d=True)
    return result.mean(axis=1)


def fit_audio(source: Path, output: Path, target: float) -> dict:
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1); start, end, runs = activity(audio, rate)
    target_frames = max(1, round(target * rate))
    active_frames = sum(run_end - run_start for run_start, run_end in runs)
    active_duration = active_frames / rate
    trimmed = mono[start:end]
    padding_ms = 0.0; tempo_correction = 0.0
    if len(trimmed) <= target_frames:
        # Natural short delivery: retain its cadence and add silence, never slow it.
        result = np.zeros(target_frames, dtype=np.float32)
        lead = min(round(rate * .025), max(0, target_frames - len(trimmed)))
        result[lead:lead + len(trimmed)] = trimmed
        padding_ms = (target_frames - len(trimmed)) / rate * 1000
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
            "stretch_percent": round(tempo_correction, 2)}
