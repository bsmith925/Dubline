#!/usr/bin/env python3
"""Dose-response of time-fitting damage: naturalness (UTMOS) vs applied manipulation.

Takes clean raw TTS takes (shipped with |stretch| and slowdown ≈ 0), re-fits each at a
ladder of requested manipulation levels using the production fitter (audio_fit.fit_audio),
and UTMOS-scores every output. The actual applied stretch/slowdown is read back from the
fitter's own metrics, so the curve is plotted against what was really done to the audio.
CPU only. Informs EXP-TIMING-008's cap.

Usage:
  python stretch_mos_curve.py JOB_DIR [JOB_DIR ...] [--max-takes 40] [--json out.json]

Levels: negative = lengthen request (slowdown, capped at 8 %, then padding);
positive = shorten request (phrase-wise rubberband compression, gap clipping).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

try:
    from app.services.audio_fit import fit_audio
except ImportError:  # bare copy next to audio_fit.py (server /tmp)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audio_fit import fit_audio  # type: ignore

LEVELS = [-15, -8, -4, 0, 5, 10, 20, 35, 55]  # % change requested on the trimmed take

_utmos = None


def utmos(wav: Path) -> float | None:
    global _utmos
    try:
        import librosa
        import torch
        if _utmos is None:
            _utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        wave, _ = librosa.load(str(wav), sr=16000, mono=True)
        if len(wave) < 1600:
            return None
        with torch.no_grad():
            return round(float(_utmos(torch.from_numpy(wave).unsqueeze(0), 16000)[0]), 3)
    except Exception as error:
        print(f"  UTMOS failed on {wav}: {error}", file=sys.stderr)
        return None


def clean_takes(folder: Path) -> list[tuple[Path, dict]]:
    """Raw takes whose shipped fit needed (almost) no manipulation: their raw audio is
    undamaged, so any manipulation we now apply is the only variable."""
    cues_file = folder / "cues.json"
    if not cues_file.is_file():
        return []
    cues = json.loads(cues_file.read_text(encoding="utf-8"))
    takes = []
    for index, cue in enumerate(cues):
        qc = cue.get("qc") or {}
        raw = folder / "generated" / f"{index + 1:06d}.wav"
        if (not cue.get("nonverbal_filler") and raw.is_file()
                and abs(float(qc.get("stretch_percent") or 0.0)) < 1.0
                and abs(float(qc.get("slowdown_percent") or 0.0)) < 1.0
                and float(qc.get("active_duration") or 0.0) >= 2.0):
            takes.append((raw, {"job": folder.name, "cue": index + 1,
                                "active_duration": qc.get("active_duration")}))
    return takes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="+", type=Path)
    parser.add_argument("--max-takes", type=int, default=40)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    takes: list[tuple[Path, dict]] = []
    for folder in args.jobs:
        takes.extend(clean_takes(folder))
    # Spread evenly across jobs/cues instead of exhausting the first job.
    takes = takes[:: max(1, len(takes) // args.max_takes)][: args.max_takes]
    print(f"{len(takes)} clean takes selected")

    import soundfile as sf
    rows = []
    with tempfile.TemporaryDirectory(prefix="mos-curve-") as temp:
        scratch = Path(temp)
        for take_index, (raw, info) in enumerate(takes):
            audio, rate = sf.read(raw, dtype="float32", always_2d=True)
            base_mos = utmos(raw)
            for level in LEVELS:
                # Trimmed length (speech + internal gaps) scaled by the requested change;
                # the fitter decides how much tempo/slowdown/padding that really takes.
                trimmed_seconds = len(audio) / rate
                target = max(0.3, trimmed_seconds / (1 + level / 100.0))
                fitted = scratch / f"{take_index:03d}-{level:+03d}.wav"
                try:
                    metrics = fit_audio(raw, fitted, target)
                except Exception as error:
                    print(f"  fit failed {raw} @ {level}%: {error}", file=sys.stderr)
                    continue
                rows.append({**info, "requested_percent": level,
                             "applied_stretch": metrics["stretch_percent"],
                             "applied_slowdown": metrics["slowdown_percent"],
                             "padding_ms": metrics["padding_ms"],
                             "raw_mos": base_mos, "fitted_mos": utmos(fitted)})
            print(f"  [{take_index + 1}/{len(takes)}] {info['job']} cue {info['cue']} done")

    by_level: dict[int, list[float]] = {}
    for row in rows:
        if row["fitted_mos"] is not None and row["raw_mos"] is not None:
            by_level.setdefault(row["requested_percent"], []).append(row["fitted_mos"] - row["raw_mos"])
    print("\nrequested % | mean ΔMOS vs raw | n")
    for level in sorted(by_level):
        deltas = by_level[level]
        print(f"{level:+11d} | {statistics.mean(deltas):+.3f} | {len(deltas)}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
