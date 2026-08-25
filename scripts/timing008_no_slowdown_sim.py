#!/usr/bin/env python3
"""EXP-TIMING-008 offline simulation: never slow the voice down — pad instead.

The dose-response curve (2026-08-25) showed the rubberband slowdown path costs ~1 MOS
at the 8 % cap (the code believed it inaudible) while padding costs nothing. This sim
re-fits every shipped raw take whose fit applied > 1 % slowdown with MAX_SLOWDOWN=1.0
(same target, same fitter otherwise) and UTMOS-scores shipped vs re-fitted — paired,
same take, CPU only. A zero-manipulation refit baseline costs ~0.1 MOS, so the measured
gain slightly UNDER-estimates the true gain.

Usage: python timing008_no_slowdown_sim.py JOBS_ROOT [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

os.environ["DUB_MAX_SLOWDOWN"] = "1.0"  # the one variable

try:
    import app.services.audio_fit as audio_fit
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import audio_fit  # type: ignore

_model = None


def mos(path: Path) -> float | None:
    global _model
    import librosa
    import torch
    if _model is None:
        _model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    wave, _ = librosa.load(str(path), sr=16000, mono=True)
    if len(wave) < 1600:
        return None
    with torch.no_grad():
        return float(_model(torch.from_numpy(wave).unsqueeze(0), 16000)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    pairs = []
    with tempfile.TemporaryDirectory(prefix="t008-") as temp:
        for job in sorted(p for p in args.root.iterdir() if (p / "cues.json").is_file()):
            cues = json.loads((job / "cues.json").read_text(encoding="utf-8"))
            for index, cue in enumerate(cues):
                qc = cue.get("qc") or {}
                raw = job / "generated" / f"{index + 1:06d}.wav"
                fitted = job / "fitted" / f"{index + 1:06d}.wav"
                target = float(cue.get("target_seconds") or 0.0)
                if (cue.get("nonverbal_filler") or not raw.is_file() or not fitted.is_file()
                        or target < 1.0 or float(qc.get("slowdown_percent") or 0.0) <= 1.0):
                    continue  # only the slowdown-affected population
                out = Path(temp) / f"{len(pairs):04d}.wav"
                try:
                    metrics = audio_fit.fit_audio(raw, out, target)
                except Exception as error:
                    print(f"fit fail {job.name} cue {index + 1}: {error}", file=sys.stderr)
                    continue
                old, new = mos(fitted), mos(out)
                if old is None or new is None:
                    continue
                pairs.append({"job": job.name, "cue": index + 1,
                              "old_slowdown": qc.get("slowdown_percent"),
                              "old_mos": round(old, 3), "new_mos": round(new, 3),
                              "new_padding_ms": metrics["padding_ms"]})
                if len(pairs) % 25 == 0:
                    print(f"{len(pairs)} pairs done", flush=True)

    deltas = sorted(p["new_mos"] - p["old_mos"] for p in pairs)
    print(f"\npairs: {len(pairs)}")
    if pairs:
        print(f"old MOS mean {statistics.mean(p['old_mos'] for p in pairs):.3f}  "
              f"new MOS mean {statistics.mean(p['new_mos'] for p in pairs):.3f}")
        print(f"paired delta mean {statistics.mean(deltas):+.3f}  "
              f"p10 {deltas[len(deltas) // 10]:+.3f}  p90 {deltas[9 * len(deltas) // 10]:+.3f}")
        print(f"improved: {sum(d > 0 for d in deltas)}/{len(deltas)}")
    if args.json:
        args.json.write_text(json.dumps(pairs, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
