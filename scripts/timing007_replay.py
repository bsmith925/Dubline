#!/usr/bin/env python3
"""EXP-TIMING-007 offline replay: would least-stretch selection have chosen differently?

Replays both selection policies over jobs that already carry measured candidates
(`qc.candidate_measurements`, recorded by EXP-TIMING-001/003 runs) — no synthesis, no
GPU. Optionally UTMOS-scores the shipped vs would-be-selected fitted takes on CPU.

Usage:
  python timing007_replay.py JOB_DIR [JOB_DIR ...] [--utmos] [--json out.json]

Caveats (recorded in the output):
- The shipped take stands in for "current": if the max-fill policy already switched a
  cue, the original take's measurements were overwritten, so the replay pool is the
  shipped take plus all candidates. This can only under-estimate the new policy's
  options, never over-estimate.
- Bundles recorded before 2026-08-25 lack slowdown_percent on candidates; manipulation
  for those is |stretch| only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

try:
    from app.services.take_selection import manipulation_percent, select_take
except ImportError:  # running from a bare copy next to take_selection.py (server /tmp)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from take_selection import manipulation_percent, select_take  # type: ignore

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


def replay_job(folder: Path, score_audio: bool) -> dict:
    cues = json.loads((folder / "cues.json").read_text(encoding="utf-8"))
    rows = []
    for index, cue in enumerate(cues):
        qc = cue.get("qc") or {}
        measurements = qc.get("candidate_measurements") or {}
        if not measurements or cue.get("nonverbal_filler"):
            continue
        target = float(cue.get("target_seconds") or 0.0)
        limit = 5.0 if cue.get("mouth_visible") else 8.0
        candidates = {int(k): dict(m) for k, m in measurements.items()}
        old = select_take(qc, candidates, target, limit, prefer_least_stretch=False)
        new = select_take(qc, candidates, target, limit, prefer_least_stretch=True)
        chosen = candidates.get(new, qc)
        shipped_text = str(cue.get("spoken_text") or "")
        switched = bool(new) and str(chosen.get("text") or "").strip() != shipped_text.strip()
        row = {
            "job": folder.name, "cue": index + 1, "target_seconds": target,
            "replayed_max_fill_choice": old,          # non-zero here = replay drift, see notes
            "least_stretch_choice": new, "switched": switched,
            "old_manipulation": round(manipulation_percent(qc), 2),
            "new_manipulation": round(manipulation_percent(chosen), 2),
            "old_fill": qc.get("active_fill_percent"),
            "new_fill": chosen.get("active_fill_percent"),
        }
        if switched:
            row["old_wav"] = str(folder / "fitted" / f"{index + 1:06d}.wav")
            row["new_wav"] = str(folder / "generated-candidates" / f"{index + 1:06d}-{new}-fitted.wav")
            if score_audio:
                row["old_mos"] = utmos(Path(row["old_wav"]))
                row["new_mos"] = utmos(Path(row["new_wav"]))
        rows.append(row)
    return {"job": folder.name, "cues": rows}


def summarize(rows: list[dict]) -> dict:
    def dist(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        values.sort()
        return {"mean": round(statistics.mean(values), 2), "p50": round(values[len(values) // 2], 2),
                "p90": round(values[int(len(values) * 0.9) - 1] if len(values) > 1 else values[-1], 2),
                "n": len(values)}
    switched = [r for r in rows if r["switched"]]
    scored = [r for r in switched if r.get("old_mos") is not None and r.get("new_mos") is not None]
    return {
        "cues_with_candidates": len(rows),
        "switched": len(switched),
        "replay_drift": sum(1 for r in rows if r["replayed_max_fill_choice"]),
        "manipulation_old": dist([r["old_manipulation"] for r in rows]),
        "manipulation_new": dist([r["new_manipulation"] if r["switched"] else r["old_manipulation"] for r in rows]),
        "fill_old": dist([r["old_fill"] for r in rows]),
        "fill_new": dist([r["new_fill"] if r["switched"] else r["old_fill"] for r in rows]),
        "mos_scored_pairs": len(scored),
        "mos_old": dist([r["old_mos"] for r in scored]),
        "mos_new": dist([r["new_mos"] for r in scored]),
        "mos_delta_mean": round(statistics.mean([r["new_mos"] - r["old_mos"] for r in scored]), 3) if scored else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="+", type=Path)
    parser.add_argument("--utmos", action="store_true", help="UTMOS-score old vs new fitted takes (CPU)")
    parser.add_argument("--json", type=Path, help="write full per-cue results here")
    args = parser.parse_args()

    all_rows: list[dict] = []
    for folder in args.jobs:
        if not (folder / "cues.json").is_file():
            print(f"skip {folder}: no cues.json", file=sys.stderr)
            continue
        result = replay_job(folder, args.utmos)
        all_rows.extend(result["cues"])
        print(f"{folder.name}: {len(result['cues'])} cues with candidates, "
              f"{sum(1 for r in result['cues'] if r['switched'])} would switch")

    summary = summarize(all_rows)
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "cues": all_rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
