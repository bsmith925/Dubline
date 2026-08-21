from __future__ import annotations

"""Engineering-defect metrics: things a viewer notices that model metrics do not.

- mouth_sharpness_ratio: Laplacian variance of the lower-face region, output / source,
  inside lip-synced intervals (blur, ghosting from frame blending).
- boundary_jump: frame-difference at lip-sync clip edges divided by the clip's median
  frame difference (hard visual cuts where rendered frames are swapped in).
- dead_air: seconds where the final mix is near-silent but the source soundtrack was
  audible (room tone / breaths lost with the dialogue).
- unmuted_source_speech: source speech not covered by the muted word spans.
- take_overlap: placed takes that overlap the next take.
- audio_default_tracks: number of audio streams flagged default in the delivered file.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .. import audio as A


def _frames(video: Path, t_end: float, step: int = 1):
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; i = 0
    while True:
        ok, f = cap.read()
        if not ok or i / fps > t_end:
            break
        if i % step == 0:
            yield i / fps, f
        i += 1
    cap.release()


def mouth_sharpness_ratio(source: Path, output: Path, intervals: list[list[float]], t_end: float,
                          landmarks: list[dict] | None = None) -> float | None:
    """Output/source sharpness in the mouth region within ``intervals`` (1.0 = as sharp as source)."""
    def region(frame, t):
        h, w = frame.shape[:2]
        box = None
        if landmarks:
            near = min(landmarks, key=lambda r: abs(r["t"] - t)) if landmarks else None
            if near and near.get("mouth_box"):
                box = near["mouth_box"]
        if box:
            x1, y1, x2, y2 = [int(v) for v in box]
            return frame[max(0, y1):y2, max(0, x1):x2]
        return frame[int(h * .45):int(h * .75), int(w * .35):int(w * .65)]
    def series(video):
        out = {}
        for t, f in _frames(video, t_end, step=3):
            if any(a <= t <= b for a, b in intervals):
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                out[round(t, 2)] = float(cv2.Laplacian(region(g, t), cv2.CV_64F).var())
        return out
    s, o = series(source), series(output)
    ratios = [o[t] / max(1e-6, s[t]) for t in s if t in o]
    return round(float(np.median(ratios)), 3) if ratios else None


def boundary_jumps(output: Path, intervals: list[list[float]], t_end: float) -> dict:
    diffs = []; prev = None
    for t, f in _frames(output, t_end):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32)
        if prev is not None:
            diffs.append((t, float(np.mean(np.abs(g - prev)))))
        prev = g
    if not diffs:
        return {"median": None, "edges": []}
    med = float(np.median([d for _, d in diffs])) or 1e-6
    fps = 1.0 / max(1e-6, diffs[1][0] - diffs[0][0]) if len(diffs) > 1 else 25.0
    edges = []
    for a, b in intervals:
        for edge in (a, b):
            near = [d for t, d in diffs if abs(t - edge) <= 1.5 / fps]
            if near:
                edges.append({"t": round(edge, 2), "jump_x_median": round(max(near) / med, 1)})
    return {"median_frame_diff": round(med, 3), "edges": edges,
            "max_jump_x_median": round(max((e["jump_x_median"] for e in edges), default=0.0), 1),
            "mean_jump_x_median": round(float(np.mean([e["jump_x_median"] for e in edges])), 1) if edges else None}


def dead_air(source_mix: Path, final_mix: Path, t_end: float, silent_db: float = -60.0, audible_db: float = -45.0) -> dict:
    src = A.speech_intervals(source_mix, thresh_db=audible_db, min_gap=0.05, min_run=0.1, t_max=t_end)
    out = A.speech_intervals(final_mix, thresh_db=silent_db, min_gap=0.05, min_run=0.05, t_max=t_end)
    # dead air = source audible minus (final above the silence floor)
    gaps = [iv for iv in A.subtract(src, out) if iv[1] - iv[0] >= 0.2]
    return {"seconds": A.total(gaps), "intervals": gaps[:20]}


def unmuted_source_speech(dialogue_stem: Path, cues: list[dict], t_end: float, pad: float = 0.035) -> dict:
    src = A.speech_intervals(dialogue_stem, t_max=t_end)
    words = [[float(w["start"]) - pad, float(w["end"]) + pad] for c in cues for w in (c.get("words") or [])
             if not c.get("nonverbal_filler")]
    left = [iv for iv in A.subtract(src, words) if iv[1] - iv[0] >= 0.15]
    return {"seconds": A.total(left), "source_speech_seconds": A.total(src), "intervals": left[:20]}


def take_overlaps(job_dir: Path, cues: list[dict]) -> dict:
    import soundfile as sf
    takes = []
    for c in cues:
        p = job_dir / "acoustically-matched" / f"{int(c['id']):06d}.wav"
        if p.is_file() and not c.get("nonverbal_filler"):
            takes.append((float(c["start"]), float(c["start"]) + sf.info(p).duration, int(c["id"])))
    takes.sort()
    overlaps = [{"prev": a[2], "next": b[2], "seconds": round(a[1] - b[0], 3)} for a, b in zip(takes, takes[1:]) if a[1] > b[0] + 0.02]
    return {"count": len(overlaps), "seconds": round(sum(o["seconds"] for o in overlaps), 3), "pairs": overlaps}


def audio_default_tracks(video: Path) -> dict:
    try:
        data = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
                                         capture_output=True, text=True).stdout)
    except Exception:
        return {"audio_streams": None, "default_audio_streams": None}
    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    return {"audio_streams": len(audio_streams),
            "default_audio_streams": sum(1 for s in audio_streams if (s.get("disposition") or {}).get("default"))}
