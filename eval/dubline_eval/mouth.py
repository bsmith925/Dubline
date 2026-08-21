from __future__ import annotations

"""Per-frame mouth aperture from a video, and articulation intervals derived from it.

The landmark pass runs in the MuseTalk venv (face_alignment + torch CUDA) as a
subprocess so this package itself stays dependency-light.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

WORKER = r'''
import json, sys, cv2, numpy as np, torch
from face_alignment import FaceAlignment, LandmarksType
video, t_max, fps, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device="cuda" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture(video); src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
step = max(1, int(round(src_fps / fps))); rows = []; index = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    t = index / src_fps
    if t > t_max: break
    if index % step == 0:
        lms = fa.get_landmarks_from_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if lms:
            pts = max(lms, key=lambda p: (p[:, 0].max() - p[:, 0].min()))
            face_h = float(pts[8, 1] - pts[27, 1])
            rows.append({"t": round(t, 3), "inner": round(float(np.linalg.norm(pts[62] - pts[66])) / max(face_h, 1), 4),
                         "face_h_px": round(face_h, 1), "faces": len(lms)})
        else:
            rows.append({"t": round(t, 3), "inner": None, "face_h_px": None, "faces": 0})
    index += 1
json.dump(rows, open(out, "w"))
'''


def mouth_series(video: Path, runtime: Path, t_max: float, fps: float = 15.0, cache: Path | None = None) -> list[dict]:
    if cache and cache.is_file():
        return json.loads(cache.read_text())
    out = (cache or Path(f"/tmp/mouth-{video.stem}.json"))
    script = out.with_suffix(".worker.py"); script.write_text(WORKER)
    subprocess.run([str(runtime), str(script), str(video), str(t_max), str(fps), str(out)],
                   check=True, capture_output=True)
    return json.loads(out.read_text())


ARTICULATION_STD = 0.012   # aperture (inner-lip gap / face height) std over a 200 ms window


def articulation_intervals(series: list[dict], window: float = 0.2, min_face_px: float = 50.0,
                           threshold: float = ARTICULATION_STD) -> list[list[float]]:
    """Intervals where the mouth is articulating: local aperture std above an absolute threshold.

    An absolute threshold (not a per-clip percentile) so a lively face is not "always
    active" and a still face is not "always moving"; 0.012 ≈ a 1.2%-of-face-height
    open/close within 200 ms, calibrated on the gingerBill talking-head baselines.
    """
    rows = [r for r in series if r.get("inner") is not None and (r.get("face_h_px") or 0) >= min_face_px]
    if len(rows) < 5:
        return []
    t = np.array([r["t"] for r in rows]); a = np.array([r["inner"] for r in rows])
    dt = np.median(np.diff(t)) if len(t) > 1 else 1 / 15
    k = max(3, int(round(window / dt)))
    motion = np.array([a[max(0, i - k // 2): i + k // 2 + 1].std() for i in range(len(a))])
    active = motion > threshold
    intervals = []
    start = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = t[i]
        elif not flag and start is not None:
            intervals.append([round(float(start), 3), round(float(t[i]), 3)]); start = None
    if start is not None:
        intervals.append([round(float(start), 3), round(float(t[-1] + dt), 3)])
    merged: list[list[float]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] < 0.15:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [iv for iv in merged if iv[1] - iv[0] >= 0.1]


def mean_aperture(series: list[dict], start: float, end: float, min_face_px: float = 50.0) -> float | None:
    values = [r["inner"] for r in series if r.get("inner") is not None and start <= r["t"] < end
              and (r.get("face_h_px") or 0) >= min_face_px]
    return round(float(np.mean(values)), 4) if values else None
