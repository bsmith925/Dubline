from __future__ import annotations

"""Audio-visual sync via the original SyncNet (LSE-D / LSE-C / AV offset).

Scores are computed on the *output* shot and on the *same* shot of the source, and
reported as both absolute values and output-minus-source deltas: many real
recordings (webcams, broadcast) are themselves offset, so absolute thresholds
mislead. Runs joonson/syncnet_python inside the MuseTalk venv.
"""

import re
import subprocess
from pathlib import Path


def _extract(video: Path, start: float, duration: float, out: Path, audio_stream: int = 0, crop: list | None = None) -> Path:
    vf = (f"crop={int(crop[2])}:{int(crop[2])}:{int(crop[0])}:{int(crop[1])}," if crop else "") + "scale=-2:480"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(video),
                    "-map", "0:v:0", "-map", f"0:a:{audio_stream}", "-r", "25", "-vf", vf,
                    "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-ar", "16000", str(out)], check=True, capture_output=True)
    return out


def _syncnet(clip: Path, repo: Path, runtime: Path, work: Path, reference: str) -> dict | None:
    env_cwd = str(repo)
    pipe = subprocess.run([str(runtime), "run_pipeline.py", "--videofile", str(clip), "--reference", reference, "--data_dir", str(work), "--overwrite"],
                          cwd=env_cwd, capture_output=True, text=True)
    if pipe.returncode:
        return {"error": (pipe.stderr or pipe.stdout)[-400:]}
    sync = subprocess.run([str(runtime), "run_syncnet.py", "--videofile", str(clip), "--reference", reference, "--data_dir", str(work)],
                          cwd=env_cwd, capture_output=True, text=True)
    text = sync.stdout + sync.stderr
    tracks = []
    for block in re.findall(r"AV offset:\s*(-?\d+)\s*.*?Min dist:\s*([\d.]+)\s*.*?Confidence:\s*([\d.]+)", text, re.S):
        tracks.append({"offset_frames": int(block[0]), "lse_d": float(block[1]), "lse_c": float(block[2])})
    # Framewise confidence (medfilt'd) is logged as an array; fraction of frames below 1.0
    # captures local desync that the clip-level median hides.
    low_fraction = None
    arrays = re.findall(r"Framewise conf:\s*\n?(\[[^\]]*\])", text, re.S)
    if arrays:
        try:
            values = [float(v) for v in re.findall(r"-?\d+\.\d+|-?\d+", arrays[-1])]
            if values:
                low_fraction = round(sum(1 for v in values if v < 1.0) / len(values), 3)
        except ValueError:
            pass
    if not tracks:
        return {"error": text[-400:] or "no face track"}
    best = max(tracks, key=lambda t: t["lse_c"])      # the speaking track is the confident one
    return {**best, "tracks": len(tracks), "low_conf_fraction": low_fraction}


def score_interval(output_video: Path, source_video: Path, start: float, duration: float, work: Path,
                   repo: Path, runtime: Path, tag: str, crop: list | None = None) -> dict:
    """Return {'output': {...}, 'source': {...}, 'delta_lse_d', 'delta_lse_c', 'delta_offset_frames'}."""
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    for name, video in (("output", output_video), ("source", source_video)):
        if not video.is_file():
            result[name] = {"error": "missing video"}
            continue
        clip = _extract(video, start, duration, work / f"{tag}-{name}.mp4", crop=crop)
        result[name] = _syncnet(clip, repo, runtime, work / "syncnet", f"{tag}-{name}") or {"error": "no result"}
    o, s = result.get("output", {}), result.get("source", {})
    if "lse_d" in o and "lse_d" in s:
        result["delta_lse_d"] = round(o["lse_d"] - s["lse_d"], 3)          # negative = better than source
        result["delta_lse_c"] = round(o["lse_c"] - s["lse_c"], 3)          # positive = better than source
        result["delta_offset_frames"] = o["offset_frames"] - s["offset_frames"]
    return result
