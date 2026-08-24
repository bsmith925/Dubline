#!/usr/bin/env python3
"""Metric validation without human scoring.

A metric earns trust by detecting failures we *inject on purpose*. Each anchor reproduces a
real Dubline failure mode on a known-good delivery; a metric passes if its response is in the
right direction and larger than its measured noise floor (eval/pareto.md).

  anchors.py build  --job <job_dir> --out <dir>    # render degraded variants
  anchors.py score  --out <dir>                    # evaluate metrics on each variant vs clean

Anchors (failure modes they stand in for):
  audio_offset_200ms   picture/audio desync (SYNC-001 class)
  picture_offset_4s    the delivered picture running ahead of audio (the 4 s bug)
  audio_speedup_30     over-compressed takes (timing track)
  silence_gap_1500ms   dead air / mouth moving in silence
  mouth_blur           renderer blur (lip-generation ceiling)
  wrong_face_mouth     lip-sync animating the wrong face (VIDEO-007 class)
  bed_removed          separation/mix losing the music-and-effects bed
  voice_quiet_10db     dub too quiet against the bed (MIX-004/006 class)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FFMPEG = ["ffmpeg", "-y", "-v", "error"]

ANCHORS: dict[str, list[str]] = {
    "audio_offset_200ms": ["-filter_complex", "[0:a]adelay=200|200[a]", "-map", "0:v:0", "-map", "[a]", "-shortest"],
    "picture_offset_4s": ["-filter_complex", "[0:v]setpts=PTS-4.0/TB[v]", "-map", "[v]", "-map", "0:a:0", "-shortest"],
    "audio_speedup_30": ["-filter_complex", "[0:a]atempo=1.3,apad[a]", "-map", "0:v:0", "-map", "[a]", "-shortest"],
    "silence_gap_1500ms": ["-filter_complex", "[0:a]volume=enable='between(t,3,4.5)':volume=0[a]", "-map", "0:v:0", "-map", "[a]"],
    "mouth_blur": ["-filter_complex", "[0:v]split[v][m];[m]crop=iw*0.3:ih*0.3:iw*0.35:ih*0.45,boxblur=8[b];[v][b]overlay=W*0.35:H*0.45[vo]",
                   "-map", "[vo]", "-map", "0:a:0"],
    "wrong_face_mouth": ["-filter_complex", "[0:v]split[v][m];[m]crop=iw*0.25:ih*0.25:iw*0.05:ih*0.5,hflip,setpts=PTS+0.4/TB[b];[v][b]overlay=W*0.35:H*0.5[vo]",
                         "-map", "[vo]", "-map", "0:a:0"],
    "voice_quiet_10db": ["-filter_complex", "[0:a]volume=-10dB[a]", "-map", "0:v:0", "-map", "[a]"],
}


def build(args: argparse.Namespace) -> None:
    job = Path(args.job); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    clean = job / "dubbed-english.mkv"
    if not clean.is_file():
        raise SystemExit(f"no delivery in {job}")
    subprocess.run([*FFMPEG, "-i", str(clean), "-t", str(args.seconds), "-c", "copy", str(out / "clean.mkv")], check=True)
    made = []
    for name, filters in ANCHORS.items():
        dst = out / f"{name}.mkv"
        subprocess.run([*FFMPEG, "-i", str(out / "clean.mkv"), *filters, "-c:v", "libx264", "-crf", "16",
                        "-c:a", "flac", str(dst)], check=True)
        made.append(name)
    (out / "anchors.json").write_text(json.dumps({"job": str(job), "anchors": made}, indent=1))
    print(f"built {len(made)} anchors in {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--job", required=True); b.add_argument("--out", required=True)
    b.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
