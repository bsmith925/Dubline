#!/usr/bin/env python3
"""Human calibration of the evaluator: blind A/B snippet pairs and metric-vs-preference analysis.

  make    build pairs from rendered outputs of different systems on identical windows
          (plus deliberately degraded variants as anchors), randomize A/B, hide the key,
          compute the harness metrics per snippet, write a scoring sheet
  analyze join the scored sheet with the key and metrics: per-metric agreement with
          the human preference, P(human prefers B | metric says B>A), and misses

Runs on the GPU host (SyncNet, landmarks). Snippets are short (≤ 12 s) and named
pair_NNN_A.mp4 / pair_NNN_B.mp4; nothing in the filename reveals the system.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIMENSIONS = ["overall", "naturalness", "timing", "voice_likeness", "translation", "lipsync", "artifacts"]


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} exit {r.returncode}: {r.stderr.strip()[-800:]}")
    return r


def cut(src: Path, start: float, end: float, out: Path, audio_stream: int = 0) -> Path:
    sh(["ffmpeg", "-y", "-v", "error", "-accurate_seek", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end-start:.3f}",
        "-map", "0:v:0", "-map", f"0:a:{audio_stream}", "-vf", "scale=-2:720", "-r", "30", "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", str(out)])
    return out


DEGRADATIONS = {
    "audio_speedup_30": ["-filter_complex", "[0:a]atempo=1.3,apad[a]", "-map", "0:v:0", "-map", "[a]", "-shortest"],
    "audio_offset_200ms": ["-filter_complex", "[0:a]adelay=200|200[a]", "-map", "0:v:0", "-map", "[a]", "-shortest"],
    "mouth_blur": ["-filter_complex", "[0:v]split[v][m];[m]crop=iw*0.3:ih*0.3:iw*0.35:ih*0.45,boxblur=8[b];[v][b]overlay=W*0.35:H*0.45[vo]",
                   "-map", "[vo]", "-map", "0:a:0"],
    "silence_gap_1500ms": ["-filter_complex", "[0:a]volume=enable='between(t,3,4.5)':volume=0[a]", "-map", "0:v:0", "-map", "[a]"],
}


def degrade(src: Path, kind: str, out: Path) -> Path:
    sh(["ffmpeg", "-y", "-v", "error", "-i", str(src), *DEGRADATIONS[kind], "-c:v", "libx264", "-crf", "18", "-c:a", "aac", str(out)])
    return out


def snippet_metrics(snippet: Path, source_snippet: Path, work: Path) -> dict:
    """Harness metrics on one snippet (audio timing, mouth motion, sync, sharpness)."""
    from eval.dubline_eval import audio as A
    from eval.dubline_eval.mouth import mouth_series, articulation_intervals, mean_aperture
    from eval.dubline_eval.metrics.av_sync import score_interval
    from eval.dubline_eval.metrics import engineering as E
    work.mkdir(parents=True, exist_ok=True)
    dur = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(snippet)]).stdout.strip())
    wav = work / (snippet.stem + ".wav"); sh(["ffmpeg", "-y", "-v", "error", "-i", str(snippet), "-ac", "1", "-ar", "16000", str(wav)])
    src_wav = work / (source_snippet.stem + ".wav")
    if not src_wav.is_file():
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(source_snippet), "-ac", "1", "-ar", "16000", str(src_wav)])
    dub = A.speech_intervals(wav, thresh_db=-40)
    musetalk_py = ROOT / "vendor/musetalk-env/bin/python"
    mouth = mouth_series(snippet, musetalk_py, dur, 15.0, work / (snippet.stem + "-mouth.json"))
    src_mouth = mouth_series(source_snippet, musetalk_py, dur, 15.0, work / (source_snippet.stem + "-mouth.json"))
    artic = articulation_intervals(mouth); src_artic = articulation_intervals(src_mouth)
    sync = score_interval(snippet, source_snippet, 0.0, dur, work / "sync", ROOT / "vendor/syncnet_python", musetalk_py, snippet.stem) \
        if (ROOT / "vendor/syncnet_python/run_syncnet.py").is_file() else {}
    so = sync.get("output", {})
    ap_src, ap_out = mean_aperture(src_mouth, 0, dur), mean_aperture(mouth, 0, dur)
    try:
        import sys as _sys; _sys.path.insert(0, str(ROOT))
        from eval.dubline_eval.metrics.naturalness import naturalness_mos
        mos = naturalness_mos(wav)
    except Exception:
        mos = None
    return {"duration": round(dur, 2), "naturalness_mos": mos, "dub_speech_s": A.total(dub), "speech_fraction": round(A.total(dub) / dur, 3),
            "mouth_motion_on_silence_s": A.total(A.subtract(artic, dub)), "speech_on_static_mouth_s": A.total(A.subtract(dub, artic)),
            "coverage_articulation": round(A.total(A.intersect(dub, src_artic)) / A.total(src_artic), 3) if src_artic else None,
            "sync_lse_c": so.get("lse_c"), "sync_lse_d": so.get("lse_d"), "sync_offset_frames": so.get("offset_frames"),
            "sync_low_conf_fraction": so.get("low_conf_fraction"),
            "aperture_ratio": round(ap_out / ap_src, 3) if ap_src and ap_out else None,
            "mouth_sharpness_ratio": E.mouth_sharpness_ratio(source_snippet, snippet, [[0, dur]], dur),
            "dead_air_s": E.dead_air(src_wav, wav, dur).get("seconds")}


def make(args) -> None:
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out); work = out / "work"; work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    pairs = []
    for clip_spec in (spec["clips"] if "clips" in spec else [spec]):
        pairs += collect_pairs(clip_spec, work, rng)
    finalize(pairs, out, work, rng, args)


def collect_pairs(spec: dict, work: Path, rng) -> list:
    pairs = []
    source = Path(spec["source"])
    for w in spec["windows"]:
        start, end, wid = w["start"], w["end"], w["id"]
        cut(source, start, end, work / f"source-{wid}.mp4", audio_stream=spec.get("source_audio_stream", 0))
        rendered = {}
        for system, path in spec["systems"].items():
            p = Path(path)
            if p.is_file():
                rendered[system] = cut(p, start, end, work / f"{system}-{wid}.mp4")
        for a, b in combinations(sorted(rendered), 2):
            pairs.append((wid, a, rendered[a], b, rendered[b]))
        anchor = spec.get("degrade_system")
        if anchor in rendered:
            for kind in spec.get("degradations", list(DEGRADATIONS)):
                d = degrade(rendered[anchor], kind, work / f"{anchor}+{kind}-{wid}.mp4")
                pairs.append((wid, anchor, rendered[anchor], f"{anchor}+{kind}", d))
    return pairs


def finalize(pairs: list, out: Path, work: Path, rng, args) -> None:
    rng.shuffle(pairs)
    key, rows = [], []
    for n, (wid, sa, pa, sb, pb) in enumerate(pairs, 1):
        flip = rng.random() < 0.5
        left, right = ((sb, pb), (sa, pa)) if flip else ((sa, pa), (sb, pb))
        pa_out, pb_out = out / f"pair_{n:03d}_A.mp4", out / f"pair_{n:03d}_B.mp4"
        pa_out.write_bytes(left[1].read_bytes()); pb_out.write_bytes(right[1].read_bytes())
        metrics_a = snippet_metrics(left[1], work / f"source-{wid}.mp4", work / "metrics") if args.metrics else {}
        metrics_b = snippet_metrics(right[1], work / f"source-{wid}.mp4", work / "metrics") if args.metrics else {}
        key.append({"pair": n, "window": wid, "A": left[0], "B": right[0], "metrics_A": metrics_a, "metrics_B": metrics_b})
        rows.append({"pair": n, **{d: "" for d in DIMENSIONS}, "notes": ""})
        print(f"pair {n:03d}: {wid}", flush=True)
    (out / "KEY-do-not-open.json").write_text(json.dumps(key, indent=1))
    with (out / "scores.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["pair", *DIMENSIONS, "notes"]); wr.writeheader(); wr.writerows(rows)
    (out / "README.txt").write_text(
        "Blind A/B calibration.\nFor each pair watch pair_NNN_A.mp4 then pair_NNN_B.mp4 (same source window).\n"
        "In scores.csv enter A, B or = (same) for each dimension:\n"
        "  overall, naturalness (speech sounds natural), timing (speech sits where it should), voice_likeness,\n"
        "  translation (meaning/fluency), lipsync (mouth matches speech), artifacts (fewer visual/audio artifacts)\n"
        "Leave a dimension blank if it does not apply. Do not open KEY-do-not-open.json until you are done.\n")
    print(f"{len(pairs)} pairs written to {out}")


def analyze(args) -> None:
    out = Path(args.out)
    key = {k["pair"]: k for k in json.loads((out / "KEY-do-not-open.json").read_text())}
    scores = [r for r in csv.DictReader((out / "scores.csv").open()) if r.get("overall", "").strip()]
    if not scores:
        print("no scored rows yet"); return
    metric_names = sorted({m for k in key.values() for m in (k["metrics_A"] or {}) if isinstance(k["metrics_A"].get(m), (int, float))})
    higher_better = {"coverage_articulation", "sync_lse_c", "aperture_ratio", "mouth_sharpness_ratio", "speech_fraction"}
    print(f"{len(scores)} scored pairs\n")
    print(f"{'metric':28} {'n':>3} {'agree':>6} {'P(pref B | metric B>A)':>24} {'P(pref A | metric A>B)':>24}")
    for m in metric_names:
        n = agree = 0; pb_n = pb_y = pa_n = pa_y = 0
        for r in scores:
            k = key[int(r["pair"])]; a, b = (k["metrics_A"] or {}).get(m), (k["metrics_B"] or {}).get(m)
            if a is None or b is None or r["overall"].strip() == "=": continue
            better_b = (b > a) if m in higher_better else (b < a)
            if m == "sync_offset_frames": better_b = abs(b) < abs(a)
            if a == b: continue
            human_b = r["overall"].strip().upper() == "B"
            n += 1; agree += int(better_b == human_b)
            if better_b: pb_n += 1; pb_y += int(human_b)
            else: pa_n += 1; pa_y += int(not human_b)
        if n:
            print(f"{m:28} {n:3d} {agree/n:6.2f} {(pb_y/pb_n if pb_n else float('nan')):24.2f} {(pa_y/pa_n if pa_n else float('nan')):24.2f}")
    # anchors: did degraded variants lose?
    lost = total = 0
    for r in scores:
        k = key[int(r["pair"])]
        if "+" in k["A"] or "+" in k["B"]:
            total += 1; degraded_is_b = "+" in k["B"]
            lost += int((r["overall"].strip().upper() == "A") == degraded_is_b)
    if total:
        print(f"\ndegraded anchors judged worse: {lost}/{total}")
    # per-system win rates
    wins: dict[str, list[int]] = {}
    for r in scores:
        k = key[int(r["pair"])]; pref = r["overall"].strip().upper()
        for side, other in (("A", "B"), ("B", "A")):
            wins.setdefault(k[side], []).append(1 if pref == side else (0 if pref == other else 0.5))
    print("\nsystem win rate (overall):")
    for s_, w in sorted(wins.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {s_:36} {sum(w)/len(w):.2f}  (n={len(w)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--spec", required=True); m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=7); m.add_argument("--metrics", action="store_true")
    a = sub.add_parser("analyze"); a.add_argument("--out", required=True)
    args = ap.parse_args()
    (make if args.cmd == "make" else analyze)(args)


if __name__ == "__main__":
    main()
