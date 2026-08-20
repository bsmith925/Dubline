#!/usr/bin/env python3
"""Round-trip evaluation of Dubline using YouTube auto-dubbed tracks as a dataset.

YouTube publishes machine-dubbed audio tracks for many English-language videos.
That gives a free benchmark with *real* ground truth:

    original English speech  --YouTube-->  Spanish/French/... dub  --Dubline-->  English

Dubline's English is scored against what the speaker actually said (the
original's captions, or a reference transcript you supply).  The harness also
records Dubline's own QC figures so runs can be compared over time.

    python scripts/eval_autodub.py --server http://isengard:8000 \\
        https://www.youtube.com/watch?v=ZLxazlP7Ppo --dub-language es --start 0:00 --end 1:30

    python scripts/eval_autodub.py --server http://isengard:8000 --list eval/videos.txt --dub-language fr

Requires yt-dlp and ffmpeg on this machine; everything else is the standard library.
Results are appended to eval/results.jsonl and printed as a table.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dubline_send import Client, choose_tracks, parse_time, upload, wait  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
LANGUAGE_NAMES = {"es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
                  "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ru": "Russian", "hi": "Hindi",
                  "id": "Indonesian", "ar": "Arabic", "pl": "Polish", "nl": "Dutch", "tr": "Turkish"}


# --------------------------------------------------------------------------- text metrics
def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)          # [music], (laughs)
    text = text.replace("’", "'").replace("—", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return edit_distance(ref, hyp) / max(1, len(ref))


def chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, Popović 2015); tolerant of paraphrase."""
    ref = " ".join(normalize(reference)); hyp = " ".join(normalize(hypothesis))
    precisions, recalls = [], []
    for size in range(1, n + 1):
        r = Counter(ref[i:i + size] for i in range(max(0, len(ref) - size + 1)))
        h = Counter(hyp[i:i + size] for i in range(max(0, len(hyp) - size + 1)))
        overlap = sum((r & h).values())
        precisions.append(overlap / max(1, sum(h.values())))
        recalls.append(overlap / max(1, sum(r.values())))
    p, r = sum(precisions) / n, sum(recalls) / n
    return (1 + beta ** 2) * p * r / max(1e-9, beta ** 2 * p + r)


def content_word_recall(reference: str, hypothesis: str) -> float:
    stop = set("the a an and or but so of to in on at for with by from as is are was were be been it this that "
               "these those i you he she we they them his her its our your my me him us do does did not no yes "
               "if then than there here have has had will would can could should just like about into over".split())
    ref = {w for w in normalize(reference) if w not in stop and len(w) > 2}
    hyp = set(normalize(hypothesis))
    return len(ref & hyp) / max(1, len(ref))


# --------------------------------------------------------------------------- youtube
def ytdlp() -> list[str]:
    """A *current* yt-dlp: $YTDLP, else `uvx yt-dlp` (always fresh), else whatever is on PATH."""
    if os.environ.get("YTDLP"):
        return os.environ["YTDLP"].split()
    if shutil.which("uvx"):
        return ["uvx", "yt-dlp"]
    return ["yt-dlp"]


def ytdlp_json(url: str) -> dict:
    result = subprocess.run([*ytdlp(), "--dump-single-json", "--no-warnings", "--skip-download", url],
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"yt-dlp failed for {url}: {result.stderr.strip()[-400:]}\n"
                         "(an outdated yt-dlp is the usual cause; set YTDLP='uvx yt-dlp' or upgrade)")
    return json.loads(result.stdout)


def pick_formats(info: dict, dub_language: str) -> tuple[str, str, str, str]:
    """Return (video_format_id, dub_audio_format_id, original_audio_format_id, original_language)."""
    formats = info["formats"]
    video = [f for f in formats if f.get("vcodec", "none") != "none" and f.get("acodec") == "none"
             and f.get("ext") == "mp4" and (f.get("height") or 0) <= 720 and str(f.get("vcodec", "")).startswith("avc1")
             and str(f.get("protocol", "")).startswith("http") and "m3u8" not in str(f.get("protocol", ""))]
    if not video:
        raise SystemExit("No 720p H.264 video-only format found")
    video_id = max(video, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0))["format_id"]
    audio = [f for f in formats if f.get("vcodec") == "none" and f.get("acodec", "none") != "none" and f.get("ext") == "m4a"
             and "m3u8" not in str(f.get("protocol", ""))]

    def is_original(f):
        note = str(f.get("format_note", "")).lower()
        return "original" in note or f.get("audio_is_original") is True or "(default)" in note

    originals = [f for f in audio if is_original(f)]
    dubs = [f for f in audio if str(f.get("language") or "").lower().startswith(dub_language.lower()) and not is_original(f)]
    if not dubs:
        langs = sorted({str(f.get("language")) for f in audio})
        raise SystemExit(f"No '{dub_language}' auto-dub track on this video. Audio languages: {langs}")
    best = lambda items: max(items, key=lambda f: f.get("abr") or 0)["format_id"]
    original_lang = (originals[0].get("language") if originals else info.get("language")) or "en"
    return video_id, best(dubs), best(originals) if originals else "", str(original_lang)


def caption_text(info: dict, language: str, start: float, end: float) -> tuple[str, str]:
    """Return (text, source) from manual captions when present, else auto-captions."""
    for source, table in (("captions", info.get("subtitles") or {}), ("auto-captions", info.get("automatic_captions") or {})):
        tracks = table.get(language) or table.get(language.split("-")[0]) or []
        track = next((t for t in tracks if t.get("ext") == "json3"), None)
        if not track:
            continue
        data = json.loads(urllib.request.urlopen(track["url"], timeout=60).read())
        words = []
        for event in data.get("events", []):
            t0 = event.get("tStartMs", 0) / 1000
            if t0 < start or t0 >= end:
                continue
            for seg in event.get("segs") or []:
                piece = seg.get("utf8", "")
                if piece.strip():
                    words.append(piece)
        text = re.sub(r"\s+", " ", " ".join(words)).strip()
        if text:
            return text, source
    return "", "none"


def fetch(url: str, info: dict, dub_language: str, stage: Path) -> tuple[Path, Path | None, str]:
    video_id, dub_id, original_id, original_lang = pick_formats(info, dub_language)
    vid = info["id"]
    dubbed = stage / f"{vid}.{dub_language}.mkv"
    original = stage / f"{vid}.{original_lang}.original.m4a"
    if not dubbed.is_file():
        subprocess.run([*ytdlp(), "-q", "--no-warnings", "-f", f"{video_id}+{dub_id}", "--merge-output-format", "mkv",
                        "-o", str(dubbed), url], check=True)
    if original_id and not original.is_file():
        subprocess.run([*ytdlp(), "-q", "--no-warnings", "-f", original_id, "-o", str(original), url], check=True)
    return dubbed, (original if original.is_file() else None), original_lang


# --------------------------------------------------------------------------- run
def evaluate(url: str, args, client: Client) -> dict:
    info = ytdlp_json(url)
    title = info.get("title", info["id"])
    print(f"\n=== {title}")
    stage = EVAL_DIR / "sources"; stage.mkdir(parents=True, exist_ok=True)
    dubbed, original_audio, original_lang = fetch(url, info, args.dub_language, stage)
    start = parse_time(args.start) or 0.0
    end = parse_time(args.end) or float(info.get("duration") or 0)
    reference, reference_source = caption_text(info, original_lang, start, end)
    if args.reference:
        reference, reference_source = Path(args.reference).read_text(encoding="utf-8"), "supplied"
    if not reference:
        print("  WARNING: no reference transcript available; translation metrics will be skipped")

    options = {"source_language": LANGUAGE_NAMES.get(args.dub_language, "auto"), "target_language": "English",
               "subtitle_mode": "speech", "audio_mode": "separate", "engine": "indextts", "emotion_mode": "auto",
               "workflow_mode": "automatic", "mastering_preset": args.preset,
               "range_start": start, "range_end": end if args.end else None, "voice_rights_confirmed": True,
               "glossary": {}}
    if args.remote_prefix:
        remote = f"{args.remote_prefix.rstrip('/')}/{dubbed.name}"
        subprocess.run(["rsync", "-az", str(dubbed), f"{args.remote_host}:{remote}"], check=True)
        job = client.json("POST", "/api/jobs/local", {"path": remote, "options": options})
    else:
        spec = {"files": [{"name": dubbed.name, "size": dubbed.stat().st_size, "kind": "video"}], "options": options}
        job = client.json("POST", "/api/jobs", spec)
        upload(client, job, [dubbed])
        job = client.json("POST", f"/api/jobs/{job['id']}/finalize")
    if job["status"] == "awaiting_selection":
        job = choose_tracks(client, job, argparse.Namespace(audio_stream=None, subtitle_stream=None, subtitle_mode="speech"))
    print(f"  job {job['id']} submitted ({args.dub_language} dub, {start:.0f}-{end:.0f}s)")
    started = time.time()
    job = wait(client, job["id"], args.poll)
    elapsed = time.time() - started

    cues = job.get("cues") or []
    hypothesis = " ".join(str(c.get("english", "")) for c in cues)
    faithful = " ".join(str(c.get("faithful_translation") or c.get("literal_translation") or "") for c in cues)
    qc = job.get("qc") or {}
    detected = job.get("detected_language") or {}
    stretches = [abs(float((c.get("qc") or {}).get("stretch_percent") or 0)) for c in cues]
    word_sims = [float((c.get("qc") or {}).get("word_similarity") or 0) for c in cues if (c.get("qc") or {}).get("word_similarity") is not None]
    judge_failed = sum(1 for c in cues if (c.get("translation_qc") or {}).get("available") and not (c.get("translation_qc") or {}).get("passed"))
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "video_id": info["id"], "title": title, "url": url,
        "dub_language": args.dub_language, "range": [start, end], "job_id": job["id"], "status": job["status"],
        "server": client.server, "elapsed_seconds": round(elapsed), "realtime_factor": round(elapsed / max(1, end - start), 2),
        "detected_language": detected.get("language"), "detected_confidence": detected.get("confidence"),
        "reference_source": reference_source, "reference_words": len(normalize(reference)),
        "hypothesis_words": len(normalize(hypothesis)), "cue_count": len(cues),
        "wer": round(wer(reference, hypothesis), 4) if reference else None,
        "chrf": round(chrf(reference, hypothesis), 4) if reference else None,
        "content_recall": round(content_word_recall(reference, hypothesis), 4) if reference else None,
        # Same metrics on the pre-adaptation faithful translation: isolates translation
        # quality from the deliberate timing-driven compression of the dub lines.
        "faithful_words": len(normalize(faithful)),
        "wer_faithful": round(wer(reference, faithful), 4) if reference else None,
        "chrf_faithful": round(chrf(reference, faithful), 4) if reference else None,
        "content_recall_faithful": round(content_word_recall(reference, faithful), 4) if reference else None,
        "qc_passed": qc.get("passed"), "qc_flagged": qc.get("flagged_count"), "qc_cue_count": qc.get("cue_count"),
        "judge_failed": judge_failed, "integrated_lufs": qc.get("integrated_lufs"), "true_peak_dbtp": qc.get("true_peak_dbtp"),
        "mean_abs_stretch_percent": round(sum(stretches) / len(stretches), 2) if stretches else None,
        "mean_backtranscription_similarity": round(sum(word_sims) / len(word_sims), 4) if word_sims else None,
        "speakers": sorted({str(c.get("speaker")) for c in cues}),
        "error": job.get("error"),
    }
    out_dir = EVAL_DIR / "runs" / f"{info['id']}-{args.dub_language}-{job['id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference.txt").write_text(reference, encoding="utf-8")
    (out_dir / "hypothesis.txt").write_text(hypothesis, encoding="utf-8")
    (out_dir / "faithful.txt").write_text(faithful, encoding="utf-8")
    (out_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if job["status"] in {"complete", "needs_review"} and not args.no_download:
        client.download(f"/api/jobs/{job['id']}/download", out_dir / "dubbed-english.mkv")
        try:
            client.download(f"/api/jobs/{job['id']}/export/dialogue", out_dir / "english-dialogue.flac")
        except Exception:
            pass
        if original_audio:
            (out_dir / original_audio.name).write_bytes(original_audio.read_bytes())
    with (EVAL_DIR / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def print_table(rows: list[dict]) -> None:
    cols = [("video_id", 11), ("dub_language", 4), ("status", 12), ("wer", 6), ("chrf", 6), ("content_recall", 7),
            ("chrf_faithful", 8),
            ("qc_flagged", 5), ("judge_failed", 5), ("mean_abs_stretch_percent", 8), ("integrated_lufs", 7),
            ("true_peak_dbtp", 6), ("realtime_factor", 6)]
    header = " ".join(f"{name[:width]:<{width}}" for name, width in cols)
    print("\n" + header); print("-" * len(header))
    for row in rows:
        cells = []
        for name, width in cols:
            value = row.get(name)
            text = "-" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))
            cells.append(f"{text[:width]:<{width}}")
        print(" ".join(cells))
    print("\nwer/chrf/content_recall compare Dubline's English against the original's transcript "
          "(auto-captions carry their own ~5-10% error, so WER will not reach 0).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="YouTube URLs of English-original videos that have auto-dubs")
    ap.add_argument("--list", type=Path, help="file with one URL per line (# comments allowed)")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--dub-language", default="es", help="auto-dub track to feed Dubline (es, fr, de, it, pt, ja, ...)")
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    ap.add_argument("--preset", default="web", choices=["cinema", "broadcast", "web", "preserve"])
    ap.add_argument("--reference", help="reference English transcript file (overrides YouTube captions)")
    ap.add_argument("--remote-prefix", help="server-side directory reachable via rsync (skips HTTP upload)")
    ap.add_argument("--remote-host", default="isengard", help="ssh host for --remote-prefix")
    ap.add_argument("--poll", type=float, default=15)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    urls = list(args.urls)
    if args.list:
        urls += [line.strip() for line in args.list.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
    if not urls:
        ap.error("give at least one URL or --list")
    client = Client(args.server)
    rows = [evaluate(url, args, client) for url in urls]
    print_table(rows)


if __name__ == "__main__":
    main()
