#!/usr/bin/env python3
"""Send a video to a remote Dubline server and fetch the dubbed result.

Standard library only, so it runs on any machine with Python 3.9+:

    python scripts/dubline_send.py --server http://isengard:8000 film.mkv
    python scripts/dubline_send.py --server http://isengard:8000 film.mkv film.srt \\
        --start 20:20 --end 22:00 --preset web --out ./dubs

What it does:
  1. creates a job (optionally with sidecar subtitles),
  2. uploads every file in resumable 16 MiB chunks (re-run the same command to
     resume an interrupted upload of the same job with --job),
  3. finalizes, answering the audio/subtitle track question if the server asks,
  4. polls until the dub is complete, printing stage/progress,
  5. downloads the MKV and the QC report (and other exports with --exports).

Use --no-wait to return right after submission and `--job ID --wait` later.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK = 16 * 1024 * 1024
VIDEO = {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mts", ".m2ts",
         ".mpeg", ".mpg", ".wmv", ".mxf", ".vob", ".3gp"}
AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}
SUBS = {".srt", ".ass", ".ssa", ".vtt", ".sub"}
TERMINAL = {"complete", "needs_review", "error", "cancelled"}


class Client:
    def __init__(self, server: str, timeout: float = 600):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: bytes | None = None,
                headers: dict | None = None, raw: bool = False):
        req = urllib.request.Request(self.server + path, data=body, method=method,
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
                return (data, resp.status, resp.headers) if raw else json.loads(data or b"null")
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            try:
                detail = json.loads(payload).get("detail", payload.decode("utf-8", "replace"))
            except ValueError:
                detail = payload.decode("utf-8", "replace")
            if exc.code == 409 and not raw:
                # The upload endpoint uses 409 + {"offset": n} to resynchronise.
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict) and "offset" in parsed:
                        return parsed
                except ValueError:
                    pass
            raise SystemExit(f"{method} {path} failed ({exc.code}): {detail}") from None
        except urllib.error.URLError as exc:
            raise SystemExit(f"Cannot reach {self.server}: {exc.reason}") from None

    def json(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        return self.request(method, path, body, {"Content-Type": "application/json"} if body else {})

    def download(self, path: str, target: Path) -> Path:
        req = urllib.request.Request(self.server + path)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp, target.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        return target


def parse_time(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    parts = value.strip().split(":")
    if not all(p.replace(".", "", 1).isdigit() for p in parts):
        raise SystemExit(f"Invalid time '{value}' (use seconds, MM:SS or HH:MM:SS)")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO:
        return "video"
    if suffix in AUDIO:
        return "audio"
    if suffix == ".idx":
        return "subtitle_index"
    if suffix in SUBS:
        return "subtitle"
    raise SystemExit(f"Unsupported file type: {path}")


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def upload(client: Client, job: dict, files: list[Path]) -> None:
    by_name = {Path(p).name: p for p in files}
    for item in job["uploads"]:
        path = by_name[item["name"]]
        size = path.stat().st_size
        offset = int(item.get("received", 0))
        if offset >= size:
            print(f"  {path.name}: already uploaded")
            continue
        started = time.time()
        with path.open("rb") as handle:
            while offset < size:
                handle.seek(offset)
                chunk = handle.read(CHUNK)
                for attempt in range(5):
                    result = client.request(
                        "PUT", f"/api/jobs/{job['id']}/files/{item['id']}", chunk,
                        {"Upload-Offset": str(offset), "Content-Type": "application/octet-stream"})
                    new_offset = int(result["offset"])
                    if new_offset == offset + len(chunk):
                        offset = new_offset
                        break
                    # Server disagrees about our position: resync and retry.
                    offset = new_offset
                    handle.seek(offset)
                    chunk = handle.read(CHUNK)
                    time.sleep(1 + attempt)
                else:
                    raise SystemExit(f"Upload of {path.name} keeps desynchronising; giving up")
                rate = offset / max(time.time() - started, 1e-6)
                print(f"\r  {path.name}: {human(offset)} / {human(size)} "
                      f"({offset / size:.1%}, {human(rate)}/s)   ", end="", flush=True)
        print()


def choose_tracks(client: Client, job: dict, args) -> dict:
    selection = job.get("media_selection", {})
    audio = selection.get("audio_streams", [])
    subs = selection.get("subtitle_streams", [])
    print("Server needs a track choice:")
    for s in audio:
        print(f"  audio #{s.get('index')}: {s.get('language', '?')} {s.get('codec', '')} "
              f"{s.get('channels', '')}ch {s.get('title', '')}")
    for s in subs:
        print(f"  subtitle #{s.get('index')}: {s.get('language', '?')} {s.get('title', '')}")
    audio_index = args.audio_stream if args.audio_stream is not None else int(audio[0]["index"])
    subtitle_index = args.subtitle_stream
    if subtitle_index is None and subs and args.subtitle_mode in {"auto", "embedded"}:
        subtitle_index = int(subs[0]["index"])
    print(f"  -> using audio #{audio_index}" + (f", subtitle #{subtitle_index}" if subtitle_index is not None else ""))
    return client.json("POST", f"/api/jobs/{job['id']}/media-tracks",
                       {"audio_index": audio_index, "subtitle_index": subtitle_index})


def wait(client: Client, job_id: str, interval: float) -> dict:
    last = None
    announced_language = False
    while True:
        job = client.json("GET", f"/api/jobs/{job_id}")
        detected = job.get("detected_language")
        if detected and not announced_language:
            print(f"Detected spoken language: {detected.get('language')} ({float(detected.get('confidence', 0)):.0%})")
            announced_language = True
        line = f"[{job['status']}] {job.get('stage', '')} {job.get('progress', 0)}%"
        if line != last:
            print(time.strftime("%H:%M:%S"), line, flush=True)
            last = line
        if job["status"] in TERMINAL:
            return job
        if job["status"] == "paused":
            print("Job is paused on the server (review/approval workflow). Resume it in the web UI or with --approve.")
            return job
        time.sleep(interval)


def fetch_results(client: Client, job: dict, out_dir: Path, exports: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(job["filename"]).stem
    if job["status"] in {"complete", "needs_review"}:
        target = client.download(f"/api/jobs/{job['id']}/download", out_dir / f"{stem}.english.dub.mkv")
        print(f"Saved {target} ({human(target.stat().st_size)})")
        try:
            target = client.download(f"/api/jobs/{job['id']}/qc", out_dir / f"{stem}.qc.html")
            print(f"Saved {target}")
        except urllib.error.HTTPError:
            print("QC report not available")
        for kind in exports:
            ext = {"srt": "srt", "csv": "csv", "edl": "edl", "clips": "zip", "mix": "flac", "dialogue": "flac"}[kind]
            try:
                target = client.download(f"/api/jobs/{job['id']}/export/{kind}", out_dir / f"{stem}.{kind}.{ext}")
                print(f"Saved {target}")
            except urllib.error.HTTPError as exc:
                print(f"Export '{kind}' not available ({exc.code})")
    else:
        print(f"Job ended with status '{job['status']}': {job.get('error') or job.get('stage')}")
        for line in (job.get("logs") or [])[-15:]:
            print("   ", line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="one video/audio file plus optional subtitle sidecars")
    ap.add_argument("--server", default=os.getenv("DUBLINE_SERVER", "http://127.0.0.1:8000"))
    ap.add_argument("--job", help="existing job id (resume upload / wait / download)")
    ap.add_argument("--remote-path", help="path of a file already on the server (no upload)")
    ap.add_argument("--start", help="dub only from this time (seconds, MM:SS or HH:MM:SS)")
    ap.add_argument("--end", help="dub only up to this time")
    ap.add_argument("--source-language", default="auto")
    ap.add_argument("--target-language", default="English")
    ap.add_argument("--subtitle-mode", default="auto", choices=["auto", "embedded", "sidecar", "speech"])
    ap.add_argument("--audio-mode", default="separate", choices=["separate", "duck", "replace"])
    ap.add_argument("--emotion-mode", default="auto", choices=["auto", "source", "text", "neutral"])
    ap.add_argument("--workflow", default="automatic", choices=["automatic", "review", "approval"])
    ap.add_argument("--preset", default="cinema", choices=["cinema", "broadcast", "web", "preserve"],
                    help="mastering loudness preset")
    ap.add_argument("--engine", default="indextts", choices=["indextts", "preview"])
    ap.add_argument("--whisper-model", default=None)
    ap.add_argument("--glossary", type=Path, help="JSON file mapping term -> pronunciation/translation")
    ap.add_argument("--audio-stream", type=int, help="audio stream index when the file has several")
    ap.add_argument("--subtitle-stream", type=int, help="embedded subtitle stream index")
    ap.add_argument("--out", type=Path, default=Path("."), help="directory for downloaded results")
    ap.add_argument("--deliver-to", help="server-side folder name under the server's DUB_DELIVERY_DIR to copy "
                                         "the finished files into (no download needed); use with --no-download")
    ap.add_argument("--no-download", action="store_true", help="do not download results when the job finishes")
    ap.add_argument("--exports", default="", help="comma list of extra exports: srt,csv,edl,clips,mix,dialogue")
    ap.add_argument("--poll", type=float, default=10, help="status poll interval in seconds")
    ap.add_argument("--no-wait", action="store_true", help="submit and exit")
    ap.add_argument("--wait", action="store_true", help="with --job: wait for an existing job")
    ap.add_argument("--approve", action="store_true", help="with --job: resume a paused job (approve translation)")
    ap.add_argument("--status", action="store_true", help="print server status and recent jobs, then exit")
    ap.add_argument("--allow-same-language", action="store_true",
                    help="re-voice audio that is already in the target language instead of stopping")
    ap.add_argument("--i-have-the-rights", action="store_true", default=True, dest="rights",
                    help="confirm you may dub this media and reproduce its voices (default: yes)")
    args = ap.parse_args()
    client = Client(args.server)
    exports = [e.strip() for e in args.exports.split(",") if e.strip()]

    if args.status:
        info = client.json("GET", "/api/system")
        print(json.dumps(info, indent=2))
        for job in client.json("GET", "/api/jobs"):
            print(f"{job['id']}  {job['status']:<18} {job.get('progress', 0):>5}%  {job['filename']}  — {job.get('stage', '')}")
        return

    options = {
        "subtitle_mode": args.subtitle_mode, "audio_mode": args.audio_mode, "engine": args.engine,
        "emotion_mode": args.emotion_mode, "source_language": args.source_language,
        "target_language": args.target_language, "workflow_mode": args.workflow,
        "mastering_preset": args.preset, "range_start": parse_time(args.start),
        "range_end": parse_time(args.end), "audio_stream_index": args.audio_stream,
        "subtitle_stream_index": args.subtitle_stream, "voice_rights_confirmed": args.rights,
        "allow_same_language": args.allow_same_language, "delivery_dir": args.deliver_to,
        "glossary": json.loads(args.glossary.read_text(encoding="utf-8")) if args.glossary else {},
    }
    if args.whisper_model:
        options["whisper_model"] = args.whisper_model

    if args.job:
        job = client.json("GET", f"/api/jobs/{args.job}")
        if args.approve and job["status"] in {"paused", "error"}:
            job = client.json("POST", f"/api/jobs/{job['id']}/control/resume")
        if job["status"] == "uploading":
            if not args.files:
                raise SystemExit("Pass the same files again to resume this upload")
            upload(client, job, [p.resolve() for p in args.files])
            job = client.json("POST", f"/api/jobs/{job['id']}/finalize")
    elif args.remote_path:
        job = client.json("POST", "/api/jobs/local", {"path": args.remote_path, "options": options})
        print(f"Job {job['id']} created from server-side file {args.remote_path}")
    else:
        if not args.files:
            ap.error("give a video file (plus optional subtitles), --remote-path, --job or --status")
        files = [p.resolve() for p in args.files]
        for p in files:
            if not p.is_file():
                raise SystemExit(f"Not a file: {p}")
        specs = [{"name": p.name, "size": p.stat().st_size, "kind": classify(p)} for p in files]
        if sum(s["kind"] in {"video", "audio"} for s in specs) != 1:
            raise SystemExit("Provide exactly one video or audio file")
        job = client.json("POST", "/api/jobs", {"files": specs, "options": options})
        print(f"Job {job['id']} created on {client.server}; uploading {len(files)} file(s)")
        upload(client, job, files)
        job = client.json("POST", f"/api/jobs/{job['id']}/finalize")

    if job["status"] == "awaiting_selection":
        job = choose_tracks(client, job, args)
    print(f"Job {job['id']} is {job['status']}: {job.get('stage', '')}")
    if args.no_wait and not args.wait:
        print(f"Check later with: {sys.argv[0]} --server {client.server} --job {job['id']} --wait")
        return
    job = wait(client, job["id"], args.poll)
    if job.get("delivered_to"):
        print(f"Server delivered the finished files to {job['delivered_to']}")
    if not args.no_download:
        fetch_results(client, job, args.out, exports)
    elif job["status"] not in {"complete", "needs_review"}:
        print(f"Job ended with status '{job['status']}': {job.get('error') or job.get('stage')}")


if __name__ == "__main__":
    main()
