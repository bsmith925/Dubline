from __future__ import annotations

"""Run an evaluation: submit a suite (or take finished job ids), collect, measure, write a bundle.

Runs on the GPU host (needs the job folders and the pipeline venvs).
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .evaluate_job import evaluate
from .schema import RunRecord
from .timeline import plot_timeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def server_models(server: str) -> dict:
    import urllib.request
    try:
        with urllib.request.urlopen(server.rstrip("/") + "/api/system", timeout=10) as resp:
            return json.loads(resp.read()).get("models") or {}
    except Exception:
        return {}


def fetch_job(server: str, job_id: str) -> dict:
    import urllib.request
    with urllib.request.urlopen(server.rstrip("/") + f"/api/jobs/{job_id}", timeout=30) as resp:
        return json.loads(resp.read())


def load_suite(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def seed_map(bundle: Path) -> dict[str, str]:
    import json as _json
    return {c["clip_id"]: c["job_id"] for c in (_json.loads(l) for l in open(bundle / "clips.jsonl"))}


def submit_suite(suite: dict, server: str, poll: float = 20.0, seeds: dict[str, str] | None = None, seed_tier: str = "translation") -> dict[str, str]:
    """Submit every clip to the server and wait; returns clip_id -> job_id."""
    from dubline_send import Client, wait  # scripts/dubline_send.py
    client = Client(server)
    jobs: dict[str, str] = {}
    for clip in suite["clips"]:
        options = {"source_language": clip.get("source_language", "auto"), "target_language": clip["target_language"],
                   "subtitle_mode": clip.get("subtitle_mode", "speech"), "audio_mode": "separate", "engine": "indextts",
                   "emotion_mode": "auto", "workflow_mode": "automatic", "mastering_preset": clip.get("preset", "web"),
                   "range_start": clip.get("start"), "range_end": clip.get("end"), "voice_rights_confirmed": True,
                   "glossary": clip.get("glossary") or {}, "delivery_dir": f"eval/{suite['name']}/{clip['id']}"}
        if seeds and clip["id"] in seeds:
            options["seed_from_job"] = seeds[clip["id"]]; options["seed_tier"] = seed_tier
        job = client.json("POST", "/api/jobs/local", {"path": clip["path"], "options": options})
        jobs[clip["id"]] = job["id"]
        print(f"{clip['id']} -> job {job['id']}", flush=True)
    for clip_id, job_id in jobs.items():
        job = wait(client, job_id, poll)
        print(f"{clip_id}: {job['status']} {job.get('stage', '')}", flush=True)
    return jobs


def run(suite_path: Path, server: str, jobs_root: Path, out_root: Path, runtimes: dict[str, Path],
        job_ids: dict[str, str] | None = None, notes: str = "", mouth_fps: float = 15.0,
        seed_bundle: Path | None = None, seed_tier: str = "translation") -> Path:
    suite = load_suite(suite_path)
    seeds = seed_map(seed_bundle) if seed_bundle else None
    jobs = job_ids or submit_suite(suite, server, seeds=seeds, seed_tier=seed_tier)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + suite["name"]
    out = out_root / run_id
    (out / "timelines").mkdir(parents=True, exist_ok=True)
    config = {"suite": suite, "jobs": jobs, "mouth_fps": mouth_fps}
    record = RunRecord(run_id=run_id, suite=suite["name"], created=time.strftime("%Y-%m-%dT%H:%M:%S"), git_commit=git_commit(),
                       config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
                       config=config, models=server_models(server), server=server, clip_ids=list(jobs), notes=notes)
    (out / "run.json").write_text(json.dumps(record.to_dict(), indent=2))
    failed: list[dict] = []
    with (out / "utterances.jsonl").open("w") as uf, (out / "clips.jsonl").open("w") as cf:
        for clip in suite["clips"]:
            job_id = jobs.get(clip["id"])
            if not job_id:
                continue
            job_dir = jobs_root / job_id
            job_record = fetch_job(server, job_id)
            if job_record.get("status") == "error":
                print(f"SKIPPED {clip['id']}: job {job_id} failed: {str(job_record.get('error'))[:200]}", flush=True)
                failed.append({"clip_id": clip["id"], "job_id": job_id, "error": str(job_record.get("error"))[:500]})
                continue
            (out / "work" / clip["id"]).mkdir(parents=True, exist_ok=True)
            (out / "work" / clip["id"] / "job.json").write_text(json.dumps(job_record, ensure_ascii=False))
            records, clip_record, timeline = evaluate(job_dir, clip["id"], runtimes, out / "work" / clip["id"],
                                                      mouth_fps=mouth_fps, job=job_record,
                                                      original=Path(clip["path"]), clip_start=float(clip.get("start") or 0.0))
            for r in records:
                uf.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            cf.write(json.dumps(clip_record.to_dict(), ensure_ascii=False) + "\n")
            (out / "work" / clip["id"] / "timeline.json").write_text(json.dumps(timeline))
            plot_timeline(timeline, f"{clip['id']} · job {job_id} · {record.git_commit}", out / "timelines" / f"{clip['id']}.png")
            print(f"evaluated {clip['id']}: {len(records)} utterances", flush=True)
    if failed:
        (out / "failed-jobs.json").write_text(json.dumps(failed, indent=1))
        print(f"{len(failed)} clip(s) had failed jobs (see failed-jobs.json)", flush=True)
    from .report import write_summary
    write_summary(out)
    print("bundle:", out)
    return out
