from __future__ import annotations

"""dubline-eval: run | compare | timeline

Examples (on the GPU host, inside vendor/index-tts/.venv):
  python -m eval.dubline_eval.cli run --suite eval/suites/core-v0.yaml --jobs clip=jobid ... --notes "baseline"
  python -m eval.dubline_eval.cli run --suite eval/suites/core-v0.yaml        # submits the suite to the server
  python -m eval.dubline_eval.cli compare eval/runs/<baseline> eval/runs/<candidate>
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(prog="dubline-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate a suite (submit jobs, or reuse finished jobs)")
    r.add_argument("--suite", type=Path, required=True)
    r.add_argument("--server", default="http://127.0.0.1:8000")
    r.add_argument("--jobs-root", type=Path, default=Path("/mnt/media/dubline/data/jobs"))
    r.add_argument("--out", type=Path, default=ROOT / "eval" / "runs")
    r.add_argument("--jobs", nargs="*", default=[], help="clip_id=job_id pairs for already finished jobs")
    r.add_argument("--notes", default="")
    r.add_argument("--seed-bundle", type=Path, default=None, help="reuse upstream job artifacts from this bundle's jobs")
    r.add_argument("--seed-tier", default="translation", choices=["asr", "translation"])
    r.add_argument("--mouth-fps", type=float, default=15.0)
    r.add_argument("--musetalk-runtime", type=Path, default=ROOT / "vendor/musetalk-env/bin/python")
    r.add_argument("--main-runtime", type=Path, default=ROOT / "vendor/index-tts/.venv/bin/python")
    r.add_argument("--whisper-cache", type=Path, default=ROOT / "vendor/whisper")
    r.add_argument("--syncnet-repo", type=Path, default=ROOT / "vendor/syncnet_python")
    c = sub.add_parser("compare", help="compare two run bundles")
    c.add_argument("baseline", type=Path); c.add_argument("candidate", type=Path)
    c.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.cmd == "run":
        from .runner import run
        jobs = dict(pair.split("=", 1) for pair in args.jobs) or None
        run(args.suite, args.server, args.jobs_root, args.out,
            {"musetalk": args.musetalk_runtime, "main": args.main_runtime, "whisper_cache": args.whisper_cache,
             "syncnet_repo": args.syncnet_repo if (args.syncnet_repo / "run_syncnet.py").is_file() else None},
            job_ids=jobs, notes=args.notes, mouth_fps=args.mouth_fps,
            seed_bundle=args.seed_bundle, seed_tier=args.seed_tier)
    else:
        from .compare import compare
        text = compare(args.baseline, args.candidate)
        if args.out:
            args.out.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
