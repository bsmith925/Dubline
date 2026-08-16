from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.diarization import assign_diarized_speakers
from app.services.speakers import analyze_speakers
from app.store import JobStore


DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the 20:20 four-speaker validation rerender")
    parser.add_argument("--source-job", default="c499b3761f65")
    parser.add_argument("--context-offset", type=float, default=60.0)
    args = parser.parse_args()

    store = JobStore(DATA / "dubstudio.sqlite3")
    original = store.get(args.source_job)
    if not original:
        raise SystemExit(f"Source job {args.source_job} does not exist")

    source_folder = Path(original["folder"]).resolve()
    if DATA.resolve() not in source_folder.parents:
        raise SystemExit("Source job folder is outside the work directory")
    context_path = DATA / "validation" / "waterboys-19m20s-3m-diarization.json"
    if not context_path.is_file():
        raise SystemExit("The verified three-minute speaker-context result is missing")

    job_id = uuid.uuid4().hex[:12]
    folder = (DATA / "jobs" / job_id).resolve()
    shutil.copytree(source_folder, folder)

    cues_path = folder / "cues.json"
    cues = json.loads(cues_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    for cue in cues:
        cue["start"] = float(cue["start"]) + args.context_offset
        cue["end"] = float(cue["end"]) + args.context_offset
    assign_diarized_speakers(cues, context)
    for cue in cues:
        cue["start"] = round(float(cue["start"]) - args.context_offset, 3)
        cue["end"] = round(float(cue["end"]) - args.context_offset, 3)
        cue["status"] = "waiting"
        cue.pop("audio", None)
        cue.pop("qc", None)
        cue.pop("acoustic_match", None)
        cue.pop("needs_review", None)
        cue.pop("review_reasons", None)

    for directory in ("speaker-references", "generated", "fitted", "acoustically-matched"):
        target = folder / directory
        if target.is_dir() and folder in target.parents:
            shutil.rmtree(target)
    for name in (
        "tts-manifest.json", "qc-asr-manifest.json", "qc-backtranscription.json",
        "english-dialogue.flac", "english-mix.flac", "dubbed-english.mkv",
        "qc-report.json", "qc-report.html",
    ):
        (folder / name).unlink(missing_ok=True)

    reference_dir = folder / "speaker-references"
    analyze_speakers(folder / "dialogue-adaptive-24k.flac", cues, reference_dir)
    cues_path.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = copy.deepcopy(original)
    for key in ("created_at", "updated_at", "output_path", "output_size", "qc",
                "qc_report_json", "qc_report_html", "error", "control"):
        payload.pop(key, None)
    payload.update({
        "id": job_id,
        "folder": str(folder),
        "status": "error",
        "stage": "Four-speaker context rerender ready",
        "progress": 34,
        "cues": cues,
        "current_cue": 0,
        "logs": [
            "Validation clone prepared from the clean 20:20 analysis",
            "Three minutes of surrounding film resolved four source voices",
            "All English performances, mixing and QC will be regenerated",
        ],
    })
    store.create(payload)
    confident = sorted({int(cue["speaker_id"]) for cue in cues
                        if float(cue.get("speaker_confidence", 0.0)) >= 0.62})
    print(json.dumps({"job_id": job_id, "voices": confident,
                      "reference_banks": len(list(reference_dir.glob("voice-*.wav"))) }))


if __name__ == "__main__":
    main()
