from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch

from app.services.audio_fit import fit_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        spec["model"], device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    results_path = Path(spec["results"]) if spec.get("results") else None
    events: list[dict] = []
    for position, item in enumerate(spec["items"]):
        raw = Path(item["raw"]); fitted = Path(item["fitted"])
        raw.parent.mkdir(parents=True, exist_ok=True); fitted.parent.mkdir(parents=True, exist_ok=True)
        reference_text = str(item.get("reference_text") or "").strip()
        # In-context cloning (reference audio + its transcript) reproduces the
        # speaker far better than the speaker-embedding-only mode; fall back to
        # x-vector mode only when no transcript is available for the reference.
        wavs, sample_rate = model.generate_voice_clone(
            text=item["text"], language=item.get("language", "English"),
            ref_audio=item["reference"], ref_text=reference_text or None,
            x_vector_only_mode=not reference_text,
        )
        sf.write(raw, wavs[0], sample_rate)
        raw_duration = len(wavs[0]) / float(sample_rate)
        metrics = fit_audio(raw, fitted, float(item["target"]), anchors=item.get("source_runs"))
        event = {"progress": (position + 1) / max(1, len(spec["items"])),
                 "cue_index": item["cue_index"], "raw_duration": round(raw_duration, 4),
                 "attempts": 1, **metrics}
        events.append(event)
        # The model library writes progress bars to the same stream, so stdout is
        # only a hint; the results file is the record the parent relies on.
        print("\n" + json.dumps(event), flush=True)
        if results_path is not None:
            temporary = results_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(events), encoding="utf-8")
            temporary.replace(results_path)
        # This worker runs in a foreign venv without pydantic; app.config exports the value.
        time.sleep(max(0.0, float(os.getenv("DUB_GPU_LINE_COOLDOWN_SECONDS", ".45"))))
    os._exit(0)


if __name__ == "__main__":
    main()
