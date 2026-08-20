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
    for position, item in enumerate(spec["items"]):
        raw = Path(item["raw"]); fitted = Path(item["fitted"])
        raw.parent.mkdir(parents=True, exist_ok=True); fitted.parent.mkdir(parents=True, exist_ok=True)
        wavs, sample_rate = model.generate_voice_clone(
            text=item["text"], language=item.get("language", "English"),
            ref_audio=item["reference"], x_vector_only_mode=True,
        )
        sf.write(raw, wavs[0], sample_rate)
        metrics = fit_audio(raw, fitted, float(item["target"]))
        print(json.dumps({"progress": (position + 1) / max(1, len(spec["items"])),
                          "cue_index": item["cue_index"], **metrics}), flush=True)
        # This worker runs in a foreign venv without pydantic; app.config exports the value.
        time.sleep(max(0.0, float(os.getenv("DUB_GPU_LINE_COOLDOWN_SECONDS", ".45"))))
    os._exit(0)


if __name__ == "__main__":
    main()
