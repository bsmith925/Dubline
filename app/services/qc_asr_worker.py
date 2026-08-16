from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import whisper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    model = whisper.load_model("turbo", download_root=spec["cache"])
    results = []
    for index, item in enumerate(spec["items"]):
        value = model.transcribe(item["audio"], task="transcribe", language="en",
                                 fp16=torch.cuda.is_available(), temperature=0, verbose=False,
                                 condition_on_previous_text=False)
        segments = value.get("segments", [])
        confidence = (sum(math.exp(min(0.0, float(x.get("avg_logprob", -2)))) for x in segments) / len(segments)
                      if segments else 0.0)
        results.append({"text": " ".join(str(value.get("text", "")).split()),
                        "confidence": round(confidence, 4)})
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": (index + 1) / max(1, len(spec["items"])), "index": index}), flush=True)
        time.sleep(max(0.0, float(os.getenv("DUB_GPU_QC_COOLDOWN_SECONDS", ".2"))))
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
