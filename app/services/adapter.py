from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

from app.config import settings
from app.services.subprocess_control import controlled_lines, terminate_process


def adapt_dialogue(cues: list[dict], folder: Path, progress: Callable[[float, int], None],
                   checkpoint: Callable[[], None], target_language: str = "English") -> list[dict]:
    model = settings.translation_model
    if not model.is_file():
        raise RuntimeError("The local Hy-MT2 scene-translation model is missing")
    manifest = folder / "adapter-manifest.json"
    output = folder / "adapted-cues.json"
    manifest.write_text(json.dumps({"cues": cues, "model": str(model), "target_language": target_language},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.adapter_worker", "--manifest", str(manifest), "--output", str(output)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-20:]
            try:
                event = json.loads(line)
                if "index" in event:
                    progress(float(event["progress"]), int(event["index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        code = process.wait()
    except BaseException:
        terminate_process(process); raise
    if code != 0 or not output.is_file():
        raise RuntimeError("Dialogue adaptation worker failed: " + "\n".join(tail[-10:]))
    return json.loads(output.read_text(encoding="utf-8"))
