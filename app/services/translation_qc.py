from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

from app.config import settings
from app.services.subprocess_control import controlled_lines, terminate_process


def validate_translations(cues: list[dict], folder: Path,
                          progress: Callable[[float, int], None],
                          checkpoint: Callable[[], None]) -> list[dict]:
    """Run an independent bilingual judge (Qwen3), never the Hy-MT2 generator."""
    model = settings.translation_qc_model
    if not model.is_file():
        for cue in cues:
            cue["translation_qc"] = {"available": False, "passed": False,
                                     "reason": "independent bilingual QC model is missing"}
        return cues
    manifest = folder / "translation-qc-manifest.json"
    output = folder / "translation-qc.json"
    manifest.write_text(json.dumps({"model": str(model), "cues": cues}, ensure_ascii=False), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.translation_qc_worker", "--manifest", str(manifest),
         "--output", str(output)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-20:]
            try:
                event = json.loads(line)
                progress(float(event["progress"]), int(event.get("index", 0)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        code = process.wait()
    except BaseException:
        if process.poll() is None:
            terminate_process(process)
        raise
    if code != 0 or not output.is_file():
        raise RuntimeError("Independent translation QC failed: " + "\n".join(tail[-10:]))
    judged = json.loads(output.read_text(encoding="utf-8"))
    by_id = {int(item["id"]): item for item in judged}
    for cue in cues:
        cue["translation_qc"] = by_id.get(int(cue["id"]), {
            "available": True, "passed": False, "adequacy": 0.0,
            "reason": "independent judge returned no evidence",
        })
    return cues
