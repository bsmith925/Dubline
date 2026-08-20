from __future__ import annotations

"""Isolated Qwen3-ASR + forced-alignment worker.

The parent process supplies speech/scene windows instead of a whole soundtrack.
This prevents autoregressive ASR hallucination across long non-speech passages and
keeps every alignment request below the aligner's five-minute limit.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def join_units(values: list[str], language: str) -> str:
    values = [value.strip() for value in values if value.strip()]
    if language in {"Japanese", "Chinese", "Cantonese"}:
        return "".join(values)
    return " ".join(values)


def grouped_cues(items, language: str, offset: float, window_index: int,
                 engine: str) -> list[dict]:
    # Zero-duration items are an explicit invalid-alignment signal.  Keeping them
    # would manufacture dialogue at the beginning of silent film sections.
    # The aligner legitimately returns ``None`` for a silent/title-card window.
    # That is an empty result for this window, not a reason to abandon Qwen for
    # the whole film and fall back to a more hallucination-prone long ASR pass.
    items = list(items or [])
    valid = [item for item in items if float(item.end_time) > float(item.start_time)]
    alignment_validity = len(valid) / max(1, len(items))
    groups: list[list] = []
    current: list = []
    for item in valid:
        if current:
            gap = float(item.start_time) - float(current[-1].end_time)
            duration = float(current[-1].end_time) - float(current[0].start_time)
            punctuation = str(current[-1].text).rstrip().endswith(("。", "！", "？", ".", "!", "?"))
            if gap > 0.62 or duration > 10.0 or (punctuation and duration > 0.7):
                groups.append(current); current = []
        current.append(item)
    if current:
        groups.append(current)

    cues = []
    for group in groups:
        durations = [float(item.end_time) - float(item.start_time) for item in group]
        plausible = sum(.025 <= value <= 1.8 for value in durations) / max(1, len(durations))
        gaps = [max(0.0, float(group[i].start_time) - float(group[i - 1].end_time))
                for i in range(1, len(group))]
        continuity = 1.0 - min(1.0, sum(gap > 1.2 for gap in gaps) / max(1, len(gaps)))
        alignment_confidence = max(0.0, min(1.0,
            .5 * alignment_validity + .32 * plausible + .18 * continuity))
        words = [{"word": str(item.text).strip(),
                  "start": round(offset + float(item.start_time), 3),
                  "end": round(offset + float(item.end_time), 3),
                  "probability": round(alignment_confidence, 3)} for item in group]
        text = join_units([word["word"] for word in words], language)
        if not text:
            continue
        cues.append({
            "start": words[0]["start"], "end": words[-1]["end"],
            "source": text, "literal_translation": text, "english": text,
            "adapted_dialogue": text, "words": words,
            "source_language": language, "translation_is_target": language == "English",
            "transcription_confidence": round(.55 + .35 * alignment_confidence, 3),
            # Word boundaries come straight from the forced aligner, so timing
            # evidence tracks alignment validity (same scale as the subtitle path).
            "timing_confidence": round(.55 + .45 * alignment_confidence, 3),
            "alignment_confidence": round(alignment_confidence, 3),
            "confidence_source": "forced-alignment validity",
            "asr_engine": engine, "asr_window": window_index,
        })
    return cues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        spec["asr_model"], dtype=torch.bfloat16, device_map="cuda:0",
        max_inference_batch_size=1, max_new_tokens=512,
        forced_aligner=spec["aligner_model"],
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
    )
    cues: list[dict] = []
    windows = spec["windows"]
    model_name = Path(spec["asr_model"]).name
    engine = f"{model_name} + Qwen3 Forced Aligner 0.6B"
    for index, window in enumerate(windows):
        window_id = int(window.get("window_id", index))
        result = model.transcribe(
            audio=window["path"], language=window.get("language"), return_time_stamps=True,
        )[0]
        language = str(result.language or window.get("language") or "Unknown")
        window_cues = grouped_cues(result.time_stamps, language, float(window["offset"]), window_id, engine)
        cues.extend(window_cues)
        args.output.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": (index + 1) / max(1, len(windows)), "window": window_id,
                          "cues": len(window_cues)}), flush=True)
        # This worker runs in a foreign venv without pydantic; app.config exports the value.
        time.sleep(max(0.0, float(os.getenv("DUB_GPU_WINDOW_COOLDOWN_SECONDS", ".35"))))
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
