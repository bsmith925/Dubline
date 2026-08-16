from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import whisper


def overlap(a: dict, b: dict) -> float:
    return max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    model = whisper.load_model(spec["model"], download_root=spec["cache"])
    cues = []
    files = [Path(path) for path in spec["files"]]
    source_language = spec.get("source_language") or None
    for chunk_index, path in enumerate(files):
        common = dict(language=source_language, fp16=torch.cuda.is_available(), verbose=False,
                      condition_on_previous_text=True, temperature=0, word_timestamps=True)
        source = model.transcribe(str(path), task="transcribe", **common)
        detected_language = source.get("language") or source_language
        # Turbo is optimized for transcription and explicitly not trained for
        # translation. Qwen performs contextual translation in the next CPU stage.
        translated = source
        offset = chunk_index * float(spec["chunk_seconds"])
        source_segments = source.get("segments", [])
        for segment in translated.get("segments", []):
            text = " ".join(str(segment.get("text", "")).split()).strip()
            if not text:
                continue
            matching = [item for item in source_segments if overlap(item, segment) > 0]
            source_text = " ".join(" ".join(str(x.get("text", "")).split()) for x in matching).strip() or text
            words = [{"word": str(word.get("word", "")).strip(),
                      "start": round(offset + float(word.get("start", segment["start"])), 3),
                      "end": round(offset + float(word.get("end", segment["end"])), 3),
                      "probability": round(float(word.get("probability", 0.0)), 4)}
                     for word in segment.get("words", [])]
            word_confidence = (sum(float(word.get("probability", 0.0)) for word in segment.get("words", [])) /
                               max(1, len(segment.get("words", []))))
            transcript_confidence = float(__import__('math').exp(min(0.0, segment.get("avg_logprob", -1.0))))
            cues.append({
                "start": round(offset + float(segment["start"]), 3),
                "end": round(offset + float(segment["end"]), 3),
                "source": source_text, "literal_translation": text, "english": text,
                "adapted_dialogue": text, "words": words, "source_language": detected_language,
                "translation_is_target": detected_language == "en",
                "transcription_confidence": round(transcript_confidence, 4),
                "alignment_confidence": round(word_confidence, 4),
                "timing_confidence": round(.55 * word_confidence + .45 * transcript_confidence, 4),
                "confidence_source": "Whisper token probability + word timestamps",
                "no_speech_probability": round(float(segment.get("no_speech_prob", 0.0)), 4),
            })
        args.output.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": (chunk_index + 1) / max(1, len(files)), "chunk": chunk_index}), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
