from __future__ import annotations

"""Isolated Whisper language-identification worker.

Reads a manifest of short 16 kHz mono WAV samples, scores every sample with
Whisper's language head, and writes the averaged distribution.  Runs in its
own process so the model's VRAM is released on exit like every other model.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    import torch
    import whisper
    from whisper.tokenizer import LANGUAGES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(spec.get("model", "turbo"), device=device, download_root=spec.get("cache"))
    totals: dict[str, float] = {}
    samples = []
    for path in spec["samples"]:
        audio = whisper.pad_or_trim(whisper.load_audio(path))
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        top = max(probs, key=probs.get)
        samples.append({"path": path, "language": top, "confidence": round(float(probs[top]), 4)})
        for code, value in probs.items():
            totals[code] = totals.get(code, 0.0) + float(value)
        print(json.dumps({"progress": len(samples) / max(1, len(spec["samples"]))}), flush=True)
    count = max(1, len(samples))
    ranked = sorted(((code, value / count) for code, value in totals.items()), key=lambda item: -item[1])[:5]
    best_code, best_value = ranked[0] if ranked else ("", 0.0)
    args.output.write_text(json.dumps({
        "code": best_code,
        "language": LANGUAGES.get(best_code, best_code).title(),
        "confidence": round(best_value, 4),
        "candidates": [{"code": code, "language": LANGUAGES.get(code, code).title(),
                        "confidence": round(value, 4)} for code, value in ranked],
        "samples": samples,
        "model": f"whisper-{spec.get('model', 'turbo')}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
