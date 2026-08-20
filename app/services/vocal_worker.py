from __future__ import annotations

"""Streaming HTDemucs recovery worker for dialogue missed by cinematic separation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model

from app.config import settings


OUTPUT_RATE = 48_000


def crossfade(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    length = min(len(previous), len(current))
    if length == 0:
        return np.empty((0, current.shape[1]), dtype=np.float32)
    phase = np.linspace(0.0, np.pi / 2, length, dtype=np.float32)[:, None]
    return previous[-length:] * np.cos(phase) ** 2 + current[:length] * np.sin(phase) ** 2


def global_level(source: Path) -> tuple[float, float]:
    """Match Demucs' whole-track mono mean/std without loading a long film into RAM."""
    count = 0
    total = 0.0
    total_sq = 0.0
    with sf.SoundFile(source) as reader:
        for block in reader.blocks(blocksize=OUTPUT_RATE * 60, dtype="float32", always_2d=True):
            mono = block.mean(axis=1, dtype=np.float64)
            count += len(mono)
            total += float(mono.sum())
            total_sq += float(np.square(mono).sum())
    mean = total / max(1, count)
    variance = max(1e-10, total_sq / max(1, count) - mean * mean)
    return mean, variance ** 0.5


def recover_file(source: Path, output: Path, model_name: str = "htdemucs",
                 block_seconds: int = 60, overlap_seconds: int = 2) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HTDemucs vocal recovery requires CUDA")
    model = get_model(model_name).eval().cpu()
    model_rate = int(model.samplerate)
    vocal_index = model.sources.index("vocals")
    mean, std = global_level(source)
    block_frames = block_seconds * OUTPUT_RATE
    overlap_frames = overlap_seconds * OUTPUT_RATE
    step_frames = block_frames - overlap_frames

    with sf.SoundFile(source) as reader:
        if reader.samplerate != OUTPUT_RATE:
            raise RuntimeError(f"Expected {OUTPUT_RATE} Hz film mix, got {reader.samplerate}")
        total_frames = len(reader)
        channels = reader.channels
        output.parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(output, mode="w", samplerate=OUTPUT_RATE, channels=channels,
                          format="FLAC", subtype="PCM_24") as writer:
            previous: np.ndarray | None = None
            position = 0
            block_index = 0
            total_blocks = max(1, int(np.ceil(max(0, total_frames - overlap_frames) / step_frames)))
            while position < total_frames:
                reader.seek(position)
                audio = reader.read(min(block_frames, total_frames - position),
                                    dtype="float32", always_2d=True)
                waveform = torch.from_numpy(np.ascontiguousarray(audio.T)).float()
                waveform = torchaudio.functional.resample(waveform, OUTPUT_RATE, model_rate)
                waveform = (waveform - mean) / std
                with torch.inference_mode():
                    sources = apply_model(
                        model, waveform[None], device="cuda", shifts=1, split=True,
                        overlap=0.25, progress=False, num_workers=0, segment=7,
                    )[0]
                vocal = sources[vocal_index].cpu() * std + mean
                vocal = torchaudio.functional.resample(vocal, model_rate, OUTPUT_RATE)
                if vocal.shape[-1] < len(audio):
                    vocal = torch.nn.functional.pad(vocal, (0, len(audio) - vocal.shape[-1]))
                current = vocal[:, :len(audio)].numpy().T
                final = position + len(audio) >= total_frames
                if previous is None:
                    if final:
                        writer.write(np.clip(current, -1, 1))
                    else:
                        writer.write(np.clip(current[:-overlap_frames], -1, 1))
                        previous = current[-overlap_frames:]
                else:
                    writer.write(np.clip(crossfade(previous, current), -1, 1))
                    if final:
                        writer.write(np.clip(current[overlap_frames:], -1, 1))
                    else:
                        writer.write(np.clip(current[overlap_frames:-overlap_frames], -1, 1))
                        previous = current[-overlap_frames:]
                del waveform, sources, vocal
                torch.cuda.empty_cache()
                block_index += 1
                print(json.dumps({"progress": min(1.0, block_index / total_blocks),
                                  "block": block_index, "blocks": total_blocks}), flush=True)
                time.sleep(settings.dub_gpu_block_cooldown_seconds)
                position += step_frames
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="htdemucs")
    args = parser.parse_args()
    recover_file(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
