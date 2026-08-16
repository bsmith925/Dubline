from __future__ import annotations

"""Bounded-memory MelBand-RoFormer vocal recovery worker."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from mel_band_roformer.inference import SafeLoaderWithTuple
from mel_band_roformer.utils import demix_track, get_model_from_config
from ml_collections import ConfigDict


OUTPUT_RATE = 48_000


def crossfade(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    length = min(len(previous), len(current))
    phase = np.linspace(0.0, np.pi / 2, length, dtype=np.float32)[:, None]
    return previous[-length:] * np.cos(phase) ** 2 + current[:length] * np.sin(phase) ** 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MelBand-RoFormer recovery requires CUDA")
    with args.config.open(encoding="utf-8") as stream:
        config = ConfigDict(yaml.load(stream, Loader=SafeLoaderWithTuple))
    model = get_model_from_config("mel_band_roformer", config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval().to("cuda:0")
    model_rate = int(config.model.sample_rate)
    block_frames, overlap_frames = OUTPUT_RATE * 60, OUTPUT_RATE * 2
    step = block_frames - overlap_frames

    with sf.SoundFile(args.input) as reader:
        if reader.samplerate != OUTPUT_RATE:
            raise RuntimeError(f"Expected {OUTPUT_RATE} Hz film mix")
        total = len(reader); channels = reader.channels
        args.output.parent.mkdir(parents=True, exist_ok=True)
        blocks = max(1, int(np.ceil(max(0, total - overlap_frames) / step)))
        with sf.SoundFile(args.output, "w", samplerate=OUTPUT_RATE, channels=channels,
                          format="FLAC", subtype="PCM_24") as writer:
            previous = None; position = 0; block_index = 0
            while position < total:
                reader.seek(position)
                audio = reader.read(min(block_frames, total - position), dtype="float32", always_2d=True)
                waveform = torch.from_numpy(np.ascontiguousarray(audio.T)).float()
                waveform = torchaudio.functional.resample(waveform, OUTPUT_RATE, model_rate)
                with torch.inference_mode():
                    stems, _ = demix_track(config, model, waveform, torch.device("cuda:0"))
                vocal = torch.from_numpy(np.asarray(stems["vocals"], dtype=np.float32))
                vocal = torchaudio.functional.resample(vocal, model_rate, OUTPUT_RATE)
                if vocal.shape[-1] < len(audio):
                    vocal = torch.nn.functional.pad(vocal, (0, len(audio) - vocal.shape[-1]))
                current = vocal[:, :len(audio)].cpu().numpy().T
                final = position + len(audio) >= total
                if previous is None:
                    writer.write(np.clip(current if final else current[:-overlap_frames], -1, 1))
                else:
                    writer.write(np.clip(crossfade(previous, current), -1, 1))
                    writer.write(np.clip(current[overlap_frames:] if final else current[overlap_frames:-overlap_frames], -1, 1))
                previous = None if final else current[-overlap_frames:]
                del waveform, vocal, stems
                torch.cuda.empty_cache()
                block_index += 1
                print(json.dumps({"progress": min(1.0, block_index / blocks), "block": block_index}), flush=True)
                time.sleep(max(0.0, float(os.getenv("DUB_GPU_BLOCK_COOLDOWN_SECONDS", ".6"))))
                position += step
    del model; torch.cuda.empty_cache(); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
