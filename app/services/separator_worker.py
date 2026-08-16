from __future__ import annotations

"""Isolated CUDA worker for Bandit v2 cinematic dialogue/music/effects separation."""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


SAMPLE_RATE = 48_000
STEMS = ("speech", "music", "sfx")


def load_bandit(repo: Path, checkpoint: Path, device: str):
    # Bandit's base class only uses LightningModule as an nn.Module during inference.
    # Avoid loading the repository's training/Ray stack into the production runtime.
    lightning = types.ModuleType("pytorch_lightning")
    lightning.LightningModule = torch.nn.Module
    sys.modules.setdefault("pytorch_lightning", lightning)
    sys.path.insert(0, str(repo))
    from src.models.bandit.bandit import Bandit

    model = Bandit(
        fs=SAMPLE_RATE, in_channels=1, stems=list(STEMS), band_type="musical", n_bands=64,
        normalize_channel_independently=False, treat_channel_as_feature=True,
        n_sqm_modules=8, emb_dim=128, rnn_dim=256, bidirectional=True, rnn_type="GRU",
        mlp_dim=512, hidden_activation="Tanh", hidden_activation_kwargs=None,
        complex_mask=True, use_freq_weights=True, n_fft=2048, win_length=2048,
        hop_length=512, window_fn="hann_window", wkwargs=None, power=None,
        center=True, normalized=True, pad_mode="reflect", onesided=True,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    state = {key.removeprefix("model."): value for key, value in saved.items() if key.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    significant_missing = [key for key in missing if not key.endswith(".window")]
    if significant_missing or unexpected:
        raise RuntimeError(f"Bandit checkpoint mismatch: missing={significant_missing[:5]}, unexpected={unexpected[:5]}")
    return model.eval().to(device)


def separate_block(model, audio: np.ndarray, device: str, batch_size: int) -> dict[str, np.ndarray]:
    from src.system.inference_handler import StandardTensorChunkedInferenceHandler

    handler = StandardTensorChunkedInferenceHandler(
        fs=SAMPLE_RATE, chunk_size_seconds=8.0, hop_size_seconds=1.0,
        inference_batch_size=batch_size, pad_mode="reflect",
    ).to(device)
    mixture = torch.from_numpy(np.ascontiguousarray(audio.T)).float()[None].to(device)
    # Complex masks require float32/float16 at view_as_complex. The upstream checkpoint
    # is float32; an outer BF16 autocast corrupts that boundary on current PyTorch.
    with torch.inference_mode():
        result = handler(mixture, model)["estimates"]
    stems = {name: result[name]["audio"][0].float().cpu().numpy().T for name in STEMS}
    del mixture, result, handler
    torch.cuda.empty_cache()
    return stems


def crossfade(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    length = min(len(previous), len(current))
    if length == 0:
        return np.empty((0, current.shape[1]), dtype=np.float32)
    phase = np.linspace(0.0, np.pi / 2, length, dtype=np.float32)[:, None]
    return previous[-length:] * np.cos(phase) ** 2 + current[:length] * np.sin(phase) ** 2


def separate_file(source: Path, dialogue_path: Path, background_path: Path, checkpoint: Path,
                  repo: Path, batch_size: int = 2, block_seconds: int = 60, overlap_seconds: int = 2) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Bandit cinematic separation requires CUDA")
    device = "cuda"
    model = load_bandit(repo, checkpoint, device)
    block_frames = block_seconds * SAMPLE_RATE
    overlap_frames = overlap_seconds * SAMPLE_RATE
    step_frames = block_frames - overlap_frames

    with sf.SoundFile(source) as reader:
        if reader.samplerate != SAMPLE_RATE:
            raise RuntimeError(f"Expected {SAMPLE_RATE} Hz film mix, got {reader.samplerate}")
        total_frames = len(reader)
        channels = reader.channels
        dialogue_path.parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(dialogue_path, mode="w", samplerate=SAMPLE_RATE, channels=channels,
                          format="FLAC", subtype="PCM_24") as dialogue_writer, \
             sf.SoundFile(background_path, mode="w", samplerate=SAMPLE_RATE, channels=channels,
                          format="FLAC", subtype="PCM_24") as background_writer:
            previous: dict[str, np.ndarray] | None = None
            position = 0
            block_index = 0
            total_blocks = max(1, int(np.ceil(max(0, total_frames - overlap_frames) / step_frames)))
            while position < total_frames:
                reader.seek(position)
                audio = reader.read(min(block_frames, total_frames - position), dtype="float32", always_2d=True)
                estimates = separate_block(model, audio, device, batch_size)
                current = {
                    "dialogue": estimates["speech"],
                    "background": estimates["music"] + estimates["sfx"],
                }
                final = position + len(audio) >= total_frames
                if previous is None:
                    if final:
                        dialogue_writer.write(np.clip(current["dialogue"], -1, 1))
                        background_writer.write(np.clip(current["background"], -1, 1))
                    else:
                        dialogue_writer.write(np.clip(current["dialogue"][:-overlap_frames], -1, 1))
                        background_writer.write(np.clip(current["background"][:-overlap_frames], -1, 1))
                        previous = {name: values[-overlap_frames:] for name, values in current.items()}
                else:
                    blended_dialogue = crossfade(previous["dialogue"], current["dialogue"])
                    blended_background = crossfade(previous["background"], current["background"])
                    dialogue_writer.write(np.clip(blended_dialogue, -1, 1))
                    background_writer.write(np.clip(blended_background, -1, 1))
                    if final:
                        dialogue_writer.write(np.clip(current["dialogue"][overlap_frames:], -1, 1))
                        background_writer.write(np.clip(current["background"][overlap_frames:], -1, 1))
                    else:
                        dialogue_writer.write(np.clip(current["dialogue"][overlap_frames:-overlap_frames], -1, 1))
                        background_writer.write(np.clip(current["background"][overlap_frames:-overlap_frames], -1, 1))
                        previous = {name: values[-overlap_frames:] for name, values in current.items()}
                block_index += 1
                print(json.dumps({"progress": min(1.0, block_index / total_blocks), "block": block_index,
                                  "blocks": total_blocks}), flush=True)
                time.sleep(max(0.0, float(os.getenv("DUB_GPU_BLOCK_COOLDOWN_SECONDS", ".6"))))
                position += step_frames
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dialogue", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    separate_file(args.input, args.dialogue, args.background, args.checkpoint, args.repo, args.batch_size)


if __name__ == "__main__":
    main()
