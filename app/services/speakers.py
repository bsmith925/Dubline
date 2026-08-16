from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def analyze_speakers(dialogue_audio: Path, cues: list[dict], output_dir: Path) -> dict[int, Path]:
    """Cluster cue-level CAMPPlus voiceprints and build a clean reference bank."""
    if not cues:
        return {}
    repo = Path(os.getenv("INDEXTTS_REPO", "vendor/index-tts")).resolve()
    model_dir = Path(os.getenv("INDEXTTS_MODEL_DIR", repo / "checkpoints")).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import torch
    import torchaudio
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(model_dir / "hf_cache" / "campplus_cn_common.bin", map_location="cpu"))
    model.eval().to(device)
    valid_indices: list[int] = []
    embeddings: list[np.ndarray] = []
    energies: dict[int, float] = {}

    with sf.SoundFile(dialogue_audio) as reader, torch.inference_mode():
        source_rate = reader.samplerate
        for index, cue in enumerate(cues):
            start = max(0, round((float(cue["start"]) - 0.12) * source_rate))
            end = min(len(reader), round((float(cue["end"]) + 0.12) * source_rate))
            if end <= start:
                continue
            reader.seek(start)
            samples = reader.read(end - start, dtype="float32", always_2d=True).mean(axis=1)
            energies[index] = float(np.sqrt(np.mean(samples * samples) + 1e-12))
            if len(samples) < source_rate * 0.65 or energies[index] < 2e-4:
                continue
            waveform = torch.from_numpy(samples)[None]
            if source_rate != 16_000:
                waveform = torchaudio.functional.resample(waveform, source_rate, 16_000)
            features = torchaudio.compliance.kaldi.fbank(
                waveform, num_mel_bins=80, dither=0, sample_frequency=16_000,
            )
            if features.shape[0] < 10:
                continue
            features = (features - features.mean(dim=0, keepdim=True)).to(device)
            embedding = model(features.unsqueeze(0)).squeeze(0).cpu().numpy()
            embedding /= np.linalg.norm(embedding) + 1e-9
            valid_indices.append(index)
            embeddings.append(embedding)

    preserve_diarization = all("speaker_id" in cue for cue in cues)
    labels = cluster_embeddings(np.stack(embeddings)) if embeddings and not preserve_diarization else np.zeros(0, dtype=int)
    if preserve_diarization:
        for index, cue in enumerate(cues):
            cue["reference_quality"] = round(energies.get(index, 0.0), 6)
        output_dir.mkdir(parents=True, exist_ok=True)
        references = build_reference_bank(dialogue_audio, cues, output_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return references
    assignments: dict[int, int] = {index: int(label) for index, label in zip(valid_indices, labels)}
    valid_assignment_indices = set(assignments)
    for index in range(len(cues)):
        assignments.setdefault(index, -1)

    # Stable, human-readable numbering follows first appearance rather than cluster internals.
    ordered: dict[int, int] = {}
    for index in range(len(cues)):
        cluster = assignments[index]
        if cluster == -1:
            cues[index]["speaker_id"] = 0
            cues[index]["speaker"] = "Uncertain voice"
            cues[index]["speaker_confidence"] = 0.0
            cues[index]["reference_quality"] = round(energies.get(index, 0.0), 6)
            continue
        if cluster not in ordered:
            ordered[cluster] = len(ordered) + 1
        speaker_id = ordered[cluster]
        assignments[index] = speaker_id
        cues[index]["speaker_id"] = speaker_id
        cues[index]["speaker"] = f"Voice {speaker_id}"
        cues[index]["speaker_confidence"] = 0.9 if index in valid_assignment_indices else 0.0
        cues[index]["reference_quality"] = round(energies.get(index, 0.0), 6)

    for index, cue in enumerate(cues):
        overlaps = any(
            other != index and min(float(cue["end"]), float(value["end"]))
            - max(float(cue["start"]), float(value["start"])) > 0.08
            for other, value in enumerate(cues)
        )
        cue["overlapping_speech"] = overlaps
        if overlaps:
            cue["speaker_confidence"] = min(float(cue.get("speaker_confidence", 0.0)), 0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    references = build_reference_bank(dialogue_audio, cues, output_dir)
    del model
    return references


def cluster_embeddings(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    # CAMPPlus cosine distance: conservative threshold avoids merging distinct actors.
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.24,
    )
    labels = clustering.fit_predict(values)
    # Avoid pathological over-segmentation caused by isolated, very short film cues.
    if len(set(labels)) > max(2, len(values) // 3):
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.34,
        )
        labels = clustering.fit_predict(values)
    return labels


def build_reference_bank(dialogue_audio: Path, cues: list[dict], output_dir: Path) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[tuple[float, int, dict]]] = defaultdict(list)
    for index, cue in enumerate(cues):
        duration = float(cue["end"]) - float(cue["start"])
        energy = float(cue.get("reference_quality", 0.0))
        # A long but wrongly assigned line is far more damaging than a short
        # reference bank.  Only clean, confident full-film assignments are
        # allowed to teach a recurring character voice.  Tentative lines still
        # retain their candidate cluster for later reconciliation, but TTS uses
        # the line's own source voice instead of contaminating another actor.
        if (int(cue.get("speaker_id", 0)) <= 0
                or float(cue.get("speaker_confidence", 0.0)) < 0.62
                or bool(cue.get("overlapping_speech"))
                or duration < 0.65):
            continue
        score = min(duration, 5.0) * np.log10(1 + energy * 10_000)
        groups[int(cue["speaker_id"])].append((score, index, cue))

    references: dict[int, Path] = {}
    with sf.SoundFile(dialogue_audio) as reader:
        rate = reader.samplerate
        for speaker_id, candidates in groups.items():
            parts = []
            seconds = 0.0
            for _, _, cue in sorted(candidates, reverse=True):
                start = max(0, round((float(cue["start"]) - 0.1) * rate))
                end = min(len(reader), round((float(cue["end"]) + 0.1) * rate))
                if end <= start:
                    continue
                reader.seek(start)
                samples = reader.read(end - start, dtype="float32", always_2d=True).mean(axis=1)
                if len(samples) < rate * 0.4:
                    continue
                parts.append(samples)
                seconds += len(samples) / rate
                if seconds >= 8.0:
                    break
                parts.append(np.zeros(round(rate * 0.08), dtype=np.float32))
            if not parts:
                continue
            montage = np.concatenate(parts)
            montage = montage[: round(rate * 12.0)]
            rms = float(np.sqrt(np.mean(montage * montage) + 1e-12))
            gain = min(12.0, 0.075 / max(rms, 1e-5))
            montage = np.clip(montage * gain, -0.95, 0.95)
            path = output_dir / f"voice-{speaker_id:03d}.wav"
            sf.write(path, montage, rate, subtype="PCM_16")
            references[speaker_id] = path
    return references


def score_speaker_similarity(cues: list[dict], generated_dir: Path,
                             references: dict[int, Path]) -> None:
    """Add CAMPPlus cosine similarity between each take and its character bank."""
    if not cues or not references:
        return
    repo = Path(os.getenv("INDEXTTS_REPO", "vendor/index-tts")).resolve()
    model_dir = Path(os.getenv("INDEXTTS_MODEL_DIR", repo / "checkpoints")).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import torch
    import torchaudio
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(model_dir / "hf_cache" / "campplus_cn_common.bin", map_location="cpu"))
    model.eval().to(device)

    def embed(path: Path) -> np.ndarray | None:
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(values.mean(axis=1))[None]
        if waveform.shape[-1] < rate * .35:
            return None
        if rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, rate, 16_000)
        features = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, dither=0,
                                                     sample_frequency=16_000)
        features = (features - features.mean(dim=0, keepdim=True)).to(device)
        with torch.inference_mode():
            value = model(features.unsqueeze(0)).squeeze(0).cpu().numpy()
        return value / (np.linalg.norm(value) + 1e-9)

    banks = {speaker_id: embed(path) for speaker_id, path in references.items() if path.is_file()}
    for index, cue in enumerate(cues, 1):
        bank = banks.get(int(cue.get("speaker_id", 0))); line = generated_dir / f"{index:06d}.wav"
        value = embed(line) if bank is not None and line.is_file() else None
        if bank is not None and value is not None:
            cue.setdefault("qc", {})["speaker_similarity"] = round(float(np.dot(bank, value)), 3)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
