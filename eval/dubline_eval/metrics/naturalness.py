from __future__ import annotations

"""No-reference speech naturalness (MOS 1-5) via UTMOS (SpeechMOS, torch.hub).

Automatic proxy for "sounds natural"; to be validated against human calibration scores
before it carries any keep/revert decision on its own (research notes: UTMOS/NISQA).
"""

from pathlib import Path

_model = None


def naturalness_mos(wav_path: Path) -> float | None:
    global _model
    try:
        import librosa
        import torch
        if _model is None:
            _model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        wave, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        if len(wave) < 1600:
            return None
        with torch.no_grad():
            score = _model(torch.from_numpy(wave).unsqueeze(0), 16000)
        return round(float(score[0]), 3)
    except Exception:
        return None
