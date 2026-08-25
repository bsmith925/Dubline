from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from app.services.audio_fit import fit_audio


def speech_like(path: Path, seconds: float = 1.0, rate: int = 24_000) -> None:
    t = np.arange(int(seconds * rate)) / rate
    tone = (0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    sf.write(path, tone, rate, subtype="PCM_16")


def test_short_take_slows_down_by_default(tmp_path: Path):
    source = tmp_path / "take.wav"; speech_like(source)
    metrics = fit_audio(source, tmp_path / "fit.wav", target=1.5)
    assert metrics["slowdown_percent"] == 8.0


def test_dub_max_slowdown_1_pads_instead(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DUB_MAX_SLOWDOWN", "1.0")
    source = tmp_path / "take.wav"; speech_like(source)
    metrics = fit_audio(source, tmp_path / "fit.wav", target=1.5)
    assert metrics["slowdown_percent"] == 0.0
    assert metrics["padding_ms"] > 400  # the spare span became padding, not slower speech


def test_invalid_env_value_falls_back_to_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DUB_MAX_SLOWDOWN", "not-a-number")
    source = tmp_path / "take.wav"; speech_like(source)
    metrics = fit_audio(source, tmp_path / "fit.wav", target=1.5)
    assert metrics["slowdown_percent"] == 8.0
