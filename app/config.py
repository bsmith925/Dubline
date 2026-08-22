from __future__ import annotations

"""Typed application settings loaded from the environment and ``.env``.

Every tunable Dubline reads lives here as a field whose name is the lower-case
form of the environment variable (``DUB_WORKDIR`` -> ``settings.dub_workdir``).
Real environment variables win over ``.env``; relative paths are resolved
against the project root so a checked-in ``.env`` works from any cwd.

The isolated model workers that run in *other* virtual environments (Qwen ASR,
pyannote, Qwen-TTS, MuseTalk) cannot import pydantic, so the server exports the
resolved settings back into ``os.environ`` once at startup
(:meth:`Settings.export_environment`); children simply inherit them.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


def venv_python(venv_dir: str | os.PathLike[str]) -> Path:
    """Interpreter path inside ``venv_dir`` for the current platform."""
    root = Path(venv_dir)
    return root / "Scripts" / "python.exe" if sys.platform == "win32" else root / "bin" / "python"


def _resolve(value: Path | str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else BASE_DIR / path).resolve()


def _resolve_runtime(value: Path | str) -> Path:
    """Accept a venv directory or an interpreter path from either platform."""
    candidate = _resolve(value)
    parts = [part.lower() for part in candidate.parts]
    is_interpreter = (len(parts) >= 2 and parts[-2] in {"scripts", "bin"}
                      and parts[-1].startswith("python"))
    if not is_interpreter:
        return venv_python(candidate)
    native = venv_python(candidate.parent.parent)
    if native != candidate and not candidate.exists() and native.exists():
        return native
    return candidate


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore",
        case_sensitive=False, validate_default=True,
    )

    # --- server -------------------------------------------------------------
    dub_engine: Literal["indextts", "qwen-tts", "preview"] = "indextts"
    dub_workdir: Path = BASE_DIR / "data"
    dub_host: str = "0.0.0.0"
    # Optional server-side delivery root: finished dubs are copied to
    # <dub_delivery_dir>/<job delivery_dir or source stem>/ as well as being downloadable.
    dub_delivery_dir: Path | None = None
    dub_port: int = Field(default=8000, ge=1, le=65535)
    hf_token: str | None = None

    # --- IndexTTS primary engine -------------------------------------------
    indextts_repo: Path = Path("vendor/index-tts")
    indextts_model_dir: Path | None = None  # defaults to <indextts_repo>/checkpoints

    # --- ASR ------------------------------------------------------------------
    whisper_model: str = "turbo"
    whisper_cache_dir: Path = Path("vendor/whisper")
    qwen_asr_model: Path = Path("vendor/qwen3-asr-0.6b-qwen")
    qwen_asr_escalation_model: Path = Path("vendor/qwen3-asr-1.7b-qwen")
    qwen_aligner_model: Path = Path("vendor/qwen3-forced-aligner-0.6b-qwen")
    qwen_asr_runtime: Path = Path("vendor/pyannote-env")

    # --- separation -------------------------------------------------------------
    bandit_repo: Path = Path("vendor/bandit-v2")
    bandit_checkpoint: Path | None = None  # defaults to <bandit_repo>/checkpoints/checkpoint-multi.ckpt
    roformer_model_dir: Path = Path("vendor/melband-roformer/melband-roformer-kim-vocals")

    # --- LLMs -------------------------------------------------------------------
    translation_model: Path = Path("vendor/hy-mt2-7b/Hy-MT2-7B-Q4_K_M.gguf")
    translation_qc_model: Path = Path("vendor/qwen3-8b/Qwen3-8B-Q4_K_M.gguf")
    dub_llama_gpu_layers: int = -1
    # Sampling per LLM role. Deterministic by default; Qwen3.x model cards suggest
    # temperature 0.7 / top_p 0.8 for non-thinking use if you prefer their defaults.
    translation_temperature: float = Field(default=0.05, ge=0, le=2)
    translation_top_p: float = Field(default=0.9, ge=0, le=1)
    translation_qc_temperature: float = Field(default=0.0, ge=0, le=2)
    translation_qc_top_p: float = Field(default=0.9, ge=0, le=1)

    # --- speakers -------------------------------------------------------------
    pyannote_model: Path = Path("vendor/pyannote-community-1")
    pyannote_runtime: Path = Path("vendor/pyannote-env")
    dub_diarization_device: Literal["cpu", "cuda"] = "cuda"
    dub_diarization_batch_size: int = Field(default=4, ge=1, le=8)
    dub_diarization_cpu_threads: int = Field(default=8, ge=1, le=12)
    opencv_face_model_dir: Path = Path("vendor/opencv-face")

    # --- TTS fallback ---------------------------------------------------------
    qwen_tts_model: Path = Path("vendor/qwen3-tts-1.7b-base")
    qwen_tts_runtime: Path = Path("vendor/qwen-tts-env")
    # Voice-clone prompt for Qwen3-TTS. "icl" (reference audio + its transcript) tracks the
    # speaker more closely but carries the reference language's phonology into the dub;
    # "xvector" (speaker embedding only) keeps the target language native. "auto" picks
    # xvector when dubbing across languages and icl when re-voicing the same language.
    qwen_tts_clone_mode: Literal["auto", "icl", "xvector"] = "auto"
    # Re-adapt and re-voice utterances whose take filled under 75% of their span.
    # Off since experiment 1 (2026-08-21): it removed 8.6 points of mean speed-up and
    # +0.08 intelligibility at no translation cost; see eval/experiments.md.
    dub_lengthen_short_takes: bool = False
    # Experiment 2: place a short take's phrases onto the source's speech runs
    # (from forced-aligned words) instead of evenly widening pauses.
    dub_place_on_source_runs: bool = False

    # --- MuseTalk lip-sync (optional) ----------------------------------------
    musetalk_enabled: bool = True
    musetalk_repo: Path = Path("vendor/MuseTalk")
    musetalk_model_dir: Path | None = None  # defaults to <musetalk_repo>/models/musetalkV15
    musetalk_runtime: Path = Path("vendor/musetalk-env")
    musetalk_max_shots: int = Field(default=0, ge=0)  # 0 = every eligible utterance
    # Lip-sync engine. EXP-LIPSYNC-001 (2026-08-21): LatentSync 1.6 doubled SyncNet confidence and
    # lifted identity 0.79 -> 0.95 vs MuseTalk at 2x runtime / 17.5 GB. Default flips after the suite run.
    lipsync_engine: Literal["musetalk", "latentsync"] = "latentsync"
    # Mastering for the web/broadcast presets: "dynamic" = one-pass loudnorm (current behaviour,
    # rides gain and lifts near-silence), "linear" = two-pass loudnorm with one static gain.
    dub_mastering_mode: Literal["dynamic", "linear", "peak_safe"] = "peak_safe"
    # peak_safe: how much the safety limiter may work on the loudest peaks (dB over the ceiling).
    dub_master_limiter_allowance_db: float = 3.0
    # MIX-004: cap on the gain that matches a take to the source performance level (was 12 dB
    # and saturated on every cue of film content: takes landed 10-17 dB under the dialogue).
    dub_level_match_max_db: float = 24.0
    # MIX-005: mix balance of the dub voice against the separated bed (stereo/mono path).
    dub_voice_gain_db: float = 0.0
    # SEP-002: rebuild the bed as film mix minus the recovery vocal stem over cues the
    # cinematic separator left out of its dialogue stem (voice otherwise plays under the dub).
    dub_adaptive_background: bool = True
    dub_duck_ratio: float = 2.0
    # EXP-AUDIO-003: mute the whole span of a re-voiced utterance, not only its aligned words.
    dub_mute_whole_utterance: bool = True
    # EXP-VIDEO-003: feed LatentSync an exact fps=25 clip instead of letting it run `ffmpeg -r 25`
    # (measured ~60 ms picture lag on 30 fps sources).
    lipsync_pre_resample_25: bool = True
    # EXP-VIDEO-004: animate the mouth over the utterance span, or only where the take is voiced.
    lipsync_extent: Literal["utterance", "voiced"] = "utterance"
    # VIDEO-007: lip-sync the dominant face track on a crop around it (multi-face shots, PiP).
    lipsync_face_crop: bool = True
    # EXP-TIMING-001: for poorly fitting takes, voice the adapter's candidates and pick by MEASURED duration.
    dub_select_by_measured_duration: bool = True
    # EXP-TRANS-001: translate one line per call (scene as context) instead of a 12-line batch.
    translation_per_line: bool = True
    # EXP-TIMING-002: adapter duration model uses measured per-language TTS speaking rates.
    adapter_language_rates: bool = False
    latentsync_repo: Path = Path("vendor/LatentSync")
    latentsync_runtime: Path = Path("vendor/latentsync-env")

    # --- GPU pacing between isolated inference calls ------------------------
    dub_gpu_line_cooldown_seconds: float = Field(default=0.45, ge=0)
    dub_gpu_block_cooldown_seconds: float = Field(default=0.6, ge=0)
    dub_gpu_qc_cooldown_seconds: float = Field(default=0.2, ge=0)
    dub_gpu_window_cooldown_seconds: float = Field(default=0.35, ge=0)

    # ------------------------------------------------------------------------
    @field_validator("dub_diarization_device", mode="before")
    @classmethod
    def _device(cls, value):
        value = str(value or "cuda").strip().lower()
        return value if value in {"cpu", "cuda"} else "cuda"

    @field_validator("dub_diarization_batch_size", "dub_diarization_cpu_threads", mode="before")
    @classmethod
    def _bounded_int(cls, value, info):
        limit = {"dub_diarization_batch_size": (4, 8), "dub_diarization_cpu_threads": (8, 12)}[info.field_name]
        try:
            return max(1, min(limit[1], int(value)))
        except (TypeError, ValueError):
            return limit[0]

    @field_validator("hf_token", mode="before")
    @classmethod
    def _blank_token(cls, value):
        value = str(value or "").strip()
        return value or None

    @field_validator(
        "dub_workdir", "indextts_repo", "whisper_cache_dir", "qwen_asr_model",
        "qwen_asr_escalation_model", "qwen_aligner_model", "bandit_repo", "roformer_model_dir",
        "translation_model", "translation_qc_model", "pyannote_model", "opencv_face_model_dir",
        "qwen_tts_model", "musetalk_repo", "latentsync_repo", mode="after",
    )
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return _resolve(value)

    @field_validator("indextts_model_dir", "bandit_checkpoint", "musetalk_model_dir", "dub_delivery_dir", mode="after")
    @classmethod
    def _optional_absolute(cls, value: Path | None) -> Path | None:
        return _resolve(value) if value is not None else None

    @field_validator("qwen_asr_runtime", "pyannote_runtime", "qwen_tts_runtime", "musetalk_runtime", "latentsync_runtime", mode="after")
    @classmethod
    def _runtime(cls, value: Path) -> Path:
        return _resolve_runtime(value)

    @model_validator(mode="after")
    def _derived_defaults(self):
        if self.indextts_model_dir is None:
            self.indextts_model_dir = self.indextts_repo / "checkpoints"
        if self.bandit_checkpoint is None:
            self.bandit_checkpoint = self.bandit_repo / "checkpoints" / "checkpoint-multi.ckpt"
        if self.musetalk_model_dir is None:
            self.musetalk_model_dir = self.musetalk_repo / "models" / "musetalkV15"
        return self

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    def export_environment(self) -> None:
        """Publish resolved values to ``os.environ`` for isolated worker processes."""
        for name, value in self.model_dump().items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "1" if value else "0"
            os.environ[name.upper()] = str(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def settings_from(environment: dict[str, str]) -> Settings:
    """Build settings from an explicit mapping (overrides the process env; ignores .env)."""
    return Settings(_env_file=None, **{key.lower(): value for key, value in environment.items()})


settings = get_settings()
