from __future__ import annotations

"""Result schema for Dubline evaluation runs.

One run evaluates one suite (a fixed, fingerprinted list of clips) under one
configuration and writes a bundle:

    eval/runs/<run_id>/
        run.json              RunRecord (identity, config hash, model versions, suite)
        utterances.jsonl      one UtteranceRecord per utterance
        clips.jsonl           one ClipRecord per clip
        summary.md            human-readable summary
        timelines/<clip>.png  unified timeline plots
        media/<clip>/...      rendered clip + stems for inspection

Raw metrics are always preserved; aggregates are computed by compare/report,
never stored in place of the components.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Identity:
    clip_id: str
    utterance_id: int
    source_start: float
    source_end: float
    speaker_id: int | None
    source_language: str | None
    target_language: str
    job_id: str


@dataclass
class SourceCharacteristics:
    source_duration: float                      # utterance span (ASR-derived; NOT ground truth)
    source_speech_duration: float | None        # VAD-active time inside the span
    source_pause_duration: float | None
    source_word_count: int
    source_speech_rate_wps: float | None
    mouth_visible_fraction: float | None        # from visual analysis
    face_area_ratio: float | None
    visible_faces: int | None
    shot_count: int | None
    overlapping_speech: bool
    source_articulation_intervals: list[list[float]] = field(default_factory=list)  # mouth-motion-derived


@dataclass
class TranslationMetrics:
    source_text: str
    target_text: str
    faithful_text: str | None
    target_char_count: int
    target_word_count: int
    judge_passed: bool | None
    judge_adequacy: float | None
    judge_reason: str | None
    untranslated_word_rate: float | None        # share of target words that are source-language leftovers
    candidates: list[str] = field(default_factory=list)


@dataclass
class TTSMetrics:
    engine: str | None
    clone_mode: str | None
    raw_duration: float | None                  # synthesized, before fitting
    raw_speech_active: float | None
    final_duration: float | None                # after fitting
    final_speech_active: float | None
    stretch_percent: float | None               # speed-up requested (>=0)
    slowdown_percent: float | None
    padding_ms: float | None
    target_speech_rate_wps: float | None
    backtranscription: str | None
    word_similarity: float | None               # intelligibility vs intended text
    language_id: dict[str, float] | None        # {"fr": 0.98, "en": 0.01}
    attempts: int | None


@dataclass
class VoiceMetrics:
    speaker_similarity: float | None
    pitch_delta_semitones: float | None
    energy_contour_similarity: float | None
    pause_ratio_delta: float | None


@dataclass
class VisualMetrics:
    lipsync_model: str | None
    lipsync_applied: bool
    lipsync_skip_reason: str | None
    lipsync_interval: list[float] | None        # [start, end] actually rendered
    lipsync_clip_length_ratio: float | None     # rendered clip duration / submitted duration (1.0 expected)
    sync_confidence: float | None               # LSE-C-style, relative to source when available
    sync_offset_frames: float | None
    mouth_motion_on_silence: float | None       # seconds of articulation while target speech is silent
    speech_on_static_mouth: float | None        # seconds of target speech while mouth is static
    coverage_articulation: float | None         # target-speech time overlapping source articulation / source articulation time
    identity_similarity_delta: float | None     # face embedding before vs after edit
    aperture_ratio: float | None                # mean output aperture / mean source aperture in the interval


@dataclass
class SystemMetrics:
    stage_seconds: dict[str, float] = field(default_factory=dict)
    retries: int = 0
    gpu_seconds: float | None = None
    peak_vram_mb: float | None = None


@dataclass
class UtteranceRecord:
    identity: Identity
    source: SourceCharacteristics
    translation: TranslationMetrics
    tts: TTSMetrics
    voice: VoiceMetrics
    visual: VisualMetrics
    system: SystemMetrics
    flags: list[str] = field(default_factory=list)   # pipeline review reasons, verbatim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClipRecord:
    clip_id: str
    job_id: str
    source_fingerprint: str
    target_language: str
    utterance_count: int
    wall_seconds: float | None
    realtime_factor: float | None
    delivery_qc_passed: bool | None
    integrated_lufs: float | None
    true_peak_dbtp: float | None
    lipsync_rendered: int
    lipsync_eligible: int
    video_identity_preserved: bool | None
    reference_transcript_metrics: dict[str, float] = field(default_factory=dict)  # WER/chrF vs ground truth when known
    paths: dict[str, str] = field(default_factory=dict)                          # rendered outputs for inspection

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    run_id: str
    suite: str
    created: str
    git_commit: str
    config_hash: str
    config: dict[str, Any]
    models: dict[str, str]
    server: str
    clip_ids: list[str]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
