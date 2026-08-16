from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from app.services.pipeline import SAMPLE_RATE, render_timeline
from app.services.pipeline import reconcile_subtitles_with_asr
from app.services.dialogue import build_adaptive_dialogue
from app.services.adapter_worker import choose_candidate, hard_line
from app.services.qc import inspect_cues
from app.services.qc import edit_distance
from app.services.tts_worker import fit_audio
from app.services.subtitles import looks_english, parse_microdvd, parse_srt
from app.services.diarization import assign_diarized_speakers
from app.services.speakers import build_reference_bank
from app.services.visual_speakers import fuse_visual_speakers
from app.store import JobStore


def test_srt_and_language_detection(tmp_path: Path):
    path = tmp_path / "film.srt"
    path.write_text("1\n00:00:01,250 --> 00:00:03,000\n<i>What are you doing?</i>\n\n2\n00:00:04,000 --> 00:00:05,000\nI'm here.\n", encoding="utf-8")
    cues = parse_srt(path)
    assert [{key: cue[key] for key in ("start", "end", "text")} for cue in cues] == [
        {"start": 1.25, "end": 3.0, "text": "What are you doing?"},
        {"start": 4.0, "end": 5.0, "text": "I'm here."},
    ]
    assert looks_english(cues)


def test_microdvd_subtitle_timing(tmp_path: Path):
    path = tmp_path / "film.sub"
    path.write_text("{24}{48}First line|second line\n", encoding="utf-8")
    assert parse_microdvd(path, 24) == [{"start": 1.0, "end": 2.0, "text": "First line second line"}]


def test_job_store_recovery(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create({"id": "abc", "status": "processing", "stage": "Voice"})
    assert store.recover_interrupted() == ["abc"]
    assert store.get("abc")["status"] == "queued"


def test_sample_accurate_overlapping_timeline(tmp_path: Path):
    fitted = tmp_path / "fitted"
    fitted.mkdir()
    for index, amplitude in ((1, 1000), (2, 2000)):
        with wave.open(str(fitted / f"{index:06d}.wav"), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(SAMPLE_RATE)
            wav.writeframes(np.full(SAMPLE_RATE, amplitude, dtype="<i2").tobytes())
    output = tmp_path / "timeline.wav"
    render_timeline([
        {"start": 0.5, "end": 1.5},
        {"start": 1.0, "end": 2.0},
    ], fitted, output, 2.5)
    with wave.open(str(output), "rb") as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert len(samples) == round(2.5 * SAMPLE_RATE)
    assert samples[0] == 0
    assert samples[round(.75 * SAMPLE_RATE)] == 1000
    assert samples[round(1.25 * SAMPLE_RATE)] == 3000
    assert samples[round(1.75 * SAMPLE_RATE)] == 2000


def test_adaptive_dialogue_recovers_only_dropped_lines(tmp_path: Path):
    import soundfile as sf
    rate = 24_000
    primary = np.zeros(rate * 2, dtype=np.float32)
    recovery = np.zeros_like(primary)
    recovery[:rate] = 0.02 * np.sin(2 * np.pi * 180 * np.arange(rate) / rate)
    primary[rate:] = recovery[rate:] = 0.02 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)
    sf.write(tmp_path / "primary.flac", primary, rate)
    sf.write(tmp_path / "recovery.flac", recovery, rate)
    cues = [{"start": 0, "end": 1}, {"start": 1, "end": 2}]
    summary = build_adaptive_dialogue(tmp_path / "primary.flac", tmp_path / "recovery.flac",
                                      cues, tmp_path / "adaptive.flac")
    assert summary["recovered_cues"] == 1
    assert cues[0]["dialogue_source"] == "HTDemucs recovery"
    assert cues[1]["dialogue_source"] == "Bandit cinematic"


def test_subtitle_cards_use_word_timing_not_long_asr_segment():
    subtitles = [{"start": 10.0, "end": 12.0, "text": "In lane seven."}]
    asr = [{"start": 0, "end": 20, "source": "long source", "source_language": "ja",
            "transcription_confidence": .8,
            "words": [{"word": "鈴", "start": 10.2, "end": 10.5},
                      {"word": "木", "start": 10.5, "end": 10.8}]}]
    cue = reconcile_subtitles_with_asr(subtitles, asr)[0]
    assert cue["start"] == 10.2 and cue["end"] == 10.8
    assert cue["source"] == "鈴木"
    assert cue["literal_translation"] == "In lane seven."


def test_unspoken_title_and_name_cards_are_not_synthesized_as_dialogue():
    subtitles = [
        {"start": 1.0, "end": 3.0, "text": "WATERBOYS"},
        {"start": 4.0, "end": 5.0, "text": "Suzuki"},
        {"start": 6.0, "end": 7.0, "text": "Stop!"},
    ]
    cues = reconcile_subtitles_with_asr(subtitles, [])
    assert [cue["english"] for cue in cues] == ["Stop!"]


def test_qc_flags_large_stretch_and_word_mismatch(tmp_path: Path):
    import soundfile as sf
    fitted = tmp_path / "fitted"; fitted.mkdir()
    sf.write(fitted / "000001.wav", np.ones(SAMPLE_RATE, dtype=np.float32) * .02, SAMPLE_RATE)
    cues = [{"start": 0, "end": 1,
             "qc": {"stretch_percent": 12, "word_similarity": .4, "wer": .5, "cer": .4,
                    "backtranscription": "wrong", "active_duration": .2},
             "speaker_confidence": .9, "reference_quality": .02, "timing_confidence": .9,
             "transcription_confidence": .9, "adaptation_confidence": .9, "alignment_confidence": .9,
             "translation_qc": {"available": True, "passed": True}}]
    summary = inspect_cues(cues, fitted)
    assert summary["flagged_count"] == 1
    assert len(cues[0]["review_reasons"]) == 2


def test_retained_english_translation_is_not_marked_uncertain():
    selected, confidence = choose_candidate("On your mark!", ["Yo-i!"], 0.79, True)
    assert selected == "On your mark!"
    assert confidence >= 0.75


def test_semantic_judge_rejects_short_but_wrong_adaptation():
    literal = "Sato, return the festival money tomorrow."
    candidates = ["Sato, bring it tomorrow.", "Sato, pay the festival money tomorrow."]
    semantic = [
        {"index": 0, "adequacy": .98, "terminology": 1.0, "register": .9},
        {"index": 1, "adequacy": .38, "terminology": .4, "register": .9},
        {"index": 2, "adequacy": .94, "terminology": .95, "register": .9},
    ]
    selected, _ = choose_candidate(literal, candidates, 2.1, True, semantic=semantic)
    assert selected != candidates[0]


def test_timing_retry_can_limit_adaptation_to_failed_lines():
    cue = {"start": 0, "end": .5, "english": "This would normally be too long.",
           "_skip_adaptation": True}
    assert not hard_line(cue)


def test_visible_mouth_uses_tighter_timing_gate(tmp_path: Path):
    import soundfile as sf
    fitted = tmp_path / "fitted"; fitted.mkdir()
    tone = np.ones(SAMPLE_RATE, dtype=np.float32) * .02
    sf.write(fitted / "000001.wav", tone, SAMPLE_RATE)
    sf.write(fitted / "000002.wav", tone, SAMPLE_RATE)
    cues = [
        {"start": 0, "end": 1, "qc": {"stretch_percent": 6, "word_similarity": 1, "wer": 0, "cer": 0,
          "backtranscription": "ok", "active_duration": .2}, "speaker_confidence": .9,
         "reference_quality": .02, "mouth_visible": False, "timing_confidence": .9,
         "transcription_confidence": .9, "adaptation_confidence": .9, "alignment_confidence": .9,
         "translation_qc": {"available": True, "passed": True}},
        {"start": 0, "end": 1, "qc": {"stretch_percent": 6, "word_similarity": 1, "wer": 0, "cer": 0,
          "backtranscription": "ok", "active_duration": .2}, "speaker_confidence": .9,
         "reference_quality": .02, "mouth_visible": True, "timing_confidence": .9,
         "transcription_confidence": .9, "adaptation_confidence": .9, "alignment_confidence": .9,
         "translation_qc": {"available": True, "passed": True}},
    ]
    inspect_cues(cues, fitted)
    assert not cues[0]["needs_review"]
    assert cues[1]["needs_review"]
    assert "exceeds 5%" in cues[1]["review_reasons"][0]


def test_repeated_active_face_can_split_a_tentative_audio_cluster():
    cues = [
        {"id": 1, "start": 0, "end": .5, "speaker_id": 2, "speaker_confidence": .4},
        {"id": 2, "start": .6, "end": 1.1, "speaker_id": 2, "speaker_confidence": .45},
        {"id": 3, "start": 1.2, "end": 2.0, "speaker_id": 2, "speaker_confidence": .9},
    ]
    visual = {
        1: {"active_face_id": 8, "active_speaker_confidence": .96, "mouth_visible": True},
        2: {"active_face_id": 8, "active_speaker_confidence": .94, "mouth_visible": True},
        3: {"active_face_id": 5, "active_speaker_confidence": .95, "mouth_visible": True},
    }
    summary = fuse_visual_speakers(cues, visual)
    assert summary["created_visual_voices"] == 1
    assert cues[0]["speaker_id"] == cues[1]["speaker_id"] != cues[2]["speaker_id"]
    assert cues[0]["speaker_assignment"].startswith("repeated active face")


def test_silence_aware_fit_pads_short_speech_without_slowing(tmp_path: Path):
    import soundfile as sf
    rate = 24_000
    audio = np.zeros(rate, dtype=np.float32)
    audio[round(.2 * rate):round(.5 * rate)] = .08 * np.sin(
        2 * np.pi * 180 * np.arange(round(.3 * rate)) / rate)
    source = tmp_path / "short.wav"; output = tmp_path / "fitted.wav"
    sf.write(source, audio, rate)
    metrics = fit_audio(source, output, 1.0)
    fitted, fitted_rate = sf.read(output)
    assert fitted_rate == rate and len(fitted) == rate
    assert metrics["stretch_percent"] == 0
    assert metrics["padding_ms"] > 400


def test_error_rates_use_real_edit_distance():
    assert edit_distance(["in", "lane", "seven"], ["in", "line", "seven"]) == 1
    assert edit_distance(list("mark"), list("marks")) == 1


def test_index_native_rate_is_resampled_to_timeline_rate(tmp_path: Path):
    import soundfile as sf
    rate = 22_050
    values = .05 * np.sin(2 * np.pi * 190 * np.arange(rate) / rate)
    source = tmp_path / "native.wav"; output = tmp_path / "timeline.wav"
    sf.write(source, values, rate)
    fit_audio(source, output, .8)
    fitted, fitted_rate = sf.read(output)
    assert fitted_rate == SAMPLE_RATE
    assert len(fitted) == round(.8 * SAMPLE_RATE)


def test_full_context_diarization_preserves_four_scene_voices():
    cues = [{"start": 62.7, "end": 65.0}, {"start": 66.8, "end": 68.2},
            {"start": 85.0, "end": 88.4}, {"start": 99.9, "end": 100.7}]
    turns = [{"start": 62.65, "end": 65.0, "speaker": "SPEAKER_03"},
             {"start": 66.75, "end": 68.2, "speaker": "SPEAKER_02"},
             {"start": 84.96, "end": 88.42, "speaker": "SPEAKER_04"},
             {"start": 99.89, "end": 100.72, "speaker": "SPEAKER_00"}]
    assign_diarized_speakers(cues, {"diarization": turns, "exclusive_diarization": turns})
    assert [cue["speaker_id"] for cue in cues] == [1, 2, 3, 4]
    assert all(cue["speaker_assignment"] == "confident" for cue in cues)


def test_uncertain_speaker_does_not_contaminate_character_bank(tmp_path: Path):
    import soundfile as sf
    audio = tmp_path / "dialogue.wav"
    sf.write(audio, np.ones(SAMPLE_RATE * 3, dtype=np.float32) * .02, SAMPLE_RATE)
    cues = [{"start": 0.0, "end": 1.0, "speaker_id": 1, "speaker_confidence": .9,
             "reference_quality": .02, "overlapping_speech": False},
            {"start": 1.0, "end": 3.0, "speaker_id": 2, "speaker_confidence": .32,
             "reference_quality": .03, "overlapping_speech": False}]
    references = build_reference_bank(audio, cues, tmp_path / "references")
    assert set(references) == {1}
