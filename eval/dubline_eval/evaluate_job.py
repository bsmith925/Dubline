from __future__ import annotations

"""Turn one finished Dubline job folder into utterance/clip records (read-only).

Runs on the GPU host next to the job folders. Uses the pipeline's own venvs for
the model-backed measurements (Whisper language-ID, FAN landmarks).
"""

import json
import re
import subprocess
from pathlib import Path

from . import audio
from .mouth import articulation_intervals, mean_aperture, mouth_series
from .schema import (ClipRecord, Identity, SourceCharacteristics, SystemMetrics, TranslationMetrics,
                     TTSMetrics, UtteranceRecord, VisualMetrics, VoiceMetrics)

LANGID_WORKER = r'''
import json, sys, whisper
paths, cache, out = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
m = whisper.load_model("turbo", device="cuda", download_root=cache)
result = {}
for p in paths:
    a = whisper.pad_or_trim(whisper.load_audio(p)); mel = whisper.log_mel_spectrogram(a, n_mels=m.dims.n_mels).to(m.device)
    _, probs = m.detect_language(mel)
    result[p] = {k: round(float(v), 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3]}
json.dump(result, open(out, "w"))
'''

SHARED_VOCAB = {"french": {"code", "internet", "web", "performance", "discussion", "question", "simple", "type",
                           "table", "film", "design", "test", "process", "message", "sport", "budget", "client",
                           "service", "standard", "important", "possible", "impossible", "direct", "final", "total",
                           "clean", "software", "developer", "bug", "commit", "agile", "manifesto", "book", "blog",
                           "tweet", "twitter", "youtube", "video"}}


def _language_ids(paths: list[Path], runtime: Path, cache_dir: Path, out: Path) -> dict[str, dict[str, float]]:
    if not paths:
        return {}
    script = out.with_suffix(".langid.py"); script.write_text(LANGID_WORKER)
    subprocess.run([str(runtime), str(script), json.dumps([str(p) for p in paths]), str(cache_dir), str(out)],
                   check=True, capture_output=True)
    return json.loads(out.read_text())


def _untranslated_rate(source: str, target: str, target_language: str) -> float | None:
    words = lambda s: [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ']{4,}", s or "")]
    src = set(words(source)); tgt = words(target)
    if not tgt:
        return None
    shared = SHARED_VOCAB.get(target_language.lower(), set())
    names = {w for w in re.findall(r"\b[A-Z][a-zà-ÿ']{3,}", source or "")}
    names = {n.lower() for n in names}
    leftovers = [w for w in tgt if w in src and w not in shared and w not in names]
    return round(len(leftovers) / len(tgt), 4)


def evaluate(job_dir: Path, clip_id: str, runtimes: dict[str, Path], work: Path,
             t_max: float | None = None, mouth_fps: float = 15.0, job: dict | None = None) -> tuple[list[UtteranceRecord], ClipRecord, dict]:
    """``job`` is the final job record from the server API (cues.json on disk predates the
    lip-sync stage and lacks its per-line results and the delivery QC)."""
    work.mkdir(parents=True, exist_ok=True)
    job = job or {}
    cues = job.get("cues") or json.loads((job_dir / "cues.json").read_text())
    qc_report = job.get("qc") or {}
    options = job.get("options") or {}
    target_language = options.get("target_language") or _guess_target(job_dir) or "English"
    duration = audio.duration(job_dir / "working-soundtrack-48k.flac")
    t_end = min(duration, t_max) if t_max else duration

    source_speech = audio.speech_intervals(job_dir / "cinema-dialogue.flac", t_max=t_end)
    dub_speech = audio.speech_intervals(job_dir / "english-dialogue.flac", thresh_db=-40, t_max=t_end)
    src_mouth = mouth_series(job_dir / "selected-source.mkv", runtimes["musetalk"], t_end, mouth_fps, work / "mouth-source.json") \
        if (job_dir / "selected-source.mkv").is_file() else []
    out_mouth = mouth_series(job_dir / "dubbed-english.mkv", runtimes["musetalk"], t_end, mouth_fps, work / "mouth-output.json") \
        if (job_dir / "dubbed-english.mkv").is_file() else []
    src_artic = articulation_intervals(src_mouth)
    out_artic = articulation_intervals(out_mouth)

    lipsync = {}
    for cfg in (job_dir / "musetalk-finishing").glob("cue-*.json") if (job_dir / "musetalk-finishing").is_dir() else []:
        cue_id = int(cfg.stem.split("-")[1])
        spec = json.loads(cfg.read_text())["task"]
        clip = Path(spec["video_path"])
        rendered = job_dir / "musetalk-finishing" / "results" / "v15" / spec["result_name"]
        probe = lambda p: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                               capture_output=True, text=True).stdout.strip() or 0)
        lipsync[cue_id] = {"submitted": probe(clip), "rendered": probe(rendered) if rendered.is_file() else None}

    takes = [job_dir / "fitted" / f"{int(c['id']):06d}.wav" for c in cues if (job_dir / "fitted" / f"{int(c['id']):06d}.wav").is_file()]
    langid = _language_ids(takes, runtimes["main"], runtimes["whisper_cache"], work / "langid.json")

    records: list[UtteranceRecord] = []
    for cue in cues:
        start, end = float(cue["start"]), float(cue["end"])
        if start >= t_end:
            continue
        q = cue.get("qc") or {}
        visual = cue.get("visual_speaker") or {}
        take = job_dir / "fitted" / f"{int(cue['id']):06d}.wav"
        take_speech = audio.speech_intervals(take, thresh_db=-40, offset=start) if take.is_file() else []
        span_src_speech = audio.clip(source_speech, start, end)
        span_src_artic = audio.clip(src_artic, start, end)
        span_dub_speech = audio.clip(dub_speech, start, end + 5.0)
        span_out_artic = audio.clip(out_artic, start, end)
        ls = lipsync.get(int(cue["id"]))
        lipsync_applied = bool(q.get("visual_lipsync"))
        lipsync_interval = [round(max(0.0, start - 0.12), 3), round(max(0.0, start - 0.12) + (ls["rendered"] or 0), 3)] if ls and ls.get("rendered") else None
        motion_on_silence = audio.total(audio.subtract(span_out_artic, span_dub_speech)) if lipsync_applied else None
        speech_on_static = audio.total(audio.subtract(audio.clip(span_dub_speech, start, end), span_out_artic)) if lipsync_applied and out_mouth else None
        coverage = (audio.total(audio.intersect(span_dub_speech, span_src_artic)) / audio.total(span_src_artic)
                    if span_src_artic else None)
        src_words = len(cue.get("words") or []) or len(str(cue.get("source", "")).split())
        tgt_words = len(str(cue.get("english", "")).split())
        records.append(UtteranceRecord(
            identity=Identity(clip_id=clip_id, utterance_id=int(cue["id"]), source_start=start, source_end=end,
                              speaker_id=cue.get("speaker_id"), source_language=cue.get("source_language"),
                              target_language=target_language, job_id=job_dir.name),
            source=SourceCharacteristics(
                source_duration=round(end - start, 3), source_speech_duration=audio.total(span_src_speech),
                source_pause_duration=round(end - start - audio.total(span_src_speech), 3), source_word_count=src_words,
                source_speech_rate_wps=round(src_words / max(0.2, audio.total(span_src_speech) or (end - start)), 2),
                mouth_visible_fraction=visual.get("mouth_visibility"), face_area_ratio=visual.get("face_area_ratio"),
                visible_faces=visual.get("visible_faces"), shot_count=None, overlapping_speech=bool(cue.get("overlapping_speech")),
                source_articulation_intervals=span_src_artic),
            translation=TranslationMetrics(
                source_text=cue.get("source", ""), target_text=cue.get("english", ""), faithful_text=cue.get("faithful_translation"),
                target_char_count=len(cue.get("english", "")), target_word_count=tgt_words,
                judge_passed=(cue.get("translation_qc") or {}).get("passed"), judge_adequacy=(cue.get("translation_qc") or {}).get("adequacy"),
                judge_reason=(cue.get("translation_qc") or {}).get("reason"),
                untranslated_word_rate=_untranslated_rate(cue.get("source", ""), cue.get("english", ""), target_language),
                candidates=list(cue.get("translation_candidates") or [])),
            tts=TTSMetrics(
                engine=q.get("tts_engine") or ("Qwen3-TTS" if (job_dir / "qwen-tts-manifest-primary-synthesis.json").is_file() else "IndexTTS"),
                clone_mode=options.get("qwen_tts_clone_mode"), raw_duration=q.get("raw_duration"), raw_speech_active=q.get("active_duration"),
                final_duration=audio.duration(take) if take.is_file() else None,
                final_speech_active=audio.total([[s - start, e - start] for s, e in take_speech]),
                stretch_percent=q.get("stretch_percent"), slowdown_percent=q.get("slowdown_percent"), padding_ms=q.get("padding_ms"),
                target_speech_rate_wps=round(tgt_words / max(0.2, audio.total(take_speech)), 2) if take_speech else None,
                backtranscription=q.get("backtranscription"), word_similarity=q.get("word_similarity"),
                language_id=langid.get(str(take)), attempts=q.get("tts_attempts")),
            voice=VoiceMetrics(speaker_similarity=q.get("speaker_similarity"), pitch_delta_semitones=q.get("pitch_delta_semitones"),
                               energy_contour_similarity=q.get("energy_contour_similarity"), pause_ratio_delta=q.get("pause_ratio_delta")),
            visual=VisualMetrics(
                lipsync_model=q.get("visual_lipsync"), lipsync_applied=lipsync_applied, lipsync_skip_reason=q.get("visual_lipsync_skipped"),
                lipsync_interval=lipsync_interval,
                lipsync_clip_length_ratio=(round(ls["rendered"] / ls["submitted"], 3) if ls and ls.get("rendered") and ls.get("submitted") else None),
                sync_confidence=None, sync_offset_frames=None,
                mouth_motion_on_silence=motion_on_silence, speech_on_static_mouth=speech_on_static,
                coverage_articulation=round(coverage, 3) if coverage is not None else None,
                identity_similarity_delta=None,
                aperture_ratio=(round(mean_aperture(out_mouth, start, end) / mean_aperture(src_mouth, start, end), 3)
                                if lipsync_applied and mean_aperture(src_mouth, start, end) and mean_aperture(out_mouth, start, end) else None)),
            system=SystemMetrics(retries=max(0, int(q.get("tts_attempts") or 1) - 1)),
            flags=list(cue.get("review_reasons") or []),
        ))

    throughput = job.get("throughput") or {}
    clip = ClipRecord(
        clip_id=clip_id, job_id=job_dir.name, source_fingerprint=_fingerprint(job_dir), target_language=target_language,
        utterance_count=len(records), wall_seconds=throughput.get("wall_seconds"), realtime_factor=throughput.get("realtime_factor"),
        delivery_qc_passed=qc_report.get("passed"), integrated_lufs=qc_report.get("integrated_lufs"), true_peak_dbtp=qc_report.get("true_peak_dbtp"),
        lipsync_rendered=sum(1 for r in records if r.visual.lipsync_applied), lipsync_eligible=sum(1 for r in records if r.visual.lipsync_applied or (r.visual.lipsync_skip_reason is None)),
        video_identity_preserved=qc_report.get("streams_preserved_exactly"),
        paths={"mkv": str(job_dir / "dubbed-english.mkv"), "voice": str(job_dir / "english-dialogue.flac"), "job": str(job_dir)},
    )
    timeline = {"t_end": t_end, "source_speech": source_speech, "dub_speech": dub_speech, "source_articulation": src_artic,
                "output_articulation": out_artic, "mouth_source": src_mouth, "mouth_output": out_mouth,
                "utterances": [{"id": r.identity.utterance_id, "start": r.identity.source_start, "end": r.identity.source_end,
                                "target_seconds": next((c.get("target_seconds") for c in cues if int(c["id"]) == r.identity.utterance_id), None),
                                "raw_tts": r.tts.raw_duration, "fill": (r.tts.final_speech_active or 0) / max(0.1, r.source.source_duration) * 100,
                                "take_speech": audio.speech_intervals(job_dir / "fitted" / f"{r.identity.utterance_id:06d}.wav", thresh_db=-40, offset=r.identity.source_start)
                                if (job_dir / "fitted" / f"{r.identity.utterance_id:06d}.wav").is_file() else []} for r in records],
                "lipsync_intervals": [{"cue_id": r.identity.utterance_id, "start": r.visual.lipsync_interval[0], "end": r.visual.lipsync_interval[1]}
                                      for r in records if r.visual.lipsync_interval],
                "asr_words": [w for c in json.loads((job_dir / "asr-dialogue-map-primary.json").read_text()) for w in c.get("words", [])]
                if (job_dir / "asr-dialogue-map-primary.json").is_file() else [],
                "asr_fragments": [{"start": c["start"], "end": c["end"]} for c in json.loads((job_dir / "asr-dialogue-map-primary.json").read_text())]
                if (job_dir / "asr-dialogue-map-primary.json").is_file() else [],
                "speaker_turns": (json.loads((job_dir / "full-film-speaker-registry" / "speaker-diarization.json").read_text()).get("diarization", [])
                                  if (job_dir / "full-film-speaker-registry" / "speaker-diarization.json").is_file() else [])}
    return records, clip, timeline


def _guess_target(job_dir: Path) -> str | None:
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream_tags=title", "-of", "csv=p=0",
                                str(job_dir / "dubbed-english.mkv")], capture_output=True, text=True).stdout.strip()
        return probe.split(" AI Dub")[0] or None
    except Exception:
        return None


def _fingerprint(job_dir: Path) -> str:
    try:
        return json.loads((job_dir / "media-fingerprint.json").read_text()).get("fingerprint", "")
    except Exception:
        return ""
