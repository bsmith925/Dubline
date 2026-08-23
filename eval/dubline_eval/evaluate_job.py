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
from . import audio as A
from .metrics.av_sync import score_interval
from .metrics import engineering as E
from .metrics.mix import mix_fidelity
from .metrics.video_fidelity import outside_edit_fidelity
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


def _crop_articulation(job_dir: Path, box: list, start: float, end: float, runtimes: dict, fps: float, work: Path, cue_id: int):
    x, y, side = [int(v) for v in box]
    out = []; series_out = []
    for name, video in (("output", job_dir / "dubbed-english.mkv"), ("source", job_dir / "selected-source.mkv")):
        clip = work / f"crop-{name}-{cue_id:03d}.mp4"
        if not clip.is_file():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-accurate_seek", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
                            "-an", "-vf", f"crop={side}:{side}:{x}:{y}", "-c:v", "libx264", "-crf", "18", str(clip)], check=True, capture_output=True)
        series = mouth_series(clip, runtimes["musetalk"], end - start, fps, work / f"crop-{name}-{cue_id:03d}-mouth.json")
        series_out.append([{**r, "t": r["t"] + start} for r in series])
        out.append([[a + start, b + start] for a, b in articulation_intervals(series)])
    return out[0], out[1], series_out[0], series_out[1]


def articulation_strength(out_series: list[dict], src_series: list[dict], dub_speech, src_speech) -> float | None:
    """Std of the inner-lip aperture during speech, output / source (1.0 = articulates as much as the actor)."""
    import numpy as np
    def std(series, intervals):
        vals = [r.get("inner") for r in series if r.get("inner") is not None and any(a <= r["t"] <= b for a, b in intervals)]
        return float(np.std(vals)) if len(vals) > 5 else None
    o, s = std(out_series, dub_speech), std(src_series, src_speech)
    return round(o / s, 3) if o is not None and s else None


def evaluate(job_dir: Path, clip_id: str, runtimes: dict[str, Path], work: Path,
             t_max: float | None = None, mouth_fps: float = 15.0, job: dict | None = None,
             original: Path | None = None, clip_start: float = 0.0) -> tuple[list[UtteranceRecord], ClipRecord, dict]:
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
        results_dir = job_dir / "musetalk-finishing" / "results"
        rendered = next((p for p in (results_dir / "latentsync" / spec["result_name"].replace(".mp4", "-30fps.mp4"),
                                     results_dir / "latentsync" / spec["result_name"],
                                     results_dir / "v15" / spec["result_name"]) if p.is_file()), results_dir / "v15" / spec["result_name"])
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
        crop_box = q.get("visual_lipsync_crop")
        if crop_box and q.get("visual_lipsync") and end - start >= 0.5:
            # The whole-frame mouth tracker misses small / secondary faces: track the crop the
            # renderer used, on both output and source, for this cue.
            span_out_artic, span_src_artic, cue_out_series, cue_src_series = _crop_articulation(job_dir, crop_box, start, end, runtimes, mouth_fps, work, int(cue["id"]))
        else:
            cue_out_series = [r for r in out_mouth if start <= r["t"] <= end]; cue_src_series = [r for r in src_mouth if start <= r["t"] <= end]
        ls = lipsync.get(int(cue["id"]))
        lipsync_applied = bool(q.get("visual_lipsync"))
        artic_strength = articulation_strength(cue_out_series, cue_src_series, span_dub_speech, span_src_speech) if lipsync_applied else None
        lipsync_interval = [round(max(0.0, start - 0.12), 3), round(max(0.0, start - 0.12) + (ls["rendered"] or 0), 3)] if ls and ls.get("rendered") else None
        if q.get("visual_lipsync_interval"):
            lipsync_interval = [float(v) for v in q["visual_lipsync_interval"]]
        motion_on_silence = audio.total(audio.subtract(span_out_artic, span_dub_speech)) if lipsync_applied else None
        speech_on_static = audio.total(audio.subtract(audio.clip(span_dub_speech, start, end), span_out_artic)) if lipsync_applied and out_mouth else None
        coverage = (audio.total(audio.intersect(span_dub_speech, span_src_artic)) / audio.total(span_src_artic)
                    if span_src_artic else None)
        matched_take = job_dir / "acoustically-matched" / f"{int(cue['id']):06d}.wav"
        resid = (E.source_residual_under_take(job_dir / "english-dialogue.flac", matched_take, start)
                 if matched_take.is_file() and (job_dir / "english-dialogue.flac").is_file() and not cue.get("nonverbal_filler") else {})
        sync = {}
        # Score A/V sync on every lip-synced cue (on its crop when one was used, so SyncNet sees
        # the active face) plus single-face cues that were not lip-synced (reference).
        if runtimes.get("syncnet_repo") and end - start >= 1.0 and (
                lipsync_applied or (cue.get("mouth_visible") and visual.get("visible_faces") == 1)):
            sync = score_interval(job_dir / "dubbed-english.mkv", job_dir / "selected-source.mkv", start, end - start,
                                  work / "sync", runtimes["syncnet_repo"], runtimes["musetalk"], f"u{int(cue['id']):03d}",
                                  crop=crop_box if lipsync_applied else None)
        so, ss = sync.get("output", {}), sync.get("source", {})
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
                engine=q.get("tts_engine") or (job.get("synthesis") or {}).get("engine") or ("Qwen3-TTS" if (job_dir / "qwen-tts-manifest-primary-synthesis.json").is_file() else "IndexTTS"),
                clone_mode=(job.get("synthesis") or {}).get("clone_mode"), raw_duration=q.get("raw_duration"), raw_speech_active=q.get("active_duration"),
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
                sync_lse_c=so.get("lse_c"), sync_lse_d=so.get("lse_d"), sync_offset_frames=so.get("offset_frames"),
                source_lse_c=ss.get("lse_c"), source_lse_d=ss.get("lse_d"), source_offset_frames=ss.get("offset_frames"),
                delta_lse_c=sync.get("delta_lse_c"), delta_lse_d=sync.get("delta_lse_d"),
                sync_low_conf_fraction=so.get("low_conf_fraction"),
                mouth_sharpness_ratio=(E.mouth_sharpness_ratio(job_dir / "selected-source.mkv", job_dir / "dubbed-english.mkv",
                                                               [lipsync_interval], t_end) if lipsync_applied and lipsync_interval else None),
                boundary_jump_x_median=None,
                render_lag_ms=None,   # superseded by the clip-level picture_offset (measured against the ORIGINAL source)
                source_residual_under_take_db=resid.get("max_window_db"),
                source_residual_seconds_above_50db=resid.get("seconds_above_-50db"),
                mouth_motion_on_silence=motion_on_silence, speech_on_static_mouth=speech_on_static,
                articulation_strength=artic_strength,
                coverage_articulation=round(coverage, 3) if coverage is not None else None,
                identity_similarity_delta=None,
                aperture_ratio=(round(mean_aperture(cue_out_series, start, end) / mean_aperture(cue_src_series, start, end), 3)
                                if lipsync_applied and mean_aperture(cue_src_series, start, end) and mean_aperture(cue_out_series, start, end) else None)),
            system=SystemMetrics(retries=max(0, int(q.get("tts_attempts") or 1) - 1)),
            flags=list(cue.get("review_reasons") or []),
        ))

    throughput = job.get("throughput") or {}
    ls_intervals = [r.visual.lipsync_interval for r in records if r.visual.lipsync_interval]
    jumps = E.boundary_jumps(job_dir / "dubbed-english.mkv", ls_intervals, t_end) if ls_intervals and (job_dir / "dubbed-english.mkv").is_file() else {}
    for r in records:
        if r.visual.lipsync_interval:
            a, b = r.visual.lipsync_interval
            mine = [e["jump_x_median"] for e in jumps.get("edges", []) if abs(e["t"] - a) < 0.05 or abs(e["t"] - b) < 0.05]
            r.visual.boundary_jump_x_median = max(mine) if mine else None
    dead = E.dead_air(job_dir / "working-soundtrack-48k.flac", job_dir / "english-mix.flac", t_end) if (job_dir / "english-mix.flac").is_file() else {}
    unmuted = E.unmuted_source_speech(job_dir / "cinema-dialogue.flac", cues, t_end)
    overlaps = E.take_overlaps(job_dir, cues)
    defaults = E.audio_default_tracks(job_dir / "dubbed-english.mkv")
    pic = (E.picture_offset(original, job_dir / "dubbed-english.mkv", clip_start, ls_intervals, t_end)
           if original is not None and original.is_file() and (job_dir / "dubbed-english.mkv").is_file() else {})
    (work / "picture-offset.json").write_text(json.dumps(pic))
    excess = (E.boundary_excess(original, job_dir / "dubbed-english.mkv", clip_start, pic.get("outside_lipsync_frames") or 0, ls_intervals, t_end)
              if original is not None and original.is_file() and ls_intervals and (job_dir / "dubbed-english.mkv").is_file() else {})
    (work / "boundary-excess.json").write_text(json.dumps(excess))
    # Entities: names as first-class objects (see docs/entity-track.md)
    from .metrics import entities as ENT
    src_by_cue = {int(c["id"]): str(c.get("source") or "") for c in cues if not c.get("nonverbal_filler")}
    dub_by_cue = {int(c["id"]): str(c.get("spoken_text") or c.get("english") or "") for c in cues if not c.get("nonverbal_filler")}
    heard_by_cue = {int(c["id"]): str((c.get("qc") or {}).get("backtranscription") or "") for c in cues if not c.get("nonverbal_filler")}
    known_keys = ENT.mid_sentence_keys(src_by_cue.values())
    all_mentions = [n for text in src_by_cue.values() for n in ENT.candidate_names(text, known_keys)]
    ent = {"consistency": ENT.consistency(ENT.cluster_names(all_mentions)),
           "preservation": ENT.preservation(src_by_cue, dub_by_cue),
           "tts_pronunciation": ENT.preservation(dub_by_cue, heard_by_cue)}
    (work / "entities.json").write_text(json.dumps(ent, ensure_ascii=False, indent=1))
    mos_total = A.total(A.subtract(out_artic, dub_speech)) if out_mouth else None
    mixfid = (mix_fidelity(job_dir / "working-soundtrack-48k.flac", job_dir / "english-mix.flac", dub_speech, t_end,
                           source_me=(job_dir / "cinema-background-adaptive.flac" if (job_dir / "cinema-background-adaptive.flac").is_file() else job_dir / "cinema-background.flac"),
                           source_dialogue=job_dir / "cinema-dialogue.flac")
              if (job_dir / "english-mix.flac").is_file() else {})
    if (job_dir / "english-mix-premaster.flac").is_file() and (job_dir / "english-mix.flac").is_file():
        from .metrics.mix import mastering_effect
        mixfid.update(mastering_effect(job_dir / "english-mix-premaster.flac", job_dir / "english-mix.flac", dub_speech, t_end))
    # face boxes from the FAN series (inner-lip landmarks are not stored; approximate the face box from face height)
    boxes = {}
    for r in src_mouth:
        if r.get("inner") is not None and (r.get("face_h_px") or 0) >= 50 and r.get("box"):
            boxes[r["t"]] = r["box"]
    vidfid = (outside_edit_fidelity(job_dir / "selected-source.mkv", job_dir / "dubbed-english.mkv", ls_intervals, t_end,
                                    face_boxes=boxes or None, heatmap_out=work / "diff-heatmap.png")
              if (job_dir / "dubbed-english.mkv").is_file() else {})
    clip = ClipRecord(
        clip_id=clip_id, job_id=job_dir.name, source_fingerprint=_fingerprint(job_dir), target_language=target_language,
        utterance_count=len(records), wall_seconds=throughput.get("wall_seconds"), realtime_factor=throughput.get("realtime_factor"),
        delivery_qc_passed=qc_report.get("passed"), integrated_lufs=qc_report.get("integrated_lufs"), true_peak_dbtp=qc_report.get("true_peak_dbtp"),
        lipsync_rendered=sum(1 for r in records if r.visual.lipsync_applied), lipsync_eligible=sum(1 for r in records if r.visual.lipsync_applied or (r.visual.lipsync_skip_reason is None)),
        video_identity_preserved=qc_report.get("streams_preserved_exactly"),
        paths={"mkv": str(job_dir / "dubbed-english.mkv"), "voice": str(job_dir / "english-dialogue.flac"), "job": str(job_dir)},
        dead_air_seconds=dead.get("seconds"), unmuted_source_speech_seconds=unmuted.get("seconds"),
        take_overlap_seconds=overlaps.get("seconds"), default_audio_streams=defaults.get("default_audio_streams"),
        mouth_motion_on_silence_total_s=mos_total, boundary_jump_max_x_median=jumps.get("max_jump_x_median"),
        boundary_excess_max_mad=excess.get("max_excess"),
        entity_consistency=ent["consistency"].get("entity_consistency"),
        entity_clusters_inconsistent=ent["consistency"].get("clusters_inconsistent"),
        translation_entity_preservation=ent["preservation"].get("translation_entity_preservation"),
        tts_entity_pronunciation=ent["tts_pronunciation"].get("translation_entity_preservation"),
        lipsync_coverage=(round(A.total(A.intersect(dub_speech, ls_intervals)) / A.total(dub_speech), 3) if dub_speech and ls_intervals else (0.0 if dub_speech else None)),
        picture_offset_outside_lipsync_frames=pic.get("outside_lipsync_frames"),
        picture_offset_inside_lipsync_frames=pic.get("inside_lipsync_frames"),
        picture_sync_inside_minus_outside_frames=(round(pic["inside_lipsync_frames"] - pic["outside_lipsync_frames"], 1)
                                                  if pic.get("inside_lipsync_frames") is not None and pic.get("outside_lipsync_frames") is not None else None),
        mouth_sharpness_ratio=(E.mouth_sharpness_ratio(job_dir / "selected-source.mkv", job_dir / "dubbed-english.mkv", ls_intervals, t_end)
                               if ls_intervals else None),
        mix_fidelity=mixfid, video_fidelity=vidfid,
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
