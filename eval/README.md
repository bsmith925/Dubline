# Dubline evaluation harness

Purpose: make every change to the dubbing pipeline measurable against a fixed corpus,
so iteration becomes `change → run suite → metrics → compare to baseline → keep/revert`.
The harness never modifies production output; it reads job artifacts and renders.

## Layout

```
eval/
  README.md                 this document (structure, schema, metric definitions)
  suites/
    core-v1.yaml            fixed clip list: source, range, source/target language,
                            fingerprint, tags (talking-head, overlap, PiP, beard, ...)
  corpus/                   fetched sources (gitignored) + fingerprints.json
  runs/<run_id>/            one bundle per run (gitignored)
    run.json                RunRecord: git commit, config hash, models, suite
    utterances.jsonl        UtteranceRecord per utterance (all raw metrics)
    clips.jsonl             ClipRecord per clip
    summary.md              human-readable summary
    timelines/<clip>.png    unified absolute-time timeline per clip
    media/<clip>/           rendered MKV, voice stem, per-utterance takes
  registry.jsonl            non-dominated configurations (quality vs compute)
  dubline_eval/             package
    schema.py               dataclasses for the records above
    corpus.py               suite loading, fetching, fingerprinting (uses yt-dlp / local files)
    collect.py              job folder → raw bundle (cues, takes, stems, video, logs)
    timeline.py             unified timeline extraction + plot (source VAD, ASR words/fragments,
                            turns, utterances, TTS take speech, dub speech, lip-sync intervals,
                            per-frame mouth aperture on source and output)
    metrics/
      audio_timing.py       speech activity (VAD), stretch/slowdown/padding, speech rate,
                            coverage of source articulation by target speech
      av_sync.py            SyncNet LSE-C/LSE-D + offset relative to the source shot;
                            mouth-motion-on-silence; speech-on-static-mouth; clip length ratio
      language.py           Whisper language-ID of each take; untranslated-word rate;
                            ASR intelligibility (back-transcription similarity)
      semantic.py           chrF / WER / content recall vs reference transcript when one exists;
                            judge verdicts recorded (never the sole signal)
      speaker.py            CAMPPlus similarity to the speaker bank; pitch/energy statistics
      visual_identity.py    SFace embedding before/after in the edited region; aperture ratio;
                            temporal consistency of the edited region
      system.py             stage latency, retries, GPU seconds, VRAM from job logs
    runner.py               submit a suite to a server, wait, collect, compute, write bundle
    compare.py              baseline vs candidate: per-metric, per-clip, per-utterance deltas,
                            worst regressions / best improvements with media paths
    report.py               summary.md and HTML report
    cli.py                  `dubline-eval run|compare|timeline|corpus`
```

`scripts/eval_autodub.py` (YouTube auto-dub round-trip) becomes a corpus source in
`corpus.py` rather than a separate harness.

## Where it runs

Metrics that need the models (Whisper language-ID, CAMPPlus, FAN landmarks, SyncNet)
run on the GPU host next to the job folders: `runner.py` is executed on the server and
invokes the pipeline's own venvs (`vendor/index-tts/.venv`, `vendor/musetalk-env`) for
those steps. `compare.py` / `report.py` only need the JSONL bundles and run anywhere.

## Distinct timing quantities (never conflated in the schema)

| quantity | where it comes from | field |
|---|---|---|
| source ASR segment / utterance span | forced aligner + merge rules | `source.source_duration` |
| source speech activity | VAD on the dialogue stem | `source.source_speech_duration`, intervals |
| speaker turn | diarization | clip timeline |
| source visible articulation | per-frame mouth aperture (FAN) | `source.source_articulation_intervals` |
| target natural speech duration | raw TTS before fitting | `tts.raw_duration`, `tts.raw_speech_active` |
| target speech after fitting | fitted take VAD | `tts.final_speech_active` |
| lip-sync generated articulation | aperture on output | `visual.aperture_ratio`, motion-on-silence |

"Coverage" is reported as three separate observables (target speech vs source articulation,
target speech vs source speech, generated articulation vs target speech), not as
`tts_duration / asr_span`.

## Metric definitions (MVP)

- **speech activity**: frame RMS (40 ms window / 20 ms hop) above −42 dBFS (source stem) or
  −40 dBFS (takes, voice stem); gaps < 120 ms merged; runs < 50 ms dropped.
- **mouth aperture**: inner-lip distance (FAN points 62–66) / face height (27→8), sampled at
  15 fps; frames with no face are null. Articulation = aperture variance in a 200 ms window
  above the clip's 30th percentile.
- **mouth_motion_on_silence** (s): time where output articulation is above threshold and target
  speech is inactive, within lip-synced intervals; reported per utterance and per clip.
- **speech_on_static_mouth** (s): time where target speech is active and output articulation is
  below threshold, within the visible-face interval.
- **coverage_articulation**: |target speech ∩ source articulation| / |source articulation| within
  the utterance span.
- **lipsync_clip_length_ratio**: rendered clip duration / submitted clip duration (expected 1.0).
- **stretch / slowdown / padding**: from `fit_audio` metrics as recorded per take.
- **language_id**: Whisper-turbo language probabilities on the fitted take.
- **untranslated_word_rate**: share of target words (≥ 4 letters, not names/numbers) that occur
  verbatim in the source text and are not in the target language's shared vocabulary list.
- **speaker_similarity**: CAMPPlus cosine vs the job's speaker bank (same embedder as QC).
- **AV sync (LSE-C/LSE-D, offset)**: original SyncNet on the output shot and on the source shot;
  reported as output minus source. Added after the MVP once SyncNet weights are provisioned.

## Comparison rules (initial, conservative)

Reject a candidate if any of: semantic fidelity (chrF vs reference, judge adequacy) drops
materially; untranslated-word rate rises; mouth-motion-on-silence or speech-on-static-mouth
increases at clip level; delivery QC or video identity regresses; any single utterance
regresses catastrophically (e.g. stretch > 50 %, language-ID flips). Otherwise a candidate
is kept if it dominates the baseline, or enters the registry as a quality/compute trade-off.
