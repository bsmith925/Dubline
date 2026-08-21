# Experiment log

One isolated change per entry. Bundles live in `eval/runs/`; `registry.jsonl` is the machine-readable record.

## Noise floor (core-v0)
Two runs of identical production code (`20260821-022651` vs `20260821-140505`): stretch p90 38.8 vs 46.2 %, coverage 0.76 vs 0.66, intelligibility 0.929 vs 0.894, speaker similarity 0.811 vs 0.796. Deltas smaller than these are not signal. LLM sampling, TTS sampling and retry paths are all stochastic; core-v1 should be large enough to shrink this.

## Experiment 1 — disable the "too-short" lengthening pass · keep
- **Hypothesis**: the length-control passes are the main source of extreme compression.
- **Variable**: `DUB_LENGTHEN_SHORT_TAKES` 1 → 0.
- **Result**: stretch mean 9.9 → 1.3 (p90 46 → 0, max 92 → 25); intelligibility 0.894 → 0.978; SyncNet ΔLSE-C vs source +0.39 → +0.47; judge unchanged; padding 1.1 → 2.0 s; coverage 0.66 → 0.59 (French 0.56 → 0.38); mouth-motion-on-silence 2.1 → 2.8 s (French +1.9 s); wall −10 %.
- **Conclusion**: keep. Compression is gone and speech is intelligible and in sync while it plays; what remains is that shorter target speech is front-loaded in the span and the mouth moves in silence afterwards — a placement problem, not a length problem. The pre-registered coverage rule (−0.05) was tripped but is inside the measured noise (0.10); recorded as an override.

## Experiment 2 — place take phrases on source articulation · revert
- **Hypothesis**: distributing a take's phrases onto the source's speech runs (word-timestamp groups) instead of uniform pause spreading raises coverage of source articulation and lowers mouth-motion-on-silence, with stretch, intelligibility and judge unchanged.
- **Variable**: `DUB_PLACE_ON_SOURCE_RUNS` 0 → 1.
- **Result**: coverage 0.585 → 0.484; mouth-motion-on-silence 2.83 → 3.30 s; SyncNet LSE-C 3.12 → 2.25; intelligibility 0.978 → 0.898; speaker similarity −0.024; judge unchanged; stretch 1.3 → 0.
- **Conclusion**: revert. Greedy placement scattered phrases and truncated the last one when it did not fit. Flag stays off. Phrase placement is not the dominant error; no further placement heuristics — timing work moves to the natural-speech → target timeline → visual alignment planner architecture.

## Experiment 3 — lip-sync clip extraction seeks accurately and cuts by duration · keep
- **Variable**: ffmpeg `-ss X -to Y -i` → `-accurate_seek -ss X -i -t (Y−X)` in `lipsync.py`. Reproduced: 12.77 s clip for a 10.02 s window on the first shot (clip began at t=0); new command gives 10.03 s.
- **Expected**: lipsync_clip_length_ratio_err → 0 on first shots; first-shot SyncNet offset/confidence improve; nothing else moves.

## EXP-LIPSYNC-001 — MuseTalk 1.5 vs LatentSync 1.6 on identical inputs · keep LatentSync
- Identical original 1080p frames and identical final French take for utterances 1–3 of job 77577c114226. Metrics: SyncNet LSE-C/D/offset vs source, ArcFace identity, LPIPS outside mouth mask, landmark jitter ratio, aperture ratio, mouth-motion-on-silence, runtime, peak VRAM. Artifacts: side-by-side, difference videos, metrics.json.
- **Result (timestamp-paired)**: LSE-C 2.02 → 4.07, LSE-D 10.19 → 8.37 (source 0.66/12.97); ArcFace identity 0.791 → 0.953; articulation amplitude 0.57 → 0.80 of source; LPIPS outside mouth 0.020 → 0.050; jitter ratio 1.01 → 1.21 (25 vs 30 fps inflates by ~1.2; u001 1.62 is real flicker); mouth-on-silence 1.0 → 1.2 s; runtime 93 → 171 s per ~9.5 s shot; VRAM 9.0 → 17.5 GB.
- **Conclusion**: adopt LatentSync 1.6 as the lip tier. Integration must handle 25→30 fps resampling and blend-back explicitly. Artifacts: eval/runs/lipsync-001 (side-by-side and difference videos, metrics.json).
- **Result**: clip-length ratio error 0.069 → 0.001 (every shot 1.00); all other metrics inside the noise floor. Keep.

## Experiment 4 — LIPSYNC_ENGINE=latentsync in the full pipeline · keep
- **Hypothesis**: EXP-LIPSYNC-001's gains (SyncNet ×2, identity 0.79 → 0.95, articulation 0.57 → 0.80) survive the pipeline's compositing and 25→30 fps resampling; delivery QC passes; runtime roughly doubles on lip-synced shots.
- **Variable**: `LIPSYNC_ENGINE` musetalk → latentsync (commit 0b…; default unchanged).
- **Result**: LSE-C 3.01 → 5.19, LSE-D 10.57 → 8.70, articulation 0.69 → 0.82, speech-on-static-mouth 0.19 → 0.09 s; coverage/motion-on-silence within noise; |AV offset| 0.8 → 1.8 frames (25→30 fps resample; fix by rendering at source fps or aligning the resample); wall +5 %; delivery QC pass. Default flipped to latentsync.

## core-v1 baseline · running
- First run of the frozen 10-clip suite on the accepted config (lengthening off, accurate clip seek, LatentSync). Reference for all later experiments; a second identical run establishes the noise floor.
