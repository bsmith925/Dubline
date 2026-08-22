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

## EXP-AUDIO-003 — mute the whole re-voiced utterance, not only aligned word spans

- **Trigger**: audible source "uhh" at 18.2–18.6 s under French "critiquer" (exp-4, job 4bf7a9d62b26), persists in a single-track file.
- **Diagnosis**: stem − best-gain take residual = −42 dB at 18.2–18.6 s (floor −100 dB elsewhere). Aligned words: "criticize" ends 18.16, "Clean" starts 19.11 — a 0.95 s hole the aligner attached to no word, so word-span muting kept it as "non-verbal performance".
- **Hypothesis**: inside a re-voiced utterance, anything in the source dialogue stem is replaced; muting the utterance span removes these bleed-throughs without touching non-verbal material *between* utterances.
- **Variable**: `DUB_MUTE_WHOLE_UTTERANCE` (false → true). Offline A/B on identical takes (`render_timeline` only), 60 s of gb-fr.
- **New metric**: `source_residual_under_take_db` / `source_residual_seconds_above_50db` (least-squares residual of the voice stem vs the placed take, per utterance).
- **Expected**: residual-under-take → floor inside utterances; preserved source outside takes unchanged.
- **Actual**: residual seconds > −50 dB **10.8 s → 1.9 s** (7 → 6 cues affected); preserved source outside takes 0.43 s → 0.43 s (identical). The 18.2 s bleed is gone. All remaining residual is in take **overrun**: every take ends 0.1–1.05 s after its utterance (cue 7: utt 59.16–67.32, take to 68.37) so the inter-utterance breath plays under the dub tail. Worst remaining: −25.6 dB at 68.1–68.3 s.
- **Decision**: keep (default on). Remaining residual is a distinct defect — take extent exceeds utterance extent — to be characterized as EXP-AUDIO-004 (mute under the placed take extent vs constrain placement; one variable), not folded in here.

## EXP-VIDEO-002 — boundary-jump decomposition (no change)

- Method: at each lip-sync clip edge of exp-4 (job 4bf7a9d62b26, 3 clips) measure the output's frame-to-frame step vs the source's at the same edge; compare the first inside frame with the co-timed source frame (face region vs background); and estimate the temporal offset of the composite vs the source on the **background** (untouched by lip-sync) every 0.5 s through the clip.
- Result: every edge is a hard cut (output step 6.2–9.7 vs source motion 0.6–4.3). Inside every clip the composite is a **constant 2–3 frames (67–100 ms) behind the source** (bg MAD 1.3 at the best offset vs up to 21 at the same instant). Not drift: frame counts match (284/284). Cue 1's first ~2.5 s matches no source frame within ±8 (face MAD 12–29): that is the "mouth moving on its own at the beginning" region and is a separate defect.
- Cause: isolated to LatentSync's `read_video`, which converts the input with `ffmpeg -r 25`. On the 30 fps clip that conversion alone shifts content by −47…−73 ms; `-vf fps=25` is exact (±13 ms) and `-r 25` on a 25 fps input is identity (MAD 0.15).
- Decomposition: ≈ 60 ms temporal lag (LatentSync input conversion) + ≤ 1 frame resample quantization + appearance delta at the face (cue 1 entry MAD 11.7, cue 2 exit 10.6) which remains after timing is fixed.

## EXP-VIDEO-003 — pre-resample the LatentSync input to exact 25 fps

- **Variable**: `LIPSYNC_PRE_RESAMPLE_25` (false → true): hand LatentSync `fps=25` output so its own `-r 25` is identity. Nothing else changes (same seed, steps, guidance, 25→30 resample).
- **Metric**: new `render_lag_ms` per lip-synced utterance (background alignment of composite vs source; negative = output shows older content). Expected −67…−100 → within ±33 ms. Secondary: `boundary_jump_x_median` at exits should drop; sync offset unchanged or better.
- Status: implemented, awaiting a run (queued after EXP-AUDIO-003 / core-v1 re-collect).

## EXP-VIDEO-001 — 25→30 fps resample method, identical LatentSync frames (no change)

- Arms on the same raw 25 fps renders (exp-4, cues 1–2; cue 3 pending): `minterpolate mi_mode=blend` (production), `fps=30` frame-hold, `minterpolate mi_mode=dup`. Reference: the raw 25 fps render itself.
- Mouth sharpness ratio vs source — raw25 0.646 / 0.619; blend 0.692 / 0.630; hold 0.744 / 0.682; dup 0.746 / 0.682.
- Duplicated-frame fraction — blend 0.12 / 0.00; hold 0.24 / 0.17; dup 0.24 / 0.17 (raw25 already 0.088 on cue 1).
- SyncNet (vs the cue audio) — blend LSE-C 3.74 / 3.26; hold 3.72 / 3.01; dup 3.72 / 2.91; offset +2 frames in every arm.
- **Conclusion**: the blurred mouth is LatentSync's own render (≈0.62–0.65 of source sharpness before any resample). Blend costs ≤0.05 more; hold recovers ≈0.05 at the price of 17–24 % duplicated frames (visible judder) and no sync gain. **Decision: no change** (keep blend). Sharpness has to come from the renderer (resolution/face-restoration track), not from the resample.
- **Result** (core-v0 run `20260822-021438`, job ea80075555bb vs exp-4 job 4bf7a9d62b26, sequential background-alignment probe every 0.5 s): offset inside clips **−2…−3 frames → 0…−1 frame** (≤ 33 ms, the 30→25→30 quantization); co-timed bg MAD now equals best-offset MAD (1.26 / 1.26 vs 1.28 / 21.0 before at motion instants). Suite metrics flat within noise (LSE-C mean 5.19 → 5.08, n = 6; sharpness 0.743/0.959 → 0.758/0.935). Cue 1's first ~2.5 s still matches no source frame — separate defect (EXP-VIDEO-004 candidate).
- **Decision**: keep (default on). Note: first harness `render_lag_ms` used frame-index seeks and reported garbage (+267/−167 ms) on MKV; rewritten to sequential reads.

## EXP-AUDIO-004 — take extent vs utterance extent (characterization, no change)

- core-v1 re-collected baseline, 108 takes: take runs past the utterance end p50 0.34 s / p90 1.3 s / max 4.7 s (96 % > 0.1 s). Split: **trailing silence inside the take** p50 0.23 / p90 1.19 / max 6.4 s (fitting pads every take to the slot, `audio_fit.py:92-127`), and **voiced overrun** p50 0.11 / p90 0.95 / max 4.6 s (50 % of takes > 0.1 s; charade-hotel-banter cue 13: 10.4 s utterance, 15.1 s of speech).
- Trimming the trailing silence alone changes nothing audible; the voiced overrun is the translation-length problem (timing track). What is *visible* is that lip-sync animates over the utterance span regardless of where the take speaks → mouth on silence (tail padding) and frozen mouth under speech (overrun). → EXP-VIDEO-004.

## EXP-VIDEO-004 — lip-sync extent = voiced extent of the take (±120 ms)

- **Variable**: `LIPSYNC_EXTENT` utterance → voiced. The rendered clip starts 120 ms before the take's first audible speech and ends 120 ms after its last; audio for the renderer is the same take trimmed to match. Nothing else changes.
- **Metrics**: `mouth_motion_on_silence_s` and `speech_on_static_mouth_s` (expected to drop on every lip-synced cue), lip-sync count/boundary jumps (expected unchanged in count, positions move), `render_lag_ms` (must stay ≤ 33 ms), LSE-C.
- Harness: the pipeline now records `qc.visual_lipsync_interval`; the harness prefers it over the `start − 0.12 + rendered length` assumption.
- **EXP-VIDEO-004 result** (`20260822-042238-core-v0` vs `021438`): mouth_motion_on_silence mean 1.62 → 1.70 s, speech_on_static_mouth 2.13 → 2.22 s, LSE-C 5.08 → 5.30 (n=6, noise). No improvement. Confound: cue 2 of gb-fr received a 4.5 s take for a 9.2 s utterance this run (adapter duration pass chose a 22-word version; TTS spoke it at 4.9 w/s) so the source mouth talks silently for 4.5 s whichever extent is used. **Decision: revert** (`LIPSYNC_EXTENT=utterance`). Real lesson recorded under EXP-TIMING-001.

## EXP-SYNC-001 — delivered picture runs 4.03 s ahead of the audio outside lip-synced windows

- **Found by**: comparing delivered frames against the ORIGINAL source (not the job's working copy): output frame 0 == source frame 121 on every gb clip; inside lip-synced windows the offset is −2 frames. New clip-level metric `picture_offset_outside/inside_lipsync_frames` (+121 / −2 on job 849b69ea8716).
- **Cause**: range extraction `ffmpeg -i src -ss 0 -t 90 -c copy -avoid_negative_ts make_zero` drops the first video GOP (source keyframes at 0 and 4.033 s) → `selected-source.mkv` audio starts at 0, video at 4.033. The lip-sync composite does `setpts=PTS-STARTPTS` on that video (rebasing it to 0) while rendered clips are cut by timestamp and placed correctly → base picture 4 s early, lip-synced windows correct, hard cut at every window edge. Explains: "mouth moving at the beginning", boundary jumps 45–130× median, exit/entry mismatches in EXP-VIDEO-002, and why `outside_edit_fidelity` (PSNR 56 dB) saw nothing — it compared against the equally shifted working copy.
- **Fix**: input-side seek (`-ss X -i src -t D -c copy`): both streams start at the keyframe ≤ X (measured 0.033/0.033 at 0, 0.033/0.013 at 30 s); pipeline now refuses a selection whose streams start > 50 ms apart.
- **Metrics**: picture offset outside/inside → 0/0 expected; boundary_jump_max should collapse; mouth_motion_on_silence_total should drop (source mouth no longer 4 s out of phase with the dub).
