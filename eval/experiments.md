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
- **EXP-SYNC-001 result** (`20260822-050917-core-v0` vs `042238`): picture offset outside lip-sync **+121 → 0 frames** (inside −2 → −1); boundary_jump_max 79.7/56.6 → 17.6/23.3; mouth_motion_on_silence mean 1.70 → 1.13 s; low-adequacy cues (<0.7) 5 → 2. Noise-prone metrics moved: coverage 0.75 → 0.59 (noise floor 0.10), LSE-C 5.30 → 4.33 (one cue at 0.76). gb-fr cue 3 was no longer lip-synced: active-speaker confidence 0.98 → 0.68 *after* alignment — the visual speaker detector preferred the 4 s-shifted picture, which says its own timing needs checking (follow-up VIDEO-005). **Decision: keep** (objective invariant). core-v1 baseline invalid for every picture metric → re-run.

## ENG-001 — exactly one default audio track (verified offline)

- Delivered MKVs carried two default audio tracks (dub + original) on every clip (`default_audio_streams` = 2). Remux now clears the default flag on original audio tracks (all other disposition flags kept); delivery QC compares original audio streams without the default flag. Verified by re-running the remux on job from `20260822-050917`: streams → (video default), (flac fra default=1), (aac default=0); media QC failures: none. Will read 1 in the next bundles.

## MIX-001 — "bed loss" on gb-es-roundtrip was not bed loss (metric fix)

- Job from `20260822-050917`: separator put the whole source into the dialogue stem (dialogue stem −19.2 dB = source mix; background stem −94 dB — the YouTube auto-dub track has no bed). In dub-silent regions the source is at −20 dB (speaking) and the final at −63 dB: that is **source speech where our dub is silent** (short takes) being muted — a timing defect, not mixing. `mix_fidelity` now evaluates the bed only where the source dialogue stem is silent too and reports `source_speech_without_dub_s` separately.

## MIX-002 / MIX-003 — bed +5…+8 dB in bed-only regions was the limiter squashing dialogue (offline A/B)

- Characterization (core-v1 baseline-2 jobs 09585a7c849e charade-office, 376fae4dc925 thom-celia, da259f657617 gb): separation faithful (background stem = source bed within 0.2 dB); mix graph preserves the dialogue/bed ratio (premaster 17.1 vs source 17.6 dB); **mastering** applied +14.2 dB to the bed but only +11.3 dB to speech (web preset, one-pass loudnorm to −16 LUFS + limiter): speech peaks were limited ~3 dB, the quiet bed was not. Mastering mode dynamic vs linear made no difference (bed Δ 8.0 vs 7.5) — the limiter, not the gain rider.
- New metric `master_dialogue_squash_db` = (final − premaster) in dub-speech regions minus the same elsewhere (premaster now kept). Baseline: −2.9 / +0.1 / −2.3 dB.
- **Variable**: `DUB_MASTERING_MODE` dynamic → `peak_safe`: two-pass measurement, one static gain = min(gain to target, gain to the −1 dBTP ceiling + allowance). Allowance tried at 0, 1, 3 dB on the same three jobs (three values, same data — noted):
  - 0 dB: squash 0.0 / −0.1 / −0.1, LUFS −18.7 / −22.9 / −20.1
  - 1 dB: squash 0.0 / 0.05 / −0.14, LUFS −17.7 / −21.9 / −19.1
  - 3 dB: squash −0.03 / +0.37 / −0.21, LUFS −16.0 / −19.9 / −17.2 (dynamic: −15.6 / −16.7 / −17.5)
- **Decision**: keep `peak_safe` with 3 dB allowance (default). Cost: programmes with a loud bed land up to ~3 dB under the −16 LUFS web target; reported in the mastering record (`gain_limited_by_peak`, `expected_lufs`). To be confirmed on the next core-v0 run via `master_dialogue_squash_db`.

## EXP-TRANS-001 — translations assigned to the wrong cue (batch output misattribution)

- core-v1 baseline-2 (`20260822-094315`): cues with judge adequacy 0.0 went 2 → 9; on tos-lab-multispeaker cues 8–11 each carry the NEXT cue's translation (cue 9 "Why does she do this" → "Ya probamos esa opción", cue 10's line) and cues 2–3 the previous one. The shift is already present in `literal_translation`: the scene-batch translation (12 numbered lines, id-labelled JSON array) returned content labelled with the wrong line numbers. The id-based schema cannot detect this.
- Judge noise floor for reference: identical core-v0 baselines both had 6 cues < 0.7 (third run 7).
- **Variable**: `TRANSLATION_PER_LINE` false → true: one line per call with the full scene as context. Suite `trans-001` = core-v0 + tos-lab-multispeaker-es.
- **Metrics**: cues with adequacy == 0 (misattribution), cues < 0.7, translation-QC failures, word_similarity, wall time (cost). Expected: misattributions → 0, adequacy ≥ baseline; cost ≈ +N LLM calls of a 7B model.
- **core-v0 reference-2** (`20260822-105241`): confirms ENG-001 (default audio tracks 2 → 1) and MIX-003 (`master_dialogue_squash_db` −0.3 / −0.24) in a full run. `bed_rms_delta_db` on gb-fr reads +9.7 dB: that is the uniform programme gain (+19 dB on a −32 dB source) lifting room tone, not bed damage; added `bed_ratio_delta_db` (dialogue−bed ratio, output minus source) as the gain-independent check.

## EXP-TIMING-001 — choose the phrasing by MEASURED duration (result)

- `20260822-113541-core-v0` vs reference-2 `105241`. 8 poorly fitting takes (5 FR, 3 ES) had 4–6 candidates voiced and measured; all 8 switched. gb-fr active-fill p10/p50/p90 **37/62/85 → 57/77/90 %**, cues under 60 % fill 4 → 2 (ES 0 → 0); padding p90 6468 → 3704 ms; mouth-on-silence total FR 10.3 → 12.8 s (n.b. counts source mouth outside lip-synced cues; noisy), LSE-C 4.31 → 4.73; judge adequacy mean 0.72 → 0.69 (within noise; candidates are adequacy-gated ≥ 0.65); wall +22 % FR / +8 % ES.
- Flagged but not caused by the variable: word_similarity 0.43 on FR cue 7 (not switched) is "quinze à vingt-cinq" vs back-transcribed "15 à 25" — metric needs number normalization; stretch 9.6 % on the same off-screen cue is run variance.
- **Decision: keep** (`DUB_SELECT_BY_MEASURED_DURATION=true`). Next on this track: extend candidate measurement to all takes (not only < 75 % fill) once cost is acceptable, and feed measured rates back into the adapter's word target.

## QC-001 — back-transcription normalizer (metric change, affects `word_similarity` from here on)

- `normalize_words` tokenized with `[a-z']+` (dropping accented letters, splitting "réécrire" → "r crire") and spelled digits in English only ("15" → "one five"). Now unicode-aware and spells numbers in the target language (en/fr/es, 0–999 999; "%" → "pour cent"/"por ciento"). Expect `word_similarity` to rise for fr/es and the FR cue-7 false 0.43 to disappear. Not a pipeline-behaviour change, but QC review flags that depend on similarity will shift.
- **EXP-TRANS-001 result** (`20260822-123139-trans-001` vs core-v1 baseline-2 for tos-lab and TIMING-001 bundle for gb): judge adequacy == 0 (misattributed) tos-lab **6 → 0**, gb-es **3 → 0**, gb-fr 0 → 0; cues < 0.7: 7 → 1, 5 → 1, 1 → 1; mean adequacy 0.62 → 0.94, 0.53 → 0.93, 0.87 → 0.82 (one cue, noise); wall 1053 → 972 s, 1042 → 972 s, 982 → 796 s (per-line calls are not slower than the batch + retries). **Decision: keep** (`TRANSLATION_PER_LINE=true`).

## VIDEO-005 — boundary jumps on charade-office / gb-screenshare were (a) the metric and (b) a 1-frame placement lag

- `boundary_jump_x_median` divides the edge step by the clip's median motion: on a static screen-share any edit reads ×89; on charade-office cue 8 the ×94 is the source's own shot cut at the window's exit frame (source step 55, output 55). The metric must subtract the source's step at the same (offset-aligned) frames — TODO.
- With the selection offset applied, inside the window the output's motion equals the source's (13 vs 13 per frame) but frames differ by ~20 MAD → a ≈1-frame temporal lag of the rendered clip. The render itself is exact vs its own source clip (0 ms for blend, fps-drop and mi_dup). The lag is in **compositing**: the clip is placed at the raw utterance time, off the frame grid.
- Offline A/B of placement on the existing renders (`picture_offset` inside vs outside, original as reference): charade 23.976 fps current −66 vs outside −65; floor −65; ceil −66; −1 frame −65; +1 −66.5. gb 30 fps: current [−1,0,0], floor [0,0,1].
- **VIDEO-006**: snap the clip start down to the base frame grid. Adopted (objective placement fix); confirmed by `picture_sync_inside_minus_outside_frames` in baseline-3+.

## MIX-004 / MIX-005 — dub voice too quiet on film content (user report on tos-lab, confirmed)

- `dialogue_to_bed_snr_db`: gb talking-head +24…+40 dB; film clips −4.8 (tos-lab), −0.8 (charade-funeral), +1.2 (rooftop), +1.5 (thom-celia), +3.3 (hotel). `bed_ratio_delta_db` tos-lab −5.9 (balance 5.9 dB worse than the source).
- Per cue on tos-lab (job faa3e3f78bd6): takes 10–17 dB under the source dialogue they replace (voiced vs voiced, median −10.5 dB) and `level_match_db` = 12.04 on 17/17 cues — the level matcher's 4× cap is saturated everywhere (second cap of 2× in the stereo placement too). Offline cap sweep 12/24/36 dB: saturated 17/17 → 11/17 → 1/17; voice stem in speech regions −22.7 → −21.0 dB (= source dialogue stem −21.3). Dialogue-to-bed SNR only −4.5 → −3.5 because in this film the separated bed (−19.8) is louder than the dialogue stem: matching the source level is necessary but not sufficient for a TTS voice.
- MIX-005 ladder (voice gain / duck ratio, offline, MKVs for listening in ~/Downloads/dubline/mix-005-ab): A current −3.8 dB; B +4 dB −2.7; C +8 dB −0.4; D duck 4:1 −4.7; E +4 & 4:1 −3.3. Ducking alone does not help. Awaiting the listener's pick before setting a default (preference, not an invariant).
- Also seen: cues 10–12 have no source dialogue in the dialogue stem while the bed is −14 dB there → separation put those lines into the bed (SEP-001, to characterize: unmuted source speech through the bed).

## VIDEO-007 — active-track lip-sync on a face crop (multi-face shots, picture-in-picture)

- Finding: all `mouth_motion_on_silence` lives in cues the lip-sync gate skipped (inside lip-synced cues: 0 s on every clip). Gate skips most cues: gb talking-head 5/9 "mouth not visible" — after 32 s the frame holds gingerBill's PiP plus a static face in the slides; the scorer multiplies confidence by `single_face_ratio/0.60`, so two coexisting tracks → confidence 0 → `mouth_visible=False` although `mouth_visibility` is 1.0 and his mouth motion is the only one. Film two-shots fail the same way ("not exactly one face": tos-lab 9/17, charade-funeral 8/9 skipped).
- **Variable**: `LIPSYNC_FACE_CROP` false → true: (1) the visual worker records the dominant track's median box and, with several tracks, accepts the ranking when one mouth dominates (dominance = 1 − second/best ≥ 0.8 instead of requiring a single face); (2) lip-sync renders a square crop (1.8× the face box) around that track and the composite pastes it back at the same position, so the renderer cannot animate another face and small faces get more pixels.
- **Metrics**: lip-synced cue count, LSE-C/LSE-D on the newly lip-synced cues (wrong-face rendering would show as low LSE-C), `mouth_motion_on_silence_total_s`, mouth sharpness, identity, boundary (paste seam) via outside-edit PSNR inside lip-sync.

## SEP-001 / SEP-002 — lines the cinematic separator dropped stay in its background stem

- core-v1 baseline-2: 16 cues across 6 clips have a digitally silent Bandit dialogue stem (−105…−119 dB) while the bed is −13…−30 dB there (tos-lab 5/17, office 4/18, rooftop 3/7). The confidence gate already routes those cues to RoFormer/HTDemucs recovery for references, but the mix kept Bandit's background → that voice plays under the dub, unmuted.
- SEP-002 (`DUB_ADAPTIVE_BACKGROUND`, default on): over such cues the bed becomes film mix − the chosen vocal stem, 35 ms fades. Offline on tos-lab job 751aed49ca62: 5 cues rebuilt; bed over them −14.0 → −17.1, −16.1 → −17.6, −30.2 → −31.8 dB (small: these spans are loud effects, which is why Bandit lost the voice); elsewhere max |Δ| = 0.0. Objective, harmless, limited benefit on this corpus. Harness reads the adaptive bed when present.

## core-v1 baseline-3 (`20260822-153143-core-v1`) vs baseline-2 (`094315`)

| metric | baseline-2 | baseline-3 |
|---|---|---|
| judge adequacy == 0 (misattributed) | 9 | **0** |
| judge adequacy < 0.7 | 24 | 15 |
| judge adequacy mean | 0.81 | 0.89 |
| padding p90 / max (ms) | 2110 / 6686 | 1253 / 9617 |
| stretch p90 (%) | 13.4 | 5.8 |
| dead air gb-fr / gb-es (s) | 14.3 / 14.9 | 7.3 / 8.8 |
| master dialogue squash (dB) | −2.9…+0.1 | −0.4…+0.6 |
| default audio tracks | 2 | 1 |
| speaker similarity mean | 0.67 | 0.70 |
| LSE-C mean (n=15) | 4.14 | 4.57 |
| wall time gb-fr / gb-es (s) | 797 / 966 | 1128 / 1122 |

Top-5 failure modes after baseline-3 (by tails):
1. **Lip-sync coverage**: most cues skipped by the visual gate (two face tracks / PiP / two-shots) → source mouth visible on silence 5–33 s per clip. → EXP-VIDEO-007 (active-track face crop) running.
2. **Dub voice vs bed on film content**: dialogue-to-bed SNR −4.8…+3 dB (gb +24…+40); level matcher saturated. → MIX-004 queued (cap), MIX-005 balance awaits listener pick.
3. **Translation length vs slot**: padding max 9.6 s, fill p10 still low on talking-head content; voiced overrun on half the takes. → timing track (candidate measurement on all takes; adapter word target from measured rates).
4. **Renderer sharpness**: mouth sharpness 0.38 (thom-celia), 0.60 (tos-lab) → lip-generation track; face crop may help small faces.
5. **Speaker similarity tail**: min 0.22, p10 ≈ 0.45 (x-vector cross-lingual) → voice track (ICL vs x-vector per language; reference selection).

## EXP-TIMING-002 — adapter duration model from measured per-language speaking rates

- Measured on baseline-3 raw takes (words per voiced second, p50): fr 3.69 (p25–75 3.1–4.1), es 3.54 (2.8–4.8), en 2.63. The adapter used 2.65 for every language, over-predicting French durations by ~40 % and asking for too few words.
- **Variable**: `ADAPTER_LANGUAGE_RATES` false → true (fr 3.7 / es 3.5 / en 2.65 in `predicted_seconds` and the word target in the prompt). Nothing else changes; TIMING-001 still measures candidates for poorly fitting takes.
- **Metrics**: active-fill p10/p50, padding p90, number of takes sent to candidate measurement (expected to fall → wall time), stretch p90 (must not rise above limits), judge adequacy.
- **EXP-VIDEO-007 first run** (`20260822-174723-core-v0` vs reference-3 `164600`): lipsync_coverage gb-fr **0.24 → 0.64** (6 cues), gb-es 0.41 → 0.71 (8 cues); crop-rendered PiP cues sharpness 1.02–1.08 (vs 0.68–0.73 full-frame). Two regressions, both fixed before the re-run: (a) gb-fr cues 1–2 (large single face) failed inside LatentSync ("Face not detected") on the 1.8× crop → crop now only for multi-face or small-face cues; (b) SyncNet was scored on the first three single-face cues only, which were exactly the failed ones (LSE-C 0.65–1.7 = source mouth vs French) → harness now scores every lip-synced cue on its crop. Re-run as VIDEO-007b. Also: a queue race started two `run_experiment` runs at once (server restart killed MIX-004 mid-run, `.env` mixed) → replaced the waiter scripts with one sequential queue (`queue_seq.sh`).
- **EXP-VIDEO-007b result** (`20260822-210233-core-v0` vs reference-3 `164600`): lipsync_coverage gb-fr **0.24 → 0.86** (2 → 8 cues; cue 3 still "uncertain active speaker" 0.68), gb-es 0.41 → 0.72 (4 → 8). SyncNet now scored on all 16 lip-synced cues: LSE-C mean 4.81 (n=6) → 5.25 (n=16), LSE-D 9.33 → 8.67; the crop-rendered PiP cues score 4.3–7.3 (the right face is animated). Mouth sharpness on cropped cues 1.00–1.06 vs 0.67–0.73 full-frame. `mouth_motion_on_silence_total` flat (10.2 → 10.9 s): the remaining motion is inside rendered windows during dub silence (renderer idles) and in cue 3 — separate item. **Decision: keep** (`LIPSYNC_FACE_CROP=true`).
- **EXP-MIX-004 result** (`20260822-220749-trans-001` vs `123139`): tos-lab take level vs source dialogue (voiced) median **−10.5 → −2.1 dB**; dialogue-to-bed SNR −4.8 → −1.9 dB; bed_ratio_delta −5.9 → −3.4 dB. gb-es: squash −0.65 → −2.43 dB (hotter takes work the limiter harder; allowance is 3 dB — watch). gb-fr bed_ratio_delta unstable (bed at −51 dB) → metric now reports it only when the source bed is above −45 dB. Speaker similarity 0.76 (n=31), adequacy 0.90. **Decision: keep** (`DUB_LEVEL_MATCH_MAX_DB=24`). MIX-005 (voice above bed) still awaits the listener's pick.

## VALID-002 — harness blind spot on face-crop cues

- In `210233` the whole-frame FAN tracker reported no articulation on the crop-rendered PiP cues (speech_on_static_mouth 3.4–6.2 s per cue, on-silence 0) while SyncNet on the crop scored LSE-C 4.3–7.3. The tracker was on the wrong/too-small face. Per-cue articulation (and `coverage_articulation`) is now computed on the renderer's crop for such cues. `aperture_ratio` still uses the whole-frame series — to move next. Values for cropped cues in bundles before this change are not trustworthy.

## EXP-TIMING-003 — measure alternative phrasings for every take

- **Variable**: `DUB_MEASURE_ALL_TAKES` false → true (TIMING-001 currently measures only takes with fill < 75 % or over the stretch limit). **Metrics**: fill p10/p50/p90, padding p90, stretch p90, judge adequacy (candidates adequacy-gated), wall time (cost). Queued on trans-001 after TIMING-002; compare against the TIMING-002 bundle.
- **EXP-TIMING-002 result** (`20260822-233749-trans-001` vs MIX-004 `220749`): gb-fr fill p10/p50/p90 65/77/80 → 73/82/93, padding p90 2873 → 1402 ms, stretch p90 0 → 3.1 %; gb-es ≈ unchanged (70/74/87 → 67/78/85); **tos-lab worse**: fill p10/p50 64/81 → 50/73, stretch p90 19.7 → 27.1 %; wall gb-fr 986 → 1913 s, gb-es 1063 → 1598 s (all 8 FR takes went to candidate measurement). Adequacy 0.89 (same). **Decision: revert** (flag kept, default off). The per-take speaking-rate spread (p25–p75 ≈ 1.5×) means any word-count model mis-sizes a third of lines; measuring candidates (TIMING-001/003) is the robust path.
- **EXP-TIMING-003 result** (`20260823-011355-trans-001` vs MIX-004 `220749`): gb-fr fill p10/p50/p90 65/77/80 → 69/86/91, padding p90 2873 → 1597 ms; gb-es 70/74/87 → 72/77/93; tos-lab 64/81/120 → 54/81/97 with stretch p90 **19.7 → 0 %** (no compressed takes left). Measured/switched: 6/4 → 9/9, 7/4 → 8/4, 5/4 → 7/7. Cost: wall gb-fr 986 → 2039 s, gb-es 1063 → 1589 s, tos 1096 → 1170 s. Adequacy 0.89 → 0.88, speaker similarity 0.76 → 0.75. **Decision: keep** (`DUB_MEASURE_ALL_TAKES=true`) as the quality point; the flag is the compute knob (Pareto: ~+60 % TTS-stage time for p50 fill +9 and zero compressed takes).

## VIDEO-008 — renderer under-articulates (characterization) → guidance scale experiment

- With crop-aware tracking (`20260823-013826-core-v0` re-collect) the rendered mouth's aperture std is ≈0.013–0.030 during dub silence and ≈0.013–0.041 during dub speech — barely different — while the source actor's speech std is 0.025–0.039 (and even his pauses 0.03–0.045). The "mouth moving in silence" inside rendered windows is the renderer's uniformly weak motion, not idle jitter; under speech it articulates ~0.6× the source. New per-cue metric `articulation_strength` (voiced aperture std, output/source).
- **Variable**: `LATENTSYNC_GUIDANCE_SCALE` 1.5 → 2.5. **Metrics**: articulation_strength (expect ↑ toward 1), LSE-C/LSE-D, mouth sharpness, identity (ArcFace in the A/B tool if needed), boundary excess. Queued on core-v0 after baseline-4.

## ASR-001 — proper noun mis-heard at the source ("Casey Muratori" → "Murata")

- Heard as "Muriati" in every gb-fr dub. Trace: Qwen3-ASR transcribed "Murata"; translation kept it faithfully; TTS said it. Not a translation regression. The job glossary (term → pronunciation on the spoken text) is the designed user input for names; suites now pass a per-clip `glossary` (gb clips: Murata → Muratori). General fix for the ASR track: proper-noun recovery (second-ASR vote or on-screen text) — open.

## MIX-006 — replicate the source's voice-over-bed balance (no preference to pick)

- Listener feedback: every MIX-005 variant beat "current", but a preference is the wrong question; the invariant is the source's own dialogue-to-bed balance. `match_source_balance`: per cue, K-weighted (voice − bed) in the output vs (dialogue − bed) in the source over the same span; bounded ±8 dB correction on the voice stem with 30 ms smoothing; a user offset can sit on top later.
- Offline on the MIX-004 tos-lab job: corrections +1.75…+3.67 dB (median +2.3) on 13 cues; harness `bed_ratio_delta_db` −3.39 → −1.82, d2b SNR −1.9 → −0.3, LUFS unchanged, no clipping. Residual reflects the harness metric being RMS-in-mix (with ducking) vs the K-weighted stem ratio — to reconcile so the target is exactly 0. Queued on trans-001 after VIDEO-008 (`DUB_MATCH_SOURCE_BALANCE=true`). MKVs: ~/Downloads/dubline/mix-006-ab/.

## ASR-001 characterization — proper nouns across core-v1 (baseline-3 transcripts vs Whisper turbo, greedy and beam-5)

- 216 capitalized tokens; 66 not confirmed by Whisper (many are sentence-initial false positives of the crude filter). Real disagreements: the same name spelled several ways by the SAME recognizer within a film — Lampert/Lampard/Lombard/Lampeth, Voss/Vasse's, Scobie/Scoby, Celia/"Luxilio"; gingerBill: "Moritore" (cue 1) vs "Muratori" (cue 3) vs "Murata" (other runs). Whisper: Muratari/Muratore, GingerBell/Gingerbill.
- Greedy vs beam-5 (Whisper): no systematic gain on names; beam better on casing/punctuation. OOV names need candidates/consensus, not decoding changes.
- → ASR-002 plan: film-level name consistency pass (cluster capitalized tokens from both recognizers by phonetic key + edit distance; consensus spelling; rewrite before translation). Metrics: spellings per cluster (Lampert 4 → 1), cross-recognizer name agreement; glossary/on-screen text as optional extra votes.

## ENTITY baseline (core-v1 baseline-3 transcripts, v0 detector: capitalized non-sentence-initial tokens, phonetic clusters)

| clip | consistency | inconsistent clusters | translation preservation | TTS pronunciation (lost) |
|---|---|---|---|---|
| gb-talking-head-fr | 0.80 | Moritore/Muratori | 0.94 | 1.0 |
| gb-es-roundtrip-en | 0.75 | (false positive) | 0.81 (Martin lost in cue 3) | 0.94 (Inger ← GingerBill) |
| tos-thom-celia-fr | – | – | 1.0 | 0.0 (Celia → "Luxilio") |
| charade-hotel-banter-fr | 0.80 | Lambert/Lampert | 0.67 (Charles lost, cue 14) | 0.75 (Joshua, Pete) |
| charade-office-es | 0.00 | Charles/Charles's · Voss/Vasse's · Lampard/Lampert/Lombard | 0.85 | 0.80 (Gideon, Vasse, Voss) |
| charade-embassy-fr | 1.0 | – | 0.54 (United States…) | 1.0 |

Known v0 weaknesses: no punctuation in ASR output → some sentence-initial false positives ("Unfortunately", "Use"); possessives. Next: proper NER (spaCy multilingual) and the title-lexicon builder (EXP-ENTITY-001) measured by consistency → 1.0 and TTS-pronunciation losses → 0.

## INCIDENT 2026-08-23 — server checkout silently stale (three runs invalid)

- Offline tests copied files into the server's working tree; `git pull -q` then refused to merge (untracked `entities.py`) and the queue scripts ignored the exit code. Server stayed at `e76a590` from 01:38 UTC.
- Invalid: **VIDEO-008** (`070236-core-v0`, guidance setting absent → identical to baseline: LSE-C 5.25 → 5.17) and **MIX-006** (`083838-trans-001`, setting absent → no corrections). **Baseline-4** (`045814-core-v1`) ran without TIMING-003's default; its stretch p90 regression (5.8 → 19.8 %) is the VIDEO-007 × slack-policy interaction (more `mouth_visible` cues → 12 % slack and 5 % limit → shorter targets → retries still over → compression), plus sub-second cues; its 4 delivery-QC failures are peak-safe loudness (−18.4…−19.4 LUFS) vs a check that did not read the mastering record — fixed.
- Fixes: working tree cleaned and pulled (`6f1e28d`); queue scripts now abort loudly on pull failure and print the commit they run; test copies go to /tmp only. Re-queued VIDEO-008 and MIX-006, then EXP-TIMING-004 (`DUB_LIPSYNC_SLACK`) on core-v1.

## EXP-ENTITY-001 — title lexicon v0 (program-wide name consensus)

- **Variable**: `ENTITY_LEXICON` false → true. After utterance merge: discover capitalized non-initial tokens across all cues, cluster by phonetic key, canonical = glossary/metadata match else most frequent spelling; rewrite aliases in `source` (original kept in `source_asr`); write `entity-lexicon.json`. Unit: Moritore→Muratori, Lampard→Lampert.
- **Metrics**: `entity_consistency` (→ 1.0), `translation_entity_preservation`, `tts_entity_pronunciation`, judge adequacy (must not drop). Suite core-v1 (charade/tos names). Known v0 gaps: Lombard not merged into Lampert (key distance), no OCR/N-best evidence yet.
- **EXP-VIDEO-008b result** (`20260823-150712-core-v0` at 6f1e28d vs crop-aware re-collect `013826`): LSE-C mean **5.25 → 5.96** (median 5.39 → 6.25), LSE-D 8.67 → 8.55, aperture_ratio 0.94 → 1.04, articulation_strength median 1.06 (n=16; max 9.6 on one cue — inspect), mouth sharpness unchanged (1.02/1.00), mouth_motion_on_silence 1.47 → 1.65 s (stronger motion overall). **Decision: keep** (`LATENTSYNC_GUIDANCE_SCALE=2.5`).
- **EXP-MIX-006b result** (`20260823-164537-trans-001` at 7a047a2 vs TIMING-003 `011355`): corrections applied on 9/10/14 cues, median +2.9/+2.9/+2.7 dB (range +1.7…+6.7); tos-lab `bed_ratio_delta` −3.33 → −1.42, d2b SNR −1.77 → −0.13; gb d2b 19.1 → 22.1 / 46 → 66; gb-es squash 0 → −1.3 dB (hotter voice; within allowance); adequacy 0.89, similarity 0.75 unchanged. **Decision: keep** (`DUB_MATCH_SOURCE_BALANCE=true`). Residual −1.4 dB: harness ratio is RMS-in-mix with ducking vs the K-weighted stem ratio the corrector targets — reconcile so the invariant reads 0.
- **EXP-TIMING-004 provisional** (`20260823-201946-core-v1` vs baseline-4 `045814`; confounds: TIMING-003 default + guidance 2.5 also differ): stretch p90 **19.8 → 9.05 %** (mean 7.6 → 7.1; max worse 186 → 314 on a 1-word sub-second cue), delivery QC 6/10 → 10/10 (QC loudness fix contributes), LSE-C 4.86 → 5.67, padding mean 401 → 471 ms (slack cost). Remaining >8 % stretch: 13 cues, all `adapted_retry` — sub-second lines and translations no candidate could shorten. Final keep/revert vs reference-5 (single-variable). To inspect: tos-rooftop cue 7 adequacy 0.0.

## TRANS-002 — junk ASR fragments must not be translated (unconditional, corpus-verified)

- tos-rooftop cue 7: source "S" (ASR junk) → translator invented "C'est fini." (judge adequacy 0.0). `is_nonverbal_filler` now also returns True for fragments with <2 alphabetic chars or a single ≤3-letter vowel-less token — such cues keep the original performance. Corpus scan across all 10 core-v1 clips: catches exactly the two "S" fragments (0.64 s each), zero real-speech collateral; unit cases: S/St/Hmm → filler, Go/No/Ok → translate.
- **EXP-ENTITY-001 result** (`20260824-003010-core-v1` vs TIMING-004 `201946`, single variable `ENTITY_LEXICON`): inconsistent name clusters across the suite **8 → 3** (and 2 of the remaining 3 were the metric counting possessives — fixed; genuine remainder: Lampert/Lampard on charade-office). Per-clip consistency: gb-talking-head 0.8 → 1.0 (Moritore/Muratori unified), gb-screenshare 0.0 → 1.0, charade-embassy 0.5 → 1.0, charade-office 0.0 → 0.33 (0.67 after the metric fix). Translation entity preservation: gb-screenshare 0.39 → 0.58, gb-es 0.81 → 0.91, gb-talking-head 0.94 → 0.94. Judge adequacy 0.885 → 0.879 (noise), word similarity 0.966. Lexicon artifact written per job (`entity-lexicon.json`). **Decision: keep** (`ENTITY_LEXICON=true`). Next on the track: ASR-002b (lexicon → trie/hotword biasing → re-decode name spans) for names no spelling in the transcript got right.
- First `naturalness_mos` (UTMOS) distribution on core-v1: mean 1.88, median 1.56, p90 2.95, max 4.24 (n=112) — low absolute values are expected for cross-lingual cloned TTS on 24 kHz takes; useful as a *relative* signal, to be validated against calibration-002 human scores before it informs any decision.

## NOISE-FLOOR (repeated seeds) — planned

Three identical trans-001 runs (same `DUB_SEED=1247`, same defaults, lip-sync off for speed). Deliverable: per-metric spread (max−min and stdev) across the three, published in `eval/pareto.md`. Rule adopted: **an experiment delta smaller than the noise floor of its metric cannot justify a keep or a revert** — such experiments must be re-run with more clips or a paired/seeded design. Metrics of interest: judge adequacy, word similarity, active-fill p10/p50, padding p90, stretch p90, naturalness MOS, entity metrics, dialogue-to-bed SNR, balance-vs-source.

## NOISE-FLOOR results (3 identical trans-001 runs, seed 1247) — and the audit they force

| metric | spread across identical runs |
|---|---|
| judge adequacy (mean) | 0.130 |
| stretch % (mean) | 3.78 |
| worst-clip stretch p90 | **23.4** |
| active fill p10 / p50 | 7.9 / 3.2 |
| dialogue-to-bed SNR | 2.95 dB |
| padding ms (mean) | 90 |
| speaker similarity | 0.016 |
| naturalness MOS | 0.073 |
| bed ratio delta | 0.21 dB |
| balance vs source | 0.010 dB |
| entity consistency | 0.000 |
| wall time | 3 min |

**Audit of prior decisions against these floors (honest corrections):**
- **EXP-TIMING-004 — REVERTED (was "provisional keep").** Against the clean comparator reference-5 (`20260824-043951`, single variable): stretch mean 7.14 vs 5.80, p90 9.05 vs 8.73, padding 471 vs 434 ms, adequacy 0.885 vs 0.893, QC 10/10 both. Every delta is inside the noise floor. The stretch improvement I credited to slack in the provisional comparison came from the QC-loudness fix and TIMING-003 defaults present in that comparison, not from the variable. `DUB_LIPSYNC_SLACK=false`.
- **EXP-TIMING-003 — keep STANDS, but on the fill metric only.** Its headline "tos-lab stretch p90 19.7 → 0" is *inside* the ±23.4 floor for worst-clip stretch p90 and must not be cited. The fill p50 gain (77 → 86, floor ±3.2) and the padding reduction survive.
- **EXP-MIX-006 — keep STANDS on `bed_ratio_delta_db` (−3.33 → −1.42, floor ±0.21) and `balance_vs_source` (floor ±0.01), NOT on `dialogue_to_bed_snr_db`** (−1.8 → −0.1 is inside the ±2.95 floor).
- **EXP-ENTITY-001, TRANS-001, MIX-004, SYNC-001, VIDEO-007 — unaffected**: entity consistency has a zero floor; TRANS-001's adequacy gain (0.62 → 0.94) is 2.5× the floor; the others moved metrics by 5–100× their floors.
- **Open gap**: LSE-C / coverage / sharpness have no floor yet (noise runs had lip-sync off) → VISUAL-NOISE-FLOOR queued (2 identical core-v0 runs). Until it lands, VIDEO-008's LSE-C 5.25 → 5.96 is *unvalidated*.

## METRIC VALIDATION without human scoring (adopted 2026-08-24)

Human A/B scoring is not required to know whether a metric works: inject a failure we already
understand, and check the metric's response against its noise floor. Results on calibration-002's
anchors (mean delta, ratio to floor):

| injected failure | detected by | effect |
|---|---|---|
| audio offset 200 ms | `sync_offset_frames` | 5.0× floor |
| speed-up 30 % | `dub_speech_s`, `dead_air_s`, `speech_fraction`, `speech_on_static_mouth_s` | 6–8× |
| mouth blur | `mouth_sharpness_ratio` (18×), `aperture_ratio` (3.9×), `sync_lse_c` (1.2×) | strong/weak |
| silence gap 1.5 s | `dub_speech_s`, `speech_fraction`, `mouth_motion_on_silence_s`, `coverage_articulation` | 3.4–4.2× |

**`naturalness_mos` (UTMOS) FAILS validation** — response to a 30 % speed-up is −0.046 against its own
±0.073 floor; blur/gap/offset all ≈ +0.007; whole-corpus range compressed to 1.24–2.37. It is retained
as a logged column but **must not inform any decision**; candidates to replace it: NISQA-TTS, or a
paired MOS predictor validated on this same anchor battery first.

`eval/anchors.py build` renders a battery covering the failure modes we have actually shipped bugs for
(picture offset 4 s, wrong-face render, bed removal, quiet voice, plus the four above), so every new
metric can be validated the same way before it is allowed to decide anything.
