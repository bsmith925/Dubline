# Dubline

Dubline is a fully local, unattended film-dubbing pipeline for Windows and CUDA. The normal interaction is deliberately small: drop a complete film (and an optional sidecar subtitle), press **Dub the complete film**, and receive an English-dubbed MKV plus a QC report. Source language, subtitle/ASR choice, separation, speaker analysis, adaptation, performance transfer, retry policy and cinema mastering are automatic. No cloud inference or hosted service is used.

The implementation is tuned for this machine: an RTX 4060 Laptop GPU with 8 GB VRAM, i7-13650HX, 32 GB RAM, and constrained local storage. Every major GPU model runs in its own process, one at a time. Exiting the process is the VRAM release boundary and also protects the UI service from native PyTorch/model faults.

## What the automatic pipeline does

1. **Preflight and ingest.** Probe every stream, chapter, timebase, language tag, subtitle, and exact source duration. The UI can process the complete film or a precise start/end section such as `20:20`–`22:00`; sidecars are trimmed and shifted automatically. Character registration still uses the original complete film. Estimate working-space requirements and report low free VRAM. Prefer a supplied M&E/DME/clean-effects track when its metadata identifies one.
2. **Cinematic sound separation.** Run the official multilingual Bandit v2 model to create dialogue and music/effects stems. Run MelBand-RoFormer and HTDemucs independently as recovery candidates. A cue-level speech/contamination gate keeps Bandit's clean result, prefers RoFormer for masked voices, or uses HTDemucs as the legacy fallback. Recovery is never used as the final M&E bed.
3. **Dialogue map.** Feed bounded subtitle/voice-activity scene windows to Qwen3-ASR 0.6B and Qwen3 Forced Aligner 0.6B on CUDA. Confidence is derived from forced-alignment validity rather than a fixed constant. Weak windows selectively run Qwen3-ASR 1.7B in a fresh process and combine its alignment evidence with 0.6B/1.7B transcript agreement. Whisper large-v3-turbo remains the unsupported-language boundary fallback and independent TTS back-transcription model. Subtitle cards remain translation evidence and are reconciled to forced word/speech timing.
4. **Audiovisual speakers.** Before scene processing, sample the original complete film and cluster recurring faces into a persistent SFace registry. Use pyannote Community-1 exclusive/overlap-aware diarization across the full film—even when only a section is selected—plus CAMPPlus voiceprints and a CPU YuNet/SFace pass which measures mouth motion. A clearly speaking registered face becomes a confidence-gated face/voice anchor; later distance, profile, obscured, and off-screen lines are matched by the global audio cluster and voiceprints. No face is treated as missing evidence rather than negative evidence, and unknown voices remain separate. Only clean assignments at or above 0.62 confidence may enter a recurring character bank. The C3ASD adapter boundary remains available, but its official repository currently publishes code without a pretrained checkpoint, so Dubline does not pretend an unavailable model is active.
5. **Translation, then audiovisual adaptation.** Run Hy-MT2 7B Q4_K_M through llama.cpp with CUDA GPU layer offloading. The first pass translates complete scenes faithfully with stable names, facts, terminology, register, jokes, and relationships. Difficult cues receive up to six variants. A separate local semantic judge ranks adequacy, terminology and register before duration, source rhythm and phonetic/viseme compatibility; visible mouths receive stricter duration and viseme weighting. Character-string similarity is only an emergency fallback, never the primary meaning score.
6. **Performance transfer.** Measure source energy, pauses, pitch, speaking rate, and stereo position. Clean cues use the individual source cue as IndexTTS's emotion/performance prompt and the character bank for timbre. Qwen's eight-way scene emotion is an isolated fallback for cues whose source performance is too weak.
7. **IndexTTS 2.5.** Run IndexTTS in its own CUDA process with BF16, low-VRAM behavior, native duration control, multilingual tokenizer support, stable kernels, and deterministic first takes.
8. **Generate–measure–reject–retry.** Detect and trim only outer silence, measure active speech, and preserve internal pauses. Short natural delivery is padded without slowing. Overlong takes get a calibrated second IndexTTS candidate, then Hy-MT2 re-adaptation if necessary. Rubber Band touches voiced phrases only. The normal correction limit is 8%; it tightens to 5% only when a confidently active mouth is visible. A line which repeatedly fails word, speaker, duration, or prosody checks gets a Qwen3-TTS 1.7B Base voice-cloned alternative, and the two engines are ranked rather than blindly replaced.
9. **Independent translation and multimodal QC.** Hy-MT2 never grades its own translation. A separately loaded Qwen3 8B bilingual judge compares source-language text directly with the final English line and fails changed facts, polarity, names, relationships, material omissions/additions, and register changes. Whisper Turbo then back-transcribes every selected take. QC records WER, CER, active-speech timing, CAMPPlus source/generated speaker similarity, source-to-bed dialogue leakage, clipping, source level, alignment, adaptation confidence, pause-pattern delta, pitch delta, and energy-contour similarity. Missing evidence fails closed instead of receiving a favourable default.
10. **Non-verbal and overlap safety.** The separated performance stem is retained on the English timeline and only forced-aligned spoken-word spans are removed. Laughter, crying, breaths, screams, grunts, singing, and crowd walla outside those spans therefore survive even when they are not synthesized. Hyphenated two-person subtitle cards become simultaneous independent cues and voices. When a mono overlap cannot be reliably disentangled, the source overlap is retained, the dominant English performance is added, and the cue is explicitly marked `Needs review`; the system does not claim a false clean separation.
11. **Acoustic finish, mix, and delivery.** Derive a six-band match EQ and early-reflection decay from each source cue instead of assigning a coarse room preset. Restore source spatial position and level, preserve a genuine 5.1 or 7.1 M&E bed when supplied, and otherwise export explicit stereo. Select cinema dialogue-gated −27 LUFS/−2 dBTP, EBU −23 LUFS/−1 dBTP, web −16 LUFS, or unmastered delivery. Add an English FLAC delivery track without changing the original tracks' default dispositions. FLAC describes the output codec; it is not a claim that model-generated speech is artifact-free.
12. **Optional visual finishing.** MuseTalk 1.5 is installed in a separate Python 3.10/CUDA environment and is disabled by default. It only accepts clear, single-face shots with strong active-speaker evidence and a remaining visible sync failure. It refuses HDR/Dolby Vision, 10/12-bit, and multi-video-stream sources so an optional pass cannot silently downgrade the picture. Group shots, distance shots, profiles without reliable landmarks, and already acceptable cues retain audio-side synchronization.
13. **Delivered-media QC.** Verify picture duration, the identity/codec/language/title/disposition of every original stream, chapters, container metadata, English FLAC presence, mastering result, loudness, true peak, missing cues, and every line result. A job is `Complete` only if every automatic cue and delivery check passes. A playable output with any unresolved evidence is `Needs review`, with no success wording that hides the exception.

## Automation policies

The drag-and-drop UI deliberately runs the fully automatic policy with English output and cinema delivery. The underlying job API retains the review and approval policies for controlled production workflows without exposing model plumbing on the start screen.

- **Fully automatic** produces the film without pausing, internally retries failed cues, and assigns `Needs review` if anything remains unresolved.
- **Review flagged lines** completes the film and shows only unresolved cues. A flagged line can be edited and regenerated without restarting the project.
- **Approval workflow** performs separation, ASR, alignment, speaker analysis, adaptation, and emotion analysis, then pauses before the long synthesis stage.

Pause, cancellation, process crashes, and restarts retain committed cue sheets, references, takes, and stems in SQLite-backed job workspaces. Queued pauses and cancellations resolve immediately; live model children use a continuously drained stream and are terminated at the next 250 ms control checkpoint. Edited cues invalidate and archive generated, fitted, acoustically matched, and alternative-engine files, so a stale performance cannot re-enter the mix. Cue sheets are stored separately from compact job state, checkpointed every 25 synthesized lines, and fetched by the browser only when their revision changes.

## Inputs, selection, and review

Drag-and-drop accepts common consumer and professional video containers (`mkv`, `mp4`, `mov`, `avi`, `webm`, `m4v`, `ts`, `mts`, `m2ts`, `mpeg`, `mpg`, `wmv`, `mxf`, `vob`, `3gp`) and audio programmes (`wav`, `flac`, `mp3`, `m4a`, `aac`, `ogg`, `opus`, `wma`, `aiff`). Multiple media files create separate queued jobs. Local paths can point to any FFmpeg-readable local file without copying it into the workspace. Network URLs, streaming/DRM sources, disc decryption, and ISO authoring are intentionally outside this local-file application.

All audio streams are probed before processing. Automatic selection rejects commentary, audio description, director, isolated-score, music-only, and karaoke labels; ambiguous uploaded media pauses for an explicit programme-track choice. Subtitle selection avoids forced/signs-only/commentary tracks, tries all supplied sidecars in ranked order, preserves `.sub`/`.idx` VobSub pairs, and falls back to ASR when bitmap text cannot be read locally. English forced cards no longer suppress speech found by ASR, and non-English subtitle wording is retained as translation evidence.

The optional approval workflow exposes the full pre-synthesis cue sheet. Source transcript, English dialogue, timing, and project-wide character name can be edited; cues can be split or merged; a pronunciation glossary can be applied; prior takes can be restored. The visible timeline is paged in groups of 500 rather than truncated. Voice cloning requires an explicit rights confirmation. References never expand into neighbouring actors: short cues are silence-padded and weak identity evidence is flagged rather than presented as an exact clone.

## 20:20 speaker validation

The Waterboys scene beginning at 20:20 exposed an important validation artifact. When the scene was cropped to 45 seconds before diarization, Community-1 returned only two speaker clusters. The same exact scene, analyzed with three minutes of surrounding film, returned four distinct speakers. The production path already analyzes the complete input film, so it has this surrounding evidence; short diagnostic clips do not.

The implementation now also prevents low-confidence cropped-scene labels from contaminating voice banks. The audiovisual validation at 20:20 correctly refuses to create an identity anchor: two to six faces are visible in the sampled group coverage, and mouth-pixel motion alone is not safe active-speaker evidence. The four voices are recovered when full-film audio context is available; the cropped diagnostic remains a deliberately audio-tentative result. The automated suite includes a four-voice full-context mapping test and a separate test proving that an uncertain cue cannot enter another character's reference montage.

Some historical 20:20 validation outputs were corrected with a precomputed context map while the failure was being diagnosed. Those stored outputs are not the production mechanism and are not used as fixtures or overrides. New jobs derive the mapping automatically from whole-film diarization, stable face IDs, and confidence-gated voice evidence; there are no scene names, timestamps, or expected speaker counts in the speaker-assignment code.

## Installed local models

- IndexTTS 2.5 checkpoints and auxiliary Wav2Vec2-BERT, semantic codec, CAMPPlus, BigVGAN, and Qwen emotion model
- Bandit v2 multilingual cinematic separator (`checkpoint-multi.ckpt`, MD5 verified)
- MelBand-RoFormer Kim vocals checkpoint (modern recovery) and HTDemucs (legacy fallback)
- Qwen3-ASR 0.6B, selective Qwen3-ASR 1.7B escalation, and Qwen3 Forced Aligner 0.6B
- Whisper large-v3-turbo (fallback boundaries and back-transcription QC)
- Hy-MT2 7B Q4_K_M GGUF (CUDA GPU-accelerated scene translation and dubbing adaptation)
- pyannote Community-1 in an isolated CUDA environment (exclusive and overlapping speaker turns)
- OpenCV YuNet and SFace for CPU face/active-mouth anchors
- Qwen3-TTS 1.7B Base in an isolated CUDA environment (failed-take alternative)
- MuseTalk 1.5, DWPose, SyncNet, S3FD, face parsing, Whisper Tiny and SD VAE in an isolated optional finishing environment
- Whisper Small remains cached but is no longer the default

pyannote is kept outside the IndexTTS environment because its current dependency stack requires a newer `protobuf`; this prevents it from changing IndexTTS's validated runtime. CAMPPlus remains the explicit fallback if that isolated model cannot run.

## Setup and run

Requirements are Python 3.11, Git, FFmpeg/FFprobe with Rubber Band support, and a recent NVIDIA driver.

```powershell
.\setup.ps1
.\run.ps1
```

Open <http://127.0.0.1:8000>. For feature films, use the local-path option to avoid copying the source. Set `DUB_WORKDIR` to a spacious SSD when the preflight estimate exceeds the system drive's free space.

Setup installs the locked CUDA environments, downloads every ungated model, verifies the Bandit and RoFormer checkpoints, and optionally downloads pyannote when `HF_TOKEN` is present. All inference remains local after setup. Review the bundled IndexTTS custom model licence before commercial distribution.

## Outputs

Each completed job contains:

- `dubbed-english.mkv` — copied picture and original streams plus a non-default English FLAC delivery track
- `english-dialogue.flac` — float-mixed, source-positioned dialogue stem
- `english-mix.flac` — English dialogue plus M&E
- `cues.json` — complete dialogue/adaptation/performance/QC state
- `qc-report.json` and `qc-report.html`
- English SRT, cue CSV, 24 fps CMX-style EDL, per-character full-length FLAC stems, and a ZIP of per-line WAV takes
- per-character reference banks, archived take history, and checkpointed per-line takes

## Known boundary

This laptop is designed for high-quality film dubbing, not full feature-length generative face replacement or Atmos object authoring. Original Atmos and other multichannel tracks are preserved; a supplied M&E bed can produce a new stereo, 5.1, or 7.1 English bed, but it does not reconstruct Atmos objects. Mono simultaneous speech cannot always be separated into clean identities; unresolved overlaps retain source performance and are reported. Voice matching is confidence-scored synthesis, not a guarantee of biometric identity. MuseTalk can fit when loaded alone but is deliberately selective because it can alter the actor's visual performance and is slow on a laptop GPU.

Primary model documentation: [IndexTTS](https://github.com/index-tts/index-tts), [Bandit v2](https://github.com/kwatcharasupat/bandit-v2), [Demucs](https://github.com/facebookresearch/demucs), [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), [C3ASD](https://github.com/jisoo-o/C3ASD), [MuseTalk](https://github.com/TMElyralab/MuseTalk), [Whisper](https://github.com/openai/whisper), and [pyannote.audio](https://github.com/pyannote/pyannote-audio).
