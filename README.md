# Dubline

<div align="center">

**Fully Local, Unattended Cinematic Film-Dubbing Pipeline with Automatic Stem Separation, ASR, LLM Adaptation, Voice Cloning, and Cinema Mastering.**

*No cloud inference · No hosted APIs · 100% On-Device*

</div>

---

## Overview

**Dubline** is an end-to-end cinematic audio dubbing system designed to translate and dub full-length feature films and video programmes into English entirely offline on local CUDA hardware. 

Drop in any video file (with optional sidecar subtitles), click **Dub**, and receive a broadcast-ready MKV with master-quality English dialogue and music/effects (M&E) stems, complete with automated quality-control (QC) reports.

```
┌─────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ Input Video │ ──► │ Cinematic Separation  │ ──► │ Multi-pass ASR        │
│ & Subtitles │     │ (Bandit v2 / RoFormer)│     │ & Forced Alignment    │
└─────────────┘     └───────────────────────┘     └───────────┬───────────┘
                                                              │
┌─────────────┐     ┌───────────────────────┐     ┌───────────▼───────────┐
│ Delivery    │ ◄── │ Studio Match EQ &     │ ◄── │ IndexTTS 2.5 /        │
│ (MKV & QC)  │     │ Cinema Mastering      │     │ Hy-MT2 Adaptation     │
└─────────────┘     └───────────────────────┘     └───────────────────────┘
```

---

## Key Features

- **Strict Process & VRAM Isolation:** Designed to run reliably on consumer GPUs (8 GB+ VRAM). Every heavy AI model (Bandit, RoFormer, pyannote, Qwen ASR, LLM, IndexTTS) executes in its own isolated subprocess where process exit guarantees clean CUDA memory release.
- **Cinematic Stem Separation:** Uses Bandit v2 for multilingual music, effects, and speech separation, with MelBand-RoFormer and HTDemucs vocal recovery gating.
- **Audiovisual Speaker Tracking:** Fuses whole-film `pyannote.audio` diarization and CAMPPlus voiceprints with OpenCV YuNet & SFace visual active-mouth tracking for accurate character identity assignment.
- **Syllable- & Timing-Aware Dialogue Adaptation:** Hy-MT2 7B LLM generates multiple dubbing candidates scored by semantic adequacy, syllable count, natural pauses, and viseme compatibility.
- **Independent Bilingual QC Judge:** Translations are audited by an independent Qwen 8B bilingual judge to catch hallucinated facts, omissions, or tone shifts before synthesis.
- **Prosody & Emotion-Matched TTS:** Synthesizes speech with IndexTTS 2.5 using source cue audio as performance/emotion prompts, with automatic Qwen3-TTS fallback.
- **Non-Verbal Sound Preservation:** Retains natural human expressions (laughter, gasps, crying, grunts, sighs) by selectively removing only forced-aligned spoken word spans from the original performance track.
- **Cinema Mastering & Spatial Matching:** Applies 6-band dynamic Match EQ, early-reflection convolution matching, dialogue-gated loudness normalization (Cinema −27 LUFS, EBU R128 −23 LUFS, or Web −16 LUFS), and true-peak limiting.

---

## Pipeline Architecture

1. **Preflight & Ingest:** Probes containers, streams, chapters, timebases, and audio layouts. Automatically trims and synchronizes sidecar subtitles or extracts internal tracks.
2. **Dialogue & M&E Separation:** Runs Bandit v2 to extract clean speech and cinema-grade Music & Effects (M&E) stems. Employs cue-level contamination gates to recover masked lines via RoFormer.
3. **Dialogue Mapping & Forced Alignment:** Uses Qwen3-ASR and Qwen3 Forced Aligner on CUDA for word-level timestamps, falling back to Whisper Large-v3-Turbo for boundary validation.
4. **Audiovisual Diarization:** Tracks speaking faces across scenes, building a persistent character voice-bank gated by strict confidence thresholds (≥ 0.62).
5. **Contextual Translation & Adaptation:** Translates scene-by-scene with Hy-MT2, evaluating up to six duration-constrained variants against source rhythm and mouth visibility.
6. **Performance Cloning & Generation:** Extracts energy contours, pitch, and cadence to drive IndexTTS 2.5 synthesis with BF16 precision.
7. **Generate–Measure–Reject Loop:** Trims outer silence, verifies active duration, and retries overlong lines. Applies non-linear Rubber Band time-stretching strictly to voiced phrases (< 8% skew, < 5% for visible mouth close-ups).
8. **Automated Quality Control (QC):** Back-transcribes every synthesized take with Whisper Turbo to calculate WER/CER, speaker cosine similarity, and M&E bed leakage.
9. **Acoustic Finish & Delivery:** Renders multichannel/stereo English FLAC delivery tracks, muxes with the original video container, and outputs comprehensive HTML/JSON QC reports.

---

## System Requirements

- **Operating System:** Windows 10/11 (64-bit) or Linux
- **GPU:** NVIDIA GPU with CUDA support (8 GB VRAM minimum; 12 GB+ recommended)
- **CPU:** 6+ cores / 12+ threads recommended
- **RAM:** 16 GB minimum (32 GB recommended)
- **Storage:** 40 GB+ free disk space for models and working caches (SSD strongly recommended)
- **External Dependencies:** Python 3.11, Git, FFmpeg (with Rubber Band audio filter support), and recent NVIDIA GPU drivers.

---

## Quick Start

### 1. Installation

Clone the repository and run the automated setup script to configure virtual environments and download required models:

```powershell
# Clone the repository
git clone https://github.com/leighrobertabbott/Dubline.git
cd Dubline

# Run setup (installs dependencies and model weights)
.\setup.ps1
```

### 2. Launch the Application

Start the local server and web interface:

```powershell
.\run.ps1
```

Open your browser at **`http://127.0.0.1:8000`**.

---

## Configuration

Dubline can be configured via environment variables or by creating a `.env` file (see [`.env.example`](.env.example)):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DUB_ENGINE` | `indextts` | Primary TTS engine (`indextts` or `qwen-tts`) |
| `DUB_WORKDIR` | `./data` | Working directory for job stems and temporary files |
| `DUB_DIARIZATION_DEVICE` | `cuda` | Hardware device for diarization (`cuda` or `cpu`) |
| `DUB_LLAMA_GPU_LAYERS` | `-1` | Number of GPU layers for LLM adaptation (`-1` = full offload) |
| `BANDIT_CHECKPOINT` | `./vendor/...` | Path to Bandit v2 separation checkpoint |
| `MUSETALK_ENABLED` | `0` | Enable optional generative visual lip-sync pass (`0` or `1`) |
| `HF_TOKEN` | *None* | Optional Hugging Face token for gated models (e.g. pyannote) |

---

## Outputs & Deliverables

Each completed dubbing job produces a self-contained project workspace containing:

- **`dubbed-english.mkv`** — Lossless video with original streams preserved and new English FLAC delivery track.
- **`english-dialogue.flac`** — Clean, source-positioned English dialogue stem.
- **`english-mix.flac`** — Fully balanced mix (Dialogue + M&E).
- **`cues.json`** — Detailed timeline of every cue, transcript, timing, speaker ID, and QC metrics.
- **`qc-report.html` & `qc-report.json`** — Visual and machine-readable QC audit (WER, speaker similarity, loudness, clip warnings).
- **`subtitles.srt` & `cues.csv`** — Time-aligned English subtitle cards and editorial cue sheets.
- **Per-Character Stems & WAV takes** — Individual character dialogue tracks and ZIP archive of every synthesized line.

---

## Model Acknowledgments & References

Dubline brings together world-class open-source research and foundational models:

- **TTS & Voice Cloning:** [IndexTTS](https://github.com/index-tts/index-tts) · [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- **Audio Separation:** [Bandit v2](https://github.com/kwatcharasupat/bandit-v2) · [MelBand-RoFormer](https://github.com/lucidrains/mel-band-roformer) · [Demucs](https://github.com/facebookresearch/demucs)
- **ASR & Alignment:** [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) · [OpenAI Whisper](https://github.com/openai/whisper)
- **Translation & Adaptation:** [Hy-MT2](https://github.com/Tencent-Hunyuan/HunyuanTranslation-2) · [Qwen2.5 / Qwen3](https://github.com/QwenLM/Qwen2.5) · [llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Diarization & Voiceprints:** [pyannote.audio](https://github.com/pyannote/pyannote-audio) · [CAMPPlus (ModelScope)](https://github.com/modelscope/3D-Speaker)
- **Visual Sync & Face Models:** [OpenCV YuNet & SFace](https://github.com/opencv/opencv_zoo) · [MuseTalk](https://github.com/TMElyralab/MuseTalk)

---

## License

This project is licensed under the [MIT License](LICENSE) (or applicable component licenses for bundled third-party models). Please review the respective model licenses for commercial usage terms.
