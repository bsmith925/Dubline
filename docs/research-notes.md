# Research notes — literature actionable for Dubline

## Timing / isochrony

**Isochrony-Aware NMT for Automatic Dubbing** (Tam, Lakew, Virkar, Mathur, Federico — AWS, arXiv:2112.08548)
- Task: translation must transfer the source's *phrase-pause structure* (pause = ≥300 ms silence; phrase = text between pauses), not just meaning. Segment-level, not sentence-level.
- Best approach is the simplest: inject literal `[pause]` markers into the SOURCE text and train/ask the model to emit `[pause]` markers in the TARGET. Implicit learning beat explicit phrase-level verbosity control (which cost translation quality) and beat the two-step MT→prosodic-alignment cascade on their combined score.
- Training data trick: synthetic pause insertion by sampling phrase lengths from a real distribution (they lacked labeled pauses).
- **Metrics we can adopt:**
  - SA (segmentation accuracy): % of utterances where the target has the same number of pauses as the source.
  - PhraseLC: % of utterances where EVERY target phrase is within ±10 % of the character length of its source phrase.
  - Acceptability = ChrF-Phrase × PhraseLC.
  - Smoothness: stability of TTS speaking rate across contiguous target phrases (their [6]).
- Relaxation: off-screen speech gets looser sync constraints — matches our mouth-visible slack policy (EXP-TIMING-004), literature-validated.
- **For us (EXP-TIMING-005 candidate):** our per-line translation prompt can carry the source's pause structure (`[pause]` where source words gap ≥300 ms) and ask for a translation with `[pause]` markers → the fitter then places phrases on those anchors (we already have anchored placement in fit_audio). Harness gains SA/PhraseLC/speaking-rate-smoothness.

**Prosodic alignment line** (Federico/Virkar et al., Interspeech/ICASSP 2020-22, arXiv:2204.02530): two-step alternative — translate, then a prosodic alignment model segments the translation onto source pauses using speaking-rate features and cross-lingual semantic matches; off-screen relaxation improves subjective scores. Our measured-candidate selection plays the same role; their relaxation idea = our slack flag.

**Intra-Sentential Speaking Rate Control in TTS for Dubbing** (Sharma et al., Interspeech 2021): control TTS rate per phrase instead of stretching audio afterwards — for us: IndexTTS/Qwen-TTS have no rate control, but candidate selection by measured duration approximates it; a rate-controllable TTS is the stronger version.

## Entity / OOV (ASR)
- **CB-Whisper** (LREC 2024): open-vocabulary keyword spotting on Whisper encoder states to recognize user-defined entities.
- **Zero-shot trie biasing with synthetic multi-pronunciations** (arXiv:2508.17796): TTS-synthesize pronunciation variants of hotwords, compile a prefix-trie, reward beam hypotheses (shallow fusion) — zero-shot on pretrained Whisper. Fits our title-lexicon: once the lexicon exists, re-decode suspicious spans with trie biasing.
- **OWSM-Biasing** (arXiv:2506.09448), **LLM-ASR hotword retrieval + RL** (arXiv:2512.21828): dynamic-vocabulary biasing; retrieval-augmented hotwords for large lists.
- Direction for ASR-002+: lexicon → biasing list → second decode pass of name spans.

## To read next
- Lip-sync generation beyond LatentSync (quality/resolution), dubbing quality evaluation (subjective protocols), AV-sync metrics beyond SyncNet, expressive S2S dubbing (VideoDubber, StreamSpeech-style), face-restoration for renderer blur.
