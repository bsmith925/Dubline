# Experiment log

One isolated change per entry. Bundles live in `eval/runs/`; `registry.jsonl` is the machine-readable record.

## Noise floor (core-v0)
Two runs of identical production code (`20260821-022651` vs `20260821-140505`): stretch p90 38.8 vs 46.2 %, coverage 0.76 vs 0.66, intelligibility 0.929 vs 0.894, speaker similarity 0.811 vs 0.796. Deltas smaller than these are not signal. LLM sampling, TTS sampling and retry paths are all stochastic; core-v1 should be large enough to shrink this.

## Experiment 1 — disable the "too-short" lengthening pass · keep
- **Hypothesis**: the length-control passes are the main source of extreme compression.
- **Variable**: `DUB_LENGTHEN_SHORT_TAKES` 1 → 0.
- **Result**: stretch mean 9.9 → 1.3 (p90 46 → 0, max 92 → 25); intelligibility 0.894 → 0.978; SyncNet ΔLSE-C vs source +0.39 → +0.47; judge unchanged; padding 1.1 → 2.0 s; coverage 0.66 → 0.59 (French 0.56 → 0.38); mouth-motion-on-silence 2.1 → 2.8 s (French +1.9 s); wall −10 %.
- **Conclusion**: keep. Compression is gone and speech is intelligible and in sync while it plays; what remains is that shorter target speech is front-loaded in the span and the mouth moves in silence afterwards — a placement problem, not a length problem. The pre-registered coverage rule (−0.05) was tripped but is inside the measured noise (0.10); recorded as an override.

## Experiment 2 — place take phrases on source articulation · running
- **Hypothesis**: distributing a take's phrases onto the source's speech runs (word-timestamp groups) instead of uniform pause spreading raises coverage of source articulation and lowers mouth-motion-on-silence, with stretch, intelligibility and judge unchanged.
- **Variable**: `DUB_PLACE_ON_SOURCE_RUNS` 0 → 1.
