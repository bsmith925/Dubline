# Entity track — names and terminology as a first-class object

**Problem.** ASR mis-hears out-of-vocabulary names (Casey *Muratori* → "Murata"), translation faithfully
preserves the wrong string, TTS pronounces it, back-transcription agrees, and every downstream metric
reports success. The transcript is not the script. Professional localization solves this with a
title-specific terminology/pronunciation sheet carried through every stage (anime is the extreme case:
invented names, attacks, honorifics with no general-language prior).

**Principle.** `ASR transcript != canonical script`. Ordinary speech can be trusted; suspicious spans
(names, terminology, unusual tokens) go through a reconciliation stage that builds a per-title lexicon
from all available evidence and then feeds ASR correction, translation preservation, TTS pronunciation
and QC from that one artifact.

```
raw media → ASR hypotheses (primary + second recognizer, N-best where available)
          → entity / unusual-token discovery over the WHOLE program
          → context resolver: title/description metadata · on-screen OCR · subtitles · prior utterances
                              · recurring speaker/name graph · second-ASR hypotheses · optional knowledge lookup
          → per-title lexicon  {canonical, aliases, type, confidence, evidence, pronunciation}
          → canonical transcript (suspicious spans re-decoded with context) → translation → TTS → QC
```

Resolution is iterative and program-wide: a lone uncertain "Murata" is resolved retroactively once
"Casey Muratori" / "Muratori's approach" / "Casey" appear elsewhere. Once an entity is established, ASR
hypotheses in its phonetic neighbourhood receive a strong prior toward it.

**Decoding note.** Greedy is not inherently "more phonetic" (the LM prior lives in the seq2seq decoder),
but beam search amplifies common lexical sequences; alternative decodings are useful as *competing
hypotheses* for the resolver, not as a fix. Measured on core-v1: greedy vs beam-5 made no systematic
difference on names.

**Evidence (core-v1, baseline-3).** The same recognizer spells one name several ways within a film:
Lampert/Lampard/Lombard/Lampeth, Voss/Vasse's, Scobie/Scoby, Moritore/Muratori/Murata. Consensus
across cues and recognizers usually contains the right stem.

## Deliverables (eval first)
1. Harness metrics: `entity_consistency_across_cues` (spellings per phonetic cluster), `cross_recognizer_entity_agreement`,
   `translation_entity_preservation`, `tts_entity_pronunciation`. Entity errors are high severity in reports.
2. Title lexicon builder v0 (EXP-ENTITY-001): whole-program discovery from both recognizers + filename/title metadata;
   phonetic clustering (metaphone + edit distance); canonical spelling by weighted consensus; lexicon artifact;
   transcript rewrite before translation; lexicon → translation `protected_names` → TTS glossary → QC.
3. Evidence sources: on-screen OCR, subtitles, N-best from the primary ASR, knowledge lookup (opt-in).
4. Resolver scoring: acoustic + context + entity prior over N-best, replacing majority vote.
