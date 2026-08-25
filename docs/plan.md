# Dubline plan (2026-08-25) — refocus on the perceptual objective

This is the standing plan. It replaces ad-hoc queue ordering. It exists because the last
iteration cycle produced real wins but also burned days on proxy metrics and knob
experiments whose deltas were inside the noise floor. The rules below are meant to make
that structurally hard to repeat.

## 1. Where iteration went wrong (read this before queueing anything)

The eval infrastructure is good. The failure was **metric literacy**, in three forms:

1. **Optimizing a proxy instead of the objective.** The timing track spent five
   experiments (TIMING-002/004, VIDEO-004, placement exp-2 — all reverted) pushing
   slot-fill % and stretch p90, while the actual perceptual cost of the fitting step went
   unmeasured. When UTMOS finally landed we learned time-fitting costs **0.50 MOS per
   take**, and jobs stretched 55–70 % score **1.4–2.3 MOS** vs **3.5–4.2** for
   unstretched jobs. Fill % is a *constraint* (don't leave dead air / a moving silent
   mouth); it was treated as the *goal*.
2. **Deciding on metrics that cannot decide.** Worst-clip stretch p90 has a ±23.4 noise
   floor; dialogue-to-bed SNR ±2.95 dB; coverage ±0.10. Several keeps were argued from
   these before the noise-floor audit corrected them. A metric with a floor that large is
   diagnostic colour, never a verdict.
3. **Scoring metrics out of their domain.** UTMOS was "failed" by scoring it on mixed
   film audio against video/timing anchors, then reinstated a day later when tested on
   the axis it actually measures. Anchor validation must match the metric's own axis.

## 2. Metric doctrine

**Decision-grade** (anchor-validated, floor from 3 identical seeded runs; a keep/revert
must move one of these by ≥ 3× its floor, pre-registered):

| metric | floor | axis |
|---|---|---|
| naturalness_mos (UTMOS, isolated takes) | ±0.073 | speech naturalness / time-fitting damage |
| judge adequacy (suite mean) | ±0.13 | translation meaning |
| entity consistency | ±0.000 | names |
| bed_ratio_delta_db | ±0.21 | mix balance |
| balance_vs_source | ±0.010 | mix balance |
| speaker similarity (mean) | ±0.016 | voice |
| active fill p50 | ±3.2 | timing (constraint) |
| padding (mean) | ±90 ms | timing (constraint) |

**Diagnostic-only — never the basis of a keep/revert:** worst-clip stretch p90 (±23.4),
dialogue-to-bed SNR (±2.95), coverage (±0.10), mouth_motion_on_silence (conflates audio
and visual claims — see timing-architecture §4.7), and **all visual metrics (LSE-C,
sharpness, articulation) until VISUAL-NOISE-FLOOR lands**. VIDEO-008's keep
(LSE-C 5.25 → 5.96) is provisional until then.

**Rules** (already adopted; restated as standing):
- An expected effect < 3× the floor does not earn a GPU run. Redesign it (more clips,
  seeded pair) or drop it.
- Seeded paired runs (`seed_from_job`, lip-sync off where visual is untouched) are the
  default: 15 min vs 78 min, and shared upstream noise shrinks the effective floor.
- A new metric decides nothing until `eval/anchors.py` validates it on its own axis.
- Every experiment report leads with one plain-language perceptual claim and the single
  decision-grade metric that supports it. The other numbers go below the fold.

## 3. The objective, ranked by what a viewer notices

1. **Speech sounds natural.** Worst validated axis today: core-v1 naturalness mean 1.88
   vs 3.5–4.2 demonstrated achievable on unstretched jobs. Up to ~2 MOS on the table.
2. **Meaning is right.** Adequacy 0.89–0.91 after TRANS-001/002 — good; maintain, don't
   chase (floor ±0.13 means small "gains" here are noise).
3. **Names are right.** Entity track on rails (lexicon keep; ASR-002b next).
4. **Mix sounds like the source.** Within floors after MIX-004/006 — maintain.
5. **Lips look plausible.** Coverage fixed (0.24 → 0.86). Renderer sharpness (~0.62 of
   source) is a model ceiling — a separate, expensive track; do not iterate knobs on it.

## 4. Priorities

### P0 — timing-for-naturalness (the ~2-MOS lever)
*(updated 2026-08-25 after the offline replay + dose-response, eval/experiments.md)*
- ~~EXP-TIMING-007~~ **cancelled before its GPU run**: the offline replay showed +0.03
  MOS (bar was +0.22). Slowdown damage saturates by ~5 %, so least-stretch tie-breaking
  cannot help. Implementation stays in the tree (`DUB_PREFER_LEAST_STRETCH`, off).
- **EXP-TIMING-008 — the lever, offline-validated**: `DUB_MAX_SLOWDOWN=1.0` (never slow
  the voice; pad instead). Paired sim on 137 affected takes: **+0.827 MOS** (11× floor),
  133/137 improved. First GPU experiment when free (after the visual noise floor): one
  seeded trans-001 pair confirming fill p50 / padding / mouth-motion / adequacy hold.
  Compression stays for overruns; its cap (~8–10 % per the dose-response) is a follow-up
  single variable.
- Add naturalness_mos to every bundle summary and to the Pareto axes (fill % vs MOS).
- **EXP-TIMING-005** (after 008): pause-structure translation (`[pause]` markers, AWS
  isochrony paper) — attacks length at the *translation* end so fitting has less to do.
  SA/PhraseLC metrics already in the harness.

### P1 — unblock and broaden (GPU, when Brad frees it)
- **VISUAL-NOISE-FLOOR** (2 identical core-v0 runs) — first GPU job. Validates or
  refutes VIDEO-008 and gates every future visual keep.
- **Speaker-similarity tail**: p10 ≈ 0.45, min 0.22 — top-5 failure mode, zero
  experiments so far. Start with characterization only (which cues, which speakers,
  reference quality?) before any knob.
- **ASR-002b**: lexicon → trie/hotword biasing → re-decode name spans (design is CPU).

### P2 — cost and ceilings (only after P0/P1 move)
- VIDEO-010: LatentSync inference steps 20 → 12 (LatentSync = 40 % of wall).
- Face restoration on the rendered crop (VIDEO-009) / LTX-2.5 decision — park until the
  visual noise floor exists and P0 lands.

## 5. Stop doing

- **No placement/slack/duration-model heuristics.** Four reverts on this class
  (TIMING-002, TIMING-004, VIDEO-004, exp-2). The standing rule from exp-2 holds:
  timing work goes through candidate measurement (TIMING-001/003) and, structurally,
  the natural-speech → target-timeline planner — not per-knob tweaks.
- No keeps/reverts argued from diagnostic-only metrics (§2 list).
- No visual keeps until the visual noise floor lands.
- No metric verdicts without same-axis anchors (the UTMOS lesson).
- No reconciling residual dB on metrics whose floor exceeds the residual.

## 6. Open items / waiting on Brad

- Timestamp (±2 s) of the "light-beam" artifact in
  `~/Downloads/dubline/exp-video-008/*.mkv` — VIDEO-008's keep is on hold for it.
- GPU handback signal (all Dubline GPU work stopped 2026-08-24 09:20). First jobs on
  return: VISUAL-NOISE-FLOOR, then EXP-TIMING-007.
