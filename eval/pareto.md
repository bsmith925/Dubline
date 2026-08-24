# Quality vs compute — Pareto view (as of 2026-08-23 evening)

Wall = sum of job wall-seconds in the bundle (dubbing only; harness evaluation adds ~5 min/clip).
Quality headline = judge adequacy mean · LSE-C mean · lip-sync coverage mean. Full per-bundle data in `runs/*/summary.md`.

## core-v0 (2 gb clips) — the same content over time
| bundle | config era | wall min | adequacy | LSE-C | coverage |
|---|---|---|---|---|---|
| 20260822-105241 | reference-2 (pre face-crop) | 29 | 0.72 | 4.31 | – |
| 20260822-210233 | + VIDEO-007b face-crop | 61 | 0.88 | 5.25 | 0.79 |
| 20260823-150712 | + guidance 2.5 (VIDEO-008) | 59 | 0.87 | **5.96** | 0.79 |

+30 min bought: 2.5× lip-sync coverage (the PiP section animated at all) and +1.65 LSE-C.

## core-v1 (10 clips) — full-suite baselines
| bundle | era | wall min | adequacy | LSE-C |
|---|---|---|---|---|
| 20260822-094315 | baseline-2 | 142 | 0.81 | 4.14 |
| 20260822-153143 | baseline-3 (+TRANS-001, TIMING-001…) | 154 | 0.89 | 4.57 |
| 20260823-045814 | baseline-4 (+VIDEO-007, MIX-004) | 179 | 0.91 | 4.86 |
| 20260823-201946 | TIMING-004 (provisional) | 189 | 0.89 | 5.67 |

## trans-001 (3 clips) — iteration suite
| bundle | variable | wall min |
|---|---|---|
| 20260822-123139 | TRANS-001 | 46 |
| 20260822-220749 | MIX-004 | 52 |
| 20260823-011355 | TIMING-003 (measure all takes) | 80 |
| 20260823-164537 | MIX-006 | 78 |

## Cost levers (queued as experiments)
- LatentSync rendering = 40 % of pipeline wall → VIDEO-010: inference_steps 20→12 A/B.
- Candidate measurement = 12 % → TIMING-006: selective measurement (uncertain fits only).
- Iteration speed: job seeding (reuse separation/ASR/translation) + MUSETALK_ENABLED=false for non-visual experiments → expected trans-001 iteration ≈ 20 min (validation run queued). Seeded pairs also share upstream noise → cleaner comparisons.

## Iteration speed (measured 2026-08-24)
| run type | wall |
|---|---|
| trans-001 full (unseeded, lip-sync on) | 78 min |
| trans-001 seeded (tier=translation, lip-sync off) | **15 min** |
| core-v1 full | 179–189 min |

Seeding reuses separation/ASR/translation artifacts from a reference job (19–20 files per clip, hard-linked) so only the stages under test re-run; seeded pairs also share upstream noise, which shrinks the comparison's noise floor.

## Noise floors (3 identical runs) — decision thresholds
adequacy ±0.13 · stretch mean ±3.8 · worst-clip stretch p90 ±23.4 · fill p10 ±7.9 · fill p50 ±3.2 · padding ±90 ms · d2b SNR ±2.95 dB · bed-ratio ±0.21 dB · balance-vs-source ±0.01 dB · speaker similarity ±0.016 · naturalness MOS ±0.073 · entity consistency ±0.000 · wall ±3 min.
