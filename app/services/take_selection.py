"""Measured-candidate take selection (EXP-TIMING-001/003/007).

The pipeline voices alternative phrasings for a cue, measures each (active duration,
stretch, slowdown), and picks one. Two policies:

- max_fill (EXP-TIMING-001/003): reward filling the slot; reject anything needing more
  compression than timing QC allows; prefer the longest acceptable take.
- least_stretch (EXP-TIMING-007): time-fitting audibly damages speech (-0.50 MOS per
  take; jobs stretched 55-70 % score 1.4-2.3 vs 3.5-4.2 unstretched), while padding is
  free. Among options whose fill is comparable, prefer the one needing the least
  time manipulation.

Kept free of pipeline imports so the offline replay tooling can reuse it verbatim.
"""
from __future__ import annotations

# An option must beat the current take's fill score by this much to switch (max_fill),
# or to count as a fill improvement worth switching for (least_stretch).
FILL_MARGIN = 0.02
# least_stretch: options within this fill-score distance of the best fill compete on
# manipulation instead of fill.
FILL_TOLERANCE = 0.05
# least_stretch: switching away from the current take purely for less manipulation
# requires at least this many percentage points of reduction.
MIN_MANIPULATION_GAIN = 1.0


def fill_score(measurement: dict, target: float) -> float:
    """Slot-fill reward in [0, 1]; -1 when the take is unusable (no speech or over the
    stretch limit timing QC enforces)."""
    active = float(measurement.get("active_duration") or 0.0)
    stretch = abs(float(measurement.get("stretch_percent") or 0.0))
    limit = float(measurement.get("_stretch_limit") or 0.0)
    if active <= 0 or stretch > limit:
        return -1.0
    return min(1.0, active / max(0.1, target))


def manipulation_percent(measurement: dict) -> float:
    """Total time manipulation the fitter applied: compression plus slowdown, in
    percentage points. Both directions damage naturalness; padding does not."""
    return abs(float(measurement.get("stretch_percent") or 0.0)) + abs(
        float(measurement.get("slowdown_percent") or 0.0))


def select_take(current: dict, candidates: dict[int, dict], target: float, limit: float,
                prefer_least_stretch: bool = False) -> int:
    """Return the winning option: 0 keeps the current take, k >= 1 selects candidate k.

    `current` and each candidate carry the fit measurements (active_duration,
    stretch_percent, slowdown_percent). `limit` is the timing-QC stretch tolerance for
    this cue (5 % mouth visible / 8 % not).
    """
    pool = {0: dict(current), **{k: dict(m) for k, m in candidates.items()}}
    for measurement in pool.values():
        measurement["_stretch_limit"] = limit
    scores = {k: fill_score(m, target) for k, m in pool.items()}

    if not prefer_least_stretch:
        best_key, best_score = 0, scores[0]
        for k in sorted(candidates):
            if scores[k] > best_score + FILL_MARGIN:
                best_key, best_score = k, scores[k]
        return best_key

    best_fill = max(scores.values())
    if best_fill < 0:
        return 0
    eligible = {k for k, score in scores.items() if score >= best_fill - FILL_TOLERANCE}
    # Least manipulation wins; ties break toward higher fill, then toward keeping the
    # current take, then toward the earlier candidate (determinism).
    winner = min(eligible, key=lambda k: (manipulation_percent(pool[k]), -scores[k], k != 0, k))
    if winner != 0 and 0 in eligible:
        gain = manipulation_percent(pool[0]) - manipulation_percent(pool[winner])
        if gain < MIN_MANIPULATION_GAIN and scores[winner] - scores[0] <= FILL_MARGIN:
            return 0
    return winner
