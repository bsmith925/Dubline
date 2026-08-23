from __future__ import annotations

"""Isochrony metrics from the IAMT literature (arXiv:2112.08548), adapted to audio:

- Source phrase structure: gaps >= 300 ms between aligned source words split an utterance into phrases.
- Target phrase structure: silence runs >= 300 ms inside the placed take.
- SA (segmentation accuracy): target has the same number of phrases as the source.
- phrase_duration_compliance: every target phrase within +-tol of its source phrase duration
  (order-wise pairing), only defined when SA holds.
"""

from pathlib import Path

from .. import audio as A

PAUSE_S = 0.3


def source_phrases(cue: dict) -> list[float]:
    words = [w for w in (cue.get("words") or []) if w.get("start") is not None]
    if not words:
        return []
    phrases = []
    start = float(words[0]["start"]); last_end = float(words[0]["end"])
    for w in words[1:]:
        if float(w["start"]) - last_end >= PAUSE_S:
            phrases.append(last_end - start)
            start = float(w["start"])
        last_end = float(w["end"])
    phrases.append(last_end - start)
    return [p for p in phrases if p > 0.05]


def take_phrases(take: Path) -> list[float]:
    if not take.is_file():
        return []
    speech = A.speech_intervals(take, thresh_db=-40, min_gap=PAUSE_S, min_run=0.05)
    return [b - a for a, b in speech if b - a > 0.05]


def pause_structure(cue: dict, take: Path, tolerance: float = 0.2) -> dict:
    src = source_phrases(cue); tgt = take_phrases(take)
    if not src or not tgt:
        return {}
    sa = len(src) == len(tgt)
    out = {"source_phrases": len(src), "target_phrases": len(tgt), "pause_sa": sa}
    if sa:
        out["phrase_duration_compliance"] = all(abs(t - s) <= max(0.25, tolerance * s) for s, t in zip(src, tgt))
    return out
