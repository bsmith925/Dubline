from __future__ import annotations

"""Audio primitives shared by the metrics: speech-activity intervals and interval algebra."""

import numpy as np
import soundfile as sf


def speech_intervals(path, thresh_db: float = -42.0, hop: float = 0.02, win: float = 0.04,
                     min_gap: float = 0.12, min_run: float = 0.05, t_max: float | None = None,
                     offset: float = 0.0) -> list[list[float]]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if t_max is not None:
        mono = mono[: int(t_max * rate)]
    w = max(1, int(win * rate)); h = max(1, int(hop * rate))
    if len(mono) < w:
        return []
    frames = np.lib.stride_tricks.sliding_window_view(mono, w)[::h]
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1) + 1e-12) + 1e-9)
    active = db > thresh_db
    intervals: list[list[float]] = []
    start = None
    for index, flag in enumerate(active):
        t = index * hop
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            intervals.append([start, t]); start = None
    if start is not None:
        intervals.append([start, len(active) * hop])
    merged: list[list[float]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [[round(s + offset, 3), round(e + offset, 3)] for s, e in merged if e - s >= min_run]


def total(intervals: list[list[float]]) -> float:
    return round(sum(e - s for s, e in intervals), 3)


def clip(intervals: list[list[float]], start: float, end: float) -> list[list[float]]:
    out = []
    for s, e in intervals:
        a, b = max(s, start), min(e, end)
        if b > a:
            out.append([a, b])
    return out


def intersect(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            lo, hi = max(s1, s2), min(e1, e2)
            if hi > lo:
                out.append([lo, hi])
    return out


def subtract(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Parts of intervals ``a`` not covered by ``b``."""
    out = []
    for s, e in a:
        pieces = [[s, e]]
        for bs, be in b:
            next_pieces = []
            for ps, pe in pieces:
                if be <= ps or bs >= pe:
                    next_pieces.append([ps, pe])
                else:
                    if bs > ps:
                        next_pieces.append([ps, bs])
                    if be < pe:
                        next_pieces.append([be, pe])
            pieces = next_pieces
        out.extend(pieces)
    return out


def duration(path) -> float:
    return round(sf.info(path).duration, 3)
