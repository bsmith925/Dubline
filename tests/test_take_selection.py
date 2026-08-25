from __future__ import annotations

from app.services.take_selection import fill_score, manipulation_percent, select_take


def m(active=0.0, stretch=0.0, slowdown=0.0):
    return {"active_duration": active, "stretch_percent": stretch, "slowdown_percent": slowdown}


def test_fill_score_rejects_over_limit_and_silent_takes():
    assert fill_score({**m(active=4.0, stretch=9.0), "_stretch_limit": 8.0}, 5.0) == -1.0
    assert fill_score({**m(active=0.0), "_stretch_limit": 8.0}, 5.0) == -1.0
    assert fill_score({**m(active=4.0, stretch=7.9), "_stretch_limit": 8.0}, 5.0) == 0.8


def test_manipulation_counts_both_directions():
    assert manipulation_percent(m(stretch=-6.0, slowdown=3.0)) == 9.0
    assert manipulation_percent({}) == 0.0


# --- max_fill (EXP-TIMING-001/003 behaviour, unchanged) ---------------------------

def test_max_fill_picks_highest_fill_over_margin():
    current = m(active=3.0)  # fill 0.60
    candidates = {1: m(active=3.05), 2: m(active=4.5)}  # 0.61, 0.90
    assert select_take(current, candidates, target=5.0, limit=8.0) == 2


def test_max_fill_keeps_current_within_margin():
    current = m(active=4.45)
    candidates = {1: m(active=4.5)}  # +0.01 fill: inside the 0.02 margin
    assert select_take(current, candidates, target=5.0, limit=8.0) == 0


def test_max_fill_rejects_candidates_over_stretch_limit():
    current = m(active=3.0, stretch=2.0)
    candidates = {1: m(active=5.0, stretch=9.0)}
    assert select_take(current, candidates, target=5.0, limit=8.0) == 0


def test_max_fill_rescues_unusable_current():
    current = m(active=3.0, stretch=12.0)  # over limit -> score -1
    candidates = {1: m(active=2.0, stretch=1.0)}
    assert select_take(current, candidates, target=5.0, limit=8.0) == 1


# --- least_stretch (EXP-TIMING-007) -----------------------------------------------

def test_least_stretch_prefers_less_manipulation_when_fills_comparable():
    current = m(active=4.8, stretch=-7.0)          # fill 0.96, manipulation 7
    candidates = {1: m(active=4.6, stretch=-0.5)}  # fill 0.92 (within 0.05), manipulation 0.5
    assert select_take(current, candidates, target=5.0, limit=8.0, prefer_least_stretch=True) == 1


def test_least_stretch_does_not_sacrifice_fill():
    current = m(active=4.8, stretch=-7.0)          # fill 0.96
    candidates = {1: m(active=3.0, stretch=0.0)}   # fill 0.60: outside tolerance
    assert select_take(current, candidates, target=5.0, limit=8.0, prefer_least_stretch=True) == 0


def test_least_stretch_keeps_current_for_marginal_gain():
    current = m(active=4.8, stretch=-0.6)
    candidates = {1: m(active=4.8, stretch=-0.1)}  # gain 0.5 < 1.0 point
    assert select_take(current, candidates, target=5.0, limit=8.0, prefer_least_stretch=True) == 0


def test_least_stretch_still_switches_for_pure_fill_gain():
    current = m(active=3.0)                        # fill 0.60, no manipulation
    candidates = {1: m(active=4.9)}                # fill 0.98, no manipulation
    assert select_take(current, candidates, target=5.0, limit=8.0, prefer_least_stretch=True) == 1


def test_least_stretch_counts_slowdown_as_manipulation():
    current = m(active=4.8, slowdown=6.0)
    candidates = {1: m(active=4.7, slowdown=0.0)}
    assert select_take(current, candidates, target=5.0, limit=8.0, prefer_least_stretch=True) == 1


def test_least_stretch_ties_break_toward_current():
    current = m(active=4.8, stretch=-2.0)
    candidates = {1: m(active=4.8, stretch=-2.0), 2: m(active=4.8, stretch=-2.0)}
    assert select_take(current, candidates, target=5.0, limit=8.0, prefer_least_stretch=True) == 0


def test_least_stretch_keeps_current_when_everything_unusable():
    current = m(active=3.0, stretch=20.0)
    candidates = {1: m(active=3.0, stretch=15.0)}
    assert select_take(current, candidates, target=5.0, limit=8.0, prefer_least_stretch=True) == 0
