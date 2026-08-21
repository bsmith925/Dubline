from __future__ import annotations

"""Compare two run bundles: per-metric, per-clip and per-utterance deltas; never hide regressions."""

from pathlib import Path

from .report import METRICS, load, summarize, value


def compare(baseline: Path, candidate: Path, top: int = 8) -> str:
    b_run, b_utts, b_clips = load(baseline)
    c_run, c_utts, c_clips = load(candidate)
    b_sum, c_sum = summarize(b_utts), summarize(c_utts)
    lines = [f"# Compare `{b_run['run_id']}` ({b_run['git_commit']}) → `{c_run['run_id']}` ({c_run['git_commit']})", "",
             "## Aggregate deltas (mean; sign shown as improvement direction)", "",
             "| metric | baseline | candidate | Δ | better? |", "|---|---|---|---|---|"]
    for name, (path, higher) in METRICS.items():
        if name in b_sum and name in c_sum:
            b, c = b_sum[name]["mean"], c_sum[name]["mean"]
            delta = c - b
            better = (delta > 0) == higher if abs(delta) > 1e-9 else None
            lines.append(f"| {name} | {b} | {c} | {delta:+.4f} | {'✓' if better else ('✗' if better is False else '–')} |")
    # per clip
    lines += ["", "## Per-clip", "", "| clip | metric | baseline | candidate | Δ |", "|---|---|---|---|---|"]
    key = lambda u: (u["identity"]["clip_id"], u["identity"]["utterance_id"])
    b_by = {key(u): u for u in b_utts}; c_by = {key(u): u for u in c_utts}
    clips = sorted({k[0] for k in list(b_by) + list(c_by)})
    for clip in clips:
        for name, (path, higher) in METRICS.items():
            bv = [float(v) for v in (value(u, path) for k, u in b_by.items() if k[0] == clip) if v is not None]
            cv = [float(v) for v in (value(u, path) for k, u in c_by.items() if k[0] == clip) if v is not None]
            if bv and cv:
                lines.append(f"| {clip} | {name} | {sum(bv)/len(bv):.4f} | {sum(cv)/len(cv):.4f} | {sum(cv)/len(cv)-sum(bv)/len(bv):+.4f} |")
    # per utterance regressions
    changes = []
    for k in set(b_by) & set(c_by):
        for name, (path, higher) in METRICS.items():
            bv, cv = value(b_by[k], path), value(c_by[k], path)
            if bv is None or cv is None:
                continue
            delta = float(cv) - float(bv)
            scale = max(abs(float(bv)), 1e-6)
            signed = delta if higher else -delta          # positive = improvement
            changes.append((signed / scale, name, k, float(bv), float(cv)))
    changes.sort()
    def rows(items):
        return [f"| {k[0]} | U{k[1]} | {name} | {bv:.4f} | {cv:.4f} | {rel*100:+.0f}% | {c_by[k]['identity']['job_id']} |"
                for rel, name, k, bv, cv in items]
    lines += ["", f"## Worst regressions (top {top})", "", "| clip | utt | metric | baseline | candidate | rel | candidate job |", "|---|---|---|---|---|---|---|"]
    lines += rows([c for c in changes if c[0] < 0][:top])
    lines += ["", f"## Largest improvements (top {top})", "", "| clip | utt | metric | baseline | candidate | rel | candidate job |", "|---|---|---|---|---|---|---|"]
    lines += rows([c for c in reversed(changes) if c[0] > 0][:top])
    # runtime
    bw = sum(float(c.get("wall_seconds") or 0) for c in b_clips); cw = sum(float(c.get("wall_seconds") or 0) for c in c_clips)
    lines += ["", "## Runtime", "", f"baseline wall {bw:.0f}s · candidate wall {cw:.0f}s · Δ {cw-bw:+.0f}s", "",
              "Media for inspection: see `clips.jsonl` → `paths` in each bundle."]
    return "\n".join(lines) + "\n"
