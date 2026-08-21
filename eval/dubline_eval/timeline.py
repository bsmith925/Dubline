from __future__ import annotations

"""Unified absolute-time timeline plot for one clip."""

import numpy as np


def plot_timeline(timeline: dict, title: str, out_png, t_start: float = 0.0, t_end: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = t_end or timeline["t_end"]
    rows = [
        ("source speech (VAD, dialogue stem)", [tuple(iv) for iv in timeline["source_speech"]], "#4a5568"),
        ("source articulation (mouth aperture)", [tuple(iv) for iv in timeline.get("source_articulation", [])], "#718096"),
        ("ASR words", [(w["start"], w["end"]) for w in timeline.get("asr_words", [])], "#2b6cb0"),
        ("ASR fragments", [(f["start"], f["end"]) for f in timeline.get("asr_fragments", [])], "#3182ce"),
        ("speaker turns", [(x["start"], x["end"]) for x in timeline.get("speaker_turns", [])], "#805ad5"),
        ("utterances (translation + TTS units)", [(u["start"], u["end"]) for u in timeline["utterances"]], "#0f766e"),
        ("TTS take speech", [tuple(iv) for u in timeline["utterances"] for iv in u.get("take_speech", [])], "#d69e2e"),
        ("dub track speech (VAD)", [tuple(iv) for iv in timeline["dub_speech"]], "#dd6b20"),
        ("output articulation (after lip-sync)", [tuple(iv) for iv in timeline.get("output_articulation", [])], "#e53e3e"),
        ("lip-sync clip interval", [(x["start"], x["end"]) for x in timeline.get("lipsync_intervals", [])], "#9b2c2c"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={"height_ratios": [3.4, 1.4]}, sharex=True)
    ax = axes[0]
    for i, (label, ivs, color) in enumerate(rows):
        y = len(rows) - i
        for a, b in ivs:
            if b > t_start and a < T:
                ax.barh(y, min(b, T) - max(a, t_start), left=max(a, t_start), height=0.6, color=color,
                        alpha=0.6 if "words" in label else 0.85, edgecolor="none")
    ax.set_yticks(range(1, len(rows) + 1)); ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10)
    for u in timeline["utterances"]:
        if t_start <= u["start"] < T:
            ax.text(u["start"] + 0.1, len(rows) - 5 + 0.42,
                    f'U{u["id"]} span {u["end"] - u["start"]:.1f}s · TTS {u.get("raw_tts") or 0:.1f}s · fill {u.get("fill") or 0:.0f}%',
                    fontsize=8, color="#0f766e", va="bottom")
    for x in timeline.get("lipsync_intervals", []):
        if t_start <= x["start"] < T:
            ax.annotate(f'clip {x["end"] - x["start"]:.1f}s', (x["start"], 1.42), fontsize=8, color="#9b2c2c")
    ax.set_xlim(t_start, T); ax.grid(axis="x", alpha=.25); ax.set_title(title, fontsize=13, loc="left")

    ax2 = axes[1]
    def series(rows_):
        ok = [r for r in rows_ if r.get("inner") is not None and (r.get("face_h_px") or 0) >= 50]
        return np.array([r["t"] for r in ok]), np.array([r["inner"] for r in ok])
    xs, ys = series(timeline.get("mouth_source", [])); xo, yo = series(timeline.get("mouth_output", []))
    if len(xs):
        ax2.plot(xs, np.clip(ys, 0, 0.5), color="#4a5568", lw=1.1, label="source mouth opening (inner lips / face height)")
    if len(xo):
        ax2.plot(xo, np.clip(yo, 0, 0.5), color="#c53030", lw=1.1, label="output mouth opening (after lip-sync)")
    for a, b in timeline["dub_speech"]:
        if b > t_start and a < T:
            ax2.axvspan(max(a, t_start), min(b, T), color="#dd6b20", alpha=.12)
    ax2.set_ylim(0, 0.35); ax2.legend(loc="upper right", fontsize=9); ax2.set_ylabel("aperture (normalized)")
    ax2.set_xlabel("seconds (media time)"); ax2.grid(alpha=.25)
    ax2.text(t_start + 0.2, 0.32, "shaded = dub speech active", fontsize=8, color="#dd6b20")
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close(fig)
