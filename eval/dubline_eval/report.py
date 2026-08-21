from __future__ import annotations

"""Aggregate a run bundle into per-metric summaries (raw records stay untouched)."""

import json
import statistics as st
from pathlib import Path

# metric name -> (path in record, higher_is_better)
METRICS: dict[str, tuple[tuple[str, ...], bool]] = {
    "speaker_similarity": (("voice", "speaker_similarity"), True),
    "word_similarity": (("tts", "word_similarity"), True),
    "judge_adequacy": (("translation", "judge_adequacy"), True),
    "untranslated_word_rate": (("translation", "untranslated_word_rate"), False),
    "stretch_percent": (("tts", "stretch_percent"), False),
    "slowdown_percent": (("tts", "slowdown_percent"), False),
    "padding_ms": (("tts", "padding_ms"), False),
    "target_language_prob": (("tts", "language_id", "__target__"), True),
    "coverage_articulation": (("visual", "coverage_articulation"), True),
    "mouth_motion_on_silence_s": (("visual", "mouth_motion_on_silence"), False),
    "speech_on_static_mouth_s": (("visual", "speech_on_static_mouth"), False),
    "lipsync_clip_length_ratio_err": (("visual", "lipsync_clip_length_ratio"), False),
    "aperture_ratio": (("visual", "aperture_ratio"), True),
    "sync_lse_c": (("visual", "sync_lse_c"), True),
    "sync_lse_d": (("visual", "sync_lse_d"), False),
    "delta_lse_c_vs_source": (("visual", "delta_lse_c"), True),
    "delta_lse_d_vs_source": (("visual", "delta_lse_d"), False),
    "sync_offset_abs_frames": (("visual", "sync_offset_frames"), False),
    "sync_low_conf_fraction": (("visual", "sync_low_conf_fraction"), False),
    "mouth_sharpness_ratio": (("visual", "mouth_sharpness_ratio"), True),
    "boundary_jump_x_median": (("visual", "boundary_jump_x_median"), False),
    "retries": (("system", "retries"), False),
}

LANG_CODES = {"english": "en", "french": "fr", "spanish": "es", "german": "de", "italian": "it", "portuguese": "pt",
              "japanese": "ja", "korean": "ko", "chinese": "zh", "russian": "ru", "arabic": "ar"}


def value(record: dict, path: tuple[str, ...]):
    node = record
    for key in path:
        if key == "__target__":
            code = LANG_CODES.get(str(record["identity"]["target_language"]).lower(), "")
            return (node or {}).get(code) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if path[-1] == "lipsync_clip_length_ratio" and node is not None:
        return abs(float(node) - 1.0)
    if path[-1] == "sync_offset_frames" and node is not None:
        return abs(float(node))
    return node


def load(bundle: Path) -> tuple[dict, list[dict], list[dict]]:
    run = json.loads((bundle / "run.json").read_text())
    utts = [json.loads(l) for l in (bundle / "utterances.jsonl").read_text().splitlines() if l.strip()]
    clips = [json.loads(l) for l in (bundle / "clips.jsonl").read_text().splitlines() if l.strip()]
    return run, utts, clips


def summarize(utts: list[dict]) -> dict[str, dict]:
    out = {}
    for name, (path, _) in METRICS.items():
        vals = [float(v) for v in (value(u, path) for u in utts) if v is not None]
        if vals:
            out[name] = {"n": len(vals), "mean": round(st.mean(vals), 4), "median": round(st.median(vals), 4),
                         "p90": round(sorted(vals)[min(len(vals) - 1, int(0.9 * len(vals)))], 4),
                         "max": round(max(vals), 4), "min": round(min(vals), 4)}
    return out


def write_summary(bundle: Path) -> None:
    run, utts, clips = load(bundle)
    summary = summarize(utts)
    lines = [f"# Eval run {run['run_id']}", "",
             f"suite `{run['suite']}` · commit `{run['git_commit']}` · config `{run['config_hash']}` · models: "
             + ", ".join(f"{k}={v}" for k, v in (run.get('models') or {}).items()), "",
             f"{len(clips)} clip(s), {len(utts)} utterance(s)", "", "## Clips", "",
             "| clip | job | utt | wall s | QC | lip-sync | dead air s | unmuted src s | take overlap s | default audio | mouth-on-silence total s | boundary jump max | sharpness | bed RMS Δ dB | bed missing | PSNR outside edit (med/min) | PSNR unedited (med/min) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in clips:
        lines.append(f"| {c['clip_id']} | {c['job_id']} | {c['utterance_count']} | {c.get('wall_seconds')} | {c.get('delivery_qc_passed')} | {c['lipsync_rendered']} | "
                     f"{c.get('dead_air_seconds')} | {c.get('unmuted_source_speech_seconds')} | {c.get('take_overlap_seconds')} | {c.get('default_audio_streams')} | "
                     f"{c.get('mouth_motion_on_silence_total_s')} | {c.get('boundary_jump_max_x_median')} | {c.get('mouth_sharpness_ratio')} | "
                     f"{(c.get('mix_fidelity') or {}).get('bed_rms_delta_db')} | {(c.get('mix_fidelity') or {}).get('bed_missing_fraction')} | "
                     f"{(c.get('video_fidelity') or {}).get('psnr_outside_edit_inside_lipsync')} | {(c.get('video_fidelity') or {}).get('psnr_unedited_frames')} |")
    lines += ["", "## Utterance metrics (raw distributions)", "", "| metric | n | mean | median | p90 | max | min |", "|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['n']} | {s['mean']} | {s['median']} | {s['p90']} | {s['max']} | {s['min']} |")
    flags: dict[str, int] = {}
    for u in utts:
        for f in u.get("flags", []):
            flags[f[:70]] = flags.get(f[:70], 0) + 1
    lines += ["", "## Pipeline review flags", ""] + [f"- {n} × {f}" for f, n in sorted(flags.items(), key=lambda kv: -kv[1])]
    (bundle / "summary.md").write_text("\n".join(lines) + "\n")
    (bundle / "summary.json").write_text(json.dumps({"metrics": summary, "flags": flags}, indent=2))
