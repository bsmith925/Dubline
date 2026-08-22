from __future__ import annotations

"""Audio-mix fidelity: did the non-dialogue bed survive?

Compares the source soundtrack with the final output in regions where the dub is
NOT speaking (dub speech masked out), and the separated M&E stem with the final
output in the same regions. Differences there mean something downstream damaged
the bed (over-aggressive separation, ducking, mastering), independent of the dub.
"""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import audio as A


def _mono(path: Path, rate: int = 48000, t_max: float | None = None) -> np.ndarray:
    tmp = Path(f"/tmp/mixfid-{path.stem}-{rate}.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate), *(["-t", f"{t_max:.3f}"] if t_max else []), str(tmp)],
                   check=True, capture_output=True)
    x, _ = sf.read(tmp, dtype="float32", always_2d=True)
    return x.mean(axis=1)


def _db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x * x) + 1e-12) + 1e-9))


def _band_spectrum(x: np.ndarray, rate: int, bands: int = 24) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    edges = np.geomspace(40, rate / 2, bands + 1)
    out = np.array([spec[(freqs >= lo) & (freqs < hi)].sum() for lo, hi in zip(edges, edges[1:])])
    return 10 * np.log10(out + 1e-9)


def mix_fidelity(source_mix: Path, final_mix: Path, dub_speech: list[list[float]], t_end: float,
                 source_me: Path | None = None, rate: int = 48000, source_dialogue: Path | None = None) -> dict:
    src = _mono(source_mix, rate, t_end); out = _mono(final_mix, rate, t_end)
    n = min(len(src), len(out)); src, out = src[:n], out[:n]
    total = [[0.0, n / rate]]
    # regions where the dub is silent (with 150 ms guard), i.e. the bed alone is audible in the output
    guarded = [[max(0, a - .15), b + .15] for a, b in dub_speech]
    quiet = [iv for iv in A.subtract(total, guarded) if iv[1] - iv[0] >= 0.5]
    # ... and where the SOURCE was not speaking either: source speech under a silent dub is a
    # timing/placement defect (reported separately), not a bed problem.
    source_speech_no_dub = 0.0
    if source_dialogue is not None and source_dialogue.is_file():
        src_speech = [[max(0, a - .1), b + .1] for a, b in A.speech_intervals(source_dialogue, thresh_db=-40, t_max=t_end)]
        source_speech_no_dub = A.total(A.intersect(quiet, src_speech))
        quiet = [iv for iv in A.subtract(quiet, src_speech) if iv[1] - iv[0] >= 0.5]
    if not quiet:
        return {"seconds_evaluated": 0.0, "source_speech_without_dub_s": round(source_speech_no_dub, 2)}
    def gather(x):
        return np.concatenate([x[int(a * rate):int(b * rate)] for a, b in quiet])
    s_q, o_q = gather(src), gather(out)
    rms_delta = _db(o_q) - _db(s_q)
    spectral = float(np.mean(np.abs(_band_spectrum(s_q, rate) - _band_spectrum(o_q, rate))))
    # per-second bed level tracking in the quiet regions: how often is the output >12 dB below the source?
    missing = 0; windows = 0
    for a, b in quiet:
        for t in np.arange(a, b - 0.5, 0.5):
            ws, wo = src[int(t * rate):int((t + .5) * rate)], out[int(t * rate):int((t + .5) * rate)]
            windows += 1
            if _db(ws) > -50 and _db(wo) < _db(ws) - 12:
                missing += 1
    result = {"seconds_evaluated": round(sum(b - a for a, b in quiet), 2), "source_speech_without_dub_s": round(source_speech_no_dub, 2),
              "bed_rms_delta_db": round(rms_delta, 2), "bed_spectral_distance_db": round(spectral, 2),
              "bed_missing_fraction": round(missing / max(1, windows), 3)}
    if source_me and source_me.is_file():
        me = _mono(source_me, rate, t_end)[:n]
        m_q = gather(me)
        result["me_vs_source_rms_delta_db"] = round(_db(m_q) - _db(s_q), 2)       # how much the separator removed
        result["output_vs_me_rms_delta_db"] = round(_db(o_q) - _db(m_q), 2)       # what mixing/mastering did to the bed
    # SNR of dub over bed inside dub speech (should be healthy, ~10-20 dB)
    loud = [iv for iv in dub_speech if iv[1] - iv[0] >= 0.3]
    if loud:
        o_l = np.concatenate([out[int(a * rate):int(b * rate)] for a, b in loud])
        result["dialogue_to_bed_snr_db"] = round(_db(o_l) - _db(o_q), 2)
    result["clipped_samples"] = int(np.sum(np.abs(out) >= 0.999))
    return result


def mastering_effect(premaster: Path, final_mix: Path, dub_speech: list[list[float]], t_end: float, rate: int = 48000) -> dict:
    """What mastering did to speech versus the rest: gain applied in dub-speech regions minus gain
    applied elsewhere (negative = speech was limited/compressed harder than the bed)."""
    pm = _mono(premaster, rate, t_end); out = _mono(final_mix, rate, t_end)
    n = min(len(pm), len(out)); pm, out = pm[:n], out[:n]
    guarded = [[max(0, a - .15), b + .15] for a, b in dub_speech]
    rest = [iv for iv in A.subtract([[0.0, n / rate]], guarded) if iv[1] - iv[0] >= 0.5]
    speech = [iv for iv in dub_speech if iv[1] - iv[0] >= 0.3]
    if not speech or not rest:
        return {}
    gather = lambda x, ivs: np.concatenate([x[int(a * rate):int(b * rate)] for a, b in ivs])
    g_speech = _db(gather(out, speech)) - _db(gather(pm, speech))
    g_rest = _db(gather(out, rest)) - _db(gather(pm, rest))
    return {"master_gain_speech_db": round(g_speech, 2), "master_gain_rest_db": round(g_rest, 2),
            "master_dialogue_squash_db": round(g_speech - g_rest, 2)}
