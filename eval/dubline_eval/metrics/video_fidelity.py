from __future__ import annotations

"""Video fidelity outside the edited region, measured on the FINAL muxed video.

Invariant: pixels that should not have changed must remain effectively unchanged.
Inside lip-synced intervals the face box (from the harness's own landmarks) is
masked out and PSNR/SSIM are computed on the remainder; outside lip-synced
intervals the whole frame must match. Also reports the temporal difference
outside the mask (flicker/re-encode noise) and a coarse frame-difference heatmap.
"""

from pathlib import Path

import cv2
import numpy as np


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    mu_a = cv2.GaussianBlur(a, (7, 7), 1.5); mu_b = cv2.GaussianBlur(b, (7, 7), 1.5)
    sa = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a ** 2; sb = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(np.mean(((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))))


def outside_edit_fidelity(source: Path, output: Path, lipsync_intervals: list[list[float]], t_end: float,
                          face_boxes: dict[float, list[float]] | None = None, step: int = 5, heatmap_out: Path | None = None) -> dict:
    cs, co = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(output))
    fps_s = cs.get(cv2.CAP_PROP_FPS) or 30.0
    psnr_in, psnr_out, ssim_in, ssim_out, tdiff = [], [], [], [], []
    heat = None; prev_o = None; i = 0
    while True:
        ok, fs = cs.read()
        if not ok or i / fps_s > t_end:
            break
        if i % step == 0:
            co.set(cv2.CAP_PROP_POS_MSEC, i / fps_s * 1000); ok2, fo = co.read()
            if not ok2:
                break
            if fo.shape != fs.shape:
                fo = cv2.resize(fo, (fs.shape[1], fs.shape[0]))
            gs = cv2.cvtColor(cv2.resize(fs, (480, 270)), cv2.COLOR_BGR2GRAY)
            go = cv2.cvtColor(cv2.resize(fo, (480, 270)), cv2.COLOR_BGR2GRAY)
            t = i / fps_s
            inside = any(a <= t <= b for a, b in lipsync_intervals)
            mask = np.ones_like(gs, dtype=bool)
            if inside and face_boxes:
                near = min(face_boxes, key=lambda k: abs(k - t))
                if abs(near - t) < 0.5:
                    x1, y1, x2, y2 = [int(v / 4) for v in face_boxes[near]]   # boxes are in source pixels; frames scaled by 4 (1920->480)
                    mask[max(0, y1):y2, max(0, x1):x2] = False
            diff = (gs.astype(np.float32) - go.astype(np.float32)) ** 2
            mse = float(diff[mask].mean()) if mask.any() else 0.0
            psnr = 99.0 if mse < 1e-6 else float(10 * np.log10(255 ** 2 / mse))
            ssim = _ssim(np.where(mask, gs, 0), np.where(mask, go, 0))
            (psnr_in if inside else psnr_out).append(psnr); (ssim_in if inside else ssim_out).append(ssim)
            heat = diff if heat is None else heat + diff
            if prev_o is not None:
                tdiff.append(float(np.abs(go.astype(np.float32) - prev_o)[mask].mean()))
            prev_o = go.astype(np.float32)
        i += 1
    cs.release(); co.release()
    if heatmap_out is not None and heat is not None:
        h = heat / max(1, len(psnr_in) + len(psnr_out)); h = np.clip(h / max(1e-6, np.percentile(h, 99)) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(heatmap_out), cv2.applyColorMap(h, cv2.COLORMAP_INFERNO))
    summary = lambda xs: (round(float(np.median(xs)), 2), round(float(np.min(xs)), 2)) if xs else (None, None)
    return {"psnr_outside_edit_inside_lipsync": summary(psnr_in), "psnr_unedited_frames": summary(psnr_out),
            "ssim_outside_edit_inside_lipsync": summary(ssim_in), "ssim_unedited_frames": summary(ssim_out),
            "temporal_diff_outside_mask_mean": round(float(np.mean(tdiff)), 3) if tdiff else None,
            "frames_compared": len(psnr_in) + len(psnr_out)}
