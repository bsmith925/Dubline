#!/usr/bin/env python3
"""EXP-LIPSYNC-001: MuseTalk 1.5 vs LatentSync 1.6 on identical inputs.

Runs on the GPU host. For each shot: identical original frames (cut from the job's
selected-source.mkv) and the identical final dubbed take (the pipeline's
acoustically-matched wav), rendered by both models, then measured.

  vendor/latentsync-env/bin/python eval/lipsync_ab.py --job /mnt/media/dubline/data/jobs/<id> \
      --cues 1 2 3 --out eval/runs/lipsync-001

Metrics per shot and model: SyncNet LSE-C/LSE-D/offset (original SyncNet, vs source),
ArcFace identity drift vs source, LPIPS outside the mouth mask, temporal mouth-landmark
jitter ratio, mouth-motion-on-silence, runtime, peak VRAM.
Artifacts: side-by-side and difference videos, metrics.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MUSETALK = ROOT / "vendor/MuseTalk"
MUSETALK_PY = ROOT / "vendor/musetalk-env/bin/python"
LATENTSYNC = ROOT / "vendor/LatentSync"
LATENTSYNC_PY = ROOT / "vendor/latentsync-env/bin/python"
SYNCNET = ROOT / "vendor/syncnet_python"


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def probe_duration(p: Path) -> float:
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]).stdout.strip() or 0)


class VramMeter:
    def __init__(self):
        self.peak = 0; self._stop = False
        self.thread = threading.Thread(target=self._poll, daemon=True)
    def _poll(self):
        while not self._stop:
            try:
                used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                          capture_output=True, text=True).stdout.strip().splitlines()[0])
                self.peak = max(self.peak, used)
            except Exception:
                pass
            time.sleep(1)
    def __enter__(self):
        self.baseline = 0
        try:
            self.baseline = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                               capture_output=True, text=True).stdout.strip().splitlines()[0])
        except Exception:
            pass
        self.thread.start(); return self
    def __exit__(self, *a):
        self._stop = True; self.thread.join(timeout=2)
    @property
    def delta_mb(self): return max(0, self.peak - self.baseline)


def run_musetalk(frames_mp4: Path, audio_wav: Path, out_dir: Path, tag: str) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / f"{tag}.yaml"
    cfg.write_text(f'task_0:\n  video_path: "{frames_mp4}"\n  audio_path: "{audio_wav}"\n  result_name: "{tag}.mp4"\n')
    t = time.time()
    with VramMeter() as vm:
        sh([str(MUSETALK_PY), "-m", "scripts.inference", "--inference_config", str(cfg), "--result_dir", str(out_dir),
            "--unet_model_path", "models/musetalkV15/unet.pth", "--unet_config", "models/musetalkV15/musetalk.json",
            "--version", "v15", "--use_float16", "--batch_size", "16", "--extra_margin", "10", "--parsing_mode", "jaw"], cwd=MUSETALK)
    result = out_dir / "v15" / f"{tag}.mp4"
    return result, {"runtime_s": round(time.time() - t, 1), "peak_vram_mb": vm.delta_mb}


def run_latentsync(frames_mp4: Path, audio_wav: Path, out_dir: Path, tag: str) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = out_dir / f"{tag}.mp4"
    t = time.time()
    with VramMeter() as vm:
        sh([str(LATENTSYNC_PY), "-m", "scripts.inference", "--unet_config_path", "configs/unet/stage2_512.yaml",
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt", "--inference_steps", "20", "--guidance_scale", "1.5",
            "--video_path", str(frames_mp4), "--audio_path", str(audio_wav), "--video_out_path", str(result), "--seed", "1247"], cwd=LATENTSYNC)
    return result, {"runtime_s": round(time.time() - t, 1), "peak_vram_mb": vm.delta_mb}


MEASURE = r'''
import json, sys, cv2, numpy as np, torch
from face_alignment import FaceAlignment, LandmarksType
src, out, mouth_json, report = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device="cuda")
import lpips
loss = lpips.LPIPS(net="alex").cuda().eval()
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]); app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as exc:
    app = None; print("insightface unavailable:", exc, file=sys.stderr)

def frames(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    while True:
        ok, f = cap.read()
        if not ok: break
        yield f
    cap.release()

def lm(frame):
    pts = fa.get_landmarks_from_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not pts: return None
    return max(pts, key=lambda p: p[:, 0].max() - p[:, 0].min())

def embed(frame):
    if app is None: return None
    faces = app.get(frame)
    if not faces: return None
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    e = f.normed_embedding; return e

ident, lpips_out, jit_src, jit_out, aperture_src, aperture_out = [], [], [], [], [], []
prev_s = prev_o = None
for i, (fs, fo) in enumerate(zip(frames(src), frames(out))):
    if fo.shape != fs.shape: fo = cv2.resize(fo, (fs.shape[1], fs.shape[0]))
    ps, po = lm(fs), lm(fo)
    if ps is None or po is None: continue
    face_h = float(ps[8, 1] - ps[27, 1]);
    if face_h < 50: continue
    aperture_src.append(float(np.linalg.norm(ps[62]-ps[66]))/face_h); aperture_out.append(float(np.linalg.norm(po[62]-po[66]))/face_h)
    if prev_s is not None:
        jit_src.append(float(np.mean(np.linalg.norm(ps[48:68]-prev_s[48:68], axis=1)))/face_h)
        jit_out.append(float(np.mean(np.linalg.norm(po[48:68]-prev_o[48:68], axis=1)))/face_h)
    prev_s, prev_o = ps, po
    if i % 3 == 0:
        # mouth mask = lower face box from landmarks, padded; LPIPS outside it: copy source pixels into the box
        x1, y1 = int(ps[48:68, 0].min() - 0.15*face_h), int(ps[33, 1])   # from nose tip down
        x2, y2 = int(ps[48:68, 0].max() + 0.15*face_h), int(ps[8, 1] + 0.1*face_h)
        fo2 = fo.copy(); fo2[max(0,y1):y2, max(0,x1):x2] = fs[max(0,y1):y2, max(0,x1):x2]
        # crop a face-centred window so the metric is about the face, not the whole 1080p frame
        cx1, cy1 = max(0, int(ps[:,0].min()-0.3*face_h)), max(0, int(ps[:,1].min()-0.5*face_h))
        cx2, cy2 = int(ps[:,0].max()+0.3*face_h), int(ps[:,1].max()+0.2*face_h)
        a = cv2.resize(fs[cy1:cy2, cx1:cx2], (256, 256)); b = cv2.resize(fo2[cy1:cy2, cx1:cx2], (256, 256))
        ta = torch.from_numpy(a[:, :, ::-1].copy()).permute(2,0,1).float().cuda()/127.5-1; tb = torch.from_numpy(b[:, :, ::-1].copy()).permute(2,0,1).float().cuda()/127.5-1
        with torch.no_grad(): lpips_out.append(float(loss(ta[None], tb[None])))
        es, eo = embed(fs), embed(fo)
        if es is not None and eo is not None: ident.append(float(np.dot(es, eo)))
res = {"frames": len(aperture_src),
       "identity_cosine_mean": round(float(np.mean(ident)), 4) if ident else None, "identity_cosine_min": round(float(np.min(ident)), 4) if ident else None,
       "lpips_outside_mouth_mean": round(float(np.mean(lpips_out)), 4) if lpips_out else None,
       "jitter_src": round(float(np.mean(jit_src)), 5) if jit_src else None, "jitter_out": round(float(np.mean(jit_out)), 5) if jit_out else None,
       "jitter_ratio": round(float(np.mean(jit_out)/max(1e-6, np.mean(jit_src))), 3) if jit_src else None,
       "aperture_ratio": round(float(np.mean(aperture_out)/max(1e-6, np.mean(aperture_src))), 3) if aperture_src else None}
json.dump({"metrics": res, "aperture_out": aperture_out, "aperture_src": aperture_src}, open(report, "w"))
print(json.dumps(res))
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--cues", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", type=Path, default=ROOT / "eval/runs/lipsync-001")
    ap.add_argument("--models", nargs="+", default=["musetalk", "latentsync"])
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    args.job = args.job.resolve()
    cues = {int(c["id"]): c for c in json.loads((args.job / "cues.json").read_text())}
    from eval.dubline_eval.metrics.av_sync import score_interval
    from eval.dubline_eval import audio as A
    results = {"job": args.job.name, "shots": {}}
    for cue_id in args.cues:
        cue = cues[cue_id]; start, end = float(cue["start"]), float(cue["end"])
        tag = f"u{cue_id:03d}"; shot = out / tag; shot.mkdir(exist_ok=True)
        frames_mp4 = shot / "source.mp4"; audio_wav = shot / "dub.wav"
        sh(["ffmpeg", "-y", "-v", "error", "-accurate_seek", "-ss", f"{start:.3f}", "-i", str(args.job / "selected-source.mkv"),
            "-t", f"{end-start:.3f}", "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(frames_mp4)])
        take = args.job / "acoustically-matched" / f"{cue_id:06d}.wav"
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(take), "-t", f"{end-start:.3f}", "-ar", "16000", "-ac", "1", "-af", f"apad=whole_dur={end-start:.3f}", str(audio_wav)])
        dub_speech = A.speech_intervals(audio_wav, thresh_db=-40)
        results["shots"][tag] = {"start": start, "end": end, "models": {}}
        for model in args.models:
            try:
                rendered, cost = (run_musetalk if model == "musetalk" else run_latentsync)(frames_mp4, audio_wav, shot / model, tag)
            except subprocess.CalledProcessError as exc:
                results["shots"][tag]["models"][model] = {"error": (exc.stderr or exc.stdout)[-3000:]}; print(tag, model, "FAILED:", (exc.stderr or exc.stdout)[-1500:], flush=True); continue
            # mux identical audio onto the render for SyncNet and for viewing
            with_audio = shot / f"{model}-{tag}.mp4"
            sh(["ffmpeg", "-y", "-v", "error", "-i", str(rendered), "-i", str(audio_wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(with_audio)])
            source_with_audio = shot / f"source-{tag}.mp4"
            if not source_with_audio.is_file():
                sh(["ffmpeg", "-y", "-v", "error", "-i", str(frames_mp4), "-i", str(audio_wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(source_with_audio)])
            sync = score_interval(with_audio, source_with_audio, 0.0, end - start, shot / "sync", SYNCNET, MUSETALK_PY, f"{model}-{tag}") if (SYNCNET / "run_syncnet.py").is_file() else {}
            report = shot / f"{model}-measure.json"
            script = shot / "measure.py"; script.write_text(MEASURE)
            measure = json.loads(subprocess.run([str(LATENTSYNC_PY), str(script), str(frames_mp4), str(rendered), "-", str(report)], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
            ap_out = json.loads(report.read_text())["aperture_out"]
            # mouth motion on silence from the rendered aperture series at the render fps
            fps = 25.0
            try:
                fps = float(eval(sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(rendered)]).stdout.strip()))
            except Exception:
                pass
            from eval.dubline_eval.mouth import articulation_intervals
            series = [{"t": i / fps, "inner": v, "face_h_px": 100} for i, v in enumerate(ap_out)]
            artic = articulation_intervals(series)
            mos = A.total(A.subtract(artic, dub_speech))
            results["shots"][tag]["models"][model] = {**cost, "sync": {k: v for k, v in sync.items() if k != "tracks"}, **measure, "mouth_motion_on_silence_s": mos,
                                                      "video": str(with_audio)}
            print(tag, model, json.dumps(results["shots"][tag]["models"][model])[:300], flush=True)
        # side-by-side + difference videos when both rendered
        vids = [shot / f"{m}-{tag}.mp4" for m in args.models if (shot / f"{m}-{tag}.mp4").is_file()]
        if len(vids) == 2:
            sh(["ffmpeg", "-y", "-v", "error", "-i", str(source_with_audio), "-i", str(vids[0]), "-i", str(vids[1]),
                "-filter_complex", "[0:v]scale=640:-2,drawtext=text='source':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[a];"
                f"[1:v]scale=640:-2,drawtext=text='{args.models[0]}':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[b];"
                f"[2:v]scale=640:-2,drawtext=text='{args.models[1]}':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[c];[a][b][c]hstack=3[v]",
                "-map", "[v]", "-map", "1:a:0", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", str(shot / f"side-by-side-{tag}.mp4")])
            for m, v in zip(args.models, vids):
                sh(["ffmpeg", "-y", "-v", "error", "-i", str(frames_mp4), "-i", str(v), "-filter_complex",
                    "[0:v][1:v]blend=all_mode=difference,eq=contrast=4:brightness=0.1,scale=960:-2[v]", "-map", "[v]", "-c:v", "libx264", "-crf", "18",
                    str(shot / f"difference-{m}-{tag}.mp4")])
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print("wrote", out / "metrics.json")


if __name__ == "__main__":
    main()
