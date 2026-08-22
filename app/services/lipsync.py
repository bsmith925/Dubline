from __future__ import annotations

"""Confidence-gated MuseTalk lip-sync for every utterance with a clean, visible mouth."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from app.config import settings
from app.services.subprocess_control import controlled_lines, terminate_process


def _enabled() -> bool:
    return settings.musetalk_enabled


def _video_fps(path: Path) -> float | None:
    try:
        text = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                               "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
        num, den = text.split("/")
        return float(num) / float(den)
    except Exception:
        return None


def _run(command: list[str], cwd: Path, checkpoint: Callable[[], None]) -> tuple[bool, str]:
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-20:]
        return process.wait() == 0, "\n".join(tail)
    except BaseException:
        if process.poll() is None:
            terminate_process(process)
        raise


def apply_selective_lipsync(source: Path, cues: list[dict], dialogue_dir: Path, folder: Path,
                            progress: Callable[[float, int], None],
                            checkpoint: Callable[[], None]) -> tuple[Path | None, dict]:
    """Return a video-only override, or None when no shot safely qualifies.

    An utterance qualifies when exactly one sizeable face is visible, its mouth is
    visible and the active-speaker evidence is strong.  Shots that fail the gate
    keep the original picture; a partially lip-synced film beats an artefacted one.
    """
    if not _enabled():
        return None, {"enabled": False, "selected": 0, "completed": 0}
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(source)],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout)
    videos = [item for item in probe.get("streams", []) if item.get("codec_type") == "video"]
    primary = videos[0] if videos else {}
    unsafe = []
    if len(videos) != 1:
        unsafe.append("multiple video streams must be preserved")
    if any(marker in str(primary.get("pix_fmt", "")) for marker in ("10", "12")):
        unsafe.append("10/12-bit picture may contain HDR")
    if str(primary.get("color_transfer", "")).lower() in {"smpte2084", "arib-std-b67"}:
        unsafe.append("HDR transfer metadata must be preserved")
    side_data = json.dumps(primary.get("side_data_list", [])).lower()
    if "dovi" in side_data or "dolby vision" in side_data:
        unsafe.append("Dolby Vision metadata must be preserved")
    if unsafe:
        return None, {"enabled": True, "ready": True, "selected": 0, "completed": 0,
                      "refused": True, "reason": "; ".join(unsafe)}
    engine = settings.lipsync_engine
    repo = settings.musetalk_repo
    runtime = settings.musetalk_runtime
    model_dir = settings.musetalk_model_dir
    if engine == "latentsync":
        repo, runtime = settings.latentsync_repo, settings.latentsync_runtime
        required = [runtime, repo / "checkpoints/latentsync_unet.pt", repo / "configs/unet/stage2_512.yaml"]
    else:
        required = [runtime, model_dir / "unet.pth", model_dir / "musetalk.json",
                    repo / "models/whisper/pytorch_model.bin"]
    if not all(path.is_file() for path in required):
        return None, {"enabled": True, "ready": False, "engine": engine, "selected": 0, "completed": 0}
    candidates = []
    skipped: dict[str, int] = {}
    for index, cue in enumerate(cues):
        visual = cue.get("visual_speaker") or {}
        duration = float(cue.get("end", 0)) - float(cue.get("start", 0))
        reason = None
        if cue.get("nonverbal_filler") or cue.get("overlapping_speech"):
            reason = "source performance kept"
        elif not cue.get("mouth_visible"):
            reason = "mouth not visible"
        elif visual.get("visible_faces") != 1:
            reason = "not exactly one face"
        elif float(visual.get("active_speaker_confidence") or 0) < .82:
            reason = "uncertain active speaker"
        elif float(visual.get("face_area_ratio") or 0) < .004:
            reason = "face too small"
        elif not .45 <= duration <= 14.0:
            reason = "duration out of range"
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            cue.setdefault("qc", {})["visual_lipsync_skipped"] = reason
            continue
        candidates.append((index, cue))
    if settings.musetalk_max_shots:
        candidates = candidates[:settings.musetalk_max_shots]
    if not candidates:
        return None, {"enabled": True, "ready": True, "selected": 0, "completed": 0, "skipped": skipped}

    work = folder / "musetalk-finishing"; work.mkdir(exist_ok=True)
    results = work / "results"; results.mkdir(exist_ok=True)
    completed: list[tuple[Path, float, float, int]] = []
    for position, (index, cue) in enumerate(candidates):
        checkpoint()
        padding = .12
        start = max(0.0, float(cue["start"]) - padding)
        end = float(cue["end"]) + padding
        clip = work / f"cue-{index + 1:06d}-source.mp4"
        audio = work / f"cue-{index + 1:06d}-audio.wav"
        # Keep the first/last 120 ms silent so the regenerated mouth blends at shot boundaries.
        # Native frame rate: MuseTalk scales its audio window by 50/fps, and keeping the
        # source cadence lets the edited shot composite back frame-accurately.
        # Seek accurately and cut by duration (input "-to" near the start of a file
        # produced clips that began at t=0; measured 1.27-1.32x over-length on first shots).
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-accurate_seek", "-ss", f"{start:.3f}", "-i", str(source),
                        "-t", f"{max(0.04, end - start):.3f}", "-an", "-c:v", "libx264", "-crf", "14",
                        "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", str(clip)],
                       check=True)
        line = dialogue_dir / f"{index + 1:06d}.wav"
        duration = max(.1, end - start)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(line), "-filter_complex",
                        f"adelay={round(padding * 1000)}:all=1,apad=whole_dur={duration:.3f}",
                        "-t", f"{duration:.3f}", "-ar", "16000", "-ac", "1", str(audio)], check=True)
        result_name = f"cue-{index + 1:06d}.mp4"
        config = work / f"cue-{index + 1:06d}.json"
        config.write_text(json.dumps({"task": {"video_path": str(clip), "audio_path": str(audio),
                                                "result_name": result_name, "engine": engine}}, indent=2), encoding="utf-8")
        if engine == "latentsync":
            rendered = results / "latentsync" / result_name
            rendered.parent.mkdir(parents=True, exist_ok=True)
            source_fps = _video_fps(clip)
            model_input = clip
            if settings.lipsync_pre_resample_25 and source_fps and abs(source_fps - 25.0) > 0.01:
                # EXP-VIDEO-003: LatentSync converts its input with `ffmpeg -r 25`, which we
                # measured to lag the picture by ~60 ms (47-73 ms) on 30 fps sources; the
                # `fps=25` filter is exact (±13 ms) and `-r 25` on a 25 fps input is identity.
                model_input = clip.with_name(clip.stem + "-25fps.mp4")
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-an", "-vf", "fps=25",
                                "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(model_input)], check=True)
            command = [str(runtime), "-m", "scripts.inference", "--unet_config_path", "configs/unet/stage2_512.yaml",
                       "--inference_ckpt_path", "checkpoints/latentsync_unet.pt", "--inference_steps", "20",
                       "--guidance_scale", "1.5", "--seed", "1247", "--video_path", str(model_input), "--audio_path", str(audio),
                       "--video_out_path", str(rendered)]
            ok, tail = _run(command, repo, checkpoint)
            generated = rendered
            if ok and generated.is_file():
                # LatentSync renders at 25 fps; bring the shot back to the source cadence so
                # the composite stays frame-accurate (the overlay enables by time).
                if source_fps and abs(source_fps - 25.0) > 0.01:
                    resampled = rendered.with_name(rendered.stem + f"-{source_fps:g}fps.mp4")
                    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(rendered), "-an", "-vf",
                                    f"minterpolate=fps={source_fps:g}:mi_mode=blend", "-c:v", "libx264", "-crf", "14",
                                    "-pix_fmt", "yuv420p", str(resampled)], check=True)
                    generated = resampled
            label = "LatentSync 1.6"
        else:
            command = [str(runtime), "-m", "scripts.inference", "--inference_config", str(config),
                       "--result_dir", str(results), "--unet_model_path", str(model_dir / "unet.pth"),
                       "--unet_config", str(model_dir / "musetalk.json"), "--whisper_dir",
                       str(repo / "models/whisper"), "--version", "v15", "--use_float16",
                       "--batch_size", "16", "--extra_margin", "10", "--parsing_mode", "jaw",
                       "--ffmpeg_path", str(Path(shutil.which("ffmpeg") or "ffmpeg").parent)]
            ok, tail = _run(command, repo, checkpoint)
            generated = results / "v15" / result_name
            label = "MuseTalk 1.5"
        if ok and generated.is_file():
            completed.append((generated, start, end, index))
            cue.setdefault("qc", {})["visual_lipsync"] = label
        else:
            cue.setdefault("qc", {})["visual_lipsync_error"] = tail[-600:]
        progress((position + 1) / len(candidates), index)
    if not completed:
        return None, {"enabled": True, "ready": True, "selected": len(candidates), "completed": 0}

    output = work / "video-override.mkv"
    inputs = ["-i", str(source)]
    graph = ["[0:v:0]setpts=PTS-STARTPTS[base0]"]
    current = "base0"
    for number, (clip, start, end, _) in enumerate(completed, 1):
        inputs += ["-i", str(clip)]
        shifted = f"clip{number}"; merged = f"base{number}"
        graph.append(f"[{number}:v:0]setpts=PTS-STARTPTS+{start:.6f}/TB[{shifted}]")
        graph.append(f"[{current}][{shifted}]overlay=eof_action=pass:shortest=0:enable='between(t,{start:.6f},{end:.6f})'[{merged}]")
        current = merged
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph),
                    "-map", f"[{current}]", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                    "-map_metadata", "0", str(output)], check=True)
    return output, {"enabled": True, "ready": True, "engine": engine, "selected": len(candidates),
                    "completed": len(completed), "skipped": skipped,
                    "cue_ids": [index + 1 for *_, index in completed]}
