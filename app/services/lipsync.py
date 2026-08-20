from __future__ import annotations

"""Optional, confidence-gated MuseTalk finishing for a few clean visible shots."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from app.config import settings
from app.services.subprocess_control import controlled_lines, terminate_process


def _enabled() -> bool:
    return settings.musetalk_enabled


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

    MuseTalk is intentionally not applied across a whole film.  A cue qualifies only
    when exactly one sizeable face is visible, the active-speaker evidence is strong,
    and audio-only timing QC still found a visible mismatch.
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
    repo = settings.musetalk_repo
    runtime = settings.musetalk_runtime
    model_dir = settings.musetalk_model_dir
    required = [runtime, model_dir / "unet.pth", model_dir / "musetalk.json",
                repo / "models/whisper/pytorch_model.bin"]
    if not all(path.is_file() for path in required):
        return None, {"enabled": True, "ready": False, "selected": 0, "completed": 0}
    candidates = []
    for index, cue in enumerate(cues):
        visual = cue.get("visual_speaker", {})
        qc = cue.get("qc", {})
        visible_failure = (abs(float(qc.get("stretch_percent") or 0.0)) > 5.0
                           or float(qc.get("energy_contour_similarity", 1.0)) < .15)
        duration = float(cue.get("end", 0)) - float(cue.get("start", 0))
        if (visible_failure and cue.get("mouth_visible") and visual.get("visible_faces") == 1
                and float(visual.get("active_speaker_confidence", 0)) >= .82
                and float(visual.get("face_area_ratio", 0)) >= .004
                and .45 <= duration <= 8.0):
            candidates.append((index, cue))
    candidates = candidates[:settings.musetalk_max_shots]
    if not candidates:
        return None, {"enabled": True, "ready": True, "selected": 0, "completed": 0}

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
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                        "-i", str(source), "-an", "-vf", "fps=25", "-c:v", "libx264", "-crf", "16", str(clip)],
                       check=True)
        line = dialogue_dir / f"{index + 1:06d}.wav"
        duration = max(.1, end - start)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(line), "-filter_complex",
                        f"adelay={round(padding * 1000)}:all=1,apad=whole_dur={duration:.3f}",
                        "-t", f"{duration:.3f}", "-ar", "16000", "-ac", "1", str(audio)], check=True)
        result_name = f"cue-{index + 1:06d}.mp4"
        config = work / f"cue-{index + 1:06d}.json"
        config.write_text(json.dumps({"task": {"video_path": str(clip), "audio_path": str(audio),
                                                "result_name": result_name}}, indent=2), encoding="utf-8")
        command = [str(runtime), "-m", "scripts.inference", "--inference_config", str(config),
                   "--result_dir", str(results), "--unet_model_path", str(model_dir / "unet.pth"),
                   "--unet_config", str(model_dir / "musetalk.json"), "--whisper_dir",
                   str(repo / "models/whisper"), "--version", "v15", "--use_float16",
                   "--batch_size", "4", "--ffmpeg_path", str(Path(shutil.which("ffmpeg") or "ffmpeg").parent)]
        ok, tail = _run(command, repo, checkpoint)
        generated = results / "v15" / result_name
        if ok and generated.is_file():
            completed.append((generated, start, end, index))
            cue.setdefault("qc", {})["visual_lipsync"] = "MuseTalk 1.5"
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
    return output, {"enabled": True, "ready": True, "selected": len(candidates),
                    "completed": len(completed), "cue_ids": [index + 1 for *_, index in completed]}
