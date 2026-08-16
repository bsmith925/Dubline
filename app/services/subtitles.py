from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path


TEXT_SUBTITLE_CODECS = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text", "microdvd"}
ENGLISH_CODES = {"en", "eng", "english"}


def audio_streams(probe: dict) -> list[dict]:
    result = []
    ordinal = 0
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        tags = stream.get("tags", {})
        disposition = stream.get("disposition", {})
        result.append({
            "index": int(stream["index"]), "ordinal": ordinal,
            "codec": stream.get("codec_name", "unknown"), "channels": int(stream.get("channels", 0) or 0),
            "layout": stream.get("channel_layout"), "language": str(tags.get("language", "und")).lower(),
            "title": tags.get("title") or tags.get("handler_name") or f"Audio track {ordinal + 1}",
            "default": bool(disposition.get("default")),
        })
        ordinal += 1
    return result


def choose_audio(streams: list[dict], preferred_index: int | None = None) -> tuple[dict, list[str]]:
    if not streams:
        raise RuntimeError("The source has no audio track")
    if preferred_index is not None:
        selected = next((item for item in streams if int(item["index"]) == int(preferred_index)), None)
        if selected is None:
            raise RuntimeError(f"Selected audio stream {preferred_index} does not exist")
        return selected, []
    reject = ("commentary", "audio description", "descriptive", "director", "isolated score",
              "music only", "karaoke")
    scored = []
    for item in streams:
        label = f"{item['title']} {item['language']}".lower()
        score = (3.0 if item["default"] else 0.0) + min(2.0, item["channels"] / 3)
        score -= 12.0 if any(marker in label for marker in reject) else 0.0
        scored.append((score, -item["ordinal"], item))
    scored.sort(reverse=True, key=lambda value: (value[0], value[1]))
    warnings = []
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 1.0:
        warnings.append("multiple plausible programme audio tracks were found; verify the selected track")
    return scored[0][2], warnings


def probe_media(path: Path) -> dict:
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return json.loads(process.stdout)


def media_duration(probe: dict) -> float:
    values = [probe.get("format", {}).get("duration")]
    values += [stream.get("duration") for stream in probe.get("streams", [])]
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            pass
    return max(parsed, default=0.0)


def subtitle_streams(probe: dict) -> list[dict]:
    result = []
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "subtitle":
            continue
        tags = stream.get("tags", {})
        result.append({
            "index": stream["index"],
            "codec": stream.get("codec_name", "unknown"),
            "language": str(tags.get("language", "und")).lower(),
            "title": tags.get("title") or f"Subtitle track {stream['index']}",
            "default": bool(stream.get("disposition", {}).get("default")),
            "forced": bool(stream.get("disposition", {}).get("forced")),
            "hearing_impaired": bool(stream.get("disposition", {}).get("hearing_impaired")),
            "text": stream.get("codec_name") in TEXT_SUBTITLE_CODECS,
        })
    return result


def choose_embedded(streams: list[dict], preferred_index: int | None = None) -> dict | None:
    compatible = [item for item in streams if item["text"]]
    if not compatible:
        return None
    if preferred_index is not None:
        return next((item for item in compatible if int(item["index"]) == int(preferred_index)), None)
    def undesirable(item: dict) -> bool:
        label = str(item.get("title", "")).lower()
        return item.get("forced") or any(marker in label for marker in
            ("forced", "signs", "songs", "commentary", "director", "trivia"))
    full = [item for item in compatible if not undesirable(item)]
    compatible = full or compatible
    return sorted(compatible, key=lambda item: (
        undesirable(item),
        item["language"] not in ENGLISH_CODES,
        item.get("hearing_impaired", False),
        not item["default"],
        item["index"],
    ))[0]


def extract_embedded(video: Path, stream: dict, output: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", f"0:{stream['index']}", str(output)],
                   check=True, capture_output=True)
    return output


def normalize_sidecar(path: Path, output: Path, fps: float = 23.976) -> Path:
    if path.suffix.lower() == ".srt":
        return path
    if path.suffix.lower() == ".sub" and is_microdvd(path):
        cues = parse_microdvd(path, fps)
        write_srt(cues, output)
        return output
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), str(output)], check=True, capture_output=True)
    return output


def is_microdvd(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:256].decode("utf-8", errors="ignore")
    except OSError:
        return False
    return bool(re.search(r"\{\d+\}\{\d+\}", sample))


def parse_microdvd(path: Path, fps: float) -> list[dict]:
    result = []
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = re.match(r"\{(\d+)\}\{(\d+)\}(.*)", raw)
        if not match:
            continue
        text = re.sub(r"\{[^}]+\}", "", match.group(3)).replace("|", " ")
        result.append({"start": int(match.group(1)) / fps, "end": int(match.group(2)) / fps, "text": clean_text(text)})
    return result


def parse_srt(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    pattern = re.compile(
        r"(?:^|\n)(?:\d+\s*\n)?(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})[^\n]*\n(.*?)(?=\n\s*\n|\Z)",
        re.DOTALL,
    )
    cues = []
    for match in pattern.finditer(content):
        start, end = timestamp(match.group(1)), timestamp(match.group(2))
        raw = re.sub(r"<[^>]+>|\{\\[^}]+\}", "", html.unescape(match.group(3)))
        annotations = re.findall(r"\[([^\]]+)\]", raw)
        raw = re.sub(r"\[[^\]]+\]", "", raw).replace("♪", " ")
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
        lines = [line for line in lines if line]
        multi = len(lines) > 1 and sum(bool(re.match(r"^[-–—]", line)) for line in lines) >= 2
        values = lines if multi else [" ".join(lines)]
        for position, value in enumerate(values):
            value = value.strip(" -–—")
            speaker = None
            label = re.match(r"^([A-Z][A-Z0-9 _.'-]{1,28}):\s*(.+)$", value)
            if label:
                speaker, value = label.group(1).title(), label.group(2)
            text = clean_text(value)
            if text:
                cues.append({"start": start, "end": end, "text": text,
                             "subtitle_speaker_hint": speaker,
                             "card_speaker_index": position if multi else 0,
                             "simultaneous_card": multi,
                             "nonverbal_annotations": annotations})
    return cues


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\{\\[^}]+\}", "", value)
    value = html.unescape(value).replace("\n", " ").replace("♪", " ")
    value = re.sub(r"\s+", " ", value).strip(" -–—")
    return value


def timestamp(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def write_srt(cues: list[dict], output: Path) -> None:
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(f"{index}\n{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}\n{cue['text']}\n")
    output.write_text("\n".join(blocks), encoding="utf-8")


def format_timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def looks_english(cues: list[dict], language: str | None = None) -> bool:
    if (language or "").lower() in ENGLISH_CODES:
        return True
    sample = " ".join(item["text"] for item in cues[:80]).lower()
    words = re.findall(r"[a-z']+", sample)
    if not words:
        return False
    common = {"the", "a", "to", "and", "you", "i", "is", "it", "of", "that", "we", "in", "this", "what", "me", "my", "for", "on"}
    ascii_ratio = sum(char.isascii() for char in sample) / max(1, len(sample))
    return ascii_ratio > 0.93 and sum(word in common for word in words) >= max(2, len(words) // 35)
