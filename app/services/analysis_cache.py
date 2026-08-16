from __future__ import annotations

"""Durable, content-addressed cache for whole-programme analysis."""

import hashlib
import json
from pathlib import Path


FINGERPRINT_VERSION = 1
SAMPLE_BYTES = 4 * 1024 * 1024


def media_fingerprint(source: Path, job_folder: Path | None = None) -> str:
    """Identify the media content without hashing a multi-gigabyte film in full.

    Size plus independently sampled beginning/middle/end blocks distinguish cuts
    and edits while keeping local ingest responsive.  The per-job manifest avoids
    reading those blocks again on pause/resume.
    """
    source = source.resolve()
    stat = source.stat()
    manifest = job_folder / "media-fingerprint.json" if job_folder else None
    if manifest and manifest.is_file():
        try:
            cached = json.loads(manifest.read_text(encoding="utf-8"))
            if (cached.get("version") == FINGERPRINT_VERSION
                    and cached.get("source") == str(source)
                    and int(cached.get("size", -1)) == stat.st_size
                    and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns):
                return str(cached["fingerprint"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    digest = hashlib.sha256()
    digest.update(f"dubline-media-v{FINGERPRINT_VERSION}:{stat.st_size}".encode("ascii"))
    offsets = sorted({0, max(0, stat.st_size // 2 - SAMPLE_BYTES // 2),
                      max(0, stat.st_size - SAMPLE_BYTES)})
    with source.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            block = stream.read(min(SAMPLE_BYTES, stat.st_size - offset))
            digest.update(offset.to_bytes(8, "little"))
            digest.update(len(block).to_bytes(8, "little"))
            digest.update(block)
    fingerprint = digest.hexdigest()
    if manifest:
        value = {"version": FINGERPRINT_VERSION, "source": str(source),
                 "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "fingerprint": fingerprint}
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(manifest)
    return fingerprint


def cache_artifact_path(job_folder: Path, fingerprint: str, artifact: str) -> Path:
    root = job_folder.parent.parent / "analysis-cache" / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    return root / artifact


def restore_json_artifact(job_folder: Path, fingerprint: str, artifact: str,
                          destination: Path, *, expected_version: int) -> bool:
    cached = cache_artifact_path(job_folder, fingerprint, artifact)
    if not cached.is_file():
        return False
    try:
        value = json.loads(cached.read_text(encoding="utf-8"))
        if int(value.get("version", -1)) != expected_version:
            return False
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(cached.read_bytes())
    temporary.replace(destination)
    return True


def store_json_artifact(job_folder: Path, fingerprint: str, artifact: str,
                        source: Path, *, expected_version: int) -> bool:
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        if int(value.get("version", -1)) != expected_version:
            return False
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    cached = cache_artifact_path(job_folder, fingerprint, artifact)
    temporary = cached.with_suffix(cached.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(cached)
    return True
