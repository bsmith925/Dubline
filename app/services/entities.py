from __future__ import annotations

"""Title lexicon v0: names and terminology as a program-level object (docs/entity-track.md).

The ASR transcript is not the canonical script. Proper nouns are discovered over the whole
program, clustered by a crude phonetic key, and each cluster gets one canonical spelling
(weighted consensus: frequency, then glossary/metadata evidence). Cue text is rewritten to
the canonical form before translation, and the lexicon is written as a job artifact so
translation (protected names), TTS (pronunciation) and QC can share it.
"""

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

STOP = {"i", "um", "uh", "oh", "okay", "yeah", "mister", "mr", "mrs", "missus", "miss", "the", "but", "and", "so", "it", "this", "that",
        "we", "you", "he", "she", "they", "what", "why", "no", "yes", "well", "now", "then", "if", "when", "because", "like", "which",
        "where", "how", "let's", "it's", "i'm", "we're", "you're", "all", "right", "alright", "hello", "hi", "thank", "thanks", "please"}


def phonetic_key(token: str) -> str:
    t = re.sub(r"['’]s$", "", token.lower())
    t = re.sub(r"[^a-zà-ÿ]", "", t)
    t = t.replace("ph", "f").replace("ck", "k").replace("c", "k").replace("q", "k").replace("z", "s").replace("x", "ks")
    t = t.replace("th", "t").replace("y", "i").replace("w", "v")
    t = re.sub(r"[aeiouàáâäèéêëìíîïòóôöùúûü]+", "a", t)
    return re.sub(r"(.)\1+", r"\1", t)


def _mentions(text: str, known: set[str] | None = None) -> list[tuple[str, int, int]]:
    out = []
    for m in re.finditer(r"\b[A-ZÀ-Þ][\w'’\-]+\b", text or ""):
        t = m.group(0)
        if len(t) <= 2 or t.lower() in STOP:
            continue
        before = text[:m.start()].rstrip()
        initial = not before or before[-1] in ".!?…:"
        if initial and not (known and phonetic_key(t) in known):
            continue
        out.append((t, m.start(), m.end()))
    return out


def build_lexicon(cues: list[dict], glossary: dict[str, str] | None = None, metadata_text: str = "") -> dict:
    texts = [str(c.get("source") or "") for c in cues if not c.get("nonverbal_filler")]
    known = {phonetic_key(t) for text in texts for t, _, _ in _mentions(text)}
    clusters: dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        for t, _, _ in _mentions(text, known):
            key = phonetic_key(t)
            match = next((k for k in clusters if k == key or SequenceMatcher(None, k, key).ratio() >= 0.8), None)
            clusters[match or key][re.sub(r"['’]s$", "", t)] += 1
    evidence_terms = [w for w in re.findall(r"\b[A-ZÀ-Þ][\w'’\-]+\b", metadata_text or "")] + list((glossary or {}).values())
    entries = []
    for key, forms in clusters.items():
        if sum(forms.values()) < 1:
            continue
        # consensus: external evidence (glossary value / metadata) wins, then frequency, then length
        ext = next((e for e in evidence_terms if phonetic_key(e) == key or SequenceMatcher(None, phonetic_key(e), key).ratio() >= 0.8), None)
        canonical = ext or max(forms.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        entries.append({"canonical": canonical, "aliases": sorted(f for f in forms if f != canonical), "mentions": sum(forms.values()),
                        "confidence": round(forms.get(canonical, 0) / sum(forms.values()), 3) if not ext else 0.95,
                        "evidence": (["glossary/metadata"] if ext else []) + ["transcript consensus"], "key": key})
    return {"version": 0, "entries": sorted(entries, key=lambda e: -e["mentions"])}


def canonicalize_cues(cues: list[dict], lexicon: dict) -> int:
    """Rewrite alias spellings in cue['source'] to the canonical form. Returns the number of rewrites."""
    alias_map = {}
    for e in lexicon["entries"]:
        for a in e["aliases"]:
            alias_map[a] = e["canonical"]
    if not alias_map:
        return 0
    pattern = re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(alias_map, key=len, reverse=True)) + r")(['’]s)?\b")
    rewrites = 0
    for cue in cues:
        text = str(cue.get("source") or "")
        new, n = pattern.subn(lambda m: alias_map[m.group(1)] + (m.group(2) or ""), text)
        if n:
            cue["source_asr"] = text
            cue["source"] = new
            rewrites += n
    return rewrites


def write_lexicon(lexicon: dict, path: Path) -> None:
    path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=1), encoding="utf-8")
