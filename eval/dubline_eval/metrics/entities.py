from __future__ import annotations

"""Entity metrics: names and terminology as first-class objects.

A wrong name propagates through translation, TTS and back-transcription, and every other
metric agrees with it. These metrics look at the names themselves:

- entity_consistency: spellings per phonetic cluster across the program's transcript
  (1.0 = every mention of a name spelled one way).
- cross_recognizer_agreement: share of primary-ASR name mentions confirmed (same phonetic
  cluster) by a second recognizer's transcript of the same cue.
- translation_entity_preservation: share of source-name clusters whose canonical spelling
  (or an alias) appears in the dub text.
- tts_entity_pronunciation: share of dub-text names found (phonetically) in the take's
  back-transcription.
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher

SENTENCE_STARTERS = {"i", "um", "uh", "oh", "okay", "yeah", "mister", "mr", "mrs", "missus", "miss", "the", "but", "and", "so", "it",
                     "this", "that", "we", "you", "he", "she", "they", "what", "why", "no", "yes", "well", "now", "then", "if", "when",
                     "because", "like", "which", "where", "how", "let's", "it's", "i'm", "we're", "you're", "all", "right", "alright",
                     "hello", "hi", "thank", "thanks", "please", "only", "keep", "boy", "always", "not", "unique", "functions", "again",
                     "global", "speed", "avoid", "mom", "earth", "look", "chicken", "i'll", "lock", "stop", "begin", "understand", "hey",
                     "thing", "your", "are", "get", "two", "one", "there", "here", "for", "with", "from", "have", "has", "been", "just"}


def _phonetic_key(token: str) -> str:
    """Crude language-neutral key: lowercase, collapse vowels/doubles, map common confusions."""
    t = re.sub(r"['’]s$", "", token.lower())
    t = re.sub(r"[^a-zà-ÿ]", "", t)
    t = t.replace("ph", "f").replace("ck", "k").replace("c", "k").replace("q", "k").replace("z", "s").replace("x", "ks")
    t = t.replace("th", "t").replace("y", "i").replace("w", "v")
    t = re.sub(r"[aeiouàáâäèéêëìíîïòóôöùúûü]+", "a", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return t


def candidate_names(text: str, known: set[str] | None = None) -> list[str]:
    """Capitalized tokens that are not sentence-initial (or are known to recur mid-sentence)."""
    out = []
    for m in re.finditer(r"\b[A-ZÀ-Þ][\w'’\-]+\b", text or ""):
        t = m.group(0)
        if len(t) <= 2 or t.lower() in SENTENCE_STARTERS:
            continue
        before = (text[:m.start()]).rstrip()
        initial = not before or before[-1] in ".!?…:"
        if initial and not (known and _phonetic_key(t) in known):
            continue
        out.append(t)
    return out


def mid_sentence_keys(texts) -> set[str]:
    """Phonetic keys of tokens seen capitalized mid-sentence anywhere in the program."""
    return {_phonetic_key(n) for text in texts for n in candidate_names(text)}


def cluster_names(mentions: list[str]) -> dict[str, list[str]]:
    """Group spellings by phonetic key with fuzzy merging of near keys."""
    clusters: dict[str, list[str]] = defaultdict(list)
    for m in mentions:
        key = _phonetic_key(m)
        if not key:
            continue
        match = next((k for k in clusters if k == key or SequenceMatcher(None, k, key).ratio() >= 0.8), None)
        clusters[match or key].append(m)
    return dict(clusters)


def consistency(clusters: dict[str, list[str]]) -> dict:
    rows = []
    for key, spellings in clusters.items():
        if len(spellings) < 2:
            continue
        # possessives are the same name, not a second spelling
        spellings = [re.sub(r"['’]s$", "", s) for s in spellings]
        forms = sorted(set(spellings), key=lambda s: -spellings.count(s))
        rows.append({"key": key, "mentions": len(spellings), "spellings": forms, "majority_share": round(spellings.count(forms[0]) / len(spellings), 3)})
    multi = [r for r in rows if len(r["spellings"]) > 1]
    return {"clusters_recurring": len(rows), "clusters_inconsistent": len(multi),
            "entity_consistency": round(1 - len(multi) / len(rows), 3) if rows else None, "inconsistent": multi[:20]}


def cross_agreement(primary_by_cue: dict[int, str], second_by_cue: dict[int, str]) -> dict:
    total = agreed = 0; misses = []
    known = mid_sentence_keys(primary_by_cue.values())
    for cue_id, text in primary_by_cue.items():
        names = candidate_names(text, known)
        if not names or cue_id not in second_by_cue:
            continue
        second_keys = {_phonetic_key(n) for n in candidate_names(second_by_cue[cue_id])} | {_phonetic_key(w) for w in re.findall(r"\w+", second_by_cue[cue_id])}
        for n in names:
            total += 1
            k = _phonetic_key(n)
            if any(k == s or SequenceMatcher(None, k, s).ratio() >= 0.8 for s in second_keys):
                agreed += 1
            else:
                misses.append((cue_id, n))
    return {"mentions": total, "agreed": agreed, "cross_recognizer_agreement": round(agreed / total, 3) if total else None, "misses": misses[:20]}


def preservation(source_by_cue: dict[int, str], dub_by_cue: dict[int, str]) -> dict:
    """Share of source names (per cue) whose phonetic key appears in that cue's dub text."""
    total = kept = 0; lost = []
    known = mid_sentence_keys(source_by_cue.values())
    for cue_id, text in source_by_cue.items():
        names = candidate_names(text, known)
        if not names:
            continue
        dub_keys = {_phonetic_key(w) for w in re.findall(r"[\w'’\-]+", dub_by_cue.get(cue_id, ""))}
        for n in names:
            total += 1
            k = _phonetic_key(n)
            if any(k == d or SequenceMatcher(None, k, d).ratio() >= 0.8 for d in dub_keys):
                kept += 1
            else:
                lost.append((cue_id, n))
    return {"mentions": total, "kept": kept, "translation_entity_preservation": round(kept / total, 3) if total else None, "lost": lost[:20]}
