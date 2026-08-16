from __future__ import annotations

"""Two-pass scene translation and cue adaptation using local Hy-MT2."""

import argparse
import difflib
import json
import math
import os
import re
from pathlib import Path


def predicted_seconds(text: str) -> float:
    words = len(re.findall(r"[\w']+", text))
    return max(0.35, words / 2.65 + text.count(",") * 0.08 + text.count(".") * 0.1)


def hard_line(cue: dict) -> bool:
    if cue.get("_skip_adaptation"):
        return False
    target = max(0.3, float(cue["end"]) - float(cue["start"]))
    ratio = predicted_seconds(cue.get("faithful_translation") or cue.get("literal_translation")
                              or cue.get("english", "")) / target
    return bool(cue.get("force_adaptation")) or ratio > 1.06 or not cue.get("translation_was_supplied", True)


def json_value(content: str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        array = re.search(r"\[[\s\S]*\]", content)
        if array:
            return json.loads(array.group(0))
        obj = re.search(r"\{[\s\S]*\}", content)
        return json.loads(obj.group(0)) if obj else None


def parse_candidates(content: str) -> list[str]:
    value = json_value(content) or {}
    names = ("natural", "shorter", "lip_compatible", "rhythmic", "literal_short", "idiomatic_short")
    return list(dict.fromkeys(" ".join(value[name].split()) for name in names
            if isinstance(value.get(name), str) and value[name].strip()
            and value[name].strip().lower() not in {"true", "false", "null"}))


def protected_names(text: str) -> set[str]:
    common = {"The", "A", "An", "In", "On", "At", "To", "From", "He", "She", "It",
              "We", "They", "I", "You", "This", "That", "What", "How", "Why", "Ready"}
    return {name for name in re.findall(r"\b[A-Z][A-Za-z'-]+\b", text) if name not in common}


def syllable_count(text: str) -> int:
    groups = re.findall(r"[aeiouy]+", text.lower())
    return max(1, len(groups))


def rhythm_score(text: str, cue: dict | None = None) -> float:
    cue = cue or {}
    punctuation_pauses = len(re.findall(r"[,;:—…]", text)) + text.count("...")
    expected = round(float(cue.get("source_performance", {}).get("pause_ratio", .15)) * 4)
    return max(0.0, 1.0 - abs(punctuation_pauses - expected) / 4)


def viseme_score(text: str, cue: dict | None = None) -> float:
    cue = cue or {}
    source_units = len(cue.get("words") or [])
    if not source_units:
        source_units = max(1, len(re.findall(r"\w+", cue.get("source", ""))))
    expected = max(1.0, source_units * .72)
    return max(0.0, 1.0 - abs(syllable_count(text) - expected) / max(expected, 3.0))


def choose_candidate(literal: str, candidates: list[str], target: float,
                     translation_is_target: bool, semantic: list[dict] | None = None,
                     cue: dict | None = None) -> tuple[str, float]:
    required_names = protected_names(literal)
    cue = cue or {}
    all_candidates = list(dict.fromkeys([literal, *candidates]))
    semantic = semantic or []
    judged = {int(item.get("index", -1)): item for item in semantic if isinstance(item, dict)}
    best, best_score, best_quality = literal, float("inf"), 0.0
    for index, candidate in enumerate(all_candidates):
        lexical = difflib.SequenceMatcher(None, literal.lower(), candidate.lower()).ratio()
        evidence = judged.get(index, {})
        adequacy = float(evidence.get("adequacy", lexical if translation_is_target else .72))
        terminology = float(evidence.get("terminology", 1.0))
        register = float(evidence.get("register", .8))
        missing_names = len([name for name in required_names if name.lower() not in candidate.lower()])
        duration_error = min(1.5, abs(predicted_seconds(candidate) - target) / target)
        rhythm_loss = 1 - rhythm_score(candidate, cue)
        viseme_loss = 1 - viseme_score(candidate, cue)
        mouth_visible = bool(cue.get("mouth_visible"))
        if mouth_visible:
            score = ((1 - adequacy) * .35 + duration_error * .22 + viseme_loss * .18
                     + (1 - terminology) * .12 + (1 - register) * .07 + rhythm_loss * .06)
        else:
            score = ((1 - adequacy) * .48 + duration_error * .24 + (1 - terminology) * .15
                     + (1 - register) * .08 + rhythm_loss * .05)
        score += missing_names * 1.5
        acceptable = adequacy >= .62 and terminology >= .65 and missing_names == 0
        if acceptable and score < best_score:
            best, best_score, best_quality = candidate, score, adequacy
    if not math.isfinite(best_score):
        best, best_score, best_quality = literal, .35, .75
    confidence = max(0.0, min(1.0, .55 * (1 - best_score) + .45 * best_quality))
    if translation_is_target and best == literal:
        confidence = max(confidence, 0.75)
    return best, confidence


def judge_candidates(llm, cue: dict, faithful: str, candidates: list[str]) -> list[dict]:
    values = list(dict.fromkeys([faithful, *candidates]))
    numbered = "\n".join(f"{i}: {value}" for i, value in enumerate(values))
    prompt = f"""Judge English dubbing candidates against the source and faithful translation.
Return only a JSON array with one object per candidate:
{{"index":0,"adequacy":0.0,"terminology":0.0,"register":0.0}}
Scores are 0 to 1. Adequacy means all meaning and facts are preserved. Penalize additions,
omissions, wrong names and changed intent. Terminology covers names and scene terms. Register
covers tone and relationship. Do not prefer brevity by itself.
Source: {cue.get('source', '')}
Faithful English: {faithful}
Candidates:
{numbered}"""
    value = json_value(ask(llm, prompt, 420))
    return value if isinstance(value, list) else []


def ask(llm, prompt: str, max_tokens: int = 900) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], temperature=0.05, top_p=0.9,
        max_tokens=max_tokens,
    )
    return str(response["choices"][0]["message"]["content"])


def scene_batches(cues: list[dict], size: int = 12) -> list[list[int]]:
    return [list(range(start, min(len(cues), start + size))) for start in range(0, len(cues), size)]


def faithful_pass(llm, cues: list[dict]) -> None:
    for batch in scene_batches(cues):
        untranslated = [index for index in batch if not cues[index].get("translation_is_target", True)]
        for index in batch:
            cue = cues[index]
            cue["translation_was_supplied"] = bool(cue.get("translation_is_target", True))
            if cue["translation_was_supplied"]:
                cue["faithful_translation"] = cue.get("literal_translation") or cue.get("english", "")
        if not untranslated:
            continue
        numbered = "\n".join(f"{index}: {cues[index].get('source', '')}" for index in batch)
        wanted = ", ".join(str(index) for index in untranslated)
        prompt = f"""Translate this complete film scene faithfully into idiomatic English.
Maintain names, terminology, facts, register, jokes, relationships and continuity across lines.
Do not shorten for timing and never romanize instead of translating.
Return only a JSON object mapping these requested line numbers to English strings: {wanted}
Scene:
{numbered}"""
        value = json_value(ask(llm, prompt)) or {}
        for index in untranslated:
            translated = value.get(str(index)) if isinstance(value, dict) else None
            if not translated:
                translated = ask(llm, f"Translate to natural English only:\n{cues[index].get('source', '')}", 180)
            translated = " ".join(str(translated).strip().strip('"').split())
            cues[index]["faithful_translation"] = translated
            cues[index]["literal_translation"] = translated
            cues[index]["english"] = translated
            cues[index]["translation_is_target"] = True
            cues[index]["translation_model"] = "Hy-MT2-7B Q4 · scene-faithful pass"


def adaptation_pass(llm, cues: list[dict], output: Path) -> None:
    difficult = [index for index, cue in enumerate(cues) if hard_line(cue)]
    for position, index in enumerate(difficult):
        cue = cues[index]
        faithful = cue.get("faithful_translation") or cue.get("literal_translation") or cue.get("english", "")
        target = max(0.3, float(cue["end"]) - float(cue["start"]))
        context = cues[max(0, index - 5):min(len(cues), index + 6)]
        context_text = "\n".join(
            f"{offset + max(0, index - 5)}: {item.get('faithful_translation') or item.get('english', '')}"
            for offset, item in enumerate(context)
        )
        previous_stretch = float(cue.get("qc", {}).get("stretch_percent") or 0.0)
        timing_limit = 5.0 if cue.get("mouth_visible") else 8.0
        urgency = (f"A previous spoken take was {previous_stretch:.1f}% too long; the shorter choice must be "
                   "materially shorter. " if previous_stretch > timing_limit else "")
        prompt = f"""Adapt one already-translated film line for dubbing. Do not translate it again.
Return strict JSON with six distinct versions:
{{"natural":"...","shorter":"...","lip_compatible":"...","rhythmic":"...","literal_short":"...","idiomatic_short":"..."}}
All values must be idiomatic English, never transliteration. Preserve meaning, names, facts and register.
{urgency}Target spoken duration: {target:.2f} seconds, roughly {max(1, round(target * 2.65))} words.
Mouth clearly visible: {bool(cue.get('mouth_visible'))}. Match vowel/syllable rhythm more closely when true.
Faithful English translation: {faithful}
Scene context:
{context_text}"""
        candidates = parse_candidates(ask(llm, prompt, 240))
        semantic = judge_candidates(llm, cue, faithful, candidates)
        selected, confidence = choose_candidate(faithful, candidates, target, True,
                                                 semantic=semantic, cue=cue)
        if previous_stretch > timing_limit:
            required_names = protected_names(faithful)
            judged = {int(item.get("index", -1)): item for item in semantic if isinstance(item, dict)}
            shorter = []
            for candidate_index, candidate in enumerate(candidates, 1):
                evidence = judged.get(candidate_index, {})
                adequacy = float(evidence.get("adequacy", 0.0))
                terminology = float(evidence.get("terminology", 0.0))
                if (predicted_seconds(candidate) < predicted_seconds(faithful) * 0.88
                        and adequacy >= .65 and terminology >= .65
                        and all(name.lower() in candidate.lower() for name in required_names)):
                    duration_fit = 1 - min(1.0, abs(predicted_seconds(candidate) - target) / target)
                    shorter.append((.65 * adequacy + .2 * terminology + .15 * duration_fit, candidate))
            if shorter:
                selected = max(shorter)[1]
                confidence = max(confidence, 0.7)
        cue["translation_candidates"] = candidates
        cue["candidate_semantic_scores"] = semantic
        cue["adaptation_scoring"] = "semantic + terminology + duration + rhythm + viseme"
        cue["adapted_dialogue"] = selected; cue["english"] = selected
        cue["adaptation_confidence"] = round(confidence, 3)
        cue["adaptation_attempts"] = 1 + len(candidates)
        cue["adaptation_model"] = "Hy-MT2-7B Q4 · duration pass"
        cue["force_adaptation"] = False
        output.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": (position + 1) / max(1, len(difficult)), "index": index,
                          "difficult": len(difficult)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8")); cues = spec["cues"]
    from llama_cpp import Llama
    gpu_layers = int(os.getenv("DUB_LLAMA_GPU_LAYERS", "-1"))
    llm = Llama(model_path=spec["model"], n_ctx=8192, n_batch=512, n_threads=10,
                n_threads_batch=12, n_gpu_layers=gpu_layers, verbose=False)
    faithful_pass(llm, cues)
    adaptation_pass(llm, cues, args.output)
    for cue in cues:
        cue.setdefault("faithful_translation", cue.get("literal_translation") or cue.get("english", ""))
        cue.setdefault("adapted_dialogue", cue.get("english", ""))
        cue.setdefault("adaptation_confidence", 0.9); cue.setdefault("adaptation_attempts", 1)
    args.output.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"progress": 1.0, "complete": True}), flush=True)


if __name__ == "__main__":
    main()
