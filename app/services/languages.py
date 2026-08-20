from __future__ import annotations

"""Target-language registry: codes, container tags and which TTS engine can speak it.

IndexTTS-2.5 speaks Chinese, English, Japanese, Spanish and Arabic; Qwen3-TTS
covers ten languages including French, German, Italian, Portuguese, Korean and
Russian.  When the requested target is outside IndexTTS's set, Qwen3-TTS takes
the primary role instead of acting only as the per-line fallback.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    name: str
    iso1: str            # ISO 639-1, used for Whisper / ASR
    iso2: str            # ISO 639-2/B, used for Matroska track tags
    indextts: str | None  # IndexTTS-2.5 language code, None when unsupported
    qwen_tts: bool        # Qwen3-TTS support


LANGUAGES: dict[str, Language] = {lang.name: lang for lang in (
    Language("English", "en", "eng", "EN", True),
    Language("Chinese", "zh", "zho", "ZH", True),
    Language("Japanese", "ja", "jpn", "JA", True),
    Language("Spanish", "es", "spa", "ES", True),
    Language("Arabic", "ar", "ara", "AR", False),
    Language("French", "fr", "fra", None, True),
    Language("German", "de", "deu", None, True),
    Language("Italian", "it", "ita", None, True),
    Language("Portuguese", "pt", "por", None, True),
    Language("Korean", "ko", "kor", None, True),
    Language("Russian", "ru", "rus", None, True),
)}
SUPPORTED_TARGETS = [name for name, lang in LANGUAGES.items() if lang.indextts or lang.qwen_tts]


def normalize_language(value: str | None, default: str = "English") -> str:
    """Accept a name or ISO code in any case and return the canonical name."""
    text = str(value or "").strip()
    if not text:
        return default
    for name, lang in LANGUAGES.items():
        if text.lower() in {name.lower(), lang.iso1, lang.iso2}:
            return name
    return text.title()


def language(name: str) -> Language:
    return LANGUAGES.get(normalize_language(name), Language(str(name), "", "und", None, False))


def iso1(name: str) -> str | None:
    return language(name).iso1 or None


def iso2(name: str) -> str:
    return language(name).iso2 or "und"


def primary_engine(target: str, requested: str = "indextts") -> str:
    """Which voice engine synthesizes the main takes for ``target``."""
    lang = language(target)
    if requested == "qwen-tts" and lang.qwen_tts:
        return "qwen-tts"
    if lang.indextts:
        return "indextts"
    if lang.qwen_tts:
        return "qwen-tts"
    raise ValueError(f"No local speech engine can synthesize {target}")
