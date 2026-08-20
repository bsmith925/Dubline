from __future__ import annotations

import os
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from app.config import settings
from app.services.subprocess_control import controlled_lines, terminate_process


EMOTION_ORDER = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


class ContextEmotionAnalyzer:
    """Runs Qwen separately so its VRAM can be released before TTS on an 8 GB GPU."""

    def __init__(self, mode: str, preview: bool = False):
        self.mode = mode
        self.analyzer = None
        if mode in {"auto", "text"} and not preview:
            repo = settings.indextts_repo
            model_dir = settings.indextts_model_dir
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from indextts.infer_v2_5 import QwenEmotion
            self.analyzer = QwenEmotion(str(model_dir / "qwen0.6bemo4-merge"))

    def analyze(self, context: str) -> dict:
        if self.mode == "source":
            return {"label": "source performance", "vector": None}
        if self.mode == "neutral":
            return {"label": "calm", "vector": [0, 0, 0, 0, 0, 0, 0, 1]}
        if self.analyzer is not None:
            detected = self.analyzer.inference(context)
            vector = [float(detected.get(name, 0.0)) for name in EMOTION_ORDER]
            label = max(range(len(vector)), key=vector.__getitem__)
            return {"label": EMOTION_ORDER[label], "vector": vector}
        return heuristic_emotion(context)

    def close(self) -> None:
        if self.analyzer is not None:
            try:
                self.analyzer.model.to("cpu")
            except Exception:
                pass
            self.analyzer = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


class IndexTTSEngine:
    """Lazy, reusable IndexTTS 2.5 adapter. One model instance is shared by the GPU worker."""

    _model = None
    _model_lock = threading.Lock()

    def __init__(self, mode: str = "indextts"):
        self.mode = mode

    def _load(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            repo = settings.indextts_repo
            model_dir = settings.indextts_model_dir
            required = ("config.yaml", "gpt.pth", "s2mel.pth", "codec.pth")
            missing = [name for name in required if not (model_dir / name).is_file()]
            if missing:
                raise RuntimeError(f"IndexTTS model files are missing: {', '.join(missing)}")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA PyTorch is not active in this Python environment")
                from indextts.infer_v2_5 import IndexTTS2
            except ImportError as exc:
                raise RuntimeError("The IndexTTS 2.5 runtime is not installed in this Python environment") from exc
            os.environ.setdefault("HF_HUB_CACHE", str(model_dir / "hf_cache"))
            self._model = IndexTTS2(
                cfg_path=str(model_dir / "config.yaml"), model_dir=str(model_dir),
                use_bf16=True, use_cuda_kernel=False, use_deepspeed=False, use_qwen_emo=False,
            )
            return self._model

    def synthesize(self, text: str, reference_audio: Path, output: Path, target_seconds: float,
                   emotion_vector: list[float] | None = None, emotion_strength: float = 0.6,
                   emotion_audio: Path | None = None, language: str = "EN",
                   use_random: bool = False) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "preview":
            self._preview_voice(text, output)
            return
        model = self._load()
        estimated = max(0.7, len(text) / 14.0)
        duration_factor = min(1.8, max(0.55, target_seconds / estimated))
        model.infer(
            spk_audio_prompt=str(reference_audio), text=text, lang=language, output_path=str(output),
            duration_factor=duration_factor, emo_vector=emotion_vector,
            emo_audio_prompt=str(emotion_audio) if emotion_audio and emotion_vector is None else None,
            emo_alpha=emotion_strength,
            use_random=use_random, verbose=False,
        )

    @staticmethod
    def _preview_voice(text: str, output: Path) -> None:
        escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")[:1000]
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            f"flite=text='{escaped}':voice=slt", "-ar", "24000", "-ac", "1", str(output),
        ], check=True, capture_output=True)


def heuristic_emotion(text: str) -> dict:
    value = text.lower()
    scores = {name: 0.0 for name in EMOTION_ORDER}
    patterns = {
        "happy": ("happy", "wonderful", "love", "great", "glad", "yes!", "thank"),
        "angry": ("angry", "hate", "damn", "idiot", "shut up", "how dare", "!"),
        "sad": ("sorry", "died", "death", "cry", "sad", "lost", "goodbye"),
        "afraid": ("afraid", "scared", "run", "hide", "help", "danger", "please!"),
        "disgusted": ("disgust", "gross", "filthy", "sickening"),
        "melancholic": ("remember", "used to", "miss you", "alone", "regret"),
        "surprised": ("what?", "really?", "impossible", "can't believe", "oh!"),
    }
    for emotion, words in patterns.items():
        scores[emotion] = min(1.0, sum(value.count(word) for word in words) * .45)
    if not any(scores.values()):
        scores["calm"] = 1.0
    vector = [scores[name] for name in EMOTION_ORDER]
    return {"label": EMOTION_ORDER[max(range(8), key=vector.__getitem__)], "vector": vector}


def analyze_context_emotions(contexts: list[str], mode: str, preview: bool, folder: Path,
                             progress: Callable[[float, int], None], checkpoint: Callable[[], None]) -> list[dict]:
    manifest = folder / "emotion-manifest.json"
    output = folder / "emotion-results.json"
    manifest.write_text(json.dumps({"contexts": contexts, "mode": mode, "preview": preview},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.emotion_worker", "--manifest", str(manifest),
         "--output", str(output)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip())
            tail = tail[-20:]
            try:
                event = json.loads(line)
                progress(float(event["progress"]), int(event["index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        code = process.wait()
    except BaseException:
        terminate_process(process)
        raise
    if code != 0 or not output.is_file():
        raise RuntimeError("Emotion analysis worker failed: " + "\n".join(tail[-10:]))
    return json.loads(output.read_text(encoding="utf-8"))


def synthesize_voice_lines(manifest_data: dict, folder: Path,
                           progress: Callable[[dict], None], checkpoint: Callable[[], None]) -> None:
    manifest = folder / "tts-manifest.json"
    manifest.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.services.tts_worker", "--manifest", str(manifest)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip())
            tail = tail[-30:]
            try:
                event = json.loads(line)
                if "progress" in event and "index" in event:
                    progress(event)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        code = process.wait()
    except BaseException:
        terminate_process(process)
        raise
    if code != 0:
        raise RuntimeError("IndexTTS worker failed: " + "\n".join(tail[-12:]))


def synthesize_qwen_fallback(items: list[dict], folder: Path,
                             progress: Callable[[dict], None], checkpoint: Callable[[], None]) -> bool:
    runtime = settings.qwen_tts_runtime
    model = settings.qwen_tts_model
    if not items or not runtime.is_file() or not (model / "model.safetensors").is_file():
        return False
    manifest = folder / "qwen-tts-manifest.json"
    results = folder / "qwen-tts-results.json"
    results.unlink(missing_ok=True)
    manifest.write_text(json.dumps({"model": str(model), "items": items, "results": str(results)},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    process = subprocess.Popen(
        [str(runtime), "-m", "app.services.qwen_tts_worker", "--manifest", str(manifest)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1,
    )
    tail = []
    reported: set[int] = set()
    try:
        for line in controlled_lines(process, checkpoint):
            tail.append(line.rstrip()); tail = tail[-20:]
            try:
                event = json.loads(line)
                if "progress" in event:
                    reported.add(int(event["cue_index"])); progress(event)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass
        code = process.wait()
    except BaseException:
        terminate_process(process)
        raise
    if code != 0:
        raise RuntimeError("Qwen3-TTS fallback failed: " + "\n".join(tail[-10:]))
    # Authoritative per-line metrics: anything stdout parsing missed is replayed here.
    if results.is_file():
        for event in json.loads(results.read_text(encoding="utf-8")):
            if int(event.get("cue_index", -1)) not in reported:
                progress(event)
    missing = [item["cue_index"] for item in items if int(item["cue_index"]) not in
               {int(e.get("cue_index", -1)) for e in (json.loads(results.read_text(encoding="utf-8")) if results.is_file() else [])}]
    if missing:
        raise RuntimeError(f"Qwen3-TTS produced no take for {len(missing)} line(s): {missing[:8]}")
    return True
