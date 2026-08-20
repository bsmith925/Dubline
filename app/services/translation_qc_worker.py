from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings
from app.services.llm_json import StructuredOutputError, array_of, ask_json, number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    cues = spec["cues"]
    target = str(spec.get("target_language") or "English").upper()
    from llama_cpp import Llama
    gpu_layers = settings.dub_llama_gpu_layers
    llm = Llama(model_path=spec["model"], n_ctx=8192, n_batch=512, n_threads=10,
                n_threads_batch=12, n_gpu_layers=gpu_layers, verbose=False)
    results = []
    # Keep every prompt well below the 8k context even when subtitle cards contain
    # long SDH/translator notes.  Fixed ten-line batches could silently truncate
    # precisely the difficult passages this independent check exists to catch.
    batches = []
    current, characters = [], 0
    for cue in cues:
        size = sum(len(str(cue.get(key, ""))) for key in
                   ("source", "faithful_translation", "literal_translation", "english"))
        if current and (len(current) >= 6 or characters + size > 14_000):
            batches.append(current); current, characters = [], 0
        current.append(cue); characters += size
    if current:
        batches.append(current)
    completed = 0
    for batch in batches:
        lines = "\n".join(
            f"ID {cue['id']}\nSOURCE ({cue.get('source_language') or 'auto'}): {cue.get('source','')}\n"
            f"FAITHFUL {target}: {cue.get('faithful_translation') or cue.get('literal_translation','')}\n"
            f"DUB {target}: {cue.get('english','')}"
            for cue in batch
        )
        prompt = f"""You are an independent bilingual film-translation quality checker.
The translations were created by a different model. Judge SOURCE directly against DUB {target};
use FAITHFUL {target} only as secondary evidence. Detect changed facts, polarity, names, relationships,
omissions, additions, mistranslated idioms and register changes. Do not reward fluency alone.
Give one verdict per ID with adequacy, names and register scores from 0 to 1 and a concise specific reason.
Pass only when adequacy >= 0.78, names >= 0.85, and no material omission/addition exists.

{lines}"""
        ids = [int(cue["id"]) for cue in batch]
        schema = array_of({
            "type": "object",
            # ``reason`` comes first on purpose: the grammar forces the model to state
            # its evidence before committing to scores, a one-line chain of thought.
            "properties": {"id": {"type": "integer", "enum": ids}, "reason": {"type": "string"},
                           "adequacy": {"type": "number"}, "names": {"type": "number"},
                           "register": {"type": "number"}, "passed": {"type": "boolean"}},
            "required": ["id", "reason", "adequacy", "names", "register", "passed"],
        }, "verdicts", min_items=len(ids), max_items=len(ids))
        valid: dict[int, dict] = {}
        try:
            for item in ask_json(llm, prompt, schema, max_tokens=120 + 90 * len(ids),
                                 temperature=settings.translation_qc_temperature,
                                 top_p=settings.translation_qc_top_p)["verdicts"]:
                valid.setdefault(int(item["id"]), item)
        except StructuredOutputError as exc:
            print(json.dumps({"warning": f"judge batch failed: {exc}"}), flush=True)
        for cue in batch:
            item = valid.get(int(cue["id"]))
            if not item:
                item = {"id": cue["id"], "adequacy": 0.0, "names": 0.0, "register": 0.0,
                        "passed": False, "reason": "independent judge returned no result"}
            item["available"] = True
            item["model"] = f"{Path(spec['model']).stem} independent bilingual judge"
            for key in ("adequacy", "names", "register"):
                item[key] = number(item.get(key))
            item["passed"] = bool(item.get("passed")) and item["adequacy"] >= .78
            results.append(item)
        completed += len(batch)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": min(1.0, completed / max(1, len(cues))),
                          "index": completed - 1}), flush=True)


if __name__ == "__main__":
    main()
