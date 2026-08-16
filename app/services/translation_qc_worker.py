from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        return json.loads(match.group(0)) if match else []


def ask(llm, prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], temperature=0.0,
        top_p=.85, max_tokens=1500,
    )
    return str(response["choices"][0]["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    cues = spec["cues"]
    from llama_cpp import Llama
    gpu_layers = int(os.getenv("DUB_LLAMA_GPU_LAYERS", "-1"))
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
            f"FAITHFUL ENGLISH: {cue.get('faithful_translation') or cue.get('literal_translation','')}\n"
            f"DUB ENGLISH: {cue.get('english','')}"
            for cue in batch
        )
        prompt = f"""You are an independent bilingual film-translation quality checker.
The translations were created by a different model. Judge SOURCE directly against DUB ENGLISH;
use FAITHFUL ENGLISH only as secondary evidence. Detect changed facts, polarity, names, relationships,
omissions, additions, mistranslated idioms and register changes. Do not reward fluency alone.
Return only a JSON array, one object per ID:
{{"id":1,"adequacy":0.0,"names":0.0,"register":0.0,"passed":false,"reason":"concise specific reason"}}
Pass only when adequacy >= 0.78, names >= 0.85, and no material omission/addition exists.

{lines}"""
        values = parse_json(ask(llm, prompt))
        valid = {int(item.get("id", -1)): item for item in values if isinstance(item, dict)}
        for cue in batch:
            item = valid.get(int(cue["id"]))
            if not item:
                item = {"id": cue["id"], "adequacy": 0.0, "names": 0.0, "register": 0.0,
                        "passed": False, "reason": "independent judge returned no result"}
            item["available"] = True
            item["model"] = "Qwen3-8B Q4 independent bilingual judge"
            item["passed"] = bool(item.get("passed")) and float(item.get("adequacy", 0)) >= .78
            results.append(item)
        completed += len(batch)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": min(1.0, completed / max(1, len(cues))),
                          "index": completed - 1}), flush=True)


if __name__ == "__main__":
    main()
