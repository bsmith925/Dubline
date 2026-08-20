from __future__ import annotations

"""Schema-constrained JSON generation for the local llama.cpp models.

``response_format`` with a JSON schema makes llama.cpp compile the schema into a
GBNF grammar, so the model can only emit a document that validates: no prose,
no reasoning blocks, no truncated arrays, and ids restricted to the values we
enumerate.  Call sites describe the shape they need and receive parsed JSON.
"""

import json
from typing import Any


class StructuredOutputError(RuntimeError):
    pass


def ask_json(llm, prompt: str, schema: dict, *, max_tokens: int = 900, temperature: float = 0.0,
             top_p: float = 0.9, retries: int = 1) -> Any:
    """Return the parsed JSON object the model produced under ``schema``.

    Grammar guarantees syntactic validity; the only failure mode left is hitting
    ``max_tokens`` mid-document, which is retried once with a larger budget.
    """
    budget = max_tokens
    last_error: Exception | None = None
    for _ in range(retries + 1):
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object", "schema": schema},
            temperature=temperature, top_p=top_p, max_tokens=budget,
        )
        choice = response["choices"][0]
        content = str(choice["message"]["content"] or "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            if choice.get("finish_reason") != "length":
                break
            budget *= 2
    raise StructuredOutputError(f"model output did not complete under the schema: {last_error}")


def number(value: Any, default: float = 0.0, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def array_of(item_schema: dict, key: str, *, min_items: int | None = None, max_items: int | None = None) -> dict:
    items: dict = {"type": "array", "items": item_schema}
    if min_items is not None:
        items["minItems"] = min_items
    if max_items is not None:
        items["maxItems"] = max_items
    return {"type": "object", "properties": {key: items}, "required": [key]}
