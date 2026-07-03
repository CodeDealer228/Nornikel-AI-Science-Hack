import json
import re

from .models import LLMExtraction

_JSON_BLOCK_RE = re.compile(r"`{3}(?:json)?\s*(\{.*?\})\s*`{3}", re.S | re.I)


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    block = _JSON_BLOCK_RE.search(stripped)
    if block:
        stripped = block.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start == -1:
        raise ValueError("response does not contain a JSON object")

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start:i + 1]
    raise ValueError("response contains an unfinished JSON object")


def parse_llm_json(text: str) -> LLMExtraction:
    obj = json.loads(extract_json_object(text))
    if hasattr(LLMExtraction, "model_validate"):
        return LLMExtraction.model_validate(obj)
    return LLMExtraction(**obj)
