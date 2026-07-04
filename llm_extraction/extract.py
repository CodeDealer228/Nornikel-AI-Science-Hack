"""Run NER+RE extraction over parsed_data/texts/Статьи chunks through Yandex
AI Studio (DeepSeek v4 flash, reasoning_effort=high), using the team's ready
system prompt + 10 few-shot examples (llm_pipeline_fewshot/ner_re_extraction_prompt.md).

This is a first practical test run (not the fully-optimized mass-extraction
orchestrator described in mass_extraction_pipeline.md) — pack=1 (one chunk per
call), modest concurrency, straightforward retry on 429/5xx. Saves one JSONL
line per chunk with the raw model output, parsed JSON (if valid), and
validation flags (mentions-are-substrings, relation endpoints exist).
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

from env_load import API_KEY, FOLDER_ID
from build_system_prompt import build_system_prompt

URL = "https://ai.api.cloud.yandex.net/v1/responses"
MODEL = f"gpt://{FOLDER_ID}/deepseek-v4-flash/latest"
HEADERS = {"Authorization": f"Api-Key {API_KEY}", "Content-Type": "application/json"}

VALID_ENTITY_TYPES = {
    "Material", "Process", "Equipment", "Property", "Experiment",
    "Publication", "Expert", "Facility",
}
VALID_PREDICATES = {
    "uses_material", "operates_at_condition", "produces_output", "described_in",
    "validated_by", "contradicts", "affiliated_with", "authored_by",
}

USER_TEMPLATE = """Документ: {doc_id} — {doc_id}
Чанк: {chunk_id} из {chunk_total}

Текст:
\"\"\"
{chunk_text}
\"\"\"

Извлеки сущности и связи по инструкции выше. Верни только JSON."""

_norm_ws = re.compile(r"\s+")


def normalize_ws(s: str) -> str:
    return _norm_ws.sub(" ", s).strip()


def extract_message_text(response_json: dict) -> str | None:
    for item in response_json.get("output", []):
        if item.get("type") == "message":
            content = item.get("content") or []
            for c in content:
                if c.get("type") == "output_text":
                    return c.get("text")
    return None


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_parsed(parsed: dict, chunk_text: str) -> dict:
    flags = []
    norm_text = normalize_ws(chunk_text)
    entities = parsed.get("entities", [])
    relations = parsed.get("relations", [])
    local_ids = {e.get("local_id") for e in entities}

    bad_type = [e["local_id"] for e in entities if e.get("type") not in VALID_ENTITY_TYPES]
    if bad_type:
        flags.append(f"INVALID_ENTITY_TYPE:{bad_type}")

    hallucinated_mentions = []
    for e in entities:
        for m in e.get("mentions", []):
            if normalize_ws(m) not in norm_text:
                hallucinated_mentions.append((e.get("local_id"), m))
    if hallucinated_mentions:
        flags.append(f"HALLUCINATED_MENTION:{len(hallucinated_mentions)}")

    bad_rel = []
    for r in relations:
        if r.get("subject") not in local_ids or r.get("object") not in local_ids:
            bad_rel.append(r)
        if r.get("predicate") not in VALID_PREDICATES:
            bad_rel.append(r)
    if bad_rel:
        flags.append(f"INVALID_RELATION:{len(bad_rel)}")

    return {
        "flags": flags,
        "n_entities": len(entities),
        "n_relations": len(relations),
        "n_hallucinated_mentions": len(hallucinated_mentions),
    }


async def call_one(client: httpx.AsyncClient, system_prompt: str, chunk: dict,
                    sem: asyncio.Semaphore, max_retries: int = 5) -> dict:
    user_msg = USER_TEMPLATE.format(
        doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"],
        chunk_total=chunk["chunk_total"], chunk_text=chunk["text"],
    )
    body = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "reasoning": {"effort": "high"},
        "max_output_tokens": 32000,
        "prompt_cache_key": "nornikel-ner-re-v1",
        "prompt_cache_retention": "24h",
    }

    result = {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]}
    delay = 2.0
    async with sem:
        for attempt in range(1, max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = await client.post(URL, headers=HEADERS, json=body, timeout=180)
            except httpx.RequestError as e:
                result["status"] = "REQUEST_ERROR"
                result["error"] = str(e)
                if attempt == max_retries:
                    return result
                await asyncio.sleep(delay)
                delay *= 2
                continue

            result["latency_s"] = round(time.perf_counter() - t0, 2)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                if attempt == max_retries:
                    result["status"] = "RATE_LIMITED_GAVE_UP"
                    return result
                await asyncio.sleep(wait)
                delay *= 2
                continue

            if r.status_code != 200:
                result["status"] = f"HTTP_{r.status_code}"
                result["error"] = r.text[:500]
                if attempt == max_retries or r.status_code < 500:
                    return result
                await asyncio.sleep(delay)
                delay *= 2
                continue

            body_json = r.json()
            if body_json.get("error"):
                result["status"] = "MODEL_CALL_ERROR"
                result["error"] = json.dumps(body_json["error"], ensure_ascii=False)
                if attempt == max_retries:
                    return result
                await asyncio.sleep(delay)
                delay *= 2
                continue

            usage = body_json.get("usage", {})
            result["usage"] = {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cached_tokens": usage.get("input_tokens_details", {}).get("cached_tokens"),
            }
            incomplete = body_json.get("incomplete_details")
            if incomplete:
                result["incomplete_details"] = incomplete

            text = extract_message_text(body_json)
            if text is None:
                result["status"] = "NO_MESSAGE_OUTPUT"
                return result

            raw_text = text
            try:
                parsed = json.loads(strip_json_fence(text))
            except json.JSONDecodeError as e:
                result["status"] = "JSON_PARSE_ERROR"
                result["error"] = str(e)
                result["raw_output"] = raw_text[:2000]
                return result

            validation = validate_parsed(parsed, chunk["text"])
            result["status"] = "OK"
            result["parsed"] = parsed
            result["validation"] = validation
            return result

    return result


async def run(chunks_path: Path, out_path: Path, concurrency: int, limit: int | None):
    system_prompt = build_system_prompt()
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = [c for c in chunks if len(c["text"].strip()) >= 30]
    if limit:
        chunks = chunks[:limit]

    print(f"[run] {len(chunks)} chunks, concurrency={concurrency}, model={MODEL}")

    sem = asyncio.Semaphore(concurrency)
    done = 0
    t_start = time.perf_counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        async with httpx.AsyncClient() as client:
            tasks = [asyncio.create_task(call_one(client, system_prompt, c, sem)) for c in chunks]
            for coro in asyncio.as_completed(tasks):
                res = await coro
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1
                if done % 10 == 0 or done == len(chunks):
                    elapsed = time.perf_counter() - t_start
                    print(f"[{done}/{len(chunks)}] elapsed={elapsed:.0f}s last_status={res.get('status')}")

    print(f"[run] done in {time.perf_counter() - t_start:.0f}s -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="chunks_statyi.jsonl")
    ap.add_argument("--out", default="extraction_results_statyi.jsonl")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(Path(args.chunks), Path(args.out), args.concurrency, args.limit))
