"""Batched NER+RE extraction: packs up to `--pack` chunks (default 40) into a
single Yandex AI Studio call instead of one chunk per call (see extract.py for
the original one-chunk-per-call version). Uses the model's large context
window to amortize the ~13k-token system prompt (instruction + few-shot) over
many chunks per call, cutting the number of calls by ~40x and the wall-clock
time proportionally, while keeping the same per-chunk output schema so
downstream consumers of the JSONL don't need to change.

System prompt: build_system_prompt_batch.py (batch contract + 2 batch few-shot
demos, same 10 curated examples as the single-chunk prompt, regrouped).

Batching risk (per mass_extraction_pipeline.md): a big pack can get truncated
by max_output_tokens, or the model can occasionally drop/merge chunk_ids. On
any sign of that (JSON parse failure, incomplete_details, or missing/extra
chunk_ids in the response) the batch is bisected and retried recursively down
to single-chunk calls, instead of silently losing chunks.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from env_load import API_KEY, FOLDER_ID
from build_system_prompt_batch import build_system_prompt_batch
from extract import (
    URL, HEADERS, VALID_ENTITY_TYPES, VALID_PREDICATES,
    extract_message_text, strip_json_fence, validate_parsed, normalize_ws,
)

MODEL = f"gpt://{FOLDER_ID}/deepseek-v4-flash/latest"
# Empirically: single-chunk output averages ~5200 tokens with the reasoning
# summary enabled, ~half that with generate_summary=False (measured on this
# corpus). Budget generously per chunk since the API doesn't hard-reject large
# max_output_tokens values -- the bisection fallback in process_batch_recursive
# is the real truncation safety net, not this cap.
MAX_OUTPUT_TOKENS_PER_CHUNK = 4000
MAX_OUTPUT_TOKENS_CAP = 200000


def build_user_message(batch: list[dict]) -> str:
    blocks = []
    for c in batch:
        blocks.append(f'### ЧАНК {c["chunk_id"]}\n"""\n{c["text"]}\n"""')
    instruction = (
        "\n\nИзвлеки сущности и связи для КАЖДОГО чанка выше по инструкции. Верни ОДИН JSON-объект "
        '{"results": [...]} — ровно одна запись на каждый chunk_id, в том же порядке, что и чанки выше.'
    )
    return "\n\n".join(blocks) + instruction


async def call_batch(client: httpx.AsyncClient, system_prompt: str, batch: list[dict],
                      sem: asyncio.Semaphore, max_retries: int = 5) -> dict:
    """Returns {"status": ..., "by_chunk_id": {chunk_id: parsed_or_error}, "raw": ...}"""
    user_msg = build_user_message(batch)
    max_out = min(MAX_OUTPUT_TOKENS_CAP, MAX_OUTPUT_TOKENS_PER_CHUNK * len(batch))
    body = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "reasoning": {"effort": "high", "generate_summary": False},
        "max_output_tokens": max_out,
        "prompt_cache_key": "nornikel-ner-re-batch-v1",
        "prompt_cache_retention": "24h",
    }

    meta = {"batch_chunk_ids": [c["chunk_id"] for c in batch], "batch_size": len(batch)}
    delay = 2.0
    async with sem:
        for attempt in range(1, max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = await client.post(URL, headers=HEADERS, json=body, timeout=600)
            except httpx.RequestError as e:
                if attempt == max_retries:
                    return {**meta, "status": "REQUEST_ERROR", "error": str(e)}
                await asyncio.sleep(delay)
                delay *= 2
                continue

            latency_s = round(time.perf_counter() - t0, 2)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                if attempt == max_retries:
                    return {**meta, "status": "RATE_LIMITED_GAVE_UP", "latency_s": latency_s}
                await asyncio.sleep(wait)
                delay *= 2
                continue

            if r.status_code != 200:
                if attempt == max_retries or r.status_code < 500:
                    return {**meta, "status": f"HTTP_{r.status_code}", "error": r.text[:500], "latency_s": latency_s}
                await asyncio.sleep(delay)
                delay *= 2
                continue

            body_json = r.json()
            if body_json.get("error"):
                if attempt == max_retries:
                    return {**meta, "status": "MODEL_CALL_ERROR",
                            "error": json.dumps(body_json["error"], ensure_ascii=False), "latency_s": latency_s}
                await asyncio.sleep(delay)
                delay *= 2
                continue

            usage = body_json.get("usage", {})
            incomplete = body_json.get("incomplete_details")

            text = extract_message_text(body_json)
            if text is None:
                return {**meta, "status": "NO_MESSAGE_OUTPUT", "latency_s": latency_s}

            try:
                parsed = json.loads(strip_json_fence(text))
                results = parsed.get("results")
                if not isinstance(results, list):
                    raise ValueError("no 'results' array in response")
            except (json.JSONDecodeError, ValueError) as e:
                return {**meta, "status": "JSON_PARSE_ERROR", "error": str(e),
                        "raw_output": text[:3000], "latency_s": latency_s,
                        "incomplete_details": incomplete}

            by_id = {}
            for res in results:
                cid = res.get("chunk_id")
                if cid is not None:
                    by_id[cid] = res

            expected_ids = [c["chunk_id"] for c in batch]
            missing = [cid for cid in expected_ids if cid not in by_id]

            return {
                **meta,
                "status": "OK",
                "by_chunk_id": by_id,
                "missing_chunk_ids": missing,
                "usage": usage,
                "incomplete_details": incomplete,
                "latency_s": latency_s,
            }

    return {**meta, "status": "GAVE_UP"}


async def process_batch_recursive(client: httpx.AsyncClient, system_prompt: str, batch: list[dict],
                                   sem: asyncio.Semaphore, min_pack: int = 1) -> list[dict]:
    """Runs one batch call; on failure or missing chunk_ids, bisects and retries
    the affected sub-batches recursively (down to single-chunk calls)."""
    res = await call_batch(client, system_prompt, batch, sem)

    needs_bisect = (
        res["status"] != "OK"
        or res.get("missing_chunk_ids")
        or (res.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
    )

    if needs_bisect:
        print(f"[bisect] pack={len(batch)} status={res.get('status')} "
              f"missing={res.get('missing_chunk_ids')} incomplete={res.get('incomplete_details')} "
              f"error={str(res.get('error'))[:200]}")

    if needs_bisect and len(batch) > min_pack:
        mid = len(batch) // 2
        left, right = batch[:mid], batch[mid:]
        left_results, right_results = await asyncio.gather(
            process_batch_recursive(client, system_prompt, left, sem, min_pack),
            process_batch_recursive(client, system_prompt, right, sem, min_pack),
        )
        return left_results + right_results

    out = []
    by_id = res.get("by_chunk_id", {})
    for c in batch:
        cid = c["chunk_id"]
        line = {"chunk_id": cid, "doc_id": c["doc_id"], "batch_size": res.get("batch_size")}
        if res["status"] != "OK":
            line["status"] = res["status"]
            line["error"] = res.get("error")
            out.append(line)
            continue
        chunk_res = by_id.get(cid)
        if chunk_res is None:
            line["status"] = "MISSING_IN_BATCH_RESPONSE"
            out.append(line)
            continue
        parsed_chunk = {"entities": chunk_res.get("entities", []), "relations": chunk_res.get("relations", [])}
        validation = validate_parsed(parsed_chunk, c["text"])
        line["status"] = "OK"
        line["parsed"] = parsed_chunk
        line["validation"] = validation
        line["latency_s"] = res.get("latency_s")
        out.append(line)
    return out


def make_packs(chunks: list[dict], pack_size: int) -> list[list[dict]]:
    return [chunks[i:i + pack_size] for i in range(0, len(chunks), pack_size)]


async def run(chunks_path: Path, out_path: Path, concurrency: int, pack_size: int, limit: int | None):
    system_prompt = build_system_prompt_batch()
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = [c for c in chunks if len(c["text"].strip()) >= 30]
    if limit:
        chunks = chunks[:limit]

    packs = make_packs(chunks, pack_size)
    print(f"[run] {len(chunks)} chunks -> {len(packs)} batches of up to {pack_size}, "
          f"concurrency={concurrency}, model={MODEL}")

    sem = asyncio.Semaphore(concurrency)
    done_chunks = 0
    done_batches = 0
    t_start = time.perf_counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        async with httpx.AsyncClient() as client:
            tasks = [asyncio.create_task(process_batch_recursive(client, system_prompt, p, sem)) for p in packs]
            for coro in asyncio.as_completed(tasks):
                lines = await coro
                for line in lines:
                    fout.write(json.dumps(line, ensure_ascii=False) + "\n")
                fout.flush()
                done_chunks += len(lines)
                done_batches += 1
                elapsed = time.perf_counter() - t_start
                print(f"[batches {done_batches}/{len(packs)}, chunks {done_chunks}/{len(chunks)}] "
                      f"elapsed={elapsed:.0f}s")

    print(f"[run] done in {time.perf_counter() - t_start:.0f}s -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="chunks_statyi_2000.jsonl")
    ap.add_argument("--out", default="extraction_results_statyi_batch.jsonl")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--pack", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(Path(args.chunks), Path(args.out), args.concurrency, args.pack, args.limit))
