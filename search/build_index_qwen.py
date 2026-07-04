from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("parsed_data/chunks.jsonl")
OUT_DIR = Path("parsed_data/search_index_qwen")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+(?:[-_][a-zA-Zа-яА-ЯёЁ0-9]+)?")


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            chunk = json.loads(line)
            text = chunk.get("text", "")

            if len(text.strip()) < 80:
                continue

            chunks.append(chunk)

    return chunks


def simple_tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def get_chunk_text(chunk: dict) -> str:
    return chunk.get("text", "") or ""


def get_source_document(chunk: dict) -> str:
    provenance = chunk.get("provenance") or {}
    return provenance.get("source_document") or chunk.get("doc_id", "") or ""


def get_heading_path(chunk: dict) -> list[str]:
    provenance = chunk.get("provenance") or {}
    return provenance.get("heading_path", []) or []


def make_embedding_text(chunk: dict) -> str:
    source = get_source_document(chunk)
    heading = " > ".join(get_heading_path(chunk))
    text = get_chunk_text(chunk)

    return f"Документ: {source}\nРаздел: {heading}\nТекст: {text}"


def get_bm25_tokens(chunk: dict, fallback_text: str) -> list[str]:
    """
    Для BM25 кладем и Natasha-леммы, и обычные токены.
    Так BM25 лучше переживает падежи и формы слов.
    """
    lemmas = chunk.get("natasha", {}).get("lemmas", []) or []
    lemma_tokens = [str(x).lower() for x in lemmas if str(x).strip()]
    raw_tokens = simple_tokenize(fallback_text)

    result = []
    seen = set()

    for token in lemma_tokens + raw_tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)

    return result


def save_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks_path = args.chunks
    out_dir = args.out_dir

    if not chunks_path.exists():
        raise FileNotFoundError(f"Не найден файл: {chunks_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[qwen-index] loading chunks from {chunks_path}")
    chunks = load_chunks(chunks_path)
    print(f"[qwen-index] chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("Нет чанков для индексации")

    embedding_texts = []
    tokenized_corpus = []

    for chunk in chunks:
        text_for_embedding = make_embedding_text(chunk)

        # Для Qwen документы кодируем без 'passage:' prefix.
        embedding_texts.append(text_for_embedding)

        tokens = get_bm25_tokens(chunk, text_for_embedding)
        tokenized_corpus.append(tokens)

    print("[qwen-index] building BM25")
    bm25 = BM25Okapi(tokenized_corpus)

    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[qwen-index] loading embedding model: {MODEL_NAME} (device={device})")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("[qwen-index] encoding chunks")
    embeddings = model.encode(
        embedding_texts,
        batch_size=2,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"[qwen-index] embeddings shape: {embeddings.shape}")
    np.save(out_dir / "embeddings.npy", embeddings)

    print("[qwen-index] saving chunks copy")
    save_jsonl(chunks, out_dir / "chunks.jsonl")

    metadata = {
        "model_name": MODEL_NAME,
        "n_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),

        # Для Qwen запросы будем кодировать через prompt_name='query'.
        "query_prompt_name": "query",

        # Для совместимости со старым searcher.
        "uses_e5_prefix": False,
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[qwen-index] done: {out_dir}")


if __name__ == "__main__":
    main()