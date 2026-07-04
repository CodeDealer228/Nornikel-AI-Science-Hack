from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


CHUNKS_PATH = Path("parsed_data/chunks.jsonl")
OUT_DIR = Path("parsed_data/search_index")

MODEL_NAME = "intfloat/multilingual-e5-base"


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
    return chunk.get("provenance", {}).get("source_document", "") or ""


def get_heading_path(chunk: dict) -> list[str]:
    return chunk.get("provenance", {}).get("heading_path", []) or []


def make_embedding_text(chunk: dict) -> str:
    """
    В embedding лучше давать не только голый текст,
    но и источник/раздел. Так поиск лучше понимает контекст.
    """
    source = get_source_document(chunk)
    heading = " > ".join(get_heading_path(chunk))
    text = get_chunk_text(chunk)

    return f"Документ: {source}\nРаздел: {heading}\nТекст: {text}"


def get_bm25_tokens(chunk: dict, fallback_text: str) -> list[str]:
    """
    Для BM25 используем и Natasha-леммы, и обычные токены.
    Это помогает, потому что запрос пока не лемматизируется.
    """
    lemmas = chunk.get("natasha", {}).get("lemmas", []) or []
    lemma_tokens = [str(x).lower() for x in lemmas if str(x).strip()]
    raw_tokens = simple_tokenize(fallback_text)

    # Убираем дубли, сохраняя порядок
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


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {CHUNKS_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[index] loading chunks from {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"[index] chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("Нет чанков для индексации")

    print("[index] preparing texts")
    embedding_texts = []
    tokenized_corpus = []

    for chunk in chunks:
        text_for_embedding = make_embedding_text(chunk)
        embedding_texts.append("passage: " + text_for_embedding)

        tokens = get_bm25_tokens(chunk, text_for_embedding)
        tokenized_corpus.append(tokens)

    print("[index] building BM25")
    bm25 = BM25Okapi(tokenized_corpus)

    with (OUT_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    print(f"[index] loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("[index] encoding chunks")
    embeddings = model.encode(
        embedding_texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"[index] embeddings shape: {embeddings.shape}")
    np.save(OUT_DIR / "embeddings.npy", embeddings)

    print("[index] saving chunks copy")
    save_jsonl(chunks, OUT_DIR / "chunks.jsonl")

    metadata = {
        "model_name": MODEL_NAME,
        "n_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "uses_e5_prefix": True,
    }

    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[index] done: {OUT_DIR}")


if __name__ == "__main__":
    main()