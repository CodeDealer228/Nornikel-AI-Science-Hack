from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .numeric_extractor import extract_numeric_expressions
from .synonyms import SynonymExpander
from .query_processor import QueryProcessor


INDEX_DIR = Path("parsed_data/search_index")

DENSE_TOP_N = 50
BM25_TOP_N = 50
DEFAULT_TOP_K = 10


TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+(?:[-_][a-zA-Zа-яА-ЯёЁ0-9]+)?")


def simple_tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def make_snippet(text: str, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def get_chunk_text(chunk: dict[str, Any]) -> str:
    return chunk.get("text", "") or ""


def get_source_document(chunk: dict[str, Any]) -> str:
    return chunk.get("provenance", {}).get("source_document", "") or ""


def get_heading_path(chunk: dict[str, Any]) -> list[str]:
    return chunk.get("provenance", {}).get("heading_path", []) or []


def make_title(chunk: dict[str, Any]) -> str:
    source = get_source_document(chunk)
    heading = " > ".join(get_heading_path(chunk))

    if heading:
        return f"{source} | {heading}"

    return source


def top_indices(scores: np.ndarray, top_n: int) -> list[int]:
    top_n = min(top_n, len(scores))
    return list(np.argsort(scores)[::-1][:top_n])


def rrf_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion.

    Объединяет несколько рейтингов.
    Нам не нужно нормализовать BM25-score и cosine-score:
    мы используем только позиции документов в выдаче.
    """
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridSearcher:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir

        metadata_path = index_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Не найден поисковый индекс: {index_dir}. "
                f"Сначала запусти: python -m search.build_index"
            )

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.chunks = load_jsonl(index_dir / "chunks.jsonl")
        self.embeddings = np.load(index_dir / "embeddings.npy")

        with (index_dir / "bm25.pkl").open("rb") as f:
            self.bm25 = pickle.load(f)

        model_name = self.metadata["model_name"]
        self.model = SentenceTransformer(model_name)

        self.uses_e5_prefix = bool(self.metadata.get("uses_e5_prefix", True))

        # Синонимы подключаются только к BM25/query expansion.
        # Dense/Qwen-запрос остается исходным.
        self.synonym_expander = SynonymExpander.from_project()
        self.query_processor = QueryProcessor(self.synonym_expander)

    def dense_search(self, query: str, top_n: int = DENSE_TOP_N) -> tuple[list[int], np.ndarray]:
        """
        Семантический поиск.

        Для e5 используем prefix 'query: '.
        Для Qwen используем prompt_name='query', если он есть в metadata.
        """
        query_prompt_name = self.metadata.get("query_prompt_name")

        if query_prompt_name:
            query_embedding = self.model.encode(
                [query],
                prompt_name=query_prompt_name,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            query_text = "query: " + query if self.uses_e5_prefix else query

            query_embedding = self.model.encode(
                [query_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        query_embedding = np.asarray(query_embedding, dtype=np.float32)[0]

        scores = self.embeddings @ query_embedding

        ranking = top_indices(scores, top_n)
        return ranking, scores

    def expand_lexical_query(self, query: str) -> list[str]:
        """
        Расширяет запрос для BM25 через synonyms.yaml.

        Важно:
        include_canonical_id=False, потому что текущий BM25-индекс
        построен без canonical_id в документах. Если позже добавим
        canonical_id на этапе build_index.py, тогда можно будет поставить True.
        """
        query_tokens = simple_tokenize(query)

        expanded_tokens = self.synonym_expander.expand_query_tokens(
            query=query,
            base_tokens=query_tokens,
            tokenizer=simple_tokenize,
            include_canonical_id=False,
        )

        return expanded_tokens

    def get_synonym_matches(self, query: str) -> list[dict[str, Any]]:
        """
        Для debug: показывает, какие synonym-группы сработали на запрос.
        """
        return self.synonym_expander.find_matches(query)

    def bm25_search(self, query: str, top_n: int = BM25_TOP_N) -> tuple[list[int], np.ndarray]:
        """
        Лексический поиск.

        На вход сюда уже должен приходить bm25_query:
        - русский перевод;
        - glossary terms;
        - aliases из synonyms.yaml.
        """
        tokens = simple_tokenize(query)

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)

        ranking = top_indices(scores, top_n)
        return ranking, scores

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        query_plan = self.query_processor.process(query)

        dense_ranking, dense_scores = self.dense_search(query_plan.dense_query)
        bm25_ranking, bm25_scores = self.bm25_search(query_plan.bm25_query)

        fused = rrf_fusion([dense_ranking, bm25_ranking])
        fused = fused[:top_k]

        results = []

        for rank, (idx, fused_score) in enumerate(fused, start=1):
            chunk = self.chunks[idx]
            text = get_chunk_text(chunk)

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.get("chunk_id"),
                    "title": make_title(chunk),
                    "source_document": get_source_document(chunk),
                    "heading_path": get_heading_path(chunk),
                    "char_start": chunk.get("provenance", {}).get("char_start"),
                    "char_end": chunk.get("provenance", {}).get("char_end"),
                    "fused_score": float(fused_score),
                    "dense_score": float(dense_scores[idx]),
                    "bm25_score": float(bm25_scores[idx]),
                    "snippet": make_snippet(text),
                    "text": text,
                    "numeric_expressions": extract_numeric_expressions(text),
                }
            )

        return results

    def get_query_plan(self, query: str) -> dict[str, Any]:
        return self.query_processor.process(query).to_dict()

    def get_dense_query(self, query: str) -> str:
        return self.query_processor.process(query).dense_query

    def get_bm25_query(self, query: str) -> str:
        return self.query_processor.process(query).bm25_query

    def get_synonym_matches(self, query: str) -> list[dict[str, Any]]:
        return self.query_processor.process(query).matched_synonyms

    def expand_lexical_query(self, query: str) -> list[str]:
        bm25_query = self.get_bm25_query(query)
        return simple_tokenize(bm25_query)