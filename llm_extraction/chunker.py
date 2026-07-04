"""Recursive markdown chunker (LangChain) for parsed_data/texts/<category>/*.md.

Splits on markdown structure first (headings, paragraph breaks) and only
falls back to sentence/word/char boundaries when a section is still too big
-- avoids cutting mid-sentence or mid-table wherever the document structure
allows it. Target size matches mass_extraction_pipeline.md's empirical
~2.2 chars/token estimate for Russian BPE: ~2200 chars =~ 1000 tokens.
"""
import json
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 2200
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n## ", "\n### ", "\n#### ",   # markdown headings first
        "\n\n",                          # paragraph breaks
        "\n",                            # line breaks
        ". ", "? ", "! ",                # sentence boundaries
        " ", "",                         # last resort
    ],
    keep_separator=True,
)


def chunk_file(md_path: Path, doc_id: str) -> Iterator[dict]:
    text = md_path.read_text(encoding="utf-8")
    pieces = _splitter.split_text(text)
    offset = 0
    for i, piece in enumerate(pieces, start=1):
        start = text.find(piece, offset)
        if start == -1:
            start = offset
        end = start + len(piece)
        offset = max(offset, start + 1)
        yield {
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_c{i:03d}",
            "chunk_total": len(pieces),
            "char_start": start,
            "char_end": end,
            "text": piece,
        }


def chunk_folder(folder: Path) -> list[dict]:
    chunks = []
    for md_path in sorted(folder.glob("*.md")):
        doc_id = md_path.stem
        chunks.extend(chunk_file(md_path, doc_id))
    return chunks


if __name__ == "__main__":
    import sys

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../parsed_data/texts/Статьи")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("chunks_statyi.jsonl")

    chunks = chunk_folder(folder)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = [len(c["text"]) for c in chunks]
    n_docs = len(set(c["doc_id"] for c in chunks))
    print(f"docs={n_docs} chunks={len(chunks)} avg_size={sum(sizes)//max(1,len(sizes))} "
          f"min={min(sizes) if sizes else 0} max={max(sizes) if sizes else 0}")
    print(f"written to {out_path}")
