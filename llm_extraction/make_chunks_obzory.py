"""Chunks parsed_data/texts/Обзоры at 2000 chars, same approach as
make_chunks_2000.py used for Статьи, plus a minimal regex cleanup of each
chunk's text AFTER chunking (see clean_chunk.py for why after, not before).
"""
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from clean_chunk import clean_chunk_text

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n## ", "\n### ", "\n#### ",
        "\n\n",
        "\n",
        ". ", "? ", "! ",
        " ", "",
    ],
    keep_separator=True,
)


def chunk_file(md_path: Path, doc_id: str):
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
            "text": clean_chunk_text(piece),
        }


def chunk_folder(folder: Path):
    chunks = []
    for md_path in sorted(folder.rglob("*.md")):
        chunks.extend(chunk_file(md_path, md_path.stem))
    return chunks


if __name__ == "__main__":
    import sys

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../parsed_data/texts/Обзоры")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("chunks_obzory_2000.jsonl")

    chunks = chunk_folder(folder)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = [len(c["text"]) for c in chunks]
    n_docs = len(set(c["doc_id"] for c in chunks))
    print(f"docs={n_docs} chunks={len(chunks)} avg_size={sum(sizes)//max(1,len(sizes))} "
          f"min={min(sizes) if sizes else 0} max={max(sizes) if sizes else 0}")
    print(f"written to {out_path}")
