"""
Shared intermediate representation for every format parser in this package,
plus the per-document image writer.

Deliberately NOT imported from a.py (see plan decision): this package stays
fully independent of the legacy script, even though the block/table shape is
similar. Image IDs here are scoped to a single document (IMAGE_0001, ...),
unlike a.py's ImageStore which deduplicates globally across the whole corpus.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ParsedBlock:
    kind: str  # "text" | "image" | "table" | "heading"
    content: Any = None
    level: int = 0
    image_id: Optional[str] = None
    caption: Optional[str] = None
    unit: Optional[int] = None  # page / slide / sheet index, 1-based

    def to_markdown(self) -> str:
        if self.kind == "heading":
            return f"{'#' * min(self.level, 6)} {self.content}\n"
        if self.kind == "image":
            lines = [f"\n[IMAGE_{self.image_id}]\n"]
            if self.caption:
                lines.append(f"_{self.caption}_\n")
            return "\n".join(lines)
        if self.kind == "table":
            return _table_to_md(self.content) + "\n"
        return f"{self.content}\n" if self.content else ""


def _table_to_md(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    rows = [[str(c).replace("\n", " ").strip() if c is not None else "" for c in r] for r in rows]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(width)]
    lines = []
    for i, row in enumerate(rows):
        line = "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
        lines.append(line)
        if i == 0:
            lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    return "\n".join(lines)


@dataclass
class ParsedDocument:
    source_file: str
    doc_type: str
    blocks: List[ParsedBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    units_total: int = 0  # page/slide/sheet count, for validation yield checks

    def to_markdown(self) -> str:
        parts = []
        if self.metadata:
            parts.append("---")
            for k, v in self.metadata.items():
                parts.append(f"{k}: {v}")
            parts.append("---\n")
        parts.append(f"# {Path(self.source_file).stem}\n")
        for b in self.blocks:
            parts.append(b.to_markdown())
        return "\n".join(parts)

    def text_length(self) -> int:
        """Total extracted content length — prose (text/heading) plus table
        cell content, since table-heavy formats (xls/xlsx) have no prose at
        all but the table content IS the meaningful extracted output."""
        total = 0
        for b in self.blocks:
            if b.kind in ("text", "heading") and b.content:
                total += len(str(b.content))
            elif b.kind == "table" and b.content:
                total += sum(len(str(c)) for row in b.content for c in row if c)
        return total

    def image_ids(self) -> List[str]:
        return [b.image_id for b in self.blocks if b.kind == "image" and b.image_id]


class ImageWriter:
    """Allocates sequential per-document IMAGE_IDs and writes files into one
    folder per source document (parsed_data/images/<mirror path>/)."""

    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self._counter = 0
        self._written = False

    def add(self, data: bytes, ext: str) -> str:
        self._counter += 1
        image_id = f"{self._counter:04d}"
        if not self._written:
            self.image_dir.mkdir(parents=True, exist_ok=True)
            self._written = True
        ext = (ext or "png").lstrip(".").lower() or "png"
        (self.image_dir / f"IMAGE_{image_id}.{ext}").write_bytes(data)
        return image_id

    def add_from_file(self, src: Path) -> str:
        return self.add(src.read_bytes(), src.suffix)

    @property
    def count(self) -> int:
        return self._counter
