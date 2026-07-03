"""
.pdf parser — single deterministic method, no OCR: pymupdf4llm.to_markdown().

pymupdf4llm embeds every extracted raster as a standard markdown image ref
(`![](path/to/file.png)`) inline in reading order. Empirically (see the plan),
an unfiltered run on a real 90-page corpus PDF pulled out 194 images ranging
424 bytes-1MB — the sub-few-KB ones are repeated page furniture (masthead
icons, bullets), not content. PDF_IMAGE_MIN_BYTES drops those; everything at
or above it becomes a real [IMAGE_ID] in the output.
"""
import re
import shutil
import tempfile
from pathlib import Path

import fitz
import pymupdf4llm

from .common import ImageWriter, ParsedBlock, ParsedDocument
from .config import PDF_IGNORE_GRAPHICS, PDF_IMAGE_FORMAT, PDF_IMAGE_MIN_BYTES, PDF_IMAGE_SIZE_LIMIT

_IMG_REF_RE = re.compile(r"!\[\]\(([^)]+)\)")


def parse(path: Path, image_writer: ImageWriter) -> ParsedDocument:
    doc = fitz.open(str(path))
    try:
        if doc.is_encrypted:
            raise ValueError("PDF is encrypted/password-protected")
        page_count = doc.page_count

        staging = Path(tempfile.mkdtemp(prefix="pdfimg_"))
        try:
            raw_md = pymupdf4llm.to_markdown(
                doc,
                write_images=True,
                image_path=str(staging),
                image_format=PDF_IMAGE_FORMAT,
                ignore_graphics=PDF_IGNORE_GRAPHICS,
                image_size_limit=PDF_IMAGE_SIZE_LIMIT,
            )
            rewritten, dropped = _rewrite_image_refs(raw_md, staging, image_writer)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        parsed = ParsedDocument(
            source_file=path.name,
            doc_type="pdf",
            metadata={"pages": page_count, "images_dropped_below_floor": dropped},
            units_total=page_count,
        )
        parsed.blocks.append(ParsedBlock(kind="text", content=rewritten))
        return parsed
    finally:
        doc.close()


def _rewrite_image_refs(md_text: str, staging: Path, image_writer: ImageWriter) -> tuple[str, int]:
    dropped = 0

    def _replace(match: re.Match) -> str:
        nonlocal dropped
        ref = match.group(1)
        img_path = Path(ref)
        if not img_path.is_absolute():
            img_path = staging / Path(ref).name
        if not img_path.is_file() or img_path.stat().st_size < PDF_IMAGE_MIN_BYTES:
            dropped += 1
            return ""
        image_id = image_writer.add_from_file(img_path)
        return f"[IMAGE_{image_id}]"

    rewritten = _IMG_REF_RE.sub(_replace, md_text)
    return rewritten, dropped
