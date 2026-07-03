"""
.pdf parser — single deterministic method: plain PyMuPDF (fitz), NOT
pymupdf4llm.

pymupdf4llm was the original choice but got dropped after empirical timing:
on a 102-page real corpus PDF, pymupdf4llm.to_markdown() took ~100s
(independent of table_strategy — tested lines_strict/lines/None, all ~100s),
while plain fitz on the SAME file did get_text() for all pages in 0.22s and
find_tables() in 3.9s — a ~450x difference with no quality loss for a
deterministic, non-ML pass. pymupdf4llm's per-page markdown reconstruction
(font clustering, block-merging, layout inference) is doing much more work
than this pass needs or than "простой понятный детерминированный парсинг"
calls for.

Bonus: plain fitz's page.get_images() only enumerates genuinely embedded
raster XObjects — no vector-graphics noise. On the same CM_03_11.pdf where
pymupdf4llm extracted 194 images (many sub-KB page-furniture fragments),
plain fitz found 119 image refs / 118 unique xrefs — a much more sensible
count, and de-duping by xref for free catches repeated logos/headers reused
across pages (the same problem a.py's ImageStore solved via content hash;
here it's solved via the PDF's own object references, which is exact rather
than heuristic).
"""
from pathlib import Path

import fitz

from .common import ImageWriter, ParsedBlock, ParsedDocument
from .config import PDF_IMAGE_MIN_BYTES

# fitz's find_tables() occasionally misdetects a normal two-column article
# layout as a table and dumps an entire paragraph into one cell (found
# empirically: a scientific-article PDF produced a "table" whose single cell
# held ~5000 characters of running prose). Real data tables in this corpus
# (prices, production figures) have short cells; anything this long is a
# layout-detection false positive, not tabular data — and the same text is
# already captured correctly via the normal per-page text blocks below, so
# dropping the bogus table loses nothing.
_TABLE_CELL_MAX_CHARS = 400


def parse(path: Path, image_writer: ImageWriter) -> ParsedDocument:
    doc = fitz.open(str(path))
    try:
        if doc.is_encrypted:
            raise ValueError("PDF is encrypted/password-protected")

        parsed = ParsedDocument(
            source_file=path.name, doc_type="pdf",
            metadata={"pages": doc.page_count}, units_total=doc.page_count,
        )

        xref_to_image_id = {}
        dropped = 0

        for page in doc:
            for block in page.get_text("blocks"):
                text = block[4]
                block_type = block[6]
                if block_type != 0 or not text:
                    continue
                for line in text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if len(line) < 80 and line.isupper():
                        parsed.blocks.append(ParsedBlock(kind="heading", content=line, level=2, unit=page.number + 1))
                    else:
                        parsed.blocks.append(ParsedBlock(kind="text", content=line, unit=page.number + 1))

            try:
                for table in page.find_tables().tables:
                    rows = table.extract()
                    if not rows:
                        continue
                    max_cell = max((len(str(c)) for row in rows for c in row if c), default=0)
                    if max_cell > _TABLE_CELL_MAX_CHARS:
                        continue  # layout-detection false positive, see comment above
                    parsed.blocks.append(ParsedBlock(kind="table", content=rows, unit=page.number + 1))
            except Exception:
                pass  # table detection is best-effort; text above already captured the content

            for img in page.get_images(full=True):
                xref = img[0]
                if xref in xref_to_image_id:
                    parsed.blocks.append(ParsedBlock(kind="image", image_id=xref_to_image_id[xref], unit=page.number + 1))
                    continue
                try:
                    extracted = doc.extract_image(xref)
                    data, ext = extracted["image"], extracted["ext"]
                except Exception:
                    continue
                if len(data) < PDF_IMAGE_MIN_BYTES:
                    dropped += 1
                    continue
                image_id = image_writer.add(data, ext)
                xref_to_image_id[xref] = image_id
                parsed.blocks.append(ParsedBlock(kind="image", image_id=image_id, unit=page.number + 1))

        parsed.metadata["images_dropped_below_floor"] = dropped
        return parsed
    finally:
        doc.close()
