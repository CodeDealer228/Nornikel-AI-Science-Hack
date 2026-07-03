"""
Handles two cases with the same logic: a real standalone .gif, and any file
that the signature gate (signatures.py) found to actually be an image
despite a different claimed extension (e.g. this corpus's one file that is
a BMP mislabeled `.xls`). Either way: verify it opens, store it as the
document's single image, no text/OCR extraction in this pass.
"""
from pathlib import Path

from PIL import Image

from .common import ImageWriter, ParsedBlock, ParsedDocument


def parse(path: Path, image_writer: ImageWriter, detected_kind: str = "") -> ParsedDocument:
    with Image.open(path) as im:
        im.verify()
    ext = detected_kind or path.suffix.lstrip(".") or "png"

    parsed = ParsedDocument(
        source_file=path.name,
        doc_type="image",
        metadata={"note": "standalone image, no text/OCR extraction in this pass"},
        units_total=1,
    )
    image_id = image_writer.add(path.read_bytes(), ext)
    parsed.blocks.append(ParsedBlock(kind="image", image_id=image_id))
    return parsed
