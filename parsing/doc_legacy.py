"""
.doc (legacy binary Word) AND .docm parser — convert via a real MS Word COM
instance, then parse the result with docx_parser.

.doc: no pure-Python library reads the old binary format reliably; Word's
own converter is deterministic (not ML/OCR) and this machine has Office
installed, so it's the natural fit.

.docm: structurally plain OOXML, but python-docx's `Document()` rejects it
outright — it validates the main part's content type exactly, and .docm
declares `application/vnd.ms-word.document.macroEnabled.main+xml` instead of
the docx-specific type, raising `ValueError: ... is not a Word file` even
though the document is perfectly well-formed (confirmed empirically against
all 3 real .docm files in this corpus). Routing it through the same Word-COM
round-trip as .doc sidesteps that check entirely — Word itself has no such
restriction — and reuses code already proven correct, rather than adding a
second, cleverer fix (e.g. patching the zip's content-types XML in memory).

FileFormat=12 is WdSaveFormat.wdFormatXMLDocument (.docx) — verified
empirically against a real corpus .doc file (produced a valid PK-zip that
python-docx opened with real extracted text) before relying on it here, since
guessing this constant wrong would silently produce a corrupt/wrong-format
file that might not even raise an error.

21 files total in this corpus (18 .doc + 3 .docm) — processed serially
through one shared Word Application instance (COM automation under
concurrency is fragile; serial is fast enough at this volume and one bad
file can't corrupt the shared app because Documents.Open/Close is scoped per
call).
"""
from pathlib import Path

import win32com.client as win32

from . import docx_parser
from .common import ImageWriter, ParsedDocument

_WD_FORMAT_DOCX = 12


class WordConverter:
    def __enter__(self) -> "WordConverter":
        self.app = win32.Dispatch("Word.Application")
        self.app.Visible = False
        self.app.DisplayAlerts = 0
        return self

    def __exit__(self, *exc_info):
        try:
            self.app.Quit()
        except Exception:
            pass

    def convert_to_docx(self, src: Path, dst: Path) -> None:
        doc = self.app.Documents.Open(str(src.resolve()), ReadOnly=True)
        try:
            doc.SaveAs(str(dst.resolve()), FileFormat=_WD_FORMAT_DOCX)
        finally:
            doc.Close(False)


def parse(path: Path, image_writer: ImageWriter, converter: WordConverter, staging_dir: Path) -> ParsedDocument:
    staging_dir.mkdir(parents=True, exist_ok=True)
    converted = staging_dir / f"{path.stem}.docx"
    try:
        converter.convert_to_docx(path, converted)
        parsed = docx_parser.parse(converted, image_writer)
        parsed.doc_type = path.suffix.lstrip(".").lower()
        parsed.metadata["converted_via"] = "word_com"
        return parsed
    finally:
        converted.unlink(missing_ok=True)
