"""
.xls (legacy binary Excel) parser — single deterministic method: xlrd.

xlrd is XLS-only since 2.0 (it dropped xlsx support), which is exactly what
we want here: pure Python, no compiled deps, no COM automation needed for
432 files. No image extraction — xlrd doesn't support it and there is no
lightweight pure-Python alternative for the legacy binary format; this is a
deliberate, logged gap (XLS_IMAGES_SKIPPED), not a silent omission.
"""
import datetime
from pathlib import Path

import xlrd

from .common import ImageWriter, ParsedBlock, ParsedDocument

_EMPTY, _TEXT, _NUMBER, _DATE, _BOOLEAN, _ERROR = range(6)


def _cell_str(sheet: "xlrd.sheet.Sheet", book: "xlrd.book.Book", r: int, c: int) -> str:
    ctype = sheet.cell_type(r, c)
    value = sheet.cell_value(r, c)
    if ctype == _EMPTY:
        return ""
    if ctype == _NUMBER:
        return str(int(value)) if float(value).is_integer() else str(value)
    if ctype == _DATE:
        try:
            dt = xlrd.xldate.xldate_as_datetime(value, book.datemode)
            return dt.date().isoformat() if dt.time() == datetime.time(0, 0) else dt.isoformat(sep=" ")
        except Exception:
            return str(value)
    if ctype == _BOOLEAN:
        return "TRUE" if value else "FALSE"
    if ctype == _ERROR:
        return f"#ERR:{xlrd.error_text_from_code.get(value, value)}"
    return str(value)


def parse(path: Path, image_writer: ImageWriter) -> ParsedDocument:
    book = xlrd.open_workbook(str(path))
    parsed = ParsedDocument(source_file=path.name, doc_type="xls", units_total=book.nsheets)
    parsed.metadata = {"sheets": book.nsheets, "images_skipped": "XLS_IMAGES_SKIPPED"}

    for sheet in book.sheets():
        parsed.blocks.append(ParsedBlock(kind="heading", content=f"Лист: {sheet.name}", level=2))
        rows = [[_cell_str(sheet, book, r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
        if rows:
            parsed.blocks.append(ParsedBlock(kind="table", content=rows))

    return parsed
