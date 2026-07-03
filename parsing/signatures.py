"""
Step-0 gate: verify a file's real content matches its extension before any
format-specific parser touches it. Found empirically necessary — one file in
this corpus (`MBR_Cu-forecast_23feb10.xls`) is a BMP image mislabeled `.xls`;
a full corpus-wide scan found no other mismatches, but the check stays on
for every file since a silent mis-dispatch (wrong parser on wrong bytes) is
exactly the class of error that's easy to miss without it.
"""
from pathlib import Path
from typing import Optional

from .config import IMAGE_SIGNATURES, SIGNATURES

_HTML_MARKERS = (b"<!doctype", b"<html", b"<?xml")


class SignatureResult:
    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind      # "match" | "image" | "html" | "unknown"
        self.detail = detail  # e.g. detected image extension, or raw header repr

    def __repr__(self):
        return f"SignatureResult({self.kind!r}, {self.detail!r})"


def check_signature(path: Path) -> SignatureResult:
    ext = path.suffix.lower()
    expected = SIGNATURES.get(ext)
    with open(path, "rb") as f:
        head = f.read(16)

    if expected is not None and head.startswith(expected):
        return SignatureResult("match")

    for sig, kind in IMAGE_SIGNATURES.items():
        if head.startswith(sig):
            return SignatureResult("image", kind)

    lowered = head.lower()
    if any(lowered.startswith(m) or m in lowered for m in _HTML_MARKERS) or head[:1] == b"<":
        return SignatureResult("html")

    return SignatureResult("unknown", repr(head))
