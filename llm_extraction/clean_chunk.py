"""Minimal regex cleanup applied to chunk text AFTER chunking (not before --
the chunker's separator hierarchy relies on raw "\n\n" paragraph breaks and
"\n## " headings to split on document structure, so cleaning first would
blunt those boundaries; cleaning the already-cut chunk text doesn't).

Two things only, per spec -- no broader normalization:
  1) collapse runs of the same whitespace character (newline, tab, space)
     down to a single occurrence -- NOT a general whitespace-class collapse,
     which would also merge different whitespace kinds together.
  2) drop image placeholders like "[IMAGE_0018]" (any number of digits) left
     over from parsing/run.py's markdown conversion.

char_start/char_end recorded during chunking still point into the original
(uncleaned) document text -- traceability is preserved; only the "text"
payload sent to the model is cleaned.
"""
import re

_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE_\d+\]")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")
_MULTI_TAB_RE = re.compile(r"\t{2,}")
_MULTI_SPACE_RE = re.compile(r" {2,}")


def clean_chunk_text(text: str) -> str:
    text = _IMAGE_PLACEHOLDER_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n", text)
    text = _MULTI_TAB_RE.sub("\t", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text
