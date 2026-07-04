"""Text normalization shared by every synonym/alias matching step in this
package. Kept as one function so the graph builder, the curated-dictionary
lookup, and (eventually) the search module's query expansion all treat the
same string the same way -- a mismatch here silently breaks matches.

Superset of chunking-branch's synonym_normalization/canonicalizer.py: also
folds Unicode sub/superscript digits and unifies dash variants, matching
search-module-update branch's search/synonyms.py normalize_for_match(), since
resources/synonyms.yaml was authored against that normalization.
"""
import re
import unicodedata

_MULTIPLE_SPACES_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[‐-―−]")

_SUBSCRIPT_TRANSLATION = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
})


def normalize(text: str) -> str:
    """Canonical form used as a dict/merge key: NOT for display."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_SUBSCRIPT_TRANSLATION)
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = _DASH_RE.sub("-", text)
    text = text.replace("«", '"').replace("»", '"').strip().strip('"').strip()
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    return text.lower()


def lang_of(text: str) -> str:
    if re.search(r"[а-яА-ЯёЁ]", text):
        return "ru"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "other"
