import re

_MULTIPLE_SPACES_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s-]")


def canonicalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, trim spaces, remove basic punctuation."""
    if not text:
        return ""

    text = text.lower()
    text = _PUNCTUATION_RE.sub("", text)
    text = _MULTIPLE_SPACES_RE.sub(" ", text).strip()
    return text
