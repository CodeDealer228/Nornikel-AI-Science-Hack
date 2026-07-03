from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Tuple

from .config import ChunkConfig
from .segmentation import SentenceSegmenter

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

@dataclass
class _Unit:
    start: int
    stop: int
    kind: str
    heading_path: Tuple[str, ...]
    is_break: bool = False

@dataclass
class RawChunk:
    char_start: int
    content_start: int
    char_end: int
    text: str
    heading_path: List[str]
    oversize: bool = False

    @property
    def overlap_prefix_chars(self) -> int:
        return self.content_start - self.char_start

def _iter_line_spans(s: str, body_start: int) -> Iterator[Tuple[int, int, str]]:
    pos, n = body_start, len(s)
    while pos < n:
        nl = s.find("\n", pos)
        if nl == -1:
            yield pos, n, s[pos:n]
            return
        yield pos, nl + 1, s[pos:nl + 1]
        pos = nl + 1

def _extract_units(s: str, body_start: int, segmenter: SentenceSegmenter,
                   section_break_level: int) -> List[_Unit]:
    units: List[_Unit] = []
    heading_stack: List[Tuple[int, str]] = []
    spans = list(_iter_line_spans(s, body_start))
    i, n = 0, len(spans)

    while i < n:
        start, stop, line = spans[i]
        bare = line.strip("\n")
        stripped = bare.strip()

        if not stripped:
            i += 1
            continue

        m = _HEADING_RE.match(bare)
        if m:
            level, text = len(m.group(1)), m.group(2)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            hp = tuple(h for _, h in heading_stack)
            units.append(_Unit(start, stop, "heading", hp, is_break=level <= section_break_level))
            i += 1
            continue

        if _TABLE_LINE_RE.match(line):
            t_start, t_stop, j = start, stop, i + 1
            while j < n and _TABLE_LINE_RE.match(spans[j][2]):
                t_stop = spans[j][1]
                j += 1
            hp = tuple(h for _, h in heading_stack)
            units.append(_Unit(t_start, t_stop, "table", hp))
            i = j
            continue

        if stripped.startswith("[IMAGE_"):
            hp = tuple(h for _, h in heading_stack)
            units.append(_Unit(start, stop, "image", hp))
            i += 1
            continue

        p_start, p_stop, j = start, stop, i + 1
        while j < n:
            l2 = spans[j][2]
            l2_bare = l2.strip("\n")
            if (not l2_bare.strip() or _HEADING_RE.match(l2_bare)
                    or _TABLE_LINE_RE.match(l2) or l2_bare.strip().startswith("[IMAGE_")):
                break
            p_stop = spans[j][1]
            j += 1
        hp = tuple(h for _, h in heading_stack)
        para = s[p_start:p_stop]
        sents = segmenter.segment(para)
        if sents:
            for sent in sents:
                units.append(_Unit(p_start + sent.start, p_start + sent.stop, "sentence", hp))
        else:
            units.append(_Unit(p_start, p_stop, "sentence", hp))
        i = j

    return units

def _ulen(u: _Unit) -> int:
    return u.stop - u.start

def _seg_len(seg: List[_Unit]) -> int:
    return seg[-1].stop - seg[0].start

def _is_context_only(seg: List[_Unit]) -> bool:
    return all(u.kind in ("heading", "image") for u in seg)

def _merge_context_forward(segments: List[List[_Unit]], cfg: ChunkConfig) -> List[List[_Unit]]:
    out: List[List[_Unit]] = []
    carry: List[_Unit] = []
    for seg in segments:
        if _is_context_only(seg) and _seg_len(seg) < cfg.max_chunk_chars:
            carry = carry + seg
            continue
        out.append(carry + seg if carry else seg)
        carry = []
    if carry:
        out.append(carry)
    return out

def _merge_small(segments: List[List[_Unit]], cfg: ChunkConfig) -> List[List[_Unit]]:
    if not segments:
        return []
    segments = _merge_context_forward(segments, cfg)
    out: List[List[_Unit]] = [segments[0]]
    for seg in segments[1:]:
        if (_seg_len(seg) < cfg.min_chunk_chars
                and not seg[0].is_break
                and _seg_len(out[-1]) + _seg_len(seg) <= cfg.max_chunk_chars):
            out[-1] = out[-1] + seg
        else:
            out.append(seg)
    if len(out) > 1 and _seg_len(out[0]) < cfg.min_chunk_chars \
            and _seg_len(out[0]) + _seg_len(out[1]) <= cfg.max_chunk_chars \
            and not out[1][0].is_break:
        out[1] = out[0] + out[1]
        out.pop(0)
    return out

def _overlap_start(content_start: int, prev_seg: List[_Unit],
                   sentence_starts: List[int], cfg: ChunkConfig) -> int:
    lo = prev_seg[0].start
    cands = [p for p in sentence_starts if lo <= p < content_start]
    if not cands:
        return content_start
    chosen = cands[-cfg.overlap_sentences:]
    for p in chosen:
        if content_start - p <= cfg.overlap_max_chars:
            return p
    return content_start

def _pack(units: List[_Unit], s: str, cfg: ChunkConfig) -> List[RawChunk]:
    segments: List[List[_Unit]] = []
    cur: List[_Unit] = []
    cur_len = 0

    for u in units:
        if u.is_break and cur:
            segments.append(cur)
            cur, cur_len = [], 0
        if cur and cur_len + _ulen(u) > cfg.target_chunk_chars:
            segments.append(cur)
            cur, cur_len = [], 0
        cur.append(u)
        cur_len += _ulen(u)
        if cur_len >= cfg.max_chunk_chars:
            segments.append(cur)
            cur, cur_len = [], 0
    if cur:
        segments.append(cur)

    segments = _merge_small(segments, cfg)
    sentence_starts = [u.start for u in units if u.kind == "sentence"]

    raws: List[RawChunk] = []
    for idx, seg in enumerate(segments):
        content_start = seg[0].start
        content_end = seg[-1].stop
        char_start = content_start
        if idx > 0:
            char_start = _overlap_start(content_start, segments[idx - 1], sentence_starts, cfg)
        anchor = next((u for u in seg if u.kind in ("sentence", "table")), seg[0])
        raws.append(RawChunk(
            char_start=char_start,
            content_start=content_start,
            char_end=content_end,
            text=s[char_start:content_end],
            heading_path=list(anchor.heading_path),
            oversize=(content_end - content_start) > cfg.max_chunk_chars,
        ))
    return raws

def strip_front_matter(s: str) -> Tuple[int, dict]:
    m = _FRONT_MATTER_RE.match(s)
    if not m:
        return 0, {}
    meta: dict = {}
    for line in s[m.start():m.end()].splitlines():
        if line == "---" or not line.strip():
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return m.end(), meta

def build_raw_chunks(s: str, segmenter: SentenceSegmenter, cfg: ChunkConfig) -> List[RawChunk]:
    body_start, _ = strip_front_matter(s)
    units = _extract_units(s, body_start, segmenter, cfg.section_break_level)
    if not units:
        return []
    return _pack(units, s, cfg)
