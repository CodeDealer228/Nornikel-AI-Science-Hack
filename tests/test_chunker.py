import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking.chunker import build_raw_chunks, strip_front_matter
from chunking.config import ChunkConfig

@dataclass
class _S:
    text: str
    start: int
    stop: int

class RegexSegmenter:
    _RE = re.compile(r"[^.!?…]*[.!?…]+|\S[^.!?…]*$", re.S)

    def segment(self, text):
        out = []
        for m in self._RE.finditer(text):
            seg = m.group(0)
            core = seg.rstrip()
            if core.strip():
                out.append(_S(core, m.start(), m.start() + len(core)))
        return out

def _para(n_sentences, word="слово"):
    return " ".join(f"{word}{i} " * 6 + "." for i in range(n_sentences))

def check(name, cond):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}")
    assert cond, name

seg = RegexSegmenter()

print("Test 1: offset fidelity + overlap on a multi-paragraph doc")
doc = "---\ntitle: X\nauthor: Ivanov\n---\n\n# Заголовок\n\n"
doc += _para(20) + "\n\n" + _para(20) + "\n\n" + _para(20) + "\n"
cfg = ChunkConfig(target_chunk_chars=600, max_chunk_chars=1200, min_chunk_chars=120,
                  overlap_sentences=2, overlap_max_chars=300, section_break_level=2)
chunks = build_raw_chunks(doc, seg, cfg)
check("produced >1 chunk", len(chunks) > 1)
for c in chunks:
    check(f"text matches slice [{c.char_start}:{c.char_end}]", c.text == doc[c.char_start:c.char_end])
check("all chunks non-empty", all(c.text.strip() for c in chunks))
check("overlap present on chunk#2", chunks[1].overlap_prefix_chars > 0)
check("char_start monotonic", all(chunks[i].char_start <= chunks[i + 1].char_start for i in range(len(chunks) - 1)))
bm, meta = strip_front_matter(doc)
check("front matter parsed", meta.get("author") == "Ivanov")
check("body starts after front matter", chunks[0].content_start >= bm)

print("Test 2: table atomicity")
table = "\n".join("| a%d | b%d | c%d |" % (i, i, i) for i in range(40))
doc2 = "# H\n\n" + _para(6) + "\n\n" + table + "\n\n" + _para(6) + "\n"
cfg2 = ChunkConfig(target_chunk_chars=300, max_chunk_chars=5000, min_chunk_chars=50,
                   overlap_sentences=1, overlap_max_chars=200, section_break_level=2)
chunks2 = build_raw_chunks(doc2, seg, cfg2)
containing = [c for c in chunks2 if "| a0 |" in c.text]
check("table appears in exactly one chunk", len(containing) == 1)
check("that chunk holds the WHOLE table (a0..a39)", "| a39 |" in containing[0].text)
for c in chunks2:
    check("t2 slice fidelity", c.text == doc2[c.char_start:c.char_end])

print("Test 3: section headings")
doc3 = "# Doc\n\n" + _para(4) + "\n\n## Методика\n\n" + _para(4) + "\n\n## Результаты\n\n" + _para(4) + "\n"
cfg3 = ChunkConfig(target_chunk_chars=99999, max_chunk_chars=99999, min_chunk_chars=1,
                   overlap_sentences=1, overlap_max_chars=200, section_break_level=2)
chunks3 = build_raw_chunks(doc3, seg, cfg3)
check("3 section chunks (Doc / Методика / Результаты)", len(chunks3) == 3)
paths = [tuple(c.heading_path) for c in chunks3]
check("heading_path includes Методика", any("Методика" in p for p in paths))
check("heading_path includes Результаты", any("Результаты" in p for p in paths))

print("Test 4: tiny doc")
doc4 = "Одно короткое предложение. И второе."
chunks4 = build_raw_chunks(doc4, seg, ChunkConfig())
check("single chunk", len(chunks4) == 1)
check("no overlap", chunks4[0].overlap_prefix_chars == 0)
check("slice fidelity", chunks4[0].text == doc4[chunks4[0].char_start:chunks4[0].char_end])

print("Test 5: empty doc")
check("empty -> 0 chunks", build_raw_chunks("---\na: b\n---\n\n   \n", seg, ChunkConfig()) == [])

print("\nALL TESTS PASSED")
