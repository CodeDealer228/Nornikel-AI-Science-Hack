"""Builds the full system-message string (instruction + 10 few-shot examples)
straight from the team's ner_re_extraction_prompt.md, so it stays byte-exact
with the source of truth instead of being hand-copied."""
import re
from pathlib import Path

PROMPT_MD = (
    Path(__file__).resolve().parent.parent
    / "llm_pipeline_fewshot" / "ner_re_extraction_prompt.md"
)


def build_system_prompt() -> str:
    content = PROMPT_MD.read_text(encoding="utf-8")

    instr_match = re.search(r"## 1\. SYSTEM PROMPT.*?```\n(.*?)\n```", content, re.DOTALL)
    if not instr_match:
        raise ValueError("could not find SYSTEM PROMPT fenced block")
    instruction = instr_match.group(1).strip()

    examples_section_match = re.search(r"## 2\. FEW-SHOT.*?(?=\n## 3\.)", content, re.DOTALL)
    if not examples_section_match:
        raise ValueError("could not find FEW-SHOT section")
    examples_section = examples_section_match.group(0)

    example_blocks = []
    pattern = re.compile(
        r"### ПРИМЕР (\d+) — ВХОД.*?```\n(.*?)\n```\s*"
        r"### ПРИМЕР \1 — ВЫХОД\s*```json\n(.*?)\n```",
        re.DOTALL,
    )
    for m in pattern.finditer(examples_section):
        n, inp, out = m.group(1), m.group(2).strip(), m.group(3).strip()
        example_blocks.append(
            f"### ПРИМЕР {n} — ВХОД\n{inp}\n\n### ПРИМЕР {n} — ВЫХОД\n{out}"
        )

    if len(example_blocks) != 10:
        raise ValueError(f"expected 10 few-shot examples, found {len(example_blocks)}")

    return instruction + "\n\n" + "\n\n".join(example_blocks)


if __name__ == "__main__":
    prompt = build_system_prompt()
    out_path = Path(__file__).resolve().parent / "system_prompt_built.txt"
    out_path.write_text(prompt, encoding="utf-8")
    print(f"OK: {len(prompt)} chars, ~{len(prompt) // 3} tokens (rough), written to {out_path}")
