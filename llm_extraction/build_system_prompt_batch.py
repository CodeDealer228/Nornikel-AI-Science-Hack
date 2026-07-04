"""Batch variant of build_system_prompt.py.

Same source of truth (../llm_pipeline_fewshot/ner_re_extraction_prompt.md)
and same 10 hand-curated few-shot pairs, but:
  1. the single-chunk JSON contract is replaced by a batch contract (input =
     several chunks tagged by chunk_id, output = {"results": [...]} with one
     entry per input chunk_id, in the same order);
  2. the 10 example pairs are regrouped into two batch demonstrations (5+5)
     showing the model the "several chunks in -> several results out" pattern,
     instead of ten separate single-chunk demonstrations.

Rationale: repackaging the already-validated 10 examples costs nothing in
annotation quality (same entities/relations JSON, just re-wrapped) while
teaching the batch shape explicitly, per mass_extraction_pipeline.md section 4.
"""
import re
from pathlib import Path

PROMPT_MD = (
    Path(__file__).resolve().parent.parent
    / "llm_pipeline_fewshot" / "ner_re_extraction_prompt.md"
)

SINGLE_CONTRACT_RE = re.compile(
    r"## КОНТРАКТ ВЫХОДНЫХ ДАННЫХ.*?(?=\n## ЧТО ЗНАЧИТ КАЖДЫЙ ТИП СУЩНОСТИ)",
    re.DOTALL,
)

BATCH_CONTRACT = """## КОНТРАКТ ВХОДНЫХ И ВЫХОДНЫХ ДАННЫХ (BATCH-РЕЖИМ)

На вход тебе подаётся НЕСКОЛЬКО чанков за один вызов (до 40), каждый со своим
уникальным chunk_id, в формате:

### ЧАНК {chunk_id}
\"\"\"
{текст чанка}
\"\"\"

(и так далее для каждого чанка пачки, в порядке их следования).

Ты обрабатываешь КАЖДЫЙ чанк НЕЗАВИСИМО от остальных: сущности и связи одного
чанка никогда не пересекаются и не ссылаются на сущности другого чанка пачки,
даже если по смыслу текста это один и тот же реальный объект (слияние
одинаковых сущностей между разными чанками — отдельный этап после этого
вызова, не твоя задача сейчас).

Верни ОДИН JSON-объект и ничего больше: без markdown-разметки, без ```json, без
пояснений до или после. Строго такая структура:

{
  "results": [
    {
      "chunk_id": "id чанка ТОЧНО как во входном блоке ЧАНК",
      "entities": [
        {
          "local_id": "e1",
          "type": "Material | Process | Equipment | Property | Experiment | Publication | Expert | Facility",
          "canonical_name": "нормализованная (лемматизированная) форма термина, слова только из текста",
          "mentions": ["точная подстрока текста №1", "точная подстрока текста №2 (если это тот же реальный объект под другим именем/аббревиатурой)"],
          "attributes": { }
        }
      ],
      "relations": [
        {
          "subject": "local_id сущности-источника ИЗ ЭТОГО ЖЕ chunk_id",
          "predicate": "uses_material | operates_at_condition | produces_output | described_in | validated_by | contradicts | affiliated_with | authored_by",
          "object": "local_id сущности-цели ИЗ ЭТОГО ЖЕ chunk_id",
          "note": "короткое пояснение связи ТОЛЬКО для predicate=contradicts, опционально; для остальных predicate поле не нужно"
        }
      ]
    }
  ]
}

## ЖЁСТКИЕ ПРАВИЛА BATCH-РЕЖИМА (нарушение = брак)

- Массив "results" должен содержать РОВНО ОДНУ запись на КАЖДЫЙ входной chunk_id,
  в ТОМ ЖЕ порядке, в котором чанки шли во входном сообщении. Не пропускай
  чанки, даже если в них не нашлось ни одной сущности — в этом случае верни
  "entities": [] и "relations": [] для этого chunk_id, а не пропускай запись.
  Не добавляй chunk_id, которых не было во входном сообщении.
- "local_id" (e1, e2, ...) уникальны ТОЛЬКО В ПРЕДЕЛАХ своего chunk_id. Двум
  разным чанкам можно и нужно независимо использовать одинаковые local_id
  (e1, e2...) — это не конфликт, т.к. они живут в разных "results"-записях.
  "subject"/"object" в relations одного чанка ссылаются ТОЛЬКО на local_id
  сущностей этого же чанка (никогда — соседнего).
- Правила для полей mentions и attributes — те же, что в однчанковом режиме
  (см. ниже определения типов), просто применяются к каждому чанку отдельно.

### Правила для поля mentions (механизм синонимов)

- Первый элемент массива mentions — это ВСЕГДА точная подстрока текста при первом
  упоминании сущности (copy-paste, с сохранением регистра и пунктуации оригинала).
- Если тот же самый реальный объект в этом же чанке упоминается ещё раз под другим
  именем, сокращением или аббревиатурой — И ты уверен, что это одна и та же сущность —
  добавь эту вторую форму ВТОРЫМ элементом массива mentions той же сущности. НЕ создавай
  для неё отдельный entity, и НЕ пиши "название (синоним)" внутри одной строки —
  каждый элемент mentions отдельный, чистый, без скобок с досочинённым текстом.
- Если ты НЕ уверен, что два упоминания — один и тот же объект, создавай ДВЕ отдельные
  сущности. Лучше не смержить два разных объекта, чем ошибочно смержить.
- Не выдумывай mentions, которых нет в тексте "для полноты" — только то, что реально
  встретилось.

### Правила для поля attributes (только для Property, опционально для Experiment)

Если сущность типа Property содержит числовое значение, обязательно заполни:
{
  "value_raw": "числовое выражение ТОЧНО как в тексте, включая знаки ≤/≥/± и текстовые
                 разделители диапазона, напр. 'S ≤ 0,05 %' или '20–30 мг/л'",
  "operator": "<= | >= | = | range | null",
  "min": число или null,
  "max": число или null,
  "unit": "единица измерения ТОЧНО как написана в тексте, напр. '%', 'г/дм3', '°C', 'мг/л', 'А/м2', 'кДж/моль', 'к.о.', 'об/мин'"
}
Если единица измерения в тексте не указана явно — unit: null, не угадывай. Если это
диапазон "20–30" — operator: "range", min: 20, max: 30. Если "≤300" — operator: "<=",
min: null, max: 300. Для остальных типов сущностей (Material, Process, Equipment,
Experiment, Publication, Expert, Facility) оставляй attributes как пустой объект {},
если в тексте нет явных числовых характеристик, которые нужно сохранить отдельно.

### Правила для relations

- subject и object — ЛОКАЛЬНЫЕ id (local_id) сущностей ИЗ ТОГО ЖЕ ЧАНКА (того же
  chunk_id). Никогда не ссылайся на сущность из другого чанка пачки или на
  сущность, которой нет в массиве entities этой же записи results.
- predicate строго один из перечисленных 8 значений (последние два — affiliated_with
  и authored_by — расширение сверх базовой онтологии задания, используй их ТОЛЬКО когда
  текст явно указывает аффилиацию автора с организацией или авторство публикации).
- Извлекай relation, только если связь ЯВНО читается из текста чанка, а не подразумевается
  общими знаниями о металлургии.
- contradicts используй ТОЛЬКО если в САМОМ ЭТОМ чанке одновременно присутствуют два
  факта (обычно два значения одной величины, или два вывода/рекомендации), которые
  прямо противоречат друг другу. Если текст просто упоминает один метод или одно
  значение без явного столкновения с другим — не используй contradicts.
- Если в чанке нет явных связей между сущностями — верни пустой массив relations: [].
"""


def _extract_examples(content: str) -> list[tuple[str, str, str]]:
    """Returns list of (n, input_text, output_json) for the 10 single-chunk examples."""
    examples_section_match = re.search(r"## 2\. FEW-SHOT.*?(?=\n## 3\.)", content, re.DOTALL)
    if not examples_section_match:
        raise ValueError("could not find FEW-SHOT section")
    examples_section = examples_section_match.group(0)

    pattern = re.compile(
        r"### ПРИМЕР (\d+) — ВХОД.*?```\n(.*?)\n```\s*"
        r"### ПРИМЕР \1 — ВЫХОД\s*```json\n(.*?)\n```",
        re.DOTALL,
    )
    out = [(m.group(1), m.group(2).strip(), m.group(3).strip()) for m in pattern.finditer(examples_section)]
    if len(out) != 10:
        raise ValueError(f"expected 10 few-shot examples, found {len(out)}")
    return out


def _wrap_batch_demo(label: str, members: list[tuple[str, str, str]]) -> str:
    import json as _json

    chunk_ids = [f"demo_{label}_{n}" for n, _, _ in members]
    input_lines = []
    for cid, (n, inp, _out) in zip(chunk_ids, members):
        input_lines.append(f'### ЧАНК {cid}\n"""\n{inp}\n"""')
    input_block = "\n\n".join(input_lines)

    results = []
    for cid, (n, _inp, out) in zip(chunk_ids, members):
        parsed = _json.loads(out)
        results.append({
            "chunk_id": cid,
            "entities": parsed.get("entities", []),
            "relations": parsed.get("relations", []),
        })
    output_block = _json.dumps({"results": results}, ensure_ascii=False, indent=2)

    return (
        f"### ПРИМЕР BATCH {label} — ВХОД (пачка из {len(members)} чанков)\n{input_block}\n\n"
        f"### ПРИМЕР BATCH {label} — ВЫХОД\n```json\n{output_block}\n```"
    )


def build_system_prompt_batch() -> str:
    content = PROMPT_MD.read_text(encoding="utf-8")

    instr_match = re.search(r"## 1\. SYSTEM PROMPT.*?```\n(.*?)\n```", content, re.DOTALL)
    if not instr_match:
        raise ValueError("could not find SYSTEM PROMPT fenced block")
    instruction = instr_match.group(1).strip()

    intro_old = (
        "Ты — экстрактор структурированных знаний для графа знаний горно-металлургической\n"
        "R&D-отрасли. На вход тебе подаётся один фрагмент текста (чанк) научной статьи, отчёта\n"
        "или патента на русском или английском языке. Твоя задача — извлечь именованные сущности\n"
        "и связи между ними СТРОГО по заданной онтологии и вернуть их в виде ОДНОГО JSON-объекта."
    )
    intro_new = (
        "Ты — экстрактор структурированных знаний для графа знаний горно-металлургической\n"
        "R&D-отрасли. На вход тебе подаётся ПАЧКА из нескольких фрагментов текста (чанков) научных\n"
        "статей, отчётов или патентов на русском или английском языке — до 40 чанков за один вызов.\n"
        "Твоя задача — для КАЖДОГО чанка пачки независимо извлечь именованные сущности и связи между\n"
        "ними СТРОГО по заданной онтологии и вернуть результаты по всем чанкам в виде ОДНОГО\n"
        "JSON-объекта (см. контракт ниже)."
    )
    if intro_old not in instruction:
        raise ValueError("could not find single-chunk intro paragraph to replace")
    instruction = instruction.replace(intro_old, intro_new)

    if not SINGLE_CONTRACT_RE.search(instruction):
        raise ValueError("could not find single-chunk КОНТРАКТ ВЫХОДНЫХ ДАННЫХ block to replace")
    instruction = SINGLE_CONTRACT_RE.sub(BATCH_CONTRACT.strip() + "\n\n", instruction)

    examples = _extract_examples(content)
    demo_a = _wrap_batch_demo("A", examples[:5])
    demo_b = _wrap_batch_demo("B", examples[5:])

    intro = (
        "Далее — 2 примера обработки ПАЧКИ из нескольких чанков за один вызов (демонстрация "
        "batch-режима на реальных абзацах корпуса, ранее провалидированных в одночанковом виде). "
        "Изучи их внимательно: они показывают, что на вход приходит несколько блоков ЧАНК, а на "
        "выход — один JSON с массивом results, по одной записи на каждый чанк, в том же порядке, "
        "с независимой нумерацией local_id внутри каждой записи."
    )

    return instruction + "\n\n" + intro + "\n\n" + demo_a + "\n\n" + demo_b


if __name__ == "__main__":
    prompt = build_system_prompt_batch()
    out_path = Path(__file__).resolve().parent / "system_prompt_batch_built.txt"
    out_path.write_text(prompt, encoding="utf-8")
    print(f"OK: {len(prompt)} chars, ~{len(prompt) // 3} tokens (rough), written to {out_path}")
