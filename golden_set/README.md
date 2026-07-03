# Golden Set

`golden_set/` хранит машинно-читаемый эталон ручной разметки чанков для R&D-карты знаний Норникеля. Этот эталон нужен для оценки extraction-пайплайна: LLM extraction, Natasha-based extraction и возможного ансамбля.

Основной файл эталонной разметки: `golden_samples.jsonl`. Черновые файлы вроде `golden_samples_draft.jsonl` можно оставлять рядом, но валидировать и использовать в метриках нужно основной файл.

## Формат

Каждая строка JSONL - один sample, привязанный к реальному `chunk_id` из chunking-пайплайна.

Обязательные поля sample:

- `sample_id` - id размеченного примера.
- `chunk_id` - id чанка.
- `document_id` - id исходного документа.
- `source_path` - путь к источнику.
- `text` - полный текст чанка.
- `entities` - список сущностей.
- `relations` - список связей.

Entity:

- `id` - уникальный id сущности внутри sample.
- `type` - тип из разрешенного списка.
- `canonical_name` - нормализованное имя сущности.
- `mentions` - реальные формы из текста чанка.
- `attributes` - дополнительные свойства, если нужны.

Mention:

- `text` - точная подстрока `sample.text`.
- `start`, `end` - optional offsets; если указаны, `sample.text[start:end]` должен совпадать с `text`.

Relation:

- `id` - id связи.
- `subject` - id entity-источника.
- `predicate` - тип связи из разрешенного списка.
- `object` - id entity-цели.
- `evidence_text` - точная цитата из `sample.text`, подтверждающая связь.
- `evidence_start`, `evidence_end` - optional offsets для evidence.
- `attributes` - дополнительные свойства, если нужны.

## Валидация

Запуск:

```bash
python -m golden_set.validate golden_set/golden_samples.jsonl
```

При успехе валидатор печатает:

```text
Golden set is valid
```

Код выхода `0` означает успех, `1` - найдены ошибки.

## Entity Types

- Material
- Substance
- Process
- Equipment
- Property
- Parameter
- Condition
- Experiment
- Publication
- TechnologySolution
- Result
- Conclusion
- Limitation
- Facility
- Organization
- Expert

## Relation Types

- has_subprocess
- replaced_by
- affects_property
- has_limitation
- has_quality_requirement
- has_distribution_requirement
- has_expected_result
- based_on
- applies_to
- uses_technology
- produces_output
- requires_expertise
- uses_equipment
- studies
- fed_through
- depends_on
- operates_at_condition
- measured_property
- operates_between
- validated_by
- performed_by
- uses_material
- described_in
- supported_by
- authored_by
- affiliated_with
- used_in_facility
- contradicts

`used_in_facility` — технологическое решение или процесс применяется/используется на конкретной площадке, фабрике, цехе, лаборатории или другом объекте Facility.

## Правила Разметки

Traceability: каждая сущность и каждая связь должны подтверждаться текстом внутри chunk.

Evidence: нельзя добавлять связь, если нет точной цитаты `evidence_text` из `text`.

Canonical name: `canonical_name` хранит нормализованное имя, а `mentions` хранят реальные формы из текста.

No hallucinations: нельзя добавлять сущность или связь по знаниям из головы, если ее нет в тексте чанка.

## Как Добавлять Samples

1. Выбрать реальный chunk из `chunks.jsonl`.
2. Скопировать его `chunk_id`, `source_path`, `document_id` и полный `text`.
3. Вручную добавить только те entities и relations, которые подтверждаются текстом.
4. Для mentions и evidence использовать точные подстроки из `text`.
5. Запустить валидатор.
6. Исправить только структурные ошибки, offsets или неточные цитаты, не добавляя факты вне текста.
