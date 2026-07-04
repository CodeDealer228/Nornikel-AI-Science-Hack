# ontology/ — Этап 2: Онтология предметной области

Источник истины для типов графа. Один модуль `ontology_types.py`,
без зависимостей — используется NER/RE-промптом
(`../llm_pipeline_fewshot/`), пайплайном нормализации
(`../synonym_normalization/`), загрузчиком Neo4j
(`../neo4j_integration/`) и Cypher-хелперами
(`../graph_reasoning/cypher_helpers.py`).

## Что определяет

- **`EntityType`** — концептуальные типы узла (Material, Process,
  Equipment, Property, Experiment, Publication, Expert, Facility, …
  + якоря Geography/Year/NumericValue). Используются в Python-типизации
  по всему пайплайну и в промпте LLM.
- **`RelationType`** — концептуальные типы отношений (uses_material,
  operates_at_condition, produces_output, described_in,
  validated_by, contradicts, … + has_geography/published_in_year/
  has_numeric_value).
- **`EntityLabel`** — Neo4j-лейбл узла. **Один `EntityType` → один
  `EntityLabel`**, чтобы граф имел per-type лейблы, а не плоский
  `(:Entity {type})`. По соглашению значение лейбла совпадает со
  значением типа.
- **`NodeLabel`** — вспомогательные лейблы (Document, Chunk, Alias).
- **`RelationshipLabel`** — вспомогательные типы рёбер (MENTIONS,
  HAS_CHUNK, KNOWN_AS, SUPPORTS, HAS_GEOGRAPHY, …).
- **`NumericOperator`** — словарь операторов числовых ограничений
  (`<=`, `>=`, `=`, `<`, `>`, `range`) для Cypher/хелперов.
- **`GeographyKind`** — словарь географии для фильтра
  «отечественная vs мировая практика» (Russia, CIS, Europe, …,
  Worldwide, Unknown) — прямо по требованию задания.

## Добавление нового типа

1. Добавить значение в `EntityType` (StrEnum).
2. Добавить то же значение в `EntityLabel` (лейбл совпадает с типом
   по соглашению).
3. Опционально — соответствующий `RelationType`.
4. Перезапустить ингест — загрузчик сам создаст новый лейбл и
   индексы (`setup_constraints` проходит по всем `EntityLabel`).

`label_for(entity_type)` — быстрый lookup `EntityType → EntityLabel`,
используется в загрузчике и Cypher-хелперах.

## Расширяемость

Расширяемость онтологии на смежные домены проверена ручной разметкой:
статья по горной вентиляции (`../ner_re_extraction/`, документ 7)
использует те же 8 типов сущностей / 6 типов отношений, что и
металлургия — меняется только словарь, а не схема.
