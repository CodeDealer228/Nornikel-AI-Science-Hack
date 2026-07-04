# ensemble/ — Этап 2: Слияние Natasha + LLM

Сливает независимые выходы Natasha и LLM-пайплайна
(`../chunking/natasha_pipeline.py` и `../llm_pipeline_fewshot/`) в
один граф-готовый набор сущностей/отношений, сохраняя провенанс и
флаги на ревью. Встроен в `../scripts/ingest.py` (`ensemble_chunk`).

## Что делает `EnsembleMerger`

- **Группировка по каноническому ключу.** Сущности группируются по
  `(type, canonicalize_text(name))` (тот же канонизатор, что в
  `../synonym_normalization/`), отношения — по
  `(source_id, relation_type, target_id)` с переводом `local_id`
  через `entity_id_map`. Поверхностные формы `ПВК` и
  `печь Ванюкова конвертерная` сходятся в одну группу.
- **Взвешенная уверенность.** LLM по умолчанию весит 0.7, Natasha —
  0.3; при совпадении из нескольких источников даётся
  `multi_source_bonus` (+0.08). Базовые `confidence` берутся из
  выходов экстракторов.
- **Выбор канонического имени и цитаты** — по `(confidence, длина)`.
  Алиасы (mentions) сливаются из всех участников группы без потерь.
- **`EnsembleDecision` на каждую сущность/отношение** — фиксирует
  источники (`natasha`/`yandex_llm`/`ensemble`), причину слияния
  (`MergeReason`) и флаг `needs_review`. Ревью поднимается, когда
  факт пришёл из одного источника с низкой уверенностью или когда
  в группе есть конфликт типов сущности.

## Файлы

| Файл | Ответственность |
|---|---|
| `merger.py` | `EnsembleMerger` — `merge`/`merge_entities`/`merge_relations`, агрегация confidence, выбор канонического имени/цитаты. |
| `models.py` | `EvidenceSource`, `MergeReason`, `EnsembleDecision`, `EnsembleResult`. |

## Не модифицирует

`EnsembleMerger` не правит выходы `../routing/` — он работает на
этапе извлечения (ingest-time), а роутер на этапе запроса
(query-time). На выходы ансамбля роутер не опирается (см.
`../routing/README.md`).
