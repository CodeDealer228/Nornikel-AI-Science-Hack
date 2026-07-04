# quality_control/ — Этап 2: Контроль качества фактов

Инспектирует извлечённые сущности/отношения на чанке и поднимает
неблокирующие `QualityIssue` — низкая уверенность, висячие концы
отношений, явные и семантические противоречия, разреженный граф.
Не валидирует онтологию (это делает `../golden_set/`) и не
выбрасывает факты — только маркирует.

## Что делает `FactQualityController`

`inspect(entities, relations)` собирает `QualityReport` из четырёх
детекторов:

| Детектор | Код issue | Severity | Срабатывает когда |
|---|---|---|---|
| `detect_low_confidence` | `low_confidence_entity` / `low_confidence_relation` | warning | `confidence < 0.45` (порог конструктора) |
| `detect_relation_endpoint_gaps` | `missing_relation_endpoint` | **error** | `source_local_id`/`target_local_id` отношения отсутствует в множестве сущностей чанка |
| `detect_extracted_contradictions` | `explicit_contradiction`, `possible_semantic_contradiction` | warning | явное `contradicts` либо пара `has_limitation`+`has_expected_result` на одной паре узлов |
| `detect_sparse_graph_signals` | `no_entities`, `entities_without_relations` | warning | пустой извлечение или сущности без связей |

`QualityReport.has_errors` — True, если есть хоть один issue с
`severity="error"` (т.е. `missing_relation_endpoint`).

## Калибровка confidence

`calibrate_entity_confidence` / `calibrate_relation_confidence` —
пересчёт уверенности с учётом источника (бонус за `ensemble`),
поддержки несколькими источниками, наличия цитаты и штрафа за
`needs_review`. На данный момент эти методы ничем в репозитории не
вызываются (ни в ингест-пайплайне, ни в тестах) — это подготовленный,
но ещё не подключённый механизм калибровки.

## Файлы

| Файл | Ответственность |
|---|---|
| `quality_controller.py` | `FactQualityController` — детекторы + калибровка. |
| `models.py` | `QualityIssue` (frozen), `QualityReport` (`has_errors`). |

## Связь с другими слоями

Детекторы противоречий здесь и в `../graph_reasoning/reasoner.py`
сознательно дублируют логику `has_limitation`/`has_expected_result`:
этот пакет работает на этапе извлечения (по чанку, до загрузки в
граф), а `GraphReasoner` — на этапе запроса (по подграфу). Состав
оппозиций держится согласованным вручную.
