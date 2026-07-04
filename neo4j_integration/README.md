# neo4j_integration/ — Этап 2: Загрузчик графа

Асинхронная загрузка извлечённых сущностей/отношений в Neo4j по
схеме с per-type лейблами и первоклассными узлами
geography/year/numeric. Потребляется `../scripts/ingest.py`
(`load_to_neo4j`) и `../api/server.py` (через `Neo4jConfig`).

## Схема

Каждый `EntityType` (Material, Process, …) отображается в свой
собственный Neo4j-лейбл — запрос `MATCH (m:Material) RETURN m`
быстрый и типобезопасный, без property-фильтра. Маппинг держится в
`../ontology/` (`label_for`).

Вспомогательные узлы:
- `:Document` — один на исходный документ
- `:Chunk` — один на текстовый чанк
- `:Alias` — известные альтернативные имена (синонимы)
- `:Geography` — географический якорь (Russia, Worldwide, …)
- `:Year` — временной якорь (год публикации)
- `:NumericValue` — числовое измерение (с min/max/unit/operator)

Отношения:
- `(doc:Document)-[:HAS_CHUNK]->(c:Chunk)`
- `(c:Chunk)-[:MENTIONS]->(e:Entity)`
- `(e:Entity)-[:KNOWN_AS]->(a:Alias)`
- `(c:Chunk)-[:SUPPORTS]->(r:REL_TYPE)` — evidence для отношений
- `(e:Entity)-[:HAS_GEOGRAPHY]->(g:Geography)`
- `(e:Entity)-[:PUBLISHED_IN_YEAR]->(y:Year)`
- `(e:Entity)-[:HAS_NUMERIC_VALUE]->(n:NumericValue)`
- доменные отношения (`uses_material`, …) грузятся как собственный
  тип: `(s:Material)-[:USES_MATERIAL]->(t:Process)`

## `Neo4jLoader`

- `setup_constraints` — уникальные констрейнты на `name` каждого
  лейбла + на `Document.id`/`Chunk.id`/`Alias.name`/`Geography.name`/
  `Year.value`/`NumericValue.id`; индексы на `Document.source`,
  `Year.value`, `NumericValue.numeric_value`/`unit`, `Geography.kind`.
  Везде `IF NOT EXISTS` — безопасно вызывать многократно.
- `load_entities` — батчами (`batch_size`), через `apoc.merge.node`
  по `[label]`, с вытягиванием geography/year/numeric из
  `attributes` сущности в типизированные поля запроса
  (`apoc.do.when` для условного MERGE якорей). Сливает `mentions`
  в `:Alias` узлы.
- `load_relations` — группирует по `relation_type`, санитизирует тип
  под Cypher-лейбл (`_safe_cypher_rel_type`), мерджит отношение и
  связывающий `:Chunk` через `(:SUPPORTS)` с `confidence`/`quote`.
- `count_by_label` — диагностика по лейблам.

Использует `apoc.*` процедуры (`apoc.merge.node`,
`apoc.merge.relationship`, `apoc.do.when`) — на целевом инстансе
Neo4j должен быть установлен плагин APOC.

**Внимание: `../docker-compose.yml` сейчас собирает образ с
`NEO4J_PLUGINS: "[]"` (без плагинов).** В таком виде реальный
`python -m scripts.ingest` (без `--skip-neo4j`) упадёт на первой же
apoc-процедуре. Чтобы прогнать загрузку в граф, добавьте APOC в
`NEO4J_PLUGINS` (например `NEO4J_PLUGINS: "[\"apoc\"]"`) и при
необходимости `NEO4J_dbms_security_procedures_unrestricted=apoc.*`.
Это известный, ещё не закрытый пробел между загрузчиком и
docker-конфигом.

## Файлы

| Файл | Ответственность |
|---|---|
| `neo4j_loader.py` | `Neo4jLoader` — схема, загрузка сущностей/отношений, `count_by_label`. Ленивый импорт `neo4j` в `__init__` — пакет грузится и без драйвера, реальная загрузка упадёт явно. |
| `neo4j_config.py` | `Neo4jConfig` (`uri`/`user`/`password`/`batch_size`) из env `NEO4J_*`. |

## Статус

Загрузчик и схема готовы. **Нет запущенного инстанса Neo4j** —
проверялся только offline-путь (`--skip-neo4j`). Поднять локально:
`docker-compose up neo4j` (см. `../docker-compose.yml`), затем
`python -m scripts.ingest` без `--skip-neo4j`.
