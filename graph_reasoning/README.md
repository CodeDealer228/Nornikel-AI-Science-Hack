# graph_reasoning/ — Этап 2: Извлечение подграфа и рассуждения

Читает граф из Neo4j и готовит контекст для ответа: вытаскивает
подграф вокруг seed-сущностей, детектит противоречия и пробелы в
знаниях, рендерит текстовый контекст для LLM-синтеза. Потребляется
дисптечером (`../agent/dispatcher.py`) и роутером
(`../routing/graph_coverage.py`).

## Что делает

- **`Neo4jSubgraphExtractor`** — асинхронный обход графа от
  seed-сущностей на 1–4 хопа, дедупликация узлов/рёбер по `element_id`,
  сборка `GraphReasoningContext` (nodes, edges, paths). Драйвер
  принимает любой объект, совместимый с `neo4j.AsyncDriver`, сам его
  не импортирует и не строит.
- **`GraphReasoner`** — обогащает контекст:
  - `detect_contradictions` — явные `contradicts`-рёбра + пары
    `has_limitation`/`has_expected_result` и
    `replaced_by`/`uses_technology` на одной паре узлов;
  - `detect_knowledge_gaps` — изолированные узлы (без рёбер),
    низкоуверенные узлы (`< 0.45`), отсутствие seed-узлов в графе,
    опора на единственный источник (`< 2` документов).
- **`GraphContextBuilder`** — рендерит `GraphReasoningContext` в
  текст (факты с цитатами и источниками, пробелы по severity,
  противоречия). Это то, что идёт в `user_prompt` синтезатора.
- **Cypher-хелперы** (`cypher_helpers.py`) — `entities_by_name`,
  `entities_by_geography`, `entities_by_year_range`,
  `entities_by_numeric_value`, `top_related`, `graph_statistics`.
  Уважают per-type лейблы из `../ontology/` (не плоский
  `(:Entity {type})`). Используются REST API `../api/server.py`
  для эндпоинтов `/stats` и `/entities`.

## Файлы

| Файл | Ответственность |
|---|---|
| `models.py` | `GraphNode`, `GraphEdge`, `GraphPath`, `GraphGap`, `GraphReasoningContext` (frozen dataclasses). |
| `neo4j_subgraph.py` | `Neo4jSubgraphExtractor` — чтение подграфа. |
| `reasoner.py` | `GraphReasoner` — противоречия, пробелы, `build_llm_context`. |
| `context_builder.py` | `GraphContextBuilder` — текстовый рендер. |
| `cypher_helpers.py` | async Cypher-хелперы по фильтрам + `graph_statistics`. |

## Зависимости

Графовые хелперы требуют поднятый Neo4j (см. корневой README, п. 1
«С чего продолжать»). Без драйвера дисптечер передаёт
`graph_extractor=None` и `_graph_query` возвращает пустой
`GraphReasoningContext` — путь не падает, только не даёт графа.
