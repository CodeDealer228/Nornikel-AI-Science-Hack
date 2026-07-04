# agent/ — Этап 3: Исполнение запроса и синтез ответа

Финальный слой пайплайна. Принимает естественноязыковой запрос,
маршрутизирует его (через `../routing/`), исполняет выбранную ветку
(граф и/или RAG) и синтезирует человекочитаемый ответ. Это то, что
вызывают CLI, REST API (`../api/`) и фронтенд (`../frontend.py`).

## Поток

```
user query
   │
   ▼
Dispatcher.dispatch()
   │  ├── QueryRouter.route()               → QueryRoutingDecision (route, signals)
   │  │      (GraphCoverageReport из routing/)
   │  │
   │  ├── route == GRAPH_ONLY  → _graph_query()   (Neo4jSubgraphExtractor + GraphReasoner)
   │  ├── route == RAG_ONLY    → _rag_query()     (RAGClient, фильтры из decision)
   │  ├── route == HYBRID      → оба параллельно (asyncio.gather)
   │  └── route == NO_DATA     → ничего не вызывает
   │
   ▼
AnswerSynthesizer.synthesize()   → SynthesisResult (NL-ответ или fallback-рендер)
   │
   ▼
DispatchResult (to_markdown / JSON)
```

Диспетчер намеренно backend-agnostic: граф подаётся как
`Neo4jSubgraphExtractor` (тот же, что в `GraphCoverageAnalyzer`),
RAG — любой объект, реализующий протокол `RAGClient`. Без драйвера
Neo4j и без RAG-бэкенда дисптечер всё ещё работает (офлайн-режим:
каждый запрос уходит в `RAG_ONLY`, `StubRAGClient` возвращает пустой
результат с маркером).

## Файлы

| Файл | Ответственность |
|---|---|
| `dispatcher.py` | `Dispatcher` + `DispatchResult`. Исполняет решение роутера, собирает граф-контекст и RAG-документы, рендерит Markdown. Сам ответ **не генерирует** — это работа синтезатора. |
| `synthesizer.py` | `AnswerSynthesizer` + `SynthesisResult`. Берёт `DispatchResult` и вызывает Yandex Foundation Models для NL-ответа. При недоступности LLM — детерминированный fallback-рендер контекста (для офлайн-демо и тестов). |
| `rag_client.py` | Протокол `RAGClient` + `StubRAGClient` + `NumericFilter`/`RAGDocument`/`RAGResult`. Реальный RAG-бэкенд подключается реализацией того же интерфейса — без правки дисптечера. |
| `rag_factory.py` | Plug-in слот для RAG-бэкенда: entry-point `kg.rag_backends` или `RAG_BACKEND=...`. По умолчанию — `stub`. |
| `cli.py` | CLI `python -m agent.cli "<запрос>"`. Флаги: `--neo4j`, `--json`, `--decision-only`, `--no-synthesis`, `-v`. Принудительно ставит UTF-8 на stdout (Windows). |

## Запуск

```bash
# офлайн (без Neo4j, без LLM — покажет routing + fallback-рендер)
python -m agent.cli "Какие методы обессоливания воды при сульфатах ≤300 мг/л?"

# только решение роутера, без исполнения и синтеза
python -m agent.cli --decision-only "никель электроэкстракция"

# полный путь с Neo4j и синтезом
python -m agent.cli --neo4j "Сравни отечественную и зарубежную практику выщелачивания никеля"

# JSON для программных потребителей
python -m agent.cli --json "покажи эксперименты по флотации"
```

## Что сознательно не сделано

- RAG-бэкенд не реализован в этом пакете — закрыт `StubRAGClient`.
  Подключение semantic/lexical индекса — через `rag_factory.py`
  (см. корневой README, раздел «С чего продолжать», п. 2).
- `Dispatcher.extract_query_filters` разбирает географию/время/числа
  эвристически (без полноценного парсера числовых диапазонов) —
  записывает, что ограничение присутствует, для пост-фильтрации
  вызывающей стороной.
