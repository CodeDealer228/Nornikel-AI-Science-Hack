# tests/ — pytest по всем слоям

Юнит-тесты по каждому слою пайплайна. Большинство — чистые, без
Neo4j и без реальных вызовов LLM (моки/фикстуры). Запуск:

```bash
python -m pytest tests/ -q        # 88 passed
```

## Покрытие

| Тест | Слой | Что проверяет |
|---|---|---|
| `test_chunker.py` | `chunking/` | алгоритм границ чанков через stub-сегментатор (без Natasha): разрывы по предложениям, цельные таблицы, перекрытие, `char_start`/`char_end`, `strip_front_matter` |
| `test_ontology.py` | `ontology/` | у каждого `EntityType` есть `EntityLabel` с тем же значением; `label_for`; `GeographyKind`/`NumericOperator` |
| `test_llm_parser.py` | `llm_pipeline_fewshot/` | парсинг JSON-ответа, валидация сущностей/отношений, `MockLLMClient`, фабрика `create_llm_client` (без реального API) |
| `test_config_and_logging.py` | `config.py`, `logging_setup.py` | настройки из env, `get_settings`/`reset_settings_cache`, structured logging |
| `test_golden_set_validator.py` | `golden_set/` | `validate_sample`: корректный сэмпл проходит, висячие `local_id`/неточные цитаты ловятся |
| `test_ensemble_routing_graph_quality.py` | `ensemble/` + `routing/` + `graph_reasoning/` + `quality_control/` | сквозная связка: ансамбль сливает одну сущность из двух источников; extraction-роутер выбирает `ENSEMBLE`; `GraphReasoner` детектит `contradicts` + low-confidence gap; `FactQualityController` ловит `missing_relation_endpoint` |
| `test_query_router.py` | `routing/` (query-time) | `QueryRouter`/`GraphCoverageAnalyzer`/`QueryEntityExtractor`: правила NO_DATA/RAG_ONLY/GRAPH_ONLY/HYBRID, маркеры geo/numeric/comparison |
| `test_dispatcher.py` | `agent/` | `Dispatcher`: ветки GRAPH_ONLY/RAG_ONLY/HYBRID/NO_DATA, фильтры в RAG, fallback без драйвера |
| `test_synthesizer.py` | `agent/` | `AnswerSynthesizer`: fallback-рендер без LLM-клиента, сборка user_prompt с контекстом/пробелами/RAG |
| `test_rag_factory.py` | `agent/rag_factory.py` | plug-in интерфейс: `register_rag_backend`, `build_rag_client` по `RAG_BACKEND`, откат на `StubRAGClient` |
| `test_api.py` | `api/` | FastAPI в офлайн-режиме через `TestClient(app)` (context manager прогоняет async-lifespan): `/health`, `/route`, `/query`, `/metrics` — 200; geo-comparison запрос даёт `markers.geography=true` |

## Заметки

- `test_api.py` skip'ается целиком, если `fastapi` не установлен —
  раньше это маскировало баг в `/query` (см. `../DECISIONS.md`,
  запись от 2026-07-04). Сейчас fastapi установлен, все 8
  API-тестов реально выполняются.
- Тесты, использующие Natasha (`test_chunker` идёт через stub, а
  query-роутер — через реальную `get_pipeline`), скачивают модели
  navec/slovnet один раз; дальше работают офлайн из кэша.
