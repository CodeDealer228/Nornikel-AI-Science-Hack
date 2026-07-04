# api/ — Этап 3: REST API (FastAPI)

Выставляет граф знаний наружу как REST API поверх дисптечера из
`../agent/`. Один модуль `server.py`, поднимается uvicorn'ом.

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/query` | полный пайплайн: routing + dispatch + синтез |
| `POST` | `/route` | только решение роутера (без графа/RAG/синтеза) |
| `GET` | `/health` | liveness-проба (всегда 200, если процесс жив) |
| `GET` | `/ready` | readiness-проба: проверяет связность с Neo4j и наличие синтезатора |
| `GET` | `/stats` | статистика графа (требует Neo4j) |
| `GET` | `/entities` | листинг сущностей по name/type/geography/year/numeric |
| `GET` | `/metrics` | Prometheus-style текстовые метрики (без auth) |

`/query` и `/route` возвращают `request_id` (uuid) и пробрасывают
его в логирование через `set_request_id`.

## Auth (опционально)

Если задана переменная окружения `API_KEY`, защищённые эндпоинты
требуют заголовок `X-API-Key: <значение>`. Пустая `API_KEY` — auth
выключен. `/health` и `/metrics` auth не требуют.

## Режимы работы

- **Offline** (по умолчанию, если Neo4j не поднята): дисптечер
  строится без graph-экстрактора; каждый запрос уходит в `RAG_ONLY`,
  RAG — `StubRAGClient` (пустой ответ с маркером). API всё равно
  отвечает 200 на `/health`, `/route`, `/query` — это и проверяется
  тестами `../tests/test_api.py`.
- **Online**: при наличии `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`
  lifespan открывает `AsyncDriver`, строит дисптечера с
  `Neo4jSubgraphExtractor` и `AnswerSynthesizer` (если настроен
  YandexGPT). `/ready` возвращает `neo4j_connected=true`.

RAG-клиент строится через `agent.rag_factory.build_rag_client()`
(учитывает `RAG_BACKEND` и зарегистрированные entry-points), с
откатом на `StubRAGClient` при ошибке.

## Запуск

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8080
```

CORS включается переменной `API_ENABLE_CORS` (по умолчанию `true`).

## Тесты

`tests/test_api.py` гоняет эндпоинты в офлайн-режиме через
`TestClient(app)` как context manager — это корректно прогоняет
async-lifespan в том же loop'е, что и запросы, и биндит
`STATE.driver`/`STATE.dispatcher`.
