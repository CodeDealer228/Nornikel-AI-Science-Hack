# Граф знаний R&D для Норникеля — материалы хакатона

Прочитайте сначала это — файл написан так, чтобы объяснить весь репозиторий сам по
себе, без необходимости присутствовать при его создании.

**Задача**: построить граф знаний R&D для горно-металлургической отрасли, который
связывает публикации, эксперименты, технологические решения, материалы,
оборудование, объекты/экспертов и выводы, и отвечает на многопараметрические
запросы на естественном языке (материал + процесс + условие + география +
временной диапазон) с фильтрацией по числовым диапазонам, верификацией источников
и сравнением отечественной/зарубежной практики. Полное задание:
[`hackathon_task.md`](hackathon_task.md).

## Парадигма

Всё в этом репозитории следует одному пайплайну:

```
  ETL  ─────────────────────────▶  Граф знаний + semantic/lexical индекс  ─────▶  Агент
  (скачивание + парсинг в Markdown) (NER/RE-извлечение, обработка синонимов,        (ответы на вопросы
                                     схема графа, поисковый индекс)                  на естественном языке)
```

## Текущий статус

| Этап | Папка | Статус |
|---|---|---|
| 1. Скачивание | [`elt/`](elt/) | ✅ готово — быстрое, параллельное, повторяет структуру удалённых папок |
| 1.1 Парсинг → Markdown | [`parsing/`](parsing/) | ✅ слой парсинга готов|
| 1.2 Чанкинг + Natasha | [`chunking/`](chunking/) | ✅ готово — сегментация, чанкер, Natasha-пайплайн, оркестратор |
| 2. Ручная разметка NER/RE | [`ner_re_extraction/`](ner_re_extraction/) | ✅ готово для 7 статей — эталонная разметка + банк примеров для few-shot |
| 2. Онтология | [`ontology/`](ontology/) | ✅ готово — `EntityType`, `RelationType`, метки узлов, `GeographyKind`, `NumericOperator` |
| 2. Синонимы/алиасы | [`synonym_normalization/`](synonym_normalization/) | ✅ реализовано — `canonicalizer`, `synonym_dictionary`, `normalize_pipeline`, `units_normalization`; встроено в `scripts/ingest` |
| 2. LLM-пайплайн NER/RE | [`llm_pipeline_fewshot/`](llm_pipeline_fewshot/) | ✅ код написан (`YandexGPTClient` + `MockLLMClient` + фабрика `create_llm_client` + батч-раннер); фабрика умеет `mock`/`real`/`deepseek` (`LLM_CLIENT_MODE`). DeepSeek хостится на Yandex Cloud — тот же `YandexGPTClient`/`/completion`/`Api-Key`, только `ds://` model URI; ключ от Yandex AI Studio, не DeepSeek. Промпт и формат выхода изначально под DeepSeek. **Прогон по корпусу выполнен** — 2 батча: `Статьи` 1314 чанков (1303 OK, 1177 с сущностями, 694 со связями, ср. задержка ~355 c) и `Обзоры` 1341 чанк |
| 2. Ансамбль Natasha + LLM | [`ensemble/`](ensemble/) | ✅ готово — `EnsembleMerger`; встроен в `scripts/ingest` |
| 2. Загрузка графовой БД | [`neo4j_integration/`](neo4j_integration/) | ✅ загрузчик + схема готовы |
| 2. Графовые рассуждения | [`graph_reasoning/`](graph_reasoning/) | ✅ готово — `GraphReasoner`, `Neo4jSubgraphExtractor`, `entities_by_*`, `graph_statistics` |
| 2. Semantic + lexical индекс | — | ⬜ не реализовано — RAG-слот закрыт `StubRAGClient`; есть plug-in интерфейс (`agent/rag_factory.py`) для внешнего бэкенда (Elasticsearch/эмбеддинги) через entry-point или `RAG_BACKEND` |
| 2. Маршрутизация запросов | [`routing/`](routing/) | ✅ готово — `query_router`, `query_entity_extractor`, `graph_coverage`; маркеры numeric / geography / temporal / comparison / definitional / causal |
| 3. Агент + синтез ответов | [`agent/`](agent/) | ✅ готово — `Dispatcher`, `AnswerSynthesizer`, `RAGClient`/`StubRAGClient`, `rag_factory`; CLI `python -m agent.cli` |
| 3. REST API | [`api/`](api/) | ✅ готово и запускается (FastAPI): `/query`, `/route`, `/health`, `/ready`, `/stats`, `/entities`, `/metrics`; опц. API-key; проверено live под uvicorn |
| 3. Фронтенд | [`frontend/`](frontend/) (SPA) · [`frontend.py`](frontend.py) (Streamlit) | ✅ готово — дизайнерское SPA (тёмная тема, magenta/cyan Норникеля): дашборд, интерактивный граф, семантический поиск с фильтрами, литобзор, сравнение, пробелы, аудит; работает офлайн по `data.json` и с API. Streamlit — скучная альтернатива |
| 3. Оценка (F1) | [`evaluation/`](evaluation/), [`golden_set/`](golden_set/), [`run_natasha_eval.py`](run_natasha_eval.py) | ✅ готово — golden set 57 сэмплов, sklearn-репорт по NER+RE; Natasha-only F1 низкий по ожиданию (детектит только PER/LOC/ORG → Expert/Facility/Organization) |

### Пайплайн end-to-end (проверено)

```
python -m elt.download && python -m elt.extract_archives
python -m parsing.run                     # → parsed_data/  (частично)
LLM_CLIENT_MODE=mock python -m scripts.ingest --skip-neo4j --limit 3   # smoke-test
python -m scripts.ingest --skip-neo4j     # полный прогон (Natasha + YandexGPT + ансамбль)
python run_natasha_eval.py                # F1 против golden_set
uvicorn api.server:app --port 8080        # REST API
python build_frontend_data.py             # → frontend/data.json (агрегат батча)
python -m http.server -d frontend 8050    # SPA → http://localhost:8050
streamlit run frontend.py                 # UI (альтернатива)
```

Smoke-test на Windows-консоли: 3 файла → 44 чанка → 306 сущностей (58 mock-LLM + 248 Natasha) за 3.4 c. Тесты: `python -m pytest tests/ -q` → **88 passed**.

## Карта репозитория

```
hackathon_task.md           исходное задание хакатона — прочитать перед любым архитектурным решением
DECISIONS.md                журнал архитектурных решений (по датам)
config.py, logging_setup.py общая конфигурация (pydantic-settings) и логирование

Этап 1 — ETL
elt/                        скачивание корпуса с Яндекс.Диска (параллельно, докачка)
parsing/                    конвертация всех форматов в Markdown + извлечение изображений
  extensions.md             полный перечень всех исходных файлов по расширениям

Этап 2 — извлечение и граф
chunking/                   сегментация Markdown → чанки; Natasha-пайплайн; оркестратор
ontology/                   типы сущностей/отношений, метки узлов, GeographyKind, NumericOperator
ner_re_extraction/          вручную выверенная разметка NER+RE по 7 статьям (эталон + few-shot банк)
  ner_re_examples.md        сама разметка (сущности, отношения, наблюдения)
  source_texts/             извлечённый текст этих 7 статей
synonym_normalization/      синонимы/алиасы: canonicalizer, dictionary, normalize_pipeline, units
llm_pipeline_fewshot/       LLM-пайплайн NER+RE: YandexGPTClient + MockLLMClient + фабрика + батч-раннер
  ner_re_extraction_prompt.md system+user промпт с 10 few-shot примерами
ensemble/                   ансамбль Natasha + LLM (EnsembleMerger)
quality_control/            контроль качества фактов: FactQualityController (low confidence, висячие концы, противоречия)
neo4j_integration/          загрузчик графа + схема Neo4j
graph_reasoning/            извлечение подграфа, GraphReasoner, entities_by_*, graph_statistics
routing/                    маршрутизация NL-запроса: query_router, entity_extractor, graph_coverage

Этап 3 — агент и доставка
agent/                      Dispatcher, AnswerSynthesizer, RAGClient/StubRAGClient, rag_factory, CLI
api/                        FastAPI REST: /query, /route, /health, /ready, /stats, /entities, /metrics
frontend/                   дизайнерское SPA (index.html + data.json): дашборд, граф, поиск, литобзор, сравнение, аудит — офлайн по data.json и с API
  build_frontend_data.py     агрегатор extraction-батча → frontend/data.json
frontend.py                 Streamlit UI (обзор графа, обход сущностей, NL-запросы, офлайн-режим) — скучная альтернатива
scripts/ingest.py           end-to-end ингест: parse → chunk → Natasha+LLM → ансамбль → Neo4j
run_natasha_eval.py         F1-оценка против golden_set
evaluation/                 матчинг предсказаний и метрики (NER/RE)
golden_set/                 эталонные сэмплы (57) + схема + валидатор
tests/                      pytest по всем слоям (88 passed)
Dockerfile, docker-compose.yml  контейнеризация (Neo4j + приложение)
```

В каждой папке — свой `README.md`: этот файл — карта, они — детали.

## Почему в репозитории нет `input_docs/` и `parsed_data/`

Исходный корпус весит несколько гигабайт (`input_docs/`, скачивается через `elt/`),
а результат парсинга в Markdown+изображения — уже ~213 МБ и продолжает расти
(`parsed_data/`, создаётся через `parsing/`). Ни то, ни другое не должно попадать
в git. Чтобы воспроизвести локально:

```bash
pip install requests python-docx python-pptx pymupdf4llm openpyxl xlrd pywin32

python -m elt.download            # -> input_docs/
python -m elt.extract_archives     # распаковать скачанные .rar/.zip

python -m parsing.run              # -> parsed_data/texts/, parsed_data/images/
```

(`parsing.run` требует установленный MS Word на машине, где запускается — легаси
`.doc`/`.docm` конвертируются через Word COM automation, см. `parsing/README.md`.)

## Что сознательно НЕ включено в репозиторий

Ранее существовал единый скрипт (`a.py`), объединявший последовательный
downloader с "проглатыванием" структуры папок и первую версию парсеров форматов.
Обе половины заменены: `elt/` устраняет баг downloader'а со сглаживанием структуры
(он молча перезаписывал одноимённые файлы из разных удалённых папок), а `parsing/`
заменяет слой парсеров реальным извлечением изображений из PDF (старая версия
только расставляла плейсхолдеры) и добавляет шаг валидации/отчёта. `a.py` устарел
и намеренно не включён — актуальным пайплайном считать `elt/` + `parsing/`.

## С чего продолжать

1. **Поднять инстанс Neo4j** (локально через `docker-compose.yml` или облачный) и
   прогнать `python -m scripts.ingest` без `--skip-neo4j` — загрузчик и схема
   готовы, но проверялся только offline-путь. Это главное, чего не хватает до
   полного end-to-end. (Внимание: в `docker-compose.yml` сейчас
   `NEO4J_PLUGINS: "[]"` — а загрузчик использует `apoc.*`; перед прогоном
   добавьте APOC в плагины, см. `neo4j_integration/README.md`.)
2. **Реальный RAG-бэкенд.** Сейчас RAG закрыт `StubRAGClient` (пустой ответ).
   Подключить semantic (эмбеддинги) и/или lexical (Elasticsearch) индекс через
   plug-in интерфейс `agent/rag_factory.py` (entry-point `kg.rag_backends` или
   `RAG_BACKEND=...`). Схема синонимов из `synonym_normalization/` рассчитана на
   оба сразу.
3. **Доразогнать `parsing.run`** на весь исходный корпус (~2000 файлов, пул PDF
   — самая долгая часть). Сейчас в `Статьи/` + `Доклады/` ~76 `.md`.
4. **Полный ингест по `Статьи/` через DeepSeek когда появятся ключи.** DeepSeek
   хостится на Yandex Cloud / Yandex AI Studio — вызывается тем же
   `YandexGPTClient` (не отдельным клиентом и не `api.deepseek.com`): режим
   `LLM_CLIENT_MODE=deepseek` собирает клиент с `ds://` model URI, аутентификация
   обычным `YANDEX_GPT_API_KEY` (ключ от Yandex AI Studio, не DeepSeek-ключ).
   URI берётся готовым из `YANDEX_GPT_MODEL_URI` (`ds://...`) или строится как
   `ds://<YANDEX_GPT_FOLDER_ID>/<DEEPSEEK_MODEL>`. Одна команда:
   `LLM_CLIENT_MODE=deepseek python -m scripts.ingest --input Статьи --skip-neo4j`
   (chunk → Natasha + DeepSeek → ансамбль → `parsed_chunks/merged.jsonl`).
   Затем сверька качества: `python run_natasha_eval.py --predictions parsed_chunks/merged.jsonl`
   против эталона `ner_re_extraction/ner_re_examples.md` и golden_set — eval
   принимает формат `merged.jsonl` напрямую (`source_document`/`entity`/
   `source_entity`/`relation_type`). Natasha-only F1 ≈ 0.04 — ожидаемо;
   полный охват типов даёт LLM.
5. **Чистка артефактов прогона** (опц.): в `extraction_results_obzory_batch.jsonl`
   замечена минимум одна усечённая JSON-строка — пересобрать батч или отфильтровать
   при загрузке.
