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
| 1.1 Парсинг → Markdown | [`parsing/`](parsing/) | 🟡 пайплайн готов, **прогон по корпусу неполный** (34 из ~2000 файлов; ~1400 PDF ещё не обработаны) |
| 2. Ручная разметка NER/RE | [`ner_re_extraction/`](ner_re_extraction/) | ✅ готово для 7 статей — эталонная разметка + банк примеров для few-shot |
| 2. Проблема синонимов + сборка графа | [`synonym_normalization/`](synonym_normalization/) | 🟡 первая версия реализована — `build_graph.py` дедуплицирует `llm_extraction/extraction_results_*.jsonl` в файловый граф (`graph/nodes.jsonl`/`edges.jsonl`), используя курируемый словарь `resources/synonyms.yaml` (перенесён из ветки `search-module-update`); загрузка в графовую БД ещё не сделана |
| 2. Автоматический LLM-пайплайн NER/RE | [`llm_extraction/`](llm_extraction/) | 🟡 пайплайн реализован и прогнан (batch-режим, DeepSeek v4 flash) на категории `Статьи`; F1 против golden_set: NER 0.22, RE 0.00 (см. `llm_extraction/README.md` за caveats — few-shot leakage, RE-матчинг каскадно зависит от NER recall); `Обзоры`/`Доклады` и полный корпус ещё не прогнаны |
| 2. Загрузка графовой БД + схема | — | ⬜ не начато (файловый граф из `synonym_normalization/` в Neo4j/etc. ещё не загружен) |
| 2. Semantic + lexical поисковый индекс | [`search/`](search/) | 🟡 первая версия реализована и проверена — гибридный поиск (dense e5/Qwen эмбеддинги + BM25 + RRF), glossary-aware перевод запроса, синонимы, числовая аннотация результатов; перенесён из ветки `search-module-update`, адаптирован под формат чанков `llm_extraction/chunker.py` |
| 3. Агент (ответы на естественном языке по графу) | — | ⬜ не начато |

## Карта репозитория

```
hackathon_task.md          исходное задание хакатона — прочитать перед любым архитектурным решением
elt/                        Этап 1: скачивание корпуса с Яндекс.Диска (параллельно, докачка)
parsing/                    Этап 1.1: конвертация всех форматов в Markdown + извлечение изображений
  extensions.md             полный перечень всех исходных файлов по расширениям
ner_re_extraction/          Этап 2: вручную выверенная разметка NER+RE по 7 статьям
  ner_re_examples.md        сама разметка (сущности, отношения, наблюдения)
  source_texts/             извлечённый текст этих 7 статей (для проверки разметки)
synonym_normalization/       Этап 2: дедупликация NER-графа — build_graph.py собирает graph/nodes.jsonl+edges.jsonl
  build_graph.py             дедуплицирует llm_extraction/extraction_results_*.jsonl в файловый граф
  graph/                     выходной граф: nodes.jsonl, edges.jsonl, merges.jsonl (аудит объединений)
llm_pipeline_fewshot/        Этап 2: готовый system+user промпт для автоматического LLM-извлечения
  ner_re_extraction_prompt.md system-промпт + 10 few-shot примеров для DeepSeek (источник правды)
llm_extraction/              Этап 2: рабочий пайплайн автоматического LLM NER/RE (прогнан на Статьях)
  extract_batch.py           основной путь: batch-вызовы к DeepSeek v4 flash + рекурсивная бисекция
  evaluate_ner_re.py         F1 NER/RE против golden_set/
  golden_set/                64-чанковая эталонная разметка для оценки
resources/                    курируемый словарь синонимов (synonyms.yaml) + намайненные кандидаты
  synonyms.yaml               120+ курируемых групп canonical_id/type/canonical_name/aliases
search/                       Этап 2: гибридный поиск (dense + BM25 + RRF) по chunks, synonym-aware запросы
  build_index.py              строит поисковый индекс (embeddings + BM25) из chunks.jsonl
  searcher.py                 HybridSearcher — dense search + BM25 + RRF fusion + numeric annotation
  run_search.py                CLI: python -m search.run_search "запрос" --index-dir ...
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

1. Доразогнать `parsing.run` на весь корпус (пул PDF — самая долгая часть, на
   ~1400 файлов стоит закладывать реальное время выполнения).
2. Выбрать графовую БД (Neo4j/Neptune/JanusGraph по заданию) и загрузить туда
   файловый граф из `synonym_normalization/graph/` (`nodes.jsonl`/`edges.jsonl`,
   уже дедуплицированный через `resources/synonyms.yaml`). Онтология уже
   опробована в `ner_re_extraction/`: типы сущностей `Material, Process,
   Equipment, Property, Experiment, Publication, Expert, Facility`; типы
   отношений `uses_material, operates_at_condition, produces_output,
   described_in, validated_by, contradicts`.
   **Не начинать с нуля** — ветка `chunking` уже содержит рабочий
   `neo4j_integration/neo4j_loader.py` (apoc-based upsert, per-type labels) и
   query-time слой (`routing/`, `graph_reasoning/`, `agent/`, `api/`), но он
   спроектирован под *другую* архитектуру приёма (живая нормализация во время
   Natasha+LLM-ensemble извлечения, не пост-обработка готового
   `extraction_results_*.jsonl`, которая есть в `main` сейчас) — при переносе
   его нужно адаптировать под схему `graph/nodes.jsonl`/`edges.jsonl`, а не
   копировать как есть. Известные баги в этом коде на момент ревью (2026-07-04):
   `graph_reasoning/neo4j_subgraph.py` матчит узлы по несуществующему лейблу
   `:Entity` (loader пишет только per-type лейблы — граф всегда будет выглядеть
   пустым), `api/server.py`'s `/query` падает на каждый вызов
   (`req.synthesis_calls` — несуществующее поле `QueryRequest`) — исправить
   прежде, чем брать этот код в работу.
3. `llm_extraction/` уже прогнан на `Статьях` (NER F1 0.22, RE F1 0.00 — см.
   `llm_extraction/README.md` за caveats). Следующий шаг — не столько "написать
   пайплайн" (готово), сколько поднять NER recall/каноникализацию сущностей
   (сейчас 0.43) и ослабить матчинг RE, прежде чем прогонять `Обзоры`/`Доклады`
   и весь корпус через оркестрацию из `mass_extraction_pipeline.md`.
4. ~~Поднять semantic и lexical поисковые индексы вместе~~ — сделано,
   см. [`search/`](search/) (dense e5/Qwen + BM25 + RRF, synonym-aware
   запросы, числовая аннотация). Не хватает: структурированного разбора
   запроса на слоты материал/процесс/условие/география/диапазон дат (сейчас
   запрос только переписывается в лучшую поисковую строку, без facet-фильтров)
   и геотегов отечественная/зарубежная практика — см. `search/README.md`.
5. Слой агента/NL-запросов поверх графа пока не начат для текущей (post-hoc,
   без Neo4j) архитектуры main — см. п.2 про уже готовый, но архитектурно
   несовместимый `agent/`+`api/` из ветки `chunking`.
