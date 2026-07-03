# chunking/ — Этап 2 (пред-извлечение): Natasha + разбиение на чанки

Мост между `parsing/` (даёт `parsed_data/texts/*.md`) и `llm_pipeline_fewshot/`
(LLM-извлечение по одному чанку за вызов). Здесь живёт **обязательный
классический NLP-слой (Natasha)**: сегментация предложений Natasha напрямую
задаёт границы чанков, а морфология/леммы/первичный NER кладутся в метаданные
чанка для последующей нормализации синонимов и lexical-индекса.

## Что делает

`parsed_data/texts/*.md` → `parsed_data/chunks.jsonl` (+ `chunk_report.json`).

Каждый чанк:
- рассчитан на один вызов LLM (цель ~2500 символов, потолок 4000);
- **не рвётся посреди предложения** (границы — от сегментатора Natasha);
- **не рвёт таблицы** (Markdown-таблица всегда целиком в одном чанке);
- уважает границы абзацев и секционных заголовков (`##` и выше — жёсткий разрыв);
- несёт **перекрытие** с предыдущим чанком (по умолчанию 2 предложения), чтобы
  сущность/отношение на стыке не потерялись;
- хранит **точную провенанс-привязку**: `char_start`/`char_end` — это буквальные
  смещения в исходном `.md`, так что `chunk.text == source[char_start:char_end]`
  (CLAUDE.md принцип 1 — источник каждого факта).

## Архитектура (SOLID)

| Файл | Ответственность |
|---|---|
| `segmentation.py` | Абстракция `SentenceSegmenter` (Protocol) + `Sentence`. Без зависимостей — шов между чистым ядром и Natasha. |
| `chunker.py` | **Чистый** алгоритм границ/перекрытия/таблиц/оффсетов. Зависит только от Protocol → полностью юнит-тестируем без Natasha. |
| `natasha_pipeline.py` | Реализация `SentenceSegmenter` на Natasha + морфология/леммы/NER. Ленивые синглтоны на процесс. Деградирует при отсутствии моделей. |
| `models.py` | Pydantic-контракт выхода (`Chunk`, `ChunkProvenance`, `NatashaAnnotation`). |
| `orchestrator.py` | Обход корпуса, `ProcessPoolExecutor`, отчёт. Паттерн `parsing/orchestrator.py`: падать поэлементно, не ронять батч. |
| `config.py` / `run.py` | Параметры (`ChunkConfig`) и CLI. |

## Запуск

```bash
pip install -r chunking/requirements.txt   # natasha + pydantic

# сначала должен быть выполнен парсинг (parsed_data/texts/*.md):
python -m parsing.run

# затем чанкинг всего корпуса:
python -m chunking.run

# быстрый прогон на первых 20 документах:
python -m chunking.run --limit 20 --workers 4

# осмотреть разбиение одного файла (без записи):
python -m chunking.run --sample parsed_data/texts/<путь>.md
```

Первый запуск Natasha скачивает модели navec/slovnet (нужен интернет один раз;
дальше работает офлайн из кэша). Если модели недоступны — сегментация (границы
чанков) всё равно работает, а NER/леммы помечаются `ner_available: false` в
отчёте, run не падает.

## Контракт выхода (одна строка `chunks.jsonl`)

```json
{
  "chunk_id": "Статьи/foo.md#0001",
  "index": 1,
  "provenance": {"source_document": "Статьи/foo.md", "char_start": 113, "char_end": 360,
                 "heading_path": ["Циркуляция католита...", "Введение"]},
  "text": "## Введение\nЭлектроэкстракция (electrowinning) никеля ...",
  "overlap_prefix_chars": 0,
  "oversize": false,
  "natasha": {"n_sentences": 3, "n_tokens": 41, "ner_available": true,
              "primary_entities": [{"text": "Институт Гипроникель", "normal": "институт гипроникель",
                                    "type": "ORG", "start": 150, "stop": 170}],
              "lemmas": ["электроэкстракция", "никель", "ванна", "скорость", "..."]},
  "doc_metadata": {"author": "Иванов И.И."}
}
```

Это ровно то, что потребляет LLM-раннер: `provenance.source_document` +
`char_start`/`char_end` идут в промпт (`ner_re_extraction_prompt.md`, поля
«Документ» / «Смещение чанка»), `primary_entities` — как дешёвые сиды сущностей
(Expert=PER, Facility/Publication-org=ORG), `lemmas` — в нормализацию алиасов и
lexical-индекс.
```
