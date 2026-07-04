# llm_extraction/ — Этап 2: Автоматический LLM-пайплайн NER/RE

Рабочий (не макетный) пайплайн автоматического извлечения сущностей и связей
из `parsed_data/texts/*.md` через Yandex AI Studio (DeepSeek v4 flash,
`reasoning_effort: high`), построенный поверх готового промпта и 10 few-shot
примеров из [`llm_pipeline_fewshot/`](../llm_pipeline_fewshot/). Промотирован
сюда из корневого `llm_extraction/` (см. `CLAUDE.md` в корне working directory)
после первого практического прогона на реальном корпусе — раньше в статусе
репозитория это была строка "пайплайн-код не написан, ждём API-токены", теперь
код есть и прогнан.

Схема разработки пайплайна («мозговой штурм по этапам») подробно расписана в
`mass_extraction_pipeline.md` (пока остаётся в корне working directory, не
здесь — это план, а не код).

## Пайплайн

```
parsed_data/texts/<категория>/*.md
        │  chunker.py / make_chunks_2000.py / make_chunks_obzory.py
        ▼
chunks_*.jsonl  (рекурсивный markdown-чанкер, ~2000–2200 симв./чанк)
        │  build_system_prompt.py / build_system_prompt_batch.py
        │  (собирают system-промпт живьём из llm_pipeline_fewshot/ner_re_extraction_prompt.md)
        ▼
extract.py (1 чанк за вызов) / extract_batch.py (до 40 чанков за вызов, с
рекурсивной бисекцией пачки при обрезании ответа или потере chunk_id)
        ▼
extraction_results_*.jsonl  (raw output + распарсенный JSON + validate_parsed())
        │  evaluate_ner_re.py
        ▼
Precision/Recall/F1 по NER и RE против golden_set/
```

| Файл | Роль |
|---|---|
| `env_load.py` | читает `YANDEX_API_KEY`/`YANDEX_FOLDER_ID` из `.claude/.env`, не печатая их; ищет `.env` вверх по дереву папок (файл лежит на два уровня ниже корня working directory, где реально хранятся креды — см. `CLAUDE.md`) |
| `chunker.py` | базовый рекурсивный markdown-чанкер (LangChain `RecursiveCharacterTextSplitter`), 2200 симв./чанк, режет по заголовкам → абзацам → предложениям |
| `make_chunks_2000.py` | тот же подход, 2000 симв., для `parsed_data/texts/Статьи` — этими чанками получен `extraction_results_statyi_batch.jsonl` |
| `make_chunks_obzory.py` | то же для `parsed_data/texts/Обзоры`, плюс `clean_chunk.py` после нарезки (ещё не прогонялось через extract_batch.py) |
| `clean_chunk.py` | лёгкая regex-очистка ПОСЛЕ нарезки: схлопывает повторяющиеся пробельные символы, убирает плейсхолдеры `[IMAGE_NNNN]` — не до нарезки, чтобы не портить границы `\n\n`/заголовков, на которые опирается чанкер |
| `build_system_prompt.py` / `build_system_prompt_batch.py` | собирают system-промпт (инструкция + few-shot) прямо из `ner_re_extraction_prompt.md`, чтобы не расходиться с источником правды при ручном копировании; batch-версия переупаковывает те же 10 примеров в batch-контракт (вход = несколько чанков, выход = `{"results": [...]}`) |
| `extract.py` | одночанковый вызов API; также источник общих констант/функций (`VALID_ENTITY_TYPES`, `validate_parsed`, парсинг ответа) для batch-версии |
| `extract_batch.py` | **основной путь**: пакует до 40 чанков в один вызов, при обрезании ответа (`max_output_tokens`) или пропавших `chunk_id` рекурсивно бисектит пачку и повторяет — так пачка целиком никогда молча не теряется |
| `evaluate_ner_re.py` | считает micro/macro/weighted Precision/Recall/F1 по NER (матчинг сущностей по стеммингованным `mentions`, `SnowballStemmer('russian')`) и по RE (точное совпадение тройки `(subject_canonical, predicate, object_canonical)`) против `golden_set/` |
| `extraction_results_statyi_batch.jsonl` | результат реального прогона `extract_batch.py` по всем чанкам `Статьи` (1334 строки/чанк-результата) — вход для `evaluate_ner_re.py` |
| `golden_set/` | 64-чанковая эталонная разметка для измерения F1 (см. `golden_set/golden_set_coverage_plan.md` за планом покрытия и `golden_set/source_texts/` за исходным текстом размеченных документов) |
| `leaked_few_shots.jsonl` | **важный caveat**: 3 из 10 few-shot примеров промпта оказались построены на тех же абзацах, что позже попали в `golden_set` — при чтении метрик ниже учитывай, что оценка не полностью "чистая" (модель могла видеть почти те же примеры в контексте) |

## Как запустить

```bash
cd llm_extraction

# 1. Нарезка (путь к parsed_data/ — три уровня вверх от этой папки, см. CLAUDE.md)
python make_chunks_2000.py ../../../parsed_data/texts/Статьи chunks_statyi_2000.jsonl

# 2. Извлечение (batch-режим, основной путь)
python extract_batch.py --chunks chunks_statyi_2000.jsonl --out extraction_results_statyi_batch.jsonl --concurrency 10 --pack 40

# 3. Оценка против golden_set
python evaluate_ner_re.py
```

Требует `.claude/.env` с `YANDEX_API_KEY`/`YANDEX_FOLDER_ID` (см. `env_load.py`)
и пакеты `httpx`, `langchain-text-splitters`, `nltk` (для `SnowballStemmer`).

## Текущие метрики (extraction_results_statyi_batch.jsonl vs golden_set, 14 общих документов)

| Задача | Micro F1 | Micro Precision | Micro Recall |
|---|---|---|---|
| NER | 0.22 | 0.15 | 0.43 |
| RE | 0.00 | 0.00 | 0.00 |

Читать эти числа с поправкой на два известных ограничения:
1. **утечка few-shot** — см. `leaked_few_shots.jsonl` выше;
2. **RE = 0** указывает не обязательно на полное отсутствие правильных связей, а
   на то, что матчинг сейчас требует точного совпадения тройки
   `(subject_canonical, predicate, object_canonical)` после стемминга — любое
   расхождение в каноническом имени сущности (а NER recall тут всего 0.43)
   каскадно обнуляет и связь. Прежде чем делать выводы о качестве
   RE-экстракции как таковой, стоит сначала поднять NER-recall и/или ослабить
   матчинг RE (например, сравнивать по типу сущности + предикату, а не по
   точному имени).

Ещё не прогонялось через `extract_batch.py`: категория `Обзоры` (чанкер для неё
есть — `make_chunks_obzory.py` — но батч-экстракция по ней не запускалась) и
любая полномасштабная оркестрация всего корпуса с батчингом/кэшированием по
токенам (описана в `mass_extraction_pipeline.md`, не реализована).
