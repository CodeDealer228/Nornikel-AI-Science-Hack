# scripts/ — Этап 2: end-to-end ингест

`scripts/ingest.py` — единственный оркестратор полного пайплайна
извлечения: обходит `.md`-файлы, чанкует, гоняет Natasha + YandexGPT,
сливает через ансамбль и грузит в Neo4j. Сплит на стадии чтобы
длинный прогон можно было возобновить после сбоя.

## Запуск

```bash
python -m scripts.ingest                       # полный пайплайн
python -m scripts.ingest --skip-llm            # только Natasha (без LLM-стоимости)
python -m scripts.ingest --skip-neo4j          # только merged.jsonl, без загрузки
python -m scripts.ingest --limit 50            # только первые 50 файлов
python -m scripts.ingest --input Статьи        # свой входной каталог
LLM_CLIENT_MODE=mock python -m scripts.ingest --skip-neo4j --limit 3   # smoke-test
```

По умолчанию: вход `./Статьи`, выход `./parsed_chunks/`.

## Что делает по шагам

1. **Обход + чанкинг** — `discover_markdown_files` → `chunk_file`
   (переиспользует `../chunking/`): `build_raw_chunks` + Natasha,
   на выходе `ChunkInput` с провенансом (`source_document`,
   `char_start`/`char_end`, `heading_path`).
2. **Natasha** — `natasha_entities_for_chunk`: PER/LOC/ORG →
   Expert/Facility/Organization (остальные типы Natasha не детектит),
   `confidence=0.6`.
3. **LLM** — `llm_extract_chunk` через `ChunkExtractor` (фабрика
   `create_llm_client` выбирает реальный YandexGPT или mock по
   `LLM_CLIENT_MODE`); неудача на чанке не роняет батч.
4. **Нормализация + ансамбль** — `ensemble_chunk`: сначала
   `normalize_entities`/`normalize_relations` из
   `../synonym_normalization/`, затем `EnsembleMerger.merge`.
5. **Поток в JSONL** — каждая запись чанка пишется в
   `parsed_chunks/merged.jsonl` сразу (resumability + отладка).
6. **Загрузка в Neo4j** — `load_to_neo4j` через `Neo4jLoader`
   (setup_constraints + load_entities + load_relations). Пропускается
   по `--skip-neo4j`; при ошибке фиксируется в `report.skipped`, а
   не падает.

## Выход

- `parsed_chunks/ingest_report.json` — сводка
  (`files_total`/`files_processed`/`files_empty`/`files_errored`,
  `chunks_total`, `entities_total`, `relations_total`, `by_source`,
  `duration_sec`, `skipped`).
- `parsed_chunks/merged.jsonl` — по строке на чанк с
  `entities`/`relations` (модельные дампы `EnrichedEntity`/
  `EnrichedRelation`).
- Neo4j — финальный загруженный граф (если не `--skip-neo4j`).

## Паттерн «громко падать на элементе, но продолжать батч»

Ошибки чанкинга/LLM/Natasha логируются и увеличивают
`files_errored`/`skipped`; обработка идёт дальше. Это тот же
паттерн, что в `../parsing/orchestrator.py` и
`../chunking/orchestrator.py`. stdout принудительно переведён в
UTF-8 (Windows-консоль cp1251 иначе роняет кириллицу и `→`).
