# evaluation/ — Этап 3: Метрики NER+RE (precision/recall/F1)

Чистая библиотека сравнения предсказаний с эталоном по
сущностям и отношениям. Используется `../run_natasha_eval.py`
для sklearn-style отчёта против `../golden_set/`.

## Что делает

- `compare_entities` / `compare_relations` — сравнивают предсказанные
  и эталонные множества, возвращают `(tp, fp, fn)`. Совпадение
  считается по канонизированному хэшу:
  `type:canonicalize_text(name)` для сущностей и
  `src_type:src-[pred]->tgt_type:tgt` для отношений. Канонизация —
  из `../synonym_normalization/canonicalizer.py`, поэтому
  `Медная руда` и `медная руда` считаются одним совпадением.
- `Evaluator` накапливает micro-счётчики по чанкам
  (`evaluate_chunk`) и отдаёт агрегированный `EvaluationReport`
  через `get_micro_report`.
- `MetricResult` считает `precision`/`recall`/`f1_score` из
  tp/fp/fn; `EvaluationReport` группирует метрики по сущностям и
  отношениям.

## Файлы

| Файл | Ответственность |
|---|---|
| `evaluation_matcher.py` | `_hash_entity`/`_hash_relation`, `compare_entities`, `compare_relations`. |
| `evaluation_metrics.py` | `MetricResult`, `EvaluationReport`. |
| `evaluator_orchestrator.py` | `Evaluator` — аккумулятор micro-метрик по чанкам. |

## Связь с `run_natasha_eval.py`

`run_natasha_eval.py` — отдельный CLI в корне репозитория. Он сам
считает Micro/Macro/Weighted F1 (не через этот пакет) и умеет
грузить три источника предсказаний: Natasha-only (по умолчанию),
внешний «flat» JSONL (`--predictions`) и DeepSeek-формат (вложенный
`parsed`, матчинг по `doc_id` → статье). Этот пакет —
переиспользуемое ядро сравнения (dataclass-метрики + matcher по
канонизированному хэшу) для программного использования.

На данный момент пакет в репозитории ничем не импортируется —
ни `run_natasha_eval.py`, ни тестами. CLI в корне дублирует
логику сравнения самостоятельно (со своими нюансами: авто-детект
формата предсказаний, матчинг по `doc_id`/окну символов). Это
сознательное замеченное расхождение: либо `run_natasha_eval.py`
должен перейти на это ядро, либо ядро — кандидат на удаление/
сокращение. Не «готово к употреблению» без одного из этих шагов.
