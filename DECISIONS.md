# DECISIONS.md

## 2026-07-04 — Дизайнерский SPA-фронтенд (пока нет ключей)

### Контекст
Базовый `frontend.py` (Streamlit) — скучный по природе платформы. Команда попросила
«дизайнерский разъёб» по ТЗ хакатона. Streamlit не даёт полного контроля над
разметкой/анимацией, поэтому сделал отдельное статическое SPA (HTML/CSS/JS, без CDN,
без сборки) в фирменном стиле Норникеля (тёмный графит + magenta/cyan-акцент). Работает
**без бэкенда и без ключей** прямо сейчас — по агрегату `data.json` из реального батча.

### Решение
1. **`build_frontend_data.py`** (корень репо) — агрегатор `extraction_results_*_batch.jsonl`
   → `frontend/data.json`: индекс сущностей (по `(type, canonical_name)`), рёбра,
   числовые значения, противоречия, пробелы (изолированные сущности + непокрытые пары
   «процесс × материал» + отсутствие зарубежных источников + low-confidence),
   технологические цепочки «материал→процесс→результат», эксперты с аффилиациями,
   дашборды (покрытие по доменам, зоны риска, активность команд). Выход: 9750 сущностей,
   4935 связей, 3483 числа, 9 противоречий, 42 пробела, 40 цепочек, 1.5 МБ.
2. **`frontend/index.html`** — одностраничное SPA, весь CSS+JS inline (без внешних
   зависимостей, работает офлайн). 9 вкладок по ТЗ:
   - **Ответ** — структурированный ответ (сущности/факты/цитаты/числа/гео/источники),
     с API — LLM-синтез, без — локальная агрегация.
   - **Дашборд** — KPI, покрытие по направлениям, донат по типам, зоны риска,
     активность команд, типы связей.
   - **Граф** — force-directed граф (140 узлов, перетаскивание/зум/карточка по клику,
     красные пунктирные рёбра `contradicts`) + цепочки материал→процесс→результат.
   - **Поиск и фильтры** — многоуровневая фильтрация (тип/география/связь/
     достоверность/числовое свойство + диапазон min/max) + таблица числовых значений.
   - **Эксперты и лаборатории** — карточки экспертов с аффилиациями + список Facility.
   - **Пробелы и противоречия** — gap-детекты + явные `contradicts`.
   - **Литературный обзор** — группировка источников, консенсус/разногласия,
     уверенность, рекомендации.
   - **Сравнение** — таблица вариант А vs Б (качественные + количественные метрики).
   - **Аудит** — журнал в localStorage (запросы/просмотры/экспорт), экспорт JSON.
   - Топбар: ролевая модель (5 ролей), индикатор API (зелёный/жёлтый), URL API +
     переподключение, экспорт Markdown/JSON-LD/PDF.
3. Маркеры запроса и бейдж маршрута зеркалят `routing/query_entity_extractor.py` и
   `routing/query_router.py` (numeric/geo/temporal/comparison/definitional/causal →
   GRAPH_ONLY/HYBRID/RAG_ONLY/NO_DATA).
4. Авто-определение API: SPA пробует `GET /health`; на связи — `/query` с LLM-синтезом,
   офлайн — `data.json`. Переключается на лету через поле URL.

### Проверка
- **jsdom runtime-smoke** (23/23): рендер всех 9 вкладок, 140 узлов графа, 173 рёбра,
  40 цепочек, 60 экспертов, 60 Facility, 42 пробела, 9 противоречий, 300 строк поиска,
  200 чисел, 4 KPI, 5 coverage-баров, 9 donut-сегментов, 8 relation-баров, 8 risk-строк,
  12 team-баров, 5 chips, 9 tabs/panels.
- **5/5 примеров-запросов** проходят без пустого ответа (GRAPH_ONLY ×4, HYBRID ×1 для
  сравнительного): ans 2655–5565 симв., lit review 3865–5839 симв.
- `node --check` — JS-синтаксис валиден. data.json валиден.
- Статика поднимается: `python -m http.server -d frontend 8050` → 200 на `/` и `/data.json`.

### Найденный и починенный баг (в ходе smoke)
`extractSeeds` матчил только полное каноническое имя как подстроку запроса — русская
морфология («обессоливания» ↔ сущность «обессоливание», «сульфаты» ↔ «сульфат») давала
0 сидов для флагманского запроса → пустой ответ + пустой литобзор. **Починено**:
токен-матчинг по 5-символьному префиксу + стоп-слова (какие/методы/практика/…), чтобы
технические термины всплывали, а query-оболочка нет. После: ans 4489 симв.,
lit review 5366 симв. (было ~100/147).

### Затронутые ограничения
- `frontend.py` (Streamlit) не удалял — оставил как скучную альтернативу.
- SPA работает по `data.json` из `extraction_results_statyi_batch.jsonl`; после
  реального DeepSeek-прогона — пересобрать `python build_frontend_data.py` (он
  автоматически возьмёт свежий `parsed_chunks/merged.jsonl`, если батч-файл ниже
  в порядке кандидатов). См. `frontend/README.md`.
- Граф-визуализация — упрощённый force-directed на каноническом SVG (без D3/cytoscape,
  чтобы не тащить CDN); 140 узлов / 173 рёбра — сэмпл топа. Для полного графа нужен
  WebWorker-сим, но для демо достаточно.
- Аудит — localStorage (демо), не серверный; ролевая модель — UI-уровень (не server-side
  enforcement). Для прод-ИБ — отдельная задача.

## 2026-07-04 — DeepSeek едет на Yandex, а не на api.deepseek.com (пересмотр)

### Контекст
Уточнение команды: «всё будем делать через DeepSeek с ключом от Yandex AI Studio».
Значит DeepSeek — это модель, хостимая на Yandex Cloud Foundation Models, а не
публичный `api.deepseek.com`. Аутентификация — яндексовская (`Api-Key` +
`YANDEX_GPT_API_KEY`/`YANDEX_GPT_FOLDER_ID`), а не DeepSeek-овский `sk-...` Bearer.
Ранее (запись «DeepSeek-клиент» ниже) я сделал отдельный `DeepSeekClient` на
`api.deepseek.com` с `DEEPSEEK_API_KEY` — это был **неправильный** endpoint для
данной схемы. Запись ниже оставлена как история; актуален этот пересмотр.

### Решение
1. **Удалён отдельный `DeepSeekClient`**, `DeepSeekError`, константы
   `DEEPSEEK_CHAT_ENDPOINT`/`DEEPSEEK_DEFAULT_MODEL` и экспорты в `__init__.py`.
   DeepSeek больше не стучится на `api.deepseek.com` и не требует `DEEPSEEK_API_KEY`.
2. **`YandexGPTClient` принимает любой `scheme://` URI.** Раньше конструктор
   проверял только `startswith("gpt://")` — `ds://`-URI попадал в обе ветки и
   ломался в `gpt://ds://...`. Теперь: полный URI с схемой (`gpt://`, `ds://`,
   …) используется verbatim; bare-имя модели требует `folder_id` (иначе явная
   ошибка вместо молчаливого `gpt:///model`). DeepSeek-on-Yandex = тот же
   `YandexGPTClient` + `ds://` model URI + обычный `YANDEX_GPT_API_KEY`.
3. **Фабрика `create_llm_client('deepseek')`** строит `YandexGPTClient` с
   `ds://`-URI: готовый `YANDEX_GPT_MODEL_URI` (если `ds://`) — as-is; иначе
   `ds://<YANDEX_GPT_FOLDER_ID>/<DEEPSEEK_MODEL>` (`DEEPSEEK_MODEL` по умолч.
   `deepseek-v3`). Без того и другого — понятная `YandexGPTError`.
4. **Синтез ответа переведён на `create_llm_client()`** в `agent/cli.py` и
   `api/server.py` (раньше хардкод `YandexGPTClient()` / гейт
   `is_configured=api_key AND folder_id`, что блокировало `ds://`-URI без
   `folder_id`). `MockLLMClient` для синтеза пропускается (он выдаёт NER/RE
   JSON-фикстуру, а не пользовательский ответ — падаем в context-only render).
   Офлайн-путь сохранён: нет ключа → `create_llm_client()` падает → catch →
   `synth=None` (как раньше при `is_configured=false`).
5. **`.env.example`** — `DEEPSEEK_API_KEY` убран; DeepSeek описан как
   Yandex-хостинг (тот же ключ, `ds://`-URI, `DEEPSEEK_MODEL=deepseek-v3`).
6. Документация (корневой `README`, `llm_pipeline_fewshot/README.md`) обновлена
   под DeepSeek-on-Yandex.

### Проверка (без ключей, stub HTTP)
- 7 проверок `_ds_via_yandex_check.py`: `create_llm_client('deepseek')` →
  `YandexGPTClient` (не отдельный класс) с `ds://b1g123/deepseek-v3`; запрос
  идёт на `https://llm.api.cloud.yandex.net/foundationModels/v1/completion` с
  `Authorization: Api-Key ...` и Yandex-телом `{modelUri, completionOptions,
  messages}` (НЕ на `api.deepseek.com`); Yandex-ответ (`result.alternatives[0]
  .message.text`) парсится; `ds://`-URI без folder_id работает; без URI и
  folder_id — понятная `YandexGPTError`; `real` с `gpt://` и `mock` не сломаны.
- **«First time with keys» через `.env`**: временный `.env` с
  `YANDEX_GPT_API_KEY` + `ds://`-URI + `LLM_CLIENT_MODE=deepseek`, env-vars
  unset inline → импорт `scripts.ingest` подхватил `.env` → `create_llm_client()`
  собрал `YandexGPTClient` с ключом и `ds://`-URI из файла.
- **Live uvicorn**: `/health` 200, `/ready` offline (`synthesis_configured:false`),
  `/query` offline → `route=rag_only, used_llm=false` — синтез-рефактор не
  сломал API.
- **Mock-ингест** (3 файла): `skipped: []`.
- `python -m pytest tests/ -q` → **88 passed**.

### Замечание
Точный `ds://` model URI для DeepSeek-on-Yandex я внешне подтвердить не смог
(Yandex-доки закрыты captcha, web-поиск пуст). Но контракт (`/completion`
endpoint, `Api-Key`, `{modelUri, completionOptions, messages}`, ответ
`result.alternatives[*].message.text`) — это существующий `YandexGPTClient`,
уже работавший в прошлых прогонах; DeepSeek отличается только схемой `ds://`.
Когда дадут ключ от Yandex AI Studio, подставьте реальный `ds://<folder>/...`
URI в `YANDEX_GPT_MODEL_URI` — правки кода не потребуется.

### Затронутые ограничения
- Онтология, промпт, few-shot, Neo4j-загрузчик, ансамбль, роутер — не менялись.
- Поведение `create_llm_client()` по умолчанию (`real`→Yandex) не изменено.
- `load_dotenv` (запись «Pre-key dry-run», баг 4) сохранён — по-прежнему грузит
  `.env` в `config.py`/`scripts/ingest.py`/`agent/cli.py`.
- Синтез через `create_llm_client()` теперь следует `LLM_CLIENT_MODE`, но
  `MockLLMClient` для синтеза пропускается (см. п. 4).

## 2026-07-04 — Фаза 1, шаг 1: mock-LLM для smoke-test

### Решение
Добавлен единый интерфейс `LLMClient` и фабрика `create_llm_client()` в `llm_pipeline_fewshot/llm_parser.py`.
Фабрика выбирает реальный YandexGPT-клиент или детерминированный `MockLLMClient`.

### Почему так
Smoke-test должен запускаться первым и не зависеть от API-ключей, сети и платных токенов. При этом текущий продовый путь не должен ломаться: режим по умолчанию оставлен `real`, а mock включается явно через env.

### Как включить mock
```bash
LLM_CLIENT_MODE=mock python -m scripts.ingest --skip-neo4j --limit 10
```

Также поддерживаются:
- `LLM_MODE=mock`
- `YANDEX_GPT_USE_MOCK=1`

### Компромиссы
`MockLLMClient` не пытается имитировать качество реальной модели. Он возвращает валидный JSON по схеме NER+RE на основе небольших фикстур из ручных примеров `ner_re_extraction/ner_re_examples.md` и fallback-ключевых слов. Этого достаточно для проверки связки end-to-end, но не для оценки precision/recall.

### Затронутые ограничения
- Онтология не менялась.
- Fine-tune не добавлялся.
- Новые зависимости не добавлялись.
- RAG/agent/router не изменялись, потому что это будущие фазы.

## 2026-07-04 — Разблокировка交付ных entry-points на Windows (дедлайн)

### Контекст
Команда использует системный Python 3.13.7 (`C:\python\python`); `.venv/` — мёртвый артефакт
(Python 3.14, падает с `Failed to import encodings module`), на него полагаться нельзя.
Перед сдачей три CLI entry-point падали на стандартной Windows-консоли (cp1251), а API-сервер
и eval-скрипт не запускались из-за отсутствующих зависимостей. 8 тестов `tests/test_api.py`
молча skip'ались, потому что `fastapi` не был установлен — и потому скрывали реальный баг в роутере.

### Решение

1. **UTF-8 stdout в CLI.** В `scripts/ingest.py`, `agent/cli.py`, `run_natasha_eval.py` добавлен
   `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` на старте. Раньше help/лог с
   кириллицей и `→`/`—` падали с `UnicodeEncodeError: 'charmap' codec can't encode character`.
   Документированный smoke-test `LLM_CLIENT_MODE=mock python -m scripts.ingest --skip-neo4j --limit 3`
   теперь проходит на голой Windows-консоли (3 файла → 44 чанка → 306 сущностей, 3.4 c).

2. **Установка отсутствующих зависимостей** в системный Python: `fastapi`, `pydantic`,
   `pydantic-settings`, `neo4j`, `python-dotenv`, `scikit-learn` (всё уже в `requirements.txt`).
   `neo4j` импортируется лениво в `Neo4jLoader.__init__` (строка 55), поэтому пакет грузился и без
   драйвера — но реальная загрузка в граф падала бы. После установки `api.server` импортируется,
   uvicorn поднимается, `run_natasha_eval.py` отрабатывает (F1-отчёт печатается).

3. **Баг в `api/server.py` /query.** Строка `if req.synthesis_calls:` обращалась к несуществующему
   атрибуту pydantic-модели `QueryRequest` (был `# type: ignore[attr-defined]` — намёк, что поля
   нет). Любой `/query` возвращал 500. Счётчик `synthesis_calls` перенесён за пределы dispatch-block
   и инкрементируется по факту `result.synthesis is not None`, а не по запросу.

4. **`has_geo_marker` не ловил падежные формы.** В `routing/query_entity_extractor.py` маркер
   географии брался только из `_GEO_RE` (формы им. падежа: «отечественная практика»), а запрос
   «Сравни отечественную и зарубежную практику выщелачивания никеля» использует винительный
   падеж — корректный паттерн лежал в `_GEO_COMPARISON_RE`, но кормил `is_geo_comparison`, не
   `has_geo_marker`. Теперь `has_geo_marker = _GEO_RE.search(query) or _GEO_COMPARISON_RE.search(query)`
   — гео-сравнение это и есть гео-маркер. Это прямо по требованию задания: различение
   отечественной/зарубежной практики.

5. **`tests/test_api.py` lifespan.** Тест вызывал `lifespan(app).__enter__()` синхронно, но
   `lifespan` — `@asynccontextmanager` (`_AsyncGeneratorContextManager`, нужны `__aenter__`/`__aexit__`).
   Переведён на `TestClient(app) as ctx` — он сам прогоняет lifespan в том же loop'е, что и запросы,
   и корректно биндит `STATE.driver`/`STATE.dispatcher`.

### Проверка
- `python -m pytest tests/ -q` → **88 passed** (было 80 passed + 8 skipped; все 8 API-тестов теперь
  реально выполняются и проходят).
- Live-проверка uvicorn: `/health`, `/route`, `/query`, `/metrics` возвращают 200; geo-comparison
  запрос даёт `markers.geography=true, comparison=true`; offline `/query` даёт `used_llm=false`.

### Компромиссы
- Зависимости поставлены в системный Python, а не в пересозданный `.venv`. `.venv` оставлен как есть
  (мёртвый) — пересоздавать перед сдачей рискованнее, чем работать в проверенном интерпретаторе.
- Качество Natasha-only извлечения низкое (NER Micro F1 ≈ 0.04, RE = 0) — это ожидаемо и отмечено
  самим скриптом: Natasha детектит только PER/LOC/ORG → Expert/Facility/Organization; полный охват
  типов требует `--predictions <jsonl>` от LLM-пайплайна. Был разблокирован прогон, не точность.

### Затронутые ограничения
- Онтология не менялась.
- Промпт LLM и few-shot не менялись.
- Neo4j по-прежнему не поднят (нет инстанса); проверялся только offline-путь.
- Документация (README) не обновлялась — статус-таблица устарела (агент/API/роутер уже существуют,
  хотя README помечает их как «не начато»); это отдельная задача.

## 2026-07-04 — DeepSeek-клиент (слот под будущие ключи, без прогона)

### Контекст
Промпт `llm_pipeline_fewshot/ner_re_extraction_prompt.md` с самого начала написан под
DeepSeek («один вызов DeepSeek на один чанк»), а артефакты прошлого прогона
(`extraction_results_statyi_batch.jsonl`, `extraction_results_obzory_batch.jsonl`) уже
в DeepSeek-формате выхода (`{doc_id, parsed:{entities,relations}}`). При этом в репо
был только *читатель* этого формата (`run_natasha_eval.py::load_deepseek_predictions`,
`frontend.py`), а самого *клиента* DeepSeek не было — `create_llm_client()` умел только
`YandexGPTClient` и `MockLLMClient`. Ключей к LLM сейчас нет; когда дадут, это будет
DeepSeek. Значит слот под него нужно подготовить заранее, чтобы в момент ключей
запустить полный ингест по `Статьи/` одной командой, а не дописывать клиент в спешке.

### Решение
1. **`DeepSeekClient`** в `llm_pipeline_fewshot/llm_parser.py` — OpenAI-совместимый
   endpoint `https://api.deepseek.com/chat/completions`, Bearer-auth, только `urllib`
   (без новых зависимостей, как у `YandexGPTClient`). Реализует тот же `LLMClient`
   protocol (`complete`/`acomplete` → `CompletionResponse`), значит drop-in для
   `ChunkExtractor` и `AnswerSynthesizer`. `deepseek-chat` и `deepseek-reasoner` на
   одном endpoint (модель выбирается полем `model`, не отдельным URL — подтверждено
   по api-docs.deepseek.com). `response_format: {"type":"json_object"}` — NER+RE
   промпт ждёт JSON. Ретраи на 429/5xx, как у Yandex. Модель передаётся в API
   verbatim — значит новые `deepseek-v4-flash`/`deepseek-v4-pro` работают без правки
   кода (старые `deepseek-chat`/`deepseek-reasoner` депрекейтнуты 2026-07-24).
2. **`DeepSeekError`** — наследник `YandexGPTError`, чтобы существующие обработчики
   (`ChunkExtractor`, `AnswerSynthesizer`, `scripts.ingest`) ловили ошибки DeepSeek
   без рефактора.
3. **Фабрика `create_llm_client()`** — добавлена ветка `deepseek` (и
   `deepseek-chat`/`deepseek-reasoner` для конкретной модели). Режим по умолчанию
   `real`→Yandex **не менялся** (принцип из записи выше: не ломать продовый путь);
   DeepSeek включается явно через `LLM_CLIENT_MODE=deepseek`. Требует `DEEPSEEK_API_KEY`,
   иначе `DeepSeekError` на конструировании.
4. **`.env.example`** — `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `LLM_CLIENT_MODE=mock`.
5. **Экспорты** `__init__.py` — `DeepSeekClient`, `DeepSeekError` в `__all__`.

### Проверка (без ключей)
- 9 структурных проверок: импорт; `deepseek` без ключа → `DeepSeekError`; с ключом
  → `DeepSeekClient` с верными `endpoint`/`model_uri`/`model`; `deepseek-reasoner` →
  тот же endpoint, модель прокинута; произвольная модель (`deepseek-v4-flash`)
  форвардится verbatim; payload OpenAI-совместимый с `response_format`; `_parse_response`
  маппит `prompt_tokens`/`completion_tokens`/`total_tokens` в `CompletionUsage`;
  `DeepSeekError` — наследник `YandexGPTError`. Все ОК.
- `python -m pytest tests/ -q` → **88 passed** (поведение не изменилось).

### Что НЕ сделано (сознательно)
- **Прогона по `Статьи/` не было** — по решению команды, только клиент. Когда дадут
  ключ: `DEEPSEEK_API_KEY=... LLM_CLIENT_MODE=deepseek python -m scripts.ingest --input Статьи --skip-neo4j`.
- Синтез ответа (`AnswerSynthesizer`) всё ещё хардкодит `YandexGPTClient` в
  `agent/cli.py` и `api/server.py` — DeepSeek туда не проброшен. Для ингеста
  (NER+RE) это не нужно; понадобится только если хочется генерить финальные ответы
  через DeepSeek, а не через Yandex.

### Затронутые ограничения
- Онтология, промпт, few-shot — не менялись.
- Neo4j не поднят; APOC в `docker-compose.yml` по-прежнему `[]` (см. запись выше и
  `neo4j_integration/README.md`).
- Поведение `create_llm_client()` по умолчанию не изменено.

## 2026-07-04 — Pre-key dry-run: пойманы и починены 3 бага «не взлетит с первого раза»

### Контекст
Ключей к DeepSeek ещё нет, но полный ингест по `Статьи/` должен запуститься с
первого раза, когда они появятся. Поэтому прогнал всю цепочку end-to-end на
реальных 60 `.md` из `Статьи/` в режиме `LLM_CLIENT_MODE=mock` (проверяет всё,
кроме HTTP-вызова DeepSeek) + отдельно проверил HTTP-слой DeepSeek через stub
OpenAI-ответ (единственный слой, недоступный без ключа) + live-проверка uvicorn.
Четыре бага всплыли и починены.

### Проверки, прошедшие чисто
- **Mock-ингест по всем 60 `Статьи/`**: 6483 сущностей, `skipped: []`, 3.3 c на
  smoke (3 файла). Связка chunk → Natasha+mock-LLM → ансамбль → `merged.jsonl`
  работает на реальных файлах.
- **HTTP-слой DeepSeek** (`_deepseek_http_check.py`, stub `urlopen`): request-shape
  (`https://api.deepseek.com/chat/completions`, `Authorization: Bearer ...`,
  OpenAI messages, `response_format: {"type":"json_object"}`, `stream:false`),
  parse (OpenAI → `CompletionResponse`, маппинг `prompt_tokens`/
  `completion_tokens`/`total_tokens`), 401 → `DeepSeekError` без ретрая,
  429 → ретрай с отказом. Все 4 проверки ОК.
- **Live uvicorn** (`api.server:app`): `/health` 200, `/ready` offline
  (`neo4j_connected:false`), `/route` geo-comparison →
  `markers.geography=true, comparison=true`, `/query` offline →
  `route=rag_only, used_llm=false`, `/metrics` 200. (Curl на Git Bash/Windows
  коверкает кириллицу через cp1251 — тела запросов надо слать из UTF-8-файлов
  через `--data-binary @file`, это артефакт curl, не API.)
- `python -m pytest tests/ -q` → **88 passed** после всех правок.
- **«First time with keys» через `.env`**: создал временный `.env` с
  `DEEPSEEK_API_KEY=...` + `LLM_CLIENT_MODE=deepseek`, запустил subprocess с
  этими env-vars *unset inline* (единственный источник — файл), импорт
  `scripts.ingest` подхватил `.env` через новый module-level `load_dotenv`, и
  `create_llm_client()` собрал `DeepSeekClient` с ключом из файла. После —
  `.env` удалён, `git check-ignore .env` подтверждает, что он в `.gitignore`
  (строка 151) — утечки ключа через коммит не будет.
- Промпт содержит «json» 18 раз → требование OpenAI-compatible
  `response_format: {"type":"json_object"}` (в промпте должно быть слово
  "json") выполнено, первого вызова 400 по этой причине не будет.

### Баг 1: eval не матчит свежий выход ingest (F1=0.0000)
`run_natasha_eval.py --predictions parsed_chunks/merged.jsonl` давал
`matched 0 entity sets`, `format: unknown`, F1=0.0000. Причина: расхождение имён
полей. `scripts.ingest` пишет `EnrichedEntity.model_dump()` → `source_document` +
`entity` + `source_entity`/`target_entity`/`relation_type` +
`source_entity_type`/`target_entity_type`; eval-читатель ждал `source_file` +
`canonical_name` + `subject`/`object`/`predicate` + `subject_type`/`object_type`.
Когда дадут ключ и запустят eval на свежем `merged.jsonl` — молча 0 совпадений.
**Починка**: `load_external_predictions` и `detect_predictions_format` сделали
двусторонними (принимают оба набора имён). После: `matched 57 entity sets`,
`format: flat`. Формат выхода ingest не трогал (на него завязаны
loader/ensemble/frontend).

### Баг 2: фронтенд показывал 0 сущностей на реальном ingest-выходе
`frontend.py::load_offline_data` читал `ent.get("canonical_name")`, а
`merged.jsonl` пишет `entity` → пустые canonical_name. **Починка**: ветка
«Pipeline format» теперь `ent.get("canonical_name") or ent.get("entity", "")` и
`rec.get("source_document") or rec.get("source_file", "")`.

### Баг 3: фронтенд возвращался на первом *существующем* файле-кандидате
`MERGED_JSONL_CANDIDATES` = `[parsed_data/chunks.jsonl, parsed_chunks/merged.jsonl,
extraction_results_statyi_batch.jsonl]`. `parsed_data/chunks.jsonl` существует,
но это выход чанкинга (нет поля `entities`/`parsed`) — 0 сущностей. Цикл
возвращался на нём и никогда не доходил до `merged.jsonl` (6483 сущностей).
**Починка**: цикл теперь возвращает первого кандидата, который *реально дал
сущности*; кандидат без сущностей — fallback, не основной. После: выбран
`merged.jsonl`, 6483 сущности, все с непустым canonical_name.

### Баг 4: `.env` не загружался — ключ из файла не доходил до `os.environ`
`.env.example` прямо инструктирует: «Скопируй в .env и подставь реальные
значения», а `python-dotenv` лежит в `requirements.txt`. Но никто не вызывал
`load_dotenv()` — каждый модуль читал `os.environ` напрямую. Значит пользователь,
который кладёт `DEEPSEEK_API_KEY` в `.env` и запускает `python -m scripts.ingest`,
получал бы `DeepSeekError("DeepSeek API key is required")` на первом же чанке —
ключ из файла до `create_llm_client()` не доходил. **Починка**: module-level
`load_dotenv(<repo>/.env, override=False)` (с явным путём к корню репо, чтобы
работало из любого cwd; guarded `try/except`, no-op если dotenv отсутствует или
`.env` нет) добавлен в три точки чтения env: `config.py` (покрывает
`api/server.py` через `get_settings()`), `scripts/ingest.py` (критичная команда
ингеста) и `agent/cli.py` (второй `python -m` entry-point, обходит `config.py`).
`override=False` — inline env vars выигрывают у файла. В `scripts/ingest.py`
попутно поднял `from pathlib import Path` выше блока `load_dotenv` (иначе
`NameError` на импорте — поймал и исправил сразу). `.env` уже в `.gitignore`
(строка 151) — утечки ключа через коммит не будет.

### Замечание про `deepseek-reasoner` (не баг, задокументировано)
По api-docs.deepseek.com: `response_format: json_object` поддерживается,
`temperature` молча игнорируется (не ошибка), `logprobs`/`top_logprobs` —
ошибка (мы их не шлём). `max_tokens` на reasoner включает цепочку рассуждений,
так что 3000 может truncate COT — имеет смысл поднять `DEEPSEEK_MAX_TOKENS`/
`max_tokens`, если переключатся на reasoner. Для дефолтного `deepseek-chat` это
не актуально. Модель форвардится в API verbatim — новые `deepseek-v4-flash`/
`deepseek-v4-pro` работают без правки кода (`deepseek-chat`/`deepseek-reasoner`
депрекейтнуты 2026-07-24).

### Затронутые ограничения
- Онтология, промпт, few-shot, Neo4j-загрузчик, ансамбль, роутер — не менялись.
- Формат `merged.jsonl` не менялся; правки только в потребителях (eval, frontend).
- Добавлен module-level `load_dotenv` в `config.py`, `scripts/ingest.py`,
  `agent/cli.py` — поведение не меняется, если `.env` нет или dotenv не
  установлен (no-op). Inline env vars по-прежнему выигрывают (`override=False`).
- Синтез через DeepSeek по-прежнему не проброшен (см. запись выше).
