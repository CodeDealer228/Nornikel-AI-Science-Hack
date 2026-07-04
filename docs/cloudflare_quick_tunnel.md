# Cloudflare Quick Tunnel demo

Эта инструкция нужна, чтобы любой член команды мог поднять такую же веб-демку
локально и получить публичную ссылку Cloudflare.

## Что важно

Cloudflare Quick Tunnel не хранит постоянные настройки в репозитории. Ссылка
выдаётся заново при каждом запуске `cloudflared tunnel --url ...`.

В репозиторий не кладём:

- `.env` с ключами;
- Cloudflare credentials;
- tunnel token;
- логи `cloudflared*.log`.

Для демо нужен один локальный порт, на котором одновременно доступны frontend и
API. В проекте для этого есть `scripts/serve_demo.py`: он отдаёт
`frontend/index.html` и проксирует API-ручки `/query`, `/deep-research`,
`/route`, `/ready`, `/health`, `/stats`, `/entities`, `/metrics` на FastAPI.

## 1. Подготовить окружение

```powershell
copy .env.example .env
```

Заполнить `.env` своими ключами. Для текущей демо-модели через Yandex AI Studio:

```env
LLM_CLIENT_MODE=deepseek
YANDEX_GPT_MODEL_URI=gpt://<folder_id>/deepseek-v4-flash/latest
DEEPSEEK_MODEL=deepseek-v4-flash/latest
YANDEX_REASONING_EFFORT=none
```

OpenRouter-поля можно оставить в `.env` как запасной режим, удалять их не нужно.

## 2. Поднять backend

```powershell
docker compose up -d neo4j api
```

Проверить, что API жив:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/ready
```

Если Neo4j пустой на новой машине, нужно загрузить граф из готовых результатов:

```powershell
python -m scripts.ingest_results --input ner_re_extraction/result
python build_frontend_data.py
```

Если данные уже загружены в Neo4j и `frontend/data.json` актуален, этот шаг
повторять не надо.

## 3. Поднять demo server

В отдельном терминале:

```powershell
python -m scripts.serve_demo --host 127.0.0.1 --port 8090 --api http://127.0.0.1:8080
```

Проверить локально:

```powershell
Invoke-WebRequest http://127.0.0.1:8090/ready
```

Открыть:

```text
http://127.0.0.1:8090/
```

## 4. Установить cloudflared

Windows:

```powershell
winget install --id Cloudflare.cloudflared
```

macOS:

```bash
brew install cloudflared
```

Linux:

```bash
cloudflared --version
```

Если команды нет, установить по инструкции Cloudflare:
https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

## 5. Опубликовать демку

В отдельном терминале:

```powershell
cloudflared tunnel --url http://127.0.0.1:8090
```

Cloudflare напечатает URL вида:

```text
https://something-something.trycloudflare.com
```

Эту ссылку можно отправлять команде. Пока процесс `cloudflared` запущен, ссылка
работает. Если процесс закрыть, ссылка перестанет работать.

## 6. Быстрый smoke-test

Публичный health-check:

```powershell
Invoke-RestMethod https://<your-subdomain>.trycloudflare.com/ready
```

Публичный запрос:

```powershell
$body = @{
  query = "Какие способы флотационного обогащения медно-никелевых шлаков применяются в отечественной и зарубежной практике, и какое извлечение никеля достигается при флотации?"
  synthesize = $true
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri https://<your-subdomain>.trycloudflare.com/query `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Ожидаемо:

- `route` обычно `hybrid`;
- `used_llm=true`, если модель доступна;
- `rag_documents` не пустой;
- `graph_text` не пустой для графовых/hybrid-вопросов.

## 7. Частые проблемы

Если публичная страница открывается, но запросы не работают, проверьте, что
Cloudflare направлен на `8090`, а не на `8080`.

Правильно:

```powershell
cloudflared tunnel --url http://127.0.0.1:8090
```

Неполноценно для демо:

```powershell
cloudflared tunnel --url http://127.0.0.1:8080
```

Если модель отвечает нестабильно, проверьте логи API:

```powershell
docker compose logs --tail=120 api
```

Если там `429`, это лимит провайдера. Если там `invalid model_uri`, проверьте
`YANDEX_GPT_MODEL_URI` и `LLM_CLIENT_MODE`.
