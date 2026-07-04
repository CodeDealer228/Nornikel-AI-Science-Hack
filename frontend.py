"""
Nornikel Knowledge Graph — Frontend (Streamlit).

A single-file UI that:
- Connects to the FastAPI backend (api/server.py) when available
- Falls back to local JSONL (parsed_chunks/merged.jsonl) when offline
- Shows graph stats, entity browser, query interface

Run:
    streamlit run frontend.py

Works without LLM/Neo4j (offline mode reads merged.jsonl).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import requests
import streamlit as st

REPO = Path(__file__).resolve().parent
DEFAULT_API = os.environ.get("API_URL", "http://localhost:8080")
MERGED_JSONL_CANDIDATES = [
    REPO / "parsed_data" / "chunks.jsonl",
    REPO / "parsed_chunks" / "merged.jsonl",
    REPO / "extraction_results_statyi_batch.jsonl",
]

ENTITY_TYPE_OPTIONS = [
    "Material", "Substance", "Process", "Equipment", "Property",
    "Parameter", "Condition", "Experiment", "Publication",
    "TechnologySolution", "Result", "Conclusion", "Limitation",
    "Facility", "Organization", "Expert",
    "Geography", "Year", "NumericValue",
]


# ============================================================ data layer

@st.cache_resource
def get_api_client():
    return requests.Session()


def api_get(path: str, timeout: float = 5.0) -> dict | None:
    try:
        r = get_api_client().get(f"{DEFAULT_API}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def api_post(path: str, payload: dict, timeout: float = 30.0) -> dict | None:
    try:
        r = get_api_client().post(f"{DEFAULT_API}{path}", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return {"_error": r.status_code, "_text": r.text[:300]}
    except Exception as exc:
        return {"_error": "offline", "_text": str(exc)}


@st.cache_data
def load_offline_data() -> dict:
    """Load predictions from a local JSONL when API is unreachable.

    The first candidate in ``MERGED_JSONL_CANDIDATES`` that actually yields
    entities wins; a candidate that exists but has no ``entities``/``parsed``
    field (e.g. ``parsed_data/chunks.jsonl`` — the chunking output, not a
    predictions file) is skipped rather than returned empty, so the UI falls
    through to ``parsed_chunks/merged.jsonl`` (what ``scripts.ingest``
    produces). If no candidate has entities, the first existing one is
    returned as an empty fallback.
    """
    fallback: dict | None = None
    for path in MERGED_JSONL_CANDIDATES:
        if not path.exists():
            continue
        data = {"entities": [], "chunks": 0, "source": ""}
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    data["chunks"] += 1
                    # DeepSeek format: parsed.entities
                    if "parsed" in rec and isinstance(rec["parsed"], dict):
                        for ent in rec["parsed"].get("entities", []) or []:
                            data["entities"].append({
                                "type": ent.get("type", ""),
                                "canonical_name": ent.get("canonical_name", ""),
                                "source_document": rec.get("doc_id", ""),
                            })
                    # Pipeline format: entities at top level (scripts.ingest
                    # merged.jsonl uses `entity` for the canonical name, not
                    # `canonical_name` — accept both so the offline UI works
                    # against the real ingest output).
                    else:
                        for ent in rec.get("entities", []) or []:
                            data["entities"].append({
                                "type": ent.get("type", ""),
                                "canonical_name": ent.get("canonical_name") or ent.get("entity", ""),
                                "source_document": rec.get("source_document") or rec.get("source_file", ""),
                            })
            data["source"] = path.name
            if data["entities"]:
                return data
            if fallback is None:
                fallback = data
        except Exception:
            continue
    return fallback or {"entities": [], "chunks": 0, "source": ""}
    return data


# ============================================================ UI

st.set_page_config(
    page_title="Nornikel KG — R&D Explorer",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header ---------------------------------------------------------------

st.title("⛏️ Nornikel Knowledge Graph")
st.caption("Горно-металлургический R&D — интерактивный обход графа знаний")

# Sidebar (status + filters) --------------------------------------------

with st.sidebar:
    st.header("Подключение")
    api_url = st.text_input("API endpoint", value=DEFAULT_API)

    api_health = api_get("/health")
    api_ready = api_get("/ready")
    online = api_health is not None
    neo4j_ok = bool(api_ready and api_ready.get("neo4j_connected"))
    synth_ok = bool(api_ready and api_ready.get("synthesis_configured"))

    if online:
        st.success(f"API online @ `{api_url}`")
    else:
        st.warning("API недоступен — работаем в офлайн-режиме (по JSONL)")

    if online:
        st.write(f"- Neo4j: {'✅' if neo4j_ok else '❌'}")
        st.write(f"- LLM (YandexGPT): {'✅' if synth_ok else '⚠️ отключен'}")

    st.divider()
    st.caption("Хакатон • Проект KG-RAG • 2026")

# Tabs ------------------------------------------------------------------

tab_home, tab_browse, tab_query, tab_about = st.tabs([
    "📊 Главная", "🔍 Сущности", "❓ Вопрос", "ℹ️ О системе",
])

# -------- tab 1: Home / dashboard --------------------------------------

with tab_home:
    st.subheader("Сводка по графу")

    stats = api_get("/stats") if online and neo4j_ok else None
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Узлов", stats.get("total_nodes", 0))
        c2.metric("Связей", stats.get("total_relationships", 0))
        c3.metric("Типов сущностей", len(stats.get("by_label", {})))

        st.markdown("##### Распределение по типам")
        labels = stats.get("by_label", {})
        if labels:
            sorted_labels = sorted(labels.items(), key=lambda x: -x[1])
            st.bar_chart(dict(sorted_labels[:15]))
    else:
        # Offline fallback
        offline = load_offline_data()
        c1, c2, c3 = st.columns(3)
        c1.metric("Чанков обработано", offline["chunks"])
        c2.metric("Сущностей извлечено", len(offline["entities"]))
        types = {e["type"] for e in offline["entities"]}
        c3.metric("Уникальных типов", len(types))

        if offline["entities"]:
            cnt = Counter(e["type"] for e in offline["entities"])
            st.markdown("##### Распределение по типам")
            st.bar_chart(dict(cnt.most_common(15)))
        st.caption(f"Источник: {offline['source'] or 'нет данных'}")

# -------- tab 2: Browse entities --------------------------------------

with tab_browse:
    st.subheader("Обзор сущностей")

    if online and neo4j_ok:
        c1, c2, c3 = st.columns(3)
        with c1:
            name_filter = st.text_input("Поиск по имени (можно через запятую)")
        with c2:
            geo_filter = st.text_input("География (например: Russia, Worldwide)")
        with c3:
            type_filter = st.selectbox("Тип", ["(любой)"] + ENTITY_TYPE_OPTIONS)

        params = []
        if name_filter:
            for n in [x.strip() for x in name_filter.split(",") if x.strip()]:
                params.append(("name", n))
        if geo_filter:
            params.append(("geography", geo_filter))
        if type_filter and type_filter != "(любой)":
            params.append(("entity_type", type_filter))
        params.append(("limit", 50))

        if st.button("Найти"):
            with st.spinner("Запрос к Neo4j..."):
                try:
                    r = get_api_client().get(f"{api_url}/entities", params=params, timeout=10)
                    if r.status_code == 200:
                        results = r.json().get("results", [])
                        st.success(f"Найдено: {len(results)}")
                        st.dataframe(results, use_container_width=True)
                    else:
                        st.error(f"Ошибка: {r.status_code}")
                except Exception as exc:
                    st.error(f"Backend недоступен: {exc}")
    else:
        # offline: simple filter over loaded entities
        offline = load_offline_data()
        if offline["entities"]:
            types = sorted({e["type"] for e in offline["entities"] if e["type"]})
            col1, col2 = st.columns([1, 3])
            with col1:
                t = st.selectbox("Тип", ["(все)"] + types)
                search = st.text_input("Содержит (подстрока)")
            with col2:
                ents = offline["entities"]
                if t != "(все)":
                    ents = [e for e in ents if e["type"] == t]
                if search:
                    s = search.lower()
                    ents = [e for e in ents if s in e["canonical_name"].lower()]
                cnt = Counter((e["type"], e["canonical_name"]) for e in ents)
                rows = [{"type": k[0], "canonical_name": k[1], "count": v}
                        for k, v in cnt.most_common(200)]
                st.dataframe(rows, use_container_width=True, height=400)
        else:
            st.info("Нет данных. Запустите `scripts/ingest.py` или `agent/cli.py`.")

# -------- tab 3: Query -------------------------------------------------

with tab_query:
    st.subheader("Задать вопрос системе")

    examples = [
        "Какие способы очистки от свинца при pH ≤ 3.5 в мировой практике?",
        "Что извлекается из медно-никелевого сырья на НМЗ?",
        "Какие эксперименты подтверждают модель разбавления вредных газов?",
        "Какие публикации авторов из Института Гипроникель за 2022-2024?",
    ]
    q = st.text_area(
        "Вопрос на естественном языке",
        value=examples[0] if not online else "",
        height=80,
        placeholder="Например: " + examples[1],
    )

    cols = st.columns([1, 1, 3])
    do_query = cols[0].button("🔎 Спросить", type="primary")
    do_route = cols[1].button("Только маршрут")

    if do_query and q.strip():
        with st.spinner("Запрос к графу + синтез..."):
            resp = api_post("/query", {"query": q.strip(), "synthesize": True})
            if not resp or resp.get("_error") == "offline":
                st.warning(
                    "API недоступен. Демонстрационный ответ — заглушка. "
                    "После запуска `uvicorn api.server:app` вопрос пойдёт через "
                    "роутер → Cypher → LLM-синтез."
                )
                st.markdown("**Что бы произошло (без модели):**")
                st.markdown("""
- Роутер классифицировал бы вопрос → GRAPH_ONLY (numeric+geography маркеры).
- Из графа извлеклись бы узлы `Process`/`Material` с привязкой к `NumericValue(pH ≤ 3.5)`
  и `Geography("Worldwide")`.
- Синтезатор (LLM) собрал бы ответ с цитатами и источниками.
""")
            elif "_error" in resp:
                st.error(f"Backend вернул ошибку: {resp.get('_error')}\n{resp.get('_text','')}")
            else:
                a, b = st.columns([2, 3])
                with a:
                    st.markdown("##### Ответ")
                    st.markdown(resp.get("answer") or "_(пусто)_")
                    st.caption(
                        f"route=`{resp.get('route')}` "
                        f"conf={resp.get('confidence'):.2f} "
                        f"cov={resp.get('coverage_score'):.2f} "
                        f"amb={resp.get('ambiguity_score'):.2f}"
                    )
                    if resp.get("used_llm"):
                        st.success("LLM-синтез использован")
                    else:
                        st.info("LLM-синтез не использован (fallback)")
                    if resp.get("reasons"):
                        st.markdown("**Причины маршрутизации:**")
                        for r in resp["reasons"]:
                            st.markdown(f"- `{r}`")
                with b:
                    if resp.get("graph_text"):
                        with st.expander("🧩 Подграф (как видит LLM)", expanded=False):
                            st.code(resp["graph_text"], language="text")
                    if resp.get("rag_documents"):
                        st.markdown("##### RAG-документы")
                        for doc in resp["rag_documents"]:
                            with st.expander(f"📄 {doc.get('title','')} (score={doc.get('score',0):.2f})"):
                                st.markdown(doc.get("snippet", ""))
                                if doc.get("matched_entities"):
                                    st.caption("Упомянутые сущности: " +
                                               ", ".join(doc["matched_entities"]))

    if do_route and q.strip():
        with st.spinner("Только классификация..."):
            resp = api_post("/route", {"query": q.strip()})
            if resp and "_error" not in resp:
                st.json(resp)
            else:
                st.warning("API недоступен — пропуск.")

# -------- tab 4: About -------------------------------------------------

with tab_about:
    st.subheader("Архитектура")
    st.markdown("""
```
  ETL  →  Парсинг в Markdown  →  Чанкование (Natasha)
                                    ↓
                          NER + RE извлечение (LLM/YandexGPT)
                                    ↓
                          Ensemble + нормализация
                                    ↓
                          Граф знаний (Neo4j, per-type labels)
                                    ↓
   Обход подграфа →  Cypher helpers + Routing → LLM-синтез
                                    ↓
                          Агент (этот UI)
```
""")
    st.markdown("""
**Что под капотом:**
- 19 типов сущностей + 33 типа связей — закрытая онтология
- Чанкование: ~2500/4000 символов, 2 предложения overlap, section-aware
- Каждая сущность сохраняет пробенанс до char_start/char_end в исходнике
- Синонимы собираются в `aliases[]` per-узел, слияние через Neo4j `MERGE`
- Конфликты между значениями одного свойства помечаются `contradicts`
- Все ребра хранят `quote + confidence` через `(Chunk)-[:SUPPORTS]->(Relation)`
""")

    st.markdown("##### Модули")
    st.code("""
elt/               скачивание корпуса с Яндекс.Диска
parsing/           DOCX/PDF/PPTX/XLS → Markdown + изображения
chunking/          Natasha-сегментация + размеры/перекрытие
llm_pipeline_fewshot/  YandexGPT-клиент + промпт + 10 few-shot
synonym_normalization/  canonicalize_text + газеттир
ensemble/          merge Natasha + LLM
neo4j_integration/ лоадер с per-type labels
graph_reasoning/   BFS от seed + противоречия + Knowledge gaps
routing/           классификация запроса
agent/             dispatcher + synthesizer + CLI
api/server.py      FastAPI backend ← вот этот UI с ним говорит
evaluation/        precision/recall/F1 по golden set
""")

    st.markdown("##### Текущий статус хакатона")
    st.info(
        "Если LLM отключен, фронт переходит в офлайн-режим: "
        "берёт JSONL с уже извлечёнными данными и показывает их в том же UI. "
        "Архитектура и онтология полностью описаны в коде, можно поднимать прод-агенты "
        "по мере восстановления API-ключей."
    )

# Footer ----------------------------------------------------------------

st.divider()
st.caption("Backend: `uvicorn api.server:app --port 8080` • UI: `streamlit run frontend.py`")
