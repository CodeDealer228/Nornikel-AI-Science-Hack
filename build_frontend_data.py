"""Aggregate extraction JSONL into a single data.json for the static frontend.

Reads the DeepSeek-format batch (or scripts.ingest merged.jsonl) and emits
frontend/data.json with everything the SPA needs to render dashboards, graph,
search, filters, experts, contradictions, knowledge gaps — without a backend.

Run:
    python build_frontend_data.py
    python -m http.server -d frontend 8050   # then open http://localhost:8050
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows console cp1251 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent
CANDIDATES = [
    REPO / "parsed_chunks" / "ner_re_results" / "merged.jsonl",
    REPO / "extraction_results_statyi_batch.jsonl",
    REPO / "parsed_chunks" / "merged.jsonl",
    REPO / "extraction_results_obzory_batch.jsonl",
]
OUT = REPO / "frontend" / "data.json"

ENTITY_TYPES = [
    "Material", "Substance", "Process", "Equipment", "Property",
    "Parameter", "Condition", "Experiment", "Publication",
    "TechnologySolution", "Result", "Conclusion", "Limitation",
    "Facility", "Organization", "Expert",
]


def _norm(name: str) -> str:
    return " ".join((name or "").split()).strip()


def load_records() -> list[dict]:
    for path in CANDIDATES:
        if not path.exists():
            continue
        recs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if recs:
            print(f"[build] loaded {len(recs)} records from {path.name}")
            return recs
    raise SystemExit("No extraction JSONL found.")


def main() -> None:
    records = load_records()

    # --- entity index: (type, canonical_name) -> aggregated entity ---
    ent_map: dict[tuple[str, str], dict] = {}
    edges: list[dict] = []
    docs: set[str] = set()
    chunk_count = 0
    numeric_values: list[dict] = []
    contradictions: list[dict] = []

    def get_ent(type_: str, name: str, doc: str) -> dict:
        key = (type_, _norm(name).lower())
        if key not in ent_map:
            ent_map[key] = {
                "id": f"{type_}::{key[1]}",
                "type": type_,
                "name": _norm(name),
                "docs": set(),
                "mentions": set(),
                "degree": 0,
                "attributes": {},
                "confidence": 0.0,
                "n_conf": 0,
            }
        e = ent_map[key]
        if doc:
            e["docs"].add(doc)
        return e

    for rec in records:
        chunk_count += 1
        doc = rec.get("doc_id") or rec.get("source_document") or rec.get("source_file") or ""
        if doc:
            docs.add(doc)
        parsed = rec.get("parsed", {}) or {}
        raw_ents = parsed.get("entities", []) or rec.get("entities", []) or []
        raw_rels = parsed.get("relations", []) or rec.get("relations", []) or []

        # build local_id -> entity for this chunk
        local: dict[str, dict] = {}
        for e in raw_ents:
            t = str(e.get("type", "")).strip()
            n = _norm(str(e.get("canonical_name") or e.get("entity") or ""))
            if not t or not n:
                continue
            ent = get_ent(t, n, doc)
            ent["n_conf"] += 1
            ent["confidence"] = max(ent["confidence"], 0.85)
            for m in (e.get("mentions") or [])[:8]:
                if m:
                    ent["mentions"].add(str(m)[:80])
            a = e.get("attributes") or {}
            if a:
                # keep latest non-empty attrs
                for k, v in a.items():
                    if v not in (None, "", [], {}):
                        ent["attributes"][k] = v
            lid = str(e.get("local_id") or "")
            if lid:
                local[lid] = {"ent": ent, "raw": e, "doc": doc}

            # numeric harvest
            if any(k in a for k in ("value", "numeric_value", "value_raw")) or ("min" in a and "max" in a):
                numeric_values.append({
                    "entity": ent["name"],
                    "type": t,
                    "property": a.get("property_name") or a.get("name") or (t if t in ("Property", "Parameter", "Condition") else ""),
                    "value": _num(a.get("value") or a.get("numeric_value")),
                    "unit": str(a.get("unit") or _unit_from_raw(a.get("value_raw")) or ""),
                    "operator": str(a.get("operator") or ""),
                    "min": _num(a.get("min")),
                    "max": _num(a.get("max")),
                    "value_raw": str(a.get("value_raw") or ""),
                    "doc": doc,
                })

        for r in raw_rels:
            pred = str(r.get("predicate") or r.get("relation_type") or "").strip()
            if not pred:
                continue
            # resolve endpoints via local_id, else by name
            s_lid = str(r.get("subject") or r.get("source_local_id") or "")
            o_lid = str(r.get("object") or r.get("target_local_id") or "")
            s_ent = local.get(s_lid, {}).get("ent")
            o_ent = local.get(o_lid, {}).get("ent")
            if s_ent is None and (r.get("subject_type") or r.get("source_entity_type")):
                s_ent = get_ent(
                    str(r.get("subject_type") or r.get("source_entity_type")),
                    str(r.get("subject_name") or r.get("source_entity") or s_lid),
                    doc,
                )
            if o_ent is None and (r.get("object_type") or r.get("target_entity_type")):
                o_ent = get_ent(
                    str(r.get("object_type") or r.get("target_entity_type")),
                    str(r.get("object_name") or r.get("target_entity") or o_lid),
                    doc,
                )
            if s_ent is None or o_ent is None:
                continue
            edge = {
                "s": s_ent["id"], "t": o_ent["id"],
                "p": pred, "doc": doc,
                "quote": str(r.get("quote") or "")[:160],
            }
            edges.append(edge)
            s_ent["degree"] += 1
            o_ent["degree"] += 1
            if pred == "contradicts":
                contradictions.append({
                    "subject": s_ent["name"], "subject_type": s_ent["type"],
                    "object": o_ent["name"], "object_type": o_ent["type"],
                    "doc": doc, "quote": edge["quote"],
                })

    # --- finalize entities ---
    entities = []
    for e in ent_map.values():
        e["docs"] = sorted(e["docs"])[:12]
        e["mentions"] = sorted(e["mentions"])[:8]
        e["doc_count"] = len(e["docs"])
        e["confidence"] = round(min(1.0, 0.6 + 0.05 * math.log1p(e["n_conf"])), 3)
        entities.append(e)
    entities.sort(key=lambda x: (-x["degree"], -x["doc_count"], x["name"]))

    # --- stats ---
    by_type = Counter(e["type"] for e in entities)
    by_pred = Counter(ed["p"] for ed in edges)
    top_docs = Counter()
    for e in entities:
        for d in e["docs"]:
            top_docs[d] += 1

    # --- experts & facilities ---
    experts = [e for e in entities if e["type"] == "Expert"][:60]
    facilities = [e for e in entities if e["type"] in ("Facility", "Organization")][:60]
    # wire expert -> org/facility via edges
    eid_by_id = {e["id"]: e for e in entities}
    for ex in experts:
        ex["affiliations"] = []
        ex["publications"] = 0
    ex_by_id = {ex["id"]: ex for ex in experts}
    for ed in edges:
        if ed["p"] in ("affiliated_with", "used_in_facility", "performed_by"):
            s, t = ex_by_id.get(ed["s"]), eid_by_id.get(ed["t"])
            if s and t:
                s["affiliations"].append(t["name"])
        if ed["p"] == "authored_by":
            # subject is Publication, object is Expert
            t = ex_by_id.get(ed["t"])
            if t:
                t["publications"] += 1

    # --- graph sample (top connected entities + their ego edges) ---
    top_ids = {e["id"] for e in entities[:120]}
    g_nodes = []
    for e in entities:
        if e["id"] in top_ids or e["degree"] >= 3:
            g_nodes.append(e)
            if len(g_nodes) >= 140:
                break
    g_ids = {n["id"] for n in g_nodes}
    g_edges = [ed for ed in edges if ed["s"] in g_ids and ed["t"] in g_ids][:400]

    # --- chains: Material -uses_material-> Process -produces_output/uses_equipment-> X ---
    by_id = {e["id"]: e for e in entities}
    proc_ids = {e["id"] for e in entities if e["type"] == "Process"}
    out_by_src = defaultdict(list)
    for ed in edges:
        out_by_src[ed["s"]].append(ed)
    chains = []
    for pid in proc_ids:
        outs = out_by_src.get(pid, [])
        mats = [ed for ed in outs if ed["p"] == "uses_material"]
        results = [ed for ed in outs if ed["p"] in ("produces_output",)]
        equip = [ed for ed in outs if ed["p"] == "uses_equipment"]
        if mats and (results or equip):
            chains.append({
                "process": by_id[pid]["name"],
                "materials": [by_id[m["t"]]["name"] for m in mats[:3]],
                "equipment": [by_id[e["t"]]["name"] for e in equip[:2]],
                "results": [by_id[r["t"]]["name"] for r in results[:3]],
                "doc": (mats or results or equip)[0]["doc"],
            })
        if len(chains) >= 40:
            break

    # --- knowledge gaps ---
    gaps = []
    # isolated entities
    isolated = [e for e in entities if e["degree"] == 0][:25]
    for e in isolated:
        gaps.append({
            "code": "isolated_entity",
            "severity": "warning",
            "message": f"Сущность «{e['name']}» ({e['type']}) не связана ни с чем в графе.",
            "entity": e["name"], "type": e["type"],
        })
    # material-process coverage matrix gaps
    mats = [e for e in entities if e["type"] == "Material"][:40]
    procs = [e for e in entities if e["type"] == "Process"][:40]
    covered = set()
    for ed in edges:
        if ed["p"] == "uses_material":
            covered.add((ed["s"], ed["t"]))
    missing_combos = 0
    for p in procs[:20]:
        for m in mats[:20]:
            if (p["id"], m["id"]) not in covered and (m["id"], p["id"]) not in covered:
                missing_combos += 1
    if missing_combos:
        gaps.append({
            "code": "material_process_unstudied",
            "severity": "error",
            "message": f"Из {20*20} пар «процесс × материал» в топе не покрыто {missing_combos} — потенциальная зона для исследований.",
        })
    # geography gap: corpus is Russian-only
    gaps.append({
        "code": "no_foreign_sources",
        "severity": "error",
        "message": "В корпусе нет зарубежных источников — сравнительный запрос «отечественная vs мировая практика» не имеет зарубежной половины. Рекомендуется подключить англоязычные публикации.",
    })
    # low-confidence
    low_conf = [e for e in entities if e["confidence"] < 0.7 and e["degree"] >= 2][:15]
    for e in low_conf:
        gaps.append({
            "code": "low_confidence",
            "severity": "warning",
            "message": f"«{e['name']}» — низкая уверенность ({e['confidence']}), требуется верификация.",
            "entity": e["name"], "type": e["type"],
        })

    # --- dashboards ---
    # coverage by domain (heuristic: keyword in doc name)
    domains = {
        "Гидрометаллургия": ["выщелач", "электроэкстр", "сорбц", "католит", "раствор"],
        "Пирометаллургия": ["плавк", "конверт", "штейн", "печь", "обжиг"],
        "Экология": ["вод", "очистк", "выброс", "сток", "сульфат", "хвост"],
        "Переработка отходов": ["отход", "шлак", "хвост", "вторич", "регенер"],
        "Геомеханика/безопасность": ["удароопас", "сейсм", "горн", "вентиляц", "пород"],
    }
    coverage = {}
    for dom, kws in domains.items():
        cnt = 0
        for d in docs:
            dl = d.lower()
            if any(k in dl for k in kws):
                cnt += 1
        coverage[dom] = cnt
    # risk zones: docs with few entities or contradictions
    doc_ent = Counter()
    for e in entities:
        for d in e["docs"]:
            doc_ent[d] += 1
    risk_zones = []
    for d in docs:
        ent_n = doc_ent.get(d, 0)
        ctr_n = sum(1 for c in contradictions if c["doc"] == d)
        if ent_n < 40 or ctr_n > 0:
            risk_zones.append({"doc": d, "entities": ent_n, "contradictions": ctr_n})
    risk_zones.sort(key=lambda x: (-x["contradictions"], x["entities"]))
    # team activity: experts by publication count
    team = sorted(experts, key=lambda x: -x["publications"])[:12]

    # --- numeric for filter: group by property ---
    numeric_props = Counter(n["property"] or n["unit"] or "— " for n in numeric_values)

    out = {
        "meta": {
            "source_file": next((p.name for p in CANDIDATES if p.exists()), ""),
            "doc_count": len(docs),
            "chunk_count": chunk_count,
            "entity_count": len(entities),
            "relation_count": len(edges),
            "numeric_count": len(numeric_values),
            "contradiction_count": len(contradictions),
            "expert_count": sum(1 for e in entities if e["type"] == "Expert"),
            "facility_count": sum(1 for e in entities if e["type"] in ("Facility", "Organization")),
        },
        "stats": {
            "entitiesByType": dict(by_type.most_common()),
            "relationsByPredicate": dict(by_pred.most_common()),
            "topDocs": dict(top_docs.most_common(15)),
            "coverage": coverage,
        },
        "entities": [
            {**{k: v for k, v in e.items() if k in ("id","type","name","degree","doc_count","confidence","mentions","docs","attributes")}}
            for e in entities[:1500]
        ],
        "graph": {"nodes": g_nodes, "edges": g_edges, "chains": chains},
        "experts": [
            {"name": e["name"], "affiliations": list(dict.fromkeys(e["affiliations"]))[:5],
             "publications": e["publications"], "doc_count": e["doc_count"]}
            for e in experts
        ],
        "facilities": [{"name": e["name"], "type": e["type"], "doc_count": e["doc_count"]} for e in facilities],
        "contradictions": contradictions[:50],
        "gaps": gaps,
        "numeric": numeric_values[:2000],
        "numericProps": dict(numeric_props.most_common(30)),
        "dashboards": {
            "coverage": coverage,
            "riskZones": risk_zones[:15],
            "team": [{"name": t["name"], "publications": t["publications"]} for t in team],
        },
        "entityTypes": ENTITY_TYPES,
        "predicates": sorted(by_pred.keys()),
        "geographies": [{"name": "Россия / отечественная", "kind": "Russia", "count": len(docs)}],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = OUT.stat().st_size // 1024
    print(f"[build] wrote {OUT} ({size_kb} KB)")
    print(f"[build] {len(entities)} entities, {len(edges)} edges, {len(numeric_values)} numeric, "
          f"{len(contradictions)} contradictions, {len(gaps)} gaps, {len(chains)} chains")


def _num(v) -> float | None:
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _unit_from_raw(raw: str | None) -> str:
    if not raw:
        return ""
    import re
    m = re.search(r"(мг/л|г/л|г/дм3|мг/дм3|°C|%|м3/ч|т/сут|кВт|МПа|А/м2|В/м3|кДж|ppm|об%)", raw)
    return m.group(1) if m else ""


if __name__ == "__main__":
    main()
