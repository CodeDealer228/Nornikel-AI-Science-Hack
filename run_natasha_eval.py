"""
F1 evaluation against golden_set/golden_set (2).jsonl.

Compares predictions (Natasha-only by default, or external JSONL via
--predictions) against golden samples. Outputs sklearn-style reports
for both NER and RE with Micro / Macro / Weighted F1.

Prediction JSONL format (when --predictions given):
  { "source_file": "...", "char_start": N, "char_end": M,
    "entities": [{"type": "...", "canonical_name": "..."}],
    "relations": [{"subject_type": "...", "subject": "...",
                  "predicate": "...", "object_type": "...", "object": "..."}] }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

# Windows console defaults to cp1251; force UTF-8 so Cyrillic output and the
# arrows/em-dashes in the report don't crash with UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

from chunking.chunker import build_raw_chunks
from chunking.config import default_config
from chunking.natasha_pipeline import get_pipeline
from synonym_normalization.canonicalizer import canonicalize_text

REPO = Path(__file__).resolve().parent
GOLDEN_DEFAULT = REPO / "golden_set" / "golden_set (2).jsonl"
ARTICLES_DIR = REPO / "Статьи"

NATASHA_TYPE_MAP = {"PER": "Expert", "LOC": "Facility", "ORG": "Organization"}


def load_golden(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gold = rec.get("gold", {}) or {}
            ents = gold.get("entities", []) or []
            rels = gold.get("relations", []) or []
            samples.append({
                "sample_id": rec.get("chunk_id", ""),
                "source_file": rec.get("source_file", ""),
                "char_start": rec.get("char_start", 0),
                "char_end": rec.get("char_end", 0),
                "entities": ents,
                "relations": rels,
            })
    return samples


def _resolve_article_path(source_file: str) -> Path | None:
    target = source_file
    for path in ARTICLES_DIR.rglob("*.md"):
        if path.name == target:
            return path
    cleaned_target = re.sub(r"[^а-яА-Яa-zA-Z0-9]+", "", target).lower()
    for path in ARTICLES_DIR.rglob("*.md"):
        cleaned_path = re.sub(r"[^а-яА-Яa-zA-Z0-9]+", "", path.name).lower()
        if cleaned_target and cleaned_path and (
            cleaned_target in cleaned_path or cleaned_path in cleaned_target
        ):
            return path
    return None


def natasha_predictions_for_sample(sample: dict, pipeline) -> list[tuple[str, str]]:
    """Return [(canonical_name, type)] from Natasha for one sample."""
    article = _resolve_article_path(sample["source_file"])
    if article is None:
        return []

    try:
        text = article.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = article.read_text(encoding="cp1251", errors="replace")

    cfg = default_config()
    raws = build_raw_chunks(text, pipeline, cfg)
    predictions: list[tuple[str, str]] = []
    s_lo, s_hi = sample["char_start"], sample["char_end"]

    for chunk in raws:
        if chunk.char_end < s_lo or chunk.char_start > s_hi:
            continue
        overlap = max(0, min(chunk.char_end, s_hi) - max(chunk.char_start, s_lo))
        if overlap < (s_hi - s_lo) * 0.3:
            continue
        try:
            ann = pipeline.annotate(chunk.text)
        except Exception:
            continue
        if not ann.ner_available:
            continue
        for ent in ann.primary_entities or []:
            mapped = NATASHA_TYPE_MAP.get(ent.type)
            if not mapped:
                continue
            raw_name = (ent.normal or ent.text or "").strip()
            if not raw_name:
                continue
            canon = canonicalize_text(raw_name)
            if canon:
                predictions.append((canon, mapped))

    seen = set(); uniq = []
    for name, t in predictions:
        k = f"{t}:{name}"
        if k in seen: continue
        seen.add(k); uniq.append((name, t))
    return uniq


def load_external_predictions(path: Path, samples: list[dict]):
    """Match external predictions to golden samples by (source_file, char overlap).

    Accepts two entity/relation field-name conventions:
      * "flat" eval format — ``source_file``, ``canonical_name``,
        ``subject``/``predicate``/``object`` + ``subject_type``/``object_type``;
      * ``scripts.ingest`` ``merged.jsonl`` format — ``source_document``,
        ``entity`` (canonical name), ``source_entity``/``relation_type``/
        ``target_entity`` + ``source_entity_type``/``target_entity_type``.
    The second is what ``python -m scripts.ingest`` actually writes, so the
    eval can be pointed straight at ``parsed_chunks/merged.jsonl``.
    """
    by_window: dict[tuple[str, int, int], dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sf = rec.get("source_file") or rec.get("source_document") or ""
            cs = int(rec.get("char_start", 0))
            ce = int(rec.get("char_end", 0))
            by_window.setdefault((sf, cs, ce), {"entities": [], "relations": []})
            by_window[(sf, cs, ce)]["entities"].extend(rec.get("entities", []) or [])
            by_window[(sf, cs, ce)]["relations"].extend(rec.get("relations", []) or [])

    ent_preds: dict[str, set[tuple[str, str]]] = {}
    rel_preds: dict[str, set[tuple[str, str]]] = {}

    for sample in samples:
        s_lo, s_hi = sample["char_start"], sample["char_end"]
        # find best-overlapping window for this sample
        best_key = None; best_overlap = 0
        for key in by_window.keys():
            ksf, kcs, kce = key
            if ksf != sample["source_file"]:
                continue
            ov = max(0, min(kce, s_hi) - max(kcs, s_lo))
            if ov > best_overlap:
                best_overlap = ov; best_key = key
        if best_key is None or best_overlap < (s_hi - s_lo) * 0.3:
            continue

        rec = by_window[best_key]
        ents = set()
        for e in rec["entities"]:
            t = str(e.get("type", "")).strip()
            n = canonicalize_text(e.get("canonical_name") or e.get("entity") or "")
            if t and n:
                ents.add((t, n))
        if ents:
            ent_preds[sample["sample_id"]] = ents

        rels = set()
        for r in rec["relations"]:
            subject = r.get("subject") or r.get("source_entity") or ""
            obj = r.get("object") or r.get("target_entity") or ""
            pred = r.get("predicate") or r.get("relation_type") or ""
            sub_t = r.get("subject_type") or r.get("source_entity_type") or ""
            obj_t = r.get("object_type") or r.get("target_entity_type") or ""
            triple = (
                str(sub_t).strip(),
                canonicalize_text(subject),
                str(pred).strip(),
                str(obj_t).strip(),
                canonicalize_text(obj),
            )
            if all([triple[0], triple[1], triple[2], triple[3], triple[4]]):
                rels.add(triple)
        if rels:
            rel_preds[sample["sample_id"]] = rels

    return ent_preds, rel_preds


def prf_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate_ner(samples, ent_preds, all_types):
    """Returns dict[type] = {tp, fp, fn} and aggregates."""
    per_type_tp = Counter(); per_type_fp = Counter(); per_type_fn = Counter()
    total_tp = total_fp = total_fn = 0

    for sample in samples:
        gold = set()
        for e in sample["entities"]:
            t = str(e.get("type", "")).strip()
            n = canonicalize_text(e.get("canonical_name", ""))
            if t and n:
                gold.add((t, n))
        pred = ent_preds.get(sample["sample_id"], set())

        for (t, n) in pred & gold:
            total_tp += 1; per_type_tp[t] += 1
        for (t, n) in pred - gold:
            total_fp += 1; per_type_fp[t] += 1
        for (t, n) in gold - pred:
            total_fn += 1; per_type_fn[t] += 1

    return total_tp, total_fp, total_fn, per_type_tp, per_type_fp, per_type_fn


def evaluate_re(samples, rel_preds, all_predicates):
    """Returns dict[predicate] = {tp, fp, fn} and aggregates."""
    per_type_tp = Counter(); per_type_fp = Counter(); per_type_fn = Counter()
    total_tp = total_fp = total_fn = 0

    def rel_hash(sub_t, sub, pred, obj_t, obj):
        return f"{sub_t}:{sub}|[{pred}]|{obj_t}:{obj}"

    for sample in samples:
        # build local_id -> (type, canonical_name) map
        loc_to_ent: dict[str, tuple[str, str]] = {}
        for e in sample["entities"]:
            lid = e.get("local_id", "")
            t = str(e.get("type", "")).strip()
            n = canonicalize_text(e.get("canonical_name", ""))
            if lid and t and n:
                loc_to_ent[lid] = (t, n)

        gold: set[tuple] = set()
        for rel in sample["relations"]:
            try:
                s_lid = rel["subject"]; o_lid = rel["object"]
                pred = str(rel["predicate"]).strip()
            except KeyError:
                continue
            if s_lid in loc_to_ent and o_lid in loc_to_ent and pred:
                s_t, s_n = loc_to_ent[s_lid]
                o_t, o_n = loc_to_ent[o_lid]
                gold.add((s_t, s_n, pred, o_t, o_n))

        pred = rel_preds.get(sample["sample_id"], set())

        for triple in pred & gold:
            total_tp += 1; per_type_tp[triple[2]] += 1
        for triple in pred - gold:
            total_fp += 1; per_type_fp[triple[2]] += 1
        for triple in gold - pred:
            total_fn += 1; per_type_fn[triple[2]] += 1

    return total_tp, total_fp, total_fn, per_type_tp, per_type_fp, per_type_fn


def report_block(title, classes, per_type_tp, per_type_fp, per_type_fn, micro_tp, micro_fp, micro_fn):
    """Print sklearn-style report for one block (NER or RE)."""
    print(f"=== {title} ===")
    if not classes:
        print("  (no classes found)")
        return

    rows = []
    for c in classes:
        tp = per_type_tp.get(c, 0)
        fp = per_type_fp.get(c, 0)
        fn = per_type_fn.get(c, 0)
        support = tp + fn  # how many are actually in golden (or sum for symmetric)
        p, r, f = prf_from_counts(tp, fp, fn)
        rows.append((c, p, r, f, tp, fp, fn, support))

    mp, mr, mf = prf_from_counts(micro_tp, micro_fp, micro_fn)
    macro_f = sum(row[3] for row in rows) / len(rows) if rows else 0.0
    total_support = sum(row[7] for row in rows)
    weighted_f = (
        sum(row[3] * row[7] for row in rows) / total_support if total_support else 0.0
    )

    print(f"Классы: {list(classes)}")
    print(f"Micro F1:   {mf:.4f}")
    print(f"Macro F1:   {macro_f:.4f}")
    print(f"Weighted F1:{weighted_f:.4f}")
    print(f"По классам:")
    # sort by support desc, then name
    for c, p, r_, f, tp, fp, fn, support in sorted(rows, key=lambda x: (-x[7], x[0])):
        print(f"  {c:<24s}: F1={f:.4f} (support={support})")


def _doc_id_to_source_file(doc_id: str) -> str:
    """Find .md in ARTICLES_DIR matching doc_id (substring + cleaned match)."""
    if not doc_id:
        return ""
    target = re.sub(r"[^а-яА-Яa-zA-Z0-9]+", "", doc_id).lower()
    for path in ARTICLES_DIR.rglob("*.md"):
        cleaned = re.sub(r"[^а-яА-Яa-zA-Z0-9]+", "", path.name).lower()
        if cleaned and target and (
            cleaned.startswith(target) or target.startswith(cleaned)
        ):
            return path.name
    return ""


def load_deepseek_predictions(path: Path, samples: list[dict]):
    """DeepSeek-style output with local_id refs; matches per-article via doc_id.

    DeepSeek records carry no char range/text, so we pool entities/relations
    per article (by mapping doc_id -> source_file) and attach each golden
    sample only to its article's pool. This avoids the corpus-level pool
    inflation that produced near-zero precision in earlier runs.
    """
    article_entities: dict[str, set[tuple[str, str]]] = defaultdict(set)
    article_relations: dict[str, set[tuple]] = defaultdict(set)
    n_chunks = 0
    n_unmapped = 0
    used_docs = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = rec.get("doc_id", "")
            sf = _doc_id_to_source_file(doc_id)
            if not sf:
                n_unmapped += 1
                continue
            used_docs.add(sf)
            parsed = rec.get("parsed", {}) or {}
            raw_ents = parsed.get("entities", []) or []
            raw_rels = parsed.get("relations", []) or []
            if not raw_ents:
                continue
            n_chunks += 1

            loc: dict[str, tuple[str, str]] = {}
            for e in raw_ents:
                lid = str(e.get("local_id", "")).strip()
                t = str(e.get("type", "")).strip()
                n = canonicalize_text(e.get("canonical_name", ""))
                if lid and t and n:
                    loc[lid] = (t, n)
                    article_entities[sf].add((t, n))
            for r in raw_rels:
                try:
                    s_lid = str(r["subject"]).strip()
                    o_lid = str(r["object"]).strip()
                    pred = str(r["predicate"]).strip()
                except KeyError:
                    continue
                if s_lid in loc and o_lid in loc and pred:
                    s_t, s_n = loc[s_lid]
                    o_t, o_n = loc[o_lid]
                    article_relations[sf].add((s_t, s_n, pred, o_t, o_n))

    print(f"  -> deepseek: {n_chunks} chunk(s) in {len(used_docs)} article(s) "
          f"({n_unmapped} chunks unmapped)", flush=True)

    ent_preds: dict[str, set[tuple[str, str]]] = {}
    rel_preds: dict[str, set[tuple]] = {}
    matched = 0
    for sample in samples:
        sf = sample["source_file"]
        if sf in article_entities:
            ent_preds[sample["sample_id"]] = article_entities[sf]
            rel_preds[sample["sample_id"]] = article_relations[sf]
            matched += 1
    print(f"  -> matched {matched}/{len(samples)} golden sample(s) to articles", flush=True)
    return ent_preds, rel_preds


def detect_predictions_format(path: Path) -> str:
    """Detect 'deepseek' (nested `parsed`) vs 'flat' (top-level entities).

    ``flat`` covers both the eval-style record (``source_file``) and the
    ``scripts.ingest`` ``merged.jsonl`` record (``source_document``) —
    ``load_external_predictions`` accepts either field-name set.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "parsed" in rec and isinstance(rec.get("parsed"), dict):
                return "deepseek"
            if "entities" in rec and ("source_file" in rec or "source_document" in rec):
                return "flat"
            return "unknown"
    return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(GOLDEN_DEFAULT), type=Path)
    ap.add_argument("--articles", default=str(ARTICLES_DIR), type=Path)
    ap.add_argument("--predictions", default=None, type=Path)
    args = ap.parse_args(argv)

    print(f"Loading golden: {args.golden}", flush=True)
    samples = load_golden(args.golden)
    print(f"  -> {len(samples)} sample(s)", flush=True)

    # determine all entity types and predicates appearing in golden
    gold_entity_types: set[str] = set()
    gold_predicates: set[str] = set()
    for sample in samples:
        for e in sample["entities"]:
            t = str(e.get("type", "")).strip()
            if t: gold_entity_types.add(t)
        for r in sample["relations"]:
            p = str(r.get("predicate", "")).strip()
            if p: gold_predicates.add(p)

    if not args.predictions:
        print("Loading Natasha pipeline...", flush=True)
        pipeline = get_pipeline()
        ent_preds: dict[str, set[tuple[str, str]]] = {}
        rel_preds: dict[str, set[tuple]] = {}
        print(f"Running Natasha on {len(samples)} samples...", flush=True)
        for i, sample in enumerate(samples, 1):
            if i % 10 == 0 or i == len(samples):
                print(f"  [{i}/{len(samples)}] {sample['sample_id']}", flush=True)
            preds_list = natasha_predictions_for_sample(sample, pipeline)
            # preds_list is [(name, type)]; evaluation expects (type, name)
            if preds_list:
                ent_preds[sample["sample_id"]] = {(t, n) for n, t in preds_list}
        natasha_types = {t for (t, n) in {x for s in ent_preds.values() for x in s}}
        all_ent_classes = sorted(gold_entity_types | natasha_types)
        all_rel_classes = sorted(gold_predicates)
    else:
        fmt = detect_predictions_format(args.predictions)
        print(f"Loading external predictions ({fmt}): {args.predictions}", flush=True)
        if fmt == "deepseek":
            ent_preds, rel_preds = load_deepseek_predictions(args.predictions, samples)
        else:
            ent_preds, rel_preds = load_external_predictions(args.predictions, samples)
        print(f"  -> matched {len(ent_preds)} entity sets, {len(rel_preds)} relation sets",
              flush=True)
        # determine classes from both gold + predictions
        ext_ent_types = set()
        for ents in ent_preds.values():
            for t, _ in ents:
                ext_ent_types.add(t)
        ext_pred_types = set()
        for rels in rel_preds.values():
            for triple in rels:
                ext_pred_types.add(triple[2])
        all_ent_classes = sorted(gold_entity_types | ext_ent_types)
        all_rel_classes = sorted(gold_predicates | ext_pred_types)
        if fmt == "deepseek":
            print("  (deepseek format: local_id resolved, corpus-level matching — see notes)", flush=True)

    # NER
    tp, fp_, fn, ptype_tp, ptype_fp, ptype_fn = evaluate_ner(samples, ent_preds, all_ent_classes)
    report_block("NER", all_ent_classes, ptype_tp, ptype_fp, ptype_fn, tp, fp_, fn)

    # RE
    rtp, rfp, rfn, ptp, pfp, pfn_ = evaluate_re(samples, rel_preds, all_rel_classes)
    report_block("RE", all_rel_classes, ptp, pfp, pfn_, rtp, rfp, rfn)

    print()
    if not args.predictions:
        print("Note: Natasha detects only PER/LOC/ORG -> Expert/Facility/Organization.")
        print("      For full type coverage use --predictions <jsonl>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
