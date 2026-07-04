"""
Deduplicates raw per-chunk NER/RE extraction output (one or more
llm_extraction/extraction_results_*.jsonl files -- more will land there as
Обзоры/Доклады/etc. get processed) into a canonical entity graph: one node
per real-world concept with an alias list, one deduped edge per
(subject, predicate, object) triple.

Design goals (see README.md for the story of why):

  * Deterministic: a node's id depends only on its own content
    (curated canonical_id, or f"{type}::{normalized_name}"), never on
    iteration/insertion order or an auto-increment counter. Re-running on
    the same inputs reproduces byte-identical nodes.jsonl/edges.jsonl;
    adding one more input file only adds/extends entries -- it never
    renumbers existing ones. This is what makes incremental runs over a
    growing set of jsonl files safe.

  * O(N): every entity is resolved with a single hashmap lookup against
    the curated dictionary (resources/synonyms.yaml, ported from the
    search-module-update branch's SynonymExpander design) or grouped by an
    O(1) key. Nothing here is pairwise-O(N^2) over the full corpus --
    that approach was tried first and did not finish in reasonable time
    once Property attributes pushed raw mentions past ~13k.

  * Only the curated dictionary auto-merges across surface forms that
    aren't identical strings. Everything else this script *notices*
    (abbreviation-initials patterns, near-duplicate names) is written to
    resources/synonym_candidates_from_ner.yaml as a suggestion, in the
    same schema search/mine_synonym_candidates.py already uses, so a
    human can promote it into resources/synonyms.yaml -- never a silent
    merge (see the open question in README.md).

Output (graph/):
    nodes.jsonl   canonical entities, one JSON object per line, sorted by id
    edges.jsonl   canonical relations, one JSON object per line, sorted
    merges.jsonl  audit trail: which raw surface forms collapsed into which
                  node, and by which rule (for expert review)

Also writes/updates:
    resources/synonym_candidates_from_ner.yaml   suggestions, NOT auto-merged
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from canonicalizer import lang_of, normalize
from synonym_dictionary import SynonymDictionary

HERE = Path(__file__).parent
SYNONYMS_YAML = HERE.parent / "resources" / "synonyms.yaml"
CANDIDATES_OUT = HERE.parent / "resources" / "synonym_candidates_from_ner.yaml"
OUT_DIR = HERE / "graph"

FUZZY_THRESHOLD = 0.90
# Property canonical_names are measurement/parameter identifiers (numeric
# value lives in attributes_history, already exact per-mention) -- fuzzy
# string similarity between them is almost never a real synonym and was
# ~95% of all suggestions in practice, drowning out the useful ones.
FUZZY_EXCLUDE_TYPES = {"Property"}
RU_STOPWORDS = {
    "и", "в", "на", "с", "со", "из", "для", "по", "к", "у", "от", "до",
    "при", "о", "об", "за", "над", "под", "а", "но", "или", "как", "не",
}


def initials_of(name: str) -> str:
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", name)
    letters = [w[0].upper() for w in words if w.lower() not in RU_STOPWORDS]
    return "".join(letters)


def looks_like_abbreviation(name: str) -> bool:
    return bool(re.fullmatch(r"[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9\-]{1,9}", name)) and any(c.isalpha() for c in name)


def iter_raw_entities(paths):
    """Yields (file, chunk_id, doc_id, local_id, entity_dict), in a fixed,
    reproducible order (sorted file paths, then file line order)."""
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("status") != "OK" or not row.get("parsed"):
                    continue
                chunk_id = row["chunk_id"]
                doc_id = row["doc_id"]
                for ent in row["parsed"].get("entities", []):
                    yield path.name, chunk_id, doc_id, ent["local_id"], ent


def iter_raw_relations(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("status") != "OK" or not row.get("parsed"):
                    continue
                chunk_id = row["chunk_id"]
                doc_id = row["doc_id"]
                for rel in row["parsed"].get("relations", []):
                    yield chunk_id, doc_id, rel


def node_id_for(entity_type: str, canonical_name: str, syn_dict: SynonymDictionary):
    """Deterministic node identity: curated canonical_id if the curated
    dictionary knows this surface form, else a stable slug of (type, name).
    Returns (node_id, resolved_group_or_None)."""
    group = syn_dict.resolve(canonical_name)
    if group is not None:
        return group.canonical_id, group
    return f"{entity_type}::{normalize(canonical_name)}", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="*", default=None,
                         help="extraction_results_*.jsonl files (default: glob all under llm_extraction/)")
    args = parser.parse_args()

    if args.input:
        paths = sorted(Path(p) for p in args.input)
    else:
        paths = sorted(Path(HERE.parent / "llm_extraction").glob("extraction_results_*.jsonl"))
    if not paths:
        print("No input files found.", file=sys.stderr)
        return 1
    print(f"Input files: {[p.name for p in paths]}", file=sys.stderr)

    syn_dict = SynonymDictionary.load(SYNONYMS_YAML)
    print(f"Loaded curated dictionary: {len(syn_dict.groups)} groups", file=sys.stderr)

    # chunk_local_to_node: chunk_id -> {local_id: node_id}, needed to resolve
    # relations in O(1) per relation (chunk_id is globally unique in this
    # corpus -- verified empirically against extract_batch.py's naming).
    chunk_local_to_node = defaultdict(dict)
    # node_id -> accumulated cluster data
    node_types = defaultdict(lambda: defaultdict(int))       # node_id -> {type: count}
    node_names = defaultdict(lambda: defaultdict(int))       # node_id -> {raw canonical_name: count}
    node_aliases = defaultdict(dict)                         # node_id -> {normalized: {"text":.., "lang":..}}
    node_source_chunks = defaultdict(list)
    node_attr_history = defaultdict(list)
    node_curated_group = {}                                  # node_id -> SynonymGroup (if resolved)
    n_raw = 0

    for fname, chunk_id, doc_id, local_id, ent in iter_raw_entities(paths):
        n_raw += 1
        etype = ent["type"]
        cname = ent["canonical_name"]
        node_id, group = node_id_for(etype, cname, syn_dict)
        chunk_local_to_node[chunk_id][local_id] = node_id
        if group is not None:
            node_curated_group[node_id] = group
            node_types[node_id][group.type] += 1
        else:
            node_types[node_id][etype] += 1
        node_names[node_id][cname] += 1
        for surface in [cname, *ent.get("mentions", [])]:
            norm = normalize(surface)
            node_aliases[node_id].setdefault(norm, {"text": surface, "lang": lang_of(surface)})
        node_source_chunks[node_id].append({"doc_id": doc_id, "chunk_id": chunk_id, "local_id": local_id})
        if ent.get("attributes"):
            node_attr_history[node_id].append({
                "value": ent["attributes"],
                "source_chunk": {"doc_id": doc_id, "chunk_id": chunk_id, "local_id": local_id},
            })

    print(f"Raw entity mentions: {n_raw}", file=sys.stderr)
    print(f"Canonical nodes: {len(node_names)}", file=sys.stderr)

    nodes = []
    merges_log = []
    for node_id in sorted(node_names):
        group = node_curated_group.get(node_id)
        if group is not None:
            canonical_name = group.canonical_name
            node_type = group.type
            for alias_text in group.aliases:
                norm = normalize(alias_text)
                node_aliases[node_id].setdefault(norm, {"text": alias_text, "lang": lang_of(alias_text)})
        else:
            names = node_names[node_id]
            canonical_name = max(names, key=lambda n: (len(n), n))
            node_type = max(node_types[node_id].items(), key=lambda kv: (kv[1], kv[0]))[0]

        norm_canonical = normalize(canonical_name)
        aliases = sorted(
            (a for norm, a in node_aliases[node_id].items() if norm != norm_canonical),
            key=lambda a: a["text"],
        )
        source_chunks = sorted(node_source_chunks[node_id], key=lambda s: (s["doc_id"], s["chunk_id"], s["local_id"]))
        attr_history = sorted(
            node_attr_history[node_id],
            key=lambda a: (a["source_chunk"]["doc_id"], a["source_chunk"]["chunk_id"], a["source_chunk"]["local_id"]),
        )

        nodes.append({
            "id": node_id,
            "type": node_type,
            "canonical_name": canonical_name,
            "curated": group is not None,
            "aliases": aliases,
            "mention_count": len(source_chunks),
            "source_chunks": source_chunks,
            "attributes_history": attr_history,
        })

        n_raw_forms = len(node_names[node_id])
        if group is not None or n_raw_forms > 1:
            merges_log.append({
                "node_id": node_id,
                "canonical_name": canonical_name,
                "rule": "curated_dictionary" if group is not None else "exact_name",
                "merged_surface_forms": sorted(node_names[node_id]),
                "n_raw_mentions": len(source_chunks),
            })

    # --- resolve relations onto canonical node ids ---------------------
    edge_buckets = defaultdict(lambda: {"count": 0, "source_chunks": []})
    unresolved_relations = 0
    for chunk_id, doc_id, rel in iter_raw_relations(paths):
        local_map = chunk_local_to_node.get(chunk_id, {})
        subj_node = local_map.get(rel["subject"])
        obj_node = local_map.get(rel["object"])
        if subj_node is None or obj_node is None:
            unresolved_relations += 1
            continue
        edge_key = (subj_node, rel["predicate"], obj_node)
        bucket = edge_buckets[edge_key]
        bucket["count"] += 1
        bucket["source_chunks"].append({
            "doc_id": doc_id, "chunk_id": chunk_id,
            "local_subject": rel["subject"], "local_object": rel["object"],
        })
    if unresolved_relations:
        print(f"WARNING: {unresolved_relations} relations referenced an unknown local_id (skipped)", file=sys.stderr)

    edges = [
        {"subject": s, "predicate": p, "object": o, "occurrence_count": b["count"], "source_chunks": b["source_chunks"]}
        for (s, p, o), b in sorted(edge_buckets.items())
    ]

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "nodes.jsonl", "w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "edges.jsonl", "w", encoding="utf-8") as f:
        for edge in edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "merges.jsonl", "w", encoding="utf-8") as f:
        for m in merges_log:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Canonical edges: {len(edges)} (raw relations: {sum(b['count'] for b in edge_buckets.values())})", file=sys.stderr)
    print(f"Nodes actually merged (curated match or >1 raw surface form): {len(merges_log)}", file=sys.stderr)

    mine_candidates(nodes)
    return 0


def mine_candidates(nodes):
    """Suggestion-only pass over nodes NOT resolved via the curated
    dictionary: abbreviation-initials pattern + blocked fuzzy near-duplicate
    names. Writes resources/synonym_candidates_from_ner.yaml in the same
    schema as resources/synonym_candidates.yaml (mined from raw text) so a
    human curator can review both with the same workflow."""
    uncurated = [n for n in nodes if not n["curated"]]

    by_type = defaultdict(list)
    for n in uncurated:
        by_type[n["type"]].append(n)

    candidates = {}

    def add_candidate(source_type, name_a, node_a, name_b, node_b):
        key = tuple(sorted([node_a, node_b]))
        if key not in candidates:
            candidates[key] = {
                "aliases": sorted({name_a, name_b}),
                "source_type": source_type,
                "count": 0,
                "examples": [{"node_id": node_a}, {"node_id": node_b}],
            }
        candidates[key]["count"] += 1

    # abbreviation-initials: O(distinct names) via hashmap, not pairwise
    for etype, entries in by_type.items():
        initials_index = {}
        for n in entries:
            name = n["canonical_name"]
            if len(name.split()) >= 2:
                key = initials_of(name)
                if key and key not in initials_index:
                    initials_index[key] = n
        for n in entries:
            name = n["canonical_name"]
            if looks_like_abbreviation(name):
                key = name.replace("-", "").upper()
                match = initials_index.get(key)
                if match is not None and match["id"] != n["id"]:
                    add_candidate("abbreviation_initials_ner", name, n["id"], match["canonical_name"], match["id"])

    # fuzzy near-duplicate, blocked by (type, first normalized word)
    for etype, entries in by_type.items():
        if etype in FUZZY_EXCLUDE_TYPES:
            continue
        blocks = defaultdict(list)
        for n in entries:
            norm = normalize(n["canonical_name"])
            first_word = norm.split(" ", 1)[0] if norm else ""
            blocks[first_word].append(n)
        for block in blocks.values():
            if len(block) < 2:
                continue
            for i, a in enumerate(block):
                for b in block[i + 1:]:
                    if abs(len(a["canonical_name"]) - len(b["canonical_name"])) > 10:
                        continue
                    score = SequenceMatcher(None, normalize(a["canonical_name"]), normalize(b["canonical_name"])).ratio()
                    if score >= FUZZY_THRESHOLD:
                        add_candidate("fuzzy_ner", a["canonical_name"], a["id"], b["canonical_name"], b["id"])

    result = sorted(candidates.values(), key=lambda c: -c["count"])
    CANDIDATES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"candidates": result}, f, allow_unicode=True, sort_keys=False)
    print(f"Wrote {len(result)} merge suggestions to {CANDIDATES_OUT}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
