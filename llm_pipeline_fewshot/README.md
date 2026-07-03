# llm_pipeline_fewshot/ — Stage 2: Automated NER+RE pipeline design

Design for the automated LLM-based NER+RE pipeline that will eventually replace/
scale up the manual work in `../ner_re_extraction/`. **Not implemented — blocked on
API token access.** This folder is the plan + prompt design + few-shot bank to run
the moment tokens are available, so no time is lost re-deriving it then.

## Why a manual pass came first

`../ner_re_extraction/` exists specifically so this pipeline has a **verified,
ontology-correct few-shot bank** to draw on, instead of hand-written toy examples.
Every few-shot example below is a real triple pulled from a real article, already
checked against the source text.

## Requirements this pipeline must meet

1. **Chunking.** Parsed documents (`parsed_data/texts/*.md`, output of `../parsing/`)
   run from a couple pages to tens of pages. An LLM call needs a bounded window, so
   documents get split into overlapping chunks (overlap so an entity/relation whose
   two ends land on opposite sides of a chunk boundary isn't lost). Chunk boundaries
   should prefer paragraph/heading breaks over hard character cuts — the parser
   already emits headings and paragraph-level blocks, so splitting can respect that
   structure instead of cutting mid-sentence.
2. **Structured output.** The model must return JSON matching the ontology
   directly loadable into the graph — not prose to be re-parsed. Roughly:
   ```json
   {
     "entities": [
       {"type": "Equipment", "name": "печь Ванюкова конвертерная", "aliases": ["ПВК"], "span": [120, 145]}
     ],
     "relations": [
       {"type": "produces_output", "source": "печь Ванюкова конвертерная", "target": "черновая медь"}
     ]
   }
   ```
   `span` (character offsets into the chunk) is what preserves traceability back to
   the source document — required by CLAUDE.md Core Principle 1 ("every extracted
   entity/relation/fact must carry a pointer back to its source").
3. **Async + worker pool.** With ~2000 documents once the full corpus is parsed,
   calls have to run concurrently, not one at a time. This mirrors the concurrency
   pattern `../parsing/orchestrator.py` already uses (worker pool per cost profile,
   one report entry per unit of work, a failure on one document never aborts the
   batch) — the same "fail loud per item, keep the batch going" convention should
   carry over here: one bad chunk/LLM error becomes a logged failure entry, not a
   crashed run.
4. **Per-chunk provenance carried through.** Every LLM call's input should include
   the source document name + chunk offset, and the output should be joined back to
   that metadata *before* being merged into the graph — this is what lets
   `described_in`/`validated_by` point at an actual document+location rather than
   just "some article".

## Prompt skeleton

```
SYSTEM:
  You are extracting entities and relations from R&D documents in the mining/
  metallurgy domain for a knowledge graph. Use exactly these entity types:
  Material, Process, Equipment, Property, Experiment, Publication, Expert, Facility.
  Use exactly these relation types: uses_material, operates_at_condition,
  produces_output, described_in, validated_by, contradicts.
  Rules:
  - Extract concrete named entities, not category names (e.g. "печь Ванюкова
    конвертерная (ПВК)", not "оборудование").
  - Preserve numeric values with their units exactly as written — never round,
    approximate, or drop units.
  - When an entity has an abbreviation or a RU/EN synonym pair, record it as an
    alias of one canonical entity, not two separate entities.
  - Return valid JSON matching the provided schema. No prose.

  [few-shot examples: 2-3 full entity+relation blocks pulled verbatim from
   ../ner_re_extraction/ner_re_examples.md, chosen to cover different entity/
   relation types and at least one `contradicts` case]

USER:
  Document: {source_filename}
  Chunk offset: {start}-{end}
  ---
  {chunk_text}
```

## Status

Design only. Once API tokens are available: implement `run.py` (async, worker
pool matching `../parsing/orchestrator.py`'s pattern), wire the few-shot block to
pull directly from `../ner_re_extraction/ner_re_examples.md` rather than being
copy-pasted (so future manual annotations automatically improve the few-shot bank),
and validate a sample of LLM output against the manual annotations as a quality
check before running the full corpus.
