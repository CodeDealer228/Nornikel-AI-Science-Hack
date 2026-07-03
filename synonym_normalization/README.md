# synonym_normalization/ — Stage 2: The synonym problem

A problem statement + proposed design for handling entity synonymy, written up
after it surfaced repeatedly during manual NER/RE annotation (see
`../ner_re_extraction/`). **Not implemented yet** — this is the design that the
extraction pipeline and graph schema both need to honor once built, per the
hackathon brief's explicit requirement (CLAUDE.md Core Principle 5: *"RU and EN
terms for the same concept must resolve to one entity, not two... normalize
synonyms during extraction, not as a bolt-on later"*).

## The problem

Every recurring entity in this corpus has multiple surface forms, and different
articles/authors pick different ones freely, often in the same sentence:

| Canonical concept | Surface forms actually seen in the corpus |
|---|---|
| Печь Ванюкова конвертерная | `ПВК`, `печь Ванюкова конвертерная`, "конвертерная печь Ванюкова" |
| Печь Ванюкова обеднительная | `ПВО`, `печь Ванюкова обеднительная`, "обеднительная печь Ванюкова" |
| Цех электролиза никеля №2 | `ЦЭН-2`, "цех электролиза никеля №2" |
| Атомно-эмиссионная спектрометрия с индуктивно связанной плазмой | `ИСП-АЭС`, `ICP-AES` (RU/EN pair) |
| Электроэкстракция | `electrowinning` (RU/EN pair — named explicitly in the hackathon brief itself) |
| Печь взвешенной плавки | `ПВП`, `fluidized bed furnace` (RU abbreviation **and** an EN translation for the same equipment — also named in the brief) |
| Порошок никелевый трубчатых печей | `ПНТП` |
| Кислородно-воздушная смесь | `КВС` |
| АО «Кольская ГМК» | "Кольская ГМК", "Кольская гидрометаллургическая компания" |
| Институт Гипроникель | `ООО «Институт Гипроникель»`, "Гипроникель" |

This is not a handful of special cases — it's the *default* state of technical
writing in this domain. Any query like "какие технические решения... описаны в
мировой практике" (a query form the brief requires supporting) will silently miss
half the relevant literature if `ПВК` and `печь Ванюкова конвертерная` are stored
as two unrelated nodes.

## Why this has to be handled at extraction time, not after

If the NER step stores whatever surface string it saw as the node's identity, the
graph ends up with N near-duplicate nodes for one real-world entity, each with a
*partial* set of the relations/facts that actually belong to the single concept.
Merging them after the fact requires re-deriving which surface forms co-refer —
strictly harder than getting it right once during extraction, when the
abbreviation and its expansion are usually sitting in the same paragraph (often
literally "ПВК (печь Ванюкова конвертерная)").

## Proposed design

1. **One canonical node per concept, with an `aliases` list**, not a separate node
   per surface form:
   ```
   (:Material {name: "никель", aliases: ["Ni", "nickel"]})
   (:Equipment {name: "печь Ванюкова конвертерная", aliases: ["ПВК"]})
   (:Process   {name: "электроэкстракция", aliases: ["electrowinning"]})
   ```
   This is also exactly the format already used in `../ner_re_extraction/ner_re_examples.md`
   (`главное название (уточнение)`) — the manual annotation was already choosing a
   canonical form and parenthesizing the alias, which maps directly onto this schema.

2. **A lexicon/gazetteer built bottom-up from what's already been extracted**,
   not a fixed dictionary written in advance. Every time extraction sees a pattern
   like `X (Y)` or `X, также известный как Y` for a recognized entity type, `Y`
   gets added to `X`'s alias list (or, if `Y` already exists as its own canonical
   node, the two nodes get merged and the graph keeps a record of the merge).

3. **Two consumers need this alias list for different reasons**, matching the
   "semantic+lexic index" half of the project paradigm:
   - **Lexical index** (exact/fuzzy string match, e.g. Elasticsearch) — indexes
     every alias, not just the canonical name, so a search for `ПВК` and a search
     for `печь Ванюкова конвертерная` both hit the same node.
   - **Semantic index** (embedding search) — canonical name + aliases are
     embedded together (or aliases are used to pull the query into the same
     embedding neighborhood as the canonical term), so RU/EN paraphrases that
     don't share a lexical root (`электроэкстракция` / `electrowinning`) still
     resolve to the same node via meaning rather than string match.

4. **Bilingual pairs specifically**: since this corpus mixes RU and EN documents,
   the alias list should carry a `lang` tag per alias (`{"text": "electrowinning",
   "lang": "en"}`) rather than being a flat list of strings — this is what lets a
   comparative "отечественная практика" vs "мировая практика" query filter
   correctly without the language of the surface form leaking into the geography
   filter (a RU article can cite a foreign `electrowinning` process, and an EN
   source can describe a domestic one).

## Open question for whoever picks this up next

Whether alias merging should be **fully automatic** (extraction pipeline commits
merges on its own confidence threshold) or require a **human-in-the-loop
confirmation** before two nodes are merged — the hackathon brief's "ручная
корректировка графа экспертами" (manual graph correction by experts) requirement
suggests the latter should at least be available, with the automatic merge as a
suggestion rather than a silent action.
