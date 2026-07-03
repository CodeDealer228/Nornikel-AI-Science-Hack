# ner_re_extraction/ — Stage 2: Manual NER+RE calibration

A hand-built, high-quality Named Entity Recognition + Relation Extraction pass
over 7 real articles from the corpus, done by careful multi-step reading rather
than a single LLM call. It's slower than one LLM pass would be, but it gives a
**verified, ontology-correct ground truth** — both to sanity-check the eventual
automated pipeline against, and to serve directly as the few-shot seed bank for
that pipeline (see `../llm_pipeline_fewshot/`).

## Ontology used (from the hackathon brief)

- **Entities**: `Material, Process, Equipment, Property, Experiment, Publication, Expert, Facility`
- **Relations**: `uses_material, operates_at_condition, produces_output, described_in, validated_by, contradicts`

## Contents

- **`ner_re_examples.md`** — the annotation itself. 7 articles, each with:
  - every entity type populated with **concrete named entities** (not entity
    categories) pulled straight from the article text, formatted
    `главное название (уточнение/аббревиатура/формула)` — e.g.
    `печь Ванюкова конвертерная (ПВК)`, `анионит Lewatit А365 (акрилатная матрица)`
  - a `### Отношения (triples)` block per article: concrete
    `entity —relation_type→ entity` triples, each traceable back to its source
    document
  - a closing "Наблюдения" section with cross-cutting findings (see below)
- **`source_texts/`** — the raw plain text extracted from each of the 7 source
  `.docx` files (via `python-docx`), so the annotation can be checked against the
  actual article text without needing the full (multi-GB) `input_docs/` corpus.

## Articles covered

1. Технология непрерывного конвертирования медного никельсодержащего сырья
2. Сорбционная очистка хлоридно-сульфатных никелевых растворов от примеси свинца
3. Результаты исследований в области перспективного производства никеля высокопремиальных марок
4. Определение меди и никеля в медно-никелевом файнштейне (способ ограничивающих стандартов)
5–6. Влияние различных факторов на окисление железа в высококонцентрированных хлоридных никелевых растворах, части 1 и 2
7. Влияние отставания вентиляционного трубопровода на эффективность проветривания тупиковой горной выработки (mine ventilation — deliberately a **different** technological domain from the other 6, to stress-test the ontology's extensibility claim)

## Key findings (see `ner_re_examples.md`'s "Наблюдения" section for full detail)

- **Synonym/bilingual pressure is real and immediate** — abbreviation ↔ full-name
  pairs (`ПВК` ↔ `печь Ванюкова конвертерная`, `ЦЭН-2` ↔ `цех электролиза никеля №2`,
  `ИСП-АЭС` ↔ `ICP-AES`) show up in nearly every article. This is explored further
  in `../synonym_normalization/`.
- **Numeric ranges + units are everywhere** (г/дм³, мг/л, °C, Вт/м³, кДж/моль, А/м²)
  — confirms the brief's "extraction errors here are unacceptable" requirement has
  to be a first-class extraction concern, not a post-process.
- **`contradicts` is not a rare edge case.** One article (doc 5) has two numeric
  values for the *same* quantity (activation energy) that directly contradict each
  other until a methodological correction is applied — a clean real-world test case
  for the ontology's `contradicts` relation, and a reminder that resolving a
  contradiction means adding context, not deleting one of the facts.
  Multi-part articles (docs 5+6, parts 1 and 2 of the same study) instead need a
  **non**-`contradicts` link, or the graph will wrongly split one continuous
  experiment into two disconnected ones.
- **Cross-domain extensibility holds**: the mine-ventilation article (doc 7) uses
  the exact same 8 entity types / 6 relation types as the metallurgy articles —
  only the vocabulary changes, not the schema.
- **Expert↔Facility affiliation** surfaces as an implicit relation the base 6
  relation types don't cover (an author's institutional affiliation), worth a
  decision call before graph load.
