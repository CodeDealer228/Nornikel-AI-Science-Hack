from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import MarianMTModel, MarianTokenizer

from .synonyms import SynonymExpander, ordered_unique


CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
LATIN_RE = re.compile(r"[a-zA-Z]")
TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+(?:[-_][a-zA-Zа-яА-ЯёЁ0-9]+)?")

BOUNDARY_CHARS = r"A-Za-zА-Яа-яЁё0-9"

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"


@dataclass(frozen=True)
class MatchedGlossaryTerm:
    canonical_id: str
    canonical_name: str
    group_type: str
    matched_alias: str
    replacement_ru: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    is_english_query: bool
    translated_query: str
    dense_query: str
    bm25_query: str
    matched_terms: list[dict[str, Any]]
    matched_synonyms: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def simple_query_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def has_cyrillic(text: str) -> bool:
    return CYRILLIC_RE.search(text) is not None


def has_latin(text: str) -> bool:
    return LATIN_RE.search(text) is not None


def is_fully_english_query(query: str) -> bool:
    """
    True, если в query есть латиница и нет кириллицы.

    "mine water desalination TDS" -> True
    "обессоливание шахтных вод" -> False
    "NCM811 pH температура" -> False
    """
    return has_latin(query) and not has_cyrillic(query)


def is_safe_latin_term(term: str) -> bool:
    """
    Для BM25 оставляем только безопасные латинские спецтермины:
    TDS, PGM, NCM811, SEM, EDS, XRF, ORP, pH и т.п.

    Не оставляем обычные английские фразы:
    mine water, desalination, total dissolved solids.
    """
    clean = term.strip()

    if not clean:
        return False

    if has_cyrillic(clean):
        return True

    if not has_latin(clean):
        return False

    if clean in {"pH", "ph", "PH"}:
        return True

    compact = re.sub(r"[^A-Za-z0-9]", "", clean)

    if not compact:
        return False

    has_digit = any(ch.isdigit() for ch in compact)
    letters = "".join(ch for ch in compact if ch.isalpha())

    # NCM811, Li2CO3, H2SO4
    if has_digit:
        return True

    # TDS, PGM, SEM, EDS, XRF, ORP, ICP
    if 2 <= len(letters) <= 10 and letters.upper() == letters:
        return True

    return False


def make_alias_regex(alias: str) -> re.Pattern[str]:
    """
    Делает regex для alias с нормальными границами.

    Пример:
    "mine water" найдёт "mine water", но не кусок внутри длинного слова.
    """
    alias = alias.strip()

    parts = re.split(r"([\s\-_–—−]+)", alias)
    pattern_parts: list[str] = []

    for part in parts:
        if not part:
            continue

        if re.fullmatch(r"[\s\-_–—−]+", part):
            pattern_parts.append(r"[\s\-_–—−]+")
        else:
            pattern_parts.append(re.escape(part))

    pattern = "".join(pattern_parts)

    left_boundary = ""
    right_boundary = ""

    if alias and re.match(rf"[{BOUNDARY_CHARS}]", alias[0]):
        left_boundary = rf"(?<![{BOUNDARY_CHARS}])"

    if alias and re.match(rf"[{BOUNDARY_CHARS}]", alias[-1]):
        right_boundary = rf"(?![{BOUNDARY_CHARS}])"

    return re.compile(left_boundary + pattern + right_boundary, flags=re.IGNORECASE)


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def clean_query_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = text.strip(" \n\t.,;:")
    return text.strip()


class EnRuTranslator:
    """
    Локальный en -> ru переводчик.

    Важное: он НЕ отвечает за доменные термины.
    Доменные термины берём из synonyms.yaml.
    """

    def __init__(
        self,
        model_name: str = TRANSLATION_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        text = text.strip()

        if not text:
            return ""

        batch = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        batch = {key: value.to(self.device) for key, value in batch.items()}

        generated = self.model.generate(
            **batch,
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )

        translated = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]

        return clean_query_text(translated)


class QueryProcessor:
    """
    Строит query-представления для dense и BM25.

    Если query английский:
    - ищем доменные термины в original query;
    - если короткий запрос почти весь состоит из терминов, переводчик не вызываем;
    - иначе переводим только промежутки между терминами;
    - dense_query = русскоязычный glossary-aware query;
    - bm25_query = dense_query + русские aliases + безопасные сокращения.

    Если query русский/смешанный:
    - dense_query = original query + русские canonical terms, если они полезны;
    - bm25_query = dense_query + русские aliases + безопасные сокращения.
    """

    def __init__(
        self,
        synonym_expander: SynonymExpander,
        translation_model_name: str = TRANSLATION_MODEL_NAME,
        enable_translation: bool = True,
    ) -> None:
        self.synonym_expander = synonym_expander
        self.translation_model_name = translation_model_name
        self.enable_translation = enable_translation
        self._translator: EnRuTranslator | None = None

        # Чтобы --show-query-plan и search() не переводили один и тот же query два раза.
        self._cache: dict[str, QueryPlan] = {}

    def process(self, query: str) -> QueryPlan:
        original_query = query.strip()

        if original_query in self._cache:
            return self._cache[original_query]

        is_english = is_fully_english_query(original_query)
        matched_terms = self._find_glossary_terms(original_query)

        if is_english:
            translated_query = self._make_english_translated_query(
                original_query=original_query,
                matched_terms=matched_terms,
            )
            dense_query = translated_query
        else:
            translated_query = original_query
            dense_query = self._make_non_english_dense_query(
                original_query=original_query,
                matched_terms=matched_terms,
            )

        bm25_query = self._make_bm25_query(
            dense_query=dense_query,
            matched_terms=matched_terms,
        )

        plan = QueryPlan(
            original_query=original_query,
            is_english_query=is_english,
            translated_query=translated_query,
            dense_query=dense_query,
            bm25_query=bm25_query,
            matched_terms=[term.to_dict() for term in matched_terms],
            matched_synonyms=self._make_matched_synonyms(matched_terms),
        )

        self._cache[original_query] = plan
        return plan

    def _make_english_translated_query(
        self,
        *,
        original_query: str,
        matched_terms: list[MatchedGlossaryTerm],
    ) -> str:
        if not matched_terms:
            return self._translate_en_to_ru(original_query)

        if self._should_use_glossary_only(original_query, matched_terms):
            return self._make_glossary_only_query(matched_terms)

        return self._translate_only_gaps_between_terms(
            original_query=original_query,
            matched_terms=matched_terms,
        )

    def _make_non_english_dense_query(
        self,
        *,
        original_query: str,
        matched_terms: list[MatchedGlossaryTerm],
    ) -> str:
        dense_terms = self._dense_terms_from_matches(matched_terms)

        if not dense_terms:
            return original_query

        # Для русских/смешанных запросов не раздуваем dense слишком сильно.
        # Но если query содержит аббревиатуру типа МПГ/ПВП/NCM811, canonical terms помогают.
        useful_terms = [
            term
            for term in dense_terms
            if term.lower() not in original_query.lower()
        ]

        return self._join_query_parts([original_query, *useful_terms])

    def _should_use_glossary_only(
            self,
            original_query: str,
            matched_terms: list[MatchedGlossaryTerm],
    ) -> bool:
        """
        Glossary-only включаем только если ВСЕ токены запроса покрыты найденными
        glossary terms.

        Пример:
        "mine water desalination TDS"
        -> все токены покрыты, можно не вызывать переводчик.

        "removal of iron copper lead from nickel chloride solutions"
        -> removal/of/from не покрыты, значит надо переводить промежутки.
        """
        tokens = list(TOKEN_RE.finditer(original_query))

        if not tokens or not matched_terms:
            return False

        if len(matched_terms) < 2:
            return False

        for token in tokens:
            token_start = token.start()
            token_end = token.end()

            is_covered = any(
                spans_overlap(token_start, token_end, term.start, term.end)
                for term in matched_terms
            )

            if not is_covered:
                return False

        return True

    def _token_coverage_by_terms(
        self,
        original_query: str,
        matched_terms: list[MatchedGlossaryTerm],
    ) -> float:
        token_spans = [(m.start(), m.end()) for m in TOKEN_RE.finditer(original_query)]

        if not token_spans:
            return 0.0

        covered = 0

        for token_start, token_end in token_spans:
            if any(
                spans_overlap(token_start, token_end, term.start, term.end)
                for term in matched_terms
            ):
                covered += 1

        return covered / len(token_spans)

    def _make_glossary_only_query(
        self,
        matched_terms: list[MatchedGlossaryTerm],
    ) -> str:
        """
        Собирает query только из русских glossary terms.

        Небольшая эвристика:
        process_* ставим раньше, чтобы было "обессоливание шахтных вод",
        а не "шахтные воды обессоливание".
        """

        def priority(term: MatchedGlossaryTerm) -> tuple[int, int]:
            canonical_id = term.canonical_id.lower()

            if canonical_id.startswith("process_"):
                return (0, term.start)

            if canonical_id.startswith(("water_", "solution_", "material_")):
                return (1, term.start)

            if canonical_id.startswith("parameter_"):
                return (2, term.start)

            return (3, term.start)

        ordered = sorted(matched_terms, key=priority)
        terms = ordered_unique([term.replacement_ru for term in ordered])
        return self._join_query_parts(terms)

    def _translate_only_gaps_between_terms(
        self,
        *,
        original_query: str,
        matched_terms: list[MatchedGlossaryTerm],
    ) -> str:
        """
        Glossary-aware translation.

        Мы НЕ отправляем доменные термины в переводчик.
        Переводим только промежутки между ними и вставляем replacement_ru из словаря.

        Пример:
        "How does mine water desalination affect TDS?"
        -> переводим "How does", "affect"
        -> вставляем "шахтные воды", "обессоливание", "сухой остаток"
        """
        parts: list[str] = []
        cursor = 0

        for term in matched_terms:
            gap = original_query[cursor:term.start]
            translated_gap = self._translate_en_to_ru(gap)

            if translated_gap:
                parts.append(translated_gap)

            parts.append(term.replacement_ru)
            cursor = term.end

        tail = original_query[cursor:]
        translated_tail = self._translate_en_to_ru(tail)

        if translated_tail:
            parts.append(translated_tail)

        result = self._join_query_parts(parts)

        if result:
            return result

        return self._make_glossary_only_query(matched_terms)

    def _make_bm25_query(
        self,
        *,
        dense_query: str,
        matched_terms: list[MatchedGlossaryTerm],
    ) -> str:
        bm25_terms = self._bm25_terms_from_matches(matched_terms)
        return self._join_query_parts([dense_query, *bm25_terms])

    def _find_glossary_terms(self, query: str) -> list[MatchedGlossaryTerm]:
        candidates: list[MatchedGlossaryTerm] = []

        for group in self.synonym_expander.groups:
            canonical_name = str(group.canonical_name)
            aliases = ordered_unique([canonical_name, *list(group.aliases)])

            # Длинные aliases ищем раньше, чтобы "mine water" победил "water".
            aliases = sorted(aliases, key=len, reverse=True)

            replacement_ru = self._choose_ru_replacement(group)

            if not replacement_ru:
                continue

            for alias in aliases:
                alias = str(alias).strip()

                if not self._is_usable_alias(alias):
                    continue

                pattern = make_alias_regex(alias)

                for match in pattern.finditer(query):
                    candidates.append(
                        MatchedGlossaryTerm(
                            canonical_id=str(group.canonical_id),
                            canonical_name=canonical_name,
                            group_type=str(group.type),
                            matched_alias=alias,
                            replacement_ru=replacement_ru,
                            start=match.start(),
                            end=match.end(),
                        )
                    )

        return self._select_non_overlapping_terms(candidates)

    def _select_non_overlapping_terms(
        self,
        candidates: list[MatchedGlossaryTerm],
    ) -> list[MatchedGlossaryTerm]:
        """
        Если нашли пересекающиеся aliases, оставляем более длинный.
        """
        selected: list[MatchedGlossaryTerm] = []

        candidates = sorted(
            candidates,
            key=lambda term: (-(term.end - term.start), term.start),
        )

        for candidate in candidates:
            if any(
                spans_overlap(candidate.start, candidate.end, item.start, item.end)
                for item in selected
            ):
                continue

            selected.append(candidate)

        return sorted(selected, key=lambda term: term.start)

    def _choose_ru_replacement(self, group: Any) -> str:
        """
        Выбирает русское canonical-представление группы.
        """
        canonical_name = str(group.canonical_name).strip()

        if has_cyrillic(canonical_name):
            return canonical_name

        for alias in group.aliases:
            alias = str(alias).strip()

            if has_cyrillic(alias):
                return alias

        if is_safe_latin_term(canonical_name):
            return canonical_name

        return ""

    def _is_usable_alias(self, alias: str) -> bool:
        alias = alias.strip()

        if not alias:
            return False

        # Одинокие буквы почти всегда дают ложные совпадения.
        # "S", "O", "C" лучше не матчить как aliases.
        compact = re.sub(r"[^A-Za-zА-Яа-яЁё0-9]", "", alias)

        if len(compact) <= 1:
            return False

        return True

    def _dense_terms_from_matches(
        self,
        matched_terms: list[MatchedGlossaryTerm],
    ) -> list[str]:
        return ordered_unique(
            [
                term.replacement_ru
                for term in matched_terms
                if has_cyrillic(term.replacement_ru) or is_safe_latin_term(term.replacement_ru)
            ]
        )

    def _bm25_terms_from_matches(
        self,
        matched_terms: list[MatchedGlossaryTerm],
    ) -> list[str]:
        """
        Для BM25 добавляем:
        - русские aliases;
        - безопасные латинские сокращения;
        - НЕ добавляем обычные английские фразы типа mine water/desalination.
        """
        terms: list[str] = []

        for term in matched_terms:
            group = self.synonym_expander._get_group_by_id(term.canonical_id)

            terms.append(term.replacement_ru)

            if group is None:
                continue

            aliases = ordered_unique([group.canonical_name, *list(group.aliases)])

            for alias in aliases:
                alias = str(alias).strip()

                if has_cyrillic(alias) or is_safe_latin_term(alias):
                    terms.append(alias)

        return ordered_unique(terms)

    def _make_matched_synonyms(
        self,
        matched_terms: list[MatchedGlossaryTerm],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}

        for term in matched_terms:
            item = grouped.setdefault(
                term.canonical_id,
                {
                    "canonical_id": term.canonical_id,
                    "type": term.group_type,
                    "canonical_name": term.canonical_name,
                    "replacement_ru": term.replacement_ru,
                    "matched_aliases": [],
                },
            )

            item["matched_aliases"].append(term.matched_alias)

        result = []

        for item in grouped.values():
            item["matched_aliases"] = ordered_unique(item["matched_aliases"])
            result.append(item)

        return result

    def _translate_en_to_ru(self, text: str) -> str:
        text = text.strip()

        if not text:
            return ""

        if not self.enable_translation:
            return text

        try:
            if self._translator is None:
                self._translator = EnRuTranslator(
                    model_name=self.translation_model_name,
                )

            return self._translator.translate(text)

        except Exception as exc:
            print(f"[WARN] Translation failed, fallback to original text: {exc}")
            return text

    @staticmethod
    def _join_query_parts(parts: list[str]) -> str:
        clean_parts = []

        for part in parts:
            part = clean_query_text(str(part))

            if part:
                clean_parts.append(part)

        return clean_query_text(" ".join(clean_parts))