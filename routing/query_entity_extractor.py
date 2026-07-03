"""
Query-time entity extractor.

Reuses the existing ``NatashaPipeline`` (chunking) for primary
named-entity recognition and the ``synonym_normalization``
canonicalizer for alias resolution. Falls back to a deterministic
regex pipeline when Natasha is unavailable so the router still
works in degraded environments.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from chunking.natasha_pipeline import get_pipeline
from synonym_normalization.canonicalizer import canonicalize_text
from synonym_normalization.synonym_dictionary import SynonymDictionary

from .query_models import ExtractedQueryEntity, QueryAnalysis

log = logging.getLogger(__name__)

# Russian + English question / interrogative cues, used to detect question
# shape even when a literal "?" is missing.
_QUESTION_CUE_RE = re.compile(
    r"\b(какие|какой|какая|каким|как|где|когда|сколько|почему|зачем|"
    r"чем|кто|что|куда|откуда|можно|есть ли|существует ли|"
    r"which|what|how|where|when|why|who|whom|whose)\b",
    re.IGNORECASE | re.UNICODE,
)

# Numeric patterns (concentrations, temperatures, ranges, percentages).
_NUMERIC_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|°c|г/л|мг/л|мг/дм3|м3/ч|м3/сут|т/сут|"
    r"кг/ч|квт|мвт|мм/сут|а/м2|а/дм3|ма/см2|"
    r"ph|атм|мпа|па|об%|ppm|°)\b",
    re.IGNORECASE | re.UNICODE,
)

# Unit-first pattern: "pH ≤ 4", "pH 10.5" — where the unit precedes the number.
_UNIT_FIRST_RE = re.compile(
    r"\b(?:pH)\s*[<>≤≥]=?\s*\d+(?:[.,]\d+)?\b",
    re.IGNORECASE | re.UNICODE,
)

_RANGE_RE = re.compile(
    r"\b(?:от|from|между|between)\s+\d+(?:[.,]\d+)?\s*(?:до|to|-)\s*\d+(?:[.,]\d+)?\b",
    re.IGNORECASE | re.UNICODE,
)

# Geographic markers (countries, regions, "Россия", "за рубежом", etc.).
_GEO_RE = re.compile(
    r"\b(россия|рф|зарубежом|за рубежом|мировая практика|отечественная практика|"
    r"снг|евросоюз|китай|индия|африка|канада|сша|норвегия|финляндия|"
    r"норникель|норильск|кольский|таймыр|печенега|никельский|"
    r"russia|usa|china|eu|europe|asia|africa|canada|norway|finland|worldwide)\b",
    re.IGNORECASE | re.UNICODE,
)

# Temporal markers.
_TEMPORAL_RE = re.compile(
    r"\b(за последние|последние|в\s+\d{4}|в\s+прошлом|с\s+\d{4}|по\s+\d{4}|"
    r"с\s+начала|годы|года|лет|год|"
    r"last|recent|since|until|by\s+\d{4}|in\s+\d{4}|over\s+the\s+past)\b",
    re.IGNORECASE | re.UNICODE,
)

# Definitional cues — the user is asking for a definition / decoding / acronym
# explanation, not for graph facts.
_DEFINITIONAL_RE = re.compile(
    r"\b(что такое|что значит|что означает|что представляет|"
    r"определение\s|определи\s|расшифруй|расшифровка|"
    r"аббревиатура|аббревиатурой|"
    r"what is|what does|define|definition of|stands for|meaning of)\b",
    re.IGNORECASE | re.UNICODE,
)

# Causal / explanatory cues — explanations live in prose, not in the graph.
_CAUSAL_RE = re.compile(
    r"\b(почему|зачем|как\s+работает|за\s+сч[её]т|благодаря\s+чему|"
    r"в\s+результате\s+чего|объясни\s|каким\s+образом\s|"
    r"why|how does|how is|how do|reason for|because of|due to|"
    r"mechanism behind|explanation)\b",
    re.IGNORECASE | re.UNICODE,
)

# Comparison cues — almost always want both structured facts and prose.
_COMPARISON_RE = re.compile(
    r"\b(сравни\s|сравните\s|сравнение\s|по\s+сравнению\s+с\s|"
    r"разница\s+между\s|отличие\s+от\s|"
    r"vs\.?|versus|or alternatively|compared to|comparison of|"
    r"differences between|distinguish)\b",
    re.IGNORECASE | re.UNICODE,
)

# Geographic-comparison cues — "Russia vs abroad", "domestic vs world practice".
_GEO_COMPARISON_RE = re.compile(
    r"\b(отечественн\w*\s+практик\w*|"
    r"зарубежн\w*\s+практик\w*|"
    r"миров\w*\s+практик\w*|"
    r"(?:росси|рф)\s*(?:vs|и|или|против|versus)\s*(?:зарубеж|миров|запад)|"
    r"(?:domestic|russian)\s+(?:vs|versus|or)\s+(?:world|abroad|foreign)|"
    r"worldwide\s+practice)\b",
    re.IGNORECASE | re.UNICODE,
)

# Russian stopwords used by the regex fallback to drop noise tokens.
_STOPWORDS = frozenset(
    """
    и в на по с к о у из за для от до без при что это как или но
    the a an of to in for on at by with and or but from into onto over
    please нужно необходимо можно какие какой какая каким как где когда сколько
    """.split()
)

# Mining/metallurgy domain seed terms. ``canonical_stem`` is matched at
# word boundaries (case-insensitive) and used to resolve aliases;
# ``canonical_label`` is the surface form we return as the canonical
# name (so "никеля" → "никель").
_DOMAIN_SEEDS: tuple[tuple[str, str], ...] = (
    ("никел", "никель"),
    ("мед", "медь"),
    ("кобал", "кобальт"),
    ("золот", "золото"),
    ("серебр", "серебро"),
    ("мпг", "МПГ"),
    ("штейн", "штейн"),
    ("шлак", "шлак"),
    ("катод", "катод"),
    ("анод", "анод"),
    ("электролит", "электролит"),
    ("католит", "католит"),
    ("анолит", "анолит"),
    ("шихт", "шихта"),
    ("руд", "руда"),
    ("концентр", "концентрат"),
    ("хвост", "хвосты"),
    ("печ", "печь"),
    ("ванн", "ванна"),
    ("флотомашин", "флотомашина"),
    ("выщелачив", "выщелачивание"),
    ("флотац", "флотация"),
    ("электроэкстракц", "электроэкстракция"),
    ("электролиз", "электролиз"),
    ("очистк", "очистка"),
    ("обессерив", "обессеривание"),
    ("обессолив", "обессоливание"),
    ("осажден", "осаждение"),
    ("пвп", "ПВП"),
    ("автоклав", "автоклав"),
    ("гидрометаллург", "гидрометаллургия"),
    ("пирометаллург", "пирометаллургия"),
    ("катодн", "катодный"),
    ("техногенн", "техногенный"),
    ("гипс", "гипс"),
    ("шахтн", "шахтная"),
    ("рудничн", "рудничная"),
    ("сульфат", "сульфат"),
    ("хлорид", "хлорид"),
    ("пропитк", "пропитка"),
    ("кек", "кек"),
    ("сер", "сера"),
    ("so2", "SO2"),
    ("материал", "материал"),
    ("оборудован", "оборудование"),
    ("эксперимент", "эксперимент"),
    ("процесс", "процесс"),
)


def _build_domain_term_regex() -> re.Pattern[str]:
    """Build a regex that matches domain stems (with optional inflections)."""
    parts: list[str] = []
    for stem, _label in _DOMAIN_SEEDS:
        # The stem is short enough that the trailing \w* will only match a
        # couple of Russian-inflection characters before hitting a word
        # boundary, so false positives on unrelated words stay rare.
        parts.append(re.escape(stem) + r"\w*")
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE | re.UNICODE)


_DOMAIN_TERM_RE = _build_domain_term_regex()

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\-]+", re.UNICODE)


class QueryEntityExtractor:
    """Extracts structured entity candidates from a user query.

    The extractor is deliberately conservative: it prefers high-precision
    matches from Natasha when available, then resolves aliases through
    the shared ``SynonymDictionary`` and finally falls back to a
    regex-based extractor for the mining/metallurgy vocabulary.
    """

    def __init__(
        self,
        synonym_dictionary: SynonymDictionary | None = None,
        min_surface_length: int = 2,
    ) -> None:
        self._synonym_dictionary = synonym_dictionary or SynonymDictionary()
        self._min_surface_length = min_surface_length
        self._pipeline = None
        try:
            self._pipeline = get_pipeline()
        except Exception as exc:  # pragma: no cover - defensive guard
            log.warning(
                "Natasha pipeline unavailable for query extractor: %s: %s",
                type(exc).__name__,
                exc,
            )
            self._pipeline = None

    def extract(self, query: str) -> tuple[ExtractedQueryEntity, ...]:
        """Return entity candidates for ``query``."""
        if not query or not query.strip():
            return ()

        natasha_entities = self._natasha_extract(query)
        regex_entities = self._regex_extract(query)

        merged: dict[str, ExtractedQueryEntity] = {}
        for entity in natasha_entities + regex_entities:
            key = canonicalize_text(entity.surface)
            if key in merged:
                existing = merged[key]
                if entity.confidence > existing.confidence:
                    merged[key] = entity
            else:
                merged[key] = entity

        ordered = sorted(
            merged.values(),
            key=lambda ent: (-ent.confidence, ent.char_start),
        )
        return tuple(ordered)

    def analyze(self, query: str) -> QueryAnalysis:
        """Build a structural analysis of ``query`` for downstream routing."""
        normalized = canonicalize_text(query)
        tokens = _TOKEN_RE.findall(query)
        token_count = len(tokens)
        word_count = len([t for t in tokens if t.lower() not in _STOPWORDS])

        seed_entities = self.extract(query)
        domain_seed_count = sum(
            1
            for ent in seed_entities
            if self._is_domain_term(ent.canonical) or ent.type_hint is not None
        )

        has_question_mark = "?" in query or bool(_QUESTION_CUE_RE.search(query))
        has_numeric_constraint = bool(
            _NUMERIC_RE.search(query)
            or _RANGE_RE.search(query)
            or _UNIT_FIRST_RE.search(query)
        )
        has_geo_marker = bool(_GEO_RE.search(query))
        has_temporal_marker = bool(_TEMPORAL_RE.search(query))
        is_definitional = bool(_DEFINITIONAL_RE.search(query))
        is_causal = bool(_CAUSAL_RE.search(query))
        is_comparison = bool(_COMPARISON_RE.search(query))
        is_geo_comparison = bool(_GEO_COMPARISON_RE.search(query))

        notes: list[str] = []
        if not query.strip():
            notes.append("empty_query")
        if token_count < 3:
            notes.append("very_short_query")
        if has_numeric_constraint:
            notes.append("has_numeric_constraint")
        if has_geo_marker:
            notes.append("has_geo_marker")
        if has_temporal_marker:
            notes.append("has_temporal_marker")
        if is_definitional:
            notes.append("definitional_query")
        if is_causal:
            notes.append("causal_query")
        if is_comparison:
            notes.append("comparison_query")
        if is_geo_comparison:
            notes.append("geo_comparison_query")

        is_low_signal = (
            token_count < 3
            or not seed_entities
            and not has_numeric_constraint
        )

        return QueryAnalysis(
            query=query,
            normalized_query=normalized,
            char_length=len(query),
            word_count=word_count,
            token_count=token_count,
            has_question_mark=has_question_mark,
            has_numeric_constraint=has_numeric_constraint,
            has_geo_marker=has_geo_marker,
            has_temporal_marker=has_temporal_marker,
            seed_entities=seed_entities,
            domain_seed_count=domain_seed_count,
            is_low_signal=is_low_signal,
            is_definitional=is_definitional,
            is_causal=is_causal,
            is_comparison=is_comparison,
            is_geo_comparison=is_geo_comparison,
            notes=tuple(notes),
        )

    def _natasha_extract(self, query: str) -> tuple[ExtractedQueryEntity, ...]:
        if self._pipeline is None:
            return ()
        try:
            annotation = self._pipeline.annotate(query)
        except Exception as exc:  # pragma: no cover - defensive guard
            log.warning("Natasha annotation failed on query: %s: %s", type(exc).__name__, exc)
            return ()

        entities: list[ExtractedQueryEntity] = []
        for primary in annotation.primary_entities or []:
            text = (primary.text or "").strip()
            if len(text) < self._min_surface_length:
                continue
            canonical = self._canonical_for_surface(primary.normal or text)
            entities.append(ExtractedQueryEntity(
                surface=text,
                canonical=canonical,
                type_hint=str(primary.type) if primary.type else None,
                char_start=int(primary.start),
                char_end=int(primary.stop),
                confidence=0.78,
                source="natasha",
            ))
        return tuple(entities)

    def _regex_extract(self, query: str) -> tuple[ExtractedQueryEntity, ...]:
        matches: list[ExtractedQueryEntity] = []
        seen: set[tuple[str, int]] = set()

        for match in _DOMAIN_TERM_RE.finditer(query):
            surface = match.group(0).strip()
            if len(surface) < self._min_surface_length:
                continue
            canonical = self._canonical_for_surface(surface)
            key = (canonicalize_text(surface), match.start())
            if key in seen:
                continue
            seen.add(key)
            matches.append(ExtractedQueryEntity(
                surface=surface,
                canonical=canonical,
                type_hint=None,
                char_start=match.start(),
                char_end=match.end(),
                confidence=0.6,
                source="regex",
            ))

        # Capitalised multi-word sequences (Latin/Cyrillic) — likely orgs / products.
        for match in re.finditer(
            r"\b(?:[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9]+"
            r"(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9]+){0,3})\b",
            query,
        ):
            surface = match.group(0).strip()
            if len(surface) < self._min_surface_length:
                continue
            lower = surface.lower()
            if any(token in _STOPWORDS for token in lower.split()):
                continue
            canonical = self._canonical_for_surface(surface)
            key = (canonicalize_text(surface), match.start())
            if key in seen:
                continue
            seen.add(key)
            matches.append(ExtractedQueryEntity(
                surface=surface,
                canonical=canonical,
                type_hint=None,
                char_start=match.start(),
                char_end=match.end(),
                confidence=0.55,
                source="regex",
            ))

        # Numeric constraints as "entities" (so the router can see them).
        for match in _NUMERIC_RE.finditer(query):
            surface = match.group(0).strip()
            if not surface:
                continue
            key = (canonicalize_text(surface), match.start())
            if key in seen:
                continue
            seen.add(key)
            matches.append(ExtractedQueryEntity(
                surface=surface,
                canonical=surface,
                type_hint="MEASURE",
                char_start=match.start(),
                char_end=match.end(),
                confidence=0.7,
                source="regex",
            ))

        return tuple(matches)

    @staticmethod
    def _is_domain_term(canonical: str) -> bool:
        normalized = canonicalize_text(canonical)
        if not normalized:
            return False
        for stem, _label in _DOMAIN_SEEDS:
            if normalized.startswith(stem) or stem.startswith(normalized):
                return True
        return False

    def _canonical_for_surface(self, surface: str) -> str:
        """Resolve ``surface`` to a canonical domain term, falling back to the synonym dict."""
        normalized = canonicalize_text(surface)
        if not normalized:
            return surface
        # Exact stem match: "никеля" → "никель".
        for stem, label in _DOMAIN_SEEDS:
            if normalized == stem or normalized.startswith(stem):
                return label
        # Synonym dictionary: catches load-time aliases (e.g. "ПВП" → "печь взвешенной плавки").
        resolved = self._synonym_dictionary.resolve(surface)
        if resolved and resolved != surface:
            return resolved
        return surface


def merge_seed_names(entities: Iterable[ExtractedQueryEntity]) -> tuple[str, ...]:
    """Deduplicate and order seed names for use as graph query seeds."""
    seen: set[str] = set()
    ordered: list[str] = []
    for ent in entities:
        key = canonicalize_text(ent.canonical or ent.surface)
        if key and key not in seen:
            seen.add(key)
            ordered.append(ent.canonical or ent.surface)
    return tuple(ordered)
