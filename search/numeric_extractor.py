from __future__ import annotations

import re
from typing import Any


# 0,025
# -0,126
# 1 000
# 1.5
# 1e-3
NUMBER_RE = r"[-+]?(?:\d{1,3}(?:\s\d{3})+|\d+)(?:[,.]\d+)?(?:[eE][+-]?\d+)?"

DASH_RE = r"(?:-|–|—|−|‒)"

OPERATOR_RE = (
    r"(?:"
    r"≤|>=|=>|≥|<|>|=|~|≈|"
    r"не\s+более|не\s+менее|"
    r"не\s+выше|не\s+ниже|"
    r"не\s+превыша(?:ет|ть)|"
    r"до|от|около|примерно|порядка|"
    r"менее|более|выше|ниже"
    r")"
)


# Это не доменный хардкод, а словарь единиц.
# Таблицы отдельно НЕ парсим: если значение без единицы, оно не извлекается.
UNIT_PATTERNS: list[tuple[str, str]] = [
    # concentration
    ("мг/дм3", r"мг\s*/\s*дм\s*[³3]"),
    ("мг/л", r"мг\s*/\s*л"),
    ("г/дм3", r"г\s*/\s*дм\s*[³3]"),
    ("г/л", r"г\s*/\s*л"),
    ("мкг/л", r"мкг\s*/\s*л"),
    ("моль/моль", r"моль\s*/\s*моль"),
    ("моль/дм3", r"моль\s*/\s*дм\s*[³3]"),
    ("моль/л", r"моль\s*/\s*л"),
    ("ммоль/л", r"ммоль\s*/\s*л"),
    ("ppm", r"ppm"),
    ("ppb", r"ppb"),

    # percent
    ("% отн.", r"%\s*отн\.?"),
    ("% абс.", r"%\s*абс\.?"),
    ("мас.%", r"мас\.?\s*%"),
    ("об.%", r"об\.?\s*%"),
    ("%", r"%"),

    # temperature
    ("°C", r"°\s*[CС]|℃"),

    # electrochemistry
    ("А/м2", r"А\s*/\s*м\s*[²2]"),
    ("кА/м2", r"кА\s*/\s*м\s*[²2]"),
    ("мА/см2", r"мА\s*/\s*см\s*[²2]"),
    ("мВ/с", r"мВ\s*/\s*с"),
    ("В/м", r"В\s*/\s*м"),
    ("мА", r"мА"),
    ("мВ", r"мВ"),
    ("А", r"А"),
    ("В", r"В"),

    # speed / flow
    ("об/мин", r"об\s*/\s*мин"),
    ("мин-1", r"мин\s*[-−]\s*1|мин⁻¹"),
    ("м/с", r"м\s*/\s*с"),
    ("мм/с", r"мм\s*/\s*с"),
    ("см/с", r"см\s*/\s*с"),
    ("м3/ч", r"м\s*[³3]\s*/\s*ч"),
    ("л/ч", r"л\s*/\s*ч"),
    ("л/мин", r"л\s*/\s*мин"),
    ("дм3/мин", r"дм\s*[³3]\s*/\s*мин"),

    # pressure
    ("МПа", r"МПа"),
    ("кПа", r"кПа"),
    ("Па", r"Па"),
    ("атм", r"атм"),
    ("бар", r"бар"),

    # size / density / surface
    ("мкм", r"мкм"),
    ("нм", r"нм"),
    ("мм", r"мм"),
    ("см", r"см"),
    ("м", r"м"),
    ("г/см3", r"г\s*/\s*см\s*[³3]"),
    ("кг/м3", r"кг\s*/\s*м\s*[³3]"),
    ("м2/г", r"м\s*[²2]\s*/\s*г"),
    ("см2/г", r"см\s*[²2]\s*/\s*г"),

    # time
    ("час", r"час(?:а|ов)?"),
    ("ч", r"ч"),
    ("мин", r"мин"),
    ("сут", r"сут"),

    # mass / volume
    # Одиночное "г" специально не добавляем, чтобы не ловить годы: 2020 г.
    ("кг", r"кг"),
    ("мг", r"мг"),
    ("дм3", r"дм\s*[³3]"),
    ("м3", r"м\s*[³3]"),
    ("мл", r"мл"),
    ("л", r"л"),

    # production / consumption
    ("т/сут", r"т\s*/\s*сут"),
    ("т/год", r"т\s*/\s*год"),
    ("кг/ч", r"кг\s*/\s*ч"),
    ("кг/т", r"кг\s*/\s*т"),
    ("кВтч/т", r"кВт\s*·?\s*ч\s*/\s*т"),

    # energy
    ("кДж/моль", r"кДж\s*/\s*моль"),
    ("Дж/моль", r"Дж\s*/\s*моль"),
]

UNIT_RE = "(?:" + "|".join(pattern for _, pattern in UNIT_PATTERNS) + ")"


NUMERIC_WITH_UNIT_RE = re.compile(
    rf"(?P<operator>{OPERATOR_RE})?\s*"
    rf"(?P<value1>{NUMBER_RE})"
    rf"(?:\s*{DASH_RE}\s*(?P<value2>{NUMBER_RE}))?"
    rf"\s*(?P<unit>{UNIT_RE})",
    flags=re.IGNORECASE,
)


FROM_TO_WITH_UNIT_RE = re.compile(
    rf"(?P<operator>от)\s*"
    rf"(?P<value1>{NUMBER_RE})"
    rf"\s*до\s*"
    rf"(?P<value2>{NUMBER_RE})"
    rf"\s*(?P<unit>{UNIT_RE})",
    flags=re.IGNORECASE,
)


PH_RE = re.compile(
    rf"(?<![A-Za-zА-Яа-яЁё0-9])"
    rf"(?P<unit>pH|ph|рН|рн)"
    rf"(?![A-Za-zА-Яа-яЁё0-9])"
    rf"\s*(?P<operator>{OPERATOR_RE})?\s*"
    rf"(?P<value1>{NUMBER_RE})"
    rf"(?:\s*{DASH_RE}\s*(?P<value2>{NUMBER_RE}))?",
    flags=re.IGNORECASE,
)


GENERIC_PARAMETER_RE = re.compile(
    r"("
    r"содержани[еяий]*|"
    r"концентрац(?:ия|ии|ию|ией)|"
    r"температур[аы]|"
    r"давлени[ея]|"
    r"скорост[ьи]|"
    r"расход|"
    r"плотност[ьи](?:\s+тока)?|"
    r"напряжени[ея]|"
    r"потенциал|"
    r"ОВП|"
    r"pH|рН|"
    r"влажност[ьи]|"
    r"размер(?:ы)?|"
    r"степень\s+извлечения|"
    r"извлечени[ея]|"
    r"выход|"
    r"сухой\s+остаток|"
    r"минерализац(?:ия|ии)"
    r")",
    flags=re.IGNORECASE,
)


NOISE_WORDS = [
    "orcid",
    "doi",
    "http",
    "https",
    "www.",
    "e-mail",
    "email",
    "удк",
    "isbn",
]


STOP_ENTITY_WORDS = {
    "и",
    "или",
    "а",
    "в",
    "во",
    "на",
    "по",
    "при",
    "для",
    "до",
    "от",
    "из",
    "с",
    "со",
    "к",
    "ко",
    "у",
    "за",
    "над",
    "под",
    "между",
    "после",
    "перед",
    "что",
    "как",
    "это",
    "данный",
    "данная",
    "данные",
    "раствор",
    "раствора",
    "растворе",
    "растворов",
    "фильтрат",
    "фильтрате",
    "электролит",
    "электролите",
    "менее",
    "более",
    "выше",
    "ниже",
    "около",
    "примерно",
    "порядка",
    "не",
    "больше",
    "меньше",
    "настоящее",
    "текущее",
    "время",
    "г",
    "года",
}


LEFT_ENTITY_BOUNDARY_WORDS = {
    "в",
    "во",
    "на",
    "из",
    "при",
    "после",
    "перед",
    "для",
    "по",
    "с",
    "со",
    "до",
    "от",
    "порядка",
    "около",
    "примерно",
    "менее",
    "более",
    "не",
    "должно",
    "может",
    "могут",
    "составляет",
    "составляла",
    "составляли",
    "достигает",
    "достигать",
    "превышает",
    "превышать",
    "равен",
    "равна",
    "равно",
}


CHEM_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$")
WORD_RE = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9+-]*$")


def normalize_text(text: str) -> str:
    return (
        text.replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2009", " ")
        .replace("‒", "-")
        .replace("−", "-")
    )


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_float(value: str) -> float:
    return float(value.replace(" ", "").replace(",", "."))


def canonicalize_unit(raw_unit: str | None) -> str | None:
    if not raw_unit:
        return None

    unit = compact_spaces(raw_unit)

    if unit.lower() in {"ph", "рн"}:
        return "pH"

    for canonical, pattern in UNIT_PATTERNS:
        if re.fullmatch(pattern, unit, flags=re.IGNORECASE):
            return canonical

    return unit


def normalize_operator(operator: str | None, has_range: bool) -> str:
    if has_range:
        return "range"

    if not operator:
        return "="

    op = compact_spaces(operator).lower()

    mapping = {
        "≤": "<=",
        "≥": ">=",
        "=>": ">=",
        "≈": "~",
        "~": "~",
        "не более": "<=",
        "не выше": "<=",
        "не превышает": "<=",
        "не превышать": "<=",
        "до": "<=",
        "менее": "<",
        "ниже": "<",
        "не менее": ">=",
        "не ниже": ">=",
        "от": ">=",
        "более": ">",
        "выше": ">",
        "около": "~",
        "примерно": "~",
        "порядка": "~",
    }

    return mapping.get(op, op)


def build_value_text(value1: str, value2: str | None, unit: str | None) -> str:
    if value2 is not None:
        value = f"{value1.strip()}-{value2.strip()}"
    else:
        value = value1.strip()

    if unit == "pH":
        return f"pH {value}"

    if unit:
        return f"{value} {unit}"

    return value


def get_line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, start)
    line_end = text.find("\n", end)

    if line_start == -1:
        line_start = 0
    else:
        line_start += 1

    if line_end == -1:
        line_end = len(text)

    return line_start, line_end


def is_in_markdown_table(text: str, start: int, end: int) -> bool:
    """
    Таблицы игнорируем полностью.
    Работает и для нормальных markdown-таблиц построчно,
    и для inline-таблиц, где вокруг много символов "|".
    """
    line_start, line_end = get_line_bounds(text, start, end)
    line = text[line_start:line_end]

    if line.count("|") >= 4:
        return True

    local = text[max(0, start - 120):min(len(text), end + 120)]

    if local.count("|") >= 4:
        return True

    return False


def is_noise_context(text: str, start: int, end: int, window: int = 100) -> bool:
    context = text[max(0, start - window):min(len(text), end + window)].lower()
    return any(word in context for word in NOISE_WORDS)


def is_bad_value_left_boundary(text: str, value_start: int) -> bool:
    """
    Проверяем границу именно числа, а не всего regex match.

    Плохо:
    NCM811
    ГОСТ123
    P-2

    Нормально:
    содержание 0,1 г/л
    до 1-2 %
    -0,126 В
    """
    if value_start <= 0:
        return False

    prev = text[value_start - 1]

    if prev.isalpha() or prev.isdigit():
        return True

    if prev in {"_", "-", "–", "—", "−", "‒"}:
        return True

    return False


def is_bad_unit_right_boundary(text: str, unit_end: int) -> bool:
    """
    Проверяем границу именно единицы.

    Плохо:
    100 Пакальнис -> "Па" внутри слова

    Нормально:
    100 Па
    0,1 г/л меди
    5 % Ni
    """
    if unit_end >= len(text):
        return False

    nxt = text[unit_end]

    if nxt.isalpha() or nxt.isdigit():
        return True

    if nxt in {"_", "/"}:
        return True

    return False


def get_numeric_value_end(match: re.Match[str]) -> int:
    value2 = match.groupdict().get("value2")

    if value2 is not None:
        return match.end("value2")

    return match.end("value1")


def should_skip_numeric_match(text: str, match: re.Match[str]) -> bool:
    start, end = match.span()

    if is_in_markdown_table(text, start, end):
        return True

    if is_noise_context(text, start, end):
        return True

    value_start = match.start("value1")
    unit_end = match.end("unit")

    if is_bad_value_left_boundary(text, value_start):
        return True

    if is_bad_unit_right_boundary(text, unit_end):
        return True

    return False


def should_skip_ph_match(text: str, match: re.Match[str]) -> bool:
    start, end = match.span()

    if is_in_markdown_table(text, start, end):
        return True

    if is_noise_context(text, start, end):
        return True

    value_start = match.start("value1")
    value_end = get_numeric_value_end(match)

    if is_bad_value_left_boundary(text, value_start):
        return True

    if is_bad_unit_right_boundary(text, value_end):
        return True

    return False


def explain_skip_reason_for_numeric_match(text: str, match: re.Match[str]) -> str | None:
    start, end = match.span()

    if is_in_markdown_table(text, start, end):
        return "table"

    if is_noise_context(text, start, end):
        return "noise_context"

    value_start = match.start("value1")
    unit_end = match.end("unit")

    if is_bad_value_left_boundary(text, value_start):
        return "bad_value_left_boundary"

    if is_bad_unit_right_boundary(text, unit_end):
        return "bad_unit_right_boundary"

    return None


def get_left_segment(text: str, start: int, window: int = 120) -> str:
    left = max(0, start - window)

    for boundary in ["\n", "|", ".", ";"]:
        pos = text.rfind(boundary, left, start)
        if pos != -1:
            left = max(left, pos + 1)

    return text[left:start]


def get_right_segment(text: str, end: int, window: int = 80) -> str:
    right = min(len(text), end + window)

    for boundary in ["\n", "|", ".", ";"]:
        pos = text.find(boundary, end, right)
        if pos != -1:
            right = min(right, pos)

    return text[end:right]


def detect_generic_parameter(text: str, start: int) -> str | None:
    segment = get_left_segment(text, start, window=120)
    matches = list(GENERIC_PARAMETER_RE.finditer(segment))

    if not matches:
        return None

    return compact_spaces(matches[-1].group(0)).lower()


def looks_like_entity_token(token: str) -> bool:
    token = token.strip(" ,:;()[]{}")

    if not token:
        return False

    low = token.lower()

    if low in STOP_ENTITY_WORDS:
        return False

    if CHEM_SYMBOL_RE.fullmatch(token):
        return True

    if not WORD_RE.fullmatch(token):
        return False

    if len(token) > 30:
        return False

    return True


def detect_right_entity(text: str, end: int) -> str | None:
    """
    Ловит короткую сущность справа:
    0,025 г/дм3 меди
    5 % Ni
    15-25 мг/дм3 - кобальта

    Не пытается собирать длинные фразы.
    """
    segment = get_right_segment(text, end, window=60)
    segment = segment.strip()
    segment = re.sub(r"^[\s,;:()\-–—]+", "", segment)

    if not segment:
        return None

    raw_tokens = segment.split()
    tokens: list[str] = []

    for token in raw_tokens[:2]:
        cleaned = token.strip(" ,:;()[]{}")

        if not cleaned:
            break

        if cleaned.lower() in STOP_ENTITY_WORDS:
            break

        if not looks_like_entity_token(cleaned):
            break

        tokens.append(cleaned)

        if CHEM_SYMBOL_RE.fullmatch(cleaned):
            break

        if len(tokens) >= 2:
            break

    if not tokens:
        return None

    return " ".join(tokens)


def detect_left_entity_near_parameter(text: str, start: int) -> str | None:
    """
    Ищет объект после параметра, но до предлога/связки.

    Хорошо:
    содержание свинца ... 0,1 г/л -> свинца
    содержаний свинца в растворе ... 0,015 мг/дм3 -> свинца
    концентрации Ni ... 100 г/дм3 -> Ni

    Не берет последнее слово перед числом, потому что это часто "растворе", "никелевом" и т.п.
    """
    segment = get_left_segment(text, start, window=140)
    segment = compact_spaces(segment)

    matches = list(GENERIC_PARAMETER_RE.finditer(segment))

    if not matches:
        return None

    tail = segment[matches[-1].end():]
    tail = compact_spaces(tail)

    if not tail:
        return None

    entity_tokens: list[str] = []

    for token in tail.split():
        cleaned = token.strip(" ,:;()[]{}")

        if not cleaned:
            continue

        low = cleaned.lower()

        if low in LEFT_ENTITY_BOUNDARY_WORDS:
            break

        if not looks_like_entity_token(cleaned):
            break

        entity_tokens.append(cleaned)

        if CHEM_SYMBOL_RE.fullmatch(cleaned):
            break

        if len(entity_tokens) >= 2:
            break

    if not entity_tokens:
        return None

    return " ".join(entity_tokens)


def build_parameter_hint(generic_parameter: str | None, entity: str | None) -> str | None:
    if generic_parameter and entity:
        if generic_parameter in {
            "содержание",
            "содержания",
            "содержаний",
            "концентрация",
            "концентрации",
            "концентрацию",
            "минерализация",
            "минерализации",
        }:
            return f"{generic_parameter} {entity}"

        return generic_parameter

    if generic_parameter:
        return generic_parameter

    if entity:
        return entity

    return None


def get_context(text: str, start: int, end: int, window: int = 90) -> dict[str, str]:
    return {
        "left_context": compact_spaces(text[max(0, start - window):start]),
        "right_context": compact_spaces(text[end:min(len(text), end + window)]),
    }


def overlaps_existing(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < old_end and end > old_start for old_start, old_end in spans)


def make_item(
    *,
    text: str,
    raw: str,
    start: int,
    end: int,
    operator: str | None,
    value1: str,
    value2: str | None,
    unit: str | None,
    source: str,
    parameter_hint: str | None = None,
) -> dict[str, Any]:
    has_range = value2 is not None
    canonical_unit = canonicalize_unit(unit)

    generic_parameter = detect_generic_parameter(text, start)
    right_entity = detect_right_entity(text, end)
    left_entity = detect_left_entity_near_parameter(text, start)

    # Сущность справа надежнее: "5 % Ni", "0,03 г/дм3 меди"
    entity = right_entity or left_entity
    hint = parameter_hint or build_parameter_hint(generic_parameter, entity)

    item: dict[str, Any] = {
        "raw": compact_spaces(raw),
        "operator": normalize_operator(operator, has_range),
        "unit": canonical_unit,
        "raw_value1": value1,
        "raw_value2": value2,
        "value_text": build_value_text(value1, value2, canonical_unit),
        "start": start,
        "end": end,
        "source": source,
    }

    if right_entity:
        item["right_entity"] = right_entity

    if left_entity:
        item["left_entity"] = left_entity

    if hint:
        item["parameter_hint"] = hint

    if has_range:
        v1 = parse_float(value1)
        v2 = parse_float(value2)
        item["min_value"] = min(v1, v2)
        item["max_value"] = max(v1, v2)
    else:
        item["value"] = parse_float(value1)

    item.update(get_context(text, start, end))

    return item


def extract_numeric_expressions(text: str) -> list[dict[str, Any]]:
    text = normalize_text(text)

    results: list[dict[str, Any]] = []
    used_spans: list[tuple[int, int]] = []

    # 1. "от X до Y unit"
    for match in FROM_TO_WITH_UNIT_RE.finditer(text):
        start, end = match.span()

        if should_skip_numeric_match(text, match):
            continue

        if overlaps_existing(start, end, used_spans):
            continue

        item = make_item(
            text=text,
            raw=match.group(0),
            start=start,
            end=end,
            operator=match.group("operator"),
            value1=match.group("value1"),
            value2=match.group("value2"),
            unit=match.group("unit"),
            source="from_to_regex",
        )

        results.append(item)
        used_spans.append((start, end))

    # 2. pH 10,8 / pH 10,8-11,4
    for match in PH_RE.finditer(text):
        start, end = match.span()

        if should_skip_ph_match(text, match):
            continue

        if overlaps_existing(start, end, used_spans):
            continue

        item = make_item(
            text=text,
            raw=match.group(0),
            start=start,
            end=end,
            operator=match.group("operator"),
            value1=match.group("value1"),
            value2=match.group("value2"),
            unit="pH",
            source="ph_regex",
            parameter_hint="pH",
        )

        results.append(item)
        used_spans.append((start, end))

    # 3. X-Y unit / <= X unit / X unit
    for match in NUMERIC_WITH_UNIT_RE.finditer(text):
        start, end = match.span()

        if should_skip_numeric_match(text, match):
            continue

        if overlaps_existing(start, end, used_spans):
            continue

        item = make_item(
            text=text,
            raw=match.group(0),
            start=start,
            end=end,
            operator=match.group("operator"),
            value1=match.group("value1"),
            value2=match.group("value2"),
            unit=match.group("unit"),
            source="unit_regex",
        )

        results.append(item)
        used_spans.append((start, end))

    results.sort(key=lambda item: item["start"])
    return results


def format_value_for_display(item: dict[str, Any]) -> str:
    value_text = item.get("value_text") or item.get("raw", "")
    operator = item.get("operator")

    if operator == "range":
        return value_text

    if operator in {"<", "<=", ">", ">=", "~"}:
        return f"{operator}{value_text}"

    return value_text


def normalize_label_text(text: str) -> str:
    return compact_spaces(text).lower().replace("ё", "е")


def canonicalize_parameter_label(parameter: str) -> str:
    p = compact_spaces(parameter)
    parts = p.split()

    if not parts:
        return p

    replacements = {
        "содержаний": "содержание",
        "содержания": "содержание",
        "содержание": "содержание",
        "концентрации": "концентрация",
        "концентрацию": "концентрация",
        "концентрация": "концентрация",
    }

    first_word = parts[0].lower()

    if first_word in replacements:
        rest = " ".join(parts[1:])

        if rest:
            return f"{replacements[first_word]} {rest}"

        return replacements[first_word]

    return p


def label_already_contains_entity(label: str, entity: str) -> bool:
    label_norm = normalize_label_text(label)
    entity_norm = normalize_label_text(entity)

    return entity_norm in label_norm.split() or entity_norm in label_norm


def format_numeric_expression(item: dict[str, Any]) -> str:
    value_text = format_value_for_display(item)

    right_entity = item.get("right_entity")
    left_entity = item.get("left_entity")
    parameter = item.get("parameter_hint")

    if parameter:
        parameter = canonicalize_parameter_label(str(parameter))

    # Самое надежное: сущность прямо справа после числа.
    # 5 % Ni, 0,2-0,5 % Cu, 0,03 г/дм3 меди.
    if right_entity:
        return f"{right_entity}: {value_text}"

    # Если parameter уже содержит entity, не дублируем.
    if parameter and left_entity:
        if label_already_contains_entity(parameter, str(left_entity)):
            return f"{parameter}: {value_text}"

        return f"{parameter} {left_entity}: {value_text}"

    if parameter:
        return f"{parameter}: {value_text}"

    if left_entity:
        return f"{left_entity}: {value_text}"

    return value_text


def format_numeric_expressions(items: list[dict[str, Any]], limit: int = 16) -> str:
    if not items:
        return "—"

    formatted: list[str] = []
    seen: set[str] = set()

    for item in items:
        value = format_numeric_expression(item)

        if value in seen:
            continue

        seen.add(value)
        formatted.append(value)

        if len(formatted) >= limit:
            break

    if len(items) > len(formatted):
        formatted.append(f"... +{len(items) - len(formatted)}")

    return "; ".join(formatted)


def debug_numeric_candidates(text: str) -> list[str]:
    text = normalize_text(text)

    rows: list[str] = []

    for match in NUMERIC_WITH_UNIT_RE.finditer(text):
        reason = explain_skip_reason_for_numeric_match(text, match)

        rows.append(
            f"{match.group(0)!r} at {match.start()}-{match.end()} -> {reason or 'OK'}"
        )

    return rows


if __name__ == "__main__":
    sample = """
    NCM811 А. А. Коржаков, ORCID 0000-0001-9361-8165.
    Установлено, что достижима глубина очистки от меди вплоть до содержаний 0,025-0,03 г/дм3 меди.
    Содержание цветных металлов было снижено с 5 % Ni и 1,5 % Cu до 1-2% Ni и 0,2-0,5% Cu.
    Содержание свинца не должно превышать 0,1 г/л, в исходных растворах оно может достигать 1-2 г/л.
    Стандартный потенциал пары Pb2+/Pb равен -0,126 В, а Ni2+/Ni -0,24 В.
    Согласно результатам удается достичь содержания в фильтрате примесей порядка: 15-25 мг/дм3 - кобальта,
    0,2-0,4 мг/дм3 – свинца, менее 0,05 мг/ дм3 – марганца и 0,02-0,04 мг/ дм3 – меди.
    Раствор разбавляется в 2-2,5 раза от первоначальной концентрации с 200-250 г/дм3 до ~100 г/дм3 Ni.
    Скорость перемешивания составляла 300-500 об/мин.
    pH 10,8-11,4 поддерживали в процессе.

    | Образец | Скорость перемешивания, об/мин | pH |
    | ------- | ------------------------------- | -- |
    | P-2     | 300                             | 11 |
    """

    print("EXTRACTED")
    print("=" * 100)

    for expression in extract_numeric_expressions(sample):
        print(format_numeric_expression(expression))

    print()
    print("DEBUG")
    print("=" * 100)

    for row in debug_numeric_candidates(sample):
        print(row)