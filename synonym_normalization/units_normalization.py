from typing import Optional

UNIT_ALIASES = {
    "г/дм3": "г/л",
    "г/дм³": "г/л",
    "мг/дм3": "мг/л",
    "мг/дм³": "мг/л",
    "℃": "°C",
    "C": "°C",
    "А/м²": "А/м2",
    "м³": "м3",
    "м³/ч": "м3/ч",
    "об/м": "об/мин",
}


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    """Normalize common measurement unit aliases."""
    if not unit:
        return None

    unit_clean = unit.strip()
    return UNIT_ALIASES.get(unit_clean, unit_clean)
