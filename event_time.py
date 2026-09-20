import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo


VIETNAM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def _plain_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", (value or "").casefold())
    return "".join(
        character for character in normalized
        if unicodedata.category(character) != "Mn"
    ).replace("đ", "d")


def _posted_date(value) -> date | None:
    if value is None:
        return None
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        value = to_native()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            # Neo4j local datetimes in this project represent Vietnam local time.
            return value.date()
        return value.astimezone(VIETNAM_TIMEZONE).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(VIETNAM_TIMEZONE)
            return parsed.date()
        except ValueError:
            try:
                return date.fromisoformat(text[:10])
            except ValueError:
                return None
    return None


def _valid_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _explicit_date(text: str, anchor: date | None) -> date | None:
    patterns = (
        (r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", (1, 2, 3)),
        (r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?!\d)", (3, 2, 1)),
        (
            r"\b(?:ngay\s+)?(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})\b",
            (3, 2, 1),
        ),
    )
    for pattern, order in patterns:
        match = re.search(pattern, text)
        if match:
            parts = [int(match.group(index)) for index in order]
            return _valid_date(*parts)

    if anchor is None:
        return None
    partial_patterns = (
        r"(?<![\d/.])(\d{1,2})[-/.](\d{1,2})(?![-/.]\d)",
        r"\b(?:ngay\s+)?(\d{1,2})\s+thang\s+(\d{1,2})(?!\s+nam\b)",
    )
    for pattern in partial_patterns:
        match = re.search(pattern, text)
        if match:
            return _valid_date(anchor.year, int(match.group(2)), int(match.group(1)))
    return None


def normalize_occurrence_date(
    time_expression: str | None,
    posted_at,
    *,
    evidence_text: str | None = None,
) -> dict:
    """Resolve event time to day precision, retaining how the day was derived."""
    anchor = _posted_date(posted_at)
    raw_text = (time_expression or "").strip()
    # Evidence is event-scoped and is safer than scanning a whole multi-event post.
    source_text = raw_text or (evidence_text or "")
    text = _plain_text(source_text)

    explicit = _explicit_date(text, anchor)
    if explicit is not None:
        return {"occurrence_date": explicit, "occurrence_date_source": "EXPLICIT"}

    if anchor is not None:
        relative_rules = (
            (r"\bhom kia\b", -2),
            (r"\bhom qua\b", -1),
            (r"\bhom nay\b", 0),
            (r"\bngay mai\b", 1),
        )
        for pattern, offset in relative_rules:
            if re.search(pattern, text):
                return {
                    "occurrence_date": anchor + timedelta(days=offset),
                    "occurrence_date_source": "RELATIVE",
                }
        match = re.search(r"\b(\d+)\s+ngay\s+(?:truoc|qua)\b", text)
        if match:
            return {
                "occurrence_date": anchor - timedelta(days=int(match.group(1))),
                "occurrence_date_source": "RELATIVE",
            }
    # posted_at is only a candidate-retrieval fallback. It is not occurrence
    # evidence and must not be persisted as the event's occurrence date.
    return {"occurrence_date": None, "occurrence_date_source": None}
