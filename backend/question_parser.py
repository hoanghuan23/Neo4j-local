import re
from datetime import date

from backend.models import ParsedQuestion


_SPACE_RE = re.compile(r"\s+")
_HOURS_RE = re.compile(r"\b(\d{1,3})\s*(?:h|giờ|tiếng)\b", re.IGNORECASE)
_DAYS_RE = re.compile(r"\b(\d{1,2})\s*ngày\b", re.IGNORECASE)
_WEEKS_RE = re.compile(r"\b(\d{1,2})\s*tuần\b", re.IGNORECASE)
_MONTH_DURATION_PATTERN = (
    r"(?<!ngày\s)\d{1,2}\s*tháng"
    r"(?:\s+(?:qua|trở\s+lại\s+đây))?"
)
_MONTHS_RE = re.compile(
    rf"\b({_MONTH_DURATION_PATTERN})\b",
    re.IGNORECASE,
)
_CALENDAR_DATE_PATTERN = (
    r"ngày\s+(?:0?[1-9]|[12]\d|3[01])\s+"
    r"tháng\s+(?:0?[1-9]|1[0-2])"
    r"(?:\s+năm\s+\d{4})?"
)
_CALENDAR_DATE_RE = re.compile(
    r"\bngày\s+(0?[1-9]|[12]\d|3[01])\s+"
    r"tháng\s+(0?[1-9]|1[0-2])"
    r"(?:\s+năm\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_TIME_MARKER = (
    r"(?:trong\s+)?\d+\s*(?:h|giờ|tiếng|ngày|tuần)"
    r"|(?:trong\s+)?tuần\s+trước"
    rf"|{_CALENDAR_DATE_PATTERN}"
    rf"|{_MONTH_DURATION_PATTERN}"
    r"|hôm\s+nay|hôm\s+qua|gần\s+đây|vừa\s+qua"
)
_PREVIOUS_WEEK_RE = re.compile(r"\b(?:trong\s+)?tuần\s+trước\b", re.IGNORECASE)
_LOCATION_AFTER_PREPOSITION_RE = re.compile(
    rf"\b(?:ở|tại|khu\s+vực)\s+(.+?)(?=\s+(?:{_TIME_MARKER})\b|[?.,!]|$)",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(
    rf"\b(?:liên\s+quan(?:\s+(?:đến|tới|giữa))?|nhắc\s+(?:đến|tới)|về)\s+"
    rf"(.+?)(?=\s+(?:(?:ở|tại|khu\s+vực)\s+|(?:{_TIME_MARKER})\b)"
    r"|[?.,!]|$)",
    re.IGNORECASE,
)
_LEADING_QUESTION_RE = re.compile(
    r"^(?:cho\s+(?:tôi|mình)\s+biết\s+|tìm\s+|tin\s+tức\s+|"
    r"(?:các\s+)?sự\s+kiện\s+|tình\s+hình\s+|có\s+gì\s+xảy\s+ra\s+)"
    r"(?:ở\s+|tại\s+)?",
    re.IGNORECASE,
)
_EVENT_LOCATION_RE = re.compile(
    r"^(?:tìm\s+)?(?:các\s+)?sự\s+kiện(?:\s+(?:ở|tại))?\s+(.+?)"
    r"(?:[?.,!]|$)",
    re.IGNORECASE,
)
_SUPPORTED_BARE_LOCATIONS = {"hà nội"}
_BROAD_LOCATION_PREFIX_RE = re.compile(
    r"^(?:(?:thành\s+phố|tỉnh|tp)\.?\s+)",
    re.IGNORECASE,
)
_ENTITY_RELATION_PREFIX_RE = re.compile(
    r"^giữa\s+(?=.+\s+(?:và|hoặc|hay)\s+.+$)",
    re.IGNORECASE,
)
_ENTITY_TIME_SUFFIX_RE = re.compile(
    rf"\s+(?:{_TIME_MARKER})(?:\s+qua)?$",
    re.IGNORECASE,
)
_LATEST_EVENTS_QUERY_RE = re.compile(
    r"^(?:(?:cho\s+(?:tôi|mình)\s+biết|tìm)\s+)?"
    r"(?:các\s+|những\s+)?sự\s+kiện\s+"
    r"(?:mới\s+nhất|mới\s+đây|gần\s+đây)$",
    re.IGNORECASE,
)


def is_latest_events_query(question: str) -> bool:
    """Return whether the question requests the default latest-event feed."""
    normalized = _SPACE_RE.sub(" ", question.strip()).strip(" \t,?.!")
    return _LATEST_EVENTS_QUERY_RE.fullmatch(normalized) is not None


def normalize_location_for_search(location: str | None) -> str | None:
    """Drop broad administrative labels while retaining the place name."""
    if location is None:
        return None
    normalized = _BROAD_LOCATION_PREFIX_RE.sub("", location.strip()).strip()
    return normalized or None


def normalize_entity_for_search(entity: str | None) -> str | None:
    """Remove relation wording while retaining the requested entity names."""
    if entity is None:
        return None
    normalized = " ".join(entity.strip().split())
    normalized = _ENTITY_TIME_SUFFIX_RE.sub("", normalized).strip()
    normalized = _ENTITY_RELATION_PREFIX_RE.sub("", normalized).strip()
    return normalized or None


def has_explicit_duration(text: str) -> bool:
    """Return whether the deterministic parser recognizes a duration."""
    return any(
        pattern.search(text)
        for pattern in (
            _HOURS_RE,
            _DAYS_RE,
            _WEEKS_RE,
            _MONTHS_RE,
            _PREVIOUS_WEEK_RE,
        )
    )


class RuleBasedQuestionParser:
    """Small deterministic parser that can later be replaced by an LLM parser."""

    def __init__(
        self,
        default_hours: int = 24,
        max_hours: int = 720,
        today_provider=date.today,
    ):
        self.default_hours = default_hours
        self.max_hours = max_hours
        self.today_provider = today_provider

    def parse(self, question: str) -> ParsedQuestion:
        text = _SPACE_RE.sub(" ", question.strip())
        if is_latest_events_query(text):
            entity = None
            location = None
        else:
            entity = normalize_entity_for_search(self._parse_entity(text))
            location = self._parse_location(text)
        return ParsedQuestion(
            location=normalize_location_for_search(location),
            entity=entity,
            hours=self._parse_hours(text),
            posted_date=self._parse_posted_date(text),
        )

    def _parse_posted_date(self, text: str) -> date | None:
        match = _CALENDAR_DATE_RE.search(text)
        if not match:
            return None
        day, month, year = match.groups()
        try:
            return date(
                int(year) if year else self.today_provider().year,
                int(month),
                int(day),
            )
        except ValueError:
            return None

    def _parse_hours(self, text: str) -> int:
        if _PREVIOUS_WEEK_RE.search(text):
            return min(7 * 24, self.max_hours)

        hours_match = _HOURS_RE.search(text)
        if hours_match:
            return min(max(int(hours_match.group(1)), 1), self.max_hours)

        days_match = _DAYS_RE.search(text)
        if days_match:
            return min(max(int(days_match.group(1)) * 24, 1), self.max_hours)

        weeks_match = _WEEKS_RE.search(text)
        if weeks_match:
            return min(max(int(weeks_match.group(1)) * 7 * 24, 1), self.max_hours)

        months_match = _MONTHS_RE.search(text)
        if months_match:
            months = int(re.match(r"\d+", months_match.group(1)).group())
            return min(max(months * 30 * 24, 1), self.max_hours)

        return min(max(self.default_hours, 1), self.max_hours)

    @staticmethod
    def _clean_location(value: str) -> str | None:
        value = re.sub(
            r"\s+(?:có\s+gì|xảy\s+ra|có\s+tin\s+gì|thế\s+nào).*$",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = value.strip(" \t,?.!")
        return value or None

    def _parse_location(self, text: str) -> str | None:
        explicit = _LOCATION_AFTER_PREPOSITION_RE.search(text)
        if explicit:
            return self._clean_location(explicit.group(1))

        if _ENTITY_RE.search(text):
            return None

        time_match = re.search(rf"\b(?:{_TIME_MARKER})\b", text, re.IGNORECASE)
        if time_match and time_match.start() > 0:
            prefix = _LEADING_QUESTION_RE.sub("", text[: time_match.start()].strip())
            return self._clean_location(prefix)

        event_location = _EVENT_LOCATION_RE.match(text)
        if event_location:
            return self._clean_location(event_location.group(1))

        if text.casefold().strip(" \t,?.!") in _SUPPORTED_BARE_LOCATIONS:
            return self._clean_location(text)

        return None

    @classmethod
    def _parse_entity(cls, text: str) -> str | None:
        match = _ENTITY_RE.search(text)
        if not match:
            return None
        return cls._clean_location(match.group(1))
