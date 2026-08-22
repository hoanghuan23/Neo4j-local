import re

from backend.models import ParsedQuestion


_SPACE_RE = re.compile(r"\s+")
_HOURS_RE = re.compile(r"\b(\d{1,3})\s*(?:h|giờ|tiếng)\b", re.IGNORECASE)
_DAYS_RE = re.compile(r"\b(\d{1,2})\s*ngày\b", re.IGNORECASE)
_TIME_MARKER = (
    r"(?:trong\s+)?\d+\s*(?:h|giờ|tiếng|ngày)"
    r"|hôm\s+nay|hôm\s+qua|gần\s+đây|vừa\s+qua"
)
_LOCATION_AFTER_PREPOSITION_RE = re.compile(
    rf"\b(?:ở|tại|khu\s+vực)\s+(.+?)(?=\s+(?:{_TIME_MARKER})\b|[?.,!]|$)",
    re.IGNORECASE,
)
_LEADING_QUESTION_RE = re.compile(
    r"^(?:cho\s+(?:tôi|mình)\s+biết\s+|tìm\s+|tin\s+tức\s+|"
    r"sự\s+kiện\s+|có\s+gì\s+xảy\s+ra\s+)(?:ở\s+|tại\s+)?",
    re.IGNORECASE,
)
_EVENT_LOCATION_RE = re.compile(
    r"^(?:tìm\s+)?(?:các\s+)?sự\s+kiện(?:\s+(?:ở|tại))?\s+(.+?)"
    r"(?:[?.,!]|$)",
    re.IGNORECASE,
)
_SUPPORTED_BARE_LOCATIONS = {"hà nội"}


class RuleBasedQuestionParser:
    """Small deterministic parser that can later be replaced by an LLM parser."""

    def __init__(self, default_hours: int = 24, max_hours: int = 720):
        self.default_hours = default_hours
        self.max_hours = max_hours

    def parse(self, question: str) -> ParsedQuestion:
        text = _SPACE_RE.sub(" ", question.strip())
        hours = self._parse_hours(text)
        location = self._parse_location(text)
        return ParsedQuestion(location=location, hours=hours)

    def _parse_hours(self, text: str) -> int:
        hours_match = _HOURS_RE.search(text)
        if hours_match:
            return min(max(int(hours_match.group(1)), 1), self.max_hours)

        days_match = _DAYS_RE.search(text)
        if days_match:
            return min(max(int(days_match.group(1)) * 24, 1), self.max_hours)

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
