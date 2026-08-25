import re
from datetime import date
from typing import Protocol

from backend.models import (
    ChatResponse,
    DetailQuery,
    DetailResult,
    EventSearchCursor,
    EventResult,
    ParsedQuestion,
)
from backend.pagination import decode_event_cursor, encode_event_cursor


_DETAIL_COMMAND_RE = re.compile(
    r"^/detail(?:\s+(.*))?$",
    re.IGNORECASE | re.DOTALL,
)
_CONTINUE_COMMAND_RE = re.compile(
    r"^(?:xem\s+tiếp|tiếp|xem\s+thêm|thêm(?:\s+nữa)?)\W*$",
    re.IGNORECASE,
)


class InvalidChatCommand(ValueError):
    pass


class QuestionParser(Protocol):
    def parse(self, question: str) -> ParsedQuestion: ...


class EventRepository(Protocol):
    def search_events(
        self,
        *,
        location: str | None,
        entity: str | None,
        hours: int,
        posted_date: date | None,
        limit: int,
        after: tuple[int, str, str] | None = None,
    ) -> list[dict]: ...

    def search_related_entities(
        self,
        *,
        subject: str,
        limit: int,
    ) -> list[dict]: ...


class AnswerGenerator(Protocol):
    def generate(
        self,
        *,
        question: str,
        parsed: ParsedQuestion,
        events: list[EventResult],
    ) -> str: ...


class TemplateAnswerGenerator:
    def generate(
        self,
        *,
        question: str,
        parsed: ParsedQuestion,
        events: list[EventResult],
        start_index: int = 1,
    ) -> str:
        del question
        area = f" tại {parsed.location}" if parsed.location else ""
        subject = f" liên quan tới {parsed.entity}" if parsed.entity else ""
        scope = f"{subject}{area}"
        if not events:
            return f"Không tìm thấy sự kiện{scope}."

        lines = [f"Tìm thấy {len(events)} sự kiện{scope}:"]
        for index, event in enumerate(events, start=start_index):
            source = event.sources[0].source
            additional_source_count = len(event.sources) - 1
            if additional_source_count:
                source = f"{source} +{additional_source_count}"
            lines.append(f"{index}. {event.description} (Nguồn: {source})")
        return "\n".join(lines)


class ChatService:
    def __init__(
        self,
        parser: QuestionParser,
        repository: EventRepository,
        answer_generator: AnswerGenerator | None = None,
    ):
        self.parser = parser
        self.repository = repository
        self.answer_generator = answer_generator or TemplateAnswerGenerator()

    def chat(
        self,
        message: str,
        limit: int,
        cursor: str | None = None,
    ) -> ChatResponse:
        if cursor is not None:
            return self._continue_search(message, limit, cursor)

        detail_match = _DETAIL_COMMAND_RE.fullmatch(message.strip())
        if detail_match:
            subject = (detail_match.group(1) or "").strip()
            if not subject:
                raise InvalidChatCommand(
                    "Thiếu chủ thể. Cú pháp đúng: /detail <tên chủ thể>"
                )
            return self._get_subject_detail(subject, limit)

        if _CONTINUE_COMMAND_RE.fullmatch(message.strip()):
            raise InvalidChatCommand(
                "Không có truy vấn trước để xem tiếp. Hãy tìm kiếm lại."
            )

        parsed = self.parser.parse(message)
        return self._search_events(
            message=message,
            parsed=parsed,
            limit=limit,
            start_index=1,
        )

    def _continue_search(
        self,
        message: str,
        limit: int,
        cursor: str,
    ) -> ChatResponse:
        try:
            decoded = decode_event_cursor(cursor)
        except ValueError as exc:
            raise InvalidChatCommand(str(exc)) from exc
        return self._search_events(
            message=message,
            parsed=decoded.query,
            limit=limit,
            start_index=decoded.returned + 1,
            after=decoded.sort_key,
            continuation=True,
        )

    def _search_events(
        self,
        *,
        message: str,
        parsed: ParsedQuestion,
        limit: int,
        start_index: int,
        after: tuple[int, str, str] | None = None,
        continuation: bool = False,
    ) -> ChatResponse:
        raw_results = self.repository.search_events(
            location=parsed.location,
            entity=parsed.entity,
            hours=parsed.hours,
            posted_date=parsed.posted_date,
            limit=limit + 1,
            after=after,
        )
        has_more = len(raw_results) > limit
        page_rows = raw_results[:limit]
        results = [EventResult.model_validate(item) for item in page_rows]
        answer_generator = (
            TemplateAnswerGenerator() if continuation else self.answer_generator
        )
        answer_kwargs = {
            "question": message,
            "parsed": parsed,
            "events": results,
        }
        if continuation:
            answer_kwargs["start_index"] = start_index

        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_event_cursor(
                EventSearchCursor(
                    query=parsed,
                    returned=start_index - 1 + len(page_rows),
                    matched_entity_count=last.get("matched_entity_count", 0),
                    posted_at=last["post"].get("posted_at") or "",
                    event_key=last["event_key"],
                )
            )
        return ChatResponse(
            answer=answer_generator.generate(**answer_kwargs),
            query=parsed,
            count=len(results),
            results=results,
            has_more=has_more,
            next_cursor=next_cursor,
            start_index=start_index,
        )

    def _get_subject_detail(self, subject: str, limit: int) -> ChatResponse:
        raw_details = self.repository.search_related_entities(
            subject=subject,
            limit=limit,
        )
        details = [DetailResult.model_validate(item) for item in raw_details]
        if not details:
            answer = (
                f'Không tìm thấy entity nào cùng xuất hiện trong bài viết với '
                f'"{subject}".'
            )
        else:
            lines = [
                f'Tìm thấy {len(details)} entity cùng xuất hiện với "{subject}":'
            ]
            for index, detail in enumerate(details, start=1):
                lines.append(
                    f"{index}. {detail.entity_name} ({detail.entity_type}): "
                    f"{detail.post_count} bài viết chung"
                )
            answer = "\n".join(lines)

        return ChatResponse(
            answer=answer,
            query=DetailQuery(subject=subject),
            count=len(details),
            details=details,
        )
