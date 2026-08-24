import re
from typing import Protocol

from backend.models import (
    ChatResponse,
    DetailQuery,
    DetailResult,
    EventResult,
    ParsedQuestion,
)


_DETAIL_COMMAND_RE = re.compile(
    r"^/detail(?:\s+(.*))?$",
    re.IGNORECASE | re.DOTALL,
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
        limit: int,
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
    ) -> str:
        del question
        area = f" tại {parsed.location}" if parsed.location else ""
        subject = f" liên quan tới {parsed.entity}" if parsed.entity else ""
        scope = f"{subject}{area}"
        if not events:
            return f"Không tìm thấy sự kiện{scope}."

        lines = [f"Tìm thấy {len(events)} sự kiện{scope}:"]
        for index, event in enumerate(events, start=1):
            source = event.post.source_name or event.post.platform
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

    def chat(self, message: str, limit: int) -> ChatResponse:
        detail_match = _DETAIL_COMMAND_RE.fullmatch(message.strip())
        if detail_match:
            subject = (detail_match.group(1) or "").strip()
            if not subject:
                raise InvalidChatCommand(
                    "Thiếu chủ thể. Cú pháp đúng: /detail <tên chủ thể>"
                )
            return self._get_subject_detail(subject, limit)

        parsed = self.parser.parse(message)
        raw_results = self.repository.search_events(
            location=parsed.location,
            entity=parsed.entity,
            hours=parsed.hours,
            limit=limit,
        )
        results = [EventResult.model_validate(item) for item in raw_results]
        return ChatResponse(
            answer=self.answer_generator.generate(
                question=message,
                parsed=parsed,
                events=results,
            ),
            query=parsed,
            count=len(results),
            results=results,
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
