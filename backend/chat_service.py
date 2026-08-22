from typing import Protocol

from backend.models import ChatResponse, EventResult, ParsedQuestion


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
