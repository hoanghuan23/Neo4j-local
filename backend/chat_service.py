from typing import Protocol

from backend.models import ChatResponse, EventResult


class EventRepository(Protocol):
    def search_events(
        self, *, location: str | None, limit: int
    ) -> list[dict]: ...


class ChatService:
    def __init__(self, parser, repository: EventRepository):
        self.parser = parser
        self.repository = repository

    def chat(self, message: str, limit: int) -> ChatResponse:
        parsed = self.parser.parse(message)
        raw_results = self.repository.search_events(
            location=parsed.location,
            limit=limit,
        )
        results = [EventResult.model_validate(item) for item in raw_results]
        return ChatResponse(
            answer=self._generate_answer(parsed.location, results),
            query=parsed,
            count=len(results),
            results=results,
        )

    @staticmethod
    def _generate_answer(
        location: str | None,
        results: list[EventResult],
    ) -> str:
        area = f" tại {location}" if location else ""
        if not results:
            return f"Không tìm thấy sự kiện{area}."

        lines = [f"Tìm thấy {len(results)} sự kiện{area}:"]
        for index, result in enumerate(results, start=1):
            source = result.post.source_name or result.post.platform
            lines.append(f"{index}. {result.description} (Nguồn: {source})")
        return "\n".join(lines)
