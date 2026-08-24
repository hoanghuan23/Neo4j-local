from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=10, ge=1, le=50)


class ParsedQuestion(BaseModel):
    intent: Literal["search_events"] = "search_events"
    location: str | None = None
    entity: str | None = None
    hours: int = Field(ge=1, le=720)


class DetailQuery(BaseModel):
    intent: Literal["detail"] = "detail"
    subject: str = Field(min_length=1)


class EntityResult(BaseModel):
    name: str
    type: str
    role: str | None = None


class PostResult(BaseModel):
    platform: str
    platform_id: str
    content: str
    url: str | None = None
    posted_at: str | None = None
    source_name: str | None = None


class SourceResult(BaseModel):
    source: str
    posted_at: str | None = None
    url: str | None = None


class EventResult(BaseModel):
    event_key: str
    type: str
    description: str
    status: str | None = None
    time_expression: str | None = None
    entities: list[EntityResult] = Field(default_factory=list)
    post: PostResult
    sources: list[SourceResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def populate_sources(self) -> "EventResult":
        if not self.sources:
            self.sources = [
                SourceResult(
                    source=self.post.source_name or self.post.platform,
                    posted_at=self.post.posted_at,
                    url=self.post.url,
                )
            ]
        return self


class DetailResult(BaseModel):
    entity_type: str
    entity_name: str
    post_count: int = Field(ge=0)


class ChatResponse(BaseModel):
    answer: str
    query: ParsedQuestion | DetailQuery
    count: int
    results: list[EventResult] = Field(default_factory=list)
    details: list[DetailResult] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    neo4j: Literal["connected", "disconnected"]
