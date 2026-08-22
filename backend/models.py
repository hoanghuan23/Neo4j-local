from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=10, ge=1, le=50)


class ParsedQuestion(BaseModel):
    intent: Literal["search_events"] = "search_events"
    location: str | None = None
    hours: int = Field(ge=1, le=720)


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


class EventResult(BaseModel):
    event_key: str
    type: str
    description: str
    status: str | None = None
    time_expression: str | None = None
    entities: list[EntityResult] = Field(default_factory=list)
    post: PostResult


class ChatResponse(BaseModel):
    answer: str
    query: ParsedQuestion
    count: int
    results: list[EventResult]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    neo4j: Literal["connected", "disconnected"]

