import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from neo4j.exceptions import Neo4jError

from backend.chat_service import ChatService
from backend.config import Settings
from backend.models import ChatRequest, ChatResponse, HealthResponse
from backend.neo4j_repository import Neo4jRepository
from backend.question_parser import RuleBasedQuestionParser


LOGGER = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, repository=None) -> FastAPI:
    settings = settings or Settings()
    repository = repository or Neo4jRepository(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository.connect()
        app.state.repository = repository
        app.state.chat_service = ChatService(
            RuleBasedQuestionParser(
                default_hours=settings.default_search_hours,
                max_hours=settings.max_search_hours,
            ),
            repository,
        )
        try:
            yield
        finally:
            repository.close()

    app = FastAPI(
        title="Neo4j Chat Search API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    def run_chat(request: Request, message: str, limit: int) -> ChatResponse:
        try:
            return request.app.state.chat_service.chat(message, limit)
        except Neo4jError as exc:
            LOGGER.exception("Neo4j search failed")
            raise HTTPException(
                status_code=503,
                detail="Không thể truy vấn Neo4j lúc này",
            ) from exc

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        connected = request.app.state.repository.ping()
        return HealthResponse(
            status="ok" if connected else "degraded",
            neo4j="connected" if connected else "disconnected",
        )

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        return run_chat(request, payload.message.strip(), payload.limit)

    @app.get("/api/search", response_model=ChatResponse)
    def search(
        request: Request,
        q: str = Query(min_length=1, max_length=2_000),
        limit: int = Query(
            default=settings.default_result_limit,
            ge=1,
            le=50,
        ),
    ) -> ChatResponse:
        return run_chat(request, q.strip(), limit)

    return app


app = create_app()

