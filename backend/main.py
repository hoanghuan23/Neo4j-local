import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from backend.chat_service import (
    ChatService,
    InvalidChatCommand,
    TemplateAnswerGenerator,
)
from backend.config import Settings
from backend.openai_services import (
    FallbackAnswerGenerator,
    FallbackQuestionParser,
    OpenAIAnswerGenerator,
    OpenAIQuestionParser,
)
from backend.models import ChatRequest, ChatResponse, HealthResponse, RelatedSearchRequest
from backend.neo4j_repository import Neo4jRepository
from backend.question_parser import RuleBasedQuestionParser


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Neo4j notifications include the full Cypher query; keep routine chat logs
# focused on model usage while still reporting operational errors.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def create_app(settings: Settings | None = None, repository=None) -> FastAPI:
    settings = settings or Settings()
    repository = repository or Neo4jRepository(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository.connect()
        app.state.repository = repository
        openai_client = None
        rule_parser = RuleBasedQuestionParser(
            default_hours=settings.default_search_hours,
            max_hours=settings.max_search_hours,
        )
        template_generator = TemplateAnswerGenerator()

        if settings.openai_api_key:
            from openai import OpenAI

            openai_client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=max(1, settings.chat_openai_timeout_seconds),
            )
            parser = FallbackQuestionParser(
                OpenAIQuestionParser(
                    client=openai_client,
                    model=settings.chat_openai_model,
                    default_hours=settings.default_search_hours,
                    input_price_per_million_usd=(
                        settings.chat_openai_input_price_per_million_usd
                    ),
                    output_price_per_million_usd=(
                        settings.chat_openai_output_price_per_million_usd
                    ),
                ),
                rule_parser,
            )
            answer_generator = FallbackAnswerGenerator(
                OpenAIAnswerGenerator(
                    client=openai_client,
                    model=settings.chat_openai_model,
                    input_price_per_million_usd=(
                        settings.chat_openai_input_price_per_million_usd
                    ),
                    output_price_per_million_usd=(
                        settings.chat_openai_output_price_per_million_usd
                    ),
                ),
                template_generator,
            )
        else:
            LOGGER.warning(
                "OPENAI_API_KEY is not configured; using deterministic "
                "question parsing and answer generation"
            )
            parser = rule_parser
            answer_generator = template_generator

        app.state.chat_service = ChatService(
            parser,
            repository,
            answer_generator,
        )
        try:
            yield
        finally:
            try:
                if openai_client is not None:
                    openai_client.close()
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

    def run_chat(
        request: Request,
        message: str,
        limit: int,
        cursor: str | None = None,
    ) -> ChatResponse:
        try:
            return request.app.state.chat_service.chat(message, limit, cursor)
        except InvalidChatCommand as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        return run_chat(
            request,
            payload.message.strip(),
            payload.limit,
            payload.cursor,
        )

    @app.post("/api/search/related", response_model=ChatResponse)
    def search_related(payload: RelatedSearchRequest, request: Request) -> ChatResponse:
        try:
            return request.app.state.chat_service.search_related(payload)
        except InvalidChatCommand as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (Neo4jError, ServiceUnavailable) as exc:
            LOGGER.exception("Neo4j related search failed")
            raise HTTPException(
                status_code=503, detail="Không thể truy vấn Neo4j lúc này",
            ) from exc

    @app.get("/api/search", response_model=ChatResponse)
    def search(
        request: Request,
        q: str = Query(min_length=1, max_length=2_000),
        limit: int = Query(
            default=settings.default_result_limit,
            ge=1,
            le=30,
        ),
        cursor: str | None = Query(default=None, max_length=4_096),
    ) -> ChatResponse:
        return run_chat(request, q.strip(), limit, cursor)

    return app


app = create_app()
