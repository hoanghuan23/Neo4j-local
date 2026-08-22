import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    cors_origins: tuple[str, ...] = _csv(
        os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173",
        )
    )
    default_search_hours: int = int(os.getenv("DEFAULT_SEARCH_HOURS", "24"))
    max_search_hours: int = int(os.getenv("MAX_SEARCH_HOURS", "720"))
    default_result_limit: int = int(os.getenv("DEFAULT_RESULT_LIMIT", "10"))

