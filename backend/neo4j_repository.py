import unicodedata
from typing import Any

from neo4j import GraphDatabase

from backend.config import Settings


SEARCH_EVENTS_QUERY = """
MATCH (post:Post)-[:HAS_EVENT_MENTION]->(mention:EventMention)
      -[:EVIDENCE_FOR]->(event:Event)
WHERE (
    $location_key IS NULL
    OR EXISTS {
      MATCH (mention)-[:HAS_PARTICIPANT]->(location:Entity)
      WHERE location.type = 'LOCATION'
        AND (
          location.normalized_name CONTAINS $location_key
          OR $location_key IN coalesce(location.aliases, [])
          OR location.search_name CONTAINS $location_search_key
        )
    }
    OR EXISTS {
      MATCH (post)-[:MENTIONS]->(location:Entity)
      WHERE location.type = 'LOCATION'
        AND (
          location.normalized_name CONTAINS $location_key
          OR $location_key IN coalesce(location.aliases, [])
          OR location.search_name CONTAINS $location_search_key
        )
    }
    OR toLower(coalesce(post.content, '')) CONTAINS $location_key
    OR toLower(coalesce(mention.description, '')) CONTAINS $location_key
    OR toLower(coalesce(event.description, '')) CONTAINS $location_key
  )
OPTIONAL MATCH (source:Source)-[:PUBLISHED]->(post)
OPTIONAL MATCH (mention)-[participation:HAS_PARTICIPANT]->(entity:Entity)
WITH post, mention, event, source,
     collect(DISTINCT CASE WHEN entity IS NULL THEN NULL ELSE {
       name: coalesce(entity.name, entity.normalized_name),
       type: entity.type,
       role: participation.role
     } END) AS entities
ORDER BY post.posted_at DESC
RETURN event.event_key AS event_key,
       coalesce(event.type, mention.type, 'OTHER') AS type,
       coalesce(event.description, mention.description, post.content) AS description,
       coalesce(mention.status, event.status) AS status,
       mention.time_expression AS time_expression,
       entities,
       {
         platform: post.platform,
         platform_id: post.platform_id,
         content: post.content,
         url: post.url,
         posted_at: toString(post.posted_at),
         source_name: source.name
       } AS post
LIMIT $limit
"""


def normalize_name(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.casefold().split()))


def make_search_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", normalize_name(value))
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    ).replace("đ", "d")


class Neo4jRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = None

    def connect(self) -> None:
        if not self.settings.neo4j_password:
            raise RuntimeError("Thiếu biến môi trường NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )
        self.driver.verify_connectivity()

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def ping(self) -> bool:
        if self.driver is None:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def search_events(
        self,
        *,
        location: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if self.driver is None:
            raise RuntimeError("Neo4j chưa được kết nối")
        location_key = normalize_name(location) if location else None
        location_search_key = make_search_name(location) if location else None
        with self.driver.session(database=self.settings.neo4j_database) as session:
            return session.run(
                SEARCH_EVENTS_QUERY,
                location_key=location_key,
                location_search_key=location_search_key,
                limit=limit,
            ).data()
