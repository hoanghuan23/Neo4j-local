import unicodedata
from typing import Any

from neo4j import GraphDatabase

from backend.config import Settings


SEARCH_EVENTS_QUERY = """
MATCH (post:Post)-[:HAS_EVENT_MENTION]->(mention:EventMention)
      -[:EVIDENCE_FOR]->(event:Event)
WHERE post.posted_at IS NOT NULL
  AND post.posted_at >= localdatetime() - duration({hours: $hours})
  AND (
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
  AND (
    $entity_key IS NULL
    OR EXISTS {
      MATCH (mention)-[:HAS_PARTICIPANT]->(related_entity:Entity)
      WHERE related_entity.normalized_name CONTAINS $entity_key
         OR $entity_key IN coalesce(related_entity.aliases, [])
         OR related_entity.search_name CONTAINS $entity_search_key
    }
    OR EXISTS {
      MATCH (post)-[:MENTIONS]->(related_entity:Entity)
      WHERE related_entity.normalized_name CONTAINS $entity_key
         OR $entity_key IN coalesce(related_entity.aliases, [])
         OR related_entity.search_name CONTAINS $entity_search_key
    }
    OR toLower(coalesce(post.content, '')) CONTAINS $entity_key
    OR toLower(coalesce(mention.description, '')) CONTAINS $entity_key
    OR toLower(coalesce(event.description, '')) CONTAINS $entity_key
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
"""


SEARCH_LEGACY_EVENTS_QUERY = """
MATCH (post:Post)-[:DESCRIBES]->(event:Event)
WHERE post.posted_at IS NOT NULL
  AND post.posted_at >= localdatetime() - duration({hours: $hours})
  AND (
    $location_key IS NULL
    OR EXISTS {
      MATCH (event)-[:HAS_PARTICIPANT]->(location:Entity)
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
    OR toLower(coalesce(event.description, '')) CONTAINS $location_key
  )
  AND (
    $entity_key IS NULL
    OR EXISTS {
      MATCH (event)-[:HAS_PARTICIPANT]->(related_entity:Entity)
      WHERE related_entity.normalized_name CONTAINS $entity_key
         OR $entity_key IN coalesce(related_entity.aliases, [])
         OR related_entity.search_name CONTAINS $entity_search_key
    }
    OR EXISTS {
      MATCH (post)-[:MENTIONS]->(related_entity:Entity)
      WHERE related_entity.normalized_name CONTAINS $entity_key
         OR $entity_key IN coalesce(related_entity.aliases, [])
         OR related_entity.search_name CONTAINS $entity_search_key
    }
    OR toLower(coalesce(post.content, '')) CONTAINS $entity_key
    OR toLower(coalesce(event.description, '')) CONTAINS $entity_key
  )
OPTIONAL MATCH (source:Source)-[:PUBLISHED]->(post)
OPTIONAL MATCH (event)-[participation:HAS_PARTICIPANT]->(entity:Entity)
WITH post, event, source,
     collect(DISTINCT CASE WHEN entity IS NULL THEN NULL ELSE {
       name: coalesce(entity.name, entity.normalized_name),
       type: entity.type,
       role: participation.role
     } END) AS entities
ORDER BY post.posted_at DESC
RETURN event.event_key AS event_key,
       coalesce(event.type, 'OTHER') AS type,
       coalesce(event.description, post.content) AS description,
       event.status AS status,
       event.time_expression AS time_expression,
       entities,
       {
         platform: post.platform,
         platform_id: post.platform_id,
         content: post.content,
         url: post.url,
         posted_at: toString(post.posted_at),
         source_name: source.name
       } AS post
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
        entity: str | None,
        hours: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        if self.driver is None:
            raise RuntimeError("Neo4j chưa được kết nối")
        location_key = normalize_name(location) if location else None
        location_search_key = make_search_name(location) if location else None
        entity_key = normalize_name(entity) if entity else None
        entity_search_key = make_search_name(entity) if entity else None
        parameters = {
            "location_key": location_key,
            "location_search_key": location_search_key,
            "entity_key": entity_key,
            "entity_search_key": entity_search_key,
            "hours": hours,
        }
        with self.driver.session(database=self.settings.neo4j_database) as session:
            current_results = session.run(
                SEARCH_EVENTS_QUERY, **parameters
            ).data()
            legacy_results = session.run(
                SEARCH_LEGACY_EVENTS_QUERY, **parameters
            ).data()

        results_by_event_key: dict[str, dict[str, Any]] = {}
        for result in current_results + legacy_results:
            results_by_event_key.setdefault(result["event_key"], result)

        return sorted(
            results_by_event_key.values(),
            key=lambda result: result["post"].get("posted_at") or "",
            reverse=True,
        )[:limit]
