import re
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
OPTIONAL MATCH (mention)-[:HAS_PARTICIPANT]->(mention_entity:Entity)
OPTIONAL MATCH (event)-[:HAS_PARTICIPANT]->(event_entity:Entity)
OPTIONAL MATCH (post)-[:MENTIONS]->(post_entity:Entity)
OPTIONAL MATCH (post)-[:HAS_EVENT_MENTION]->(sibling_mention:EventMention)
WITH post, mention, event,
     collect(DISTINCT mention_entity)
       + collect(DISTINCT event_entity) AS event_entities,
     collect(DISTINCT post_entity) AS post_entities,
     count(DISTINCT sibling_mention) AS sibling_event_count
WITH post, mention, event,
     [term IN $entity_terms WHERE
      any(related_entity IN event_entities WHERE
        coalesce(related_entity.normalized_name, '') CONTAINS term.key
        OR term.key IN coalesce(related_entity.aliases, [])
        OR (size(split(term.key, ' ')) > 1 AND
            coalesce(related_entity.search_name, '') CONTAINS term.search_key)
      )
      OR (size(split(term.key, ' ')) > 1 AND (
        toLower(coalesce(mention.description, '')) CONTAINS term.key
        OR toLower(coalesce(event.description, '')) CONTAINS term.key
      ))
     ] AS event_matched_terms,
     [term IN $entity_terms WHERE
      any(related_entity IN post_entities WHERE
        coalesce(related_entity.normalized_name, '') CONTAINS term.key
        OR term.key IN coalesce(related_entity.aliases, [])
        OR (size(split(term.key, ' ')) > 1 AND
            coalesce(related_entity.search_name, '') CONTAINS term.search_key)
      )
      OR (size(split(term.key, ' ')) > 1
          AND toLower(coalesce(post.content, '')) CONTAINS term.key)
     ] AS post_matched_terms,
     sibling_event_count
WITH post, mention, event,
     CASE
       WHEN size(event_matched_terms) > 0 THEN size(event_matched_terms)
       WHEN size(post_matched_terms) > 0 AND sibling_event_count = 1 THEN 1
       ELSE 0
     END AS matched_entity_count
WHERE size($entity_terms) = 0 OR matched_entity_count > 0
OPTIONAL MATCH (source:Source)-[:PUBLISHED]->(post)
OPTIONAL MATCH (mention)-[participation:HAS_PARTICIPANT]->(entity:Entity)
WITH post, mention, event, source,
     matched_entity_count,
     collect(DISTINCT CASE WHEN entity IS NULL THEN NULL ELSE {
       name: coalesce(entity.name, entity.normalized_name),
       type: entity.type,
       role: participation.role
     } END) AS entities
ORDER BY matched_entity_count DESC, post.posted_at DESC
RETURN event.event_key AS event_key,
       coalesce(event.type, mention.type, 'OTHER') AS type,
       coalesce(event.description, mention.description, post.content) AS description,
       coalesce(mention.status, event.status) AS status,
       mention.time_expression AS time_expression,
       matched_entity_count,
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
OPTIONAL MATCH (event)-[:HAS_PARTICIPANT]->(event_entity:Entity)
OPTIONAL MATCH (post)-[:MENTIONS]->(post_entity:Entity)
OPTIONAL MATCH (post)-[:DESCRIBES]->(sibling_event:Event)
WITH post, event,
     collect(DISTINCT event_entity) AS event_entities,
     collect(DISTINCT post_entity) AS post_entities,
     count(DISTINCT sibling_event) AS sibling_event_count
WITH post, event,
     [term IN $entity_terms WHERE
      any(related_entity IN event_entities WHERE
        coalesce(related_entity.normalized_name, '') CONTAINS term.key
        OR term.key IN coalesce(related_entity.aliases, [])
        OR (size(split(term.key, ' ')) > 1 AND
            coalesce(related_entity.search_name, '') CONTAINS term.search_key)
      )
      OR (size(split(term.key, ' ')) > 1
          AND toLower(coalesce(event.description, '')) CONTAINS term.key)
     ] AS event_matched_terms,
     [term IN $entity_terms WHERE
      any(related_entity IN post_entities WHERE
        coalesce(related_entity.normalized_name, '') CONTAINS term.key
        OR term.key IN coalesce(related_entity.aliases, [])
        OR (size(split(term.key, ' ')) > 1 AND
            coalesce(related_entity.search_name, '') CONTAINS term.search_key)
      )
      OR (size(split(term.key, ' ')) > 1
          AND toLower(coalesce(post.content, '')) CONTAINS term.key)
     ] AS post_matched_terms,
     sibling_event_count
WITH post, event,
     CASE
       WHEN size(event_matched_terms) > 0 THEN size(event_matched_terms)
       WHEN size(post_matched_terms) > 0 AND sibling_event_count = 1 THEN 1
       ELSE 0
     END AS matched_entity_count
WHERE size($entity_terms) = 0 OR matched_entity_count > 0
OPTIONAL MATCH (source:Source)-[:PUBLISHED]->(post)
OPTIONAL MATCH (event)-[participation:HAS_PARTICIPANT]->(entity:Entity)
WITH post, event, source,
     matched_entity_count,
     collect(DISTINCT CASE WHEN entity IS NULL THEN NULL ELSE {
       name: coalesce(entity.name, entity.normalized_name),
       type: entity.type,
       role: participation.role
     } END) AS entities
ORDER BY matched_entity_count DESC, post.posted_at DESC
RETURN event.event_key AS event_key,
       coalesce(event.type, 'OTHER') AS type,
       coalesce(event.description, post.content) AS description,
       event.status AS status,
       event.time_expression AS time_expression,
       matched_entity_count,
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


SEARCH_RELATED_ENTITIES_QUERY = """
MATCH (post:Post)-[:MENTIONS]->(subject:Entity)
WHERE coalesce(subject.normalized_name, toLower(subject.name), '')
        CONTAINS $subject_key
   OR coalesce(subject.search_name, '') CONTAINS $subject_search_key
   OR $subject_key IN coalesce(subject.aliases, [])
WITH collect(DISTINCT subject) AS candidates
WITH CASE
       WHEN any(candidate IN candidates WHERE
         coalesce(candidate.normalized_name, toLower(candidate.name), '')
           = $subject_key
         OR coalesce(candidate.search_name, '') = $subject_search_key
         OR $subject_key IN coalesce(candidate.aliases, [])
       )
       THEN [candidate IN candidates WHERE
         coalesce(candidate.normalized_name, toLower(candidate.name), '')
           = $subject_key
         OR coalesce(candidate.search_name, '') = $subject_search_key
         OR $subject_key IN coalesce(candidate.aliases, [])
       ]
       ELSE candidates
     END AS selected_subjects
UNWIND selected_subjects AS subject
MATCH (post:Post)-[:MENTIONS]->(subject)
MATCH (post)-[:MENTIONS]->(related:Entity)
WHERE NOT related IN selected_subjects
RETURN coalesce(related.type, 'UNKNOWN') AS entity_type,
       coalesce(related.name, related.normalized_name, 'Không rõ')
         AS entity_name,
       count(DISTINCT post) AS post_count
ORDER BY post_count DESC, entity_name
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


_ENTITY_ALTERNATIVE_RE = re.compile(
    r"\s+(?:và|hoặc|hay)\s+|\s*[,;]\s*",
    re.IGNORECASE,
)


def make_entity_terms(value: str | None) -> list[dict[str, str]]:
    if not value:
        return []
    alternatives = (
        alternative.strip()
        for alternative in _ENTITY_ALTERNATIVE_RE.split(value)
    )
    return [
        {
            "key": normalize_name(alternative),
            "search_key": make_search_name(alternative),
        }
        for alternative in alternatives
        if alternative
    ]


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
        parameters = {
            "location_key": location_key,
            "location_search_key": location_search_key,
            "entity_terms": make_entity_terms(entity),
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
            event_key = result["event_key"]
            existing = results_by_event_key.get(event_key)
            result_rank = (
                result.get("matched_entity_count", 0),
                result["post"].get("posted_at") or "",
            )
            existing_rank = (
                existing.get("matched_entity_count", 0),
                existing["post"].get("posted_at") or "",
            ) if existing is not None else None
            if existing_rank is None or result_rank > existing_rank:
                results_by_event_key[event_key] = result

        return sorted(
            results_by_event_key.values(),
            key=lambda result: (
                result.get("matched_entity_count", 0),
                result["post"].get("posted_at") or "",
            ),
            reverse=True,
        )[:limit]

    def search_related_entities(
        self,
        *,
        subject: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if self.driver is None:
            raise RuntimeError("Neo4j chưa được kết nối")
        parameters = {
            "subject_key": normalize_name(subject),
            "subject_search_key": make_search_name(subject),
            "limit": limit,
        }
        with self.driver.session(database=self.settings.neo4j_database) as session:
            return session.run(
                SEARCH_RELATED_ENTITIES_QUERY,
                **parameters,
            ).data()
