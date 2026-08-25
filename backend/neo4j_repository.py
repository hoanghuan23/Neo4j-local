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
     (
       $location_key IS NULL
       OR any(related_entity IN event_entities WHERE
         related_entity.type = 'LOCATION' AND (
           coalesce(related_entity.normalized_name, '') CONTAINS $location_key
           OR $location_key IN coalesce(related_entity.aliases, [])
           OR coalesce(related_entity.search_name, '')
                CONTAINS $location_search_key
         )
       )
       OR toLower(coalesce(mention.description, '')) CONTAINS $location_key
       OR toLower(coalesce(event.description, '')) CONTAINS $location_key
       OR (sibling_event_count = 1 AND (
         any(related_entity IN post_entities WHERE
           related_entity.type = 'LOCATION' AND (
             coalesce(related_entity.normalized_name, '')
                  CONTAINS $location_key
             OR $location_key IN coalesce(related_entity.aliases, [])
             OR coalesce(related_entity.search_name, '')
                  CONTAINS $location_search_key
           )
         )
         OR toLower(coalesce(post.content, '')) CONTAINS $location_key
       ))
     ) AS location_matches,
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
     [term IN $topic_terms WHERE
      any(related_entity IN event_entities WHERE
        coalesce(related_entity.normalized_name, '') CONTAINS term.key
        OR term.key IN coalesce(related_entity.aliases, [])
        OR coalesce(related_entity.search_name, '') CONTAINS term.search_key
      )
      OR toLower(coalesce(mention.description, '')) CONTAINS term.key
      OR toLower(coalesce(event.description, '')) CONTAINS term.key
     ] AS event_matched_topic_terms,
     [term IN $topic_terms WHERE
      any(related_entity IN post_entities WHERE
        coalesce(related_entity.normalized_name, '') CONTAINS term.key
        OR term.key IN coalesce(related_entity.aliases, [])
        OR coalesce(related_entity.search_name, '') CONTAINS term.search_key
      )
      OR toLower(coalesce(post.content, '')) CONTAINS term.key
     ] AS post_matched_topic_terms,
     sibling_event_count
WITH post, mention, event,
     location_matches,
     CASE
       WHEN size(event_matched_terms) > 0 THEN size(event_matched_terms)
       WHEN size(post_matched_terms) > 0 AND sibling_event_count = 1 THEN 1
       ELSE 0
     END AS matched_entity_count,
     CASE
       WHEN size(event_matched_topic_terms) > 0
         THEN size(event_matched_topic_terms)
       WHEN size(post_matched_topic_terms) > 0 AND sibling_event_count = 1
         THEN size(post_matched_topic_terms)
       ELSE 0
     END AS matched_topic_count
WHERE (
  (
    ($location_key IS NOT NULL
      OR size($entity_terms) > 0
      OR size($topic_terms) = 0)
    AND location_matches
    AND (size($entity_terms) = 0 OR matched_entity_count > 0)
  )
  OR matched_topic_count > 0
)
OPTIONAL MATCH (source:Source)-[:PUBLISHED]->(post)
OPTIONAL MATCH (mention)-[participation:HAS_PARTICIPANT]->(entity:Entity)
WITH post, mention, event, source,
     matched_entity_count, matched_topic_count,
     collect(DISTINCT CASE WHEN entity IS NULL THEN NULL ELSE {
       name: coalesce(entity.name, entity.normalized_name),
       type: entity.type,
       role: participation.role
     } END) AS entities
ORDER BY matched_entity_count DESC, matched_topic_count DESC,
         post.posted_at DESC
RETURN event.event_key AS event_key,
       coalesce(event.type, mention.type, 'OTHER') AS type,
       coalesce(event.description, mention.description, post.content) AS description,
       coalesce(mention.status, event.status) AS status,
       mention.time_expression AS time_expression,
       matched_entity_count,
       matched_topic_count,
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
OPTIONAL MATCH (event)-[:HAS_PARTICIPANT]->(event_entity:Entity)
OPTIONAL MATCH (post)-[:MENTIONS]->(post_entity:Entity)
OPTIONAL MATCH (post)-[:DESCRIBES]->(sibling_event:Event)
WITH post, event,
     collect(DISTINCT event_entity) AS event_entities,
     collect(DISTINCT post_entity) AS post_entities,
     count(DISTINCT sibling_event) AS sibling_event_count
WITH post, event,
     (
       $location_key IS NULL
       OR any(related_entity IN event_entities WHERE
         related_entity.type = 'LOCATION' AND (
           coalesce(related_entity.normalized_name, '') CONTAINS $location_key
           OR $location_key IN coalesce(related_entity.aliases, [])
           OR coalesce(related_entity.search_name, '')
                CONTAINS $location_search_key
         )
       )
       OR toLower(coalesce(event.description, '')) CONTAINS $location_key
       OR (sibling_event_count = 1 AND (
         any(related_entity IN post_entities WHERE
           related_entity.type = 'LOCATION' AND (
             coalesce(related_entity.normalized_name, '')
                  CONTAINS $location_key
             OR $location_key IN coalesce(related_entity.aliases, [])
             OR coalesce(related_entity.search_name, '')
                  CONTAINS $location_search_key
           )
         )
         OR toLower(coalesce(post.content, '')) CONTAINS $location_key
       ))
     ) AS location_matches,
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
     [term IN $topic_terms WHERE
      any(related_entity IN event_entities WHERE
        coalesce(related_entity.normalized_name, '') CONTAINS term.key
        OR term.key IN coalesce(related_entity.aliases, [])
        OR coalesce(related_entity.search_name, '') CONTAINS term.search_key
      )
      OR toLower(coalesce(event.description, '')) CONTAINS term.key
     ] AS event_matched_topic_terms,
     [term IN $topic_terms WHERE
      any(related_entity IN post_entities WHERE
        coalesce(related_entity.normalized_name, '') CONTAINS term.key
        OR term.key IN coalesce(related_entity.aliases, [])
        OR coalesce(related_entity.search_name, '') CONTAINS term.search_key
      )
      OR toLower(coalesce(post.content, '')) CONTAINS term.key
     ] AS post_matched_topic_terms,
     sibling_event_count
WITH post, event,
     location_matches,
     CASE
       WHEN size(event_matched_terms) > 0 THEN size(event_matched_terms)
       WHEN size(post_matched_terms) > 0 AND sibling_event_count = 1 THEN 1
       ELSE 0
     END AS matched_entity_count,
     CASE
       WHEN size(event_matched_topic_terms) > 0
         THEN size(event_matched_topic_terms)
       WHEN size(post_matched_topic_terms) > 0 AND sibling_event_count = 1
         THEN size(post_matched_topic_terms)
       ELSE 0
     END AS matched_topic_count
WHERE (
  (
    ($location_key IS NOT NULL
      OR size($entity_terms) > 0
      OR size($topic_terms) = 0)
    AND location_matches
    AND (size($entity_terms) = 0 OR matched_entity_count > 0)
  )
  OR matched_topic_count > 0
)
OPTIONAL MATCH (source:Source)-[:PUBLISHED]->(post)
OPTIONAL MATCH (event)-[participation:HAS_PARTICIPANT]->(entity:Entity)
WITH post, event, source,
     matched_entity_count, matched_topic_count,
     collect(DISTINCT CASE WHEN entity IS NULL THEN NULL ELSE {
       name: coalesce(entity.name, entity.normalized_name),
       type: entity.type,
       role: participation.role
     } END) AS entities
ORDER BY matched_entity_count DESC, matched_topic_count DESC,
         post.posted_at DESC
RETURN event.event_key AS event_key,
       coalesce(event.type, 'OTHER') AS type,
       coalesce(event.description, post.content) AS description,
       event.status AS status,
       event.time_expression AS time_expression,
       matched_entity_count,
       matched_topic_count,
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


def make_topic_terms(
    topic: str | None,
    search_terms: list[str] | None,
) -> list[dict[str, str]]:
    """Normalize and deduplicate Gemini's topical query expansion."""
    values = [topic, *(search_terms or [])]
    terms: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if not value or not value.strip():
            continue
        key = normalize_name(value)
        if key in seen:
            continue
        seen.add(key)
        terms.append({"key": key, "search_key": make_search_name(value)})
    return terms


def _post_identity(post: dict[str, Any]) -> tuple[Any, ...]:
    platform = post.get("platform")
    platform_id = post.get("platform_id")
    if platform and platform_id:
        return ("platform_id", platform, platform_id)

    url = post.get("url")
    if url:
        return ("url", url)

    return (
        "fields",
        platform,
        platform_id,
        post.get("content"),
        post.get("posted_at"),
        post.get("source_name"),
    )


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
        topic: str | None = None,
        search_terms: list[str] | None = None,
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
            "topic_terms": make_topic_terms(topic, search_terms),
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
        posts_by_event_key: dict[
            str, dict[tuple[Any, ...], dict[str, Any]]
        ] = {}
        for result in current_results + legacy_results:
            event_key = result["event_key"]
            post = result["post"]
            posts_by_event_key.setdefault(event_key, {})[
                _post_identity(post)
            ] = post
            existing = results_by_event_key.get(event_key)
            result_rank = (
                result.get("matched_entity_count", 0),
                result.get("matched_topic_count", 0),
                result["post"].get("posted_at") or "",
            )
            existing_rank = (
                existing.get("matched_entity_count", 0),
                existing.get("matched_topic_count", 0),
                existing["post"].get("posted_at") or "",
            ) if existing is not None else None
            if existing_rank is None or result_rank > existing_rank:
                results_by_event_key[event_key] = result

        for event_key, result in results_by_event_key.items():
            primary_post = result["post"]
            primary_identity = _post_identity(primary_post)
            other_posts = [
                post
                for identity, post in posts_by_event_key[event_key].items()
                if identity != primary_identity
            ]
            other_posts.sort(
                key=lambda post: post.get("posted_at") or "",
                reverse=True,
            )
            posts = [primary_post, *other_posts]
            result["sources"] = [
                {
                    "source": post.get("source_name") or post.get("platform"),
                    "posted_at": post.get("posted_at"),
                    "url": post.get("url"),
                }
                for post in posts
            ]

        return sorted(
            results_by_event_key.values(),
            key=lambda result: (
                result.get("matched_entity_count", 0),
                result.get("matched_topic_count", 0),
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
