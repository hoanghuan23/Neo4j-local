import json
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Any

from neo4j import GraphDatabase

from backend.event_candidate_search import EVENT_CANDIDATES_QUERY, candidate_search_parameters
from backend.config import Settings
from backend.event_search_queries import (
    SEARCH_EVENTS_QUERY,
    SEARCH_LEGACY_EVENTS_QUERY,
    SEARCH_RELATED_EVENTS_QUERY,
    SEARCH_RELATED_LEGACY_EVENTS_QUERY,
)
from backend.question_parser import normalize_entity_for_search


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
    normalized_value = normalize_entity_for_search(value)
    if not normalized_value:
        return []
    alternatives = (
        alternative.strip()
        for alternative in _ENTITY_ALTERNATIVE_RE.split(normalized_value)
    )
    return [
        {
            "key": normalize_name(alternative),
            "search_key": make_search_name(alternative),
        }
        for alternative in alternatives
        if alternative
    ]


def _matching_excerpt(value: str, term: str, context: int = 80) -> str:
    """Find a folded keyword while keeping offsets into the original text."""
    folded, offsets = [], []
    for index, character in enumerate(value):
        for part in unicodedata.normalize("NFD", character.casefold()):
            if unicodedata.category(part) != "Mn":
                folded.append(part.replace("đ", "d"))
                offsets.append(index)
    key = make_search_name(term)
    start = "".join(folded).find(key)
    if start < 0:
        return value[:2 * context]
    left = max(0, offsets[start] - context)
    right = min(len(value), offsets[start + len(key) - 1] + context + 1)
    return ("…" if left else "") + value[left:right] + ("…" if right < len(value) else "")


def _format_relation_reason(raw: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    reason = dict(raw)
    text = reason.pop("text", reason.get("excerpt") or "")
    reason["post"] = {
        "platform": post.get("platform"),
        "platform_id": post.get("platform_id"),
    }
    reason.setdefault("via_entity", None)
    reason.setdefault("relationship", None)
    reason["excerpt"] = None
    if reason["kind"] == "text_match":
        reason["excerpt"] = _matching_excerpt(text, reason["query_term"])
        field = "nội dung" if reason["evidence_field"] == "post.content" else "mô tả"
        reason["label"] = f"Khớp từ khóa trong {field}: {reason['query_term']}"
    else:
        reason["label"] = f"Liên quan qua: {reason['via_entity']['name']}"
    return reason


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


def _post_matches_date(
    post: dict[str, Any],
    posted_date: date,
    utc_offset_hours: int,
) -> bool:
    value = post.get("posted_at")
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False

    local_timezone = timezone(timedelta(hours=utc_offset_hours))
    if parsed.tzinfo is None:
        local_posted_at = parsed + timedelta(hours=utc_offset_hours)
    else:
        local_posted_at = parsed.astimezone(local_timezone)
    return local_posted_at.date() == posted_date


LOCATE_EVENT_QUERY = EVENT_CANDIDATES_QUERY + """
WITH event, matched_terms, match_coverage, candidate_score
ORDER BY candidate_score DESC, event.event_key
LIMIT $limit
CALL {
  WITH event
  MATCH (post:Post)-[:HAS_EVENT_MENTION]->
        (:EventMention)-[:EVIDENCE_FOR]->(event)
  RETURN post
  UNION
  WITH event
  MATCH (post:Post)-[:DESCRIBES]->(event)
  RETURN post
}
WITH DISTINCT event, post, matched_terms, match_coverage, candidate_score
OPTIONAL MATCH (source:Source)-[:PUBLISHED]->(post)
OPTIONAL MATCH (post)-[:MENTIONS]->
               (location:Entity {type: 'LOCATION'})
OPTIONAL MATCH path =
  (location)-[:PART_OF|IN_REGION*0..10]->
  (ancestor:Entity {type: 'LOCATION'})
WHERE all(node IN nodes(path) WHERE node.type = 'LOCATION')
  AND all(node IN nodes(path)
          WHERE single(other IN nodes(path) WHERE other = node))
RETURN DISTINCT
       event.event_key AS event_key,
       candidate_score, match_coverage, matched_terms,
       event.description AS event_description,
       post.platform AS post_platform,
       post.platform_id AS post_id,
       post.url AS post_url,
       post.content AS post_content,
       source.name AS source_name,
       toString(post.posted_at) AS posted_at,
       location.name AS mentioned_location,
       [node IN nodes(path) | node.name] AS location_chain,
       [edge IN relationships(path) | type(edge)] AS relations,
       length(path) AS depth
ORDER BY candidate_score DESC, event_key, depth DESC, post_platform, post_id;
"""


class Neo4jRepository:
    def locate_event(self, *, description: str, limit: int) -> list[dict[str, Any]]:
        if self.driver is None:
            raise RuntimeError("Neo4j chưa kết nối")
        parameters = candidate_search_parameters(description)
        if not parameters["candidate_terms"]:
            return []
        with self.driver.session(database=self.settings.neo4j_database, default_access_mode="READ") as session:
            return session.run(LOCATE_EVENT_QUERY, **parameters, limit=limit).data()

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

    def search_related_events(
        self, *, location: str | None, entity: str | None, hours: int,
        limit: int, posted_date: date | None = None,
        after: tuple[int, str, str] | None = None,
    ) -> list[dict[str, Any]]:
        return self.search_events(
            location=location, entity=entity, hours=hours, limit=limit,
            posted_date=posted_date, after=after, _related=True,
        )

    def search_events(
        self,
        *,
        location: str | None,
        entity: str | None,
        hours: int,
        limit: int,
        posted_date: date | None = None,
        after: tuple[int, str, str] | None = None,
        _related: bool = False,
    ) -> list[dict[str, Any]]:
        if self.driver is None:
            raise RuntimeError("Neo4j chưa được kết nối")
        location_key = normalize_name(location) if location else None
        location_search_key = make_search_name(location) if location else None
        terms = [{"field": "entity", **term} for term in make_entity_terms(entity)]
        if location_key:
            terms.insert(0, {"field": "location", "key": location_key,
                             "search_key": location_search_key})
        parameters = {
            "terms": terms,
            "fold_characters": [chr(code) for code in range(0x300, 0x370)
                                if unicodedata.category(chr(code)) == "Mn"],
            "hours": hours,
            "posted_date": posted_date.isoformat() if posted_date else None,
            "posted_at_utc_offset_hours": (
                self.settings.posted_at_utc_offset_hours
            ),
        }
        with self.driver.session(database=self.settings.neo4j_database) as session:
            current_results = session.run(
                SEARCH_EVENTS_QUERY, **parameters
            ).data()
            legacy_results = session.run(
                SEARCH_LEGACY_EVENTS_QUERY, **parameters
            ).data()
            if _related:
                # Exclude the complete direct set across both schemas/posts.
                direct_rows = current_results + legacy_results
                if posted_date is not None:
                    direct_rows = [row for row in direct_rows if _post_matches_date(
                        row["post"], posted_date,
                        self.settings.posted_at_utc_offset_hours,
                    )]
                direct_keys = {row["event_key"] for row in direct_rows}
                current_results = session.run(
                    SEARCH_RELATED_EVENTS_QUERY, **parameters
                ).data()
                legacy_results = session.run(
                    SEARCH_RELATED_LEGACY_EVENTS_QUERY, **parameters
                ).data()
                current_results = [row for row in current_results
                                   if row["event_key"] not in direct_keys]
                legacy_results = [row for row in legacy_results
                                  if row["event_key"] not in direct_keys]

        results_by_event_key: dict[str, dict[str, Any]] = {}
        posts_by_event_key: dict[
            str, dict[tuple[Any, ...], dict[str, Any]]
        ] = {}
        combined_results = current_results + legacy_results
        if posted_date is not None:
            combined_results = [
                result
                for result in combined_results
                if _post_matches_date(
                    result["post"],
                    posted_date,
                    self.settings.posted_at_utc_offset_hours,
                )
            ]

        reasons_by_event_key: dict[str, dict[str, dict[str, Any]]] = {}
        for result in combined_results:
            event_key = result["event_key"]
            post = result["post"]
            event_reasons = reasons_by_event_key.setdefault(event_key, {})
            for raw_reason in result.get("relation_reasons", []):
                reason = _format_relation_reason(raw_reason, post)
                event_reasons[json.dumps(reason, sort_keys=True, ensure_ascii=False)] = reason
            posts_by_event_key.setdefault(event_key, {})[
                _post_identity(post)
            ] = post
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

        for event_key, result in results_by_event_key.items():
            result["relation_reasons"] = [
                reasons_by_event_key[event_key][key]
                for key in sorted(reasons_by_event_key[event_key])
            ]
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

        sorted_results = sorted(
            results_by_event_key.values(),
            key=lambda result: (
                result.get("matched_entity_count", 0),
                result["post"].get("posted_at") or "",
                result["event_key"],
            ),
            reverse=True,
        )
        if after is not None:
            sorted_results = [
                result
                for result in sorted_results
                if (
                    result.get("matched_entity_count", 0),
                    result["post"].get("posted_at") or "",
                    result["event_key"],
                )
                < after
            ]
        return sorted_results[:limit]

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
