from knowledge_settings import (
    EVENT_RELATION_TYPES,
    KNOWLEDGE_CLASSIFIER_PROMPT_VERSION,
    KNOWLEDGE_PROMPT_VERSION,
    OLLAMA_LOG_PREVIEW_CHARS,
    OLLAMA_MODEL,
)
from knowledge_extraction import normalize_name, prepare_entity
from knowledge_validation import build_anonymous_participant_key


ENTITY_MERGE_QUERY = """
MATCH (p:Post {
    platform: $platform,
    platform_id: $post_id
})

// Prefer the canonical key, but also resolve an incoming spelling through an
// already-known alias. The toLower/trim fallback supports aliases written by
// older versions before aliases were normalized on write.
OPTIONAL MATCH (candidate:Entity {type: $entity_type})
WHERE candidate.normalized_name IN $identity_names
   OR any(alias IN coalesce(candidate.aliases, [])
          WHERE alias IN $identity_names
             OR toLower(trim(alias)) IN $identity_names)
WITH p, candidate,
     CASE
         WHEN candidate.normalized_name = $normalized_name THEN 0
         WHEN candidate.normalized_name IN $identity_names THEN 1
         ELSE 2
     END AS match_priority
ORDER BY match_priority
WITH p, head(collect(candidate)) AS existing

CALL (p, existing) {
    WITH p, existing
    WHERE existing IS NOT NULL
    RETURN existing AS e, false AS created

    UNION

    WITH p, existing
    WHERE existing IS NULL
    MERGE (new_entity:Entity {
        normalized_name: $normalized_name,
        type: $entity_type
    })
    RETURN new_entity AS e, true AS created
}

SET e.aliases = reduce(
        aliases = [],
        alias IN [existing_alias IN coalesce(e.aliases, []) |
                  toLower(trim(existing_alias))]
                 + $identity_names |
        CASE WHEN alias IN aliases THEN aliases ELSE aliases + alias END
    ),
    e.name = CASE
        WHEN created OR e.name IS NULL THEN $display_name
        WHEN $is_canonical AND e.normalized_name = $normalized_name
        THEN $display_name
        ELSE e.name
    END,
    e.search_name = CASE
        WHEN created OR e.search_name IS NULL THEN $search_name
        WHEN $is_canonical AND e.normalized_name = $normalized_name
        THEN $search_name
        ELSE e.search_name
    END,
    e.resolution_confidence = CASE
        WHEN e.resolution_confidence = 'HIGH' OR $confidence = 'HIGH'
        THEN 'HIGH'
        WHEN e.resolution_confidence = 'MEDIUM' OR $confidence = 'MEDIUM'
        THEN 'MEDIUM'
        ELSE $confidence
    END,
    e.needs_review = CASE
        WHEN e.resolution_confidence = 'HIGH' OR $confidence = 'HIGH'
        THEN false
        ELSE true
    END

MERGE (p)-[:MENTIONS]->(e)
RETURN e.normalized_name AS normalized_name, e.type AS entity_type
"""


def _merge_entity(tx, platform: str, post_id: str, entity: dict) -> dict | None:
    prepared = prepare_entity(entity)
    if prepared is None:
        return None
    result = tx.run(
        ENTITY_MERGE_QUERY,
        platform=platform,
        post_id=post_id,
        **prepared,
    )
    record = result.single()
    if record is not None:
        resolved_name = record.get("normalized_name")
        resolved_type = record.get("entity_type")
        if isinstance(resolved_name, str):
            prepared["normalized_name"] = resolved_name
        if isinstance(resolved_type, str):
            prepared["entity_type"] = resolved_type
    return prepared


def save_entities(session, platform: str, post_id: str, entities: list[dict]) -> int:
    """Legacy persistence wrapper kept for one release as a rollback path."""
    saved_count = 0
    for raw_entity in entities:
        if _merge_entity(session, platform, post_id, raw_entity) is not None:
            saved_count += 1

    session.run(
        """
        MATCH (p:Post {
            platform: $platform,
            platform_id: $post_id
        })
        SET p.entity_processed = true,
            p.entity_processed_at = datetime()
        """,
        platform=platform,
        post_id=post_id,
    ).consume()
    return saved_count


def upsert_entities(tx, platform: str, post_id: str, entities: list[dict]) -> dict:
    entity_lookup = {}
    for entity in entities:
        prepared = _merge_entity(tx, platform, post_id, entity)
        if prepared is not None:
            entity_lookup[entity["local_id"]] = prepared
    return entity_lookup


_PARTICIPANT_TITLE_PREFIXES = (
    "u.s. president ",
    "us president ",
    "president ",
    "tổng thống mỹ ",
    "tổng thống ",
    "mr. ",
    "mr ",
    "ông ",
)


def _participant_identity_names(normalized_text: str) -> list[str]:
    identity_names = [normalized_text]
    for prefix in _PARTICIPANT_TITLE_PREFIXES:
        if normalized_text.startswith(prefix):
            stripped_name = normalized_text[len(prefix):].strip()
            if stripped_name:
                identity_names.append(stripped_name)
            break
    return identity_names


def _resolve_unique_entity_by_name(tx, normalized_text: str) -> dict | None:
    """Resolve an anonymous name only when one existing Entity owns it."""
    record = tx.run(
        """
        MATCH (candidate:Entity)
        WHERE candidate.normalized_name IN $identity_names
           OR any(alias IN coalesce(candidate.aliases, [])
                  WHERE alias IN $identity_names
                     OR toLower(trim(alias)) IN $identity_names)
        WITH collect(DISTINCT candidate)[..2] AS candidates
        RETURN CASE WHEN size(candidates) = 1
                    THEN candidates[0].normalized_name END AS normalized_name,
               CASE WHEN size(candidates) = 1
                    THEN candidates[0].type END AS entity_type
        """,
        identity_names=_participant_identity_names(normalized_text),
    ).single()
    if record is None:
        return None
    normalized_name = record.get("normalized_name")
    entity_type = record.get("entity_type")
    if not isinstance(normalized_name, str) or not isinstance(entity_type, str):
        return None
    return {
        "normalized_name": normalized_name,
        "entity_type": entity_type,
    }


def _link_resolved_participant(
    tx,
    platform: str,
    post_id: str,
    mention_key: str,
    entity: dict,
    participant: dict,
) -> None:
    tx.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
        MATCH (mention:EventMention {mention_key: $mention_key})
        MATCH (entity:Entity {
            normalized_name: $normalized_name,
            type: $entity_type
        })
        MERGE (p)-[:MENTIONS]->(entity)
        MERGE (mention)-[relation:HAS_PARTICIPANT {
            role: $role
        }]->(entity)
        SET relation.confidence = $confidence
        """,
        platform=platform,
        post_id=post_id,
        mention_key=mention_key,
        normalized_name=entity["normalized_name"],
        entity_type=entity["entity_type"],
        role=participant["role"],
        confidence=participant["confidence"],
    ).consume()


def _delete_stale_events(
    tx,
    platform: str,
    post_id: str,
    mention_keys: list[str],
) -> None:
    tx.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
        OPTIONAL MATCH (p)-[:HAS_EVENT_MENTION]->(old:EventMention)
        WHERE NOT old.mention_key IN $mention_keys
        WITH p, collect(DISTINCT old) AS stale
        FOREACH (mention IN stale | DETACH DELETE mention)
        WITH p
        OPTIONAL MATCH (p)-[:DESCRIBES]->(legacy:Event)
        WHERE coalesce(legacy.schema_version, 1) = 1
          AND NOT (legacy)<-[:EVIDENCE_FOR]-(:EventMention)
        WITH p, collect(DISTINCT legacy) AS legacy_events
        FOREACH (event IN legacy_events | DETACH DELETE event)
        WITH p
        MATCH (p)-[description:DESCRIBES]->(:Event)
        DELETE description
        """,
        platform=platform,
        post_id=post_id,
        mention_keys=mention_keys,
    ).consume()


def upsert_events(
    tx,
    platform: str,
    post_id: str,
    events: list[dict],
    entity_lookup: dict,
) -> None:
    mention_keys = [event.get("mention_key", event["event_key"]) for event in events]
    _delete_stale_events(tx, platform, post_id, mention_keys)
    post_key = f"{platform}:{post_id}"

    for event in events:
        mention_key = event.get("mention_key", event["event_key"])
        tx.run(
            """
            MATCH (p:Post {platform: $platform, platform_id: $post_id})
            MERGE (mention:EventMention {mention_key: $mention_key})
            ON CREATE SET mention.created_at = datetime(),
                          mention.consolidation_status = 'PENDING'
            ON MATCH SET mention.consolidation_status = CASE
                WHEN coalesce(mention.type, '') <> $event_type
                  OR coalesce(mention.description, '') <> $description
                  OR coalesce(mention.evidence_text, '') <> $evidence_text
                  OR coalesce(mention.status, '') <> $status
                  OR coalesce(mention.time_expression, '')
                     <> coalesce($time_expression, '')
                THEN 'PENDING'
                ELSE mention.consolidation_status
            END
            SET mention.type = $event_type,
                mention.description = $description,
                mention.evidence_text = $evidence_text,
                mention.status = $status,
                mention.time_expression = $time_expression,
                mention.confidence = $confidence,
                mention.platform = $platform,
                mention.post_id = $post_id,
                mention.knowledge_model = $knowledge_model,
                mention.knowledge_prompt_version = $knowledge_prompt_version,
                mention.updated_at = datetime()
            WITH p, mention
            OPTIONAL MATCH (mention)-[:EVIDENCE_FOR]->(known:Event)
            WITH p, mention, head(collect(known)) AS known
            CALL (p, known) {
                WITH p, known
                WHERE known IS NOT NULL
                RETURN known AS event

                UNION

                WITH p, known
                WHERE known IS NULL
                MERGE (created:Event {event_key: $event_key})
                ON CREATE SET created.created_at = datetime(),
                              created.first_seen_at = p.posted_at,
                              created.schema_version = 2
                RETURN created AS event
            }
            SET event.type = coalesce(event.type, $event_type),
                event.description = coalesce(event.description, $description),
                event.status = coalesce(event.status, $status),
                event.last_seen_at = CASE
                    WHEN event.last_seen_at IS NULL OR p.posted_at > event.last_seen_at
                    THEN p.posted_at ELSE event.last_seen_at END,
                event.updated_at = datetime()
            REMOVE event.evidence_text, event.time_expression, event.confidence
            REMOVE event.start_year, event.end_year
            MERGE (p)-[:HAS_EVENT_MENTION]->(mention)
            MERGE (mention)-[:EVIDENCE_FOR]->(event)
            MERGE (p)-[:DESCRIBES]->(event)
            WITH mention
            OPTIONAL MATCH (mention)-[participant:HAS_PARTICIPANT]->()
            DELETE participant
            """,
            platform=platform,
            post_id=post_id,
            mention_key=mention_key,
            event_key=event["event_key"],
            event_type=event["type"],
            description=event["description"],
            evidence_text=event["evidence_text"],
            status=event["status"],
            time_expression=event["time_expression"],
            confidence=event["confidence"],
            knowledge_model=OLLAMA_MODEL,
            knowledge_prompt_version=KNOWLEDGE_PROMPT_VERSION,
        ).consume()

        for participant in event["participants"]:
            if participant["entity_id"]:
                entity = entity_lookup.get(participant["entity_id"])
                if entity is None:
                    continue
                tx.run(
                    """
                    MATCH (mention:EventMention {mention_key: $mention_key})
                    MATCH (entity:Entity {
                        normalized_name: $normalized_name,
                        type: $entity_type
                    })
                    MERGE (mention)-[relation:HAS_PARTICIPANT {
                        role: $role
                    }]->(entity)
                    SET relation.confidence = $confidence
                    """,
                    mention_key=mention_key,
                    normalized_name=entity["normalized_name"],
                    entity_type=entity["entity_type"],
                    role=participant["role"],
                    confidence=participant["confidence"],
                ).consume()
                continue

            normalized_text = normalize_name(participant["participant_text"])
            resolved_entity = _resolve_unique_entity_by_name(tx, normalized_text)
            if resolved_entity is not None:
                _link_resolved_participant(
                    tx,
                    platform,
                    post_id,
                    mention_key,
                    resolved_entity,
                    participant,
                )
                continue
            participant_key = build_anonymous_participant_key(
                platform,
                post_id,
                mention_key,
                participant,
            )
            tx.run(
                """
                MATCH (p:Post {platform: $platform, platform_id: $post_id})
                MATCH (mention:EventMention {mention_key: $mention_key})
                MERGE (anonymous:AnonymousParticipant {
                    participant_key: $participant_key
                })
                ON CREATE SET anonymous.created_at = datetime(),
                              anonymous.name = $normalized_text
                SET anonymous.post_key = CASE
                        WHEN $participant_scope = 'POST_LOCAL' THEN $post_key
                        ELSE null
                    END,
                    anonymous.participant_text = $participant_text,
                    anonymous.normalized_text = $normalized_text,
                    anonymous.participant_scope = $participant_scope,
                    anonymous.platform = $platform,
                    anonymous.updated_at = datetime()
                MERGE (p)-[:HAS_ANONYMOUS_PARTICIPANT]->(anonymous)
                MERGE (mention)-[relation:HAS_PARTICIPANT {
                    role: $role
                }]->(anonymous)
                SET relation.confidence = $confidence
                """,
                platform=platform,
                post_id=post_id,
                mention_key=mention_key,
                participant_key=participant_key,
                post_key=post_key,
                participant_text=participant["participant_text"],
                normalized_text=normalized_text,
                participant_scope=participant["participant_scope"],
                role=participant["role"],
                confidence=participant["confidence"],
            ).consume()

    tx.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
              -[relation:HAS_ANONYMOUS_PARTICIPANT]
              ->(anonymous:AnonymousParticipant)
        WHERE NOT EXISTS {
            MATCH (p)-[:HAS_EVENT_MENTION]->(:EventMention)
                     -[:HAS_PARTICIPANT]->(anonymous)
        }
        DELETE relation
        WITH DISTINCT anonymous
        WHERE NOT (anonymous)<-[:HAS_PARTICIPANT]-(:EventMention)
          AND NOT (anonymous)<-[:HAS_ANONYMOUS_PARTICIPANT]-(:Post)
        DELETE anonymous
        """,
        platform=platform,
        post_id=post_id,
    ).consume()

    tx.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
              -[:HAS_EVENT_MENTION]->(:EventMention)-[:EVIDENCE_FOR]->(event:Event)
        MERGE (p)-[:DESCRIBES]->(event)
        """,
        platform=platform,
        post_id=post_id,
    ).consume()

    tx.run(
        """
        MATCH (event:Event {schema_version: 2})
        WHERE NOT (event)<-[:EVIDENCE_FOR]-(:EventMention)
        DETACH DELETE event
        """
    ).consume()

    refresh_canonical_event_projections(tx)


def refresh_canonical_event_projections(tx) -> None:
    """Rebuild materialized Event fields from their source mentions."""
    tx.run(
        """
        MATCH (event:Event {schema_version: 2})
        OPTIONAL MATCH (event)-[old:HAS_PARTICIPANT]->()
        DELETE old
        """
    ).consume()
    tx.run(
        """
        MATCH (mention:EventMention)-[support:HAS_PARTICIPANT]->(participant)
        MATCH (mention)-[:EVIDENCE_FOR]->(event:Event {schema_version: 2})
        WITH event, participant, support.role AS role,
             max(support.confidence) AS confidence
        MERGE (event)-[relation:HAS_PARTICIPANT {role: role}]->(participant)
        SET relation.confidence = confidence
        """
    ).consume()
    refresh_canonical_event_relations(tx)


def refresh_canonical_event_relations(tx) -> None:
    """Rebuild canonical relations and suppress self-links after merges."""
    tx.run(
        """
        MATCH (source:Event {schema_version: 2})-[relation]->(target:Event)
        WHERE type(relation) IN $relation_types
        DELETE relation
        """,
        relation_types=sorted(EVENT_RELATION_TYPES),
    ).consume()
    for relation_type in sorted(EVENT_RELATION_TYPES):
        tx.run(
            f"""
            MATCH (source_mention:EventMention)-[support:{relation_type}]
                  ->(target_mention:EventMention)
            MATCH (source_mention)-[:EVIDENCE_FOR]->(source:Event)
            MATCH (target_mention)-[:EVIDENCE_FOR]->(target:Event)
            WHERE source <> target
            WITH source, target, collect(DISTINCT support.evidence_text) AS evidence
            MERGE (source)-[relation:{relation_type}]->(target)
            SET relation.evidence_texts = evidence,
                relation.knowledge_prompt_version = $knowledge_prompt_version
            """,
            knowledge_prompt_version=KNOWLEDGE_PROMPT_VERSION,
        ).consume()
    tx.run(
        """
        MATCH (event:Event {schema_version: 2})
        OPTIONAL MATCH (post:Post)-[:HAS_EVENT_MENTION]->(mention:EventMention)
                       -[:EVIDENCE_FOR]->(event)
        WITH event, count(DISTINCT mention) AS member_count,
             min(post.posted_at) AS first_seen_at,
             max(post.posted_at) AS last_seen_at
        SET event.member_count = member_count,
            event.first_seen_at = first_seen_at,
            event.last_seen_at = last_seen_at,
            event.updated_at = datetime()
        """
    ).consume()


def upsert_event_relations(
    tx,
    events: list[dict],
    relations: list[dict],
) -> None:
    local_to_key = {
        event["local_id"]: event.get("mention_key", event["event_key"])
        for event in events
    }
    mention_keys = list(local_to_key.values())
    tx.run(
        """
        MATCH (source:EventMention)-[relation]->(target:EventMention)
        WHERE source.mention_key IN $mention_keys
          AND type(relation) IN $relation_types
        DELETE relation
        """,
        mention_keys=mention_keys,
        relation_types=sorted(EVENT_RELATION_TYPES),
    ).consume()

    for relation in relations:
        relation_type = relation["type"]
        if relation_type not in EVENT_RELATION_TYPES:
            continue
        query = f"""
            MATCH (source:EventMention {{mention_key: $source_mention_key}})
            MATCH (target:EventMention {{mention_key: $target_mention_key}})
            MERGE (source)-[relation:{relation_type}]->(target)
            SET relation.evidence_text = $evidence_text,
                relation.knowledge_prompt_version = $knowledge_prompt_version
        """
        tx.run(
            query,
            source_mention_key=local_to_key[relation["source_event_id"]],
            target_mention_key=local_to_key[relation["target_event_id"]],
            evidence_text=relation["evidence_text"],
            knowledge_prompt_version=KNOWLEDGE_PROMPT_VERSION,
        ).consume()

    refresh_canonical_event_relations(tx)


def save_knowledge_tx(
    tx,
    platform: str,
    post_id: str,
    knowledge: dict,
    classification: dict | None = None,
    classifier_decision: str | None = None,
) -> dict:
    post_exists = tx.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
        RETURN count(p) AS post_count
        """,
        platform=platform,
        post_id=post_id,
    ).single()["post_count"]
    if post_exists != 1:
        raise ValueError(f"Không tìm thấy Post {platform}:{post_id}")

    if classifier_decision == "SKIPPED":
        tx.run(
            """
            MATCH (p:Post {platform: $platform, platform_id: $post_id})
            OPTIONAL MATCH (p)-[mention:MENTIONS]->(:Entity)
            DELETE mention
            """,
            platform=platform,
            post_id=post_id,
        ).consume()

    generic_entity_keys = knowledge.get("generic_entity_keys", [])
    if generic_entity_keys:
        tx.run(
            """
            MATCH (p:Post {platform: $platform, platform_id: $post_id})
            UNWIND $generic_entity_keys AS item
            MATCH (p)-[mention:MENTIONS]->(entity:Entity {
                normalized_name: item.normalized_name,
                type: item.type
            })
            DELETE mention
            """,
            platform=platform,
            post_id=post_id,
            generic_entity_keys=generic_entity_keys,
        ).consume()

    entity_lookup = upsert_entities(tx, platform, post_id, knowledge["entities"])
    upsert_events(tx, platform, post_id, knowledge["events"], entity_lookup)
    upsert_event_relations(tx, knowledge["events"], knowledge["event_relations"])
    tx.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
        SET p.entity_processed = true,
            p.entity_processed_at = datetime(),
            p.knowledge_processed = true,
            p.knowledge_processed_at = datetime(),
            p.knowledge_model = $knowledge_model,
            p.knowledge_prompt_version = $knowledge_prompt_version,
            p.knowledge_error = null,
            p.knowledge_retry_count = 0
        FOREACH (_ IN CASE WHEN $classifier_decision IS NULL THEN [] ELSE [1] END |
            SET p.knowledge_classifier_should_deep_analyze = $classifier_should_deep,
                p.knowledge_classifier_reason_code = $classifier_reason_code,
                p.knowledge_classifier_decision = $classifier_decision,
                p.knowledge_classifier_model = $knowledge_model,
                p.knowledge_classifier_prompt_version = $classifier_prompt_version,
                p.knowledge_classified_at = datetime()
            REMOVE p.knowledge_classifier_has_entity,
                   p.knowledge_classifier_has_event
        )
        """,
        platform=platform,
        post_id=post_id,
        knowledge_model=OLLAMA_MODEL,
        knowledge_prompt_version=KNOWLEDGE_PROMPT_VERSION,
        classifier_should_deep=(
            classification.get("should_deep_analyze") if classification else None
        ),
        classifier_reason_code=(
            classification.get("reason_code") if classification else None
        ),
        classifier_decision=classifier_decision,
        classifier_prompt_version=KNOWLEDGE_CLASSIFIER_PROMPT_VERSION,
    ).consume()
    return {
        "entities": len(entity_lookup),
        "events": len(knowledge["events"]),
        "event_relations": len(knowledge["event_relations"]),
    }


def mark_knowledge_failure(tx, platform: str, post_id: str, error: str) -> None:
    tx.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
        WITH p, CASE
            WHEN coalesce(p.knowledge_prompt_version, '')
                 <> $knowledge_prompt_version
            THEN 1
            ELSE coalesce(p.knowledge_retry_count, 0) + 1
        END AS next_retry_count
        SET p.knowledge_processed = false,
            p.knowledge_model = $knowledge_model,
            p.knowledge_prompt_version = $knowledge_prompt_version,
            p.knowledge_error = $knowledge_error,
            p.knowledge_retry_count = next_retry_count
        """,
        platform=platform,
        post_id=post_id,
        knowledge_model=OLLAMA_MODEL,
        knowledge_prompt_version=KNOWLEDGE_PROMPT_VERSION,
        knowledge_error=error[:OLLAMA_LOG_PREVIEW_CHARS],
    ).consume()


def create_entity_schema(session) -> None:
    session.run("""
        CREATE CONSTRAINT entity_identity_unique IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE (e.normalized_name, e.type) IS UNIQUE
        """).consume()
    session.run("""
        CREATE TEXT INDEX entity_search_name IF NOT EXISTS
        FOR (e:Entity) ON (e.search_name)
        """).consume()


def create_knowledge_schema(session) -> None:
    create_entity_schema(session)
    session.run("""
        CREATE CONSTRAINT event_key_unique IF NOT EXISTS
        FOR (event:Event)
        REQUIRE event.event_key IS UNIQUE
        """).consume()
    session.run("""
        CREATE CONSTRAINT anonymous_participant_key_unique IF NOT EXISTS
        FOR (participant:AnonymousParticipant)
        REQUIRE participant.participant_key IS UNIQUE
        """).consume()
    session.run("""
        CREATE CONSTRAINT event_mention_key_unique IF NOT EXISTS
        FOR (mention:EventMention)
        REQUIRE mention.mention_key IS UNIQUE
        """).consume()
    session.run("""
        CREATE INDEX event_mention_consolidation_status IF NOT EXISTS
        FOR (mention:EventMention) ON (mention.consolidation_status)
        """).consume()
