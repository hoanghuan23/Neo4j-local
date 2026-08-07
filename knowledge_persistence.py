from knowledge_settings import (
    EVENT_RELATION_TYPES,
    KNOWLEDGE_PROMPT_VERSION,
    OLLAMA_LOG_PREVIEW_CHARS,
    OLLAMA_MODEL,
)
from knowledge_extraction import prepare_entity
from knowledge_validation import build_anonymous_participant_key


ENTITY_MERGE_QUERY = """
MATCH (p:Post {
    platform: $platform,
    platform_id: $post_id
})

MERGE (e:Entity {
    normalized_name: $normalized_name,
    type: $entity_type
})
ON CREATE SET
    e.name = $display_name,
    e.search_name = $search_name,
    e.aliases = [$name],
    e.resolution_confidence = $confidence,
    e.needs_review = NOT $is_canonical
ON MATCH SET
    e.aliases = CASE
        WHEN e.aliases IS NULL THEN [e.name, $name]
        WHEN NOT $name IN e.aliases THEN e.aliases + $name
        ELSE e.aliases
    END,
    e.name = CASE
        WHEN $is_canonical THEN $display_name
        ELSE e.name
    END,
    e.search_name = CASE
        WHEN $is_canonical THEN $search_name
        ELSE coalesce(e.search_name, $search_name)
    END,
    e.resolution_confidence = CASE
        WHEN e.resolution_confidence = 'HIGH' OR $confidence = 'HIGH'
        THEN 'HIGH'
        WHEN e.resolution_confidence = 'MEDIUM' OR $confidence = 'MEDIUM'
        THEN 'MEDIUM'
        ELSE 'LOW'
    END,
    e.needs_review = CASE
        WHEN e.resolution_confidence = 'HIGH' OR $confidence = 'HIGH'
        THEN false
        ELSE true
    END

MERGE (p)-[:MENTIONS]->(e)
"""


def _merge_entity(tx, platform: str, post_id: str, entity: dict) -> dict | None:
    prepared = prepare_entity(entity)
    if prepared is None:
        return None
    tx.run(
        ENTITY_MERGE_QUERY,
        platform=platform,
        post_id=post_id,
        **prepared,
    ).consume()
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


def _delete_stale_events(
    tx,
    platform: str,
    post_id: str,
    event_keys: list[str],
) -> None:
    tx.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
        MATCH (p)-[:DESCRIBES]->(old:Event)
        WHERE NOT old.event_key IN $event_keys
        DETACH DELETE old
        """,
        platform=platform,
        post_id=post_id,
        event_keys=event_keys,
    ).consume()


def upsert_events(
    tx,
    platform: str,
    post_id: str,
    events: list[dict],
    entity_lookup: dict,
) -> None:
    event_keys = [event["event_key"] for event in events]
    _delete_stale_events(tx, platform, post_id, event_keys)
    post_key = f"{platform}:{post_id}"

    for event in events:
        tx.run(
            """
            MATCH (p:Post {platform: $platform, platform_id: $post_id})
            MERGE (event:Event {event_key: $event_key})
            ON CREATE SET event.created_at = datetime()
            SET event.type = $event_type,
                event.description = $description,
                event.evidence_text = $evidence_text,
                event.status = $status,
                event.time_expression = $time_expression,
                event.confidence = $confidence,
                event.knowledge_model = $knowledge_model,
                event.knowledge_prompt_version = $knowledge_prompt_version,
                event.updated_at = datetime()
            REMOVE event.start_year, event.end_year
            MERGE (p)-[:DESCRIBES]->(event)
            WITH event
            OPTIONAL MATCH (event)-[participant:HAS_PARTICIPANT]->()
            DELETE participant
            """,
            platform=platform,
            post_id=post_id,
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
                    MATCH (event:Event {event_key: $event_key})
                    MATCH (entity:Entity {
                        normalized_name: $normalized_name,
                        type: $entity_type
                    })
                    MERGE (event)-[relation:HAS_PARTICIPANT {
                        role: $role
                    }]->(entity)
                    SET relation.confidence = $confidence
                    """,
                    event_key=event["event_key"],
                    normalized_name=entity["normalized_name"],
                    entity_type=entity["entity_type"],
                    role=participant["role"],
                    confidence=participant["confidence"],
                ).consume()
                continue

            participant_key = build_anonymous_participant_key(
                platform,
                post_id,
                event["event_key"],
                participant,
            )
            tx.run(
                """
                MATCH (p:Post {platform: $platform, platform_id: $post_id})
                MATCH (event:Event {event_key: $event_key})
                MERGE (anonymous:AnonymousParticipant {
                    participant_key: $participant_key
                })
                SET anonymous.post_key = $post_key,
                    anonymous.participant_text = $participant_text,
                    anonymous.name = $participant_text
                MERGE (p)-[:HAS_ANONYMOUS_PARTICIPANT]->(anonymous)
                MERGE (event)-[relation:HAS_PARTICIPANT {
                    role: $role
                }]->(anonymous)
                SET relation.confidence = $confidence
                """,
                platform=platform,
                post_id=post_id,
                event_key=event["event_key"],
                participant_key=participant_key,
                post_key=post_key,
                participant_text=participant["participant_text"],
                role=participant["role"],
                confidence=participant["confidence"],
            ).consume()

    tx.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
              -[:HAS_ANONYMOUS_PARTICIPANT]->(anonymous:AnonymousParticipant)
        WHERE NOT (anonymous)<-[:HAS_PARTICIPANT]-(:Event)
        DETACH DELETE anonymous
        """,
        platform=platform,
        post_id=post_id,
    ).consume()


def upsert_event_relations(
    tx,
    events: list[dict],
    relations: list[dict],
) -> None:
    local_to_key = {event["local_id"]: event["event_key"] for event in events}
    event_keys = list(local_to_key.values())
    tx.run(
        """
        MATCH (source:Event)-[relation]->(target:Event)
        WHERE source.event_key IN $event_keys
          AND type(relation) IN $relation_types
        DELETE relation
        """,
        event_keys=event_keys,
        relation_types=sorted(EVENT_RELATION_TYPES),
    ).consume()

    for relation in relations:
        relation_type = relation["type"]
        if relation_type not in EVENT_RELATION_TYPES:
            continue
        query = f"""
            MATCH (source:Event {{event_key: $source_event_key}})
            MATCH (target:Event {{event_key: $target_event_key}})
            MERGE (source)-[relation:{relation_type}]->(target)
            SET relation.evidence_text = $evidence_text,
                relation.knowledge_prompt_version = $knowledge_prompt_version
        """
        tx.run(
            query,
            source_event_key=local_to_key[relation["source_event_id"]],
            target_event_key=local_to_key[relation["target_event_id"]],
            evidence_text=relation["evidence_text"],
            knowledge_prompt_version=KNOWLEDGE_PROMPT_VERSION,
        ).consume()


def save_knowledge_tx(
    tx,
    platform: str,
    post_id: str,
    knowledge: dict,
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
        """,
        platform=platform,
        post_id=post_id,
        knowledge_model=OLLAMA_MODEL,
        knowledge_prompt_version=KNOWLEDGE_PROMPT_VERSION,
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
            WHEN coalesce(p.knowledge_model, '') <> $knowledge_model
              OR coalesce(p.knowledge_prompt_version, '')
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
