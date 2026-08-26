import argparse
import hashlib
import json

from neo4j import GraphDatabase

from knowledge_consolidation import (
    _resolve_prompt,
    _summary_prompt,
    _validated_decisions,
    consolidate_pending_mentions,
    select_candidates,
)
from knowledge_gemini import GeminiKnowledgeCaller
from knowledge_persistence import create_knowledge_schema
from knowledge_settings import (
    EVENT_AUTO_MERGE_THRESHOLD,
    EVENT_CONSOLIDATION_SCHEMA,
    EVENT_SUMMARY_SCHEMA,
    EVENT_RELATION_TYPES,
    KNOWLEDGE_PROMPT_VERSION,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)


def _mention_key(platform: str, post_id: str, event_key: str) -> str:
    identity = f"legacy|{platform}|{post_id}|{event_key}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_legacy_rows(session, entity_name: str) -> list[dict]:
    rows = []
    for record in session.run(
        """
        MATCH (post:Post)-[:DESCRIBES]->(event:Event)
        WHERE coalesce(event.schema_version, 1) = 1
          AND (
            toLower(post.content) CONTAINS toLower($entity_name)
            OR EXISTS {
                MATCH (event)-[:HAS_PARTICIPANT]->(entity:Entity)
                WHERE entity.normalized_name = toLower($entity_name)
                   OR toLower(entity.name) = toLower($entity_name)
            }
          )
        OPTIONAL MATCH (event)-[:HAS_PARTICIPANT]->(participant)
        WITH post, event,
             collect(DISTINCT coalesce(
                 participant.normalized_name,
                 participant.normalized_text,
                 participant.name
             )) AS participants
        RETURN post.platform AS platform,
               post.platform_id AS post_id,
               post.posted_at AS posted_at,
               event.event_key AS event_key,
               event.type AS type,
               event.title AS title,
               event.description AS description,
               event.evidence_text AS evidence_text,
               event.status AS status,
               event.time_expression AS time_expression,
               event.confidence AS confidence,
               event.created_at AS created_at,
               participants
        ORDER BY post.posted_at
        """,
        entity_name=entity_name,
    ):
        item = dict(record)
        item["mention_key"] = _mention_key(
            item["platform"], item["post_id"], item["event_key"]
        )
        item["current_event_key"] = item["event_key"]
        rows.append(item)
    return rows


def migrate_legacy_row(tx, row: dict) -> str:
    parameters = {**row, "title": row.get("title")}
    tx.run(
        """
        MATCH (post:Post {platform: $platform, platform_id: $post_id})
              -[:DESCRIBES]->(event:Event {event_key: $event_key})
        MERGE (mention:EventMention {mention_key: $mention_key})
        ON CREATE SET mention.created_at = datetime()
        SET mention.type = $type,
            mention.title = coalesce($title, $description),
            mention.title_needs_backfill = $title IS NULL,
            mention.description = $description,
            mention.evidence_text = $evidence_text,
            mention.status = $status,
            mention.time_expression = $time_expression,
            mention.confidence = $confidence,
            mention.platform = $platform,
            mention.post_id = $post_id,
            mention.consolidation_status = 'PENDING',
            mention.updated_at = datetime(),
            event.schema_version = 2,
            event.legacy_event_keys = CASE
                WHEN event.event_key IN coalesce(event.legacy_event_keys, [])
                THEN event.legacy_event_keys
                ELSE coalesce(event.legacy_event_keys, []) + event.event_key
            END
        MERGE (post)-[:HAS_EVENT_MENTION]->(mention)
        MERGE (mention)-[:EVIDENCE_FOR]->(event)
        WITH event, mention
        OPTIONAL MATCH (event)-[participant:HAS_PARTICIPANT]->(target)
        WITH mention, participant, target
        WHERE participant IS NOT NULL
        MERGE (mention)-[copy:HAS_PARTICIPANT {role: participant.role}]->(target)
        SET copy.confidence = participant.confidence
        """,
        **parameters,
    ).consume()
    return row["mention_key"]


def migrate_legacy_relations(tx, mention_keys: list[str]) -> None:
    """Copy same-post Event relations to their provenance mentions."""
    for relation_type in sorted(EVENT_RELATION_TYPES):
        tx.run(
            f"""
            MATCH (post:Post)-[:HAS_EVENT_MENTION]->(source:EventMention)
            MATCH (post)-[:HAS_EVENT_MENTION]->(target:EventMention)
            MATCH (source)-[:EVIDENCE_FOR]->(source_event:Event)
                  -[legacy:{relation_type}]->(target_event:Event)
            MATCH (target)-[:EVIDENCE_FOR]->(target_event)
            WHERE source.mention_key IN $mention_keys
              AND target.mention_key IN $mention_keys
              AND source <> target
            MERGE (source)-[relation:{relation_type}]->(target)
            SET relation.evidence_text = coalesce(
                    legacy.evidence_text,
                    head(coalesce(legacy.evidence_texts, []))
                ),
                relation.knowledge_prompt_version = $prompt_version
            """,
            mention_keys=mention_keys,
            prompt_version=KNOWLEDGE_PROMPT_VERSION,
        ).consume()


def dry_run(rows: list[dict], caller: GeminiKnowledgeCaller) -> dict:
    events = [
        {
            "event_key": row["event_key"],
            "type": row["type"],
            "title": row.get("title") or row["description"],
            "description": row["description"],
            "descriptions": [row["description"]],
            "status": row["status"],
            "participants": row["participants"],
            "first_seen_at": row["posted_at"],
            "last_seen_at": row["posted_at"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    same_pairs = []
    possible = []
    different = []
    seen_pairs = set()
    for row in rows:
        candidates = select_candidates(row, events)
        if not candidates:
            continue
        decisions = _validated_decisions(
            caller(_resolve_prompt(row, candidates), EVENT_CONSOLIDATION_SCHEMA),
            candidates,
        )
        for decision in decisions:
            pair = tuple(sorted((row["event_key"], decision["candidate_event_key"])))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            item = {"events": pair, **decision}
            if (
                decision["decision"] == "SAME_EVENT"
                and decision["confidence"] >= EVENT_AUTO_MERGE_THRESHOLD
            ):
                same_pairs.append(item)
            elif decision["decision"] == "POSSIBLE_SAME_EVENT":
                possible.append(item)
            else:
                different.append(item)

    descriptions = []
    if same_pairs:
        grouped_keys = {key for item in same_pairs for key in item["events"]}
        sources = [
            {
                key: row.get(key)
                for key in (
                    "mention_key", "type", "title", "description", "evidence_text",
                    "status", "time_expression",
                )
            }
            for row in rows if row["event_key"] in grouped_keys
        ]
        descriptions.append(caller(_summary_prompt(sources), EVENT_SUMMARY_SCHEMA))
    return {
        "selected": len(rows),
        "same_event": same_pairs,
        "possible_same_event": possible,
        "different_event": different,
        "expected_descriptions": descriptions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill EventMention và gom Event legacy theo entity"
    )
    parser.add_argument("--entity", default="vetc")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    caller = GeminiKnowledgeCaller()
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    try:
        with driver.session(database="neo4j") as session:
            rows = load_legacy_rows(session, args.entity)
            if not args.apply:
                print(json.dumps(dry_run(rows, caller), ensure_ascii=False, indent=2, default=str))
                return

            create_knowledge_schema(session)
            source_posts = {(row["platform"], row["post_id"]) for row in rows}
            mention_keys = [
                session.execute_write(migrate_legacy_row, row) for row in rows
            ]
            session.execute_write(migrate_legacy_relations, mention_keys)
            stats = consolidate_pending_mentions(
                session,
                call_model=caller,
                mention_keys=mention_keys,
            )
            migrated_posts = session.run(
                """
                MATCH (post:Post)-[:HAS_EVENT_MENTION]->(mention:EventMention)
                WHERE mention.mention_key IN $mention_keys
                RETURN count(DISTINCT post) AS count
                """,
                mention_keys=mention_keys,
            ).single()["count"]
            if migrated_posts != len(source_posts):
                raise RuntimeError(
                    f"Mất liên kết Post: trước={len(source_posts)}, sau={migrated_posts}"
                )
            print(json.dumps({
                "selected_posts": len(source_posts),
                "migrated_mentions": len(mention_keys),
                "consolidation": stats,
            }, ensure_ascii=False, indent=2))
    finally:
        caller.print_cost_summary(
            target_posts=caller.usage.requests,
            stage_label="VETC MIGRATION",
        )
        caller.close()
        driver.close()


if __name__ == "__main__":
    main()
