"""Retry LOCATION_HIERARCHY without rerunning knowledge extraction."""

import argparse
import json
import logging

from neo4j import GraphDatabase

from knowledge_gemini import GeminiKnowledgeCaller
from knowledge_relations.location_hierarchy import enrich_location_hierarchy
from knowledge_settings import (
    LOCATION_HIERARCHY_MODULE_VERSION,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)


def load_candidates(session, *, limit: int) -> list[dict]:
    return [dict(record) for record in session.run(
        """
        MATCH (post:Post)-[:MENTIONS]->(location:Entity {type: 'LOCATION'})
        WHERE post.knowledge_classifier_decision = 'DEEP'
          AND (coalesce(post.location_hierarchy_status, '') <> 'COMPLETED'
               OR coalesce(post.location_hierarchy_version, '') <> $version)
        WITH DISTINCT post
        RETURN post.platform AS platform, post.platform_id AS post_id,
               post.content AS content
        ORDER BY post.posted_at DESC LIMIT $limit
        """, version=LOCATION_HIERARCHY_MODULE_VERSION, limit=limit)]


def _knowledge_from_mentions(session, platform: str, post_id: str) -> dict:
    entities = []
    for index, record in enumerate(session.run(
        """
        MATCH (post:Post {platform: $platform, platform_id: $post_id})
              -[:MENTIONS]->(location:Entity {type: 'LOCATION'})
        RETURN location.name AS name, location.normalized_name AS canonical_name
        ORDER BY location.normalized_name
        """, platform=platform, post_id=post_id), start=1):
        entities.append({
            "local_id": f"location_{index}", "name": record["name"],
            "canonical_name": record["canonical_name"], "type": "LOCATION",
        })
    return {"entities": entities, "events": [], "event_relations": []}


def backfill_location_hierarchy(session, call_model, *, limit: int = 100) -> dict:
    candidates = load_candidates(session, limit=limit)
    summary = {"selected": len(candidates), "completed": 0, "failed": 0}
    for post in candidates:
        knowledge = _knowledge_from_mentions(session, post["platform"], post["post_id"])
        result = enrich_location_hierarchy(
            session, post["platform"], post["post_id"], post["content"], knowledge,
            call_model=call_model,
        )
        if result["errors"]:
            summary["failed"] += 1
        else:
            summary["completed"] += 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill LOCATION_HIERARCHY")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit phải lớn hơn 0")
    logging.basicConfig(level=logging.WARNING)
    caller = GeminiKnowledgeCaller()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database="neo4j") as session:
            print(json.dumps(backfill_location_hierarchy(session, caller, limit=args.limit), ensure_ascii=False, indent=2))
    finally:
        caller.close()
        driver.close()


if __name__ == "__main__":
    main()
