import argparse
import json
import logging

from neo4j import GraphDatabase

from event_titles import generate_event_title
from knowledge_gemini import GeminiKnowledgeCaller
from knowledge_settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def load_candidates(session, *, event_key=None, limit=None) -> list[dict]:
    return [
        dict(record)
        for record in session.run(
            """
            MATCH (event:Event)
            WHERE coalesce(event.description, '') <> ''
              AND ($event_key IS NULL OR event.event_key = $event_key)
              AND (
                coalesce(event.title, '') = ''
                OR coalesce(event.title_needs_backfill, false)
              )
            RETURN event.event_key AS event_key,
                   event.description AS description,
                   event.title AS current_title
            ORDER BY event.event_key
            LIMIT $limit
            """,
            event_key=event_key,
            limit=limit if limit is not None else 1_000_000_000,
        )
    ]


def _write_title(
    session,
    *,
    event_key: str,
    title: str,
    needs_backfill: bool,
) -> None:
    session.run(
        """
        MATCH (event:Event {event_key: $event_key})
        SET event.title = $title,
            event.title_needs_backfill = $needs_backfill,
            event.title_updated_at = datetime()
        WITH event
        OPTIONAL MATCH (mention:EventMention)-[:EVIDENCE_FOR]->(event)
        WITH mention, coalesce(mention.title, '') = '' AS mention_missing_title
        SET mention.title = coalesce(mention.title, $title),
            mention.title_needs_backfill = CASE
                WHEN mention_missing_title THEN $needs_backfill
                ELSE coalesce(mention.title_needs_backfill, false)
            END
        """,
        event_key=event_key,
        title=title,
        needs_backfill=needs_backfill,
    ).consume()


def backfill_event_titles(
    session,
    call_model,
    *,
    apply: bool = False,
    event_key: str | None = None,
    limit: int | None = None,
) -> dict:
    candidates = load_candidates(
        session,
        event_key=event_key,
        limit=limit,
    )
    previews = []
    updated = 0
    fallback = 0
    for candidate in candidates:
        title, needs_backfill = generate_event_title(
            candidate["description"],
            call_model,
        )
        fallback += int(needs_backfill)
        previews.append({
            "event_key": candidate["event_key"],
            "current_title": candidate.get("current_title"),
            "title": title,
            "needs_backfill": needs_backfill,
        })
        if apply:
            _write_title(
                session,
                event_key=candidate["event_key"],
                title=title,
                needs_backfill=needs_backfill,
            )
            updated += 1
    return {
        "selected": len(candidates),
        "updated": updated,
        "fallback": fallback,
        "events": previews,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tạo title ngắn cho các Event cũ trong Neo4j",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--event-key")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit phải lớn hơn 0")

    caller = GeminiKnowledgeCaller()
    result = {"selected": 0}
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    try:
        with driver.session(database="neo4j") as session:
            result = backfill_event_titles(
                session,
                caller,
                apply=args.apply,
                event_key=args.event_key,
                limit=args.limit,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        caller.print_cost_summary(
            target_posts=result["selected"],
            stage_label="backfill Event title",
        )
        caller.close()
        driver.close()


if __name__ == "__main__":
    main()
