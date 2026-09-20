import argparse
import json

from neo4j import GraphDatabase

from event_time import normalize_occurrence_date
from knowledge_settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def load_mentions(session, *, limit: int | None = None) -> list[dict]:
    return [
        dict(record)
        for record in session.run(
            """
            MATCH (post:Post)-[:HAS_EVENT_MENTION]->(mention:EventMention)
            WHERE mention.occurrence_date IS NULL
               OR mention.occurrence_date_source IN ['POSTED_AT_FALLBACK', 'UNKNOWN']
            RETURN mention.mention_key AS mention_key,
                   mention.time_expression AS time_expression,
                   mention.evidence_text AS evidence_text,
                   post.posted_at AS posted_at
            ORDER BY mention.mention_key
            LIMIT $limit
            """,
            limit=limit if limit is not None else 1_000_000_000,
        )
    ]


def backfill_occurrence_dates(
    session,
    *,
    apply: bool = False,
    limit: int | None = None,
) -> dict:
    mentions = load_mentions(session, limit=limit)
    rows = []
    counts = {"EXPLICIT": 0, "RELATIVE": 0, "UNRESOLVED": 0}
    for mention in mentions:
        normalized = normalize_occurrence_date(
            mention.get("time_expression"),
            mention.get("posted_at"),
            evidence_text=mention.get("evidence_text"),
        )
        source = normalized["occurrence_date_source"]
        counts[source if source is not None else "UNRESOLVED"] += 1
        rows.append({
            "mention_key": mention["mention_key"],
            "occurrence_date": normalized["occurrence_date"],
            "occurrence_date_source": source,
        })

    if apply and rows:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (mention:EventMention {mention_key: row.mention_key})
            SET mention.occurrence_date = row.occurrence_date,
                mention.occurrence_date_source = row.occurrence_date_source,
                mention.updated_at = datetime()
            """,
            rows=rows,
        ).consume()
    return {
        "selected": len(mentions),
        "updated": len(rows) if apply else 0,
        "sources": counts,
        "mentions": [
            dict(row, occurrence_date=(
                row["occurrence_date"].isoformat()
                if row["occurrence_date"] is not None else None
            ))
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chuẩn hóa occurrence_date cho EventMention hiện có",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit phải lớn hơn 0")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    try:
        with driver.session(database="neo4j") as session:
            result = backfill_occurrence_dates(
                session,
                apply=args.apply,
                limit=args.limit,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
