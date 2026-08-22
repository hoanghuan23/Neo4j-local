import os
from uuid import uuid4

import pytest
from neo4j import GraphDatabase

from backend.config import Settings
from backend.neo4j_repository import Neo4jRepository


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_NEO4J_INTEGRATION") != "1",
    reason="set RUN_NEO4J_INTEGRATION=1 to test the local Neo4j instance",
)


def test_search_events_filters_recent_old_and_missing_posted_at():
    settings = Settings()
    marker = f"codex-time-filter-{uuid4().hex}"
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    rows = [
        {"suffix": "recent", "age_hours": 2},
        {"suffix": "old", "age_hours": 26},
        {"suffix": "missing", "age_hours": None},
    ]

    try:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                UNWIND $rows AS row
                CREATE (post:Post {
                    platform: 'codex-test',
                    platform_id: $marker + '-' + row.suffix,
                    content: 'Nội dung kiểm thử entity filter',
                    posted_at: CASE
                        WHEN row.age_hours IS NULL THEN NULL
                        ELSE localdatetime()
                             - duration({hours: row.age_hours})
                    END
                })
                CREATE (mention:EventMention {
                    mention_key: $marker + '-mention-' + row.suffix,
                    description: 'Mention kiểm thử entity filter'
                })
                CREATE (event:Event {
                    event_key: $marker + '-event-' + row.suffix,
                    description: 'Sự kiện kiểm thử entity filter'
                })
                CREATE (entity:Entity {
                    entity_id: $marker + '-entity-' + row.suffix,
                    name: $marker,
                    normalized_name: $marker + '-' + row.suffix,
                    search_name: $marker + '-' + row.suffix,
                    type: 'PERSON'
                })
                CREATE (post)-[:HAS_EVENT_MENTION]->(mention)
                CREATE (mention)-[:EVIDENCE_FOR]->(event)
                CREATE (mention)-[:HAS_PARTICIPANT]->(entity)
                """,
                rows=rows,
                marker=marker,
            ).consume()

        repository = Neo4jRepository(settings)
        repository.driver = driver
        results = repository.search_events(
            location=None,
            entity=marker,
            hours=24,
            limit=10,
        )

        assert [item["event_key"] for item in results] == [
            f"{marker}-event-recent"
        ]
    finally:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                MATCH (node)
                WHERE node.platform_id STARTS WITH $marker
                   OR node.mention_key STARTS WITH $marker
                   OR node.event_key STARTS WITH $marker
                   OR node.entity_id STARTS WITH $marker
                DETACH DELETE node
                """,
                marker=marker,
            ).consume()
        driver.close()
