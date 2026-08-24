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


def test_search_events_ranks_event_matching_both_entities_first():
    settings = Settings()
    marker = f"codex-entity-rank-{uuid4().hex}"
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    try:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                CREATE (huan:Entity {
                    entity_id: $marker + '-entity-huan',
                    name: 'Huấn',
                    normalized_name: $marker + '-huan',
                    search_name: $marker + '-huan',
                    type: 'PERSON'
                })
                CREATE (phu:Entity {
                    entity_id: $marker + '-entity-phu',
                    name: 'Phú Lê',
                    normalized_name: $marker + '-phu',
                    search_name: $marker + '-phu',
                    type: 'PERSON'
                })
                CREATE (shared_post:Post {
                    platform: 'codex-test',
                    platform_id: $marker + '-post-shared',
                    content: 'Sự kiện chung',
                    posted_at: localdatetime() - duration({hours: 10})
                })
                CREATE (shared_mention:EventMention {
                    mention_key: $marker + '-mention-shared',
                    description: 'Huấn tặng quà cho Phú Lê'
                })
                CREATE (shared_event:Event {
                    event_key: $marker + '-event-shared',
                    description: 'Huấn tặng quà cho Phú Lê'
                })
                CREATE (single_post:Post {
                    platform: 'codex-test',
                    platform_id: $marker + '-post-single',
                    content: 'Sự kiện riêng mới hơn',
                    posted_at: localdatetime() - duration({hours: 2})
                })
                CREATE (single_mention:EventMention {
                    mention_key: $marker + '-mention-single',
                    description: 'Sự kiện riêng của Phú Lê'
                })
                CREATE (single_event:Event {
                    event_key: $marker + '-event-single',
                    description: 'Sự kiện riêng của Phú Lê'
                })
                CREATE (shared_post)-[:HAS_EVENT_MENTION]->(shared_mention)
                CREATE (shared_mention)-[:EVIDENCE_FOR]->(shared_event)
                CREATE (shared_mention)-[:HAS_PARTICIPANT]->(huan)
                CREATE (shared_mention)-[:HAS_PARTICIPANT]->(phu)
                CREATE (single_post)-[:HAS_EVENT_MENTION]->(single_mention)
                CREATE (single_mention)-[:EVIDENCE_FOR]->(single_event)
                CREATE (single_mention)-[:HAS_PARTICIPANT]->(phu)
                """,
                marker=marker,
            ).consume()

        repository = Neo4jRepository(settings)
        repository.driver = driver
        results = repository.search_events(
            location=None,
            entity=f"{marker}-huan và {marker}-phu",
            hours=24,
            limit=10,
        )

        assert [item["event_key"] for item in results] == [
            f"{marker}-event-shared",
            f"{marker}-event-single",
        ]
        assert [item["matched_entity_count"] for item in results] == [2, 1]
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


def test_location_filter_is_event_scoped_for_current_and_legacy_schemas():
    settings = Settings()
    marker = f"codex-location-scope-{uuid4().hex}"
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    try:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                CREATE (location:Entity {
                    test_marker: $marker,
                    entity_id: $marker + '-location',
                    name: $marker,
                    normalized_name: $marker,
                    search_name: $marker,
                    type: 'LOCATION'
                })
                CREATE (current_post:Post {
                    test_marker: $marker,
                    platform: 'codex-test',
                    platform_id: $marker + '-current-post',
                    content: 'Bài tổng hợp tại ' + $marker,
                    posted_at: localdatetime() - duration({hours: 1})
                })
                CREATE (current_post)-[:MENTIONS]->(location)
                FOREACH (row IN [
                    {suffix: 'linked', description: 'Event có cạnh địa điểm'},
                    {suffix: 'text', description: 'Event tại ' + $marker},
                    {suffix: 'sibling', description: 'Event không liên quan'}
                ] |
                    CREATE (mention:EventMention {
                        test_marker: $marker,
                        mention_key: $marker + '-current-mention-' + row.suffix,
                        description: row.description
                    })
                    CREATE (event:Event {
                        test_marker: $marker,
                        event_key: $marker + '-current-event-' + row.suffix,
                        description: row.description
                    })
                    CREATE (current_post)-[:HAS_EVENT_MENTION]->(mention)
                    CREATE (mention)-[:EVIDENCE_FOR]->(event)
                )
                WITH location, current_post
                MATCH (current_post)-[:HAS_EVENT_MENTION]->(linked:EventMention)
                WHERE linked.mention_key ENDS WITH '-linked'
                CREATE (linked)-[:HAS_PARTICIPANT {role: 'LOCATION'}]->(location)
                CREATE (single_post:Post {
                    test_marker: $marker,
                    platform: 'codex-test',
                    platform_id: $marker + '-current-single-post',
                    content: 'Bài đơn tại ' + $marker,
                    posted_at: localdatetime() - duration({hours: 1})
                })
                CREATE (single_mention:EventMention {
                    test_marker: $marker,
                    mention_key: $marker + '-current-single-mention',
                    description: 'Event đơn không nhắc địa điểm'
                })
                CREATE (single_event:Event {
                    test_marker: $marker,
                    event_key: $marker + '-current-single-event',
                    description: 'Event đơn không nhắc địa điểm'
                })
                CREATE (single_post)-[:MENTIONS]->(location)
                CREATE (single_post)-[:HAS_EVENT_MENTION]->(single_mention)
                CREATE (single_mention)-[:EVIDENCE_FOR]->(single_event)
                CREATE (legacy_post:Post {
                    test_marker: $marker,
                    platform: 'codex-test',
                    platform_id: $marker + '-legacy-post',
                    content: 'Bài legacy tổng hợp tại ' + $marker,
                    posted_at: localdatetime() - duration({hours: 1})
                })
                CREATE (legacy_post)-[:MENTIONS]->(location)
                FOREACH (row IN [
                    {suffix: 'linked', description: 'Legacy có cạnh địa điểm'},
                    {suffix: 'text', description: 'Legacy tại ' + $marker},
                    {suffix: 'sibling', description: 'Legacy không liên quan'}
                ] |
                    CREATE (event:Event {
                        test_marker: $marker,
                        event_key: $marker + '-legacy-event-' + row.suffix,
                        description: row.description
                    })
                    CREATE (legacy_post)-[:DESCRIBES]->(event)
                )
                WITH location, legacy_post
                MATCH (legacy_post)-[:DESCRIBES]->(legacy_linked:Event)
                WHERE legacy_linked.event_key ENDS WITH '-linked'
                CREATE (legacy_linked)-[:HAS_PARTICIPANT {role: 'LOCATION'}]
                       ->(location)
                CREATE (legacy_single_post:Post {
                    test_marker: $marker,
                    platform: 'codex-test',
                    platform_id: $marker + '-legacy-single-post',
                    content: 'Bài legacy đơn tại ' + $marker,
                    posted_at: localdatetime() - duration({hours: 1})
                })
                CREATE (legacy_single_event:Event {
                    test_marker: $marker,
                    event_key: $marker + '-legacy-single-event',
                    description: 'Legacy đơn không nhắc địa điểm'
                })
                CREATE (legacy_single_post)-[:MENTIONS]->(location)
                CREATE (legacy_single_post)-[:DESCRIBES]->(legacy_single_event)
                """,
                marker=marker,
            ).consume()

        repository = Neo4jRepository(settings)
        repository.driver = driver
        results = repository.search_events(
            location=marker,
            entity=None,
            hours=24,
            limit=20,
        )

        assert {item["event_key"] for item in results} == {
            f"{marker}-current-event-linked",
            f"{marker}-current-event-text",
            f"{marker}-current-single-event",
            f"{marker}-legacy-event-linked",
            f"{marker}-legacy-event-text",
            f"{marker}-legacy-single-event",
        }
    finally:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                "MATCH (node {test_marker: $marker}) DETACH DELETE node",
                marker=marker,
            ).consume()
        driver.close()
