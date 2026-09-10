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
            entity=f"giữa {marker}-huan và {marker}-phu",
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
            f"{marker}-current-single-event",
            f"{marker}-legacy-event-linked",
            f"{marker}-legacy-single-event",
        }
        related = repository.search_related_events(
            location=marker, entity=None, hours=24, limit=20,
        )
        assert {item["event_key"] for item in related} == {
            f"{marker}-current-event-text", f"{marker}-legacy-event-text",
        }
        assert all(item["relation_reasons"] for item in related)
    finally:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                "MATCH (node {test_marker: $marker}) DETACH DELETE node",
                marker=marker,
            ).consume()
        driver.close()


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('hierarchy_relation', ['PART_OF', 'IN_REGION', 'SUBORDINATE_TO'])
def test_related_hierarchy_direct_children_and_exclusion(legacy, hierarchy_relation):
    organization = hierarchy_relation == 'SUBORDINATE_TO'
    entity_type = 'ORGANIZATION' if organization else 'LOCATION'
    reason_kind = 'organization_hierarchy' if organization else 'location_hierarchy'
    settings = Settings()
    marker = f'codex-related-{uuid4().hex}'
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        with driver.session(database=settings.neo4j_database) as session:
            session.run('''
                CREATE (parent:Entity {test_marker: $marker, type: $entity_type,
                    normalized_name: $marker, aliases: [$marker + '-alias'],
                    search_name: $marker})
                CREATE (child:Entity {test_marker: $marker, type: $entity_type, name: 'Quận'})
                CREATE (leaf:Entity {test_marker: $marker, type: $entity_type, name: 'Phường'})
                CREATE (wrong:Entity {test_marker: $marker, type: 'PERSON'})
                CREATE (outside:Entity {test_marker: $marker, type: $entity_type, name: 'Ngoài'})
                CREATE (child)-[:HIERARCHY_RELATION]->(parent)
                CREATE (leaf)-[:HIERARCHY_RELATION]->(child)
                CREATE (reverse:Entity {test_marker: $marker, type: $entity_type, name: 'Ngược'})
                CREATE (parent)-[:HIERARCHY_RELATION]->(reverse)
                CREATE (invalid:Entity {test_marker: $marker, type: 'PERSON', name: 'Sai loại'})
                CREATE (invalid)-[:HIERARCHY_RELATION]->(parent)
                CREATE (outside)-[:HIERARCHY_RELATION]->(wrong)-[:HIERARCHY_RELATION]->(parent)
            '''.replace('HIERARCHY_RELATION', hierarchy_relation), marker=marker, entity_type=entity_type).consume()
            rows = [
                ('child', 'Quận', 'participant', False, 1),
                ('leaf', 'Phường', 'participant', False, 1),
                ('reverse', 'Ngược', 'participant', False, 1),
                ('invalid', 'Sai loại', 'participant', False, 1),
                ('post', 'Quận', 'post', False, 1),
                ('multi', 'Quận', 'post', True, 1),
                ('outside', 'Ngoài', 'participant', False, 1),
                ('old', 'Quận', 'participant', False, 72),
                ('direct', 'Quận', 'participant', False, 1),
                ('description', 'Quận', 'participant', False, 1),
                ('shared', 'Quận', 'participant', False, 1),
            ]
            for suffix, place, binding, multi, age in rows:
                relation = ('CREATE (post)-[:DESCRIBES]->(event) WITH post, event, event AS mention, location'
                            if legacy else '''CREATE (mention:EventMention {test_marker: $marker,
                                mention_key: $marker + $suffix})
                                CREATE (post)-[:HAS_EVENT_MENTION]->(mention)
                                CREATE (mention)-[:EVIDENCE_FOR]->(event)
                                WITH post, event, mention, location''')
                session.run('''
                    MATCH (location:Entity {test_marker: $marker, name: $place})
                    CREATE (post:Post {test_marker: $marker, platform: 'codex-test',
                        platform_id: $marker + $suffix,
                        content: CASE WHEN $suffix = 'direct' THEN $marker ELSE 'Bài thử' END,
                        posted_at: localdatetime() - duration({hours: $age})})
                    CREATE (event:Event {test_marker: $marker, event_key: $marker + $suffix,
                        description: CASE WHEN $suffix = 'description' THEN $marker ELSE 'Sự kiện' END})
                ''' + relation + ('''
                    CREATE (post)-[:MENTIONS]->(location)
                ''' if binding == 'post' else '''
                    CREATE (mention)-[:HAS_PARTICIPANT]->(location)
                ''') + ('''
                    CREATE (extra:Event {test_marker: $marker, event_key: $marker + '-extra'})
                    CREATE (post)-[:DESCRIBES]->(extra)
                ''' if multi and legacy else '''
                    CREATE (extra:EventMention {test_marker: $marker, mention_key: $marker + '-extra'})
                    CREATE (post)-[:HAS_EVENT_MENTION]->(extra)
                ''' if multi else ''), marker=marker, suffix=suffix, place=place, age=age).consume()
            # A separate direct post in the other schema excludes the shared event.
            session.run('''
                MATCH (event:Event {event_key: $marker + 'shared'})
                MATCH (parent:Entity {test_marker: $marker, normalized_name: $marker})
                CREATE (post:Post {test_marker: $marker, platform: 'codex-test',
                    platform_id: $marker + '-direct-copy', content: $marker,
                    posted_at: localdatetime()})
                CREATE (post)-[:DESCRIBES]->(event)
                CREATE (post)-[:MENTIONS]->(parent)
            ''', marker=marker).consume()
        repository = Neo4jRepository(settings)
        repository.driver = driver
        args = dict(location=None if organization else marker,
                    entity=marker if organization else None, hours=48, limit=20)
        results = repository.search_related_events(**args)
        assert {r['event_key'] for r in results} == {
            marker + suffix for suffix in ('child', 'post', 'direct', 'description')
        }
        for result in results:
            hierarchy = [r for r in result['relation_reasons'] if r['kind'] == reason_kind]
            assert hierarchy
            assert all(r['relationship'] == hierarchy_relation for r in hierarchy)
            assert all(r['via_entity']['name'] == 'Quận' for r in hierarchy)
            assert all(r['label'] == 'Liên quan qua: Quận' for r in hierarchy)
        if organization:
            alias_results = repository.search_related_events(**{**args, 'entity': marker + '-alias'})
            assert marker + 'child' in {r['event_key'] for r in alias_results}
            assert repository.search_related_events(**{**args, 'entity': None, 'location': marker + '-alias'}) == []
            assert marker + 'child' not in {r['event_key'] for r in repository.search_events(**args)}
        page = repository.search_related_events(**{**args, 'limit': 1})
        last = page[-1]
        rest = repository.search_related_events(**args, after=(
            last['matched_entity_count'], last['post']['posted_at'], last['event_key'],
        ))
        assert [r['event_key'] for r in page + rest] == [r['event_key'] for r in results]
        assert repository.search_related_events(**{**args, 'location': marker + '-missing'}) == []
        assert repository.search_related_events(**{**args, 'entity': 'nonexistent-person'}) == []
    finally:
        with driver.session(database=settings.neo4j_database) as session:
            session.run('MATCH (n {test_marker: $marker}) DETACH DELETE n', marker=marker).consume()
        driver.close()


@pytest.fixture
def precision_graph():
    """Use a distinct future date so literal Hanoi queries don't read user data."""
    from datetime import date

    settings = Settings()
    marker = f'codex-precision-{uuid4().hex}'
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
    )
    repository = Neo4jRepository(settings)
    repository.driver = driver

    def run(query, **params):
        with driver.session(database=settings.neo4j_database) as session:
            return session.run(query, marker=marker, **params).data()

    def entity(key, name, entity_type='ORGANIZATION', **properties):
        run('''CREATE (e:Entity {test_marker: $marker, entity_id: $marker + $key})
               SET e += $properties''', key=key,
            properties={'name': name, 'type': entity_type, **properties})

    def event(key, *, legacy=False, content='Bài thử', description='Sự kiện thử',
              participants=(), mentions=(), post_key=None, age=0):
        post_key = post_key or key
        run('''MERGE (p:Post {test_marker: $marker, platform_id: $marker + $post_key})
               SET p.platform = 'codex-test', p.content = $content,
                   p.posted_at = localdatetime('2099-01-02T08:00:00') - duration({hours: $age})
               MERGE (e:Event {test_marker: $marker, event_key: $marker + $key})
               SET e.description = $description
            ''' + ('''MERGE (p)-[:DESCRIBES]->(e) WITH p, e AS binding
            ''' if legacy else '''
               MERGE (m:EventMention {test_marker: $marker, mention_key: $marker + $post_key + $key})
               SET m.description = $description
               MERGE (p)-[:HAS_EVENT_MENTION]->(m)
               MERGE (m)-[:EVIDENCE_FOR]->(e)
               WITH p, m AS binding
            ''') + '''
               CALL {
                 WITH binding
                 UNWIND $participants AS id
                 MATCH (entity:Entity {entity_id: $marker + id})
                 MERGE (binding)-[:HAS_PARTICIPANT]->(entity)
               }
               WITH p
               UNWIND $mentions AS id
               MATCH (entity:Entity {entity_id: $marker + id})
               MERGE (p)-[:MENTIONS]->(entity)
            ''', key=key, post_key=post_key, content=content, description=description,
            participants=list(participants), mentions=list(mentions), age=age)
        return marker + key

    def search(*, related=False, **params):
        return (repository.search_related_events if related else repository.search_events)(
            **{'location': None, 'entity': None, 'hours': 24,
               'posted_date': date(2099, 1, 2), 'limit': 100, **params},
        )

    try:
        yield entity, event, search, run, marker
    finally:
        run('MATCH (n {test_marker: $marker}) DETACH DELETE n')
        driver.close()


@pytest.mark.parametrize('legacy', [False, True])
def test_hanoi_university_is_related_with_real_entity_evidence(precision_graph, legacy):
    from backend.models import EventResult

    entity, event, search, run, marker = precision_graph
    entity('hanoi', 'Hà Nội', 'LOCATION')
    entity('university', 'Đại học Y Hà Nội', aliases=['ĐHYHN'])
    content = ('Một nửa trong top 20 người dẫn đầu kỳ thi bác sĩ nội trú chọn '
               'Sản phụ khoa ở Đại học Y Hà Nội, khiến chuyên ngành này ở trường chốt sổ sớm')
    key = event('university-event', legacy=legacy, content=content,
                description=content, mentions=['university'])
    assert search(location='Hà Nội') == []
    assert search(entity='Hà Nội') == []
    for query in ({'location': 'Hà Nội'}, {'entity': 'Ha Noi'}):
        rows = search(related=True, **query)
        assert [row['event_key'] for row in rows] == [key]
        result = EventResult.model_validate(rows[0])
        reason = next(r for r in result.relation_reasons if r.kind == 'entity_name_match')
        assert reason.label == 'Liên quan qua: Đại học Y Hà Nội'
        assert reason.via_entity.id == marker + 'university'
        assert reason.via_entity.type == 'ORGANIZATION'
        assert reason.post.platform_id == marker + 'university-event'
        assert any(r.kind == 'text_match' for r in result.relation_reasons)
    assert [r['event_key'] for r in search(entity='Đại học Y Hà Nội')] == [key]
    assert search(related=True, entity='ĐHYHN') == []
    assert all(r['relation_reasons'] == [] for r in search(entity='ĐHYHN'))


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('entity_type', ['PERSON', 'ORGANIZATION', 'LOCATION'])
def test_direct_normalized_names_aliases_and_partial_names(precision_graph, legacy, entity_type):
    entity, event, search, run, marker = precision_graph
    # Missing search_name, mixed whitespace, decomposed accents and array names.
    entity('target', ['Tên khác', '  Nguyễn\tVăn  An  '], entity_type,
           aliases=[' Bí danh  Đặc Biệt '])
    key = event('target-event', legacy=legacy, participants=['target'])
    for term in ('nguyen van an', 'Nguyễn Văn An', 'BI DANH DAC BIET'):
        assert [r['event_key'] for r in search(entity=term)] == [key]
        assert search(related=True, entity=term) == []
    assert search(entity='Văn An') == []
    assert [r['event_key'] for r in search(related=True, entity='Văn An')] == [key]
    if entity_type == 'LOCATION':
        assert [r['event_key'] for r in search(location='nguyen van an')] == [key]


@pytest.mark.parametrize('legacy', [False, True])
def test_related_text_only_and_combined_filters(precision_graph, legacy):
    entity, event, search, run, marker = precision_graph
    entity('hanoi', 'Hà Nội', 'LOCATION')
    entity('person', 'Nguyễn An', 'PERSON')
    entity('school', 'Đại học Y Hà Nội')
    text_key = event('text', legacy=legacy, content='Tin tại Hà Nội về Nguyễn An')
    mixed_key = event('mixed', legacy=legacy, participants=['person', 'school'])
    direct_key = event('direct', legacy=legacy, participants=['person', 'hanoi'])
    event('missing-person', legacy=legacy, participants=['school'])
    event('old', legacy=legacy, participants=['school', 'person'], age=48)
    query = dict(location='Hà Nội', entity='Nguyễn An')
    assert {r['event_key'] for r in search(**query)} == {direct_key}
    rows = search(related=True, **query)
    assert {r['event_key'] for r in rows} == {text_key, mixed_key}
    text = next(r for r in rows if r['event_key'] == text_key)
    assert {r['query_field'] for r in text['relation_reasons']} == {'location', 'entity'}
    assert all(r['kind'] == 'text_match' and r['via_entity'] is None
               for r in text['relation_reasons'])
    assert search(related=True, location='Hà Nội', entity='Không có') == []
    assert direct_key in {r['event_key'] for r in search(entity='Không có hoặc Nguyễn An')}


@pytest.mark.parametrize('legacy', [False, True])
def test_mixed_schema_post_scoping_and_event_participants(precision_graph, legacy):
    entity, event, search, run, marker = precision_graph
    entity('hanoi', 'Hà Nội', 'LOCATION')
    entity('school', 'Đại học Y Hà Nội')
    # One event represented in both schemas must still count as one event.
    single = event('single', legacy=legacy, mentions=['hanoi'])
    event('single', legacy=not legacy, mentions=['hanoi'])
    # Different events in the same post across schemas must count as multiple.
    event('multi-one', legacy=legacy, post_key='multi', mentions=['hanoi', 'school'],
          content='Đại học Y Hà Nội')
    event('multi-two', legacy=not legacy, post_key='multi', mentions=['hanoi', 'school'],
          content='Đại học Y Hà Nội')
    linked = event('linked', legacy=legacy)
    run('''MATCH (e:Event {event_key: $marker + 'linked'}),
                 (n:Entity {entity_id: $marker + 'hanoi'})
           CREATE (e)-[:HAS_PARTICIPANT]->(n)''')
    assert {r['event_key'] for r in search(location='Hà Nội')} == {single, linked}
    assert search(related=True, location='Hà Nội') == []
