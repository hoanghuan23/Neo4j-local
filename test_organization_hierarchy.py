import os
from uuid import uuid4

import pytest

from knowledge_relations.organization_hierarchy import (
    identity, choose, extract_context, propose, resolve_entity, persist_edges,
)


def node(id, name, type='ORGANIZATION', parents=()):
    return {'node_id': id, 'name': name, 'normalized_name': name.lower(),
            'aliases': [], 'type': type, 'parents': list(parents)}


@pytest.mark.parametrize('name', ['Công an TP.HCM', 'Công an TP. Hồ Chí Minh', 'Công an Thành phố Hồ Chí Minh'])
def test_city_variants(name):
    assert identity(name) == identity('Công an Thành phố Hồ Chí Minh')


def test_ambiguous_and_context_conflict():
    a = node('a', 'Công an xã Bình Minh', parents=[{'relationship': 'JURISDICTION', 'node_id': 'x'}])
    b = node('b', a['name'], parents=[{'relationship': 'JURISDICTION', 'node_id': 'y'}])
    assert choose({'name': a['name']}, [a, b])[1] == 'AMBIGUOUS_OR_CONFLICT'
    assert choose({'name': a['name']}, [a, b], [('JURISDICTION', 'y')])[0] == b
    assert choose({'name': a['name']}, [a], [('JURISDICTION', 'y')])[0] is None
    assert choose({'name': 'Phòng Cảnh sát Công an xã Bình Minh'}, [a])[1] == 'CREATE'


def test_content_validation():
    entities = [{'local_id': 'a', 'name': 'Đơn vị A', 'type': 'ORGANIZATION'},
                {'local_id': 'b', 'name': 'Tổ chức B', 'type': 'ORGANIZATION'}]
    text = 'Đơn vị A thuộc Tổ chức B'
    edge = {'source_entity_id': 'a', 'target_entity_id': 'b', 'relationship': 'SUBORDINATE_TO', 'evidence_text': text}
    assert len(extract_context(text, {'entities': entities}, lambda *_: {'relations': [edge]})) == 1
    for bad in ({'evidence_text': 'invented'}, {'relationship': 'JURISDICTION'}, {'target_entity_id': 'missing'}):
        assert extract_context(text, {'entities': entities}, lambda *_: {'relations': [{**edge, **bad}]}) == []


def test_rules_require_vietnam_and_specific_area():
    graph = [node('vn', 'Việt Nam', 'LOCATION'), node('hp', 'Hải Phòng', 'LOCATION',
        [{'relationship': 'PART_OF', 'node_id': 'vn'}]), node('bo', 'Bộ Công an'), node('ca', 'Công an Hải Phòng')]
    knowledge = {'entities': [{'local_id': 'a', 'name': 'Công an Hải Phòng', 'type': 'ORGANIZATION'}]}
    result = propose(graph, knowledge, [])
    assert {(e['relationship'], e['target_node_id']) for e in result['edges']} == {('JURISDICTION', 'hp'), ('SUBORDINATE_TO', 'bo')}
    graph[1]['parents'] = []
    assert all(e['relationship'] != 'SUBORDINATE_TO' for e in propose(graph, knowledge, [])['edges'])


@pytest.mark.skipif(os.getenv('RUN_NEO4J_INTEGRATION') != '1', reason='requires Neo4j')
def test_transactional_identity_and_edges_preserve_manual_data():
    from neo4j import GraphDatabase
    from knowledge_settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from knowledge_extraction import prepare_entity
    marker = 'org-test-' + uuid4().hex
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            tx = session.begin_transaction()
            try:
                row = tx.run('''CREATE (p:Post {platform:$m, platform_id:$m})
                    CREATE (a:Entity {type:'ORGANIZATION', name:$name, normalized_name:$name})
                    CREATE (b:Entity {type:'ORGANIZATION', name:$m, normalized_name:$m})
                    CREATE (c:Entity {type:'ORGANIZATION', name:$m, normalized_name:$m})
                    CREATE (a)-[r:SUBORDINATE_TO {manual:true}]->(b)
                    RETURN elementId(a) AS a, elementId(b) AS b, elementId(c) AS c''',
                    m=marker, name='Công an Thành phố Hồ Chí Minh '+marker).single()
                entity = {'local_id': 'a', 'name': 'Công an TP.HCM '+marker,
                          'canonical_name': 'Công an TP.HCM '+marker, 'type': 'ORGANIZATION', 'resolution_confidence': 'HIGH'}
                prepared = prepare_entity(entity)
                for _ in range(2):
                    resolved = resolve_entity(tx, marker, marker, entity, prepared)
                    assert resolved['node_id'] == row['a']
                assert tx.run('MATCH (p:Post {platform:$m})-[:MENTIONS]->(n) RETURN count(n) AS c', m=marker).single()['c'] == 1
                # A second node with the same stored key must not receive participants.
                tx.run("CREATE (:Entity {type:'ORGANIZATION', normalized_name:$name, name:$other})",
                       name=resolved['normalized_name'], other=marker+' unrelated').consume()
                from knowledge_persistence import upsert_events
                event = {'event_key': marker, 'mention_key': marker, 'type': 'OTHER',
                         'description': marker, 'evidence_text': marker, 'status': 'COMPLETED',
                         'time_expression': None, 'confidence': 1.0,
                         'participants': [{'entity_id': 'a', 'role': 'ACTOR', 'confidence': 1.0}]}
                upsert_events(tx, marker, marker, [event], {'a': resolved})
                participants = tx.run("""MATCH (m:EventMention {mention_key:$m})-[:HAS_PARTICIPANT]->(e)
                    RETURN elementId(e) AS id""", m=marker).data()
                assert participants == [{'id': row['a']}]
                edge = {'source_node_id': row['a'], 'target_node_id': row['b'], 'relationship': 'SUBORDINATE_TO',
                        'source': 'RULE', 'rule_id': 'test', 'evidence_text': None}
                assert persist_edges(tx, [edge], marker, marker) == []
                assert persist_edges(tx, [{**edge, 'target_node_id': row['c']}], marker, marker)
                assert persist_edges(tx, [{**edge, 'source_node_id': row['b'], 'target_node_id': row['a']}], marker, marker)
                properties = tx.run('MATCH ()-[r:SUBORDINATE_TO]->() WHERE startNode(r) IS NOT NULL AND elementId(startNode(r))=$a RETURN properties(r) AS p', a=row['a']).single()['p']
                assert properties == {'manual': True}
            finally:
                tx.rollback()


def test_enrichment_failure_keeps_base_and_sets_failed():
    from unittest.mock import Mock
    from knowledge_relations.organization_hierarchy import enrich_organization_hierarchy
    session = Mock()
    knowledge = {'entities': [{'local_id': 'a', 'name': 'A', 'type': 'ORGANIZATION'},
                              {'local_id': 'b', 'name': 'B', 'type': 'ORGANIZATION'}]}
    result = enrich_organization_hierarchy(session, 'x', 'y', 'A thuộc B', knowledge,
                                          call_model=Mock(side_effect=RuntimeError('offline')))
    assert result == {'errors': 1}
    assert "'FAILED'" in session.run.call_args.args[0]
    session.execute_write.assert_not_called()


def test_pipeline_orders_context_base_location_organization():
    from concurrent.futures import Future
    from unittest.mock import Mock
    from knowledge_pipeline import _save_extracted_post
    order = []
    future = Future()
    knowledge = {'entities': [{'local_id': 'a', 'name': 'Công an Hải Phòng', 'type': 'ORGANIZATION'},
                              {'local_id': 'b', 'name': 'Hải Phòng', 'type': 'LOCATION'}], 'events': [], 'event_relations': []}
    future.set_result({'knowledge': knowledge, 'relation_routes': {'event_routes': [], 'pair_routes': []},
                       'classification': {'should_deep_analyze': True}, 'classifier_decision': 'DEEP'})
    session = Mock()
    session.execute_write.side_effect = lambda *_: order.append('base') or {'entities': 2, 'events': 0, 'event_relations': 0}
    result = _save_extracted_post(session, {'platform': 'x', 'post_id': 'y', 'content': 'text'}, future,
        original_index=1, completed=1, total=1,
        organization_context_fn=lambda *_: order.append('context') or [],
        enrich_locations_fn=lambda *_: order.append('location') or {},
        enrich_organizations_fn=lambda *_: order.append('organization') or {})
    assert result == 'deep'
    assert order == ['context', 'base', 'location', 'organization']


def test_preview_does_not_write_and_reports_conflict():
    from unittest.mock import Mock, patch
    from knowledge_relations.organization_hierarchy import enrich_organization_hierarchy
    graph = [node('a', 'A', parents=[{'relationship': 'SUBORDINATE_TO', 'node_id': 'b'}]), node('b', 'B'), node('c', 'C')]
    knowledge = {'entities': [{'local_id': 'a', 'name': 'A', 'type': 'ORGANIZATION'},
                              {'local_id': 'c', 'name': 'C', 'type': 'ORGANIZATION'}],
                 'organization_context': [{'source_entity_id': 'a', 'target_entity_id': 'c',
                    'relationship': 'SUBORDINATE_TO', 'evidence_text': 'A thuộc C'}]}
    session = Mock()
    with patch('knowledge_relations.organization_hierarchy.load_graph', return_value=graph):
        result = enrich_organization_hierarchy(session, 'x', 'y', 'A thuộc C', knowledge, preview=True)
    assert result['reviews']
    assert result['edges'] == []
    session.run.assert_not_called()
    session.execute_write.assert_not_called()


def test_retry_uses_saved_input_and_restores_resolution_before_enrichment():
    import json
    from unittest.mock import Mock, patch
    from preview_organization_content import run
    knowledge = {'entities': [{'local_id': 'a', 'name': 'A', 'type': 'ORGANIZATION'}], 'events': []}
    session = Mock()
    session.run.return_value = [{'platform': 'x', 'post_id': 'y', 'content': 'A', 'snapshot': json.dumps(knowledge)}]
    tx = Mock()
    session.execute_write.side_effect = lambda fn, *args: fn(tx, *args)
    with patch('knowledge_persistence.upsert_entities', return_value={'a': {}}) as resolve, \
         patch('knowledge_persistence.upsert_events') as events, \
         patch('preview_organization_content.enrich_organization_hierarchy', return_value={'reviews': []}) as enrich:
        result = run(session, retry=True)
    resolve.assert_called_once()
    events.assert_called_once()
    assert enrich.call_args.kwargs['preview'] is False
    assert result[0]['post_id'] == 'y'
    assert "'PENDING','FAILED','NEEDS_REVIEW'" in session.run.call_args.args[0]
