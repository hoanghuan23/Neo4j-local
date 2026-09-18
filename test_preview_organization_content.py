from unittest.mock import MagicMock, patch

import pytest

from preview_organization_content import save_content


def test_empty_content_never_touches_database():
    session = MagicMock()
    with pytest.raises(ValueError):
        save_content(session, '  ', MagicMock())
    assert not session.mock_calls


def test_import_writes_content_and_knowledge_in_same_transaction():
    session = MagicMock()
    tx = MagicMock()
    session.execute_write.side_effect = lambda fn: fn(tx)
    knowledge = {'entities': [], 'events': [], 'event_relations': []}
    with patch('knowledge_extraction.extract_knowledge', return_value=knowledge), \
         patch('knowledge_persistence.create_knowledge_schema'), \
         patch('knowledge_persistence.save_knowledge_tx', return_value={'events': 0}) as save:
        first = save_content(session, 'Một nội dung thử nghiệm.', MagicMock())
        second = save_content(session, 'Một nội dung thử nghiệm.', MagicMock())
    assert first['post_id'] == second['post_id']
    assert first['mode'] == 'saved_to_neo4j'
    assert save.call_args.args[0] is tx
    assert save.call_args.args[1:3] == ('manual', first['post_id'])
    params = tx.run.call_args.kwargs
    assert params['content'] == 'Một nội dung thử nghiệm.'
    assert 'knowledge_analysis' in tx.run.call_args.args[0]
    assert 'knowledge_relation_routes' not in tx.run.call_args.args[0]


def test_ambiguous_location_keeps_source_local_identity():
    from knowledge_persistence import _merge_entity
    tx = MagicMock()
    tx.run.return_value.single.side_effect = [None, {
        'normalized_name': 'thanh xuân', 'entity_type': 'LOCATION', 'node_id': 'local-node'
    }]
    entity = {'name': 'Thanh Xuân', 'canonical_name': 'Thanh Xuân',
              'type': 'LOCATION', 'resolution_confidence': 'HIGH'}
    result = _merge_entity(tx, 'manual', 'post-1', entity)
    assert result['node_id'] == 'local-node'
    assert 'location_mention_key' in tx.run.call_args.args[0]
    assert "resolution_status='NEEDS_REVIEW'" in tx.run.call_args.args[0]
    assert tx.run.call_args.kwargs['post_id'] == 'post-1'


def test_location_run_saves_by_default(monkeypatch):
    import preview_location_content as location
    import io
    monkeypatch.setattr('sys.argv', ['preview_location_content.py'])
    monkeypatch.setattr('sys.stdin', io.StringIO(''))
    with patch.object(location, 'GeminiKnowledgeCaller') as caller, \
         patch('neo4j.GraphDatabase.driver') as driver, \
         patch('preview_organization_content.save_content', return_value={'mode': 'saved_to_neo4j'}) as save:
        assert location.main() == 0
    assert save.call_args.args[1] == location.CONTENT
    caller.return_value.close.assert_called_once()
    driver.assert_called_once()
