from concurrent.futures import Future
from unittest.mock import Mock, patch

import pytest
import knowledge_pipeline as pipeline
import knowledge_persistence as persistence
import knowledge_settings as settings
from knowledge_relation_router import normalize_relation_routes


def test_config_defaults_and_unimplemented_module():
    assert {k for k, v in settings.KNOWLEDGE_MODULES.items() if v} == {'ENTITY_HIERARCHY'}
    with patch.dict(settings.KNOWLEDGE_MODULES, {'TEMPORAL_RELATION': True}):
        with pytest.raises(ValueError, match='chưa được triển khai'):
            pipeline.process_new_posts(Mock())


def test_detected_modules_normalized_without_events():
    result = normalize_relation_routes('text', {'entities': [], 'events': []}, {
        'detected_modules': ['ENTITY_HIERARCHY', 'INVALID', 'ENTITY_HIERARCHY', None],
        'event_routes': [], 'pair_routes': [],
    })
    assert result['detected_modules'] == ['ENTITY_HIERARCHY']


@pytest.mark.parametrize('skip', [True, False])
def test_persistence_records_detection_and_does_not_schedule_disabled_hierarchy(skip):
    tx = Mock()
    tx.run.return_value.single.return_value = {'post_count': 1}
    knowledge = {'entities': [{'type': 'LOCATION'}, {'type': 'ORGANIZATION'}],
                 'events': [], 'event_relations': []}
    with patch.object(persistence, 'upsert_entities', return_value={}), \
         patch.object(persistence, 'upsert_events') as events, \
         patch.object(persistence, 'upsert_event_relations') as relations:
        persistence.save_knowledge_tx(tx, 'test', '1', knowledge,
            classifier_decision='SKIPPED' if skip else 'DEEP',
            detected_modules=['ENTITY_HIERARCHY'], runnable_modules=set())
    calls = tx.run.call_args_list
    saved = next(c.kwargs for c in calls if 'SET p.modules_pending' in c.args[0])
    assert saved['modules'] == ([] if skip else ['ENTITY_HIERARCHY'])
    assert 'status' not in saved
    assert saved['router_version'] == (None if skip else settings.RELATION_ROUTER_PROMPT_VERSION)
    assert not any('hierarchy_status' in c.args[0] for c in calls)
    assert events.call_args.kwargs['replace_participants'] is False
    relations.assert_not_called()


@pytest.mark.parametrize('detected,enabled,should_run', [
    ([], True, False), (['ENTITY_HIERARCHY'], False, False),
    (['ENTITY_HIERARCHY'], True, True),
])
def test_hierarchy_requires_both_detection_and_configuration(detected, enabled, should_run):
    future = Future()
    future.set_result({'knowledge': {'entities': [{'type': 'LOCATION'}], 'events': [], 'event_relations': []},
        'relation_routes': {'detected_modules': detected, 'event_routes': [], 'pair_routes': []},
        'classification': {'should_deep_analyze': True}, 'classifier_decision': 'DEEP'})
    session = Mock()
    session.execute_write.return_value = {'entities': 1, 'events': 0, 'event_relations': 0}
    enrich, participant, relations, consolidate = Mock(), Mock(), Mock(), []
    with patch.dict(settings.KNOWLEDGE_MODULES, {'ENTITY_HIERARCHY': enabled}):
        result = pipeline._save_extracted_post(session,
            {'platform': 'test', 'post_id': '1', 'content': 'text'}, future,
            original_index=1, completed=1, total=1, enrich_locations_fn=enrich,
            extract_participants_fn=participant, extract_event_relations_fn=relations,
            mention_keys_out=consolidate)
    assert result == 'deep'
    assert enrich.called is should_run
    participant.assert_not_called()
    relations.assert_not_called()
    assert consolidate == []


def test_skipped_does_not_call_router_or_extractors():
    extract, router, participant, relations = Mock(), Mock(), Mock(), Mock()
    result = pipeline._extract_post(lambda _: {'should_deep_analyze': False}, extract,
        lambda c, k, p, i: k, router, 'test', '1', 'text', participant, relations)
    for fn in (extract, router, participant, relations):
        fn.assert_not_called()
    assert result['relation_routes']['detected_modules'] == []


def test_pipeline_calls_real_persistence_signature_for_skipped():
    session, tx = Mock(), Mock()
    tx.run.return_value.single.return_value = {'post_count': 1}
    session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(tx, *args, **kwargs)
    with patch.object(pipeline, '_load_posts', return_value=[{'platform': 'test', 'post_id': '1', 'content': 'text'}]), \
         patch.object(pipeline, 'create_knowledge_schema'), \
         patch.object(persistence, 'upsert_entities', return_value={}), \
         patch.object(persistence, 'upsert_events'):
        summary = pipeline.process_new_posts(session, classify_post_fn=lambda _: {'should_deep_analyze': False})
    assert summary['skipped'] == 1
    saved = next(c.kwargs for c in tx.run.call_args_list if 'SET p.modules_pending' in c.args[0])
    assert saved['modules'] == []
    assert 'status' not in saved
    classifier = next(c.kwargs for c in tx.run.call_args_list if 'classifier_decision' in c.kwargs)
    assert classifier['classifier_decision'] == 'SKIPPED'


def test_enabled_participant_runs_after_base_and_preserves_keys():
    from test_event_extraction_modules import base, CONTENT, participant
    from knowledge_validation import validate_knowledge
    knowledge = validate_knowledge(CONTENT, base(), 'test', '1')
    future = Future()
    future.set_result({'knowledge': knowledge,
        'relation_routes': {'detected_modules': ['PARTICIPANT_ROLE'], 'event_routes': [], 'pair_routes': []},
        'classification': {'should_deep_analyze': True}, 'classifier_decision': 'DEEP'})
    session = Mock()
    session.execute_write.return_value = {'entities': 1, 'events': 2, 'event_relations': 0}
    def enrich(content, value):
        import copy
        assert session.execute_write.call_count == 1
        enriched = copy.deepcopy(value)
        enriched['events'][0]['participants'] = [participant(value['entities'][0]['local_id'], role='ACTOR')]
        return enriched
    with patch.dict(settings.KNOWLEDGE_MODULES, {'PARTICIPANT_ROLE': True}):
        outcome = pipeline._save_extracted_post(session,
            {'platform': 'test', 'post_id': '1', 'content': CONTENT}, future,
            original_index=1, completed=1, total=1, extract_participants_fn=enrich)
    assert outcome == 'deep'
    assert session.execute_write.call_count == 2
    saved = session.execute_write.call_args.args[3]
    assert saved['events'][0]['participants']
    assert saved['events'][0]['mention_key'] == knowledge['events'][0]['mention_key']
    assert saved['events'][0]['event_key'] == knowledge['events'][0]['event_key']


@pytest.mark.parametrize('location_errors,organization_errors', [(0, 0), (1, 0), (0, 1)])
def test_entity_completion_requires_all_branches(location_errors, organization_errors):
    future = Future()
    future.set_result({'knowledge': {'entities': [{'type': 'LOCATION'}, {'type': 'ORGANIZATION'}],
        'events': [], 'event_relations': []},
        'relation_routes': {'detected_modules': ['ENTITY_HIERARCHY'], 'event_routes': [], 'pair_routes': []},
        'classification': {'should_deep_analyze': True}, 'classifier_decision': 'DEEP'})
    session = Mock()
    session.execute_write.return_value = {'entities': 2, 'events': 0, 'event_relations': 0}
    result = pipeline._save_extracted_post(session,
        {'platform': 'test', 'post_id': '1', 'content': 'text'}, future,
        original_index=1, completed=1, total=1,
        organization_context_fn=lambda *_: [],
        enrich_locations_fn=lambda *_: {'errors': location_errors},
        enrich_organizations_fn=lambda *_: {'errors': organization_errors})
    assert result == 'deep'
    completions = [c for c in session.execute_write.call_args_list if c.args[0] is persistence.mark_module_completed]
    assert bool(completions) == (location_errors == organization_errors == 0)


def test_module_output_and_completion_share_transaction():
    tx = Mock()
    knowledge = {'events': [], 'event_relations': []}
    with patch.object(persistence, 'upsert_event_relations') as write, \
         patch.object(persistence, 'mark_module_completed') as complete:
        persistence.save_module_knowledge_tx(tx, 'test', '1', knowledge, 'EVENT_RELATION')
        complete.assert_called_once_with(tx, 'test', '1', 'EVENT_RELATION')
        write.side_effect = RuntimeError('write failed')
        complete.reset_mock()
        with pytest.raises(RuntimeError):
            persistence.save_module_knowledge_tx(tx, 'test', '1', knowledge, 'EVENT_RELATION')
        complete.assert_not_called()
