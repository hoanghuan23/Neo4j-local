"""Contracts for the separate occurrence, participant and relation stages."""

import copy
from unittest.mock import Mock, patch

import pytest

import knowledge_pipeline as pipeline
from knowledge_relations.participant_role import extract_participants
from knowledge_relations.event_relation import extract_event_relations
from knowledge_settings import EVENT_RELATION_TYPES
from knowledge_validation import validate_knowledge
from tests.test_knowledge_validation_structure import entity, event, participant


CONTENT = 'Alice performed the first occurrence. This caused the second occurrence.'


def base():
    return {
        'entities': [entity('person', 'Alice')],
        'events': [event('first', 'OTHER', 'Alice performed the first occurrence'),
                   event('second', 'OTHER', 'This caused the second occurrence')],
    }


def test_participants_validate_before_keys_and_remap_identities():
    knowledge = base()
    before = copy.deepcopy(knowledge)
    model = Mock(return_value={'events': [
        {'event_id': 'first', 'participants': [
            participant('person', role='ACTOR'),
            participant(participant_text='công an', participant_scope='GLOBAL_ROLE'),
            participant(participant_text='một người', participant_scope='POST_LOCAL'),
            participant('missing'), participant('person', role='INVALID'),
        ]},
        {'event_id': 'unknown', 'participants': [participant('person')]},
    ]})
    enriched = extract_participants(CONTENT, knowledge, model)
    assert knowledge == before
    assert enriched['events'][1]['participants'] == []
    final = validate_knowledge(CONTENT, enriched, 'test', 'post')
    participants = final['events'][0]['participants']
    assert participants[0]['entity_id'] == 'e1'
    assert participants[0]['role'] == 'ACTOR'
    assert participants[0]['participant_scope'] is None
    assert [p['participant_scope'] for p in participants[1:]] == ['GLOBAL_ROLE', 'POST_LOCAL']
    without = validate_knowledge(CONTENT, before, 'test', 'post')
    assert final['events'][0]['mention_key'] != without['events'][0]['mention_key']
    assert final['events'][0]['event_key'] != without['events'][0]['event_key']
    model.assert_called_once()


@pytest.mark.parametrize('relation_type,marker', [
    ('APPROVES', 'approved'), ('CAUSES', 'caused'), ('ENABLES', 'enabled'),
    ('PRECEDES', 'before'), ('RELATED_TO', 'linked'),
])
def test_all_relation_types_and_invalid_items(relation_type, marker):
    content = CONTENT + f' The first occurrence {marker} the second occurrence.'
    evidence = f'The first occurrence {marker} the second occurrence'
    valid = dict(source_event_id='first', target_event_id='second',
                 type=relation_type, evidence_text=evidence)
    model = Mock(return_value={'event_relations': [
        valid, dict(valid), dict(valid, target_event_id='first'),
        dict(valid, target_event_id='unknown'), dict(valid, evidence_text='invented'),
        dict(valid, type='INVALID'), None,
    ]})
    knowledge = base()
    enriched = extract_event_relations(content, knowledge, model)
    assert 'event_relations' not in knowledge
    final = validate_knowledge(content, enriched, 'test', 'post')
    assert final['event_relations'] == [dict(valid, source_event_id='ev1', target_event_id='ev2')]
    schema = model.call_args.args[1]
    assert set(schema['properties']['event_relations']['items']['properties']['type']['enum']) == EVENT_RELATION_TYPES
    model.assert_called_once()


def test_causal_relation_without_explicit_marker_is_dropped():
    model = Mock(return_value={'event_relations': [dict(
        source_event_id='first', target_event_id='second', type='CAUSES',
        evidence_text='Alice performed the first occurrence',
    )]})
    result = extract_event_relations(CONTENT, base(), model)
    assert validate_knowledge(CONTENT, result)['event_relations'] == []


def test_empty_and_single_event_avoid_model_calls():
    model = Mock()
    empty = {'entities': [], 'events': []}
    assert extract_participants('', empty, model) == empty
    assert extract_event_relations('', empty, model)['event_relations'] == []
    single = base()
    single['events'] = single['events'][:1]
    assert extract_event_relations(CONTENT, single, model)['event_relations'] == []
    model.assert_not_called()


@pytest.mark.parametrize('function,key', [
    (extract_participants, 'events'), (extract_event_relations, 'event_relations'),
])
@pytest.mark.parametrize('raw', [None, {}, [], {'wrong': []}])
def test_invalid_model_envelope_raises(function, key, raw):
    model = Mock(return_value=raw)
    with pytest.raises(ValueError):
        function(CONTENT, base(), model)
    model.assert_called_once()


@pytest.mark.parametrize('function', [extract_participants, extract_event_relations])
def test_model_failure_propagates(function):
    with pytest.raises(RuntimeError, match='model unavailable'):
        function(CONTENT, base(), Mock(side_effect=RuntimeError('model unavailable')))


def test_pipeline_stage_order_and_stable_final_keys():
    order = []
    knowledge = base()
    participant_model = Mock(return_value={'events': [{'event_id': 'first', 'participants': [participant('person', role='ACTOR')]}]})
    relation_model = Mock(return_value={'event_relations': [dict(
        source_event_id='first', target_event_id='second', type='CAUSES',
        evidence_text='This caused the second occurrence',
    )]})

    def record(name, function):
        def invoke(*args):
            order.append(name)
            return function(*args)
        return invoke

    result = pipeline._extract_post(
        record('classifier', lambda _: {'should_deep_analyze': True}),
        record('extraction', lambda _: knowledge),
        record('validation', validate_knowledge),
        'test', 'post', CONTENT,
        record('participants', lambda c, k: extract_participants(c, k, participant_model)),
        record('relations', lambda c, k: extract_event_relations(c, k, relation_model)),
    )
    assert order == ['classifier', 'extraction', 'validation']
    expected = base()
    assert result['knowledge'] == validate_knowledge(CONTENT, expected, 'test', 'post')
    participant_model.assert_not_called()
    relation_model.assert_not_called()


@pytest.mark.parametrize('enabled,deep', [(False, True), (True, False)])
def test_skip_and_entity_only_do_not_call_modules(enabled, deep):
    participants, relations = Mock(), Mock()
    with patch.object(pipeline, 'KNOWLEDGE_PIPELINE_ENABLED', enabled):
        pipeline._extract_post(
            lambda _: {'should_deep_analyze': deep}, lambda _: base(),
            validate_knowledge, 'test', 'post', CONTENT, participants, relations,
        )
    participants.assert_not_called()
    relations.assert_not_called()


@pytest.mark.parametrize('failed_stage', ['participants', 'relations'])
def test_module_failure_keeps_committed_base(failed_stage):
    participants = Mock(side_effect=lambda _c, k: k)
    relations = Mock(side_effect=lambda _c, k: k)
    failing = participants if failed_stage == 'participants' else relations
    failing.side_effect = ValueError('invalid model response')
    session = Mock()
    session.execute_write.return_value = {'entities': 1, 'events': 2, 'event_relations': 0}
    module = 'PARTICIPANT_ROLE' if failed_stage == 'participants' else 'EVENT_RELATION'
    with patch.dict(pipeline.KNOWLEDGE_MODULES, {module: True}), patch.object(pipeline, '_load_posts', return_value=[dict(platform='test', post_id='post', content=CONTENT)]), \
         patch.object(pipeline, 'create_knowledge_schema'):
        summary = pipeline.process_new_posts(
            session, extract_knowledge_fn=lambda _: base(),
            classify_post_fn=lambda _: {'should_deep_analyze': True},
            extract_participants_fn=participants, extract_event_relations_fn=relations,
        )
    assert summary['deep'] == 1
    failing.assert_called_once()
    assert session.execute_write.call_count == 1
    assert session.execute_write.call_args.args[0] is pipeline.save_knowledge_tx
    assert session.execute_write.call_args.kwargs['runnable_modules'] == {module}


def test_legacy_consolidation_imports_point_to_new_implementation():
    import knowledge_consolidation as legacy
    from knowledge_relations import event_hierarchy
    assert legacy.consolidate_pending_mentions is event_hierarchy.consolidate_pending_mentions
    assert legacy._merge_events is event_hierarchy._merge_events


def test_entrypoint_passes_shared_model_to_both_modules():
    from scripts import extract_entities as entrypoint
    model = Mock()
    with patch.object(entrypoint, '_process_new_posts', return_value={}) as process, \
         patch.object(entrypoint, 'extract_participants') as participants, \
         patch.object(entrypoint, 'extract_event_relations') as relations:
        entrypoint.process_new_posts(Mock(), call_model=model)
        kwargs = process.call_args.kwargs
        kwargs['extract_participants_fn'](CONTENT, base())
        kwargs['extract_event_relations_fn'](CONTENT, base())
    participants.assert_called_once_with(CONTENT, base(), call_model=model)
    relations.assert_called_once_with(CONTENT, base(), call_model=model)


def test_new_model_stages_are_identified_in_usage_logs():
    from knowledge_gemini import _stage_for_schema
    from knowledge_settings import PARTICIPANT_EXTRACTION_SCHEMA, EVENT_RELATION_SCHEMA
    assert _stage_for_schema(PARTICIPANT_EXTRACTION_SCHEMA) == 'participant_extraction'
    assert _stage_for_schema(EVENT_RELATION_SCHEMA) == 'event_relation'


def test_participant_bad_list_fails_but_bad_ids_are_ignored():
    with pytest.raises(ValueError):
        extract_participants(CONTENT, base(), Mock(return_value={'events': [
            {'event_id': 'first', 'participants': None},
        ]}))
    model = Mock(return_value={'events': [
        {'event_id': [], 'participants': []},
        {'event_id': 'first', 'participants': [participant('person', role='ACTOR')]},
        {'event_id': 'first', 'participants': []},
    ]})
    result = extract_participants(CONTENT, base(), model)
    assert result['events'][0]['participants'] == [participant('person', role='ACTOR')]
