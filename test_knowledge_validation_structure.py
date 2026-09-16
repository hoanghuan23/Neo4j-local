from unittest.mock import Mock

import pytest

from knowledge_extraction import extract_knowledge
from knowledge_validation import validate_knowledge
from test_extract_entities import KnowledgeValidationTests as _Fixtures

entity = _Fixtures.entity
event = _Fixtures.event
participant = _Fixtures.participant
del _Fixtures


def test_namespace_remapping_aliases_duplicates_and_references(caplog):
    caplog.set_level('DEBUG', logger='knowledge_settings')
    raw = {
        'entities': [entity('same', 'Alice'), entity('alias', 'Alice'),
                     entity('same', 'Bob'), {}],
        'events': [event('same', 'OTHER', 'First occurrence', [
            participant('alias', role='ACTOR'), participant('missing'),
            participant('missing', participant_text='a witness'), {},
        ]), event('second', 'OTHER', 'Second occurrence'),
            event('same', 'OTHER', 'Duplicate occurrence')],
        'event_relations': [
            {'source_event_id': 'same', 'target_event_id': 'second',
             'type': 'RELATED_TO', 'evidence_text': 'First occurrence. Second occurrence'},
            {'source_event_id': 'alias', 'target_event_id': 'second',
             'type': 'RELATED_TO', 'evidence_text': 'First occurrence'},
        ],
    }
    result = validate_knowledge('First occurrence. Second occurrence. Duplicate occurrence', raw, 'test', 'p1')
    assert [x['local_id'] for x in result['entities']] == ['e1']
    assert [x['local_id'] for x in result['events']] == ['ev1', 'ev2']
    assert result['events'][0]['participants'] == [
        dict(participant('e1', role='ACTOR'), participant_scope=None)
    ]
    assert len(result['event_relations']) == 1
    assert result['event_relations'][0]['source_event_id'] == 'ev1'
    assert result['event_relations'][0]['target_event_id'] == 'ev2'
    assert 'post=p1' in caplog.text
    assert 'missing_or_duplicate_id' in caplog.text
    assert 'dangling_entity_reference' in caplog.text


def test_event_limit_removes_dependent_relation():
    events = [event(str(i), 'OTHER', f'Occurrence {i}') for i in range(6)]
    result = validate_knowledge(' '.join(e['evidence_text'] for e in events), {
        'events': events,
        'event_relations': [{'source_event_id': '0', 'target_event_id': '5',
                            'type': 'RELATED_TO', 'evidence_text': 'Occurrence 0'}],
    })
    assert len(result['events']) == 5
    assert result['event_relations'] == []


def test_empty_collections_and_no_structural_llm_retry():
    for invalid in (None, {}, 'null', 12):
        model = Mock(return_value={'entities': invalid, 'events': invalid, 'event_relations': invalid})
        with pytest.raises(ValueError):
            extract_knowledge('Không có dữ liệu.', call_model=model)
        model.assert_called_once()


def test_invalid_participant_is_local_and_extra_fields_removed():
    raw_event = event('custom', 'OTHER', 'Occurrence', [
        participant(participant_text='a witness', participant_scope='INVALID'),
        participant(participant_text='null'),
        participant(participant_text='someone', role='INVALID'),
        participant(participant_text='someone', confidence=2),
    ])
    raw_event['extra'] = 'ignored'
    result = validate_knowledge('Occurrence', {'events': [raw_event], 'extra': True})
    saved = result['events'][0]
    assert len(saved['participants']) == 1
    assert saved['participants'][0]['participant_scope'] == 'POST_LOCAL'
    assert 'extra' not in saved and 'extra' not in result


def test_extraction_removes_legacy_participants_without_mutating_response():
    raw_event = event('ev1', 'OTHER', 'Occurrence')
    raw_event.update(title='Một sự việc cụ thể được mô tả trực tiếp trong nội dung bài viết', participants=None)
    model = Mock(return_value={'entities': [], 'events': [raw_event]})
    assert 'participants' not in extract_knowledge('Occurrence', call_model=model)['events'][0]
    model.assert_called_once()
    assert raw_event['participants'] is None


def test_structured_placeholders_are_not_stringified():
    raw = {'entities': [entity('e1', {})], 'events': [
        event('ev1', 'OTHER', 'Occurrence', [participant(participant_text={})]),
        event('ev2', 'OTHER', {}),
    ]}
    result = validate_knowledge('Occurrence', raw)
    assert result['entities'] == []
    assert len(result['events']) == 1
    assert result['events'][0]['participants'] == []
