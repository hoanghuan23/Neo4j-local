from unittest.mock import Mock

from knowledge_relations.event_hierarchy import summarize_event
from knowledge_relation_router import _compact_knowledge


TITLE = 'Một sự việc cụ thể được mô tả trực tiếp trong nội dung bài viết'


def mention(key='m1', **changes):
    return dict(dict(mention_key=key, title=TITLE, description='Occurrence',
                     evidence_text='Occurrence', type='OTHER', status='COMPLETED',
                     time_expression=None, title_needs_backfill=False), **changes)


def session_for(rows):
    session = Mock()
    session.run.side_effect = lambda query, **kwargs: (
        rows if 'RETURN mention.mention_key' in query else Mock()
    )
    return session


def test_identical_mentions_reuse_source_without_llm():
    session = session_for([mention(), mention('m2')])
    model = Mock()
    assert summarize_event(session, 'event', model)
    model.assert_not_called()
    saved = session.run.call_args.kwargs
    assert saved['description'] == 'Occurrence'
    assert saved['source_keys'] == ['m1']


def test_batch_cache_skips_unchanged_but_invalidates_changed_sources():
    rows = [mention(), mention('m2', description='Another occurrence')]
    session = session_for(rows)
    model = Mock(return_value=dict(title=TITLE, description='Summary', type='OTHER',
                                   status='COMPLETED', source_mention_keys=['m1', 'm2']))
    cache = {}
    assert summarize_event(session, 'event', model, summary_cache=cache)
    writes = sum('SET event.title' in call.args[0] for call in session.run.call_args_list)
    assert not summarize_event(session, 'event', model, summary_cache=cache)
    assert sum('SET event.title' in call.args[0] for call in session.run.call_args_list) == writes
    model.assert_called_once()
    rows[1]['status'] = 'ALLEGED'
    assert summarize_event(session, 'event', model, summary_cache=cache)
    assert model.call_count == 2


def test_conflicting_status_is_not_deduplicated():
    session = session_for([mention(), mention('m2', status='ALLEGED')])
    model = Mock(return_value=dict(title=TITLE, description='Summary', type='OTHER',
                                   status='ALLEGED', source_mention_keys=['m1', 'm2']))
    summarize_event(session, 'event', model)
    model.assert_called_once()


def test_failed_persistence_does_not_cache_summary():
    session = session_for([mention()])
    original = session.run.side_effect

    def fail(query, **kwargs):
        if 'SET event.title' in query:
            raise RuntimeError('write failed')
        return original(query, **kwargs)

    session.run.side_effect = fail
    cache = {}
    import pytest
    with pytest.raises(RuntimeError, match='write failed'):
        summarize_event(session, 'event', Mock(), summary_cache=cache)
    assert cache == {}


def test_router_payload_retains_distinct_evidence_aliases_and_roles():
    knowledge = {'entities': [dict(local_id='e1', name='A', canonical_name='A', type='PERSON'),
                              dict(local_id='e2', name='B', canonical_name='Bee', type='PERSON')],
                 'events': [dict(local_id='ev1', type='OTHER', title=TITLE,
                                 description='Occurrence', evidence_text='Different source',
                                 time_expression='today', participants=[dict(
                                     entity_id='e1', participant_text=None, participant_scope=None,
                                     role='ACTOR', confidence=.9)])]}
    compact = _compact_knowledge(knowledge)
    assert 'canonical_name' not in compact['entities'][0]
    assert compact['entities'][1]['canonical_name'] == 'Bee'
    saved = compact['events'][0]
    assert 'title' not in saved
    assert saved['evidence_text'] == 'Different source'
    assert saved['time_expression'] == 'today'
    assert saved['participants'] == [dict(entity_id='e1', role='ACTOR')]
    assert knowledge['events'][0]['title'] == TITLE


def test_match_payload_keeps_distinct_sources_and_all_candidates():
    import json
    from knowledge_relations.event_hierarchy import _resolve_prompt
    current = mention()
    candidates = [dict(event_key='candidate1', type='OTHER', status='COMPLETED',
                       description='Occurrence',
                       descriptions=['Occurrence', 'Distinct detail', 'Distinct detail'],
                       score_components={'total': .9}, retrieval_score=.9),
                  dict(event_key='candidate2', type='OTHER', status='ALLEGED',
                       description='Conflicting account', descriptions=[])]
    prompt = _resolve_prompt(current, candidates)
    payload = json.loads(prompt.split('Dữ liệu:\n', 1)[1])
    assert len(payload['candidates']) == 2
    assert payload['candidates'][0]['descriptions'] == ['Distinct detail']
    assert payload['candidates'][1]['status'] == 'ALLEGED'
    assert 'evidence_text' not in payload['mention']
    assert 'semantic_score_components' not in prompt
    assert candidates[0]['descriptions'] == ['Occurrence', 'Distinct detail', 'Distinct detail']
    current['evidence_text'] = 'Separate original evidence'
    changed = json.loads(_resolve_prompt(current, candidates).split('Dữ liệu:\n', 1)[1])
    assert changed['mention']['evidence_text'] == 'Separate original evidence'
