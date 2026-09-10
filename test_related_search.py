from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

from backend.config import Settings
from backend.main import create_app
from backend.neo4j_repository import Neo4jRepository
from test_backend import PagingRepository, pagination_event


class RelatedRepository(PagingRepository):
    def search_related_events(self, **kwargs):
        return super().search_events(**kwargs)


def test_related_paging_and_cursor_isolation():
    repository = RelatedRepository([pagination_event(i) for i in range(1, 5)])
    query = dict(intent='search_events', location='Hà Nội', entity=None, hours=48)
    with TestClient(create_app(Settings(gemini_api_key=''), repository)) as client:
        payload = dict(query=query, limit=2)
        first = client.post('/api/search/related', json=payload).json()
        assert first['count'] == 2
        assert first['has_more']
        assert 'Sự kiện liên quan' in first['answer']
        second = client.post('/api/search/related', json={
            **payload, 'cursor': first['next_cursor'],
        }).json()
        assert second['start_index'] == 3
        assert not second['has_more']
        assert {r['event_key'] for r in first['results']}.isdisjoint(
            r['event_key'] for r in second['results'])
        assert client.post('/api/chat', json={
            'message': 'tiếp', 'cursor': first['next_cursor'],
        }).status_code == 400
        assert client.post('/api/search/related', json={
            **payload, 'query': {**query, 'hours': 24},
            'cursor': first['next_cursor'],
        }).status_code == 400
        direct = client.post('/api/chat', json={
            'message': 'sự kiện Hà Nội', 'limit': 2,
        }).json()
        for cursor in [direct['next_cursor'], 'invalid']:
            assert client.post('/api/search/related', json={
                **payload, 'cursor': cursor,
            }).status_code == 400


@pytest.mark.parametrize('location', [None, '', '   '])
def test_related_requires_location_or_entity(location):
    with TestClient(create_app(Settings(gemini_api_key=''), RelatedRepository())) as client:
        assert client.post('/api/search/related', json={
            'query': {'location': location, 'hours': 48},
        }).status_code == 422


def test_related_empty_and_database_failure():
    repository = RelatedRepository()
    with TestClient(create_app(Settings(gemini_api_key=''), repository)) as client:
        payload = {'query': {'location': 'Hà Nội', 'hours': 48}}
        response = client.post('/api/search/related', json=payload)
        assert response.status_code == 200
        assert response.json()['results'] == []
        assert response.json()['next_cursor'] is None
        repository.search_related_events = MagicMock(side_effect=ServiceUnavailable('offline'))
        assert client.post('/api/search/related', json=payload).status_code == 503


def test_repository_excludes_full_direct_set_before_merging_and_paging():
    repository = Neo4jRepository(Settings())
    repository.driver = MagicMock()
    session = repository.driver.session.return_value.__enter__.return_value
    duplicate = pagination_event(2)
    duplicate['post'] = {**duplicate['post'], 'platform_id': 'another-post'}
    session.run.return_value.data.side_effect = [
        [pagination_event(4)], [pagination_event(3)],
        [pagination_event(i) for i in range(1, 5)], [duplicate],
    ]
    rows = repository.search_related_events(
        location='Hà Nội', entity='Lan', hours=48, limit=1,
    )
    assert [r['event_key'] for r in rows] == ['event-02']
    assert len(rows[0]['sources']) == 2
    for call in session.run.call_args_list:
        assert call.kwargs['hours'] == 48
        assert {'field': 'entity', 'key': 'lan', 'search_key': 'lan'} in call.kwargs['terms']


def test_related_accepts_entity_only_and_preserves_reason_metadata():
    event = pagination_event(1)
    reason = {
        'kind': 'entity_name_match', 'query_field': 'entity', 'query_term': 'hà nội',
        'via_entity': {'id': 'university', 'name': 'Đại học Y Hà Nội', 'type': 'ORGANIZATION'},
        'evidence_field': 'entity.name', 'excerpt': None, 'relationship': None,
        'post': {'platform': 'facebook', 'platform_id': 'post-1'},
        'label': 'Liên quan qua: Đại học Y Hà Nội',
    }
    event['relation_reasons'] = [reason]
    with TestClient(create_app(Settings(gemini_api_key=''), RelatedRepository([event]))) as client:
        response = client.post('/api/search/related', json={
            'query': {'entity': ' Hà Nội ', 'hours': 48},
        })
        assert response.status_code == 200
        body = response.json()
        assert body['query']['entity'] == 'Hà Nội'
        assert body['query']['location'] is None
        assert body['results'][0]['relation_reasons'] == [reason]


@pytest.mark.parametrize('scope', [None, 'related_locations', 'related_events'])
@pytest.mark.parametrize('version', [None, 1])
def test_old_or_unversioned_cursors_require_new_search(scope, version):
    import base64
    import json

    query = dict(location='Hà Nội', hours=48)
    cursor = dict(query=query, returned=1, matched_entity_count=1,
                  posted_at='2026-08-01T00:00:00', event_key='event-1')
    if scope is not None:
        cursor['scope'] = scope
    if version is not None:
        cursor['version'] = version
    encoded = base64.urlsafe_b64encode(json.dumps(cursor).encode()).decode()
    with TestClient(create_app(Settings(gemini_api_key=''), RelatedRepository())) as client:
        for endpoint, payload in [
            ('/api/chat', {'message': 'tiếp', 'cursor': encoded}),
            ('/api/search/related', {'query': query, 'cursor': encoded}),
        ]:
            response = client.post(endpoint, json=payload)
            assert response.status_code == 400
            assert 'Hãy tìm kiếm lại' in response.json()['detail']


def test_related_merges_and_deduplicates_reasons_across_posts_and_schemas():
    from copy import deepcopy

    repository = Neo4jRepository(Settings())
    repository.driver = MagicMock()
    session = repository.driver.session.return_value.__enter__.return_value
    first = pagination_event(1)
    first['relation_reasons'] = [{
        'kind': 'entity_name_match', 'query_field': 'location', 'query_term': 'hà nội',
        'via_entity': {'id': 'university', 'name': 'Đại học Y Hà Nội', 'type': 'ORGANIZATION'},
        'evidence_field': 'entity.name',
    }]
    second = deepcopy(first)
    second['post']['platform_id'] = 'another-post'
    second['relation_reasons'] = [{
        'kind': 'text_match', 'query_field': 'location', 'query_term': 'ha noi',
        'evidence_field': 'post.content', 'text': 'x' * 120 + ' Đại học Y Hà Nội ' + 'y' * 120,
    }]
    session.run.return_value.data.side_effect = [[], [], [first, second], [deepcopy(first)]]
    rows = repository.search_related_events(location='Hà Nội', entity=None, hours=48, limit=10)
    assert len(rows) == 1
    assert len(rows[0]['sources']) == 2
    reasons = rows[0]['relation_reasons']
    assert len(reasons) == 2
    text = next(reason for reason in reasons if reason['kind'] == 'text_match')
    assert text['via_entity'] is None
    assert 'Hà Nội' in text['excerpt']
    assert len(text['excerpt']) < 200
    assert text['post']['platform_id'] == 'another-post'
