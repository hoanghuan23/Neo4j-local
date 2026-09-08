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
        assert 'các địa điểm thuộc Hà Nội' in first['answer']
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
def test_related_requires_location(location):
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
        assert call.kwargs['entity_terms'][0]['key'] == 'lan'
