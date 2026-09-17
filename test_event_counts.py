from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.neo4j_repository import Neo4jRepository
from test_backend import pagination_event


@pytest.mark.parametrize("related", [False, True])
@pytest.mark.parametrize("size", [0, 3])
def test_api_total_counts_distinct_events_before_pagination(related, size):
    repository = Neo4jRepository(Settings())
    repository.driver = MagicMock()
    repository.connect = MagicMock()
    repository.close = MagicMock()
    session = repository.driver.session.return_value.__enter__.return_value
    rows = [pagination_event(i) for i in range(1, size + 1)]
    legacy = deepcopy(rows)
    for row in legacy:
        row["post"]["platform_id"] += "-another-post"
    # Related results contain a direct event which must not count in the total.
    direct = pagination_event(4)
    batches = [[direct], [], rows + [direct], legacy] if related else [rows, legacy]
    session.run.return_value.data.side_effect = lambda: deepcopy(
        batches[(session.run.call_count - 1) % len(batches)]
    )
    endpoint = "/api/search/related" if related else "/api/chat"
    payload = ({"query": {"location": "Hà Nội", "hours": 48}} if related else
               {"message": "sự kiện Hà Nội 48h qua"})
    with TestClient(create_app(Settings(gemini_api_key=""), repository)) as client:
        first = client.post(endpoint, json={**payload, "limit": 2})
        assert first.status_code == 200
        first = first.json()
        assert first["total_count"] == size
        assert first["count"] == min(size, 2)
        assert first["has_more"] == (size > 2)
        if size:
            second = client.post(endpoint, json={
                **payload, "limit": 2, "cursor": first["next_cursor"],
            }).json()
            assert second["total_count"] == size
            assert second["count"] == 1
            assert second["start_index"] == 3
            assert not second["has_more"]
            assert len(second["results"][0]["sources"]) == 2


@pytest.mark.parametrize("related", [False, True])
def test_total_count_respects_hot_filter_and_exhausted_cursor(related):
    repository = Neo4jRepository(Settings())
    repository.driver = MagicMock()
    session = repository.driver.session.return_value.__enter__.return_value
    hot, warm = pagination_event(1), pagination_event(2)
    hot["post"]["metric_tier"] = "hot"
    warm["post"]["metric_tier"] = "warm"
    session.run.return_value.data.side_effect = (
        [[], [], [hot, warm], []] if related else [[hot, warm], []]
    )
    search = repository.search_related_events if related else repository.search_events
    page = search(location="Hà Nội", entity=None, hours=48, limit=2,
                  hot_only=True, after=(float("-inf"), "", ""))
    assert page == []
    assert page.total_count == 1
