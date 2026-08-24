from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.neo4j_repository import (
    SEARCH_EVENTS_QUERY,
    SEARCH_LEGACY_EVENTS_QUERY,
    SEARCH_RELATED_ENTITIES_QUERY,
    Neo4jRepository,
    make_entity_terms,
)
from backend.question_parser import RuleBasedQuestionParser


class FakeRepository:
    def __init__(self, results=None, detail_results=None):
        self.results = results or []
        self.detail_results = detail_results or []
        self.search_args = None
        self.detail_search_args = None
        self.connected = False

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False

    def ping(self):
        return self.connected

    def search_events(self, **kwargs):
        self.search_args = kwargs
        return self.results

    def search_related_entities(self, **kwargs):
        self.detail_search_args = kwargs
        return self.detail_results


def test_parser_extracts_location_and_hours():
    parsed = RuleBasedQuestionParser().parse("Hà Nội 24h qua có gì?")

    assert parsed.intent == "search_events"
    assert parsed.location == "Hà Nội"
    assert parsed.hours == 24


def test_parser_supports_explicit_location_and_days():
    parsed = RuleBasedQuestionParser().parse("Có sự kiện gì tại Đà Nẵng 2 ngày qua?")

    assert parsed.location == "Đà Nẵng"
    assert parsed.hours == 48


def test_parser_supports_short_hanoi_queries():
    parser = RuleBasedQuestionParser()

    assert parser.parse("hà nội").location == "hà nội"
    assert parser.parse("sự kiện hà nội").location == "hà nội"


def test_parser_extracts_entity_without_treating_it_as_location():
    parsed = RuleBasedQuestionParser().parse(
        "Sự kiện liên quan tới Phú Lê trong 48h qua"
    )

    assert parsed.entity == "Phú Lê"
    assert parsed.location is None
    assert parsed.hours == 48


def test_parser_supports_multiple_entities_and_weeks():
    parsed = RuleBasedQuestionParser().parse(
        "các sự kiện liên quan Huấn Hoa Hồng hoặc Phú Lê trong 2 tuần qua"
    )

    assert parsed.entity == "Huấn Hoa Hồng hoặc Phú Lê"
    assert parsed.location is None
    assert parsed.hours == 336

    parsed_with_and = RuleBasedQuestionParser().parse(
        "sự kiện liên quan tới huấn và phú lê trong 2 tuần qua"
    )
    assert parsed_with_and.entity == "huấn và phú lê"
    assert parsed_with_and.hours == 336


def test_entity_alternatives_become_or_search_terms():
    assert make_entity_terms("Huấn Hoa Hồng hoặc Phú Lê") == [
        {"key": "huấn hoa hồng", "search_key": "huan hoa hong"},
        {"key": "phú lê", "search_key": "phu le"},
    ]
    assert make_entity_terms("huấn và phú lê") == [
        {"key": "huấn", "search_key": "huan"},
        {"key": "phú lê", "search_key": "phu le"},
    ]


def test_event_query_filters_by_time_and_can_match_post_content():
    assert "post.posted_at IS NOT NULL" in SEARCH_EVENTS_QUERY
    assert "localdatetime() - duration({hours: $hours})" in SEARCH_EVENTS_QUERY
    assert "toLower(coalesce(post.content, '')) CONTAINS $location_key" in (
        SEARCH_EVENTS_QUERY
    )
    assert "[term IN $entity_terms WHERE" in (
        SEARCH_EVENTS_QUERY
    )
    assert "ORDER BY matched_entity_count DESC" in SEARCH_EVENTS_QUERY
    assert "MATCH (post:Post)-[:DESCRIBES]->(event:Event)" in (
        SEARCH_LEGACY_EVENTS_QUERY
    )
    assert "MATCH (event)-[:HAS_PARTICIPANT]->(event_entity:Entity)" in (
        SEARCH_LEGACY_EVENTS_QUERY
    )


def test_repository_merges_current_and_legacy_results_by_event_key():
    current_duplicate = {
        "event_key": "event-shared",
        "description": "Kết quả từ schema mới",
        "post": {"posted_at": "2026-08-20T08:00:00"},
    }
    current_result = {
        "event_key": "event-current",
        "description": "Chỉ có trong schema mới",
        "post": {"posted_at": "2026-08-22T08:00:00"},
    }
    legacy_duplicate = {
        "event_key": "event-shared",
        "description": "Kết quả trùng từ schema cũ",
        "post": {"posted_at": "2026-08-20T08:00:00"},
    }
    legacy_result = {
        "event_key": "event-legacy",
        "description": "Chỉ có trong schema cũ",
        "post": {"posted_at": "2026-08-21T08:00:00"},
    }
    session = MagicMock()
    session.run.side_effect = [
        Mock(data=Mock(return_value=[current_duplicate, current_result])),
        Mock(data=Mock(return_value=[legacy_duplicate, legacy_result])),
    ]
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    repository = Neo4jRepository(Settings())
    repository.driver = driver

    results = repository.search_events(
        location=None,
        entity="Huấn Hoa Hồng",
        hours=720,
        limit=3,
    )

    assert [result["event_key"] for result in results] == [
        "event-current",
        "event-legacy",
        "event-shared",
    ]
    assert results[-1]["description"] == "Kết quả từ schema mới"
    assert session.run.call_count == 2
    assert session.run.call_args_list[0].args[0] == SEARCH_EVENTS_QUERY
    assert session.run.call_args_list[1].args[0] == SEARCH_LEGACY_EVENTS_QUERY


def test_repository_ranks_shared_entity_events_before_individual_events():
    shared_event = {
        "event_key": "event-shared-entities",
        "description": "Huấn tặng quà cho Phú Lê",
        "matched_entity_count": 2,
        "post": {"posted_at": "2026-08-20T08:00:00"},
    }
    individual_event = {
        "event_key": "event-one-entity",
        "description": "Sự kiện riêng của Phú Lê",
        "matched_entity_count": 1,
        "post": {"posted_at": "2026-08-22T08:00:00"},
    }
    session = MagicMock()
    session.run.side_effect = [
        Mock(data=Mock(return_value=[individual_event, shared_event])),
        Mock(data=Mock(return_value=[])),
    ]
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    repository = Neo4jRepository(Settings())
    repository.driver = driver

    results = repository.search_events(
        location=None,
        entity="Huấn và Phú Lê",
        hours=336,
        limit=10,
    )

    assert [result["event_key"] for result in results] == [
        "event-shared-entities",
        "event-one-entity",
    ]


def test_repository_returns_distinct_sources_for_each_event():
    def event_row(post_id, posted_at, description="Sự kiện đã gộp"):
        return {
            "event_key": "event-shared",
            "description": description,
            "matched_entity_count": 1,
            "post": {
                "platform": "facebook",
                "platform_id": post_id,
                "content": f"Nội dung {post_id}",
                "posted_at": posted_at,
                "source_name": f"Nguồn {post_id}",
            },
        }

    posts = [
        event_row(f"post-{index}", f"2026-08-{19 + index:02d}T08:00:00")
        for index in range(1, 6)
    ]
    legacy_duplicate = event_row(
        "post-1", "2026-08-20T08:00:00", "Kết quả schema cũ"
    )
    session = MagicMock()
    session.run.side_effect = [
        Mock(data=Mock(return_value=posts)),
        Mock(data=Mock(return_value=[legacy_duplicate])),
    ]
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    repository = Neo4jRepository(Settings())
    repository.driver = driver

    results = repository.search_events(
        location=None,
        entity=None,
        hours=24,
        limit=10,
    )

    assert len(results) == 1
    assert results[0]["post"]["platform_id"] == "post-5"
    assert results[0]["sources"] == [
        {
            "source": f"Nguồn post-{index}",
            "posted_at": f"2026-08-{19 + index:02d}T08:00:00",
            "url": None,
        }
        for index in range(5, 0, -1)
    ]


def test_chat_returns_structured_graph_results():
    repository = FakeRepository(
        [
            {
                "event_key": "event-1",
                "type": "ACCIDENT",
                "description": "Một vụ tai nạn đã xảy ra.",
                "status": "REPORTED",
                "time_expression": "sáng nay",
                "entities": [
                    {"name": "Hà Nội", "type": "LOCATION", "role": "LOCATION"}
                ],
                "post": {
                    "platform": "facebook",
                    "platform_id": "post-1",
                    "content": "Nội dung bài viết",
                    "url": "https://example.test/post-1",
                    "posted_at": "2026-08-22T08:00:00+07:00",
                    "source_name": "Nguồn thử nghiệm",
                },
            }
        ]
    )
    app = create_app(Settings(gemini_api_key=""), repository)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "Hà Nội 24h qua có gì?", "limit": 5},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["query"] == {
        "intent": "search_events",
        "location": "Hà Nội",
        "entity": None,
        "hours": 24,
    }
    assert body["results"][0]["post"]["platform_id"] == "post-1"
    assert body["results"][0]["sources"] == [
        {
            "source": "Nguồn thử nghiệm",
            "posted_at": "2026-08-22T08:00:00+07:00",
            "url": "https://example.test/post-1",
        }
    ]
    assert repository.search_args == {
        "location": "Hà Nội",
        "entity": None,
        "hours": 24,
        "limit": 5,
    }
    assert body["answer"].startswith("Tìm thấy 1 sự kiện tại Hà Nội:")


def test_detail_command_returns_related_entity_post_counts():
    repository = FakeRepository(
        detail_results=[
            {
                "entity_type": "LOCATION",
                "entity_name": "Việt Nam",
                "post_count": 30,
            },
            {
                "entity_type": "PERSON",
                "entity_name": "Phú Lê",
                "post_count": 5,
            },
        ]
    )
    app = create_app(Settings(gemini_api_key=""), repository)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "  /DETAIL   hà nội  ", "limit": 20},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == {"intent": "detail", "subject": "hà nội"}
    assert body["count"] == 2
    assert body["results"] == []
    assert body["details"] == [
        {
            "entity_type": "LOCATION",
            "entity_name": "Việt Nam",
            "post_count": 30,
        },
        {
            "entity_type": "PERSON",
            "entity_name": "Phú Lê",
            "post_count": 5,
        },
    ]
    assert repository.detail_search_args == {"subject": "hà nội", "limit": 20}
    assert repository.search_args is None
    assert "30 bài viết chung" in body["answer"]


def test_detail_command_requires_a_subject():
    repository = FakeRepository()
    app = create_app(Settings(gemini_api_key=""), repository)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "/detail", "limit": 10},
        )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Thiếu chủ thể")
    assert repository.search_args is None
    assert repository.detail_search_args is None


def test_repository_searches_related_entities_with_normalized_subject():
    rows = [
        {
            "entity_type": "LOCATION",
            "entity_name": "Việt Nam",
            "post_count": 30,
        }
    ]
    session = MagicMock()
    session.run.return_value.data.return_value = rows
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    repository = Neo4jRepository(Settings())
    repository.driver = driver

    assert repository.search_related_entities(subject="Hà Nội", limit=10) == rows
    session.run.assert_called_once_with(
        SEARCH_RELATED_ENTITIES_QUERY,
        subject_key="hà nội",
        subject_search_key="ha noi",
        limit=10,
    )


def test_detail_query_prefers_exact_subject_before_contains_matches():
    assert "WITH collect(DISTINCT subject) AS candidates" in (
        SEARCH_RELATED_ENTITIES_QUERY
    )
    assert "WHEN any(candidate IN candidates WHERE" in (
        SEARCH_RELATED_ENTITIES_QUERY
    )
    assert "ELSE candidates" in SEARCH_RELATED_ENTITIES_QUERY
    assert "WHERE NOT related IN selected_subjects" in (
        SEARCH_RELATED_ENTITIES_QUERY
    )


def test_search_endpoint_and_empty_answer():
    repository = FakeRepository()
    app = create_app(Settings(gemini_api_key=""), repository)

    with TestClient(app) as client:
        response = client.get("/api/search", params={"q": "Ở Huế hôm nay có gì?"})
        health = client.get("/health")

    assert response.status_code == 200
    assert response.json()["answer"] == "Không tìm thấy sự kiện tại Huế."
    assert repository.search_args == {
        "location": "Huế",
        "entity": None,
        "hours": 24,
        "limit": 10,
    }
    assert health.json() == {"status": "ok", "neo4j": "connected"}


def test_chat_endpoint_uses_gemini_parser_and_answer_generator():
    repository = FakeRepository(
        [
            {
                "event_key": "event-1",
                "type": "ACCIDENT",
                "description": "Một vụ tai nạn đã xảy ra.",
                "status": "REPORTED",
                "time_expression": "sáng nay",
                "entities": [],
                "post": {
                    "platform": "facebook",
                    "platform_id": "post-1",
                    "content": "Nội dung bài viết",
                    "url": "https://example.test/post-1",
                    "posted_at": "2026-08-22T08:00:00",
                    "source_name": "Nguồn thử nghiệm",
                },
            }
        ]
    )
    gemini_client = Mock()
    gemini_client.models.generate_content.side_effect = [
        SimpleNamespace(
            parsed={
                "intent": "search_events",
                "location": "Đà Nẵng",
                "hours": 48,
            }
        ),
        SimpleNamespace(parsed={"answer": "Câu trả lời Gemini có kiểm chứng."}),
    ]

    with patch("google.genai.Client", return_value=gemini_client):
        app = create_app(
            Settings(gemini_api_key="test-key", chat_gemini_model="test-model"),
            repository,
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "message": "Đà Nẵng 2 ngày qua có sự kiện gì?",
                    "limit": 5,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Câu trả lời Gemini có kiểm chứng."
    assert body["query"] == {
        "intent": "search_events",
        "location": "Đà Nẵng",
        "entity": None,
        "hours": 48,
    }
    assert body["count"] == 1
    assert repository.search_args == {
        "location": "Đà Nẵng",
        "entity": None,
        "hours": 48,
        "limit": 5,
    }
    assert gemini_client.models.generate_content.call_count == 2
    gemini_client.close.assert_called_once_with()
