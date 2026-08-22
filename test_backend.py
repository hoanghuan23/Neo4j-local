from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.neo4j_repository import SEARCH_EVENTS_QUERY
from backend.question_parser import RuleBasedQuestionParser


class FakeRepository:
    def __init__(self, results=None):
        self.results = results or []
        self.search_args = None
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


def test_event_query_does_not_filter_by_time_and_can_match_post_content():
    assert "datetime()" not in SEARCH_EVENTS_QUERY
    assert "duration(" not in SEARCH_EVENTS_QUERY
    assert "toLower(coalesce(post.content, '')) CONTAINS $location_key" in (
        SEARCH_EVENTS_QUERY
    )


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
    app = create_app(Settings(), repository)

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
        "hours": 24,
    }
    assert body["results"][0]["post"]["platform_id"] == "post-1"
    assert repository.search_args == {"location": "Hà Nội", "limit": 5}
    assert body["answer"].startswith("Tìm thấy 1 sự kiện tại Hà Nội:")


def test_search_endpoint_and_empty_answer():
    repository = FakeRepository()
    app = create_app(Settings(), repository)

    with TestClient(app) as client:
        response = client.get("/api/search", params={"q": "Ở Huế hôm nay có gì?"})
        health = client.get("/health")

    assert response.status_code == 200
    assert response.json()["answer"] == "Không tìm thấy sự kiện tại Huế."
    assert health.json() == {"status": "ok", "neo4j": "connected"}
