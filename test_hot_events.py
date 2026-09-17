from datetime import date
from unittest.mock import MagicMock

import pytest

from backend.chat_service import ChatService
from backend.config import Settings
from backend.neo4j_repository import Neo4jRepository
from backend.pagination import decode_event_cursor
from backend.question_parser import RuleBasedQuestionParser
from test_backend import pagination_event


@pytest.mark.parametrize('question', [
    'các sự kiện hot trong ngày', 'Sự kiện hot hôm nay?',
    'Tìm những sự kiện hot trong ngày hôm nay',
])
def test_hot_parser_uses_machine_date(question):
    parsed = RuleBasedQuestionParser(today_provider=lambda: date(2026, 9, 15)).parse(question)
    assert parsed.hot_only
    assert parsed.posted_date == date(2026, 9, 15)
    assert parsed.location is None
    assert parsed.entity is None


def test_hot_events_filter_rank_deduplicate_and_continue():
    def row(index, score, tier='hot', posted_at='2026-09-15T01:00:00'):
        result = pagination_event(index)
        result['post'].update(last_engagement_velocity=score, metric_tier=tier, posted_at=posted_at)
        return result

    # Higher score wins even when the source is older; warm/out-of-day sources
    # cannot contribute their score. Duplicate events span both graph schemas.
    current = [row(1, 10), row(2, 20), row(3, None), row(4, 1000, 'warm'),
               row(5, 2000, posted_at='2026-09-14T01:00:00'), row(6, 0), row(7, -1), row(9, None)]
    legacy = [row(1, 30, posted_at='2026-09-14T18:00:00'), row(8, 20)]
    repository = Neo4jRepository(Settings())
    repository.driver = MagicMock()
    session = repository.driver.session.return_value.__enter__.return_value
    session.run.side_effect = lambda query, **kwargs: MagicMock(
        data=lambda: legacy if '[:DESCRIBES]' in query.split('WHERE')[0] else current
    )
    service = ChatService(
        RuleBasedQuestionParser(today_provider=lambda: date(2026, 9, 15)), repository,
    )
    first = service.chat('các sự kiện hot trong ngày', limit=2)
    assert [r.event_key for r in first.results] == ['event-01', 'event-08']
    assert first.results[0].post.last_engagement_velocity == 30
    cursor = decode_event_cursor(first.next_cursor)
    assert cursor.query.hot_only
    assert cursor.query.posted_date == date(2026, 9, 15)
    assert cursor.sort_key[0] == 20
    second = service.chat('xem tiếp', limit=2, cursor=first.next_cursor)
    assert [r.event_key for r in second.results] == ['event-02', 'event-06']
    third = service.chat('xem tiếp', limit=1, cursor=second.next_cursor)
    assert [r.event_key for r in third.results] == ['event-07']
    fourth = service.chat('xem tiếp', limit=1, cursor=third.next_cursor)
    assert [r.event_key for r in fourth.results] == ['event-09']
    fifth = service.chat('xem tiếp', limit=1, cursor=fourth.next_cursor)
    assert [r.event_key for r in fifth.results] == ['event-03']
    assert not fifth.has_more
    for call in session.run.call_args_list:
        assert call.kwargs['hot_only'] is True
        assert call.kwargs['posted_date'] == '2026-09-15'
        assert "post.metric_tier = 'hot'" in call.args[0]


def test_ordinary_today_query_does_not_enable_hot_sort():
    assert not RuleBasedQuestionParser().parse('sự kiện hôm nay').hot_only


@pytest.mark.parametrize('question', [
    'sự kiện hot trong tuần', 'Các sự kiện hot tuần này?',
    'Tìm những sự kiện hot trong tuần',
])
def test_weekly_hot_query_uses_seven_days(question):
    parsed = RuleBasedQuestionParser().parse(question)
    assert parsed.hot_only
    assert parsed.hours == 168
    assert parsed.posted_date is None
    assert parsed.location is None
    assert parsed.entity is None


@pytest.mark.parametrize('question', [
    'các sự kiện hot Hà Nội',
    'các sự kiện hot tại Hà Nội',
    'Tìm những sự kiện hot ở Hà Nội?',
    'sự kiện hot khu vực Hà Nội',
])
def test_hot_location_query(question):
    parsed = RuleBasedQuestionParser().parse(question)
    assert parsed.hot_only
    assert parsed.location == 'Hà Nội'
    assert parsed.entity is None
    assert parsed.hours == 24
    assert parsed.posted_date is None


@pytest.mark.parametrize('time_text,hours,today', [
    ('hôm nay', 24, True), ('trong ngày', 24, True),
    ('trong tuần', 168, False), ('tuần này', 168, False),
    ('trong 48 giờ', 48, False),
])
@pytest.mark.parametrize('template', ['sự kiện hot tại Hà Nội {}', 'sự kiện hot {} tại Hà Nội'])
def test_hot_location_with_time(time_text, hours, today, template):
    parsed = RuleBasedQuestionParser().parse(template.format(time_text))
    assert parsed.hot_only
    assert parsed.location == 'Hà Nội'
    assert parsed.hours == hours
    assert parsed.posted_date == (date.today() if today else None)


def test_hot_location_is_forwarded_to_repository():
    from backend.models import EventSearchPage
    repository = MagicMock()
    repository.search_events.return_value = EventSearchPage([], total_count=0)
    service = ChatService(RuleBasedQuestionParser(), repository)
    service.chat('các sự kiện hot Hà Nội', limit=10)
    assert repository.search_events.call_args.kwargs['location'] == 'Hà Nội'
    assert repository.search_events.call_args.kwargs['hot_only'] is True
