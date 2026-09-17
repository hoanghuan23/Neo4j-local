from unittest.mock import Mock

import pytest

from backend.chat_service import ChatService
from backend.models import EventSearchPage
from backend.question_parser import RuleBasedQuestionParser


@pytest.mark.parametrize('question', [
    'các sự kiện trong tuần', 'sự kiện trong tuần',
    'Sự kiện tuần này?', 'Tìm những sự kiện trong tuần',
    'Cho mình biết các sự kiện tuần này',
])
def test_weekly_feed_uses_seven_days_without_filters(question):
    parser = RuleBasedQuestionParser(default_hours=48)
    parsed = parser.parse(question)
    assert parsed.hours == 168
    assert parsed.posted_date is None
    assert parsed.location is None
    assert parsed.entity is None
    assert not parsed.hot_only

    repository = Mock()
    repository.search_events.return_value = EventSearchPage([], total_count=0)
    ChatService(parser, repository).chat(question, limit=10)
    args = repository.search_events.call_args.kwargs
    assert args['hours'] == 168
    assert args['location'] is None
    assert args['entity'] is None
    assert args['posted_date'] is None
    assert not args.get('hot_only', False)


def test_weekly_feed_respects_max_hours():
    assert RuleBasedQuestionParser(max_hours=72).parse('các sự kiện trong tuần').hours == 72
