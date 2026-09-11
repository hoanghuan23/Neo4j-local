from unittest.mock import Mock

import pytest

from backend.chat_service import ChatService, InvalidChatCommand
from backend.models import ChatResponse
from backend.question_parser import parse_event_location_question


@pytest.mark.parametrize('question', [
    'Sự kiện "Bà cụ nhặt ve chai tìm thấy vàng" diễn ra ở đâu?',
    'Sự kiện “Bà cụ nhặt ve chai tìm thấy vàng” xảy ra tại đâu',
])
def test_extract_description(question):
    assert parse_event_location_question(question) == 'Bà cụ nhặt ve chai tìm thấy vàng'


def test_normal_location_search_is_unchanged():
    assert parse_event_location_question('Sự kiện ở Hà Nội') is None


def test_chat_returns_chains_and_sources_without_using_general_parser():
    parser, repository = Mock(), Mock()
    base = dict(event_key='e1', event_description='Bà cụ nhặt vàng',
                post_id='1', post_platform='facebook', post_url='https://example.com/post')
    repository.locate_event.return_value = [
        dict(base, location_chain=['TP HCM'], relations=[], depth=0),
        dict(base, location_chain=['TP HCM', 'Miền Nam', 'Việt Nam'],
             relations=['IN_REGION', 'PART_OF'], depth=2),
    ]
    result = ChatService(parser, repository).chat('Sự kiện "Bà cụ nhặt vàng" diễn ra ở đâu?', 10)
    assert result.query.intent == 'locate_event'
    assert result.count == 1
    assert len(result.location_candidates) == 2
    assert 'TP HCM → Miền Nam → Việt Nam' in result.answer
    assert 'https://example.com/post' in result.answer
    assert 'chưa xác nhận' in result.answer
    parser.parse.assert_not_called()
    repository.locate_event.assert_called_once_with(description='Bà cụ nhặt vàng', limit=10)
    assert ChatResponse.model_validate(result.model_dump()).query.intent == 'locate_event'


@pytest.mark.parametrize('rows, expected', [
    ([], 'Không tìm thấy'),
    ([dict(event_key='e1')], 'Chưa có địa điểm'),
])
def test_missing_evidence(rows, expected):
    repository = Mock()
    repository.locate_event.return_value = rows
    result = ChatService(Mock(), repository).chat('Sự kiện "nhặt vàng" diễn ra ở đâu', 10)
    assert expected in result.answer


def test_location_question_rejects_search_cursor():
    with pytest.raises(InvalidChatCommand):
        ChatService(Mock(), Mock()).chat('Sự kiện "nhặt vàng" diễn ra ở đâu', 10, 'cursor')


def test_groups_posts_and_preserves_location_provenance():
    repository = Mock()
    base = dict(event_key='e1', event_description='Cưa cây khiến 1 người tử vong',
                post_platform='facebook')
    located = dict(base, post_id='2', mentioned_location='Thanh Hóa')
    repository.locate_event.return_value = [
        dict(base, post_id='1'),
        dict(located, location_chain=['Thanh Hóa']),
        dict(located, location_chain=['Thanh Hóa', 'miền Trung', 'Việt Nam']),
        dict(located, location_chain=['Thanh Hóa', 'miền Trung', 'Việt Nam']),
        dict(base, post_id='3'),
    ]
    result = ChatService(Mock(), repository).chat(
        'sự kiện cưa cây khiến 1 người tử vong diễn ra ở đâu', 10)
    assert result.count == 1
    event, = result.location_events
    assert event.location_status == 'mentioned'
    assert len(event.sources) == 3
    location, = event.locations
    assert location.location_chain == ['Thanh Hóa', 'miền Trung', 'Việt Nam']
    assert location.source_ids == [event.sources[1].source_id]
    assert 'Chưa có địa điểm' not in result.answer
    assert len(result.location_candidates) == 5
    assert ChatResponse.model_validate(result.model_dump()).location_events == result.location_events


def test_distinct_events_unknown_locations_and_multiple_branches():
    repository = Mock()
    repository.locate_event.return_value = [
        dict(event_key='e1', post_id='1', mentioned_location='A', location_chain=['A', 'B']),
        dict(event_key='e1', post_id='2', mentioned_location='A', location_chain=['A', 'C']),
        dict(event_key='e1', post_id='3', mentioned_location='D'),
        dict(event_key='e2', post_id='1'),
    ]
    result = ChatService(Mock(), repository).chat('Sự kiện cưa cây diễn ra ở đâu', 10)
    assert result.count == 2
    first, second = result.location_events
    assert [loc.location_chain for loc in first.locations] == [['A', 'B'], ['A', 'C'], ['D']]
    assert [loc.source_ids for loc in first.locations] == [['source-1'], ['source-2'], ['source-3']]
    assert second.location_status == 'unknown'
    assert second.locations == []
    assert len(second.sources) == 1


def test_location_sources_include_name_and_posted_at_after_deduplication():
    repository = Mock()
    base = dict(event_key='e1', post_platform='facebook', post_id='1',
                source_name='Báo Thanh Hóa', posted_at='2026-09-11T08:30:00+07:00',
                mentioned_location='Thanh Hóa')
    repository.locate_event.return_value = [
        dict(base, location_chain=['Thanh Hóa']),
        dict(base, location_chain=['Thanh Hóa', 'Việt Nam']),
        dict(event_key='e1', post_id='2'),
    ]
    result = ChatService(Mock(), repository).chat('Sự kiện cưa cây diễn ra ở đâu', 10)
    first, second = result.location_events[0].sources
    assert first.source_name == 'Báo Thanh Hóa'
    assert first.posted_at == '2026-09-11T08:30:00+07:00'
    assert second.source_name is None
    assert second.posted_at is None
    payload = ChatResponse.model_validate(result.model_dump()).model_dump()
    assert payload['location_events'][0]['sources'][0]['posted_at'] == base['posted_at']
    assert payload['location_candidates'][0]['source_name'] == base['source_name']
