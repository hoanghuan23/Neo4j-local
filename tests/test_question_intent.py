from unittest.mock import Mock

from backend.chat_service import ChatService
from backend.models import ParsedQuestion
from backend.question_parser import RuleBasedQuestionParser


def test_model_clarification_prevents_search():
    repository = Mock()
    parser = Mock(parse=Mock(return_value=ParsedQuestion(
        hours=24, clarification_question='Bạn muốn nói đến tổ chức nào?',
    )))
    response = ChatService(parser, repository).chat('sự kiện của tổ chức đó', 10)
    assert response.answer == 'Bạn muốn nói đến tổ chức nào?'
    repository.search_events.assert_not_called()


def test_clear_topic_search_stays_broad():
    repository = Mock()
    repository.search_events.return_value = []
    ChatService(RuleBasedQuestionParser(), repository).chat(
        'sự kiện về công an tại Hồ Chí Minh', 10,
    )
    assert repository.search_events.call_args.kwargs['entity'] == 'công an'
    assert repository.search_events.call_args.kwargs['location'] == 'Hồ Chí Minh'
