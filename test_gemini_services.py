import json
import logging
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.chat_service import ChatService, TemplateAnswerGenerator
from backend.gemini_services import (
    FallbackAnswerGenerator,
    FallbackQuestionParser,
    GeminiAnswerGenerator,
    GeminiQuestionParser,
)
from backend.models import EventResult, ParsedQuestion
from backend.question_parser import RuleBasedQuestionParser


class FakeTypes:
    @staticmethod
    def GenerateContentConfig(**kwargs):
        return kwargs

    @staticmethod
    def AutomaticFunctionCallingConfig(**kwargs):
        return kwargs


def event_data():
    return {
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
            "posted_at": "2026-08-22T08:00:00",
            "source_name": "Nguồn thử nghiệm",
        },
    }


@pytest.mark.parametrize(
    ("question", "payload", "location", "entity", "hours"),
    [
        (
            "Hà Nội 24h qua có gì?",
            {"intent": "search_events", "location": "Hà Nội", "hours": 24},
            "Hà Nội",
            None,
            24,
        ),
        (
            "Đà Nẵng 2 ngày qua có sự kiện gì?",
            {"intent": "search_events", "location": "Đà Nẵng", "hours": 48},
            "Đà Nẵng",
            None,
            48,
        ),
        (
            "Sự kiện liên quan tới Phú Lê",
            {
                "intent": "search_events",
                "location": None,
                "entity": "Phú Lê",
                "hours": 24,
            },
            None,
            "Phú Lê",
            24,
        ),
    ],
)
def test_gemini_question_parser_returns_validated_structure(
    question, payload, location, entity, hours
):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed=payload,
        text=json.dumps(payload),
    )
    parser = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
    )

    parsed = parser.parse(question)

    assert parsed == ParsedQuestion(
        location=location,
        entity=entity,
        hours=hours,
    )
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "test-model"
    assert question in call["contents"]
    assert call["config"]["response_schema"] is ParsedQuestion


def test_gemini_question_parser_broadens_administrative_location():
    payload = {
        "intent": "search_events",
        "location": "thành phố Lạng Sơn",
        "hours": 168,
    }
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed=payload,
        text=json.dumps(payload),
    )
    parser = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
    )

    parsed = parser.parse("Sự kiện thành phố Lạng Sơn trong 1 tuần")

    assert parsed.location == "Lạng Sơn"
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "tên địa lý ngắn gọn" in prompt
    assert "'thành phố Lạng Sơn', 'tỉnh Lạng Sơn' -> 'Lạng Sơn'" in prompt


def test_gemini_question_parser_normalizes_between_entities():
    payload = {
        "intent": "search_events",
        "location": None,
        "entity": "giữa Hà Nội và Lào Cai",
        "hours": 24,
    }
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=payload)
    parser = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
    )

    parsed = parser.parse("sự kiện có liên quan giữa hà nội và lào cai")

    assert parsed.entity == "Hà Nội và Lào Cai"
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "chỉ trả 'A và B'" in prompt


def test_gemini_question_parser_uses_deterministic_month_duration():
    payload = {
        "intent": "search_events",
        "location": None,
        "entity": "giữa Hà Nội và Lào Cai 1 tháng trở lại đây",
        "hours": 168,
    }
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=payload)
    parser = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
        default_hours=168,
    )

    parsed = parser.parse(
        "sự kiện có liên quan giữa hà nội và lào cai 1 tháng trở lại đây"
    )

    assert parsed.entity == "Hà Nội và Lào Cai"
    assert parsed.hours == 720


def test_gemini_question_parser_keeps_model_hours_for_unrecognized_duration():
    payload = {
        "intent": "search_events",
        "location": "Hà Nội",
        "entity": None,
        "hours": 360,
    }
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=payload)
    parser = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
        default_hours=168,
    )

    parsed = parser.parse("sự kiện Hà Nội trong nửa tháng qua")

    assert parsed.hours == 360


def test_gemini_question_parser_uses_deterministic_exact_date():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed={
            "intent": "search_events",
            "location": "Hà Nội",
            "hours": 24,
        },
    )
    parser = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
    )

    parsed = parser.parse("sự kiện Hà Nội ngày 24 tháng 8 năm 2025")

    assert parsed.posted_date == date(2025, 8, 24)


def test_gemini_question_parser_uses_configured_default_hours_in_prompt():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed={"location": "Hà Nội", "hours": 168},
    )
    parser = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
        default_hours=168,
    )

    parsed = parser.parse("sự kiện Hà Nội")

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "Nếu không nêu khoảng thời gian, dùng hours=168" in prompt
    assert parsed.hours == 168


def test_gemini_question_parser_keeps_broad_topic_as_search_condition():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed={
            "intent": "search_events",
            "location": None,
            "entity": None,
            "hours": 168,
        },
    )
    parser = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
        default_hours=168,
    )

    parsed = parser.parse("bóng đá việt nam")

    assert parsed.entity == "bóng đá việt nam"
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "'bóng đá Việt Nam'" in prompt
    assert "phải được giữ trong entity, không trả null" in prompt


def test_parsed_question_schema_uses_original_search_fields():
    schema = ParsedQuestion.model_json_schema()

    assert schema["required"] == ["hours"]
    assert set(schema["properties"]) == {
        "intent",
        "location",
        "entity",
        "hours",
        "posted_date",
    }


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(parsed=None, text=""),
        SimpleNamespace(parsed={"intent": "unsupported", "hours": 24}),
        SimpleNamespace(
            parsed={
                "intent": "search_events",
                "location": "Hà Nội",
                "hours": 721,
            }
        ),
    ],
)
def test_question_parser_falls_back_on_invalid_gemini_output(response):
    client = Mock()
    client.models.generate_content.return_value = response
    primary = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
    )
    parser = FallbackQuestionParser(primary, RuleBasedQuestionParser())

    parsed = parser.parse("Hà Nội 24h qua có gì?")

    assert parsed == ParsedQuestion(location="Hà Nội", hours=24)


def test_question_parser_falls_back_on_client_error():
    primary = Mock()
    primary.parse.side_effect = TimeoutError("timeout")
    parser = FallbackQuestionParser(primary, RuleBasedQuestionParser())

    parsed = parser.parse("Có sự kiện gì tại Đà Nẵng 2 ngày qua?")

    assert parsed.location == "Đà Nẵng"
    assert parsed.hours == 48


def test_gemini_parser_uses_168_hours_for_previous_week():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed={"location": "Lạng Sơn", "hours": 168},
    )
    parser = GeminiQuestionParser(
        client=client,
        types_module=FakeTypes,
        model="test-model",
    )

    parsed = parser.parse("tình hình Lạng Sơn trong tuần trước")

    assert parsed.location == "Lạng Sơn"
    assert parsed.hours == 168


def test_gemini_answer_generator_uses_structured_graph_data():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        parsed={"answer": "Trong 24 giờ qua có một vụ tai nạn tại Hà Nội."}
    )
    generator = GeminiAnswerGenerator(
        client=client,
        types_module=FakeTypes,
        model="test-model",
    )
    event = EventResult.model_validate(event_data())
    assert event.title == event.description

    answer = generator.generate(
        question="Hà Nội 24h qua có gì?",
        parsed=ParsedQuestion(location="Hà Nội", hours=24),
        events=[event],
    )

    assert answer == "Trong 24 giờ qua có một vụ tai nạn tại Hà Nội."
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert '"event_key": "event-1"' in contents
    assert '"source_name": "Nguồn thử nghiệm"' in contents
    assert '"title":' not in contents
    assert "không thêm" in contents
    assert "mỗi sự kiện thành một mục riêng" in contents
    assert "danh sách Markdown đánh số" in contents
    assert "không gộp nhiều sự kiện" in contents
    assert "Giữ nguyên thứ tự của mảng events" in contents


def test_answer_generator_falls_back_on_invalid_output():
    primary = Mock()
    primary.generate.side_effect = ValueError("invalid response")
    generator = FallbackAnswerGenerator(primary, TemplateAnswerGenerator())
    event = EventResult.model_validate(event_data())

    answer = generator.generate(
        question="Hà Nội 24h qua có gì?",
        parsed=ParsedQuestion(location="Hà Nội", hours=24),
        events=[event],
    )

    assert answer.startswith("Tìm thấy 1 sự kiện tại Hà Nội:")


def test_answer_generator_skips_gemini_for_empty_results():
    primary = Mock()
    generator = FallbackAnswerGenerator(primary, TemplateAnswerGenerator())

    answer = generator.generate(
        question="Ở Huế hôm nay có gì?",
        parsed=ParsedQuestion(location="Huế", hours=24),
        events=[],
    )

    assert answer == "Không tìm thấy sự kiện tại Huế."
    primary.generate.assert_not_called()


def test_chat_service_preserves_results_with_injected_gemini_dependencies():
    parser = Mock()
    parser.parse.return_value = ParsedQuestion(
        location="Hà Nội",
        hours=24,
    )
    repository = Mock()
    repository.search_events.return_value = [event_data()]
    answer_generator = Mock()
    answer_generator.generate.return_value = "Câu trả lời Gemini"
    service = ChatService(parser, repository, answer_generator)

    response = service.chat("Hà Nội 24h qua có gì?", limit=5)

    assert response.answer == "Câu trả lời Gemini"
    assert response.count == 1
    assert response.results[0].event_key == "event-1"
    repository.search_events.assert_called_once_with(
        location="Hà Nội",
        entity=None,
        hours=24,
        posted_date=None,
        limit=6,
        after=None,
    )


def test_chat_continuation_reuses_query_without_parser_or_answer_model():
    parser = Mock()
    parser.parse.return_value = ParsedQuestion(location="Hà Nội", hours=24)
    first_event = event_data()
    first_event["event_key"] = "event-2"
    first_event["post"] = {
        **first_event["post"],
        "platform_id": "post-2",
        "posted_at": "2026-08-23T08:00:00",
    }
    second_event = event_data()
    repository = Mock()
    repository.search_events.side_effect = [
        [first_event, second_event],
        [second_event],
    ]
    answer_generator = Mock()
    answer_generator.generate.return_value = "Câu trả lời Gemini"
    service = ChatService(parser, repository, answer_generator)

    first = service.chat("Hà Nội 24h qua có gì?", limit=1)
    second = service.chat("xem tiếp", limit=1, cursor=first.next_cursor)

    assert second.start_index == 2
    assert second.results[0].event_key == "event-1"
    assert "2. Một vụ tai nạn" in second.answer
    parser.parse.assert_called_once_with("Hà Nội 24h qua có gì?")
    answer_generator.generate.assert_called_once()
    assert repository.search_events.call_args_list[1].kwargs == {
        "location": "Hà Nội",
        "entity": None,
        "hours": 24,
        "posted_date": None,
        "limit": 2,
        "after": (0, "2026-08-23T08:00:00", "event-2"),
    }


def test_template_answer_describes_entity_as_subject_not_location():
    generator = TemplateAnswerGenerator()
    parsed = ParsedQuestion(entity="Phú Lê", hours=24)

    answer = generator.generate(
        question="Sự kiện liên quan tới Phú Lê",
        parsed=parsed,
        events=[],
    )

    assert answer == "Không tìm thấy sự kiện liên quan tới Phú Lê."


def test_logs_tokens_and_cost_for_parser_and_answer(caplog):
    parser_client = Mock()
    parser_client.models.generate_content.return_value = SimpleNamespace(
        parsed={"intent": "search_events", "entity": "Phú Lê", "hours": 24},
        usage_metadata=SimpleNamespace(
            prompt_token_count=1_000,
            candidates_token_count=100,
            thoughts_token_count=50,
            total_token_count=1_150,
        ),
    )
    answer_client = Mock()
    answer_client.models.generate_content.return_value = SimpleNamespace(
        parsed={"answer": "Một sự kiện kiểm thử."},
        usage_metadata=SimpleNamespace(
            prompt_token_count=2_000,
            candidates_token_count=200,
            thoughts_token_count=100,
            total_token_count=2_300,
        ),
    )
    parser = GeminiQuestionParser(
        client=parser_client,
        types_module=FakeTypes,
        model="test-model",
    )
    generator = GeminiAnswerGenerator(
        client=answer_client,
        types_module=FakeTypes,
        model="test-model",
    )

    with caplog.at_level(logging.INFO, logger="backend.gemini_services"):
        parsed = parser.parse("Sự kiện liên quan tới Phú Lê")
        generator.generate(
            question="Sự kiện liên quan tới Phú Lê",
            parsed=parsed,
            events=[EventResult.model_validate(event_data())],
        )

    logs = caplog.text
    assert "stage=question_parser" in logs
    assert "input_tokens=1000" in logs
    assert "output_tokens=100" in logs
    assert "thinking_tokens=50" in logs
    assert "total_cost_usd=0.00047500" in logs
    assert "stage=answer_generator" in logs
    assert "input_tokens=2000" in logs
    assert "output_tokens=200" in logs
    assert "thinking_tokens=100" in logs
    assert "total_cost_usd=0.00095000" in logs
