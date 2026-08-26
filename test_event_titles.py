from unittest.mock import Mock

from backfill_event_titles import backfill_event_titles
from event_titles import (
    generate_event_title,
    is_valid_event_title,
    resolve_event_title,
)


VALID_TITLE = (
    "Người đàn ông mặc đồng phục bảo vệ hành hung tài xế xe ôm công nghệ"
)


def test_valid_title_does_not_call_model():
    call_model = Mock()

    title, needs_backfill = resolve_event_title(
        "Mô tả đầy đủ",
        VALID_TITLE,
        call_model,
    )

    assert title == VALID_TITLE
    assert needs_backfill is False
    call_model.assert_not_called()


def test_invalid_title_is_repaired_once():
    call_model = Mock(return_value={"title": VALID_TITLE})

    title, needs_backfill = resolve_event_title(
        "Người đàn ông hành hung tài xế xe ôm công nghệ.",
        "Quá ngắn",
        call_model,
    )

    assert title == VALID_TITLE
    assert needs_backfill is False
    assert call_model.call_count == 1


def test_invalid_repair_falls_back_to_description():
    description = "Người đàn ông hành hung tài xế xe ôm công nghệ."
    call_model = Mock(return_value={"title": "Vẫn ngắn"})

    title, needs_backfill = resolve_event_title(
        description,
        "Quá ngắn",
        call_model,
    )

    assert title == description
    assert needs_backfill is True
    assert call_model.call_count == 1


def test_backfill_generation_retries_once():
    call_model = Mock(side_effect=[
        {"title": "Quá ngắn"},
        {"title": VALID_TITLE},
    ])

    title, needs_backfill = generate_event_title("Mô tả", call_model)

    assert title == VALID_TITLE
    assert needs_backfill is False
    assert call_model.call_count == 2
    assert is_valid_event_title(title)


def test_backfill_dry_run_does_not_write():
    session = Mock()
    session.run.return_value = [{
        "event_key": "event-1",
        "description": "Người đàn ông hành hung tài xế xe ôm công nghệ.",
        "current_title": None,
    }]
    call_model = Mock(return_value={"title": VALID_TITLE})

    result = backfill_event_titles(session, call_model)

    assert result["selected"] == 1
    assert result["updated"] == 0
    assert result["events"][0]["title"] == VALID_TITLE
    assert session.run.call_count == 1


def test_backfill_apply_writes_title_without_description_parameter():
    session = Mock()
    write_result = Mock()
    session.run.side_effect = [
        [{
            "event_key": "event-1",
            "description": "Người đàn ông hành hung tài xế xe ôm công nghệ.",
            "current_title": None,
        }],
        write_result,
    ]
    call_model = Mock(return_value={"title": VALID_TITLE})

    result = backfill_event_titles(session, call_model, apply=True)

    assert result["updated"] == 1
    write_kwargs = session.run.call_args_list[1].kwargs
    assert write_kwargs["title"] == VALID_TITLE
    assert "description" not in write_kwargs
    write_result.consume.assert_called_once_with()
