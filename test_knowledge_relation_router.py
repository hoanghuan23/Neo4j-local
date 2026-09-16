import copy
from unittest.mock import Mock, patch

import pytest

from knowledge_relation_router import classify_relation_routes, normalize_relation_routes
from knowledge_settings import RELATION_GROUPS, RELATION_ROUTER_SCHEMA, KNOWLEDGE_MODULES


CONTENT = "Bộ Công an cho biết mưa lớn gây ngập. Người dân phản đối việc đóng đường."


def event(local_id="ev1", time_expression=None):
    return dict(local_id=local_id, type="OTHER", description="mưa lớn gây ngập",
                evidence_text="mưa lớn gây ngập", time_expression=time_expression,
                participants=[])


@pytest.mark.parametrize("count,expected", [
    (0, []),
    (1, ["EVENT_HIERARCHY", "PARTICIPANT_ROLE"]),
    (2, ["EVENT_HIERARCHY", "EVENT_RELATION", "PARTICIPANT_ROLE"]),
])
def test_code_event_thresholds(count, expected):
    knowledge = {"events": [event(f"ev{i}") for i in range(count)]}
    model = Mock(return_value={"detected_modules": []})
    assert classify_relation_routes(CONTENT, knowledge, model) == {"detected_modules": expected}
    model.assert_called_once()


@pytest.mark.parametrize("entity_type,expected", [
    ("LOCATION", ["ENTITY_HIERARCHY"]),
    ("ORGANIZATION", ["ENTITY_HIERARCHY"]),
    ("PERSON", []),
])
def test_entity_hierarchy_without_events(entity_type, expected):
    model = Mock(return_value={"detected_modules": []})
    result = classify_relation_routes(CONTENT, {"entities": [{"type": entity_type}]}, model)
    assert result == {"detected_modules": expected}


def test_distinct_valid_ids_only():
    knowledge = {"events": [event(), event(), None, {}, event(""), event(" ")]}
    assert normalize_relation_routes(CONTENT, knowledge, {"detected_modules": []}) == {
        "detected_modules": ["EVENT_HIERARCHY", "PARTICIPANT_ROLE"]}


@pytest.mark.parametrize("time", ["2/9", "  hôm qua  ", "từ tháng 1 đến tháng 3"])
def test_temporal_from_extraction_is_not_delegated(time):
    model = Mock(return_value={"detected_modules": []})
    result = classify_relation_routes(CONTENT, {"events": [event(time_expression=time)]}, model)
    assert "TEMPORAL_RELATION" in result["detected_modules"]
    instructions = model.call_args.args[0].split("<base_knowledge>")[0]
    assert "- TEMPORAL_RELATION:" not in instructions
    assert "- CLAIM_PROVENANCE:" in instructions
    assert "- STANCE_PERSPECTIVE:" in instructions


@pytest.mark.parametrize("time", [None, "", "  ", " null ", "NONE", "nil", "n/a", 123])
def test_missing_temporal_delegates_without_inferring_from_status(time):
    current = event(time_expression=time)
    current["status"] = "COMPLETED"
    knowledge = {"events": [current], "posted_at": "2026-01-01"}
    model = Mock(return_value={"detected_modules": []})
    assert "TEMPORAL_RELATION" not in classify_relation_routes(CONTENT, knowledge, model)["detected_modules"]
    assert "- TEMPORAL_RELATION:" in model.call_args.args[0]
    assert "(PRECEDES) giữa hai Event" in model.call_args.args[0]
    model.return_value = {"detected_modules": ["TEMPORAL_RELATION"]}
    assert "TEMPORAL_RELATION" in classify_relation_routes(CONTENT, knowledge, model)["detected_modules"]


def test_merge_filters_deduplicates_sorts_and_preserves_input():
    knowledge = {"entities": [{"type": "LOCATION"}], "events": [event()]}
    before = copy.deepcopy(knowledge)
    raw = {"detected_modules": ["STANCE_PERSPECTIVE", "INVALID", None, {}, [],
                               "CLAIM_PROVENANCE", "STANCE_PERSPECTIVE", "EVENT_RELATION"]}
    assert normalize_relation_routes(CONTENT, knowledge, raw) == {"detected_modules": [
        "CLAIM_PROVENANCE", "ENTITY_HIERARCHY", "EVENT_HIERARCHY", "PARTICIPANT_ROLE",
        "STANCE_PERSPECTIVE"]}
    assert knowledge == before


def test_llm_cannot_add_structural_modules():
    model = Mock(return_value={"detected_modules": sorted(RELATION_GROUPS)})
    assert classify_relation_routes(CONTENT, {}, model) == {"detected_modules": [
        "CLAIM_PROVENANCE", "STANCE_PERSPECTIVE", "TEMPORAL_RELATION"]}


def test_strict_schema_and_no_detail_output_instructions():
    model = Mock(return_value={"detected_modules": []})
    classify_relation_routes(CONTENT, {}, model)
    prompt, schema = model.call_args.args
    assert schema == RELATION_ROUTER_SCHEMA
    assert schema["required"] == ["detected_modules"]
    assert set(schema["properties"]) == {"detected_modules"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["detected_modules"]["items"]["enum"] == sorted(RELATION_GROUPS)
    for old in ("event_routes", "pair_routes", "route_details", "reason", "evidence_text", "action"):
        assert old not in prompt
    assert "Phân biệt stance của tác giả Post với stance của người được trích dẫn" in prompt


def test_empty_input_skips_model():
    model = Mock()
    assert classify_relation_routes("  ", {}, model) == {"detected_modules": []}
    model.assert_not_called()


def test_code_only_with_no_semantic_context_skips_model():
    model = Mock()
    assert classify_relation_routes("", {"entities": [{"type": "LOCATION"}]}, model) == {
        "detected_modules": ["ENTITY_HIERARCHY"]}
    model.assert_not_called()


@pytest.mark.parametrize("knowledge", [
    {"entities": [{"name": "Công ty A", "type": "ORGANIZATION"}]},
    {"events": [event()]},
    {"events": [{"local_id": "ev1", "participants": [{"participant_text": "người dân"}]}]},
])
def test_knowledge_text_still_calls_model_without_content(knowledge):
    model = Mock(return_value={"detected_modules": ["CLAIM_PROVENANCE"]})
    assert "CLAIM_PROVENANCE" in classify_relation_routes("", knowledge, model)["detected_modules"]
    model.assert_called_once()


def test_detection_independent_of_execution_configuration():
    model = Mock(return_value={"detected_modules": ["CLAIM_PROVENANCE", "STANCE_PERSPECTIVE"]})
    with patch.dict(KNOWLEDGE_MODULES, {name: False for name in RELATION_GROUPS}):
        assert classify_relation_routes(CONTENT, {}, model) == {
            "detected_modules": ["CLAIM_PROVENANCE", "STANCE_PERSPECTIVE"]}
    model.assert_called_once()


@pytest.mark.parametrize("raw", [None, [], {}, {"detected_modules": None}, {"detected_modules": "bad"}])
def test_invalid_output_raises_even_with_code_decisions(raw):
    model = Mock(return_value=raw)
    with pytest.raises(ValueError):
        classify_relation_routes(CONTENT, {"events": [event()]}, model)


def test_model_error_propagates():
    with pytest.raises(RuntimeError, match="model failed"):
        classify_relation_routes(CONTENT, {}, Mock(side_effect=RuntimeError("model failed")))
