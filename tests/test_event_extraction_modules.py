"""Contracts for the separate occurrence and event-relation stages."""

from unittest.mock import Mock, patch

import pytest

import knowledge_pipeline as pipeline
from knowledge_relations.event_relation import extract_event_relations
from knowledge_settings import EVENT_RELATION_TYPES
from knowledge_validation import validate_knowledge
from tests.test_knowledge_validation_structure import entity, event


CONTENT = "Alice performed the first occurrence. This caused the second occurrence."


def base():
    return {
        "entities": [entity("person", "Alice")],
        "events": [
            event("first", "OTHER", "Alice performed the first occurrence"),
            event("second", "OTHER", "This caused the second occurrence"),
        ],
    }


@pytest.mark.parametrize("relation_type,marker", [
    ("APPROVES", "approved"),
    ("CAUSES", "caused"),
    ("ENABLES", "enabled"),
    ("PRECEDES", "before"),
    ("RELATED_TO", "linked"),
])
def test_all_relation_types_and_invalid_items(relation_type, marker):
    content = CONTENT + f" The first occurrence {marker} the second occurrence."
    evidence = f"The first occurrence {marker} the second occurrence"
    valid = {
        "source_event_id": "first",
        "target_event_id": "second",
        "type": relation_type,
        "evidence_text": evidence,
    }
    model = Mock(return_value={"event_relations": [
        valid,
        dict(valid),
        dict(valid, target_event_id="first"),
        dict(valid, target_event_id="unknown"),
        dict(valid, evidence_text="invented"),
        dict(valid, type="INVALID"),
        None,
    ]})

    enriched = extract_event_relations(content, base(), model)
    final = validate_knowledge(content, enriched, "test", "post")

    assert final["event_relations"] == [
        dict(valid, source_event_id="ev1", target_event_id="ev2")
    ]
    schema = model.call_args.args[1]
    assert set(
        schema["properties"]["event_relations"]["items"]["properties"]["type"]["enum"]
    ) == EVENT_RELATION_TYPES


def test_causal_relation_without_explicit_marker_is_dropped():
    model = Mock(return_value={"event_relations": [{
        "source_event_id": "first",
        "target_event_id": "second",
        "type": "CAUSES",
        "evidence_text": "Alice performed the first occurrence",
    }]})
    result = extract_event_relations(CONTENT, base(), model)
    assert validate_knowledge(CONTENT, result)["event_relations"] == []


def test_empty_and_single_event_avoid_model_calls():
    model = Mock()
    empty = {"entities": [], "events": []}
    assert extract_event_relations("", empty, model)["event_relations"] == []
    single = base()
    single["events"] = single["events"][:1]
    assert extract_event_relations(CONTENT, single, model)["event_relations"] == []
    model.assert_not_called()


@pytest.mark.parametrize("raw", [None, {}, [], {"wrong": []}])
def test_invalid_model_envelope_raises(raw):
    model = Mock(return_value=raw)
    with pytest.raises(ValueError):
        extract_event_relations(CONTENT, base(), model)
    model.assert_called_once()


def test_model_failure_propagates():
    with pytest.raises(RuntimeError, match="model unavailable"):
        extract_event_relations(
            CONTENT,
            base(),
            Mock(side_effect=RuntimeError("model unavailable")),
        )


def test_pipeline_stage_order_and_stable_final_keys():
    order = []
    knowledge = base()
    relation_model = Mock()

    def record(name, function):
        def invoke(*args):
            order.append(name)
            return function(*args)
        return invoke

    result = pipeline._extract_post(
        record("classifier", lambda _: {"should_deep_analyze": True}),
        record("extraction", lambda _: knowledge),
        record("validation", validate_knowledge),
        "test",
        "post",
        CONTENT,
        record("relations", lambda c, k: extract_event_relations(c, k, relation_model)),
    )

    assert order == ["classifier", "extraction", "validation"]
    assert result["knowledge"] == validate_knowledge(CONTENT, base(), "test", "post")
    relation_model.assert_not_called()


@pytest.mark.parametrize("enabled,deep", [(False, True), (True, False)])
def test_skip_and_entity_only_do_not_call_modules(enabled, deep):
    relations = Mock()
    with patch.object(pipeline, "KNOWLEDGE_PIPELINE_ENABLED", enabled):
        pipeline._extract_post(
            lambda _: {"should_deep_analyze": deep},
            lambda _: base(),
            validate_knowledge,
            "test",
            "post",
            CONTENT,
            relations,
        )
    relations.assert_not_called()


def test_module_failure_keeps_committed_base():
    relations = Mock(side_effect=ValueError("invalid model response"))
    session = Mock()
    session.execute_write.return_value = {
        "entities": 1,
        "events": 2,
        "event_relations": 0,
    }
    with patch.dict(pipeline.KNOWLEDGE_MODULES, {"EVENT_RELATION": True}), \
            patch.object(pipeline, "_load_posts", return_value=[{
                "platform": "test", "post_id": "post", "content": CONTENT,
            }]), patch.object(pipeline, "create_knowledge_schema"):
        summary = pipeline.process_new_posts(
            session,
            extract_knowledge_fn=lambda _: base(),
            classify_post_fn=lambda _: {"should_deep_analyze": True},
            extract_event_relations_fn=relations,
        )

    assert summary["deep"] == 1
    relations.assert_called_once()
    assert session.execute_write.call_count == 1
    assert session.execute_write.call_args.args[0] is pipeline.save_knowledge_tx
    assert session.execute_write.call_args.kwargs["runnable_modules"] == {"EVENT_RELATION"}


def test_legacy_consolidation_imports_point_to_new_implementation():
    import knowledge_consolidation as legacy
    from knowledge_relations import event_hierarchy

    assert legacy.consolidate_pending_mentions is event_hierarchy.consolidate_pending_mentions
    assert legacy._merge_events is event_hierarchy._merge_events


def test_entrypoint_passes_shared_model_to_event_relation_module():
    from scripts import extract_entities as entrypoint

    model = Mock()
    with patch.object(entrypoint, "_process_new_posts", return_value={}) as process, \
            patch.object(entrypoint, "extract_event_relations") as relations:
        entrypoint.process_new_posts(Mock(), call_model=model)
        process.call_args.kwargs["extract_event_relations_fn"](CONTENT, base())
    relations.assert_called_once_with(CONTENT, base(), call_model=model)


def test_event_relation_stage_is_identified_in_usage_logs():
    from knowledge_gemini import _stage_for_schema
    from knowledge_settings import EVENT_RELATION_SCHEMA

    assert _stage_for_schema(EVENT_RELATION_SCHEMA) == "event_relation"
