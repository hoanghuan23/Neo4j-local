import unittest
from unittest.mock import Mock

from knowledge_relation_router import (
    classify_relation_routes,
    normalize_relation_routes,
)


CONTENT = (
    "Bộ Công an cho biết mưa lớn gây ngập tại quận Thanh Xuân vào ngày 2/9. "
    "Người dân phản đối việc đóng đường."
)


def event(local_id, evidence, participants=None):
    return {
        "local_id": local_id,
        "type": "OTHER",
        "title": local_id,
        "description": evidence,
        "evidence_text": evidence,
        "time_expression": None,
        "participants": participants or [],
    }


class RelationRouterTests(unittest.TestCase):
    def test_does_not_call_model_without_events(self):
        call_model = Mock()

        result = classify_relation_routes(
            CONTENT,
            {"entities": [], "events": []},
            call_model=call_model,
        )

        self.assertEqual(result, {"event_routes": [], "pair_routes": []})
        call_model.assert_not_called()

    def test_calls_model_once_and_passes_strict_schema(self):
        knowledge = {
            "entities": [],
            "events": [event("ev1", "mưa lớn gây ngập")],
        }
        call_model = Mock(return_value={"event_routes": [], "pair_routes": []})

        result = classify_relation_routes(
            CONTENT,
            knowledge,
            call_model=call_model,
        )

        self.assertEqual(result["event_routes"][0]["event_id"], "ev1")
        self.assertEqual(call_model.call_count, 1)
        prompt = call_model.call_args.args[0]
        self.assertIn("không chọn chỉ vì xảy ra trước/sau", prompt)
        self.assertIn(
            "Phân biệt stance của tác giả Post với stance của người được trích dẫn",
            prompt,
        )
        schema = call_model.call_args.args[1]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {"event_routes", "pair_routes"})

    def test_normalizes_invalid_missing_and_duplicate_routes(self):
        knowledge = {
            "entities": [],
            "events": [
                event(
                    "ev1",
                    "Bộ Công an cho biết mưa lớn gây ngập",
                    [{"entity_id": "e1", "role": "SPEAKER"}],
                ),
                event("ev2", "Người dân phản đối việc đóng đường"),
            ],
        }
        causal = {
            "relation_group": "CAUSAL_RELATION",
            "action": "USE_BASE_DATA",
            "reason": "Có từ gây.",
            "evidence_text": "mưa lớn gây ngập",
        }
        raw = {
            "event_routes": [
                {
                    "event_id": "ev1",
                    "relation_groups": ["CAUSAL_RELATION", "CLAIM_PROVENANCE"],
                    "route_details": [
                        {
                            "relation_group": "CAUSAL_RELATION",
                            "action": "ENRICH",
                            "reason": "Sai scope.",
                            "evidence_text": "mưa lớn gây ngập",
                        },
                        {
                            "relation_group": "CLAIM_PROVENANCE",
                            "action": "USE_BASE_DATA",
                            "reason": "Có nguồn phát biểu.",
                            "evidence_text": "Bộ Công an cho biết",
                        },
                        {
                            "relation_group": "TEMPORAL_RELATION",
                            "action": "ENRICH",
                            "reason": "Evidence bịa.",
                            "evidence_text": "ngày không tồn tại",
                        },
                    ],
                },
                {
                    "event_id": "unknown",
                    "relation_groups": ["TEMPORAL_RELATION"],
                    "route_details": [],
                },
            ],
            "pair_routes": [
                {
                    "event_a_id": "ev2",
                    "event_b_id": "ev1",
                    "relation_groups": ["CAUSAL_RELATION"],
                    "route_details": [causal],
                },
                {
                    "event_a_id": "ev1",
                    "event_b_id": "ev2",
                    "relation_groups": ["CAUSAL_RELATION"],
                    "route_details": [causal],
                },
                {
                    "event_a_id": "ev1",
                    "event_b_id": "ev1",
                    "relation_groups": ["CAUSAL_RELATION"],
                    "route_details": [causal],
                },
            ],
        }

        result = normalize_relation_routes(CONTENT, knowledge, raw)

        self.assertEqual(
            [route["event_id"] for route in result["event_routes"]],
            ["ev1", "ev2"],
        )
        ev1 = result["event_routes"][0]
        self.assertEqual(
            ev1["relation_groups"],
            ["CLAIM_PROVENANCE", "PARTICIPANT_ROLE"],
        )
        self.assertEqual(ev1["route_details"][0]["action"], "ENRICH")
        self.assertEqual(ev1["route_details"][1]["action"], "USE_BASE_DATA")
        self.assertEqual(result["event_routes"][1]["relation_groups"], [])
        self.assertEqual(len(result["pair_routes"]), 1)
        pair = result["pair_routes"][0]
        self.assertEqual((pair["event_a_id"], pair["event_b_id"]), ("ev1", "ev2"))
        self.assertEqual(pair["route_details"][0]["action"], "ENRICH")

    def test_participant_fallback_is_marked_for_enrichment(self):
        knowledge = {
            "entities": [],
            "events": [
                event(
                    "ev1",
                    "Người dân phản đối việc đóng đường",
                    [{"entity_id": None, "role": "PARTICIPANT"}],
                )
            ],
        }

        result = normalize_relation_routes(
            CONTENT,
            knowledge,
            {"event_routes": [], "pair_routes": []},
        )

        detail = result["event_routes"][0]["route_details"][0]
        self.assertEqual(detail["relation_group"], "PARTICIPANT_ROLE")
        self.assertEqual(detail["action"], "ENRICH")


if __name__ == "__main__":
    unittest.main()
