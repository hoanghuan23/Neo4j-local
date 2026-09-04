import copy
import unittest
from unittest.mock import Mock

from knowledge_relations.participant_role import enrich_participant_roles


CONTENT = "Người bảo vệ hành hung thanh niên shipper trước cửa hàng."


def knowledge():
    return {
        "entities": [{"local_id": "e1", "name": "Cửa hàng"}],
        "events": [
            {
                "local_id": "ev1",
                "description": "Người bảo vệ hành hung thanh niên shipper.",
                "evidence_text": "Người bảo vệ hành hung thanh niên shipper",
                "participants": [
                    {
                        "entity_id": None,
                        "participant_text": "Người bảo vệ",
                        "participant_scope": "POST_LOCAL",
                        "role": "PARTICIPANT",
                        "confidence": 0.8,
                    },
                    {
                        "entity_id": None,
                        "participant_text": "thanh niên shipper",
                        "participant_scope": "POST_LOCAL",
                        "role": "ACTOR",
                        "confidence": 0.7,
                    },
                ],
            }
        ],
        "event_relations": [],
    }


def routes(action="ENRICH"):
    return {
        "event_routes": [
            {
                "event_id": "ev1",
                "route_details": [
                    {
                        "relation_group": "PARTICIPANT_ROLE",
                        "action": action,
                    }
                ],
            }
        ],
        "pair_routes": [],
    }


class ParticipantRoleTests(unittest.TestCase):
    def test_use_base_data_returns_same_object_without_model_call(self):
        base = knowledge()
        call_model = Mock()

        result = enrich_participant_roles(
            CONTENT, base, routes("USE_BASE_DATA"), call_model=call_model
        )

        self.assertIs(result, base)
        call_model.assert_not_called()

    def test_enriches_roles_by_index_and_preserves_every_other_field(self):
        base = knowledge()
        before = copy.deepcopy(base)
        call_model = Mock(
            return_value={
                "assignments": [
                    {
                        "event_id": "ev1",
                        "participant_index": 0,
                        "role": "ACTOR",
                        "evidence_text": "Người bảo vệ hành hung",
                    },
                    {
                        "event_id": "ev1",
                        "participant_index": 1,
                        "role": "VICTIM",
                        "evidence_text": "thanh niên shipper",
                    },
                ]
            }
        )

        result = enrich_participant_roles(
            CONTENT, base, routes(), call_model=call_model
        )

        self.assertEqual(
            [item["role"] for item in result["events"][0]["participants"]],
            ["ACTOR", "VICTIM"],
        )
        self.assertEqual(base, before)
        for index in range(2):
            expected = dict(before["events"][0]["participants"][index])
            actual = dict(result["events"][0]["participants"][index])
            expected.pop("role")
            actual.pop("role")
            self.assertEqual(actual, expected)

    def test_ignores_invalid_duplicate_and_unrouted_assignments(self):
        base = knowledge()
        call_model = Mock(
            return_value={
                "assignments": [
                    {
                        "event_id": "ev1",
                        "participant_index": 0,
                        "role": "ACTOR",
                        "evidence_text": "Người bảo vệ",
                    },
                    {
                        "event_id": "ev1",
                        "participant_index": 0,
                        "role": "VICTIM",
                        "evidence_text": "Người bảo vệ",
                    },
                    {
                        "event_id": "ev1",
                        "participant_index": 9,
                        "role": "TARGET",
                        "evidence_text": "thanh niên shipper",
                    },
                    {
                        "event_id": "ev1",
                        "participant_index": 1,
                        "role": "TARGET",
                        "evidence_text": "evidence bịa",
                    },
                    {
                        "event_id": "ev2",
                        "participant_index": 0,
                        "role": "ACTOR",
                        "evidence_text": "Người bảo vệ",
                    },
                ]
            }
        )

        result = enrich_participant_roles(
            CONTENT, base, routes(), call_model=call_model
        )

        self.assertEqual(
            [item["role"] for item in result["events"][0]["participants"]],
            ["ACTOR", "ACTOR"],
        )

    def test_model_failure_and_invalid_output_keep_base_data(self):
        for output in (ValueError("failed"), {}, {"assignments": "bad"}):
            with self.subTest(output=output):
                base = knowledge()
                call_model = Mock(
                    side_effect=output if isinstance(output, Exception) else None,
                    return_value=output if not isinstance(output, Exception) else None,
                )
                result = enrich_participant_roles(
                    CONTENT, base, routes(), call_model=call_model
                )
                self.assertIs(result, base)


if __name__ == "__main__":
    unittest.main()
