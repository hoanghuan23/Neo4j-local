import os
import unittest
from unittest.mock import Mock, patch

import extract_entities_ollama as subject


class NormalizationTests(unittest.TestCase):
    def test_normalize_name_preserves_accents_and_collapses_whitespace(self):
        self.assertEqual(subject.normalize_name("  ĐÀ   NẴNG  "), "đà nẵng")

    def test_search_name_removes_vietnamese_accents(self):
        self.assertEqual(subject.make_search_name("Đặng Thái Sơn"), "dang thai son")

    def test_normalize_null_handles_model_null_strings_recursively(self):
        self.assertEqual(
            subject.normalize_null(
                {"status": "null", "participant": {"entity_id": " NONE "}}
            ),
            {"status": None, "participant": {"entity_id": None}},
        )


class OllamaClientTests(unittest.TestCase):
    @patch.object(subject.requests, "post")
    def test_empty_model_response_logs_metadata_and_raises_clear_error(self, post):
        response = post.return_value
        response.status_code = 200
        response.content = b'{"response":""}'
        response.json.return_value = {
            "model": subject.OLLAMA_MODEL,
            "response": "",
            "done": True,
            "done_reason": "stop",
        }

        with self.assertLogs(subject.LOGGER, level="ERROR") as logs:
            with self.assertRaisesRegex(ValueError, "response rỗng"):
                subject.call_ollama("prompt", {})

        output = "\n".join(logs.output)
        self.assertIn("reason=stop", output)
        self.assertNotIn("context", output)

    @patch.object(subject.requests, "post")
    def test_invalid_model_json_logs_raw_response(self, post):
        response = post.return_value
        response.status_code = 200
        response.content = b'{"response":"not-json"}'
        response.json.return_value = {"response": "not-json", "done": True}

        with self.assertLogs(subject.LOGGER, level="ERROR") as logs:
            with self.assertRaisesRegex(ValueError, "không phải JSON hợp lệ"):
                subject.call_ollama("prompt", {})

        self.assertIn("raw_response_preview='not-json'", "\n".join(logs.output))
        self.assertFalse(post.call_args.kwargs["json"]["think"])
        self.assertEqual(post.call_count, subject.OLLAMA_MAX_ATTEMPTS)


class ExtractionTests(unittest.TestCase):
    def test_entity_schema_contains_valid_event_fields(self):
        properties = subject.ENTITY_SCHEMA["properties"]
        event_schema = properties["events"]["items"]
        event_properties = event_schema["properties"]

        self.assertEqual(
            subject.ENTITY_SCHEMA["required"],
            ["entities", "events", "event_relations"],
        )
        self.assertIn("event_relations", properties)
        self.assertNotIn("event_realations", properties)
        self.assertNotIn("start_year", event_properties)
        self.assertNotIn("end_year", event_properties)
        self.assertNotIn("start_years", event_properties)
        self.assertEqual(event_properties["confidence"]["minimum"], 0)
        self.assertEqual(event_properties["confidence"]["maximum"], 1)
        self.assertEqual(properties["events"]["maxItems"], 5)

    def test_knowledge_schema_is_strict_and_uses_bounded_enums(self):
        schema = subject.KNOWLEDGE_SCHEMA
        event = schema["properties"]["events"]["items"]
        participant = event["properties"]["participants"]["items"]
        relation = schema["properties"]["event_relations"]["items"]

        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(event["additionalProperties"])
        self.assertFalse(participant["additionalProperties"])
        self.assertEqual(
            set(event["properties"]["status"]["enum"]),
            subject.EVENT_STATUSES,
        )
        self.assertEqual(
            set(relation["properties"]["type"]["enum"]),
            subject.EVENT_RELATION_TYPES,
        )
        self.assertIn("evidence_text", event["required"])
        self.assertIn("participant_text", participant["required"])

    @patch.object(subject, "call_ollama")
    def test_extract_knowledge_returns_all_sections(self, call_ollama):
        expected = {
            "entities": [],
            "events": [
                {
                    "local_id": "ev1",
                    "type": "MEETING",
                    "description": "A meeting",
                    "status": None,
                    "time_expression": None,
                    "confidence": 0.8,
                    "participants": [],
                }
            ],
            "event_relations": [],
        }
        call_ollama.return_value = expected

        self.assertEqual(subject.extract_knowledge("A meeting happened."), expected)
        self.assertIs(call_ollama.call_args.args[1], subject.ENTITY_SCHEMA)

    @patch.object(subject, "call_ollama")
    def test_prompt_rejects_non_events_and_limits_event_count(self, call_ollama):
        call_ollama.return_value = {
            "entities": [],
            "events": [],
            "event_relations": [],
        }

        subject.extract_knowledge(
            "Bộ phim mà mình cực mong chờ phần 2 mà chưa thấy, "
            "bác nào biết phim tương tự k ạ"
        )

        prompt = call_ollama.call_args.args[0]
        self.assertIn("KHÔNG tạo Event cho cảm xúc, mong muốn, sở thích", prompt)
        self.assertIn("Tối đa 5 Event", prompt)
        self.assertIn("hất/tạt/ném vào người là ASSAULT", prompt)
        self.assertIn("trả events là []", prompt)
        self.assertIn('"ông Đoàn Bảo Châu" phải tạo Entity', prompt)
        self.assertIn("bắt buộc dùng entity_id trỏ tới Entity", prompt)
        self.assertIn("Không bao giờ đặt họ tên đầy đủ", prompt)
        self.assertIn("rà lần lượt từng câu trong toàn bộ văn bản", prompt)
        self.assertIn("không tham gia Event", prompt)
        self.assertIn("số tiền, mức phạt, số", prompt)
        self.assertIn("Có thể lấy nhiều câu liền kề", prompt)
        self.assertIn("35 triệu đồng và trừ 10 điểm", prompt)
        self.assertIn("Cơ quan A là ACTOR", prompt)
        self.assertIn("KIỂM TRA ID TRƯỚC KHI TRẢ JSON", prompt)
        self.assertIn('participant_text = "lực lượng tìm kiếm"', prompt)
        self.assertIn("tuyệt đối không tạo tham chiếu e2", prompt)

    @patch.object(subject, "call_ollama")
    def test_extract_entities_uses_canonical_schema(self, call_ollama):
        expected = [
            {
                "name": "President Trump",
                "canonical_name": "Donald Trump",
                "type": "PERSON",
                "resolution_confidence": "HIGH",
            }
        ]
        call_ollama.return_value = {"entities": expected}

        self.assertEqual(subject.extract_entities("President Trump spoke."), expected)
        self.assertIs(call_ollama.call_args.args[1], subject.ENTITY_SCHEMA)

    def test_high_confidence_aliases_share_one_key(self):
        aliases = [
            "Trump",
            "Tổng thống Trump",
            "Donald Trump",
            "President Donald Trump",
        ]
        keys = {
            subject.prepare_entity(
                {
                    "name": alias,
                    "canonical_name": "Donald Trump",
                    "type": "PERSON",
                    "resolution_confidence": "HIGH",
                }
            )["normalized_name"]
            for alias in aliases
        }
        self.assertEqual(keys, {"donald trump"})

    def test_low_confidence_does_not_use_inferred_canonical_name(self):
        entity = subject.prepare_entity(
            {
                "name": "Trump",
                "canonical_name": "Donald Trump",
                "type": "PERSON",
                "resolution_confidence": "LOW",
            }
        )
        self.assertEqual(entity["normalized_name"], "trump")
        self.assertFalse(entity["is_canonical"])

    def test_other_trump_subjects_keep_distinct_keys(self):
        melania = subject.prepare_entity(
            {
                "name": "Melania Trump",
                "canonical_name": "Melania Trump",
                "type": "PERSON",
                "resolution_confidence": "HIGH",
            }
        )
        administration = subject.prepare_entity(
            {
                "name": "Trump administration",
                "canonical_name": "Trump administration",
                "type": "ORGANIZATION",
                "resolution_confidence": "HIGH",
            }
        )
        self.assertNotEqual(
            (melania["normalized_name"], melania["entity_type"]),
            (administration["normalized_name"], administration["entity_type"]),
        )

    def test_hashtags_and_handles_are_rejected(self):
        for name in ("@AnnaJonesSky", "#ZiaYusufUK"):
            with self.subTest(name=name):
                self.assertIsNone(
                    subject.prepare_entity(
                        {
                            "name": name,
                            "canonical_name": name,
                            "type": "PERSON",
                            "resolution_confidence": "HIGH",
                        }
                    )
                )

    def test_entity_is_rejected_when_only_canonical_name_has_marker(self):
        self.assertIsNone(
            subject.prepare_entity(
                {
                    "name": "Anna Jones",
                    "canonical_name": "@AnnaJonesSky",
                    "type": "PERSON",
                    "resolution_confidence": "HIGH",
                }
            )
        )


class KnowledgeValidationTests(unittest.TestCase):
    @staticmethod
    def entity(local_id, name, entity_type="PERSON", confidence="MEDIUM"):
        return {
            "local_id": local_id,
            "name": name,
            "canonical_name": name,
            "type": entity_type,
            "resolution_confidence": confidence,
        }

    @staticmethod
    def event(
        local_id,
        event_type,
        evidence,
        participants=None,
        status="COMPLETED",
        confidence=0.9,
    ):
        return {
            "local_id": local_id,
            "type": event_type,
            "description": evidence,
            "evidence_text": evidence,
            "status": status,
            "time_expression": None,
            "confidence": confidence,
            "participants": participants or [],
        }

    @staticmethod
    def participant(
        entity_id=None,
        participant_text=None,
        role="PARTICIPANT",
        confidence=0.8,
    ):
        return {
            "entity_id": entity_id,
            "participant_text": participant_text,
            "role": role,
            "confidence": confidence,
        }

    def test_generic_entities_are_removed_and_become_anonymous_participants(self):
        content = "A Maryland man pushed another man into Baltimore harbor."
        raw = {
            "entities": [
                self.entity("e1", "Maryland man"),
                self.entity("e2", "Baltimore", "LOCATION", "HIGH"),
            ],
            "events": [
                self.event(
                    "ev1",
                    "ASSAULT",
                    "A Maryland man pushed another man into Baltimore harbor",
                    [
                        self.participant("e1", role="ACTOR"),
                        self.participant(participant_text="another man", role="VICTIM"),
                        self.participant("e2", role="LOCATION"),
                    ],
                )
            ],
            "event_relations": [],
        }

        knowledge = subject.validate_knowledge(content, raw, "facebook", "post-1")

        self.assertEqual(
            [entity["local_id"] for entity in knowledge["entities"]], ["e2"]
        )
        actor = knowledge["events"][0]["participants"][0]
        self.assertIsNone(actor["entity_id"])
        self.assertEqual(actor["participant_text"], "Maryland man")
        self.assertEqual(actor["role"], "ACTOR")
        self.assertEqual(
            knowledge["generic_entity_keys"],
            [{"normalized_name": "maryland man", "type": "PERSON"}],
        )

    def test_conflicting_participant_text_drops_entity_reference(self):
        content = (
            "người bán bất ngờ hất cả xô nước về phía cụ bà 89 tuổi "
            "tại Đồng Nai"
        )
        raw = {
            "entities": [self.entity("e1", "Đồng Nai", "LOCATION", "HIGH")],
            "events": [
                self.event(
                    "ev1",
                    "ASSAULT",
                    content,
                    [
                        self.participant(
                            "e1", participant_text="cụ bà 89 tuổi", role="TARGET"
                        )
                    ],
                )
            ],
            "event_relations": [],
        }

        participant = subject.validate_knowledge(content, raw)["events"][0][
            "participants"
        ][0]

        self.assertIsNone(participant["entity_id"])
        self.assertEqual(participant["participant_text"], "cụ bà 89 tuổi")
        self.assertEqual(participant["role"], "TARGET")

    def test_matching_participant_text_keeps_entity_reference(self):
        content = "A storm hit Đồng Nai."
        raw = {
            "entities": [self.entity("e1", "Đồng Nai", "LOCATION", "HIGH")],
            "events": [
                self.event(
                    "ev1",
                    "ASSAULT",
                    content.rstrip("."),
                    [
                        self.participant(
                            "e1", participant_text="  ĐỒNG   NAI ", role="TARGET"
                        )
                    ],
                )
            ],
            "event_relations": [],
        }

        participant = subject.validate_knowledge(content, raw)["events"][0][
            "participants"
        ][0]

        self.assertEqual(participant["entity_id"], "e1")
        self.assertIsNone(participant["participant_text"])

    def test_canonical_participant_text_keeps_entity_reference(self):
        content = "Tổng thống Trump, tức Donald Trump, warned Iran."
        raw = {
            "entities": [
                {
                    **self.entity("e1", "Tổng thống Trump", confidence="HIGH"),
                    "canonical_name": "Donald Trump",
                }
            ],
            "events": [
                self.event(
                    "ev1",
                    "STATEMENT",
                    content.rstrip("."),
                    [
                        self.participant(
                            "e1", participant_text="Donald Trump", role="SPEAKER"
                        )
                    ],
                )
            ],
            "event_relations": [],
        }

        participant = subject.validate_knowledge(content, raw)["events"][0][
            "participants"
        ][0]

        self.assertEqual(participant["entity_id"], "e1")
        self.assertIsNone(participant["participant_text"])

    def test_deduplicated_entity_id_retains_its_raw_alias_for_participants(self):
        content = "President Trump warned Iran."
        raw = {
            "entities": [
                self.entity("e1", "Donald Trump", confidence="HIGH"),
                {
                    **self.entity("e2", "President Trump", confidence="HIGH"),
                    "canonical_name": "Donald Trump",
                },
            ],
            "events": [
                self.event(
                    "ev1",
                    "STATEMENT",
                    content.rstrip("."),
                    [
                        self.participant(
                            "e2", participant_text="President Trump", role="SPEAKER"
                        )
                    ],
                )
            ],
            "event_relations": [],
        }

        knowledge = subject.validate_knowledge(content, raw)
        participant = knowledge["events"][0]["participants"][0]

        self.assertEqual(
            [entity["local_id"] for entity in knowledge["entities"]], ["e1"]
        )
        self.assertEqual(participant["entity_id"], "e1")
        self.assertIsNone(participant["participant_text"])

    def test_all_known_generic_examples_and_event_names_are_rejected(self):
        entities = [
            self.entity("e1", "a man"),
            self.entity("e2", "Maryland man"),
            self.entity("e3", "the victim"),
            self.entity("e4", "a House panel", "ORGANIZATION"),
            self.entity("e5", "Italian community", "ORGANIZATION"),
            self.entity(
                "e6", "World Dog Surfing Championships", "ORGANIZATION", "HIGH"
            ),
            self.entity("e7", "Chicago Cubs", "ORGANIZATION", "HIGH"),
            self.entity("e8", "Ukraine", "ORGANIZATION", "HIGH"),
            self.entity("e9", "Manchester City", "ORGANIZATION", "HIGH"),
            self.entity("e10", "Camera", "ORGANIZATION", "HIGH"),
            self.entity("e11", "Camera an ninh", "ORGANIZATION", "HIGH"),
        ]

        validated = subject.validate_entities(entities)

        self.assertEqual(
            [entity["name"] for entity in validated["entities"]],
            ["Chicago Cubs", "Ukraine", "Manchester City"],
        )
        self.assertEqual(validated["entities"][1]["type"], "LOCATION")
        self.assertEqual(validated["entities"][2]["type"], "ORGANIZATION")

    def test_event_name_is_not_saved_as_anonymous_location_participant(self):
        content = "The Chicago Cubs competed in the World Dog Surfing Championships."
        raw = {
            "entities": [self.entity("e1", "Chicago Cubs", "ORGANIZATION", "HIGH")],
            "events": [
                self.event(
                    "ev1",
                    "SPORTS_EVENT",
                    content.rstrip("."),
                    [
                        self.participant("e1", role="ACTOR"),
                        self.participant(
                            participant_text="World Dog Surfing Championships",
                            role="LOCATION",
                        ),
                    ],
                )
            ],
            "event_relations": [],
        }

        participants = subject.validate_knowledge(content, raw)["events"][0][
            "participants"
        ]

        self.assertEqual(len(participants), 1)
        self.assertEqual(participants[0]["entity_id"], "e1")

    def test_invalid_status_becomes_unknown_and_invalid_confidence_is_rejected(self):
        content = "Pamela Anderson said her career returned."
        raw = {
            "entities": [],
            "events": [
                self.event(
                    "ev1",
                    "STATEMENT",
                    "Pamela Anderson said her career returned",
                    status="PAST",
                ),
                self.event(
                    "ev2",
                    "STATEMENT",
                    "Pamela Anderson said her career returned",
                    confidence=1.1,
                ),
            ],
            "event_relations": [],
        }

        knowledge = subject.validate_knowledge(content, raw)

        self.assertEqual(len(knowledge["events"]), 1)
        self.assertEqual(knowledge["events"][0]["status"], "UNKNOWN")

    def test_validation_corrects_event_type_from_single_matching_trigger(self):
        content = "người bán bất ngờ hất cả xô nước về phía người đàn ông"
        raw = {
            "entities": [],
            "events": [self.event("ev1", "STATEMENT", content)],
            "event_relations": [],
        }

        knowledge = subject.validate_knowledge(content, raw, "facebook", "post-1")
        correctly_typed = {
            **raw,
            "events": [self.event("ev1", "ASSAULT", content)],
        }
        expected = subject.validate_knowledge(
            content, correctly_typed, "facebook", "post-1"
        )

        self.assertEqual(len(knowledge["events"]), 1)
        self.assertEqual(knowledge["events"][0]["type"], "ASSAULT")
        self.assertEqual(
            knowledge["events"][0]["event_key"],
            expected["events"][0]["event_key"],
        )

    def test_validation_keeps_matching_model_type_when_evidence_has_many_actions(self):
        content = "Alice warned Bob and attacked him."
        raw = {
            "entities": [],
            "events": [self.event("ev1", "ASSAULT", content.rstrip("."))],
            "event_relations": [],
        }

        knowledge = subject.validate_knowledge(content, raw)

        self.assertEqual(knowledge["events"][0]["type"], "ASSAULT")

    def test_validation_keeps_model_type_when_many_other_types_match(self):
        content = "Alice warned Bob and attacked him."
        raw = {
            "entities": [],
            "events": [self.event("ev1", "MEETING", content.rstrip("."))],
            "event_relations": [],
        }

        knowledge = subject.validate_knowledge(content, raw)

        self.assertEqual(knowledge["events"][0]["type"], "MEETING")

    def test_validation_keeps_model_type_when_evidence_matches_no_type(self):
        content = "The incident occurred on Saturday in Bimini."
        raw = {
            "entities": [],
            "events": [self.event("ev1", "OTHER", content.rstrip("."))],
            "event_relations": [],
        }

        knowledge = subject.validate_knowledge(content, raw)

        self.assertEqual(len(knowledge["events"]), 1)
        self.assertEqual(knowledge["events"][0]["type"], "OTHER")

    def test_duplicate_event_is_removed_and_unverified_event_is_kept(self):
        content = (
            "A seaplane crashed in Bimini on Saturday. "
            "The incident occurred on Saturday in Bimini."
        )
        crash = self.event(
            "ev1", "ACCIDENT", "A seaplane crashed in Bimini on Saturday"
        )
        duplicate = dict(crash, local_id="ev2")
        context = self.event(
            "ev3", "OTHER", "The incident occurred on Saturday in Bimini"
        )
        raw = {
            "entities": [],
            "events": [crash, duplicate, context],
            "event_relations": [],
        }

        knowledge = subject.validate_knowledge(content, raw)

        self.assertEqual(
            [event["local_id"] for event in knowledge["events"]], ["ev1", "ev3"]
        )

    def test_validation_keeps_at_most_five_events(self):
        sentences = [f"Alice launched product {index}" for index in range(1, 7)]
        content = ". ".join(sentences) + "."
        raw = {
            "entities": [],
            "events": [
                self.event(f"ev{index}", "OTHER", sentence)
                for index, sentence in enumerate(sentences, start=1)
            ],
            "event_relations": [],
        }

        knowledge = subject.validate_knowledge(content, raw)

        self.assertEqual(
            [event["local_id"] for event in knowledge["events"]],
            ["ev1", "ev2", "ev3", "ev4", "ev5"],
        )

    def test_wrong_taxonomy_and_evidence_not_in_post_are_rejected(self):
        content = "Buster Olney spoke about the Chicago Cubs."
        raw = {
            "entities": [],
            "events": [
                self.event("ev1", "MATCH", "Buster Olney spoke about the Chicago Cubs"),
                self.event("ev2", "STATEMENT", "Buster Olney resigned"),
            ],
            "event_relations": [],
        }

        self.assertEqual(subject.validate_knowledge(content, raw)["events"], [])

    def test_event_key_is_stable_when_entity_local_id_changes(self):
        content = "Donald Trump warned Iran."

        def raw(entity_id):
            return {
                "entities": [self.entity(entity_id, "Donald Trump", confidence="HIGH")],
                "events": [
                    self.event(
                        "ev1",
                        "STATEMENT",
                        "Donald Trump warned Iran",
                        [self.participant(entity_id, role="SPEAKER")],
                    )
                ],
                "event_relations": [],
            }

        first = subject.validate_knowledge(content, raw("e1"), "facebook", "p1")
        second = subject.validate_knowledge(content, raw("person9"), "facebook", "p1")

        self.assertEqual(
            first["events"][0]["event_key"], second["events"][0]["event_key"]
        )

    def test_relations_require_valid_references_and_explicit_evidence(self):
        content = "Alice approved the plan before Bob announced it."
        raw = {
            "entities": [],
            "events": [
                self.event("ev1", "APPROVAL", "Alice approved the plan"),
                self.event("ev2", "STATEMENT", "Bob announced it"),
            ],
            "event_relations": [
                {
                    "source_event_id": "ev1",
                    "type": "PRECEDES",
                    "target_event_id": "ev2",
                    "evidence_text": "before Bob announced it",
                },
                {
                    "source_event_id": "ev1",
                    "type": "CAUSES",
                    "target_event_id": "ev2",
                    "evidence_text": "Alice approved the plan",
                },
                {
                    "source_event_id": "missing",
                    "type": "RELATED_TO",
                    "target_event_id": "ev2",
                    "evidence_text": "Bob announced it",
                },
            ],
        }

        relations = subject.validate_knowledge(content, raw)["event_relations"]

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["type"], "PRECEDES")

    def test_participant_requires_identity_and_confidence_in_range(self):
        content = "A man pushed the victim."
        raw = {
            "entities": [],
            "events": [
                self.event(
                    "ev1",
                    "ASSAULT",
                    "A man pushed the victim",
                    [
                        self.participant(role="ACTOR"),
                        self.participant(
                            participant_text="the victim",
                            role="VICTIM",
                            confidence=-0.1,
                        ),
                        self.participant(
                            participant_text="a man", role="ACTOR", confidence=0
                        ),
                    ],
                )
            ],
            "event_relations": [],
        }

        participants = subject.validate_knowledge(content, raw)["events"][0][
            "participants"
        ]
        self.assertEqual(len(participants), 1)
        self.assertEqual(participants[0]["participant_text"], "a man")
        self.assertEqual(participants[0]["confidence"], 0)

    def test_dangling_entity_id_recovers_unique_anonymous_participant(self):
        content = "nữ tài xế cho biết đã lùi xe và nữ tài xế bị phạt."
        raw = {
            "entities": [],
            "events": [
                self.event(
                    "ev1",
                    "STATEMENT",
                    "nữ tài xế cho biết đã lùi xe",
                    [self.participant("e3", role="SPEAKER", confidence=1.0)],
                ),
                self.event(
                    "ev2",
                    "OTHER",
                    "nữ tài xế bị phạt",
                    [self.participant("e3", role="TARGET", confidence=1.0)],
                ),
            ],
            "event_relations": [],
        }

        events = subject.validate_knowledge(content, raw)["events"]

        self.assertEqual(len(events), 2)
        self.assertEqual(
            [event["participants"][0]["participant_text"] for event in events],
            ["nữ tài xế", "nữ tài xế"],
        )
        self.assertTrue(
            all(event["participants"][0]["entity_id"] is None for event in events)
        )

    def test_dangling_id_recovers_search_force_as_anonymous_actor(self):
        content = (
            "Ngày 5/8, lực lượng tìm kiếm, quy tập hài cốt liệt sĩ "
            "tiếp tục phát hiện và quy tập 12 bộ hài cốt."
        )
        raw = {
            "entities": [],
            "events": [
                self.event(
                    "ev1",
                    "OTHER",
                    content,
                    [self.participant("e2", role="ACTOR", confidence=1.0)],
                )
            ],
            "event_relations": [],
        }

        event = subject.validate_knowledge(content, raw)["events"][0]

        self.assertEqual(len(event["participants"]), 1)
        self.assertEqual(
            event["participants"][0],
            {
                "entity_id": None,
                "participant_text": "lực lượng tìm kiếm",
                "role": "ACTOR",
                "confidence": 1.0,
            },
        )

    def test_anonymous_key_is_shared_across_events_and_roles_in_one_post(self):
        participant = {
            "entity_id": None,
            "participant_text": "nữ tài xế",
            "role": "SPEAKER",
            "confidence": 1.0,
        }
        first = subject.build_anonymous_participant_key(
            "facebook", "post-1", "event-1", participant
        )
        participant["role"] = "TARGET"
        second = subject.build_anonymous_participant_key(
            "facebook", "post-1", "event-2", participant
        )

        self.assertEqual(first, second)


class PersistenceTests(unittest.TestCase):
    def test_create_entity_schema_creates_constraint_and_index(self):
        session = Mock()
        session.run.return_value.consume.return_value = None

        subject.create_entity_schema(session)

        self.assertEqual(session.run.call_count, 2)
        self.assertIn("CREATE CONSTRAINT", session.run.call_args_list[0].args[0])
        self.assertIn("CREATE TEXT INDEX", session.run.call_args_list[1].args[0])

    def test_create_knowledge_schema_is_additive(self):
        session = Mock()
        session.run.return_value.consume.return_value = None

        subject.create_knowledge_schema(session)

        self.assertEqual(session.run.call_count, 4)
        queries = "\n".join(call.args[0] for call in session.run.call_args_list)
        self.assertIn("entity_identity_unique", queries)
        self.assertIn("event_key_unique", queries)
        self.assertIn("anonymous_participant_key_unique", queries)

    def test_save_entities_uses_canonical_key_and_alias(self):
        session = Mock()
        session.run.return_value.consume.return_value = None
        entities = [
            {
                "name": "Tổng thống Trump",
                "canonical_name": "Donald Trump",
                "type": "PERSON",
                "resolution_confidence": "HIGH",
            }
        ]

        saved_count = subject.save_entities(session, "facebook", "post-1", entities)

        self.assertEqual(saved_count, 1)
        save_call = session.run.call_args_list[0]
        self.assertEqual(save_call.kwargs["normalized_name"], "donald trump")
        self.assertEqual(save_call.kwargs["name"], "Tổng thống Trump")
        self.assertTrue(save_call.kwargs["is_canonical"])

    def test_upsert_event_removes_legacy_year_properties(self):
        tx = Mock()
        tx.run.return_value.consume.return_value = None
        event = {
            "event_key": "event-1",
            "type": "OTHER",
            "description": "nữ tài xế bị xử phạt",
            "evidence_text": "nữ tài xế bị xử phạt",
            "status": "COMPLETED",
            "time_expression": None,
            "confidence": 1.0,
            "participants": [],
        }

        subject.upsert_events(tx, "facebook", "post-1", [event], {})

        queries = "\n".join(call.args[0] for call in tx.run.call_args_list)
        self.assertIn("REMOVE event.start_year, event.end_year", queries)
        event_call = next(
            call for call in tx.run.call_args_list if "MERGE (event:Event" in call.args[0]
        )
        self.assertNotIn("start_year", event_call.kwargs)
        self.assertNotIn("end_year", event_call.kwargs)

    def test_save_knowledge_marks_success_inside_same_transaction(self):
        tx = Mock()
        tx.run.return_value.consume.return_value = None
        tx.run.return_value.single.return_value = {"post_count": 1}
        knowledge = {
            "entities": [],
            "events": [],
            "event_relations": [],
            "generic_entity_keys": [],
        }

        counts = subject.save_knowledge_tx(tx, "facebook", "post-1", knowledge)

        self.assertEqual(counts, {"entities": 0, "events": 0, "event_relations": 0})
        queries = "\n".join(call.args[0] for call in tx.run.call_args_list)
        self.assertIn("p.knowledge_processed = true", queries)
        self.assertIn("p.knowledge_retry_count = 0", queries)

    def test_mark_failure_increments_retry_without_marking_success(self):
        tx = Mock()
        tx.run.return_value.consume.return_value = None

        subject.mark_knowledge_failure(tx, "facebook", "post-1", "bad JSON")

        query = tx.run.call_args.args[0]
        self.assertIn("p.knowledge_processed = false", query)
        self.assertIn("coalesce(p.knowledge_retry_count, 0) + 1", query)
        self.assertIn("p.knowledge_retry_count = next_retry_count", query)
        self.assertEqual(tx.run.call_args.kwargs["knowledge_error"], "bad JSON")


@unittest.skipUnless(
    os.getenv("RUN_NEO4J_INTEGRATION") == "1",
    "set RUN_NEO4J_INTEGRATION=1 to exercise the local Neo4j instance",
)
class Neo4jIntegrationTests(unittest.TestCase):
    platform = "codex-test"
    post_id = "knowledge-pipeline-integration"
    content = "Smoke Test Organization said hello before a witness approved the plan."

    @classmethod
    def setUpClass(cls):
        cls.driver = subject.GraphDatabase.driver(
            subject.NEO4J_URI,
            auth=(subject.NEO4J_USER, subject.NEO4J_PASSWORD),
        )

    @classmethod
    def tearDownClass(cls):
        cls.driver.close()

    def cleanup(self, session):
        session.run(
            """
            MATCH (p:Post {platform: $platform, platform_id: $post_id})
                  -[:DESCRIBES]->(event:Event)
            DETACH DELETE event
            """,
            platform=self.platform,
            post_id=self.post_id,
        ).consume()
        session.run(
            """
            MATCH (p:Post {platform: $platform, platform_id: $post_id})
                  -[:HAS_ANONYMOUS_PARTICIPANT]->(anonymous:AnonymousParticipant)
            DETACH DELETE anonymous
            """,
            platform=self.platform,
            post_id=self.post_id,
        ).consume()
        session.run(
            """
            MATCH (p:Post {platform: $platform, platform_id: $post_id})
            DETACH DELETE p
            """,
            platform=self.platform,
            post_id=self.post_id,
        ).consume()
        session.run("""
            MATCH (entity:Entity {
                normalized_name: 'smoke test organization',
                type: 'ORGANIZATION'
            })
            WHERE NOT (entity)<-[:MENTIONS]-()
            DETACH DELETE entity
            """).consume()

    def setUp(self):
        with self.driver.session(database="neo4j") as session:
            self.cleanup(session)
            session.run(
                """
                CREATE (:Post {
                    platform: $platform,
                    platform_id: $post_id,
                    content: $content
                })
                """,
                platform=self.platform,
                post_id=self.post_id,
                content=self.content,
            ).consume()

    def tearDown(self):
        with self.driver.session(database="neo4j") as session:
            self.cleanup(session)

    def build_knowledge(self):
        raw = {
            "entities": [
                {
                    "local_id": "e1",
                    "name": "Smoke Test Organization",
                    "canonical_name": "Smoke Test Organization",
                    "type": "ORGANIZATION",
                    "resolution_confidence": "HIGH",
                }
            ],
            "events": [
                {
                    "local_id": "ev1",
                    "type": "STATEMENT",
                    "description": "Smoke Test Organization said hello",
                    "evidence_text": "Smoke Test Organization said hello",
                    "status": "COMPLETED",
                    "time_expression": None,
                    "confidence": 0.9,
                    "participants": [
                        {
                            "entity_id": "e1",
                            "participant_text": None,
                            "role": "SPEAKER",
                            "confidence": 0.9,
                        },
                    ],
                },
                {
                    "local_id": "ev2",
                    "type": "APPROVAL",
                    "description": "a witness approved the plan",
                    "evidence_text": "a witness approved the plan",
                    "status": "COMPLETED",
                    "time_expression": None,
                    "confidence": 0.85,
                    "participants": [
                        {
                            "entity_id": None,
                            "participant_text": "a witness",
                            "role": "ACTOR",
                            "confidence": 0.8,
                        }
                    ],
                },
            ],
            "event_relations": [
                {
                    "source_event_id": "ev1",
                    "type": "PRECEDES",
                    "target_event_id": "ev2",
                    "evidence_text": "before a witness approved the plan",
                }
            ],
        }
        knowledge = subject.validate_knowledge(
            self.content, raw, self.platform, self.post_id
        )
        return knowledge

    def test_save_knowledge_is_idempotent_with_all_relationships(self):
        knowledge = self.build_knowledge()

        with self.driver.session(database="neo4j") as session:
            session.execute_write(
                subject.save_knowledge_tx,
                self.platform,
                self.post_id,
                knowledge,
            )
            session.execute_write(
                subject.save_knowledge_tx,
                self.platform,
                self.post_id,
                knowledge,
            )
            record = session.run(
                """
                MATCH (p:Post {platform: $platform, platform_id: $post_id})
                OPTIONAL MATCH (p)-[:DESCRIBES]->(event:Event)
                WITH p, count(DISTINCT event) AS events
                OPTIONAL MATCH (p)-[:DESCRIBES]->(:Event)
                               -[participant:HAS_PARTICIPANT]->()
                WITH p, events, count(participant) AS participants
                OPTIONAL MATCH (p)-[:HAS_ANONYMOUS_PARTICIPANT]
                                 ->(anonymous:AnonymousParticipant)
                RETURN events,
                       participants,
                       count(DISTINCT anonymous) AS anonymous,
                       p.knowledge_processed AS processed
                """,
                platform=self.platform,
                post_id=self.post_id,
            ).single()
            relation_count = session.run(
                """
                MATCH (p:Post {platform: $platform, platform_id: $post_id})
                      -[:DESCRIBES]->(:Event)-[relation:PRECEDES]->(:Event)
                RETURN count(relation) AS relation_count
                """,
                platform=self.platform,
                post_id=self.post_id,
            ).single()["relation_count"]

        self.assertEqual(record["events"], 2)
        self.assertEqual(record["participants"], 2)
        self.assertEqual(record["anonymous"], 1)
        self.assertEqual(relation_count, 1)
        self.assertTrue(record["processed"])

    def test_transaction_rolls_back_all_knowledge_when_write_fails(self):
        knowledge = self.build_knowledge()

        def fail_after_write(tx):
            subject.save_knowledge_tx(tx, self.platform, self.post_id, knowledge)
            raise RuntimeError("force rollback")

        with self.driver.session(database="neo4j") as session:
            with self.assertRaisesRegex(RuntimeError, "force rollback"):
                session.execute_write(fail_after_write)
            record = session.run(
                """
                MATCH (p:Post {platform: $platform, platform_id: $post_id})
                OPTIONAL MATCH (p)-[:DESCRIBES]->(event:Event)
                OPTIONAL MATCH (p)-[:MENTIONS]->(entity:Entity)
                RETURN count(DISTINCT event) AS events,
                       count(DISTINCT entity) AS entities,
                       p.knowledge_processed AS processed
                """,
                platform=self.platform,
                post_id=self.post_id,
            ).single()

        self.assertEqual(record["events"], 0)
        self.assertEqual(record["entities"], 0)
        self.assertIsNone(record["processed"])


if __name__ == "__main__":
    unittest.main()
