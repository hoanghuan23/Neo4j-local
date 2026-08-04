import unittest
from unittest.mock import Mock, patch

import extract_entities_ollama as subject


class NormalizationTests(unittest.TestCase):
    def test_normalize_name_preserves_accents_and_collapses_whitespace(self):
        self.assertEqual(subject.normalize_name("  ĐÀ   NẴNG  "), "đà nẵng")

    def test_search_name_removes_vietnamese_accents(self):
        self.assertEqual(subject.make_search_name("Đặng Thái Sơn"), "dang thai son")


class ExtractionTests(unittest.TestCase):
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


class PersistenceTests(unittest.TestCase):
    def test_create_entity_schema_creates_constraint_and_index(self):
        session = Mock()
        session.run.return_value.consume.return_value = None

        subject.create_entity_schema(session)

        self.assertEqual(session.run.call_count, 2)
        self.assertIn("CREATE CONSTRAINT", session.run.call_args_list[0].args[0])
        self.assertIn("CREATE TEXT INDEX", session.run.call_args_list[1].args[0])

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


if __name__ == "__main__":
    unittest.main()
