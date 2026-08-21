import threading
import unittest
from unittest.mock import Mock, patch

import knowledge_pipeline as subject


class KnowledgePipelineConcurrencyTests(unittest.TestCase):
    @patch.object(subject, "create_knowledge_schema")
    @patch.object(subject, "validate_knowledge")
    @patch.object(subject, "_load_posts")
    def test_extracts_with_two_workers_but_writes_on_main_thread(
        self,
        load_posts,
        validate_knowledge,
        create_schema,
    ):
        load_posts.return_value = [
            {"platform": "facebook", "post_id": "1", "content": "one"},
            {"platform": "tiktok", "post_id": "2", "content": "two"},
        ]
        validate_knowledge.return_value = {
            "entities": [],
            "events": [],
            "event_relations": [],
        }
        barrier = threading.Barrier(2)
        extraction_threads = set()

        def extract(content):
            extraction_threads.add(threading.get_ident())
            barrier.wait(timeout=2)
            return {"entities": [], "events": [], "event_relations": []}

        main_thread = threading.get_ident()
        write_threads = []
        session = Mock()

        def execute_write(*args):
            write_threads.append(threading.get_ident())
            return {"entities": 0, "events": 0, "event_relations": 0}

        session.execute_write.side_effect = execute_write

        with patch.object(subject, "KNOWLEDGE_WORKERS", 2):
            summary = subject.process_new_posts(
                session,
                extract_knowledge_fn=extract,
                classify_post_fn=lambda _content: {
                    "has_entity_candidate": True,
                    "has_event_candidate": False,
                },
            )

        self.assertEqual(len(extraction_threads), 2)
        self.assertNotIn(main_thread, extraction_threads)
        self.assertEqual(write_threads, [main_thread, main_thread])
        self.assertEqual(summary["deep"], 2)
        create_schema.assert_called_once_with(session)

    @patch.object(subject, "create_knowledge_schema")
    @patch.object(subject, "validate_knowledge")
    @patch.object(subject, "_load_posts")
    def test_classifier_skip_does_not_run_deep_extraction(
        self,
        load_posts,
        validate_knowledge,
        _create_schema,
    ):
        load_posts.return_value = [
            {"platform": "facebook", "post_id": "1", "content": "a caption"}
        ]
        validate_knowledge.return_value = {
            "entities": [],
            "events": [],
            "event_relations": [],
            "generic_entity_keys": [],
        }
        extract = Mock()
        session = Mock()
        session.execute_write.return_value = {
            "entities": 0,
            "events": 0,
            "event_relations": 0,
        }

        summary = subject.process_new_posts(
            session,
            extract_knowledge_fn=extract,
            classify_post_fn=lambda _content: {
                "has_entity_candidate": False,
                "has_event_candidate": False,
            },
        )

        extract.assert_not_called()
        self.assertEqual(summary["skipped"], 1)
        save_call = session.execute_write.call_args.args
        self.assertEqual(save_call[-1], "SKIPPED")

    def test_extract_post_runs_deep_for_either_candidate_type(self):
        for classification in (
            {"has_entity_candidate": True, "has_event_candidate": False},
            {"has_entity_candidate": False, "has_event_candidate": True},
        ):
            with self.subTest(classification=classification):
                extract = Mock(return_value={"entities": [], "events": []})
                result = subject._extract_post(
                    lambda _content: classification,
                    extract,
                    "facebook",
                    "1",
                    "content",
                )

                extract.assert_called_once_with("content")
                self.assertEqual(result["classifier_decision"], "DEEP")

    def test_load_posts_applies_configured_limit(self):
        session = Mock()
        session.run.return_value = []

        with (
            patch.object(subject, "KNOWLEDGE_PIPELINE_ENABLED", True),
            patch.object(subject, "POST_LIMIT", 30),
        ):
            subject._load_posts(session)

        query = session.run.call_args.args[0]
        self.assertIn("LIMIT $post_limit", query)
        self.assertEqual(session.run.call_args.kwargs["post_limit"], 30)

    def test_load_posts_prioritizes_hot_metric_tier_in_both_modes(self):
        for pipeline_enabled in (True, False):
            with self.subTest(pipeline_enabled=pipeline_enabled):
                session = Mock()
                session.run.return_value = []

                with patch.object(
                    subject,
                    "KNOWLEDGE_PIPELINE_ENABLED",
                    pipeline_enabled,
                ):
                    subject._load_posts(session)

                query = session.run.call_args.args[0]
                hot_priority = (
                    "toLower(trim(coalesce(p.metric_tier, ''))) = 'hot'"
                )
                self.assertIn(hot_priority, query)
                self.assertLess(
                    query.index(hot_priority),
                    query.index("p.posted_at DESC"),
                )


if __name__ == "__main__":
    unittest.main()
