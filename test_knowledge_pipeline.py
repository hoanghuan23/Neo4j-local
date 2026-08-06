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
            subject.process_new_posts(session, extract_knowledge_fn=extract)

        self.assertEqual(len(extraction_threads), 2)
        self.assertNotIn(main_thread, extraction_threads)
        self.assertEqual(write_threads, [main_thread, main_thread])
        create_schema.assert_called_once_with(session)

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


if __name__ == "__main__":
    unittest.main()
