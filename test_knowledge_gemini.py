import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from knowledge_gemini import GeminiKnowledgeCaller


class FakeTypes:
    @staticmethod
    def GenerateContentConfig(**kwargs):
        return kwargs


class GeminiKnowledgeCallerTests(unittest.TestCase):
    def test_accumulates_actual_usage_and_calculates_standard_cost(self):
        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(
                text=json.dumps(
                    {"entities": [], "events": [], "event_relations": []}
                ),
                usage_metadata=SimpleNamespace(
                    prompt_token_count=1_000_000,
                    candidates_token_count=100_000,
                    thoughts_token_count=200_000,
                ),
            ),
            SimpleNamespace(
                text=json.dumps(
                    {"entities": [], "events": [], "event_relations": []}
                ),
                usage_metadata=SimpleNamespace(
                    prompt_token_count=20,
                    candidates_token_count=30,
                    thoughts_token_count=None,
                ),
            ),
        ]
        caller = GeminiKnowledgeCaller(
            client=client,
            types_module=FakeTypes,
        )

        caller("post one", {})
        caller("post two", {})

        self.assertEqual(caller.usage.requests, 2)
        self.assertEqual(caller.usage.input_tokens, 1_000_020)
        self.assertEqual(caller.usage.output_tokens, 100_030)
        self.assertEqual(caller.usage.thinking_tokens, 200_000)

        output = io.StringIO()
        with patch("sys.stdout", output):
            caller.print_cost_summary(target_posts=50)

        summary = output.getvalue()
        self.assertIn("Số request có usage thực tế: 2/50", summary)
        self.assertIn("Chi phí input (Standard): $0.25000500", summary)
        self.assertIn("Chi phí output (Standard): $0.45004500", summary)
        self.assertIn("TỔNG CHI PHÍ (USD, Standard): $0.70005000", summary)

    def test_requires_usage_metadata(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text="{}",
            usage_metadata=None,
        )
        caller = GeminiKnowledgeCaller(
            client=client,
            types_module=FakeTypes,
        )

        with self.assertRaisesRegex(ValueError, "usage_metadata"):
            caller("post", {})


if __name__ == "__main__":
    unittest.main()
