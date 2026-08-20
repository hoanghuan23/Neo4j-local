import json
import unittest
from unittest.mock import Mock, patch

import knowledge_tracing as subject


class KnowledgeTracingTests(unittest.TestCase):
    def test_formats_llm_inputs_and_outputs_for_langsmith(self):
        self.assertEqual(
            subject._llm_inputs({"prompt": "xin chào", "output_schema": {}}),
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "xin chào"}],
                    }
                ]
            },
        )
        traced = subject._llm_outputs({"entities": []})
        content = traced["choices"][0]["message"]["content"]
        self.assertEqual(json.loads(content), {"entities": []})

    @patch.object(subject, "get_current_run_tree")
    def test_attaches_reasoning_token_breakdown(self, current_run):
        run = Mock()
        current_run.return_value = run

        subject.set_langsmith_usage(
            input_tokens=10,
            output_tokens=20,
            reasoning_tokens=5,
        )

        run.set.assert_called_once_with(
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 25,
                "total_tokens": 35,
                "output_token_details": {"text": 20, "reasoning": 5},
            }
        )


if __name__ == "__main__":
    unittest.main()
