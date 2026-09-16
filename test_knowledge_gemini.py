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

    @staticmethod
    def AutomaticFunctionCallingConfig(**kwargs):
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

        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(
            config["automatic_function_calling"],
            {"disable": True},
        )

        self.assertEqual(caller.usage.requests, 2)
        self.assertEqual(caller.usage.input_tokens, 1_000_020)
        self.assertEqual(caller.usage.output_tokens, 100_030)
        self.assertEqual(caller.usage.thinking_tokens, 200_000)

        output = io.StringIO()
        with patch("sys.stdout", output):
            caller.print_cost_summary(target_posts=50)

        summary = output.getvalue()
        self.assertIn("Số request có usage thực tế: 2", summary)
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


def test_stage_logs_and_failed_json_usage(caplog, capsys):
    from knowledge_settings import KNOWLEDGE_SCHEMA, EVENT_TITLE_SCHEMA
    caplog.set_level('INFO', logger='knowledge.api')
    client = Mock()
    metadata = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=3)
    client.models.generate_content.side_effect = [
        SimpleNamespace(text='{}', usage_metadata=metadata),
        SimpleNamespace(text='not json', usage_metadata=metadata),
        RuntimeError('network failure'),
    ]
    caller = GeminiKnowledgeCaller(client=client, types_module=FakeTypes)
    caller('private content', KNOWLEDGE_SCHEMA)
    import pytest
    with pytest.raises(ValueError):
        caller('private content', EVENT_TITLE_SCHEMA)
    with pytest.raises(RuntimeError):
        caller('private content', EVENT_TITLE_SCHEMA)
    caller.print_cost_summary(target_posts=1)
    summary = capsys.readouterr().out
    assert 'THEO BƯỚC' not in summary
    assert 'extraction:' not in summary
    assert 'title:' not in summary
    assert 'Tổng lần gọi API: 3' in summary
    assert 'Request/post' not in summary
    assert caller.usage.requests == 2
    assert caller.usage.input_tokens == 200
    assert caller.usage.billable_output_tokens == 46
    assert 'CHI PHÍ RIÊNG extract_knowledge (1 lần gọi API)' in summary
    assert 'TỔNG CHI PHÍ extract_knowledge: $0.00005950' in summary
    assert 'function=extract_knowledge' not in caplog.text
    assert 'stage=title' not in caplog.text
    assert 'private content' not in caplog.text


def test_summary_totals_only_deep_extraction_calls(capsys):
    from knowledge_settings import KNOWLEDGE_CLASSIFIER_SCHEMA, KNOWLEDGE_SCHEMA

    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text='{}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20, thoughts_token_count=3,
        ),
    )
    caller = GeminiKnowledgeCaller(client=client, types_module=FakeTypes)
    for index in range(100):
        caller('classify', KNOWLEDGE_CLASSIFIER_SCHEMA)
        if index < 80:
            caller('extract', KNOWLEDGE_SCHEMA)

    caller.print_cost_summary(target_posts=100)
    summary = capsys.readouterr().out
    assert 'CHI PHÍ RIÊNG extract_knowledge (80 lần gọi API)' in summary
    assert 'Chi phí input extract_knowledge: $0.00200000' in summary
    assert 'Chi phí output extract_knowledge (gồm thinking): $0.00276000' in summary
    assert 'TỔNG CHI PHÍ extract_knowledge: $0.00476000' in summary
    assert 'TỔNG CHI PHÍ (USD, Standard): $0.01071000' in summary


def test_post_context_is_thread_local_and_resets():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from knowledge_gemini import log_post_calls, _POST_CONTEXT
    barrier = Barrier(2)

    @log_post_calls
    def process(platform, post_id):
        barrier.wait(timeout=5)
        return _POST_CONTEXT.get()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(process, 'facebook', 'p1')
        second = executor.submit(process, 'facebook', 'p2')
        assert first.result() == 'facebook:p1'
        assert second.result() == 'facebook:p2'
    assert _POST_CONTEXT.get() == 'batch'
