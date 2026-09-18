import json
import threading
import logging
import time
import inspect
from contextvars import ContextVar
from functools import wraps
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from knowledge_settings import (
    GEMINI_API_KEY,
    GEMINI_INPUT_PRICE_PER_MILLION,
    GEMINI_MODEL,
    GEMINI_OUTPUT_PRICE_PER_MILLION,
    GEMINI_TIMEOUT_SECONDS,
)
from knowledge_tracing import (
    set_langsmith_model,
    set_langsmith_usage,
    trace_llm,
)


API_LOGGER = logging.getLogger("knowledge.api")
_POST_CONTEXT = ContextVar("gemini_post", default="batch")


def log_post_calls(function):
    """Attach post identity within each worker; never include source text."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        post = values.get("post", values)
        token = _POST_CONTEXT.set(f"{post.get('platform', '')}:{post.get('post_id', '')}")
        try:
            return function(*args, **kwargs)
        finally:
            _POST_CONTEXT.reset(token)
    return wrapped


def _stage_for_schema(schema):
    import knowledge_settings as settings
    for name, stage in (
        ("KNOWLEDGE_CLASSIFIER_SCHEMA", "classifier"),
        ("KNOWLEDGE_SCHEMA", "extraction"),
        ("EVENT_TITLE_SCHEMA", "title"),
        ("PARTICIPANT_ROLE_SCHEMA", "participant_role"),
        ("PARTICIPANT_EXTRACTION_SCHEMA", "participant_extraction"),
        ("EVENT_RELATION_SCHEMA", "event_relation"),
        ("LOCATION_HIERARCHY_SCHEMA", "location_hierarchy"),
        ("EVENT_CONSOLIDATION_SCHEMA", "consolidation_match"),
        ("EVENT_SUMMARY_SCHEMA", "consolidation_summary"),
    ):
        if schema == getattr(settings, name):
            return stage
    if set(schema.get("properties", {})) == {"relations"}:
        return "organization_hierarchy"
    return "unknown"


TOKENS_PER_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class GeminiUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def billable_output_tokens(self) -> int:
        return self.output_tokens + self.thinking_tokens


class GeminiKnowledgeCaller:
    """Structured Gemini caller that accumulates actual API token usage."""

    def __init__(
        self,
        *,
        api_key: str | None = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        client: Any = None,
        types_module: Any = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("Chưa cấu hình GEMINI_API_KEY trong .env")
            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise RuntimeError(
                    "Chưa cài google-genai. Hãy chạy: pip install google-genai"
                ) from error
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(GEMINI_TIMEOUT_SECONDS * 1_000),
                ),
            )
            types_module = types
        elif types_module is None:
            raise ValueError("types_module là bắt buộc khi truyền client tùy chỉnh")

        self.client = client
        self.model = model
        self.types = types_module
        self._usage = GeminiUsage()
        self._usage_lock = threading.Lock()
        self._stages = {}
        self._attempts = 0

    @trace_llm(
        name="gemini-knowledge-extraction",
        provider="google_genai",
        model=GEMINI_MODEL,
    )
    def __call__(self, prompt: str, output_schema: dict) -> dict:
        stage = _stage_for_schema(output_schema)
        started = time.monotonic()
        record = {"usage": GeminiUsage()}
        with self._usage_lock:
            self._attempts += 1
            request_id = self._attempts
        status = "ok"
        try:
            return self._request(prompt, output_schema, record)
        except Exception as error:
            status = type(error).__name__
            raise
        finally:
            usage = record["usage"]
            elapsed = time.monotonic() - started
            with self._usage_lock:
                stats = self._stages.setdefault(stage, {
                    "calls": 0, "errors": 0, "usage_calls": 0,
                    "input": 0, "output": 0, "thinking": 0,
                })
                stats["calls"] += 1
                stats["errors"] += status != "ok"
                stats["usage_calls"] += usage.requests
                stats["input"] += usage.input_tokens
                stats["output"] += usage.output_tokens
                stats["thinking"] += usage.thinking_tokens
            cost = self._cost(usage.input_tokens, usage.billable_output_tokens)
            API_LOGGER.debug(
                "Gemini function=%s call=%s stage=%s post=%s status=%s seconds=%.2f "
                "input=%s output=%s thinking=%s usage_known=%s cost_usd=%.8f",
                "extract_knowledge" if stage == "extraction" else stage,
                request_id, stage, _POST_CONTEXT.get(), status, elapsed,
                usage.input_tokens, usage.output_tokens, usage.thinking_tokens,
                bool(usage.requests), cost,
            )

    @staticmethod
    def _cost(input_tokens, output_tokens):
        return (Decimal(input_tokens) * Decimal(GEMINI_INPUT_PRICE_PER_MILLION)
                + Decimal(output_tokens) * Decimal(GEMINI_OUTPUT_PRICE_PER_MILLION)) / TOKENS_PER_MILLION

    def _request(self, prompt, output_schema, record):
        set_langsmith_model(self.model)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=output_schema,
                automatic_function_calling=(
                    self.types.AutomaticFunctionCallingConfig(disable=True)
                ),
            ),
        )
        usage = _usage_from_response(response)
        self._add_usage(usage)
        record["usage"] = usage
        set_langsmith_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.thinking_tokens,
        )

        raw_response = getattr(response, "text", None)
        if not raw_response:
            raise ValueError("Gemini trả về nội dung rỗng")
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Gemini trả về nội dung không phải JSON hợp lệ"
            ) from error
        if not isinstance(result, dict):
            raise ValueError("Gemini không trả về JSON object theo schema yêu cầu")

        return result

    def _add_usage(self, usage: GeminiUsage) -> None:
        with self._usage_lock:
            current = self._usage
            self._usage = GeminiUsage(
                requests=current.requests + 1,
                input_tokens=current.input_tokens + usage.input_tokens,
                output_tokens=current.output_tokens + usage.output_tokens,
                thinking_tokens=(
                    current.thinking_tokens + usage.thinking_tokens
                ),
            )

    @property
    def usage(self) -> GeminiUsage:
        with self._usage_lock:
            return self._usage

    def print_cost_summary(
        self,
        *,
        target_posts: int,
        stage_label: str | None = None,
    ) -> None:
        usage = self.usage
        input_price = Decimal(GEMINI_INPUT_PRICE_PER_MILLION)
        output_price = Decimal(GEMINI_OUTPUT_PRICE_PER_MILLION)
        input_cost = (
            Decimal(usage.input_tokens) * input_price / TOKENS_PER_MILLION
        )
        output_cost = (
            Decimal(usage.billable_output_tokens)
            * output_price
            / TOKENS_PER_MILLION
        )

        print("\n" + "=" * 72)
        label = f" - {stage_label}" if stage_label else ""
        print(
            f"TỔNG KẾT CHI PHÍ GEMINI{label} "
            f"CHO {target_posts} POST"
        )
        print(f"Model: {self.model}")
        print(f"Số request có usage thực tế: {usage.requests}")
        with self._usage_lock:
            attempts = self._attempts
            stages = {stage: dict(stats) for stage, stats in self._stages.items()}
        print(f"Tổng lần gọi API: {attempts}")
        stage_names = dict((
            ("classifier", "lọc ban đầu (classify_knowledge_potential)"),
            ("extraction", "extract_knowledge"),
        ))
        additional_names = {
            "title": "tạo tiêu đề sự kiện (title)",
            "participant_role": "vai trò tham gia (participant_role)",
            "participant_extraction": "trích xuất bên tham gia (participant_extraction)",
            "event_relation": "quan hệ sự kiện (event_relation)",
            "location_hierarchy": "phân cấp địa điểm (location_hierarchy)",
            "organization_hierarchy": "phân cấp tổ chức (organization_hierarchy)",
            "consolidation_match": "đối chiếu sự kiện (consolidation_match)",
            "consolidation_summary": "tổng hợp mô tả sự kiện (consolidation_summary)",
            "unknown": "chưa xác định (unknown)",
        }
        for stage in sorted(stages):
            if stage not in stage_names:
                stage_names[stage] = additional_names.get(stage, stage)
        for stage, name in stage_names.items():
            stats = stages.get(stage, {})
            stage_input_cost = self._cost(stats.get("input", 0), 0)
            stage_output_cost = self._cost(
                0, stats.get("output", 0) + stats.get("thinking", 0)
            )
            print(f"\nCHI PHÍ RIÊNG {name} ({stats.get('calls', 0)} lần gọi API)")
            print(f"Chi phí input {name}: ${stage_input_cost:.8f}")
            print(f"Chi phí output {name}: ${stage_output_cost:.8f}")
            print(f"TỔNG CHI PHÍ {name}: ${stage_input_cost + stage_output_cost:.8f}")
            missing_usage = stats.get("calls", 0) - stats.get("usage_calls", 0)
            if missing_usage:
                print(f"{name}: {missing_usage} lần gọi thiếu usage, chưa tính được chi phí.")
        print("\nCHI PHÍ TOÀN BỘ PIPELINE")
        print(f"Input tokens thực tế: {usage.input_tokens:,}")
        print(f"Output tokens thực tế: {usage.output_tokens:,}")
        print(
            "Billable output tokens: "
            f"{usage.billable_output_tokens:,}"
        )
        print(f"Chi phí input (Standard): ${input_cost:.8f}")
        print(f"Chi phí output (Standard): ${output_cost:.8f}")
        print(f"TỔNG CHI PHÍ (USD, Standard): ${input_cost + output_cost:.8f}")

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def _usage_from_response(response: Any) -> GeminiUsage:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        raise ValueError("Gemini không trả về usage_metadata")

    input_tokens = getattr(metadata, "prompt_token_count", None)
    output_tokens = getattr(metadata, "candidates_token_count", None)
    if input_tokens is None or output_tokens is None:
        raise ValueError(
            "Gemini usage_metadata thiếu prompt_token_count hoặc "
            "candidates_token_count"
        )
    return GeminiUsage(
        requests=1,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        thinking_tokens=int(
            getattr(metadata, "thoughts_token_count", None) or 0
        ),
    )


def call_gemini(prompt: str, output_schema: dict) -> dict:
    """Call the configured Gemini model and release the client afterward."""
    caller = GeminiKnowledgeCaller()
    try:
        return caller(prompt, output_schema)
    finally:
        caller.close()
