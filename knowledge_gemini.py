import json
import threading
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from knowledge_settings import (
    GEMINI_API_KEY,
    GEMINI_INPUT_PRICE_PER_MILLION,
    GEMINI_MODEL,
    GEMINI_OUTPUT_PRICE_PER_MILLION,
    LOGGER,
)
from knowledge_tracing import (
    set_langsmith_model,
    set_langsmith_usage,
    trace_llm,
)


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
            client = genai.Client(api_key=api_key)
            types_module = types
        elif types_module is None:
            raise ValueError("types_module là bắt buộc khi truyền client tùy chỉnh")

        self.client = client
        self.model = model
        self.types = types_module
        self._usage = GeminiUsage()
        self._usage_lock = threading.Lock()

    @trace_llm(
        name="gemini-knowledge-extraction",
        provider="google_genai",
        model=GEMINI_MODEL,
    )
    def __call__(self, prompt: str, output_schema: dict) -> dict:
        set_langsmith_model(self.model)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=output_schema,
            ),
        )
        usage = _usage_from_response(response)
        self._add_usage(usage)
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

        LOGGER.info(
            "Gemini hoàn tất | model=%s | input_tokens=%s | "
            "output_tokens=%s | thinking_tokens=%s",
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.thinking_tokens,
        )
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
            f"CHO TỐI ĐA {target_posts} REQUEST"
        )
        print(f"Model: {self.model}")
        print(f"Số request có usage thực tế: {usage.requests}/{target_posts}")
        print(f"Input tokens thực tế: {usage.input_tokens:,}")
        print(f"Output tokens thực tế: {usage.output_tokens:,}")
        print(f"Thinking tokens thực tế: {usage.thinking_tokens:,}")
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
