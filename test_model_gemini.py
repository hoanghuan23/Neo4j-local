import argparse
import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from dotenv import load_dotenv


POST_TARGET = {
    "platform": "facebook",
    "post_ids": [
        "1520067480155258",
    ],
}


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    thinking_tokens: int

    @property
    def billable_output_tokens(self) -> int:
        return self.output_tokens + self.thinking_tokens


@dataclass(frozen=True)
class AnalysisCost:
    input_cost: Decimal
    output_cost: Decimal

    @property
    def total_cost(self) -> Decimal:
        return self.input_cost + self.output_cost


MODEL_PRICING = {
    "gemini-3.1-flash-lite": ModelPricing(Decimal("0.25"), Decimal("1.50")),
    "gemini-3.5-flash-lite": ModelPricing(Decimal("0.30"), Decimal("2.50")),
}
DEFAULT_MODEL = "gemini-3.5-flash-lite"
TOKENS_PER_MILLION = Decimal("1000000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test một model Gemini trên nhiều Post bằng prompt, schema và "
            "validation hiện tại mà không ghi kết quả vào Neo4j."
        )
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_PRICING),
        default=DEFAULT_MODEL,
        help=f"Tên model Gemini (mặc định: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def token_usage_from_response(response: Any) -> TokenUsage:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        raise ValueError("Gemini không trả về usage_metadata; không thể tính chi phí")

    input_tokens = getattr(metadata, "prompt_token_count", None)
    output_tokens = getattr(metadata, "candidates_token_count", None)
    thinking_tokens = getattr(metadata, "thoughts_token_count", None)

    if input_tokens is None or output_tokens is None:
        raise ValueError(
            "Gemini usage_metadata thiếu prompt_token_count hoặc "
            "candidates_token_count; không thể tính chi phí"
        )

    return TokenUsage(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        thinking_tokens=int(thinking_tokens or 0),
    )


def calculate_cost(usage: TokenUsage, pricing: ModelPricing) -> AnalysisCost:
    return AnalysisCost(
        input_cost=(
            Decimal(usage.input_tokens)
            * pricing.input_per_million
            / TOKENS_PER_MILLION
        ),
        output_cost=(
            Decimal(usage.billable_output_tokens)
            * pricing.output_per_million
            / TOKENS_PER_MILLION
        ),
    )


def add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        left.input_tokens + right.input_tokens,
        left.output_tokens + right.output_tokens,
        left.thinking_tokens + right.thinking_tokens,
    )


def add_cost(left: AnalysisCost, right: AnalysisCost) -> AnalysisCost:
    return AnalysisCost(
        left.input_cost + right.input_cost,
        left.output_cost + right.output_cost,
    )


class GeminiCaller:
    def __init__(self, client: Any, model: str, types_module: Any) -> None:
        self.client = client
        self.model = model
        self.types = types_module
        self.last_usage: TokenUsage | None = None

    def __call__(self, prompt: str, output_schema: dict) -> dict:
        self.last_usage = None
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=output_schema,
            ),
        )

        self.last_usage = token_usage_from_response(response)
        raw_response = getattr(response, "text", None)
        if not raw_response:
            raise ValueError("Gemini trả về nội dung rỗng")

        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise ValueError("Gemini trả về nội dung không phải JSON hợp lệ") from error

        if not isinstance(result, dict):
            raise ValueError("Gemini không trả về JSON object theo schema yêu cầu")
        return result


def print_usage_and_cost(
    usage: TokenUsage,
    cost: AnalysisCost,
    *,
    total_label: str = "Tổng chi phí Post",
    stage_label: str | None = None,
) -> None:
    suffix = f" - {stage_label}" if stage_label else ""
    print(f"\n--- TOKEN VÀ CHI PHÍ ƯỚC TÍNH{suffix} (STANDARD) ---")
    print(f"Input tokens: {usage.input_tokens:,}")
    print(f"Output tokens: {usage.output_tokens:,}")
    print(f"Thinking tokens: {usage.thinking_tokens:,}")
    print(f"Billable output tokens: {usage.billable_output_tokens:,}")
    print(f"Chi phí input: ${cost.input_cost:.8f}")
    print(f"Chi phí output: ${cost.output_cost:.8f}")
    print(f"{total_label}: ${cost.total_cost:.8f}")


def main() -> None:
    args = parse_args()
    platform = POST_TARGET["platform"]
    post_ids = POST_TARGET["post_ids"]

    if not platform or not post_ids:
        raise SystemExit(
            "Hãy gán platform và post_ids trong biến POST_TARGET ở đầu file."
        )

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Chưa cấu hình GEMINI_API_KEY trong .env")

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise SystemExit(
            "Chưa cài google-genai. Hãy chạy: pip install google-genai"
        ) from error

    from neo4j import GraphDatabase

    from knowledge_extraction import (
        classify_knowledge_potential,
        extract_knowledge,
    )
    from knowledge_settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
    from knowledge_validation import validate_knowledge

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    try:
        with driver.session(database="neo4j") as session:
            records = session.run(
                """
                UNWIND range(0, size($post_ids) - 1) AS requested_order
                WITH requested_order, $post_ids[requested_order] AS post_id
                OPTIONAL MATCH (p:Post {
                    platform: $platform,
                    platform_id: post_id
                })
                RETURN post_id, p.content AS content
                ORDER BY requested_order
                """,
                platform=platform,
                post_ids=post_ids,
            ).data()
    finally:
        driver.close()

    pricing = MODEL_PRICING[args.model]
    client = genai.Client(api_key=api_key)
    call_gemini = GeminiCaller(client, args.model, types)

    classifier_total_usage = TokenUsage(0, 0, 0)
    classifier_total_cost = AnalysisCost(Decimal("0"), Decimal("0"))
    deep_total_usage = TokenUsage(0, 0, 0)
    deep_total_cost = AnalysisCost(Decimal("0"), Decimal("0"))
    classified_count = 0
    skipped_count = 0
    deep_count = 0

    print(f"Model: {args.model}")
    print(
        "Giá Standard / 1M token: "
        f"input ${pricing.input_per_million}, "
        f"output ${pricing.output_per_million}"
    )
    total_started_at = time.perf_counter()

    try:
        for index, record in enumerate(records, start=1):
            post_id = record["post_id"]
            content = record["content"]

            print(f"\n{'=' * 80}")
            print(f"Post {index}/{len(records)}: {platform}:{post_id}")

            if not content:
                print("Không tìm thấy nội dung Post, bỏ qua.")
                continue

            print(f"\nNội dung:\n{content}")

            started_at = time.perf_counter()
            classified_count += 1
            classification = classify_knowledge_potential(
                content,
                call_model=call_gemini,
            )
            classifier_usage = call_gemini.last_usage
            if classifier_usage is None:
                raise ValueError(
                    "Không lấy được token usage sau khi gọi classifier"
                )
            classifier_cost = calculate_cost(classifier_usage, pricing)
            needs_deep = classification["should_deep_analyze"]
            decision = "DEEP" if needs_deep else "SKIPPED"

            print("\n--- CLASSIFIER OUTPUT ---")
            print(json.dumps(classification, ensure_ascii=False, indent=2))
            print(f"Decision: {decision}")
            print_usage_and_cost(
                classifier_usage,
                classifier_cost,
                total_label="Chi phí classifier",
                stage_label=f"CLASSIFIER - POST {index}/{len(records)}",
            )

            deep_usage = TokenUsage(0, 0, 0)
            deep_cost = AnalysisCost(Decimal("0"), Decimal("0"))
            if needs_deep:
                deep_count += 1
                raw_result = extract_knowledge(content, call_model=call_gemini)
                deep_usage = call_gemini.last_usage
                if deep_usage is None:
                    raise ValueError(
                        "Không lấy được token usage sau khi phân tích sâu"
                    )
                deep_cost = calculate_cost(deep_usage, pricing)

                print("\n--- RAW DEEP MODEL OUTPUT ---")
                print(json.dumps(raw_result, ensure_ascii=False, indent=2))
                print_usage_and_cost(
                    deep_usage,
                    deep_cost,
                    total_label="Chi phí phân tích sâu",
                    stage_label=f"PHÂN TÍCH SÂU - POST {index}/{len(records)}",
                )
            else:
                skipped_count += 1
                raw_result = {
                    "entities": [],
                    "events": [],
                    "event_relations": [],
                }
                print("\nClassifier kết luận SKIPPED; không gọi prompt knowledge-v10.")

            validated_result = validate_knowledge(
                content,
                raw_result,
                platform,
                post_id,
            )
            print("\n--- OUTPUT SAU VALIDATION ---")
            print(json.dumps(validated_result, ensure_ascii=False, indent=2))

            post_usage = add_usage(classifier_usage, deep_usage)
            post_cost = add_cost(classifier_cost, deep_cost)
            print_usage_and_cost(
                post_usage,
                post_cost,
                total_label="Tổng chi phí Post",
                stage_label=f"TỔNG POST {index}/{len(records)}",
            )
            elapsed_seconds = time.perf_counter() - started_at
            print(f"\nThời gian xử lý: {elapsed_seconds:.2f} giây")

            classifier_total_usage = add_usage(
                classifier_total_usage,
                classifier_usage,
            )
            classifier_total_cost = add_cost(
                classifier_total_cost,
                classifier_cost,
            )
            deep_total_usage = add_usage(deep_total_usage, deep_usage)
            deep_total_cost = add_cost(deep_total_cost, deep_cost)
    finally:
        client.close()

    total_elapsed_seconds = time.perf_counter() - total_started_at
    print(f"\n{'=' * 80}")
    print("TỔNG KẾT TẤT CẢ POST")
    print(
        f"Classifier: {classified_count} post | "
        f"DEEP: {deep_count} | SKIPPED: {skipped_count}"
    )
    print_usage_and_cost(
        classifier_total_usage,
        classifier_total_cost,
        total_label="Tổng chi phí classifier",
        stage_label="TỔNG CLASSIFIER",
    )
    print_usage_and_cost(
        deep_total_usage,
        deep_total_cost,
        total_label="Tổng chi phí phân tích sâu",
        stage_label="TỔNG PHÂN TÍCH SÂU",
    )
    print_usage_and_cost(
        add_usage(classifier_total_usage, deep_total_usage),
        add_cost(classifier_total_cost, deep_total_cost),
        total_label="Tổng chi phí tất cả Post",
        stage_label="TOÀN BỘ PIPELINE",
    )
    print(f"\nTổng thời gian xử lý tất cả Post: {total_elapsed_seconds:.2f} giây")


if __name__ == "__main__":
    main()
