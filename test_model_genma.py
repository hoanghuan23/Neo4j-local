import argparse
import json
import os
import time

POST_TARGET = {
    "platform": "facebook",
    "post_id": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test một model Ollama trên một Post bằng prompt, schema và "
            "validation hiện tại mà không ghi kết quả vào Neo4j."
        )
    )
    parser.add_argument(
        "--model",
        default="gemma4:e2b",
        help="Tên model Groq (mặc định: gemma4:e2b)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    platform = POST_TARGET["platform"]
    post_id = POST_TARGET["post_id"]

    if not platform or not post_id:
        raise SystemExit(
            "Hãy gán platform và post_id trong biến POST_TARGET ở đầu file."
        )

    # knowledge_settings đọc model khi được import, vì vậy phải đặt biến môi
    # trường trước khi import các module của knowledge pipeline.
    os.environ["KNOWLEDGE_MODEL"] = args.model

    from neo4j import GraphDatabase

    from knowledge_extraction import extract_knowledge, call_ollama
    from knowledge_settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
    from knowledge_validation import validate_knowledge

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    try:
        with driver.session(database="neo4j") as session:
            record = session.run(
                """
                MATCH (p:Post {
                    platform: $platform,
                    platform_id: $post_id
                })
                RETURN p.content AS content
                """,
                platform=platform,
                post_id=post_id,
            ).single()
    finally:
        driver.close()

    if record is None or not record["content"]:
        raise SystemExit(
            f"Không tìm thấy nội dung Post {platform}:{post_id}"
        )

    content = record["content"]
    print(f"Post: {platform}:{post_id}")
    print(f"Model: {args.model}")
    print(f"\nNội dung:\n{content}")

    started_at = time.perf_counter()
    raw_result = extract_knowledge(content, call_model=call_ollama)
    elapsed_seconds = time.perf_counter() - started_at

    print("\n--- RAW MODEL OUTPUT ---")
    print(json.dumps(raw_result, ensure_ascii=False, indent=2))

    validated_result = validate_knowledge(
        content,
        raw_result,
        platform,
        post_id,
    )
    print("\n--- OUTPUT SAU VALIDATION ---")
    print(json.dumps(validated_result, ensure_ascii=False, indent=2))
    print(f"\nThời gian xử lý: {elapsed_seconds:.2f} giây")


if __name__ == "__main__":
    main()
