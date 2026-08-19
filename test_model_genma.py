import argparse
import json
import os
import time

POST_TARGET = {
    "platform": "facebook",
    "post_ids": [
        "1746842876801900",
        "1087785800299246",
        "1407371134860669",
        "28533300939609810",
        "1558674646298852",
        "1407326191531830",
        "1539175764913865",
        "1407318151532634",
        "1407530841593664",
        "1397976562538211",
        "1406656645014417",
        "1398012205867980",
        "1087726483638511",
        "1406699508343464",
        "1397995525869648"
    ],
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
        default="qwen2.5:7b",
        help="Tên model Ollama (mặc định: llama3.2:latest)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    platform = POST_TARGET["platform"]
    post_ids = POST_TARGET["post_ids"]

    if not platform or not post_ids:
        raise SystemExit(
            "Hãy gán platform và post_ids trong biến POST_TARGET ở đầu file."
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

    print(f"Model: {args.model}")
    total_started_at = time.perf_counter()
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

    total_elapsed_seconds = time.perf_counter() - total_started_at
    print(f"\nTổng thời gian xử lý tất cả Post: {total_elapsed_seconds:.2f} giây")


if __name__ == "__main__":
    main()
