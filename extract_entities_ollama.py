import json
import logging
import unicodedata

import requests
from neo4j import GraphDatabase

logging.getLogger("neo4j").setLevel(logging.ERROR)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "huanhoang"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_TIMEOUT_SECONDS = 600
POST_LIMIT = 100

ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION"}
CONFIDENCE_LEVELS = {"HIGH", "MEDIUM", "LOW"}

ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "canonical_name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": sorted(ENTITY_TYPES),
                    },
                    "resolution_confidence": {
                        "type": "string",
                        "enum": sorted(CONFIDENCE_LEVELS),
                    },
                },
                "required": [
                    "name",
                    "canonical_name",
                    "type",
                    "resolution_confidence",
                ],
            },
        }
    },
    "required": ["entities"],
}


def normalize_name(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.strip().casefold().split()))


def make_search_name(value: str) -> str:
    normalized = unicodedata.normalize("NFD", normalize_name(value))
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return without_accents.replace("đ", "d")


def call_ollama(prompt: str, output_schema: dict) -> dict:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "format": output_schema,
            "options": {"temperature": 0},
            "prompt": prompt,
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return json.loads(response.json()["response"])


def extract_entities(content: str) -> list[dict]:
    prompt = f"""
        Bạn là hệ thống trích xuất và phân giải thực thể có tên NER.

        Chỉ trích xuất PERSON, ORGANIZATION và LOCATION xuất hiện trực tiếp
        trong văn bản. Mỗi kết quả phải có:
        - `name`: đúng cách gọi xuất hiện trong văn bản, giữ nguyên ngôn ngữ.
        - `canonical_name`: tên phổ biến, đầy đủ của đúng chủ thể, bỏ danh xưng,
          chức vụ và tiền tố không thuộc tên riêng.
        - `type`: PERSON, ORGANIZATION hoặc LOCATION.
        - `resolution_confidence`: HIGH chỉ khi ngữ cảnh đủ chắc chắn;
          MEDIUM hoặc LOW nếu tên có thể chỉ nhiều chủ thể.

        Quy tắc bắt buộc:
        1. Không dịch tên và không tạo thực thể không xuất hiện trong văn bản.
        2. Có thể dùng kiến thức phổ biến và ngữ cảnh để mở rộng alias thành tên
           canonical, nhưng không được đoán khi danh tính còn mơ hồ.
        3. Các alias của cùng chủ thể phải có cùng `canonical_name` và `type`.
        4. Nếu không chắc, giữ tên đã bỏ danh xưng làm `canonical_name` và đặt
           confidence MEDIUM hoặc LOW.
        5. Không gộp người với chính quyền, tổ chức, gia đình hoặc người khác có
           chung một phần tên.
        6. Loại bỏ khoảng trắng thừa và dấu câu không thuộc tên.
        7. Không trích xuất đại từ, quan hệ, nghề nghiệp, số lượng, sản phẩm,
           đồ vật, sự kiện hoặc nhóm chung chung.
        8. Chỉ LOCATION khi đó là tên riêng của quốc gia, tỉnh, thành phố, quận,
           huyện hoặc địa danh; cụm như "3.400 viên kim cương nhập lậu" không
           phải LOCATION.
        9. Không trả về thực thể trùng cùng `canonical_name` và `type`.
        10. Không trích xuất hashtag hoặc username/handle; mọi kết quả có ký tự
            `#` hoặc `@` đều không hợp lệ.
        11. Chỉ trả về JSON đúng schema, không giải thích thêm.

        Ví dụ phân giải:
        - "Trump", "Tổng thống Trump", "Donald Trump" và
          "President Donald Trump" trong ngữ cảnh nói về tổng thống Hoa Kỳ
          -> canonical_name "Donald Trump", type PERSON, confidence HIGH.
        - "Melania Trump" -> canonical_name "Melania Trump", type PERSON;
          không gộp vào Donald Trump.
        - "Trump administration" -> canonical_name "Trump administration",
          type ORGANIZATION; không gộp vào Donald Trump.
        - Một họ "Trump" không có ngữ cảnh nhận diện -> confidence LOW.

        Văn bản:
        ```text
        {content}
        ```
        """.strip()

    return call_ollama(prompt, ENTITY_SCHEMA).get("entities", [])


def prepare_entity(entity: dict) -> dict | None:
    name = " ".join(str(entity.get("name", "")).strip().split())
    canonical_name = " ".join(
        str(entity.get("canonical_name", "")).strip().split()
    )
    entity_type = str(entity.get("type", "")).strip().upper()
    confidence = str(entity.get("resolution_confidence", "")).strip().upper()

    if (
        not name
        or entity_type not in ENTITY_TYPES
        or "#" in name
        or "@" in name
        or "#" in canonical_name
        or "@" in canonical_name
    ):
        return None
    if confidence not in CONFIDENCE_LEVELS:
        confidence = "LOW"

    is_canonical = confidence == "HIGH" and bool(canonical_name)
    display_name = canonical_name if is_canonical else name
    normalized_name = normalize_name(display_name)
    if not normalized_name:
        return None

    return {
        "name": name,
        "display_name": display_name,
        "normalized_name": normalized_name,
        "search_name": make_search_name(normalized_name),
        "entity_type": entity_type,
        "confidence": confidence,
        "is_canonical": is_canonical,
    }


def save_entities(session, platform: str, post_id: str, entities: list[dict]) -> int:
    saved_count = 0
    for raw_entity in entities:
        entity = prepare_entity(raw_entity)
        if entity is None:
            continue

        session.run(
            """
            MATCH (p:Post {
                platform: $platform,
                platform_id: $post_id
            })

            MERGE (e:Entity {
                normalized_name: $normalized_name,
                type: $entity_type
            })
            ON CREATE SET
                e.name = $display_name,
                e.search_name = $search_name,
                e.aliases = [$name],
                e.resolution_confidence = $confidence,
                e.needs_review = NOT $is_canonical
            ON MATCH SET
                e.aliases = CASE
                    WHEN e.aliases IS NULL THEN [e.name, $name]
                    WHEN NOT $name IN e.aliases THEN e.aliases + $name
                    ELSE e.aliases
                END,
                e.name = CASE
                    WHEN $is_canonical THEN $display_name
                    ELSE e.name
                END,
                e.search_name = CASE
                    WHEN $is_canonical THEN $search_name
                    ELSE coalesce(e.search_name, $search_name)
                END,
                e.resolution_confidence = CASE
                    WHEN e.resolution_confidence = 'HIGH' OR $confidence = 'HIGH'
                    THEN 'HIGH'
                    WHEN e.resolution_confidence = 'MEDIUM' OR $confidence = 'MEDIUM'
                    THEN 'MEDIUM'
                    ELSE 'LOW'
                END,
                e.needs_review = CASE
                    WHEN e.resolution_confidence = 'HIGH' OR $confidence = 'HIGH'
                    THEN false
                    ELSE true
                END

            MERGE (p)-[:MENTIONS]->(e)
            """,
            platform=platform,
            post_id=post_id,
            **entity,
        ).consume()
        saved_count += 1

    session.run(
        """
        MATCH (p:Post {
            platform: $platform,
            platform_id: $post_id
        })
        SET p.entity_processed = true,
            p.entity_processed_at = datetime()
        """,
        platform=platform,
        post_id=post_id,
    ).consume()
    return saved_count


def create_entity_schema(session) -> None:
    session.run(
        """
        CREATE CONSTRAINT entity_identity_unique IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE (e.normalized_name, e.type) IS UNIQUE
        """
    ).consume()
    session.run(
        """
        CREATE TEXT INDEX entity_search_name IF NOT EXISTS
        FOR (e:Entity) ON (e.search_name)
        """
    ).consume()


def process_new_posts(session) -> None:
    create_entity_schema(session)

    posts = list(
        session.run(
            """
            MATCH (p:Post)
            WHERE p.content IS NOT NULL
              AND trim(p.content) <> ''
              AND coalesce(p.entity_processed, false) = false
              AND p.platform IN ['facebook', 'tiktok']
            RETURN p.platform AS platform,
                   p.platform_id AS post_id,
                   p.content AS content
            ORDER BY p.posted_at DESC
            LIMIT $post_limit
            """,
            post_limit=POST_LIMIT,
        )
    )
    print(f"Tìm thấy {len(posts)} post để xử lý.")

    for index, post in enumerate(posts, start=1):
        platform = post["platform"]
        post_id = post["post_id"]
        print(f"\n[{index}/{len(posts)}] Đang xử lý {platform} post {post_id}")
        try:
            entities = extract_entities(post["content"])
            print(json.dumps(entities, ensure_ascii=False, indent=2))
            saved_count = save_entities(session, platform, post_id, entities)
            print(f"Đã lưu {saved_count}/{len(entities)} entity hợp lệ.")
        except Exception as error:
            print(f"Lỗi post {post_id}: {error}")


def main() -> None:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        with driver.session(database="neo4j") as session:
            process_new_posts(session)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
