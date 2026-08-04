import sqlite3
from datetime import datetime
from typing import Any
from neo4j import GraphDatabase

SQLITE_DB_PATH = "/media/pc1799/New Volume/huan/Tiktok-Api/data/tiktok.db"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "huanhoang"
NEO4J_DATABASE = "neo4j"

IMPORT_LIMIT = 500


def parse_sqlite_datetime(value: Any) -> datetime | None:
    """Chuyển DATETIME của SQLite thành datetime để Neo4j lưu đúng kiểu thời gian."""
    if value is None or isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    # Hỗ trợ cả "2026-07-22 09:08:37" và chuỗi ISO có hậu tố Z.
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def get_posts() -> list[dict[str, Any]]:
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.execute("""
            SELECT
                p.tiktok_video_id AS post_platform_id,
                p.tiktok_url AS post_url,
                p.description AS content,
                p.duration_seconds,
                p.cover_url,
                p.posted_at,

                s.source_type,
                s.identifier AS source_identifier,
                COALESCE(s.display_name, s.identifier) AS source_name,
                s.tiktok_url AS source_url,

                -- Tránh trùng identifier giữa user, hashtag, sound và keyword.
                s.source_type || ':' || s.identifier AS source_platform_id
            FROM posts AS p
            JOIN sources AS s ON s.id = p.source_id
            WHERE p.description IS NOT NULL
              AND TRIM(p.description) <> ''
              AND COALESCE(p.is_deleted, 0) = 0
            ORDER BY p.posted_at DESC
            LIMIT 500
            """)

        posts = [dict(row) for row in cursor.fetchall()]
        for post in posts:
            post["posted_at"] = parse_sqlite_datetime(post["posted_at"])

        return posts
    finally:
        connection.close()


def import_post(tx, row: dict[str, Any]) -> None:
    tx.run(
        """
        MERGE (s:Source {
            platform: 'tiktok',
            platform_id: $source_platform_id
        })
        SET
            s.name = $source_name,
            s.type = $source_type,
            s.identifier = $source_identifier,
            s.url = $source_url

        MERGE (p:Post {
            platform: 'tiktok',
            platform_id: $post_platform_id
        })
        ON CREATE SET
            p.entity_processed = false,
            p.knowledge_processed = false,
            p.knowledge_retry_count = 0
        SET
            p.content = $content,
            p.url = $post_url,
            p.posted_at = $posted_at,
            p.duration_seconds = $duration_seconds,
            p.cover_url = $cover_url,
            p.has_images = false,
            p.has_videos = true

        MERGE (s)-[:PUBLISHED]->(p)
        """,
        **row,
    )


def main() -> None:
    posts = get_posts()

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()

        with driver.session(database=NEO4J_DATABASE) as session:
            for post in posts:
                session.execute_write(import_post, post)

        print(f"Đã import {len(posts)} video TikTok vào Neo4j.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
