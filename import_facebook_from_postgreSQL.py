import os

import psycopg2
from neo4j import GraphDatabase
from dotenv import load_dotenv

from knowledge_settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

load_dotenv()

POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "facebook_scraper"),
    "user": os.getenv("POSTGRES_USER", "scraper"),
    "password": os.environ["POSTGRES_PASSWORD"],
}


def get_posts():
    connection = psycopg2.connect(**POSTGRES_CONFIG)

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    p.facebook_post_id,
                    p.facebook_url AS post_url,
                    p.content,
                    p.posted_at,
                    p.has_images,
                    p.has_videos,
                    p.metric_tier,
                    s.facebook_id,
                    s.facebook_url AS source_url,
                    s.source_name,
                    s.source_type
                FROM posts p
                JOIN sources s ON s.id = p.source_id
                WHERE p.is_tracked IS TRUE
                ORDER BY p.posted_at DESC;
            """)

            columns = [column[0] for column in cursor.description]

            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        connection.close()


def get_existing_post_ids(session):
    result = session.run("""
        MATCH (p:Post {platform: 'facebook'})
        WHERE p.platform_id IS NOT NULL
        RETURN p.platform_id AS platform_id
    """)

    return {str(record["platform_id"]) for record in result}


def import_post(tx, row):
    tx.run(
        """
        MERGE (s:Source {
            platform: 'facebook',
            platform_id: $source_platform_id
        })
        SET
            s.name = $source_name,
            s.type = $source_type,
            s.url = $source_url

        MERGE (p:Post {
            platform: 'facebook',
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
            p.has_images = $has_images,
            p.has_videos = $has_videos,
            p.metric_tier = $metric_tier

        MERGE (s)-[:PUBLISHED]->(p)
    """,
        source_platform_id=row["facebook_id"],
        source_name=row["source_name"],
        source_type=row["source_type"],
        source_url=row["source_url"],
        post_platform_id=str(row["facebook_post_id"]),
        post_url=row["post_url"],
        content=row["content"],
        posted_at=row["posted_at"],
        has_images=row["has_images"],
        has_videos=row["has_videos"],
        metric_tier=row["metric_tier"],
    )


def main():
    posts = get_posts()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session(database="neo4j") as session:
            existing_post_ids = get_existing_post_ids(session)
            new_posts = []
            seen_post_ids = set(existing_post_ids)
            invalid_posts = 0

            for post in posts:
                post_id = post["facebook_post_id"]

                if post_id is None:
                    invalid_posts += 1
                    continue

                post_id = str(post_id)
                if post_id in seen_post_ids:
                    continue

                seen_post_ids.add(post_id)
                new_posts.append(post)

            for post in new_posts:
                session.execute_write(import_post, post)

        skipped_posts = len(posts) - len(new_posts) - invalid_posts
        print(
            f"Đã import {len(new_posts)} post mới vào Neo4j. "
            f"Bỏ qua {skipped_posts} post đã có"
            + (f" và {invalid_posts} post không có ID." if invalid_posts else ".")
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
