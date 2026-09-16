"""Cập nhật velocity từ PostgreSQL cho Post Facebook đã có trong Neo4j.

Chạy: python3 import_facebook_engagement_velocity.py
Lấy mọi post có posted_at trong 7 ngày gần nhất, không lọc tier/is_tracked.
Giá trị NULL từ PostgreSQL sẽ xóa thuộc tính tương ứng trong Neo4j.
"""

import sys
from pathlib import Path

import psycopg2
from neo4j import GraphDatabase

# Allow direct execution after moving this script into import_database.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from import_database.import_facebook_from_postgreSQL import POSTGRES_CONFIG
from knowledge_settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


BATCH_SIZE = 1000


def update_velocity_batch(tx, rows):
    record = tx.run(
        """
        UNWIND $rows AS row
        OPTIONAL MATCH (p:Post {
            platform: 'facebook',
            platform_id: row.platform_id
        })
        SET p.last_engagement_velocity = row.velocity
        WITH row, count(p) AS matched_nodes
        RETURN count(CASE WHEN matched_nodes > 0 THEN 1 END) AS updated_posts
        """,
        rows=rows,
    ).single()
    return record["updated_posts"]


def sync_recent_velocities(connection, session, batch_size=BATCH_SIZE):
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0")

    stats = {"read_posts": 0, "updated_posts": 0, "missing_posts": 0, "invalid_posts": 0}
    # Named cursor đọc từng lô, không nạp toàn bộ kết quả vào RAM.
    with connection.cursor(name="recent_facebook_velocities") as cursor:
        cursor.execute(
            """
            SELECT facebook_post_id, last_engagement_velocity
            FROM posts
            WHERE posted_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
              AND posted_at <= CURRENT_TIMESTAMP
            """
        )
        while True:
            posts = cursor.fetchmany(batch_size)
            if not posts:
                break

            stats["read_posts"] += len(posts)
            rows = []
            for post_id, velocity in posts:
                if post_id is None:
                    stats["invalid_posts"] += 1
                    continue
                rows.append({
                    "platform_id": str(post_id),
                    "velocity": float(velocity) if velocity is not None else None,
                })

            if rows:
                updated = session.execute_write(update_velocity_batch, rows)
                stats["updated_posts"] += updated
                stats["missing_posts"] += len(rows) - updated

    return stats


def main():
    connection = psycopg2.connect(**POSTGRES_CONFIG)
    try:
        connection.set_session(readonly=True)
        with GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        ) as driver:
            with driver.session(database="neo4j") as session:
                stats = sync_recent_velocities(connection, session)
        print(
            f"Đã đọc {stats['read_posts']} post trong 7 ngày gần nhất; "
            f"cập nhật last_engagement_velocity cho {stats['updated_posts']} post; "
            f"{stats['missing_posts']} post chưa có trong Neo4j; "
            f"bỏ qua {stats['invalid_posts']} post không có ID."
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
