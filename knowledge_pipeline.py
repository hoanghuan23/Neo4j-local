import json

from knowledge_settings import (
    KNOWLEDGE_MAX_RETRIES,
    KNOWLEDGE_PIPELINE_ENABLED,
    KNOWLEDGE_PROMPT_VERSION,
    LOGGER,
    OLLAMA_MODEL,
    POST_LIMIT,
)
from knowledge_extraction import extract_knowledge
from knowledge_persistence import (
    create_entity_schema,
    create_knowledge_schema,
    mark_knowledge_failure,
    save_entities,
    save_knowledge_tx,
)
from knowledge_validation import validate_knowledge


def _load_posts(session) -> list:
    if KNOWLEDGE_PIPELINE_ENABLED:
        return list(
            session.run(
                """
                MATCH (p:Post)
                WHERE p.content IS NOT NULL
                  AND trim(p.content) <> ''
                  AND p.platform IN ['facebook', 'tiktok']
                  AND (
                    coalesce(p.knowledge_processed, false) = false
                    OR coalesce(p.knowledge_model, '') <> $knowledge_model
                    OR coalesce(p.knowledge_prompt_version, '')
                       <> $knowledge_prompt_version
                  )
                  AND (
                    coalesce(p.knowledge_retry_count, 0) < $max_retries
                    OR coalesce(p.knowledge_model, '') <> $knowledge_model
                    OR coalesce(p.knowledge_prompt_version, '')
                       <> $knowledge_prompt_version
                  )
                RETURN p.platform AS platform,
                       p.platform_id AS post_id,
                       p.content AS content
                ORDER BY p.posted_at DESC
                LIMIT $post_limit
                """,
                post_limit=POST_LIMIT,
                max_retries=KNOWLEDGE_MAX_RETRIES,
                knowledge_model=OLLAMA_MODEL,
                knowledge_prompt_version=KNOWLEDGE_PROMPT_VERSION,
            )
        )
    return list(
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


def process_new_posts(session, extract_knowledge_fn=extract_knowledge) -> None:
    if KNOWLEDGE_PIPELINE_ENABLED:
        create_knowledge_schema(session)
    else:
        create_entity_schema(session)

    posts = _load_posts(session)
    print(f"Tìm thấy {len(posts)} post để xử lý.")

    for index, post in enumerate(posts, start=1):
        platform = post["platform"]
        post_id = post["post_id"]
        content = post["content"]
        print(f"\n[{index}/{len(posts)}] Đang xử lý {platform} post {post_id}")
        try:
            raw_knowledge = extract_knowledge_fn(content)
            if not KNOWLEDGE_PIPELINE_ENABLED:
                entities = raw_knowledge["entities"]
                print(json.dumps(entities, ensure_ascii=False, indent=2))
                saved_count = save_entities(session, platform, post_id, entities)
                print(f"Đã lưu {saved_count}/{len(entities)} entity hợp lệ.")
                continue

            knowledge = validate_knowledge(content, raw_knowledge, platform, post_id)
            print(json.dumps(knowledge, ensure_ascii=False, indent=2))
            counts = session.execute_write(
                save_knowledge_tx, platform, post_id, knowledge
            )
            print(
                "Đã lưu "
                f"{counts['entities']} Entity, {counts['events']} Event, "
                f"{counts['event_relations']} quan hệ Event."
            )
        except Exception as error:
            LOGGER.exception("Lỗi xử lý post %s", post_id)
            if KNOWLEDGE_PIPELINE_ENABLED:
                try:
                    session.execute_write(
                        mark_knowledge_failure,
                        platform,
                        post_id,
                        str(error),
                    )
                except Exception:
                    LOGGER.exception(
                        "Không thể cập nhật trạng thái lỗi cho %s", post_id
                    )
            print(f"Lỗi post {post_id}: {error}")
