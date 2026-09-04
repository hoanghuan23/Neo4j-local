import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from langsmith import traceable

from knowledge_settings import (
    KNOWLEDGE_MAX_RETRIES,
    KNOWLEDGE_PIPELINE_ENABLED,
    KNOWLEDGE_WORKERS,
    LOGGER,
    POST_LIMIT,
)
from knowledge_extraction import classify_knowledge_potential, extract_knowledge
from knowledge_relation_router import classify_relation_routes
from knowledge_relations.participant_role import enrich_participant_roles
from knowledge_persistence import (
    create_entity_schema,
    create_knowledge_schema,
    mark_knowledge_failure,
    save_entities,
    save_knowledge_tx,
)
from knowledge_validation import validate_knowledge


@traceable(
    name="process-knowledge-post",
    run_type="chain",
    tags=["knowledge-pipeline"],
    process_inputs=lambda inputs: {
        "platform": inputs["platform"],
        "post_id": inputs["post_id"],
        "content": inputs["content"],
    },
)
def _extract_post(
    classify_post_fn,
    extract_knowledge_fn,
    validate_knowledge_fn,
    classify_relations_fn,
    platform: str,
    post_id: str,
    content: str,
    enrich_participants_fn=enrich_participant_roles,
) -> dict:
    if not KNOWLEDGE_PIPELINE_ENABLED:
        return {
            "classification": None,
            "classifier_decision": None,
            "knowledge": extract_knowledge_fn(content),
            "relation_routes": {"event_routes": [], "pair_routes": []},
        }

    classification = classify_post_fn(content)
    needs_deep_extraction = classification["should_deep_analyze"]
    raw_knowledge = (
        extract_knowledge_fn(content)
        if needs_deep_extraction
        else {"entities": [], "events": [], "event_relations": []}
    )
    knowledge = validate_knowledge_fn(content, raw_knowledge, platform, post_id)
    relation_routes = (
        classify_relations_fn(content, knowledge)
        if needs_deep_extraction
        else {"event_routes": [], "pair_routes": []}
    )
    if needs_deep_extraction:
        knowledge = enrich_participants_fn(content, knowledge, relation_routes)
    return {
        "classification": classification,
        "classifier_decision": "DEEP" if needs_deep_extraction else "SKIPPED",
        "knowledge": knowledge,
        "relation_routes": relation_routes,
    }


def _load_posts(session) -> list:
    if KNOWLEDGE_PIPELINE_ENABLED:
        return list(
            session.run(
                """
                MATCH (p:Post)
                WHERE p.content IS NOT NULL
                  AND trim(p.content) <> ''
                  AND p.platform IN ['facebook', 'tiktok']
                  AND coalesce(p.knowledge_processed, false) = false
                  AND coalesce(p.knowledge_retry_count, 0) < $max_retries
                RETURN p.platform AS platform,
                       p.platform_id AS post_id,
                       p.content AS content
                ORDER BY
                    CASE
                        WHEN toLower(trim(coalesce(p.metric_tier, ''))) = 'hot'
                        THEN 0
                        ELSE 1
                    END,
                    CASE
                        WHEN coalesce(p.entity_processed, false) = false THEN 0
                        WHEN coalesce(p.knowledge_processed, false) = false THEN 1
                        ELSE 2
                    END,
                    p.posted_at DESC
                LIMIT $post_limit
                """,
                post_limit=POST_LIMIT,
                max_retries=KNOWLEDGE_MAX_RETRIES,
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
            ORDER BY
                CASE
                    WHEN toLower(trim(coalesce(p.metric_tier, ''))) = 'hot'
                    THEN 0
                    ELSE 1
                END,
                p.posted_at DESC
            LIMIT $post_limit
            """,
            post_limit=POST_LIMIT,
        )
    )


def process_new_posts(
    session,
    extract_knowledge_fn=extract_knowledge,
    classify_post_fn=classify_knowledge_potential,
    classify_relations_fn=classify_relation_routes,
    enrich_participants_fn=enrich_participant_roles,
    consolidate_fn=None,
) -> dict:
    if KNOWLEDGE_PIPELINE_ENABLED:
        create_knowledge_schema(session)
    else:
        create_entity_schema(session)

    posts = _load_posts(session)
    print(
        f"Tìm thấy {len(posts)} post để xử lý "
        f"với {KNOWLEDGE_WORKERS} worker."
    )

    with ThreadPoolExecutor(max_workers=KNOWLEDGE_WORKERS) as executor:
        future_to_post = {
            executor.submit(
                _extract_post,
                classify_post_fn,
                extract_knowledge_fn,
                validate_knowledge,
                classify_relations_fn,
                post["platform"],
                post["post_id"],
                post["content"],
                enrich_participants_fn,
            ): (index, post)
            for index, post in enumerate(posts, start=1)
        }
        summary = {
            "total": len(posts),
            "skipped": 0,
            "deep": 0,
            "failed": 0,
            "relation_routes": [],
            "relation_router": {
                "events": 0,
                "pairs": 0,
                "groups": {},
                "actions": {"USE_BASE_DATA": 0, "ENRICH": 0},
            },
        }
        batch_mention_keys = []
        for completed, future in enumerate(as_completed(future_to_post), start=1):
            original_index, post = future_to_post[future]
            outcome = _save_extracted_post(
                session,
                post,
                future,
                original_index=original_index,
                completed=completed,
                total=len(posts),
                mention_keys_out=batch_mention_keys,
                relation_routes_out=summary["relation_routes"],
                relation_router_summary=summary["relation_router"],
            )
            summary[outcome] += 1

    consolidation = {
        "mentions": 0,
        "events_created": 0,
        "auto_merged": 0,
        "possible": 0,
        "descriptions_updated": 0,
        "failed": 0,
    }
    if KNOWLEDGE_PIPELINE_ENABLED and consolidate_fn is not None:
        try:
            consolidation = consolidate_fn(
                session,
                mention_keys=batch_mention_keys,
            )
        except Exception:
            LOGGER.exception("Lỗi bước consolidation cuối batch")
            consolidation["failed"] += 1
    summary["consolidation"] = consolidation

    print(
        "\nTổng kết pipeline: "
        f"{summary['total']} post, {summary['skipped']} skipped, "
        f"{summary['deep']} deep, {summary['failed']} lỗi."
    )
    print(
        "Consolidation: "
        f"{consolidation['mentions']} mention, "
        f"{consolidation['events_created']} Event mới, "
        f"{consolidation['auto_merged']} auto-merge, "
        f"{consolidation['possible']} nghi vấn, "
        f"{consolidation['descriptions_updated']} mô tả cập nhật, "
        f"{consolidation['failed']} lỗi."
    )
    return summary


def _save_extracted_post(
    session,
    post,
    future,
    *,
    original_index: int,
    completed: int,
    total: int,
    mention_keys_out: list[str] | None = None,
    relation_routes_out: list[dict] | None = None,
    relation_router_summary: dict | None = None,
) -> str:
    """Persist one validated extraction on the main thread."""
    platform = post["platform"]
    post_id = post["post_id"]
    content = post["content"]
    print(
        f"\n[{completed}/{total}] Hoàn tất trích xuất {platform} post {post_id} "
        f"(thứ tự ban đầu: {original_index})"
    )
    try:
        extraction_result = future.result()
        knowledge = extraction_result["knowledge"]
        relation_routes = extraction_result["relation_routes"]
        classification = extraction_result["classification"]
        classifier_decision = extraction_result["classifier_decision"]
        if not KNOWLEDGE_PIPELINE_ENABLED:
            entities = knowledge["entities"]
            print(json.dumps(entities, ensure_ascii=False, indent=2))
            saved_count = save_entities(session, platform, post_id, entities)
            print(f"Đã lưu {saved_count}/{len(entities)} entity hợp lệ.")
            return "deep"

        print(json.dumps(knowledge, ensure_ascii=False, indent=2))
        print(json.dumps(relation_routes, ensure_ascii=False, indent=2))
        counts = session.execute_write(
            save_knowledge_tx,
            platform,
            post_id,
            knowledge,
            classification,
            classifier_decision,
        )
        if mention_keys_out is not None:
            mention_keys_out.extend(
                event.get("mention_key", event["event_key"])
                for event in knowledge["events"]
            )
        if classifier_decision == "DEEP" and relation_routes_out is not None:
            relation_routes_out.append(
                {
                    "platform": platform,
                    "post_id": post_id,
                    **relation_routes,
                }
            )
        if classifier_decision == "DEEP" and relation_router_summary is not None:
            _accumulate_relation_router_summary(
                relation_router_summary,
                relation_routes,
            )
        print(
            "Đã lưu "
            f"{counts['entities']} Entity, {counts['events']} Event, "
            f"{counts['event_relations']} quan hệ Event."
        )
        return "skipped" if classifier_decision == "SKIPPED" else "deep"
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
        return "failed"


def _accumulate_relation_router_summary(summary: dict, routes: dict) -> None:
    event_routes = routes.get("event_routes", [])
    pair_routes = routes.get("pair_routes", [])
    summary["events"] += len(event_routes)
    summary["pairs"] += len(pair_routes)
    for route in [*event_routes, *pair_routes]:
        for detail in route.get("route_details", []):
            group = detail["relation_group"]
            action = detail["action"]
            summary["groups"][group] = summary["groups"].get(group, 0) + 1
            summary["actions"][action] = summary["actions"].get(action, 0) + 1
