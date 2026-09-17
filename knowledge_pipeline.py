from knowledge_gemini import log_post_calls

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from langsmith import traceable

from knowledge_settings import (
    KNOWLEDGE_MAX_RETRIES,
    KNOWLEDGE_MODULES,
    validate_module_config,
    KNOWLEDGE_PIPELINE_ENABLED,
    KNOWLEDGE_WORKERS,
    LOGGER,
    POST_LIMIT,
)
from knowledge_extraction import classify_knowledge_potential, extract_knowledge
from knowledge_relation_router import classify_relation_routes
from knowledge_relations.participant_role import extract_participants
from knowledge_relations.event_relation import extract_event_relations
from knowledge_relations.entity_hierarchy.location_hierarchy import enrich_location_hierarchy
from knowledge_relations.entity_hierarchy.organization_hierarchy import extract_context, enrich_organization_hierarchy
from knowledge_persistence import (
    create_entity_schema,
    create_knowledge_schema,
    mark_knowledge_failure,
    save_entities,
    save_knowledge_tx,
    save_module_knowledge_tx,
    mark_module_completed,
    complete_consolidated_modules,
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
@log_post_calls
def _extract_post(
    classify_post_fn,
    extract_knowledge_fn,
    validate_knowledge_fn,
    classify_relations_fn,
    platform: str,
    post_id: str,
    content: str,
    extract_participants_fn=extract_participants,
    extract_event_relations_fn=extract_event_relations,
) -> dict:
    if not KNOWLEDGE_PIPELINE_ENABLED:
        return {
            "classification": None,
            "classifier_decision": None,
            "knowledge": extract_knowledge_fn(content),
            "relation_routes": {"detected_modules": []},
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
        else {"detected_modules": []}
    )
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
    extract_participants_fn=extract_participants,
    extract_event_relations_fn=extract_event_relations,
    enrich_locations_fn=enrich_location_hierarchy,
    consolidate_fn=None,
    enrich_organizations_fn=enrich_organization_hierarchy,
    organization_context_fn=extract_context,
) -> dict:
    if KNOWLEDGE_PIPELINE_ENABLED:
        validate_module_config()
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
                extract_participants_fn,
                extract_event_relations_fn,
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
                "groups": {},
            },
            "location_hierarchy": {
                "locations": 0, "content_edges": 0, "osm_edges": 0,
                "parents_created": 0, "parents_reused": 0,
                "skipped": 0, "errors": 0,
            },
        }
        batch_mention_keys = []
        analyzed_counts = {"organizations": set(), "locations": set(), "events": 0}
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
                enrich_locations_fn=enrich_locations_fn,
                location_summary=summary["location_hierarchy"],
                extract_participants_fn=extract_participants_fn,
                extract_event_relations_fn=extract_event_relations_fn,
                enrich_organizations_fn=enrich_organizations_fn,
                organization_context_fn=organization_context_fn,
                analyzed_counts=analyzed_counts,
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
    if (KNOWLEDGE_PIPELINE_ENABLED and KNOWLEDGE_MODULES.get("EVENT_HIERARCHY")
            and batch_mention_keys and consolidate_fn is not None):
        try:
            consolidation = consolidate_fn(
                session,
                mention_keys=batch_mention_keys,
            )
            session.execute_write(complete_consolidated_modules, batch_mention_keys)
        except Exception:
            LOGGER.exception("Lỗi bước consolidation cuối batch")
            consolidation["failed"] += 1
    summary["consolidation"] = consolidation
    summary["analyzed"] = {
        "organizations": len(analyzed_counts["organizations"]),
        "locations": len(analyzed_counts["locations"]),
        "events": analyzed_counts["events"],
    }

    print(
        "\nTổng kết pipeline: "
        f"{summary['total']} post, {summary['skipped']} skipped, "
        f"{summary['deep']} deep, {summary['failed']} lỗi."
    )
    print(
        "Kết quả batch: "
        f"{summary['analyzed']['organizations']} node Organization, "
        f"{summary['analyzed']['locations']} node Location "
        "(node được lưu, không trùng trong batch, gồm cả node đã có); "
        f"{summary['analyzed']['events']} sự kiện được phân tích và lưu (EventMention)."
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


@log_post_calls
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
    enrich_locations_fn=enrich_location_hierarchy,
    location_summary: dict | None = None,
    extract_participants_fn=extract_participants,
    extract_event_relations_fn=extract_event_relations,
    enrich_organizations_fn=enrich_organization_hierarchy,
    organization_context_fn=extract_context,
    analyzed_counts=None,
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

        runnable_modules = {
            name for name in relation_routes["detected_modules"]
            if classifier_decision == "DEEP" and KNOWLEDGE_MODULES.get(name, False)
        }
        print(json.dumps(knowledge, ensure_ascii=False, indent=2))
        print(json.dumps(relation_routes, ensure_ascii=False, indent=2))
        if "ENTITY_HIERARCHY" in runnable_modules and any(e.get("type") == "ORGANIZATION" for e in knowledge["entities"]):
            try:
                knowledge["organization_context"] = organization_context_fn(content, knowledge)
            except Exception:
                LOGGER.exception("Organization context unavailable; retry after base persistence")
        counts = session.execute_write(
            save_knowledge_tx,
            platform,
            post_id,
            knowledge,
            classification,
            classifier_decision,
            detected_modules=relation_routes["detected_modules"],
            runnable_modules=runnable_modules,
        )
        if analyzed_counts is not None:
            node_ids = counts.get("entity_node_ids", {})
            analyzed_counts["organizations"].update(node_ids.get("ORGANIZATION", []))
            analyzed_counts["locations"].update(node_ids.get("LOCATION", []))
            analyzed_counts["events"] += counts["events"]
        for module, extract_fn in (
            ("PARTICIPANT_ROLE", extract_participants_fn),
            ("EVENT_RELATION", extract_event_relations_fn),
        ):
            if module not in runnable_modules:
                continue
            try:
                enriched = extract_fn(content, knowledge)
                enriched = validate_knowledge(content, enriched, platform, post_id)
                # Participant enrichment must not change already persisted identities.
                identities = {e["local_id"]: e for e in knowledge["events"]}
                for event in enriched["events"]:
                    original = identities[event["local_id"]]
                    for key in ("mention_key", "event_key"):
                        event[key] = original[key]
                if "organization_context" in knowledge:
                    enriched["organization_context"] = knowledge["organization_context"]
                session.execute_write(save_module_knowledge_tx, platform, post_id, enriched, module)
                knowledge = enriched
            except Exception:
                LOGGER.exception("Module %s thất bại sau khi lưu nền cho %s", module, post_id)
        hierarchy_ok = True
        if (
            "ENTITY_HIERARCHY" in runnable_modules
            and any(entity.get("type") == "LOCATION" for entity in knowledge["entities"])
        ):
            try:
                hierarchy = enrich_locations_fn(
                    session, platform, post_id, content, knowledge
                )
            except Exception:
                # Base extraction is already committed. Hierarchy has its own
                # retry/backfill lifecycle and must never reclassify this post
                # as a knowledge-extraction failure.
                LOGGER.exception(
                    "LOCATION hierarchy thất bại sau khi đã lưu base cho %s",
                    post_id,
                )
                hierarchy = {"errors": 1}
            hierarchy_ok = not bool(hierarchy.get("errors", 0))
            if location_summary is not None:
                for key, value in hierarchy.items():
                    location_summary[key] = location_summary.get(key, 0) + value
        if "ENTITY_HIERARCHY" in runnable_modules and any(e.get("type") == "ORGANIZATION" for e in knowledge["entities"]):
            try:
                organization = enrich_organizations_fn(session, platform, post_id, content, knowledge)
                hierarchy_ok = hierarchy_ok and not bool(organization.get("errors", 0))
            except Exception:
                hierarchy_ok = False
                LOGGER.exception("ORGANIZATION hierarchy failed after base persistence for %s", post_id)
        if "ENTITY_HIERARCHY" in runnable_modules and hierarchy_ok:
            try:
                session.execute_write(mark_module_completed, platform, post_id, "ENTITY_HIERARCHY")
            except Exception:
                LOGGER.exception("Không thể cập nhật modules_completed cho %s", post_id)
        if mention_keys_out is not None and "EVENT_HIERARCHY" in runnable_modules:
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
    for group in sorted(set(routes.get("detected_modules", []))):
        summary["groups"][group] = summary["groups"].get(group, 0) + 1
