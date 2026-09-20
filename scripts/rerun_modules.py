"""Chạy các module được chỉ định trên bài đã phân tích.

Chạy từ thư mục gốc: python -m scripts.rerun_modules
Cấu hình MODULES_TO_RERUN độc lập với KNOWLEDGE_MODULES của pipeline bài mới.
Hiện hỗ trợ EVENT_HIERARCHY: xử lý mention PENDING/ERROR của bài đăng
hôm nay/hôm qua, giữ nguyên phạm vi và thứ tự của tác vụ gộp Event.
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from neo4j import GraphDatabase

from knowledge_gemini import GeminiKnowledgeCaller
from knowledge_persistence import complete_consolidated_modules
from knowledge_relations.event_hierarchy import consolidate_pending_mentions
from knowledge_settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EVENT_MENTION_LIMIT

MODULES_TO_RERUN = ("EVENT_HIERARCHY",)
_gemini_caller: GeminiKnowledgeCaller | None = None

EVENT_CONSOLIDATION_TIMEZONE = "Asia/Ho_Chi_Minh"
# Các posted_at không có timezone trong nguồn được hiểu là UTC.
EVENT_POSTED_AT_NAIVE_TIMEZONE = "UTC"


def consolidate_recent_posts(session, call_model=None, *, now=None) -> dict:
    timezone = ZoneInfo(EVENT_CONSOLIDATION_TIMEZONE)
    current = datetime.now(timezone) if now is None else now.astimezone(timezone)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    rows = list(session.run(
        """
        MATCH (p:Post)-[:HAS_EVENT_MENTION]->(m:EventMention)
              -[:EVIDENCE_FOR]->(:Event)
        WHERE p.knowledge_processed = true
          AND p.knowledge_classifier_decision = 'DEEP'
          AND p.posted_at IS NOT NULL
          AND coalesce(m.consolidation_status, 'PENDING') IN ['PENDING', 'ERROR']
        WITH p, m,
             datetime({datetime: p.posted_at, timezone: $naive_timezone}) AS posted_time
        WHERE posted_time >= datetime($start) AND posted_time <= datetime($end)
        RETURN DISTINCT p.platform AS platform, p.platform_id AS post_id,
               posted_time AS posted_at, m.mention_key AS mention_key,
               m.created_at AS mention_created_at
        ORDER BY posted_at DESC, mention_created_at DESC, mention_key
        LIMIT $mention_limit
        """,
        # Avoid driver serialization of ZoneInfo datetimes (can segfault).
        start=start.isoformat(), end=current.isoformat(),
        naive_timezone=EVENT_POSTED_AT_NAIVE_TIMEZONE,
        mention_limit=EVENT_MENTION_LIMIT,
    ))
    mention_keys = list(dict.fromkeys(row["mention_key"] for row in rows))
    total = len({(row["platform"], row["post_id"]) for row in rows})
    print(f"Chỉ gộp Event: {start.isoformat()} đến {current.isoformat()}; "
          f"{total} bài, {len(mention_keys)}/{EVENT_MENTION_LIMIT} mention tối đa, "
          "posted_at giảm dần.")
    if not mention_keys:
        return {"total": 0, "consolidation": {}}
    stats = consolidate_pending_mentions(
        session,
        call_model=call_model if call_model is not None else get_gemini_caller(),
        mention_keys=mention_keys,
        preserve_mention_order=True,
    )
    session.execute_write(complete_consolidated_modules, mention_keys)
    print(f"Kết quả gộp Event: {stats}")
    return {"total": total, "consolidation": stats}


def get_gemini_caller() -> GeminiKnowledgeCaller:
    """Create one Gemini client and reuse it for the whole pipeline."""
    global _gemini_caller
    if _gemini_caller is None:
        _gemini_caller = GeminiKnowledgeCaller()
    return _gemini_caller


def rerun_modules(session, call_model=None, *, modules=None, now=None) -> dict:
    """Validate every requested module before executing any database writes."""
    selected = tuple(dict.fromkeys(MODULES_TO_RERUN if modules is None else modules))
    handlers = {"EVENT_HIERARCHY": consolidate_recent_posts}
    unsupported = set(selected) - handlers.keys()
    if unsupported:
        raise ValueError(f"Module chưa hỗ trợ chạy lại: {', '.join(sorted(unsupported))}")
    results = {
        name: handlers[name](session, call_model=call_model, now=now)
        for name in selected
    }
    return {
        "total": sum(result["total"] for result in results.values()),
        "modules": results,
    }


def main() -> None:
    logging.getLogger("knowledge.api").setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    print(f"Chạy lại module trên Neo4j: {NEO4J_URI}")
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        with driver.session(database="neo4j") as session:
            summary = rerun_modules(session)
            if _gemini_caller is not None:
                _gemini_caller.print_cost_summary(
                    target_posts=summary["total"],
                    stage_label="chạy lại module",
                )
    finally:
        if _gemini_caller is not None:
            _gemini_caller.close()
        driver.close()


if __name__ == "__main__":
    main()
