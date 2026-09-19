from datetime import datetime, timezone
import os
from unittest.mock import Mock, patch

import pytest
from neo4j import GraphDatabase

import rerun_modules as subject
from knowledge_relations import event_hierarchy


@pytest.mark.skipif(os.getenv("RUN_NEO4J_INTEGRATION") != "1", reason="requires local Neo4j")
def test_posted_at_conversion_accepts_local_and_zoned_datetimes():
    # Exercise the actual conversion expression used by the batch query.
    fake = Mock()
    fake.run.return_value = []
    subject.consolidate_recent_posts(fake)
    query = fake.run.call_args.args[0]
    expression = query.split("WITH p, m,", 1)[1].split(" AS posted_time", 1)[0].strip()
    with GraphDatabase.driver(
        subject.NEO4J_URI, auth=(subject.NEO4J_USER, subject.NEO4J_PASSWORD)
    ) as driver, driver.session(database="neo4j") as session:
        rows = list(session.run(
            "UNWIND [localdatetime('2026-09-14T06:15:16'), "
            "datetime('2026-09-14T13:15:16+07:00')] AS value "
            "WITH {posted_at: value} AS p RETURN " + expression + " AS value",
            naive_timezone="UTC",
        ))
    assert len(rows) == 2
    assert rows[0]["value"] == rows[1]["value"]
    assert rows[0]["value"].hour == 6


def test_recent_batch_uses_vietnam_calendar_days_and_ordered_keys():
    session = Mock()
    session.run.return_value = [
        {"platform": "facebook", "post_id": "new", "mention_key": "new-mention"},
        {"platform": "facebook", "post_id": "old", "mention_key": "old-mention"},
    ]
    model = Mock()
    # UTC Sep 16 is already Sep 17 in Vietnam.
    now = datetime(2026, 9, 16, 18, tzinfo=timezone.utc)
    with patch.object(subject, "consolidate_pending_mentions", return_value={}) as run:
        result = subject.consolidate_recent_posts(session, model, now=now)
    assert session.run.call_args.kwargs["start"] == "2026-09-16T00:00:00+07:00"
    assert session.run.call_args.kwargs["end"] == "2026-09-17T01:00:00+07:00"
    assert result["total"] == 2
    run.assert_called_once_with(
        session, call_model=model,
        mention_keys=["new-mention", "old-mention"], preserve_mention_order=True,
    )


def test_empty_batch_does_not_create_model_or_consolidate_all_mentions():
    session = Mock()
    session.run.return_value = []
    with patch.object(subject, "get_openai_caller") as model, patch.object(
        subject, "consolidate_pending_mentions"
    ) as run:
        assert subject.consolidate_recent_posts(session)["total"] == 0
    model.assert_not_called()
    run.assert_not_called()
    session.execute_write.assert_not_called()


def test_consolidation_preserves_requested_order_after_loading():
    mentions = [{"mention_key": key, "current_event_key": key} for key in ["old", "new"]]
    with patch.object(event_hierarchy, "_load_pending_mentions", return_value=mentions), patch.object(
        event_hierarchy, "_load_canonical_events", return_value=[]
    ), patch.object(event_hierarchy, "_refresh_current_event", return_value=None) as refresh:
        event_hierarchy.consolidate_pending_mentions(
            Mock(), Mock(), ["new", "old"], preserve_mention_order=True,
        )
    assert [call.args[1]["mention_key"] for call in refresh.call_args_list] == ["new", "old"]
