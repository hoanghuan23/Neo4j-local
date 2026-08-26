import os
import unittest

from neo4j import GraphDatabase

from knowledge_consolidation import (
    _validated_decisions,
    action_family,
    candidate_score,
    select_candidates,
    consolidate_pending_mentions,
)
from knowledge_settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from migrate_event_mentions import migrate_legacy_row


class EventConsolidationTests(unittest.TestCase):
    def mention(self, description):
        return {
            "mention_key": "m1",
            "current_event_key": "current",
            "type": "OTHER",
            "description": description,
            "evidence_text": description,
            "participants": ["vetc"],
            "posted_at": None,
        }

    def event(self, key, description):
        return {
            "event_key": key,
            "type": "OTHER",
            "description": description,
            "descriptions": [description],
            "participants": ["vetc"],
            "last_seen_at": None,
        }

    def test_vetc_stop_wordings_are_candidates(self):
        mention = self.mention("VETC quyết định chưa áp dụng chính sách phí")
        event = self.event("stop", "VETC tạm dừng thu phí ví điện tử")

        selected = select_candidates(mention, [event])

        self.assertEqual([item["event_key"] for item in selected], ["stop"])
        self.assertGreater(candidate_score(mention, event), 0.5)

    def test_start_and_stop_are_rejected_before_model(self):
        mention = self.mention("VETC tạm dừng thu phí ví điện tử")
        event = self.event("start", "VETC bắt đầu áp dụng phí ví điện tử")

        self.assertEqual(select_candidates(mention, [event]), [])

    def test_apology_is_not_a_stop_candidate(self):
        mention = self.mention("VETC xin lỗi khách hàng")
        event = self.event("stop", "VETC tạm dừng thu phí ví điện tử")

        self.assertEqual(action_family(mention["description"]), "APOLOGY")
        self.assertEqual(select_candidates(mention, [event]), [])

    def test_investigation_of_same_assault_is_a_candidate(self):
        mention = self.mention(
            "Công an phường Hoàng Mai đang xác minh vụ người đàn ông mặc đồ "
            "bảo vệ hành hung tài xế xe ôm công nghệ tại Louis City"
        )
        mention.update({
            "type": "INVESTIGATION",
            "participants": ["công an phường hoàng mai"],
        })
        event = self.event(
            "assault",
            "Người đàn ông mặc đồ bảo vệ hành hung tài xế xe ôm công nghệ "
            "tại Louis City Hoàng Mai",
        )
        event.update({
            "type": "ASSAULT",
            "participants": ["tài xế xe ôm công nghệ", "louis city hoàng mai"],
        })

        selected = select_candidates(mention, [event])

        self.assertEqual([item["event_key"] for item in selected], ["assault"])
        self.assertGreaterEqual(selected[0]["retrieval_score"], 0.20)

    def test_unrelated_investigation_is_not_a_candidate(self):
        mention = self.mention(
            "Công an xác minh vụ bảo vệ hành hung tài xế tại Louis City"
        )
        mention["type"] = "INVESTIGATION"
        mention["participants"] = ["công an phường hoàng mai"]
        event = self.event(
            "unrelated",
            "Một người bị bắt giữ trong vụ trộm xe máy tại quận khác",
        )
        event["type"] = "ARREST"
        event["participants"] = ["nghi phạm"]

        self.assertEqual(select_candidates(mention, [event]), [])

    def test_invalid_or_unknown_model_decisions_are_dropped(self):
        candidates = [self.event("known", "VETC tạm dừng thu phí")]
        raw = {
            "decisions": [
                {
                    "candidate_event_key": "known",
                    "decision": "SAME_EVENT",
                    "confidence": 0.95,
                    "reason": "Cùng diễn biến",
                },
                {
                    "candidate_event_key": "missing",
                    "decision": "SAME_EVENT",
                    "confidence": 1,
                    "reason": "Không tồn tại",
                },
            ]
        }

        self.assertEqual(
            _validated_decisions(raw, candidates),
            [{
                "candidate_event_key": "known",
                "decision": "SAME_EVENT",
                "confidence": 0.95,
                "reason": "Cùng diễn biến",
            }],
        )

@unittest.skipUnless(
    os.getenv("RUN_NEO4J_INTEGRATION") == "1",
    "set RUN_NEO4J_INTEGRATION=1 to exercise the local Neo4j instance",
)
class EventConsolidationIntegrationTests(unittest.TestCase):
    platform = "codex-consolidation-test"

    @classmethod
    def setUpClass(cls):
        cls.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    @classmethod
    def tearDownClass(cls):
        cls.driver.close()

    def setUp(self):
        with self.driver.session(database="neo4j") as session:
            self.cleanup(session)
            session.run(
                """
                UNWIND [
                    {post_id: '1', mention_key: 'codex-m1', event_key: 'codex-e1',
                     description: 'VETC tạm dừng thu phí ví điện tử'},
                    {post_id: '2', mention_key: 'codex-m2', event_key: 'codex-e2',
                     description: 'VETC quyết định chưa áp dụng chính sách phí'}
                ] AS item
                CREATE (post:Post {
                    platform: $platform,
                    platform_id: item.post_id,
                    posted_at: datetime()
                })
                CREATE (mention:EventMention {
                    mention_key: item.mention_key,
                    type: 'OTHER',
                    description: item.description,
                    evidence_text: item.description,
                    status: 'COMPLETED',
                    consolidation_status: 'PENDING',
                    created_at: datetime()
                })
                CREATE (event:Event {
                    event_key: item.event_key,
                    type: 'OTHER',
                    description: item.description,
                    status: 'COMPLETED',
                    schema_version: 2,
                    created_at: datetime(),
                    last_seen_at: datetime()
                })
                CREATE (post)-[:HAS_EVENT_MENTION]->(mention)
                CREATE (mention)-[:EVIDENCE_FOR]->(event)
                CREATE (post)-[:DESCRIBES]->(event)
                """,
                platform=self.platform,
            ).consume()

    def tearDown(self):
        with self.driver.session(database="neo4j") as session:
            self.cleanup(session)

    def cleanup(self, session):
        session.run(
            """
            MATCH (post:Post {platform: $platform})
            OPTIONAL MATCH (post)-[:HAS_EVENT_MENTION]->(mention:EventMention)
            OPTIONAL MATCH (mention)-[:EVIDENCE_FOR]->(event:Event)
            DETACH DELETE post, mention, event
            """,
            platform=self.platform,
        ).consume()

    def test_merges_two_mentions_and_writes_aggregate_description(self):
        def model(_prompt, schema):
            if "decisions" in schema["properties"]:
                return {
                    "decisions": [{
                        "candidate_event_key": "codex-e2",
                        "decision": "SAME_EVENT",
                        "confidence": 0.97,
                        "reason": "Cùng quyết định tạm dừng/chưa áp dụng",
                    }]
                }
            return {
                "title": "VETC quyết định chưa áp dụng và tạm dừng chính sách thu phí ví điện tử",
                "description": (
                    "VETC quyết định chưa áp dụng và tạm dừng chính sách "
                    "thu phí ví điện tử."
                ),
                "type": "OTHER",
                "status": "COMPLETED",
                "source_mention_keys": ["codex-m1", "codex-m2"],
            }

        with self.driver.session(database="neo4j") as session:
            stats = consolidate_pending_mentions(
                session,
                model,
                mention_keys=["codex-m1", "codex-m2"],
            )
            record = session.run(
                """
                MATCH (post:Post {platform: $platform})-[:DESCRIBES]->(event:Event)
                WITH event, count(DISTINCT post) AS posts
                MATCH (mention:EventMention)-[:EVIDENCE_FOR]->(event)
                RETURN count(DISTINCT event) AS events,
                       posts,
                       count(DISTINCT mention) AS mentions,
                       event.title AS title,
                       event.description AS description,
                       event.member_count AS member_count
                """,
                platform=self.platform,
            ).single()

        self.assertEqual(stats["auto_merged"], 1)
        self.assertEqual(record["events"], 1)
        self.assertEqual(record["posts"], 2)
        self.assertEqual(record["mentions"], 2)
        self.assertEqual(record["member_count"], 2)
        self.assertIn("VETC quyết định", record["title"])
        self.assertIn("chưa áp dụng", record["description"])

    def test_legacy_event_is_converted_idempotently(self):
        row = {
            "platform": self.platform,
            "post_id": "3",
            "mention_key": "codex-legacy-mention",
            "event_key": "codex-legacy-event",
            "type": "OTHER",
            "description": "VETC tạm dừng thu phí",
            "evidence_text": "VETC tạm dừng thu phí",
            "status": "COMPLETED",
            "time_expression": None,
            "confidence": 0.9,
        }
        with self.driver.session(database="neo4j") as session:
            session.run(
                """
                CREATE (post:Post {
                    platform: $platform,
                    platform_id: '3',
                    content: 'VETC tạm dừng thu phí'
                })
                CREATE (event:Event {
                    event_key: 'codex-legacy-event',
                    type: 'OTHER',
                    description: 'VETC tạm dừng thu phí',
                    evidence_text: 'VETC tạm dừng thu phí',
                    status: 'COMPLETED'
                })
                CREATE (post)-[:DESCRIBES]->(event)
                """,
                platform=self.platform,
            ).consume()
            session.execute_write(migrate_legacy_row, row)
            session.execute_write(migrate_legacy_row, row)
            record = session.run(
                """
                MATCH (post:Post {platform: $platform, platform_id: '3'})
                      -[:HAS_EVENT_MENTION]->(mention:EventMention)
                      -[:EVIDENCE_FOR]->(event:Event)
                RETURN count(DISTINCT mention) AS mentions,
                       count(DISTINCT event) AS events,
                       event.schema_version AS schema_version
                """,
                platform=self.platform,
            ).single()

        self.assertEqual(record["mentions"], 1)
        self.assertEqual(record["events"], 1)
        self.assertEqual(record["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
