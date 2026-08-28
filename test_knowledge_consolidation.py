import os
import unittest
from datetime import date

from neo4j import GraphDatabase

from knowledge_consolidation import (
    _resolve_prompt,
    _validated_decisions,
    action_family,
    best_auto_merge_decision,
    candidate_score,
    candidate_score_components,
    comparison_profile,
    effective_match_decision,
    evaluate_merge_guard,
    select_candidates,
    consolidate_pending_mentions,
)
from knowledge_settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from migrate_event_mentions import dry_run, migrate_legacy_row


class EventConsolidationTests(unittest.TestCase):
    @staticmethod
    def participant(name, role="ACTOR", identified=True):
        return {"name": name, "role": role, "identified": identified}

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

        self.assertEqual(action_family(mention["description"]), "APOLOGIZE")
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

    def test_attend_synonyms_share_an_action_family(self):
        self.assertEqual(action_family("Infantino dự khán chung kết"), "ATTEND")
        self.assertEqual(action_family("Chủ tịch FIFA xem trận chung kết"), "ATTEND")
        self.assertEqual(
            action_family("Infantino có mặt trên khán đài"), "ATTEND"
        )

    def test_same_actor_and_attend_action_rank_above_lexical_similarity(self):
        mention = self.mention("Infantino dự khán chung kết ASEAN Cup")
        mention["participants"] = [self.participant("gianni infantino")]
        event = self.event(
            "final",
            "Chủ tịch FIFA xem trận quyết định giữa Việt Nam và Thái Lan",
        )
        event["participants"] = [self.participant("gianni infantino")]

        components = candidate_score_components(mention, event)

        self.assertEqual(components["action"], 0.30)
        self.assertEqual(components["actor"], 0.25)
        self.assertGreaterEqual(candidate_score(mention, event), 0.60)

    def test_location_or_context_participant_does_not_count_as_actor(self):
        mention = self.mention("Infantino dự khán chung kết tại Hà Nội")
        mention["participants"] = [self.participant("hà nội", "LOCATION")]
        event = self.event("final", "Madam Pang xem chung kết tại Hà Nội")
        event["participants"] = [self.participant("hà nội", "LOCATION")]

        components = candidate_score_components(mention, event)

        self.assertEqual(components["actor"], 0.0)
        self.assertEqual(components["location"], 0.05)

    def test_different_named_main_actors_are_a_hard_conflict(self):
        mention = self.mention("Infantino dự khán chung kết Việt Nam - Thái Lan")
        mention["participants"] = [
            self.participant("gianni infantino"),
            self.participant("việt nam", "TARGET"),
        ]
        event = self.event(
            "madam-pang", "Madam Pang xem chung kết Việt Nam - Thái Lan"
        )
        event["participants"] = [
            self.participant("madam pang"),
            self.participant("việt nam", "TARGET"),
        ]

        guard = evaluate_merge_guard(mention, event)
        effective = effective_match_decision({
            "candidate_event_key": "madam-pang",
            "decision": "SAME_EVENT",
            "confidence": 0.99,
            "reason": "Cùng trận",
        }, mention, event)

        self.assertEqual(guard["status"], "BLOCK")
        self.assertIn("MAIN_ACTOR_CONFLICT", guard["reason_codes"])
        self.assertEqual(effective["decision"], "DIFFERENT_EVENT")

    def test_different_actions_in_same_trip_are_not_candidates(self):
        mention = self.mention("Infantino khảo sát sân vận động 60.000 chỗ")
        mention["participants"] = [self.participant("gianni infantino")]
        event = self.event("attend", "Infantino dự khán chung kết ASEAN Cup")
        event["participants"] = [self.participant("gianni infantino")]

        self.assertEqual(action_family(mention["description"]), "INSPECT")
        self.assertEqual(select_candidates(mention, [event]), [])
        self.assertEqual(evaluate_merge_guard(mention, event)["status"], "BLOCK")

    def test_additional_target_and_location_are_not_conflicts(self):
        mention = self.mention("Infantino dự khán chung kết ASEAN Cup")
        mention["participants"] = [self.participant("gianni infantino")]
        event = self.event(
            "detailed",
            "Infantino dự khán chung kết ASEAN Cup giữa Việt Nam và Thái Lan tại Hà Nội",
        )
        event["participants"] = [
            self.participant("gianni infantino"),
            self.participant("việt nam", "TARGET"),
            self.participant("thái lan", "TARGET"),
            self.participant("hà nội", "LOCATION"),
        ]

        self.assertEqual(evaluate_merge_guard(mention, event)["status"], "PASS")

    def test_full_occurrence_date_conflict_blocks_high_confidence_merge(self):
        mention = self.mention("Infantino dự khán trận ngày 25/8/2026")
        mention.update({
            "time_expression": "25/8/2026",
            "participants": [self.participant("gianni infantino")],
        })
        event = self.event("other-date", "Infantino dự khán trận ngày 27/8/2026")
        event.update({
            "occurrence_times": ["27/8/2026"],
            "participants": [self.participant("gianni infantino")],
        })

        components = candidate_score_components(mention, event)
        effective = effective_match_decision({
            "candidate_event_key": "other-date",
            "decision": "SAME_EVENT",
            "confidence": 0.99,
            "reason": "Cùng actor và hành động",
        }, mention, event)

        self.assertEqual(components["time"], -0.45)
        self.assertEqual(effective["decision"], "DIFFERENT_EVENT")
        self.assertIn("OCCURRENCE_DATE_CONFLICT", effective["guard_reason_codes"])

    def test_partial_dates_use_posted_year_and_match_time_of_day_variant(self):
        mention = self.mention("Infantino thăm PVF ngày 27/8")
        mention.update({
            "type": "VISIT",
            "time_expression": "27/8",
            "posted_at": date(2026, 8, 28),
            "participants": [self.participant("gianni infantino")],
        })
        event = self.event("pvf", "Infantino thăm PVF sáng 27/8")
        event.update({
            "type": "VISIT",
            "occurrence_times": ["sáng 27/8"],
            "first_seen_at": date(2026, 8, 27),
            "participants": [self.participant("gianni infantino")],
        })

        left = comparison_profile(mention)
        right = comparison_profile(event)
        guard = evaluate_merge_guard(mention, event)

        self.assertEqual(left["occurrence_dates"], ["2026-08-27"])
        self.assertEqual(right["occurrence_dates"], ["2026-08-27"])
        self.assertFalse(left["has_unparsed_time"])
        self.assertFalse(right["has_unparsed_time"])
        self.assertEqual(guard["status"], "PASS")

    def test_ambiguous_participants_and_partial_time_require_review(self):
        mention = self.mention("Một quan chức dự khán chung kết ngày 25/8")
        mention.update({
            "time_expression": "25/8",
            "participants": [self.participant("một quan chức", identified=False)],
        })
        event = self.event("unknown", "Một lãnh đạo xem chung kết cuối tháng 8")
        event.update({
            "occurrence_times": ["cuối tháng 8"],
            "participants": [self.participant("một lãnh đạo", identified=False)],
        })

        guard = evaluate_merge_guard(mention, event)

        self.assertEqual(guard["status"], "REVIEW")
        self.assertIn("MAIN_ACTOR_ANONYMOUS_OR_UNSTABLE", guard["reason_codes"])
        self.assertIn("OCCURRENCE_TIME_UNCERTAIN", guard["reason_codes"])

    def test_low_confidence_same_event_becomes_possible(self):
        mention = self.mention("Infantino dự khán chung kết")
        mention["participants"] = [self.participant("gianni infantino")]
        event = self.event("candidate", "Infantino xem trận chung kết")
        event["participants"] = [self.participant("gianni infantino")]

        effective = effective_match_decision({
            "candidate_event_key": "candidate",
            "decision": "SAME_EVENT",
            "confidence": 0.70,
            "reason": "Có vẻ cùng trận",
        }, mention, event)

        self.assertEqual(effective["guard_status"], "PASS")
        self.assertEqual(effective["decision"], "POSSIBLE_SAME_EVENT")

    def test_semantic_score_precedes_confidence_for_merge_choice(self):
        best = best_auto_merge_decision([
            {
                "candidate_event_key": "context-only",
                "decision": "SAME_EVENT",
                "confidence": 0.99,
                "retrieval_score": 0.35,
            },
            {
                "candidate_event_key": "same-occurrence",
                "decision": "SAME_EVENT",
                "confidence": 0.92,
                "retrieval_score": 0.70,
            },
        ])

        self.assertEqual(best["candidate_event_key"], "same-occurrence")

    def test_investigation_follow_up_is_not_blocked_by_different_actor(self):
        mention = self.mention(
            "Công an xác minh vụ bảo vệ hành hung tài xế tại Louis City"
        )
        mention.update({
            "type": "INVESTIGATION",
            "participants": [
                self.participant("công an", "ACTOR"),
                self.participant("tài xế", "VICTIM", identified=False),
            ],
        })
        event = self.event(
            "assault", "Bảo vệ hành hung tài xế tại Louis City"
        )
        event.update({
            "type": "ASSAULT",
            "participants": [
                self.participant("bảo vệ", "ACTOR", identified=False),
                self.participant("tài xế", "VICTIM", identified=False),
            ],
        })

        self.assertEqual(evaluate_merge_guard(mention, event)["status"], "PASS")

    def test_resolver_prompt_centers_occurrence_and_omits_retrieval_score(self):
        mention = self.mention("Infantino dự khán chung kết")
        event = dict(self.event("candidate", "Chủ tịch FIFA xem chung kết"),
                     retrieval_score=0.80,
                     score_components={"action": 0.30, "total": 0.60})

        prompt = _resolve_prompt(mention, [event])

        self.assertIn("cùng một occurrence", prompt)
        self.assertIn("Madam Pang", prompt)
        self.assertIn("semantic_score_components", prompt)
        self.assertNotIn('"retrieval_score"', prompt)

    def test_migration_dry_run_reports_guard_override(self):
        rows = []
        for event_key, actor in (("infantino", "gianni infantino"),
                                 ("madam-pang", "madam pang")):
            rows.append({
                "event_key": event_key,
                "mention_key": f"mention-{event_key}",
                "current_event_key": event_key,
                "type": "SPORTS_EVENT",
                "title": "Dự khán chung kết",
                "description": f"{actor} dự khán chung kết Việt Nam - Thái Lan",
                "evidence_text": f"{actor} dự khán chung kết Việt Nam - Thái Lan",
                "status": "COMPLETED",
                "time_expression": "25/8/2026",
                "participants": [self.participant(actor)],
                "posted_at": None,
                "created_at": None,
            })

        def model(_prompt, _schema):
            return {"decisions": [{
                "candidate_event_key": "madam-pang",
                "decision": "SAME_EVENT",
                "confidence": 0.99,
                "reason": "Cùng trận",
            }]}

        report = dry_run(rows, model)

        self.assertEqual(report["same_event"], [])
        self.assertEqual(len(report["different_event"]), 1)
        self.assertEqual(len(report["guard_overrides"]), 1)
        self.assertIn(
            "MAIN_ACTOR_CONFLICT",
            report["guard_overrides"][0]["guard_reason_codes"],
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
                WITH collect(mention) AS mentions
                MERGE (actor:Entity {
                    normalized_name: 'codex-vetc-consolidation-actor',
                    type: 'ORGANIZATION'
                })
                ON CREATE SET actor.name = 'Codex VETC consolidation actor'
                FOREACH (mention IN mentions |
                    MERGE (mention)-[:HAS_PARTICIPANT {role: 'ACTOR'}]->(actor)
                )
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
        session.run(
            """
            MATCH (actor:Entity {
                normalized_name: 'codex-vetc-consolidation-actor',
                type: 'ORGANIZATION'
            })
            DETACH DELETE actor
            """
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
