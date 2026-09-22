"""Candidate ranking across topics, using Neo4j's actual regex/Cypher engine."""
import os
from uuid import uuid4

import pytest
from neo4j import GraphDatabase

from backend.config import Settings
from backend.event_candidate_search import candidate_search_parameters
from backend.neo4j_repository import LOCATE_EVENT_QUERY


def test_empty_and_filler_queries_have_no_terms():
    for text in ('', '?!.*', 'ở tại và của'):
        assert candidate_search_parameters(text)['candidate_terms'] == []


def test_duplicate_words_do_not_inflate_match_requirement():
    first = candidate_search_parameters('cháy nhà máy')
    repeated = candidate_search_parameters('cháy cháy nhà máy')
    assert first['candidate_terms'] == repeated['candidate_terms']
    assert first['candidate_min_matches'] == repeated['candidate_min_matches']


@pytest.mark.skipif(os.getenv('RUN_NEO4J_INTEGRATION') != '1',
                    reason='requires local Neo4j')
@pytest.mark.parametrize('question,target,other', [
    ('bà cụ nhặt ve chai tìm thấy vàng',
     'Một bà cụ làm nghề nhặt ve chai phát hiện một túi vàng và mang nộp công an',
     'Giá vàng tăng mạnh hôm nay'),
    ('nhà máy cháy', 'Cháy lớn tại nhà máy sản xuất giấy',
     'Nhà văn giới thiệu tác phẩm mới'),
    ('cháy nhà máy', 'Cháy nhà máy sản xuất giấy',
     'Máy bơm chữa cháy gần nhà'),
    ('chay nha may', 'Cháy lớn tại nhà máy sản xuất giấy',
     'Nhà văn giới thiệu tác phẩm mới'),
    ('xe tải va chạm xe máy', 'Xe máy va chạm với xe tải trên quốc lộ',
     'Triển lãm xe điện khai mạc'),
    ('khai mạc lễ hội âm nhạc', 'Lễ hội âm nhạc mùa hè chính thức khai mạc',
     'Lễ khánh thành cầu vượt'),
    ('cứu hộ người dân ngập lụt', 'Lực lượng cứu hộ sơ tán người dân trong vùng ngập',
     'Người dân tham gia ngày hội đọc sách'),
    ('ABC 123', 'Công ty ABC ra mắt sản phẩm 123', 'Công ty XABC ra mắt 1234'),
])
def test_general_retrieval_and_ranking(question, target, other):
    settings = Settings()
    marker = 'candidate-test-' + uuid4().hex
    keys = [marker + '-' + name for name in ('target', 'other', 'orphan')]
    with GraphDatabase.driver(settings.neo4j_uri,
                              auth=(settings.neo4j_user, settings.neo4j_password)) as driver:
        with driver.session(database=settings.neo4j_database) as session:
            # Roll back every fixture, including on assertion failure.
            with session.begin_transaction() as tx:
                tx.run('''
                    CREATE (target:Event {event_key: $keys[0]})
                    SET target.description = CASE WHEN $legacy THEN null ELSE $target END,
                        target.title = CASE WHEN $legacy THEN $target ELSE null END
                    CREATE (other:Event {event_key: $keys[1], title: $other})
                    CREATE (orphan:Event {event_key: $keys[2], description: $question})
                    CREATE (p:Post {platform: 'test', platform_id: $marker})
                    CREATE (m:EventMention {mention_key: $marker})
                    FOREACH (_ IN CASE WHEN $legacy THEN [] ELSE [1] END |
                        CREATE (p)-[:HAS_EVENT_MENTION]->(m)-[:EVIDENCE_FOR]->(target))
                    CREATE (p)-[:DESCRIBES]->(target)
                    CREATE (p)-[:DESCRIBES]->(other)
                ''', keys=keys, target=target, other=other, question=question,
                       marker=marker, legacy=question == 'ABC 123').consume()
                query = LOCATE_EVENT_QUERY.replace(
                    'WHERE size($candidate_terms)',
                    'WHERE event.event_key IN $test_keys AND size($candidate_terms)', 1)
                rows = tx.run(query, **candidate_search_parameters(question),
                              limit=1, test_keys=keys).data()
                assert [row['event_key'] for row in rows] == [keys[0]]
                assert rows[0]['match_coverage'] >= 0.5
                assert rows[0]['matched_terms']
                all_rows = tx.run(query, **candidate_search_parameters(question),
                                  limit=10, test_keys=keys).data()
                assert [row['event_key'] for row in all_rows] == (
                    keys[:2] if question == 'cháy nhà máy' else [keys[0]])
                unrelated = tx.run(query, **candidate_search_parameters('thiên thạch vũ trụ'),
                                   limit=10, test_keys=keys).data()
                assert unrelated == []
                tx.rollback()
