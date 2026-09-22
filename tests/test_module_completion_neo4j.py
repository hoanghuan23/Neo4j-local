"""Transactional integration checks; all fixture writes are rolled back."""
import os
from uuid import uuid4

import pytest
from neo4j import GraphDatabase

from knowledge_persistence import mark_module_completed, complete_consolidated_modules, save_knowledge_tx
from knowledge_settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

pytestmark = pytest.mark.skipif(os.getenv('RUN_NEO4J_INTEGRATION') != '1',
                                reason='requires local Neo4j')


def test_completion_preserves_history_without_pending():
    marker = uuid4().hex
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session(database='neo4j') as session:
            with session.begin_transaction() as tx:
                tx.run('''CREATE (p:Post {platform:'test-modules', platform_id:$id,
                    modules_completed:['PARTICIPANT_ROLE']})
                    CREATE (p)-[:HAS_EVENT_MENTION]->(:EventMention {
                        mention_key:$first, consolidation_status:'RESOLVED'})
                    CREATE (p)-[:HAS_EVENT_MENTION]->(:EventMention {
                        mention_key:$second, consolidation_status:'ERROR'})''',
                    id=marker, first=marker+'1', second=marker+'2').consume()
                def properties():
                    return dict(tx.run('MATCH (p:Post {platform_id:$id}) RETURN properties(p) AS p',
                                       id=marker).single()['p'])
                mark_module_completed(tx, 'test-modules', marker, 'PARTICIPANT_ROLE')
                mark_module_completed(tx, 'test-modules', marker, 'EVENT_RELATION')
                mark_module_completed(tx, 'test-modules', marker, 'EVENT_RELATION')
                assert properties()['modules_completed'] == ['PARTICIPANT_ROLE', 'EVENT_RELATION']
                complete_consolidated_modules(tx, [marker+'1'])
                assert 'EVENT_HIERARCHY' not in properties()['modules_completed']
                tx.run("MATCH (m:EventMention {mention_key:$key}) SET m.consolidation_status='RESOLVED'",
                       key=marker+'2').consume()
                complete_consolidated_modules(tx, [marker+'2'])
                complete_consolidated_modules(tx, [marker+'1'])
                assert properties()['modules_completed'] == ['PARTICIPANT_ROLE', 'EVENT_RELATION', 'EVENT_HIERARCHY']
                empty = {'entities': [], 'events': [], 'event_relations': []}
                save_knowledge_tx(tx, 'test-modules', marker, empty,
                                  classifier_decision='SKIPPED', runnable_modules=set())
                assert properties()['modules_completed'] == ['PARTICIPANT_ROLE', 'EVENT_RELATION', 'EVENT_HIERARCHY']
                tx.run('MATCH (p:Post {platform_id:$id}) REMOVE p.modules_completed', id=marker).consume()
                save_knowledge_tx(tx, 'test-modules', marker, empty, runnable_modules=set())
                assert properties()['modules_completed'] == []
                mark_module_completed(tx, 'test-modules', marker, 'ENTITY_HIERARCHY')
                assert properties()['modules_completed'] == ['ENTITY_HIERARCHY']
                tx.rollback()
