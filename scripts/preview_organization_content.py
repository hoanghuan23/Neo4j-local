"""Import content and analyzed knowledge into Neo4j, or preview stored decisions."""
import argparse
import hashlib
import json
from pathlib import Path

from neo4j import GraphDatabase
from knowledge_settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from knowledge_relations.entity_hierarchy.organization_hierarchy import enrich_organization_hierarchy


def save_content(session, content, call_model):
    """Persist a manual article using the same graph schema as the QA backend.

    Content-derived identity makes repeated imports update the same source post.
    Base knowledge and the article commit atomically; enrichment can be retried.
    """
    from knowledge_extraction import extract_knowledge
    from knowledge_validation import validate_knowledge
    from knowledge_relations.event_relation import extract_event_relations
    from knowledge_relations.entity_hierarchy.organization_hierarchy import extract_context
    from knowledge_relations.entity_hierarchy.location_hierarchy import enrich_location_hierarchy
    from knowledge_persistence import (create_knowledge_schema, save_knowledge_tx,
                                       save_module_knowledge_tx, mark_module_completed)
    from knowledge_pipeline import runnable_modules_for
    from knowledge_settings import validate_module_config

    if not content.strip():
        raise ValueError('Content không được rỗng')
    validate_module_config()
    platform = 'manual'
    post_id = hashlib.sha256(content.encode('utf-8')).hexdigest()
    knowledge = extract_knowledge(content, call_model=call_model)
    knowledge = validate_knowledge(content, knowledge, platform, post_id)
    modules = runnable_modules_for(knowledge)
    if 'EVENT_RELATION' in modules:
        knowledge = extract_event_relations(content, knowledge, call_model=call_model)
    knowledge = validate_knowledge(content, knowledge, platform, post_id)
    has_organizations = any(e['type'] == 'ORGANIZATION' for e in knowledge['entities'])
    if has_organizations and 'ENTITY_HIERARCHY' in modules:
        knowledge['organization_context'] = extract_context(content, knowledge, call_model=call_model)

    create_knowledge_schema(session)
    session.run('''CREATE CONSTRAINT manual_content_id_unique IF NOT EXISTS
        FOR (p:ManualContent) REQUIRE p.platform_id IS UNIQUE''').consume()

    def persist(tx):
        tx.run('''MERGE (p:Post:ManualContent {platform: $platform, platform_id: $post_id})
            ON CREATE SET p.created_at=datetime(), p.posted_at=localdatetime('UTC')
            SET p.posted_at=localdatetime(p.posted_at),
                p.content=$content, p.updated_at=datetime(),
                p.knowledge_analysis=$analysis''',
            platform=platform, post_id=post_id, content=content,
            analysis=json.dumps(knowledge, ensure_ascii=False)).consume()
        counts = save_knowledge_tx(tx, platform, post_id, knowledge,
                                  {'should_deep_analyze': True, 'reason_code': 'MANUAL'}, 'DEEP',
                                  runnable_modules=modules)
        for module in ('EVENT_RELATION',):
            if module in modules:
                save_module_knowledge_tx(tx, platform, post_id, knowledge, module)
        return counts

    counts = session.execute_write(persist)
    result = {'mode': 'saved_to_neo4j', 'platform': platform, 'post_id': post_id,
              'content': content, 'counts': counts, 'knowledge': knowledge}
    # Base data remains queryable if an external enrichment service fails.
    for key, enabled, enrich in (
        ('location_hierarchy', any(e['type'] == 'LOCATION' for e in knowledge['entities']), enrich_location_hierarchy),
        ('organization_hierarchy', has_organizations, enrich_organization_hierarchy),
    ):
        if enabled and 'ENTITY_HIERARCHY' in modules:
            try:
                result[key] = enrich(session, platform, post_id, content, knowledge, call_model=call_model)
            except Exception as error:
                result[key] = {'errors': 1, 'error': str(error)}
    if 'ENTITY_HIERARCHY' in modules and all(
        not result.get(key, {}).get('errors', 0)
        for key in ('location_hierarchy', 'organization_hierarchy')
    ):
        session.execute_write(mark_module_completed, platform, post_id, 'ENTITY_HIERARCHY')
    return result


def run(session, *, limit=100, retry=False):
    rows = list(session.run('''MATCH (p:Post)
        WHERE p.organization_hierarchy_input IS NOT NULL
          AND p.organization_hierarchy_status IN ['PENDING','FAILED','NEEDS_REVIEW']
        RETURN p.platform AS platform, p.platform_id AS post_id,
               p.content AS content, p.organization_hierarchy_input AS snapshot
        ORDER BY p.platform, p.platform_id LIMIT $limit''', limit=limit))
    results = []
    for row in rows:
        knowledge = json.loads(row['snapshot'])
        # Retry resolution before enrichment, so previously skipped mentions and
        # participants are restored by the same transactional persistence path.
        if retry:
            from knowledge_persistence import upsert_entities, upsert_events
            from knowledge_relations.entity_hierarchy.organization_hierarchy import extract_context
            try:
                knowledge['organization_context'] = extract_context(row['content'], knowledge)
            except Exception:
                # Enrichment records FAILED below; leave existing base data intact.
                knowledge.pop('organization_context', None)
            else:
                def resolve(tx):
                    tx.run('''MATCH (p:Post {platform:$platform, platform_id:$post_id})
                        SET p.organization_resolution_reviews=[]''', platform=row['platform'], post_id=row['post_id']).consume()
                    lookup = upsert_entities(tx, row['platform'], row['post_id'], knowledge['entities'], knowledge['organization_context'])
                    upsert_events(tx, row['platform'], row['post_id'], knowledge.get('events', []), lookup)
                session.execute_write(resolve)
        results.append({'platform': row['platform'], 'post_id': row['post_id'],
            **enrich_organization_hierarchy(session, row['platform'], row['post_id'], row['content'], knowledge, preview=not retry)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry', action='store_true', help='Write retries; default is read-only preview')
    parser.add_argument('--limit', type=int, default=100)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--content', help='Phân tích và lưu content trực tiếp vào Neo4j')
    source.add_argument('--file', type=Path, help='Phân tích và lưu file UTF-8 vào Neo4j')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    content = args.file.read_text(encoding='utf-8') if args.file is not None else args.content
    if content is not None and (not content.strip() or args.retry):
        parser.error('Content không được rỗng; không kết hợp content với --retry')
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session(default_access_mode='WRITE' if args.retry or content is not None else 'READ') as session:
            if content is None:
                result = run(session, limit=args.limit, retry=args.retry)
            else:
                from knowledge_gemini import GeminiKnowledgeCaller
                caller = GeminiKnowledgeCaller()
                try:
                    result = save_content(session, content, caller)
                finally:
                    caller.close()
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
