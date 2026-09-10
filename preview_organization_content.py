"""Preview stored organization decisions; optionally retry unfinished posts."""
import argparse
import json

from neo4j import GraphDatabase
from knowledge_settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from knowledge_relations.organization_hierarchy import enrich_organization_hierarchy


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
            from knowledge_relations.organization_hierarchy import extract_context
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
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session(default_access_mode='WRITE' if args.retry else 'READ') as session:
            print(json.dumps(run(session, limit=args.limit, retry=args.retry), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
