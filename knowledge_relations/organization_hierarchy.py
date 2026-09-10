"""Conservative organization identity resolution and evidence-based enrichment."""
import json
import re

from knowledge_extraction import normalize_name, make_search_name, call_ollama
from knowledge_relations.location_hierarchy import _search_variants

VERSION = 'organization-hierarchy-v1'
RELATIONS = ('SUBORDINATE_TO', 'JURISDICTION')
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['relations'],
          'properties': {'relations': {'type': 'array', 'items': {
              'type': 'object', 'additionalProperties': False,
              'required': ['source_entity_id', 'target_entity_id', 'relationship', 'evidence_text'],
              'properties': {**{k: {'type': 'string'} for k in (
                  'source_entity_id', 'target_entity_id', 'evidence_text')},
                  'relationship': {'type': 'string', 'enum': list(RELATIONS)}}}}}}


def identity(name):
    value = normalize_name(name or '')
    value = re.sub(r'\btp\s*\.?\s*(?:hcm|hồ\s+chí\s+minh)\b', 'thành phố hồ chí minh', value)
    value = re.sub(r'\btp\s*\.?\s*', 'thành phố ', value)
    return ' '.join(re.sub(r'[,.;:()–—-]', ' ', value).split())


def names(node):
    return [v for v in [node.get('name'), node.get('normalized_name'), *(node.get('aliases') or [])] if isinstance(v, str) and v]


def load_graph(tx):
    return [dict(r) for r in tx.run('''
        MATCH (n:Entity) WHERE n.type IN ['ORGANIZATION', 'LOCATION']
        OPTIONAL MATCH (n)-[r:SUBORDINATE_TO|JURISDICTION|PART_OF|IN_REGION]->(parent:Entity)
        RETURN elementId(n) AS node_id, n.name AS name, n.normalized_name AS normalized_name,
               n.type AS type, n.aliases AS aliases, n.level AS level,
               n.search_name AS search_name,
               collect({relationship:type(r), node_id:elementId(parent), name:parent.name}) AS parents
    ''')]


def resolve_location(name, graph):
    keys = set(_search_variants(name))
    keys.add(make_search_name(re.sub(r"^(?:tỉnh|thành phố)\s+", "", name, flags=re.I)))
    candidates = [n for n in graph if n['type'] == 'LOCATION' and
                  (keys.intersection(make_search_name(v) for v in names(n)) or n.get('search_name') in keys)]
    admin = [n for n in candidates if n.get('level') is not None]
    return admin[0] if len(admin) == 1 else candidates[0] if len(candidates) == 1 else None


def local_area(name):
    value = identity(name)
    match = re.search(r'\b(?:thành phố|tỉnh|xã|phường)\s+(.+)$', value)
    if match:
        return match.group(0)
    match = re.fullmatch(r'công an (.+)', value)
    return match.group(1) if match else None


def choose(entity, graph, hints=()):
    keys = {identity(v) for v in (entity.get('name'), entity.get('canonical_name')) if v}
    candidates = [n for n in graph if n['type'] == 'ORGANIZATION' and keys.intersection(identity(v) for v in names(n))]
    original = candidates[:]
    for relation, target in hints:
        candidates = [n for n in candidates if not any(
            p['relationship'] == relation and p['node_id'] != target for p in n['parents'])]
        matching = [n for n in candidates if any(p['relationship'] == relation and p['node_id'] == target for p in n['parents'])]
        if matching:
            candidates = matching
    if len(candidates) == 1:
        return candidates[0], 'REUSE'
    if original:
        return None, 'AMBIGUOUS_OR_CONFLICT'
    name = identity(entity.get('canonical_name') or entity.get('name'))
    if name in {'công an', 'cơ quan cảnh sát điều tra', 'cảnh sát', 'quân đội', 'bộ chỉ huy quân sự'}:
        return None, 'INSUFFICIENT_NAME'
    return None, 'CREATE'


def extract_context(content, knowledge, call_model=None):
    entities = knowledge.get('entities', [])
    by_id = {e['local_id']: e for e in entities if e.get('type') in ('ORGANIZATION', 'LOCATION')}
    if len(by_id) < 2:
        return []
    prompt = '''Chỉ trích quan hệ tổ chức được nêu rõ trong content. SUBORDINATE_TO:
đơn vị thuộc tổ chức; JURISDICTION: tổ chức có phạm vi quản lý địa bàn.
Không coi trụ sở, nơi xảy ra sự kiện, nơi công tác là địa bàn quản lý.
Không dùng kiến thức nền. Chỉ dùng ID trong entities. evidence_text là đoạn
nguyên văn chứa tên cả hai thực thể. Không đủ bằng chứng trả relations rỗng.
'''+json.dumps({'entities': list(by_id.values()), 'content': content}, ensure_ascii=False)
    raw = (call_model or call_ollama)(prompt, SCHEMA)
    result = []
    for edge in raw.get('relations', []) if isinstance(raw, dict) else []:
        if not isinstance(edge, dict):
            continue
        source, target = by_id.get(edge.get('source_entity_id')), by_id.get(edge.get('target_entity_id'))
        relation, evidence = edge.get('relationship'), edge.get('evidence_text')
        if not source or not target or source == target or relation not in RELATIONS:
            continue
        if source['type'] != 'ORGANIZATION' or target['type'] != ('LOCATION' if relation == 'JURISDICTION' else 'ORGANIZATION'):
            continue
        if not isinstance(evidence, str) or not evidence.strip() or evidence not in content:
            continue
        if not all(any(normalize_name(v) in normalize_name(evidence) for v in
                       (e.get('name'), e.get('canonical_name')) if v) for e in (source, target)):
            continue
        result.append({**edge, 'source': 'CONTENT', 'rule_id': None})
    return result


def hints_for(entity, graph, entities=(), context=()):
    hints = []
    area = local_area(entity.get('canonical_name') or entity.get('name'))
    location = resolve_location(area, graph) if area else None
    if location:
        hints.append(('JURISDICTION', location['node_id']))
    by_id = {e['local_id']: e for e in entities}
    for edge in context:
        if edge['source_entity_id'] != entity.get('local_id'):
            continue
        target = by_id.get(edge['target_entity_id'])
        if not target:
            continue
        node = resolve_location(target.get('canonical_name') or target['name'], graph) if target['type'] == 'LOCATION' else choose(target, graph)[0]
        if node:
            hints.append((edge['relationship'], node['node_id']))
    return hints


def resolve_entity(tx, platform, post_id, entity, prepared, entities=(), context=()):
    graph = load_graph(tx)
    node, decision = choose(entity, graph, hints_for(entity, graph, entities, context))
    if decision not in ('REUSE', 'CREATE'):
        tx.run('''MATCH (p:Post {platform:$platform, platform_id:$post_id})
            SET p.organization_resolution_reviews = coalesce(p.organization_resolution_reviews, []) + $review''',
            platform=platform, post_id=post_id, review=json.dumps({'entity': entity, 'reason': decision}, ensure_ascii=False)).consume()
        return None
    row = tx.run('''MATCH (p:Post {platform:$platform, platform_id:$post_id})
        CALL (p) {
            WITH p WHERE $node_id IS NOT NULL
            MATCH (e:Entity) WHERE elementId(e)=$node_id RETURN e
            UNION
            WITH p WHERE $node_id IS NULL
            CREATE (e:Entity {type:'ORGANIZATION', name:$display_name,
                normalized_name:$normalized_name, search_name:$search_name,
                resolution_confidence:$confidence, needs_review:$needs_review}) RETURN e
        }
        SET e.aliases = reduce(acc=coalesce(e.aliases, []), a IN $identity_names |
            CASE WHEN a IN acc THEN acc ELSE acc+a END)
        MERGE (p)-[:MENTIONS]->(e)
        RETURN elementId(e) AS node_id, e.normalized_name AS normalized_name
        ''', platform=platform, post_id=post_id, node_id=node['node_id'] if node else None,
        needs_review=prepared['confidence'] != 'HIGH', **prepared).single()
    return {**prepared, **dict(row)} if row else None


def is_vietnam(node, graph):
    by_id = {n['node_id']: n for n in graph}
    pending, seen = [node], set()
    while pending:
        current = pending.pop()
        if current['node_id'] in seen:
            continue
        seen.add(current['node_id'])
        if current['type'] == 'LOCATION' and any(make_search_name(v) in ('viet nam', 'vietnam') for v in names(current)):
            return True
        pending.extend(by_id[p['node_id']] for p in current['parents'] if
                       p['relationship'] in ('PART_OF', 'IN_REGION', 'JURISDICTION') and p['node_id'] in by_id)
    return False


def propose(graph, knowledge, context):
    edges, reviews, decisions = [], [], []
    entities = knowledge.get('entities', [])
    resolved = {}
    for entity in entities:
        if entity['type'] != 'ORGANIZATION':
            continue
        node, decision = choose(entity, graph, hints_for(entity, graph, entities, context))
        decisions.append({'entity': entity, 'decision': decision, 'node_id': node['node_id'] if node else None})
        if node:
            resolved[entity['local_id']] = node
        else:
            reviews.append({'entity': entity['local_id'], 'reason': decision})
    def add(node, target, relationship, source, evidence=None, rule=None):
        edges.append({'source_node_id': node['node_id'], 'target_node_id': target['node_id'],
                      'relationship': relationship, 'source': source, 'evidence_text': evidence, 'rule_id': rule})
    for edge in context:
        node = resolved.get(edge['source_entity_id'])
        target_entity = next((e for e in entities if e['local_id'] == edge['target_entity_id']), None)
        target = resolved.get(edge['target_entity_id']) if edge['relationship'] == 'SUBORDINATE_TO' else (
            resolve_location(target_entity.get('canonical_name') or target_entity['name'], graph) if target_entity else None)
        if node and target:
            add(node, target, edge['relationship'], 'CONTENT', edge['evidence_text'])
        else:
            reviews.append({'reason': 'UNRESOLVED_CONTENT_TARGET', 'edge': edge})
    for entity in entities:
        node = resolved.get(entity['local_id'])
        if not node:
            continue
        name = identity(entity.get('canonical_name') or entity['name'])
        area = local_area(name)
        location = resolve_location(area, graph) if area else None
        # Only public security / military names have a name-based jurisdiction rule.
        police = bool(re.search(r'\bcông an\b', name))
        military = bool(re.match(r'bộ (?:chỉ huy quân sự|tư lệnh (?:vùng \d+ hải quân|bộ đội biên phòng))\b', name))
        if not (police or military):
            continue
        if location:
            add(node, location, 'JURISDICTION', 'RULE', rule='public-unit-area-v1')
        elif area:
            reviews.append({'entity': entity['local_id'], 'reason': 'UNRESOLVED_AREA'})
        target_name = 'Bộ Công an' if police else 'Bộ Quốc phòng'
        target = choose({'name': target_name}, graph)[0]
        vietnam = is_vietnam(location, graph) if location else is_vietnam(node, graph)
        if not vietnam:
            context_areas = {e['target_node_id'] for e in edges if
                             e['source_node_id'] == node['node_id'] and e['relationship'] == 'JURISDICTION'}
            vietnam = any(is_vietnam(n, graph) for n in graph if n['node_id'] in context_areas)
        if target and target['node_id'] == node['node_id']:
            continue
        if target and vietnam:
            add(node, target, 'SUBORDINATE_TO', 'RULE', rule='vn-police-v1' if police else 'vn-military-v1')
        elif not any(p['relationship'] == 'SUBORDINATE_TO' for p in node['parents']):
            reviews.append({'entity': entity['local_id'], 'reason': 'UNRESOLVED_COUNTRY_OR_PARENT'})
    accepted = []
    by_id = {n['node_id']: n for n in graph}
    for edge in edges:
        source, target, relation = edge['source_node_id'], edge['target_node_id'], edge['relationship']
        conflicting = any(e['source_node_id'] == source and e['relationship'] == relation
                          and e['target_node_id'] != target for e in edges)
        conflicting |= any(p['relationship'] == relation and p['node_id'] != target
                           for p in by_id[source]['parents'])
        pending, seen = [target], set()
        while pending and relation == 'SUBORDINATE_TO':
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(p['node_id'] for p in by_id.get(current, {}).get('parents', [])
                           if p['relationship'] == relation)
            pending.extend(e['target_node_id'] for e in edges
                           if e['source_node_id'] == current and e['relationship'] == relation)
        if conflicting or source == target or source in seen:
            reviews.append({'reason': 'CONFLICT_CYCLE_OR_INVALID_ENDPOINT', 'edge': edge})
        elif edge not in accepted:
            accepted.append(edge)
    return {'decisions': decisions, 'edges': accepted, 'reviews': reviews}


def persist_edges(tx, edges, platform, post_id):
    reviews = []
    for edge in edges:
        relation = edge['relationship']
        if relation not in RELATIONS:
            raise ValueError('Unsupported organization relation')
        row = tx.run('''MATCH (a:Entity), (b:Entity)
            WHERE elementId(a)=$source_node_id AND elementId(b)=$target_node_id
              AND a.type='ORGANIZATION' AND b.type=$target_type AND a<>b
              AND NOT EXISTS { MATCH (a)-[r:'''+relation+''']->(other) WHERE other<>b }
              '''+('AND NOT EXISTS { MATCH (b)-[:SUBORDINATE_TO*1..]->(a) }' if relation == 'SUBORDINATE_TO' else '')+'''
            MERGE (a)-[r:'''+relation+''']->(b)
            ON CREATE SET r.source=$source, r.evidence_text=$evidence_text, r.rule_id=$rule_id,
                r.platform=$platform, r.post_id=$post_id, r.module_version=$version,
                r.created_at=datetime(), r.updated_at=datetime()
            RETURN elementId(r) AS id''', **edge, target_type='LOCATION' if relation == 'JURISDICTION' else 'ORGANIZATION',
            platform=platform, post_id=post_id, version=VERSION).single()
        if not row:
            reviews.append({'reason': 'CONFLICT_CYCLE_OR_INVALID_ENDPOINT', 'edge': edge})
    return reviews


def enrich_organization_hierarchy(session, platform, post_id, content, knowledge, *, call_model=None, preview=False):
    try:
        context = knowledge.get('organization_context')
        if context is None:
            context = extract_context(content, knowledge, call_model)
        result = propose(load_graph(session), knowledge, context)
        if preview:
            return result
        result['reviews'].extend(session.execute_write(persist_edges, result['edges'], platform, post_id))
        session.run('''MATCH (p:Post {platform:$platform, platform_id:$post_id})
            SET p.organization_hierarchy_status=CASE WHEN size($reviews)>0 OR
                size(coalesce(p.organization_resolution_reviews, []))>0 THEN 'NEEDS_REVIEW' ELSE 'DONE' END,
                p.organization_hierarchy_reviews=$reviews, p.organization_hierarchy_error=null,
                p.organization_hierarchy_version=$version, p.organization_hierarchy_processed_at=datetime()
            ''', platform=platform, post_id=post_id, reviews=[json.dumps(r, ensure_ascii=False) for r in result['reviews']], version=VERSION).consume()
        return result
    except Exception as exc:
        if preview:
            raise
        session.run('''MATCH (p:Post {platform:$platform, platform_id:$post_id})
            SET p.organization_hierarchy_status='FAILED', p.organization_hierarchy_error=$error,
                p.organization_hierarchy_version=$version''', platform=platform, post_id=post_id,
            error=str(exc), version=VERSION).consume()
        return {'errors': 1}
