"""Shared direct/related predicates for current and legacy event schemas.

All evidence is selected in Neo4j, before repository deduplication and paging.
The same evidence expressions drive both eligibility and related explanations.
"""


def _fold(value: str) -> str:
    # Neo4j 5.26 supports Unicode normalization. Fold accents on aliases too,
    # including graphs written before search_name was populated.
    return (
        "replace(reduce(folded = toLower(normalize(trim(coalesce(" + value + ", '')), NFD)), "
        "character IN $fold_characters | replace(folded, character, '')), 'đ', 'd')"
    )


def _names(node: str) -> str:
    values = (
        f"[{{field: 'entity.name', value: {node}.name}}, "
        f"{{field: 'entity.normalized_name', value: {node}.normalized_name}}, "
        f"{{field: 'entity.search_name', value: {node}.search_name}}] + "
        f"[alias IN coalesce({node}.aliases, []) | "
        "{field: 'entity.aliases', value: alias}]"
    )
    # Consolidation can leave name/search_name as arrays. Treat every value as
    # a candidate name without stringifying an array into a false exact match.
    values = (
        f"reduce(entries = [], property IN ({values}) | entries + "
        "[value IN [] + coalesce(property.value, []) WHERE value IS :: STRING | "
        "{field: property.field, value: value}])"
    )
    # Match whitespace-normalized names without changing stored display values.
    folded = _fold("entry.value")
    for whitespace in ("\\t", "\\n", "\\r", "\\u00a0"):
        folded = f"replace({folded}, '{whitespace}', ' ')"
    folded = (
        f"reduce(name = '', word IN [word IN split({folded}, ' ') WHERE word <> ''] | "
        "name + CASE WHEN name = '' THEN '' ELSE ' ' END + word)"
    )
    return (
        f"[entry IN ({values}) WHERE entry.value IS NOT NULL AND trim(entry.value) <> '' | "
        f"{{field: entry.field, value: {folded}}}]"
    )


def _entity(node: str) -> str:
    return (
        f"{{id: coalesce({node}.entity_id, elementId({node})), "
        f"name: head([] + coalesce({node}.name, {node}.normalized_name, 'Không rõ')), "
        f"type: coalesce({node}.type, 'UNKNOWN')}}"
    )


def build_event_query(*, legacy: bool, related: bool) -> str:
    binding = (
        "MATCH (post:Post)-[:DESCRIBES]->(event:Event)"
        if legacy else
        "MATCH (post:Post)-[:HAS_EVENT_MENTION]->(mention:EventMention)"
        "-[:EVIDENCE_FOR]->(event:Event)"
    )
    mention = "event" if legacy else "mention"
    context = "post, event" if legacy else "post, mention, event"
    participants = "collect(DISTINCT event_entity)"
    mention_match = ""
    if not legacy:
        mention_match = "OPTIONAL MATCH (mention)-[:HAS_PARTICIPANT]->(mention_entity:Entity)"
        participants += " + collect(DISTINCT mention_entity)"
    texts = "[{field: 'event.description', value: event.description}]"
    if not legacy:
        texts += " + [{field: 'mention.description', value: mention.description}]"
    texts += (
        " + CASE WHEN sibling_event_count = 1 THEN "
        "[{field: 'post.content', value: post.content}] ELSE [] END"
    )
    exact = "any(name IN candidate.names WHERE name.value = term.search_key)"
    parent_match = (
        "(parent.relationship <> 'SUBORDINATE_TO' OR term.field = 'entity') AND "
        "any(name IN parent.names WHERE name.value = term.search_key)"
    )
    direct = (
        "any(candidate IN candidates WHERE "
        "(term.field <> 'location' OR candidate.entity.type = 'LOCATION') "
        f"AND {exact})"
    )
    # Reasons contain only graph/text evidence. Labels and bounded excerpts are
    # assembled in Python, so presentation does not complicate the predicates.
    reasons = f"""
        reduce(reasons = [], candidate IN candidates | reasons +
          [name IN candidate.names WHERE name.value CONTAINS term.search_key
             AND NOT ({exact}) | {{
            kind: 'entity_name_match', query_field: term.field, query_term: term.key,
            via_entity: candidate.entity, evidence_field: name.field
          }}] +
          [parent IN candidate.parents WHERE
             {parent_match} | {{
            kind: CASE WHEN parent.relationship = 'SUBORDINATE_TO'
                       THEN 'organization_hierarchy' ELSE 'location_hierarchy' END,
            query_field: term.field, query_term: term.key,
            via_entity: candidate.entity, evidence_field: 'entity.name',
            relationship: parent.relationship
          }}]) +
        [evidence IN texts WHERE {_fold('evidence.value')} CONTAINS term.search_key | {{
          kind: 'text_match', query_field: term.field, query_term: term.key,
          evidence_field: evidence.field, text: evidence.value
        }}]
    """ if related else "[]"
    event_match = (
        "any(candidate IN candidates WHERE candidate.at_event AND "
        "any(name IN candidate.names WHERE name.value = term.search_key))"
    )
    if related:
        event_match = (
            "any(candidate IN candidates WHERE candidate.at_event AND ("
            "any(name IN candidate.names WHERE name.value CONTAINS term.search_key) OR "
            "any(parent IN candidate.parents WHERE "
            f"{parent_match}))) OR "
            "any(evidence IN texts WHERE evidence.field <> 'post.content' AND "
            + _fold('evidence.value') + " CONTAINS term.search_key)"
        )
    eligible = "matched.direct OR size(matched.reasons) > 0" if related else "matched.direct"
    return f"""
{binding}
WHERE post.posted_at IS NOT NULL
  AND (($posted_date IS NULL
        AND post.posted_at >= localdatetime() - duration({{hours: $hours}}))
    OR ($posted_date IS NOT NULL
        AND date(post.posted_at + duration({{
          hours: $posted_at_utc_offset_hours
        }})) = date($posted_date)))
{mention_match}
OPTIONAL MATCH (event)-[:HAS_PARTICIPANT]->(event_entity:Entity)
OPTIONAL MATCH (post)-[:MENTIONS]->(post_entity:Entity)
WITH {context}, {participants} AS event_entities,
     collect(DISTINCT post_entity) AS post_entities
CALL {{
  WITH post
  OPTIONAL MATCH (post)-[:HAS_EVENT_MENTION]->(sibling:EventMention)
  OPTIONAL MATCH (sibling)-[:EVIDENCE_FOR]->(current_event:Event)
  WITH post, collect(DISTINCT current_event) AS current_events,
       collect(DISTINCT CASE WHEN current_event IS NULL THEN sibling END) AS unlinked_mentions
  OPTIONAL MATCH (post)-[:DESCRIBES]->(legacy_event:Event)
  WITH current_events, unlinked_mentions, collect(DISTINCT legacy_event) AS legacy_events
  RETURN size(current_events) + size([e IN legacy_events WHERE NOT e IN current_events])
       + size(unlinked_mentions) AS sibling_event_count
}}
CALL {{
  WITH event_entities, post_entities, sibling_event_count
  UNWIND event_entities + CASE WHEN sibling_event_count = 1
                              THEN post_entities ELSE [] END AS candidate
  WITH DISTINCT candidate, event_entities
  OPTIONAL MATCH (candidate)-[hierarchy:PART_OF|IN_REGION|SUBORDINATE_TO]->(parent:Entity)
  WHERE candidate <> parent AND (
    (type(hierarchy) IN ['PART_OF', 'IN_REGION']
      AND candidate.type = 'LOCATION' AND parent.type = 'LOCATION') OR
    (type(hierarchy) = 'SUBORDINATE_TO'
      AND candidate.type = 'ORGANIZATION' AND parent.type = 'ORGANIZATION')
  )
  WITH candidate, event_entities, collect(CASE WHEN parent IS NULL THEN NULL ELSE {{
       names: {_names('parent')}, relationship: type(hierarchy)
     }} END) AS parents
  RETURN collect({{entity: {_entity('candidate')}, names: {_names('candidate')},
                  at_event: candidate IN event_entities, parents: parents}}) AS candidates
}}
WITH {context}, candidates, {texts} AS texts
WITH {context}, [term IN $terms | {{
       field: term.field, direct: {direct},
       event_match: {event_match},
       reasons: {reasons}
     }}] AS matches
WHERE all(matched IN [m IN matches WHERE m.field = 'location'] WHERE {eligible})
  AND (none(m IN matches WHERE m.field = 'entity') OR
       any(matched IN [m IN matches WHERE m.field = 'entity'] WHERE {eligible}))
  {"AND any(m IN matches WHERE size(m.reasons) > 0)" if related else ""}
WITH {context}, reduce(reasons = [], matched IN matches | reasons + matched.reasons) AS relation_reasons,
     CASE WHEN any(m IN matches WHERE m.field = 'entity' AND m.event_match)
          THEN size([m IN matches WHERE m.field = 'entity' AND m.event_match])
          WHEN any(matched IN matches WHERE matched.field = 'entity' AND ({eligible})) THEN 1
          ELSE 0 END AS matched_entity_count
OPTIONAL MATCH (source:Source)-[:PUBLISHED]->(post)
OPTIONAL MATCH ({mention})-[participation:HAS_PARTICIPANT]->(entity:Entity)
WITH {context}, source, matched_entity_count, relation_reasons,
     collect(DISTINCT CASE WHEN entity IS NULL THEN NULL ELSE {{
       name: head([] + coalesce(entity.name, entity.normalized_name)),
       type: entity.type, role: participation.role
     }} END) AS entities
ORDER BY matched_entity_count DESC, post.posted_at DESC
RETURN event.event_key AS event_key,
       coalesce(event.type, {mention}.type, 'OTHER') AS type,
       coalesce(event.title, {mention}.title, event.description,
                {mention}.description, post.content) AS title,
       coalesce(event.description, {mention}.description, post.content) AS description,
       coalesce({mention}.status, event.status) AS status,
       {mention}.time_expression AS time_expression,
       matched_entity_count, entities, relation_reasons,
       {{platform: post.platform, platform_id: post.platform_id, content: post.content,
         url: post.url, posted_at: toString(post.posted_at), source_name: source.name}} AS post
"""


SEARCH_EVENTS_QUERY = build_event_query(legacy=False, related=False)
SEARCH_LEGACY_EVENTS_QUERY = build_event_query(legacy=True, related=False)
SEARCH_RELATED_EVENTS_QUERY = build_event_query(legacy=False, related=True)
SEARCH_RELATED_LEGACY_EVENTS_QUERY = build_event_query(legacy=True, related=True)
