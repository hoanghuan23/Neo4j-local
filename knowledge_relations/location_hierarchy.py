"""Evidence-first LOCATION hierarchy enrichment."""

import json
import re
import unicodedata

from langsmith import traceable

from knowledge_extraction import call_ollama, location_identity_names, make_search_name, normalize_name
from knowledge_settings import (
    LOCATION_HIERARCHY_MODULE_VERSION,
    LOCATION_HIERARCHY_SCHEMA,
    LOGGER,
)

ADMIN_FIELDS = (
    "neighbourhood", "quarter", "suburb", "borough", "city_district",
    "district", "municipality", "city", "town", "county",
    "state_district", "state", "region", "country",
)
OSM_MATCH_FIELDS = (*ADMIN_FIELDS, "village", "hamlet")
OSM_INTERMEDIATE_PARENT_FIELDS = ("state", "suburb")
LOCATION_PREFIXES = ("tp ", "tp. ", "thanh pho ", "city of ")
OSM_ADMIN_SUFFIX_PATTERN = re.compile(
    r"\s+(?:province|country)\s*$",
    flags=re.IGNORECASE,
)


def clean_osm_admin_name(value: str) -> str:
    """Remove generic English administrative suffixes from OSM labels."""
    cleaned = " ".join(value.split())
    return OSM_ADMIN_SUFFIX_PATTERN.sub("", cleaned).strip()


def _location_inputs(knowledge: dict, resolved: list[dict]) -> list[dict]:
    by_name = {}
    for location in resolved:
        for name in (location.get("normalized_name"), *(location.get("aliases") or [])):
            if isinstance(name, str) and name:
                by_name[normalize_name(name)] = location
    locations, seen = [], set()
    for entity in knowledge.get("entities", []):
        if not isinstance(entity, dict) or entity.get("type") != "LOCATION":
            continue
        location = next((by_name.get(normalize_name(value)) for value in (
            entity.get("canonical_name"), entity.get("name")
        ) if isinstance(value, str) and value.strip() and by_name.get(normalize_name(value))), None)
        if location is None or location["node_id"] in seen:
            continue
        seen.add(location["node_id"])
        locations.append({"entity_id": entity.get("local_id"), **location})
    return locations


def normalize_content_edges(content: str, locations: list[dict], raw: object) -> list[dict]:
    if not isinstance(raw, dict) or not isinstance(raw.get("relations"), list):
        return []
    by_id = {item["entity_id"]: item for item in locations if isinstance(item.get("entity_id"), str)}
    edges, seen = [], set()
    for item in raw["relations"]:
        if not isinstance(item, dict):
            continue
        child, parent = by_id.get(item.get("source_entity_id")), by_id.get(item.get("target_entity_id"))
        evidence = item.get("evidence_text")
        evidence_search = make_search_name(evidence) if isinstance(evidence, str) else ""
        if (child is None or parent is None or child["node_id"] == parent["node_id"]
                or not isinstance(evidence, str) or not evidence.strip()
                or normalize_name(evidence) not in normalize_name(content)
                or make_search_name(child["name"]) not in evidence_search
                or make_search_name(parent["name"]) not in evidence_search):
            continue
        key = (child["node_id"], parent["node_id"])
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "source_node_id": key[0], "target_node_id": key[1], "source": "CONTENT",
            "evidence_text": " ".join(evidence.split()), "parent_level": None,
            "osm_id": None, "osm_type": None,
        })
    return edges


@traceable(name="location-hierarchy-content", run_type="chain", tags=["location-hierarchy"])
def extract_content_edges(content: str, knowledge: dict, resolved: list[dict], call_model=None) -> list[dict]:
    locations = _location_inputs(knowledge, resolved)
    if len(locations) < 2:
        return []
    compact = [{"entity_id": item["entity_id"], "name": item["name"]} for item in locations]
    prompt = f"""
Bạn là module LOCATION_HIERARCHY. Chỉ trích xuất PART_OF được thể hiện trực tiếp
trong content. source và target phải tham chiếu LOCATION trong danh sách. Không
tạo địa danh mới, không dùng kiến thức nền và không suy diễn cây địa lý.
evidence_text phải là đoạn nguyên văn trong content chứa cả hai địa danh.
Địa chỉ nêu trực tiếp các thành phần bằng dấu phẩy, ngoặc hoặc dấu gạch ngang
cũng là bằng chứng chứa địa điểm: "tại số 96 phố Cầu Đất - Hải Phòng" cho phép
"96 phố Cầu Đất" PART_OF "Hải Phòng" nếu cả hai có trong locations.
Không coi mọi dấu gạch ngang là địa chỉ: tuyến "Hà Nội - Hải Phòng", so sánh
hai nơi hay credit "Clip: Hải Phòng" không chứng minh quan hệ PART_OF.
Nếu không đủ bằng chứng, trả relations rỗng. Chỉ trả JSON đúng schema.
<locations>{json.dumps(compact, ensure_ascii=False)}</locations>
<content>{content}</content>
""".strip()
    call_model = call_model or call_ollama
    return normalize_content_edges(content, locations, call_model(prompt, LOCATION_HIERARCHY_SCHEMA))


def normalize_vi(text: str, *, strip_accents: bool = False) -> str:
    value = unicodedata.normalize("NFD", text).lower().strip()
    if strip_accents:
        value = "".join(char for char in value if not unicodedata.combining(char))
        value = value.replace("đ", "d")
    return value


def match_score(query: str, result: dict) -> float:
    exact = normalize_vi(query)
    folded = normalize_vi(query, strip_accents=True)
    if not exact or not folded:
        return 0.0
    values = [result.get("name")]
    for field in ("address", "namedetails"):
        if isinstance(result.get(field), dict):
            values.extend(result[field].values())
    score = 0.0
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        name = normalize_vi(value)
        bare = normalize_vi(value, strip_accents=True)
        if exact == name:
            return 1.0
        if folded == bare:
            score = max(score, 0.8)
        elif exact in name:
            score = max(score, 0.6)
        elif folded in bare:
            score = max(score, 0.4)
    return score


ADMIN_NAME_PREFIXES = {
    "tỉnh": "province", "thành phố": "city", "phường": "ward",
    "xã": "commune", "quận": "district", "huyện": "county",
    "thị xã": "town", "thị trấn": "township",
}


def _explicit_admin_types(value: str, name: str) -> set[str]:
    text = normalize_name(value)
    bare = normalize_name(name)
    for prefix in ADMIN_NAME_PREFIXES:
        if bare.startswith(prefix + " "):
            bare = bare[len(prefix):].strip()
            break
    return {level for prefix, level in ADMIN_NAME_PREFIXES.items()
            if re.search(r"(?<!\w)" + re.escape(prefix + " " + bare) + r"(?!\w)", text)}


def _candidate_admin_types(row: dict, query: str) -> set[str]:
    # Inspect the object's names, never a province mentioned in its address.
    details = row.get("namedetails") or {}
    names = [row.get("name"), *(value for key, value in details.items()
             if key.split(":", 1)[0] in {"name", "official_name"})]
    levels = set()
    for name in names:
        if not isinstance(name, str):
            continue
        levels.update(_explicit_admin_types(name, query))
        for level in set(ADMIN_NAME_PREFIXES.values()):
            if re.search(r"\s" + level + r"$", normalize_name(name)):
                levels.add(level)
    return levels


def resolve_content_location(location, locations, edges, hints, geocode_fn=None, *, content=""):
    """Use explicit descendants to disambiguate an administrative parent."""
    if geocode_fn is None:
        from photon_api import geocode_with_hints as geocode_fn
    query = location["name"]
    candidates = geocode_fn(query, hints=hints)
    admin_types = _explicit_admin_types(content, query)
    if not admin_types:
        admin_types = _explicit_admin_types(query, query)
    if len(admin_types) == 1:
        candidates = [row for row in candidates
                      if _candidate_admin_types(row, query) == admin_types]
    elif len(admin_types) > 1:
        # The same bare name refers to several levels in this article.
        # Keep the ambiguity instead of choosing via descendant geography.
        return candidates
    descendants = {location["node_id"]}
    while True:
        expanded = descendants | {edge["source_node_id"] for edge in edges
                                  if edge["target_node_id"] in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    descendants.discard(location["node_id"])
    if len(candidates) < 2 or not descendants:
        return candidates

    def region(row):
        address = row.get("address", {})
        value = address.get("state") or address.get("city")
        if not isinstance(value, str) or not value.strip():
            return None
        country = address.get("country_code") or address.get("country")
        if not isinstance(country, str) or not country.strip():
            return None
        return (make_search_name(country), make_search_name(clean_osm_admin_name(value)))

    evidence = set()
    for child in locations:
        if child["node_id"] not in descendants:
            continue
        rows = geocode_fn(child["name"], hints=[query])
        # A matching address alone must not identify an unrelated POI.
        rows = [row for row in rows if make_search_name(child["name"]) in _candidate_name_keys(row)]
        regions = {region(row) for row in rows}
        if len(regions) == 1 and None not in regions:
            evidence.update(regions)
    if len(evidence) != 1:
        return candidates
    query_key = make_search_name(query)
    matches = [row for row in candidates
               if region(row) in evidence
               and query_key in _candidate_name_keys(row)
               and any(isinstance(row.get("address", {}).get(field), str)
                       and make_search_name(clean_osm_admin_name(row["address"][field])) == query_key
                       for field in OSM_MATCH_FIELDS if field != "country")
               and row.get("category", row.get("class", "place")) in ("place", "boundary")]
    return matches if len(matches) == 1 else candidates


def _candidate_name_keys(result: dict) -> set[str]:
    """Names of the object itself, excluding brand, references and address names."""
    values = [result.get("name")]
    details = result.get("namedetails")
    if isinstance(details, dict):
        for key, value in details.items():
            if key.split(":", 1)[0] in {"name", "official_name", "short_name", "alt_name", "_place_name"}:
                values.append(value)
    return {make_search_name(clean_osm_admin_name(name))
            for value in values if isinstance(value, str)
            for name in value.split(";") if name.strip()}


def administrative_chain(query: str, results: object) -> dict | None:
    """Build a hierarchy from one candidate, accepting its exact multilingual names."""
    if not isinstance(results, list) or len(results) != 1:
        return None
    query_key = make_search_name(query)
    for result in results:
        address = result.get("address") if isinstance(result, dict) else None
        if not isinstance(address, dict):
            continue
        own_names = _candidate_name_keys(result)
        # Legacy address-only fixtures remain supported. With object names available,
        # an address mentioning the query does not identify the object itself.
        if own_names and query_key not in own_names:
            continue
        matched_fields = []
        for field in OSM_MATCH_FIELDS:
            value = address.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            value = clean_osm_admin_name(value)
            if value and make_search_name(value) == query_key:
                matched_fields.append(field)
        if len(matched_fields) > 1 or (not matched_fields and query_key not in own_names):
            continue

        chain = [{"name": clean_osm_admin_name(query), "level": None}]
        seen = {query_key, *own_names}
        # Photon exposes the local administrative parent as district. Do not
        # attach an administrative object to a district below its own level.
        district = address.get("district")
        if (isinstance(district, str) and district.strip()
                and not any(field in ADMIN_FIELDS and ADMIN_FIELDS.index(field) >= ADMIN_FIELDS.index("district")
                            for field in matched_fields)):
            district = clean_osm_admin_name(district)
            district_key = make_search_name(district)
            if district and district_key not in seen:
                seen.add(district_key)
                chain.append({"name": district, "level": ADMIN_FIELDS.index("district")})
        for field in ("city", *OSM_INTERMEDIATE_PARENT_FIELDS):
            value = address.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            value = clean_osm_admin_name(value)
            value_key = make_search_name(value)
            if not value or value_key in seen:
                continue
            seen.add(value_key)
            chain.append({"name": value, "level": ADMIN_FIELDS.index(field)})
            break

        field = "country"
        value = address.get(field)
        if isinstance(value, str) and value.strip():
            value = clean_osm_admin_name(value)
            value_key = make_search_name(value)
            if value and value_key not in seen:
                chain.append({"name": value, "level": ADMIN_FIELDS.index(field)})
        if len(chain) < 2:
            continue
        return {
            "chain": chain, "osm_id": str(result.get("osm_id")) if result.get("osm_id") is not None else None,
            "osm_type": result.get("osm_type"),
        }
    return None


def _search_variants(name: str) -> list[str]:
    search = make_search_name(name)
    bare = search
    for prefix in LOCATION_PREFIXES:
        if bare.startswith(prefix):
            bare = bare[len(prefix):].strip()
            break
    ward_variants = [make_search_name(value) for value in location_identity_names(name)]
    return list(dict.fromkeys((search, bare, f"tp {bare}", f"tp. {bare}", f"thanh pho {bare}", f"city of {bare}", *ward_variants)))


def load_post_locations(session, platform: str, post_id: str) -> list[dict]:
    return [dict(record) for record in session.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})-[:MENTIONS]->(location:Entity {type: 'LOCATION'})
        RETURN elementId(location) AS node_id, location.name AS name,
               location.normalized_name AS normalized_name, location.aliases AS aliases,
               location.search_name AS search_name, location.osm_id AS osm_id,
               location.osm_type AS osm_type ORDER BY location.normalized_name
        """, platform=platform, post_id=post_id)]


def _persist_edges_tx(tx, edges: list[dict]) -> dict:
    counts = {"created": 0, "skipped": 0, "persisted_edges": []}
    for edge in edges:
        record = tx.run(
            """
            MATCH (child:Entity), (parent:Entity)
            WHERE elementId(child) = $source_node_id AND elementId(parent) = $target_node_id
              AND child.type = 'LOCATION' AND parent.type = 'LOCATION' AND child <> parent
              AND child.level IS NULL
              AND NOT EXISTS { MATCH (child)-[:PART_OF|IN_REGION]->(:Entity {type: 'LOCATION'}) }
              AND NOT EXISTS { MATCH (parent)-[:PART_OF|IN_REGION*1..]->(child) }
            MERGE (child)-[relation:PART_OF]->(parent)
            ON CREATE SET relation.created_at = datetime(), relation._location_created = true
            WITH relation, coalesce(relation._location_created, false) AS created,
                 relation.source AS old_source
            SET relation.source = CASE
                    WHEN old_source = 'ADMIN_DATA' THEN 'ADMIN_DATA'
                    WHEN old_source = 'CONTENT' OR $source = 'CONTENT' THEN 'CONTENT'
                    ELSE 'PHOTON' END,
                relation.evidence_text = CASE WHEN $source = 'CONTENT' THEN $evidence_text ELSE relation.evidence_text END,
                relation.osm_id = coalesce(relation.osm_id, $osm_id),
                relation.osm_type = coalesce(relation.osm_type, $osm_type),
                relation.parent_level = coalesce(relation.parent_level, $parent_level),
                relation.module_version = $module_version, relation.updated_at = datetime()
            REMOVE relation._location_created
            RETURN created AS created
            """, **edge, module_version=LOCATION_HIERARCHY_MODULE_VERSION).single()
        if record is None:
            counts["skipped"] += 1
        else:
            counts["created"] += int(bool(record.get("created")))
            counts["persisted_edges"].append(edge)
    return counts


def _has_existing_hierarchy(tx, node_id: str) -> bool:
    record = tx.run(
        "MATCH (child:Entity {type: 'LOCATION'}) WHERE elementId(child) = $node_id "
        "RETURN child.level IS NOT NULL OR EXISTS { "
        "MATCH (child)-[:PART_OF|IN_REGION]->(:Entity {type: 'LOCATION'}) "
        "} AS has_parent", node_id=node_id,
    ).single()
    return bool(record and record.get("has_parent"))


def _upsert_osm_chain_tx(tx, child_node_id: str, hierarchy: dict) -> dict:
    current_id = child_node_id
    counts = {"parents_created": 0, "parents_reused": 0, "osm_edges": 0, "skipped": 0}
    if _has_existing_hierarchy(tx, child_node_id):
        return counts
    tx.run(
        "MATCH (location:Entity) WHERE elementId(location) = $node_id "
        "SET location.osm_id = coalesce(location.osm_id, $osm_id), location.osm_type = coalesce(location.osm_type, $osm_type)",
        node_id=child_node_id, osm_id=hierarchy.get("osm_id"), osm_type=hierarchy.get("osm_type")).consume()
    for parent in hierarchy["chain"][1:]:
        normalized, search_name = normalize_name(parent["name"]), make_search_name(parent["name"])
        search_names = _search_variants(parent["name"])
        records = list(tx.run(
            """
            MATCH (candidate:Entity {type: 'LOCATION'})
            WHERE candidate.normalized_name = $normalized_name
               OR $normalized_name IN coalesce(candidate.aliases, [])
               OR candidate.search_name IN $search_names
               OR toLower(trim(candidate.name)) = $normalized_name
               OR candidate.normalized_name IN $search_names
            WITH collect(candidate) AS candidates
            WITH candidates,
                 [candidate IN candidates WHERE candidate.level IS NOT NULL] AS administrative
            UNWIND CASE WHEN size(administrative) = 1
                        THEN administrative ELSE candidates END AS candidate
            RETURN elementId(candidate) AS node_id LIMIT 2
            """, normalized_name=normalized, search_names=search_names))
        if len(records) == 1:
            parent_id = records[0]["node_id"]
            counts["parents_reused"] += 1
        elif len(records) > 1:
            counts["skipped"] += 1
            break
        else:
            record = tx.run(
                """
                CREATE (parent:Entity {normalized_name: $normalized_name, type: 'LOCATION', name: $name,
                    search_name: $search_name, aliases: [$normalized_name], resolution_confidence: 'HIGH', needs_review: false})
                RETURN elementId(parent) AS node_id
                """, normalized_name=normalized, name=parent["name"], search_name=search_name).single()
            parent_id = record["node_id"]
            counts["parents_created"] += 1
        edge_counts = _persist_edges_tx(tx, [{
            "source_node_id": current_id, "target_node_id": parent_id, "source": "PHOTON",
            "evidence_text": None, "parent_level": parent["level"], "osm_id": hierarchy.get("osm_id"),
            "osm_type": hierarchy.get("osm_type"),
        }])
        counts["osm_edges"] += edge_counts["created"]
        counts["skipped"] += edge_counts["skipped"]
        # Once attached to an existing location, its ancestry belongs to the DB.
        if len(records) == 1 or edge_counts["skipped"]:
            break
        current_id = parent_id
    return counts


def _set_status(session, platform: str, post_id: str, status: str, error=None) -> None:
    session.run(
        """
        MATCH (p:Post {platform: $platform, platform_id: $post_id})
        SET p.location_hierarchy_status = $status, p.location_hierarchy_version = $version,
            p.location_hierarchy_processed_at = datetime(), p.location_hierarchy_error = $error
        """, platform=platform, post_id=post_id, status=status,
        version=LOCATION_HIERARCHY_MODULE_VERSION, error=error).consume()


def enrich_location_hierarchy(session, platform: str, post_id: str, content: str, knowledge: dict, *, call_model=None, geocode_fn=None) -> dict:
    if geocode_fn is None:
        # Import lazily because Photon reuses this module's matching helpers.
        from photon_api import geocode_with_hints as geocode_fn
    summary = {"locations": 0, "content_edges": 0, "osm_edges": 0, "parents_created": 0, "parents_reused": 0, "skipped": 0, "errors": 0}
    resolved = load_post_locations(session, platform, post_id)
    locations = _location_inputs(knowledge, resolved)
    summary["locations"] = len(locations)
    try:
        edges = extract_content_edges(content, knowledge, resolved, call_model)
    except Exception:
        LOGGER.exception("Không thể trích xuất LOCATION hierarchy từ content")
        edges = []
    persisted_edges = []
    if edges:
        counts = session.execute_write(_persist_edges_tx, edges)
        summary["content_edges"], summary["skipped"] = counts["created"], counts["skipped"]
        persisted_edges = counts["persisted_edges"]
    parents = {edge["target_node_id"] for edge in persisted_edges}
    children = {edge["source_node_id"] for edge in persisted_edges}
    hint_locations = sorted(locations, key=lambda item: (
        0 if item["node_id"] in parents else 1 if item["node_id"] in children else 2
    ))
    try:
        for location in locations:
            if _has_existing_hierarchy(session, location["node_id"]):
                continue
            hints = [item["name"] for item in hint_locations if item["node_id"] != location["node_id"]]
            candidates = resolve_content_location(location, locations, persisted_edges, hints, geocode_fn, content=content)
            if len(candidates) > 1:
                session.run(
                    "MATCH (location:Entity {type: 'LOCATION'}) WHERE elementId(location) = $node_id "
                    "SET location.needs_review = true", node_id=location["node_id"],
                ).consume()
                summary["skipped"] += 1
                continue
            hierarchy = administrative_chain(location["name"], candidates)
            if hierarchy is None:
                summary["skipped"] += 1
                continue
            result = session.execute_write(_upsert_osm_chain_tx, location["node_id"], hierarchy)
            for key in ("parents_created", "parents_reused", "osm_edges", "skipped"):
                summary[key] += result[key]
    except Exception as error:
        summary["errors"] += 1
        _set_status(session, platform, post_id, "FAILED", str(error)[:2000])
        LOGGER.warning("LOCATION hierarchy lỗi cho %s:%s: %s", platform, post_id, error)
        return summary
    _set_status(session, platform, post_id, "COMPLETED")
    return summary
