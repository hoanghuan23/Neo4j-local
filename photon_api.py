"""Photon geocoding for LOCATION hierarchy enrichment and preview."""

import json

import requests

from knowledge_relations.location_hierarchy import (
    ADMIN_FIELDS,
    clean_osm_admin_name,
    match_score,
    normalize_vi,
)

PHOTON_URL = "https://photon.komoot.io/api/"


def search_photon(query, *, limit=1, request_get=None):
    response = (request_get or requests.get)(
        PHOTON_URL,
        params={"q": query, "limit": limit},
        headers={"User-Agent": "location-hierarchy-test/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        raise ValueError("Photon payload không hợp lệ")
    return data["features"][:limit]


def _candidate(feature):
    properties = feature.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Photon feature thiếu properties")
    address = {
        field: properties[field]
        for field in (*ADMIN_FIELDS, "village", "hamlet", "street", "postcode", "housenumber")
        if properties.get(field)
    }
    if properties.get("countrycode"):
        address["country_code"] = properties["countrycode"].lower()
    return {
        "name": properties.get("name"),
        "osm_id": properties.get("osm_id"),
        "osm_type": {"N": "node", "W": "way", "R": "relation"}.get(
            properties.get("osm_type"), properties.get("osm_type")
        ),
        "category": properties.get("osm_key"),
        "type": properties.get("osm_value"),
        "address": address,
        "namedetails": {"name": properties["name"]} if properties.get("name") else {},
    }


def geocode_with_hints(query, hints=None, request_get=None):
    """Return at most one matching Photon candidate for hierarchy enrichment."""
    if not query.strip():
        return []

    def select(rows):
        scored = [(match_score(query, row), row) for row in rows]
        scored = [(score, row) for score, row in scored if score > 0]
        if not scored:
            return []
        best = max(score for score, _ in scored)
        winners = [row for score, row in scored if score == best]
        return winners if len(winners) == 1 else [row for _, row in scored]

    def key(value):
        return normalize_vi(clean_osm_admin_name(value), strip_accents=True)

    seen = {key(query)}
    hint_keys = []
    for hint in hints or []:
        hint_key = key(hint)
        if not hint_key or hint_key in seen:
            continue
        seen.add(hint_key)
        hint_keys.append(hint_key)
        rows = [_candidate(feature) for feature in search_photon(
            f"{query}, {hint}", request_get=request_get
        )]
        selected = select([row for row in rows if any(
            isinstance(value, str) and key(value) == hint_key
            for field, value in row["address"].items() if field in ADMIN_FIELDS
        )])
        if len(selected) == 1:
            return selected

    rows = [_candidate(feature) for feature in search_photon(query, request_get=request_get)]
    for hint_key in hint_keys:
        selected = select([row for row in rows if any(
            isinstance(value, str) and key(value) == hint_key
            for field, value in row["address"].items() if field in ADMIN_FIELDS
        )])
        if len(selected) == 1:
            return selected
    return select(rows)


if __name__ == "__main__":
    print(json.dumps(search_photon("thái nguyên", limit=1),
                     ensure_ascii=False, indent=2))
