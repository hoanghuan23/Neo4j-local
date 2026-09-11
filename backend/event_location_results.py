"""Aggregate location mentions without assigning them to unrelated source posts."""

from backend.models import (
    EventLocationCandidate, EventLocationEvidence, EventLocationGroup,
    EventLocationSource,
)


def group_event_locations(candidates: list[EventLocationCandidate]) -> list[EventLocationGroup]:
    rows_by_event: dict[str, list[EventLocationCandidate]] = {}
    for row in candidates:
        rows_by_event.setdefault(row.event_key, []).append(row)

    groups = []
    for event_key, rows in rows_by_event.items():
        sources = {}
        locations = {}
        for row in rows:
            if row.post_id:
                key = ("id", row.post_platform, row.post_id)
            elif row.post_url:
                key = ("url", row.post_url)
            else:
                key = ("content", row.post_platform, row.post_content)
            has_source = any((row.post_id, row.post_url, row.post_content))
            if has_source and key not in sources:
                sources[key] = EventLocationSource(
                    source_id=f"source-{len(sources) + 1}",
                    post_platform=row.post_platform, post_id=row.post_id,
                    post_url=row.post_url, post_content=row.post_content,
                    source_name=row.source_name, posted_at=row.posted_at,
                )
            elif has_source:
                if not sources[key].source_name:
                    sources[key].source_name = row.source_name
                if not sources[key].posted_at:
                    sources[key].posted_at = row.posted_at
            chain = tuple(row.location_chain or ([row.mentioned_location] if row.mentioned_location else []))
            if not chain:
                continue
            mentioned = row.mentioned_location or chain[0]
            evidence = locations.setdefault((mentioned, chain), EventLocationEvidence(
                mentioned_location=mentioned, location_chain=list(chain),
            ))
            if has_source and sources[key].source_id not in evidence.source_ids:
                evidence.source_ids.append(sources[key].source_id)

        # Cypher returns every hierarchy prefix. Display only complete branches;
        # retain distinct branches and the sources that actually supplied each.
        full_locations = [evidence for (mentioned, chain), evidence in locations.items()
                          if not any(other_mentioned == mentioned and len(other) > len(chain)
                                     and other[:len(chain)] == chain
                                     for other_mentioned, other in locations)]
        groups.append(EventLocationGroup(
            event_key=event_key,
            event_description=next((row.event_description for row in rows if row.event_description), None),
            location_status="mentioned" if full_locations else "unknown",
            locations=full_locations, sources=list(sources.values()),
        ))
    return groups
