import hashlib
import re

from event_titles import is_valid_event_title
from knowledge_settings import (
    ANONYMOUS_PARTICIPANT_PATTERN,
    EVENT_ACTION_TRIGGERS,
    EVENT_NAME_PATTERN,
    EVENT_RELATION_TYPES,
    EVENT_ROLES,
    EVENT_STATUSES,
    EVENT_TYPES,
    GLOBAL_PARTICIPANT_ROLE_EXACT,
    MAX_EVENTS_PER_POST,
    PARTICIPANT_SCOPES,
    RELATION_EVIDENCE_MARKERS,
)
from knowledge_extraction import (
    _clean_text,
    _enum_value,
    _evidence_in_content,
    _normalized_source_text,
    _valid_confidence,
    classify_entity_type,
    is_generic_entity,
    make_search_name,
    normalize_name,
    normalize_null,
    prepare_entity,
)


def validate_entities(raw_entities) -> dict:
    result = {
        "entities": [],
        "entity_id_map": {},
        "entity_identities": {},
        "entity_names": {},
        "entity_types": {},
        "generic_participants": {},
        "generic_entity_keys": [],
    }
    if not isinstance(raw_entities, list):
        return result

    seen_local_ids = set()
    seen_entity_keys = {}
    seen_entity_names = {}
    seen_generic_keys = set()

    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue
        local_id = _clean_text(raw.get("local_id"))
        if not local_id or local_id in seen_local_ids:
            continue
        seen_local_ids.add(local_id)

        name = _clean_text(raw.get("name"))
        entity_type = classify_entity_type(raw)
        if is_generic_entity(raw):
            if name:
                result["generic_participants"][local_id] = name
            if name and entity_type:
                key = (normalize_name(name), entity_type)
                if key not in seen_generic_keys:
                    result["generic_entity_keys"].append(
                        {"normalized_name": key[0], "type": key[1]}
                    )
                    seen_generic_keys.add(key)
            continue

        prepared = prepare_entity(raw)
        if prepared is None:
            continue
        raw_entity_names = {
            normalized_name
            for candidate in (name, _clean_text(raw.get("canonical_name")))
            if (normalized_name := normalize_name(candidate))
        }
        entity_key = (prepared["normalized_name"], prepared["entity_type"])
        if entity_key in seen_entity_keys:
            kept_local_id = seen_entity_keys[entity_key]
            shared_entity_names = seen_entity_names[entity_key]
            shared_entity_names.update(raw_entity_names)
            result["entity_names"][local_id] = shared_entity_names
            result["entity_id_map"][local_id] = kept_local_id
            result["entity_identities"][local_id] = "|".join(entity_key)
            result["entity_types"][local_id] = prepared["entity_type"]
            continue

        normalized = {
            "local_id": local_id,
            "name": prepared["name"],
            "canonical_name": _clean_text(raw.get("canonical_name")),
            "type": prepared["entity_type"],
            "resolution_confidence": prepared["confidence"],
        }
        result["entities"].append(normalized)
        result["entity_id_map"][local_id] = local_id
        result["entity_identities"][local_id] = "|".join(entity_key)
        result["entity_names"][local_id] = raw_entity_names
        result["entity_types"][local_id] = prepared["entity_type"]
        seen_entity_keys[entity_key] = local_id
        seen_entity_names[entity_key] = raw_entity_names

    return result


def _contains_marker(text: str, marker: str) -> bool:
    return (
        re.search(
            rf"(?<!\w){re.escape(marker)}(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )


def resolve_event_type(event_type: str, evidence_text: str) -> str:
    evidence = normalize_name(evidence_text)
    matching_types = {
        candidate_type
        for candidate_type, triggers in EVENT_ACTION_TRIGGERS.items()
        if any(_contains_marker(evidence, trigger) for trigger in triggers)
    }
    if len(matching_types) == 1:
        verified_type = next(iter(matching_types))
        if verified_type != event_type:
            return verified_type
    # No match or multiple matches cannot verify a unique replacement. Keep the
    # model-provided taxonomy instead of dropping the event or guessing its type.
    return event_type


def has_actionable_event(event_type: str, evidence_text: str) -> bool:
    return resolve_event_type(event_type, evidence_text) is not None


def _participant_signature(participant: dict) -> str:
    if participant["entity_id"]:
        identity = (
            f"entity:{participant.get('_entity_identity', participant['entity_id'])}"
        )
    else:
        identity = f"anonymous:{normalize_name(participant['participant_text'])}"
    return f"{identity}:{participant['role']}"


def _infer_anonymous_participant_text(raw_event: dict) -> str:
    """Recover one unambiguous anonymous description from an event."""
    candidates = {}
    for field in ("description", "evidence_text"):
        text = _clean_text(raw_event.get(field))
        for match in ANONYMOUS_PARTICIPANT_PATTERN.finditer(text):
            candidate = _clean_text(match.group(0))
            candidates.setdefault(normalize_name(candidate), candidate)
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    return ""


def _resolve_participant_scope(value, participant_text: str) -> str:
    explicit_scope = _enum_value(value, PARTICIPANT_SCOPES)
    if explicit_scope is not None:
        return explicit_scope
    if normalize_name(participant_text) in GLOBAL_PARTICIPANT_ROLE_EXACT:
        return "GLOBAL_ROLE"
    return "POST_LOCAL"


def _event_signature(event: dict) -> str:
    participants = sorted(
        _participant_signature(participant) for participant in event["participants"]
    )
    time_expression = normalize_name(event.get("time_expression") or "")
    return "|".join(
        [
            event["type"],
            _normalized_source_text(event["evidence_text"]),
            time_expression,
            *participants,
        ]
    )


def validate_events(
    raw_events,
    content: str,
    entity_validation: dict,
) -> dict:
    result = {"events": [], "event_id_map": {}}
    if not isinstance(raw_events, list):
        return result

    seen_local_ids = set()
    seen_signatures = {}
    valid_entity_ids = entity_validation["entity_id_map"]
    valid_entity_names = entity_validation["entity_names"]
    valid_entity_types = entity_validation["entity_types"]
    generic_participants = entity_validation["generic_participants"]

    for raw in raw_events:
        if len(result["events"]) >= MAX_EVENTS_PER_POST:
            break
        if not isinstance(raw, dict):
            continue
        local_id = _clean_text(raw.get("local_id"))
        if not local_id or local_id in seen_local_ids:
            continue
        seen_local_ids.add(local_id)

        event_type = _enum_value(raw.get("type"), EVENT_TYPES)
        title = _clean_text(raw.get("title"))
        description = _clean_text(raw.get("description"))
        evidence_text = _clean_text(raw.get("evidence_text"))
        confidence = _valid_confidence(raw.get("confidence"))
        if (
            event_type is None
            or not description
            or not evidence_text
            or confidence is None
            or not _evidence_in_content(evidence_text, content)
        ):
            continue
        event_type = resolve_event_type(event_type, evidence_text)

        status = _enum_value(raw.get("status"), EVENT_STATUSES) or "UNKNOWN"
        participants = []
        seen_participants = set()
        raw_participants = raw.get("participants")
        if not isinstance(raw_participants, list):
            raw_participants = []

        for raw_participant in raw_participants:
            if not isinstance(raw_participant, dict):
                continue
            role = _enum_value(raw_participant.get("role"), EVENT_ROLES)
            participant_confidence = _valid_confidence(
                raw_participant.get("confidence")
            )
            if role is None or participant_confidence is None:
                continue

            raw_entity_id = _clean_text(raw_participant.get("entity_id"))
            participant_text = _clean_text(raw_participant.get("participant_text"))
            entity_id = None
            if raw_entity_id in valid_entity_ids:
                normalized_participant = normalize_name(participant_text)
                text_matches_entity = (
                    not participant_text
                    or normalized_participant in valid_entity_names[raw_entity_id]
                    or (
                        role == "LOCATION"
                        and valid_entity_types[raw_entity_id] == "LOCATION"
                        and any(
                            _contains_marker(normalized_participant, entity_name)
                            for entity_name in valid_entity_names[raw_entity_id]
                        )
                    )
                )
                if text_matches_entity:
                    entity_id = valid_entity_ids[raw_entity_id]
                    participant_text = ""
            elif raw_entity_id in generic_participants:
                participant_text = (
                    participant_text or generic_participants[raw_entity_id]
                )
            elif raw_entity_id:
                entity_id = None
                if not participant_text:
                    participant_text = _infer_anonymous_participant_text(raw)

            if participant_text and EVENT_NAME_PATTERN.search(participant_text):
                continue
            if not entity_id and not participant_text:
                continue
            participant = {
                "entity_id": entity_id,
                "participant_text": participant_text or None,
                "participant_scope": (
                    None
                    if entity_id
                    else _resolve_participant_scope(
                        raw_participant.get("participant_scope"), participant_text
                    )
                ),
                "role": role,
                "confidence": participant_confidence,
            }
            if entity_id:
                participant["_entity_identity"] = entity_validation[
                    "entity_identities"
                ][raw_entity_id]
            signature = _participant_signature(participant)
            if signature in seen_participants:
                continue
            seen_participants.add(signature)
            participants.append(participant)

        time_expression = normalize_null(raw.get("time_expression"))
        if time_expression is not None:
            time_expression = _clean_text(time_expression) or None

        event = {
            "local_id": local_id,
            "type": event_type,
            "title": title if is_valid_event_title(title) else description,
            "title_needs_backfill": bool(
                raw.get("title_needs_backfill")
                or not is_valid_event_title(title)
            ),
            "description": description,
            "evidence_text": evidence_text,
            "status": status,
            "time_expression": time_expression,
            "confidence": confidence,
            "participants": participants,
        }
        signature = _event_signature(event)
        if signature in seen_signatures:
            result["event_id_map"][local_id] = seen_signatures[signature]
            continue

        result["events"].append(event)
        result["event_id_map"][local_id] = local_id
        seen_signatures[signature] = local_id

    return result


def _relation_evidence_is_explicit(relation_type: str, evidence_text: str) -> bool:
    if relation_type == "RELATED_TO":
        return True
    evidence = normalize_name(evidence_text)
    return any(
        _contains_marker(evidence, marker)
        for marker in RELATION_EVIDENCE_MARKERS.get(relation_type, set())
    )


def validate_event_relations(
    raw_relations,
    content: str,
    event_validation: dict,
) -> list[dict]:
    if not isinstance(raw_relations, list):
        return []
    event_id_map = event_validation["event_id_map"]
    relations = []
    seen = set()

    for raw in raw_relations:
        if not isinstance(raw, dict):
            continue
        raw_source = _clean_text(raw.get("source_event_id"))
        raw_target = _clean_text(raw.get("target_event_id"))
        relation_type = _enum_value(raw.get("type"), EVENT_RELATION_TYPES)
        evidence_text = _clean_text(raw.get("evidence_text"))
        source = event_id_map.get(raw_source)
        target = event_id_map.get(raw_target)
        if (
            relation_type is None
            or source is None
            or target is None
            or source == target
            or not evidence_text
            or not _evidence_in_content(evidence_text, content)
            or not _relation_evidence_is_explicit(relation_type, evidence_text)
        ):
            continue
        signature = (source, relation_type, target)
        if signature in seen:
            continue
        seen.add(signature)
        relations.append(
            {
                "source_event_id": source,
                "type": relation_type,
                "target_event_id": target,
                "evidence_text": evidence_text,
            }
        )
    return relations


def build_mention_key(platform: str, post_id: str, event: dict) -> str:
    identity = f"{platform}|{post_id}|{_event_signature(event)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_event_key(platform: str, post_id: str, event: dict) -> str:
    """Build the initial canonical key; it never changes with its summary."""
    mention_key = build_mention_key(platform, post_id, event)
    return hashlib.sha256(f"canonical|{mention_key}".encode("utf-8")).hexdigest()


def build_anonymous_participant_key(
    platform: str,
    post_id: str,
    event_key: str,
    participant: dict,
) -> str:
    normalized_text = normalize_name(participant["participant_text"])
    participant_scope = participant.get("participant_scope", "POST_LOCAL")
    if participant_scope == "GLOBAL_ROLE":
        identity = f"global_role|{platform}|{normalized_text}"
    else:
        identity = f"post_local|{platform}|{post_id}|{normalized_text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def validate_knowledge(
    content: str,
    raw_knowledge: dict,
    platform: str = "",
    post_id: str = "",
) -> dict:
    raw = normalize_null(raw_knowledge if isinstance(raw_knowledge, dict) else {})
    entity_validation = validate_entities(raw.get("entities", []))
    event_validation = validate_events(
        raw.get("events", []), content, entity_validation
    )
    relations = validate_event_relations(
        raw.get("event_relations", []), content, event_validation
    )

    events = event_validation["events"]
    if platform and post_id:
        for event in events:
            event["mention_key"] = build_mention_key(platform, post_id, event)
            event["event_key"] = build_event_key(platform, post_id, event)
    for event in events:
        for participant in event["participants"]:
            participant.pop("_entity_identity", None)

    return {
        "entities": entity_validation["entities"],
        "events": events,
        "event_relations": relations,
        "generic_entity_keys": entity_validation["generic_entity_keys"],
    }
