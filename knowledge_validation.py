import hashlib
import re

from knowledge_settings import (
    EVENT_ACTION_TRIGGERS,
    EVENT_NAME_PATTERN,
    EVENT_RELATION_TYPES,
    EVENT_ROLES,
    EVENT_STATUSES,
    EVENT_TYPES,
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
        "generic_participants": {},
        "generic_entity_keys": [],
    }
    if not isinstance(raw_entities, list):
        return result

    seen_local_ids = set()
    seen_entity_keys = {}
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
        entity_key = (prepared["normalized_name"], prepared["entity_type"])
        if entity_key in seen_entity_keys:
            kept_local_id = seen_entity_keys[entity_key]
            result["entity_id_map"][local_id] = kept_local_id
            result["entity_identities"][local_id] = "|".join(entity_key)
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
        seen_entity_keys[entity_key] = local_id

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


def has_actionable_event(event_type: str, evidence_text: str) -> bool:
    evidence = normalize_name(evidence_text)
    if not evidence:
        return False
    if event_type == "OTHER" and re.fullmatch(
        r"(?:the )?(?:incident|event|scene) (?:occurred|happened|took place) "
        r"(?:on|in|at|near|during) .+",
        evidence.rstrip("."),
    ):
        return False
    triggers = EVENT_ACTION_TRIGGERS.get(event_type, set())
    if any(_contains_marker(evidence, trigger) for trigger in triggers):
        return True
    # A conservative fallback for explicit English verb forms in OTHER events.
    if event_type == "OTHER":
        words = re.findall(r"\b[a-z]+\b", make_search_name(evidence))
        return any(
            len(word) > 4 and word.endswith(("ed", "ing"))
            for word in words
            if word not in {"during", "following", "including", "pending"}
        )
    return False


def _participant_signature(participant: dict) -> str:
    if participant["entity_id"]:
        identity = (
            f"entity:{participant.get('_entity_identity', participant['entity_id'])}"
        )
    else:
        identity = f"anonymous:{normalize_name(participant['participant_text'])}"
    return f"{identity}:{participant['role']}"


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
    generic_participants = entity_validation["generic_participants"]

    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        local_id = _clean_text(raw.get("local_id"))
        if not local_id or local_id in seen_local_ids:
            continue
        seen_local_ids.add(local_id)

        event_type = _enum_value(raw.get("type"), EVENT_TYPES)
        description = _clean_text(raw.get("description"))
        evidence_text = _clean_text(raw.get("evidence_text"))
        confidence = _valid_confidence(raw.get("confidence"))
        if (
            event_type is None
            or not description
            or not evidence_text
            or confidence is None
            or not _evidence_in_content(evidence_text, content)
            or not has_actionable_event(event_type, evidence_text)
        ):
            continue

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
                entity_id = valid_entity_ids[raw_entity_id]
                participant_text = ""
            elif raw_entity_id in generic_participants:
                participant_text = (
                    participant_text or generic_participants[raw_entity_id]
                )
            elif raw_entity_id:
                # An unresolved reference is only salvageable when text exists.
                entity_id = None

            if participant_text and EVENT_NAME_PATTERN.search(participant_text):
                continue
            if not entity_id and not participant_text:
                continue
            participant = {
                "entity_id": entity_id,
                "participant_text": participant_text or None,
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
        start_year = normalize_null(raw.get("start_year"))
        end_year = normalize_null(raw.get("end_year"))
        if isinstance(start_year, bool) or not isinstance(
            start_year, (int, type(None))
        ):
            start_year = None
        if isinstance(end_year, bool) or not isinstance(end_year, (int, type(None))):
            end_year = None

        event = {
            "local_id": local_id,
            "type": event_type,
            "description": description,
            "evidence_text": evidence_text,
            "status": status,
            "time_expression": time_expression,
            "start_year": start_year,
            "end_year": end_year,
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


def build_event_key(platform: str, post_id: str, event: dict) -> str:
    identity = f"{platform}|{post_id}|{_event_signature(event)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_anonymous_participant_key(
    platform: str,
    post_id: str,
    event_key: str,
    participant: dict,
) -> str:
    identity = "|".join(
        [
            platform,
            post_id,
            event_key,
            normalize_name(participant["participant_text"]),
            participant["role"],
        ]
    )
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
