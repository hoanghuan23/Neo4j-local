import logging

from neo4j import GraphDatabase

import knowledge_extraction as _extraction
from knowledge_extraction import (
    classify_entity_type,
    is_generic_entity,
    make_search_name,
    normalize_name,
    normalize_null,
    prepare_entity,
)
from knowledge_gemini import GeminiKnowledgeCaller
from knowledge_persistence import (
    create_entity_schema,
    create_knowledge_schema,
    mark_knowledge_failure,
    save_entities,
    save_knowledge_tx,
    upsert_entities,
    upsert_event_relations,
    upsert_events,
)
from knowledge_pipeline import _load_posts
from knowledge_pipeline import process_new_posts as _process_new_posts
from knowledge_relations.participant_role import extract_participants
from knowledge_relations.event_relation import extract_event_relations
from knowledge_relations.entity_hierarchy.location_hierarchy import enrich_location_hierarchy
from knowledge_relations.entity_hierarchy.organization_hierarchy import (
    extract_context,
    enrich_organization_hierarchy,
)
from knowledge_relations.event_hierarchy import consolidate_pending_mentions
from knowledge_settings import *
from knowledge_validation import (
    build_anonymous_participant_key,
    build_event_key,
    build_mention_key,
    has_actionable_event,
    validate_entities,
    validate_event_relations,
    validate_events,
    validate_knowledge,
)

_gemini_caller: GeminiKnowledgeCaller | None = None


def get_gemini_caller() -> GeminiKnowledgeCaller:
    """Create one Gemini client and reuse it for the whole pipeline."""
    global _gemini_caller
    if _gemini_caller is None:
        _gemini_caller = GeminiKnowledgeCaller()
    return _gemini_caller


def call_gemini(prompt: str, output_schema: dict) -> dict:
    return get_gemini_caller()(prompt, output_schema)


def extract_knowledge(content: str) -> dict:
    """Extract raw knowledge with the configured Gemini model."""
    return _extraction.extract_knowledge(content, call_model=call_gemini)


def extract_entities(content: str) -> list[dict]:
    """Compatibility wrapper for callers that only need named entities."""
    return extract_knowledge(content)["entities"]


def process_new_posts(session, call_model=None) -> dict:
    """Process and consolidate posts entirely with the configured Gemini model."""
    if call_model is None:
        call_model = get_gemini_caller()

    def consolidate_batch(session, mention_keys=None):
        kwargs = {"call_model": call_model}
        if mention_keys is not None:
            kwargs["mention_keys"] = mention_keys
        return consolidate_pending_mentions(session, **kwargs)

    return _process_new_posts(
        session,
        classify_post_fn=lambda content: (
            _extraction.classify_knowledge_potential(
                content,
                call_model=call_model,
            )
        ),
        extract_knowledge_fn=lambda content: _extraction.extract_knowledge(
            content,
            call_model=call_model,
        ),
        extract_participants_fn=lambda content, knowledge: extract_participants(
            content, knowledge, call_model=call_model,
        ),
        extract_event_relations_fn=lambda content, knowledge: extract_event_relations(
            content, knowledge, call_model=call_model,
        ),
        enrich_locations_fn=lambda session, platform, post_id, content, knowledge: (
            enrich_location_hierarchy(
                session, platform, post_id, content, knowledge,
                call_model=call_model,
            )
        ),
        organization_context_fn=lambda content, knowledge: extract_context(
            content, knowledge, call_model=call_model,
        ),
        enrich_organizations_fn=lambda session, platform, post_id, content, knowledge: (
            enrich_organization_hierarchy(
                session, platform, post_id, content, knowledge,
                call_model=call_model,
            )
        ),
        consolidate_fn=consolidate_batch,
    )


def main() -> None:
    logging.getLogger("knowledge.api").setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        with driver.session(database="neo4j") as session:
            summary = process_new_posts(session)
            if _gemini_caller is not None:
                _gemini_caller.print_cost_summary(
                    target_posts=summary["total"],
                    stage_label="toàn bộ pipeline",
                )
    finally:
        if _gemini_caller is not None:
            _gemini_caller.close()
        driver.close()


if __name__ == "__main__":
    main()
