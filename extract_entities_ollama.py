import logging

import requests
from neo4j import GraphDatabase

import knowledge_extraction as _extraction
from knowledge_extraction import (
    call_ollama,
    classify_entity_type,
    is_generic_entity,
    make_search_name,
    normalize_name,
    normalize_null,
    parse_ollama_payload,
    prepare_entity,
)
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
from knowledge_gemini import GeminiKnowledgeCaller
from knowledge_settings import *
from knowledge_validation import (
    build_anonymous_participant_key,
    build_event_key,
    has_actionable_event,
    validate_entities,
    validate_event_relations,
    validate_events,
    validate_knowledge,
)


def extract_knowledge(content: str) -> dict:
    """Extract raw knowledge while preserving the legacy patch point."""
    return _extraction.extract_knowledge(content, call_model=call_ollama)


def extract_entities(content: str) -> list[dict]:
    """Compatibility wrapper for callers that only need named entities."""
    return extract_knowledge(content)["entities"]


def process_new_posts(session) -> None:
    """Process posts with Gemini and print actual token-based cost."""
    caller = GeminiKnowledgeCaller()
    try:
        _process_new_posts(
            session,
            extract_knowledge_fn=lambda content: _extraction.extract_knowledge(
                content,
                call_model=caller,
            ),
        )
    finally:
        caller.print_cost_summary(target_posts=POST_LIMIT)
        caller.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        with driver.session(database="neo4j") as session:
            process_new_posts(session)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
