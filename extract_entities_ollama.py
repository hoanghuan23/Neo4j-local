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
from knowledge_consolidation import consolidate_pending_mentions
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

# Legacy public patch point retained for older callers and tests. Runtime
# behavior remains Ollama-backed; the historical name cannot be removed yet.
call_groq = call_ollama


def extract_knowledge(content: str) -> dict:
    """Extract raw knowledge while preserving the legacy patch point."""
    return _extraction.extract_knowledge(content, call_model=call_groq)


def extract_entities(content: str) -> list[dict]:
    """Compatibility wrapper for callers that only need named entities."""
    return extract_knowledge(content)["entities"]


def process_new_posts(session) -> dict:
    """Process and consolidate posts entirely with the configured Ollama model."""
    return _process_new_posts(
        session,
        classify_post_fn=lambda content: (
            _extraction.classify_knowledge_potential(
                content,
                call_model=call_ollama,
            )
        ),
        extract_knowledge_fn=lambda content: _extraction.extract_knowledge(
            content,
            call_model=call_ollama,
        ),
        consolidate_fn=lambda session: consolidate_pending_mentions(
            session,
            call_model=call_ollama,
        ),
    )


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
