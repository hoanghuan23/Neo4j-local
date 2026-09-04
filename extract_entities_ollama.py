import logging

# Ollama imports are retained only as a disabled compatibility fallback for
# older tools/tests. The runtime pipeline below uses Gemini exclusively.
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
from knowledge_relation_router import classify_relation_routes
from knowledge_relations.participant_role import enrich_participant_roles
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

_gemini_caller: GeminiKnowledgeCaller | None = None


def get_gemini_caller() -> GeminiKnowledgeCaller:
    """Create one Gemini client and reuse it for the whole pipeline."""
    global _gemini_caller
    if _gemini_caller is None:
        _gemini_caller = GeminiKnowledgeCaller()
    return _gemini_caller


def call_gemini(prompt: str, output_schema: dict) -> dict:
    return get_gemini_caller()(prompt, output_schema)


# Disabled Ollama runtime path (kept above only as a compatibility fallback):
# call_groq = call_ollama


def call_groq(prompt: str, output_schema: dict) -> dict:
    """Legacy patch point; route historical callers through Gemini."""
    return call_gemini(prompt, output_schema)


def extract_knowledge(content: str) -> dict:
    """Extract raw knowledge while preserving the legacy patch point."""
    return _extraction.extract_knowledge(content, call_model=call_groq)


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
        classify_relations_fn=lambda content, knowledge: classify_relation_routes(
            content,
            knowledge,
            call_model=call_model,
        ),
        enrich_participants_fn=lambda content, knowledge, routes: (
            enrich_participant_roles(
                content,
                knowledge,
                routes,
                call_model=call_model,
            )
        ),
        consolidate_fn=consolidate_batch,
    )


def main() -> None:
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
