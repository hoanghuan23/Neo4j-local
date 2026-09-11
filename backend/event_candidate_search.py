"""Domain-independent lexical retrieval for free-text event descriptions.

This is candidate retrieval, not a semantic or factual confirmation. Match at
least half the distinct content tokens (at least two for multi-token queries),
then prefer coverage and adjacent phrases. All values remain query parameters.
"""

import math
import re
import unicodedata

from backend.event_search_queries import _fold


# Keep topical words, names, numbers and negation. Remove only grammatical filler.
_STOP_WORDS = frozenset(
    "là và của có một những các đã đang được bị thì mà với tại ở từ đến "
    "cho để này đó ấy vào về trong trên dưới ra".split()
)


def _normalize(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def candidate_search_parameters(description: str) -> dict:
    # Remove accented stop words before folding to avoid dropping content words
    # such as “vàng” just because their folded forms resemble grammatical words.
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", description.casefold()))
    content = [_normalize(word) for word in words if word not in _STOP_WORDS]
    terms = list(dict.fromkeys(content))
    phrases = list(dict.fromkeys(
        (_normalize(left), _normalize(right))
        for left, right in zip(words, words[1:])
        if left not in _STOP_WORDS and right not in _STOP_WORDS
    ))
    boundary = r"[^\p{L}\p{N}]"

    def pattern(parts: tuple[str, ...]) -> str:
        return (r"(?s).*?(?<![\p{L}\p{N}])" +
                (boundary + "+").join(re.escape(part) for part in parts) +
                r"(?![\p{L}\p{N}]).*")

    return {
        "candidate_terms": [{"term": term, "pattern": pattern((term,))} for term in terms],
        "candidate_phrases": [pattern(phrase) for phrase in phrases],
        "candidate_min_matches": min(len(terms), max(2, math.ceil(len(terms) * 0.5))),
        "fold_characters": [chr(code) for code in range(0x300, 0x370)
                            if unicodedata.category(chr(code)) == "Mn"],
    }


# Produces one scored row per sourced Event, before any result limit or location
# expansion. Both the current and legacy evidence schemas are supported.
EVENT_CANDIDATES_QUERY = f"""
MATCH (event:Event)
WHERE size($candidate_terms) > 0
  AND (EXISTS {{ MATCH (:Post)-[:HAS_EVENT_MENTION]->(:EventMention)-[:EVIDENCE_FOR]->(event) }}
       OR EXISTS {{ MATCH (:Post)-[:DESCRIBES]->(event) }})
WITH event, [{_fold('event.title')}, {_fold('event.description')}] AS candidate_texts
WITH event, candidate_texts,
     [term IN $candidate_terms WHERE any(text IN candidate_texts WHERE text =~ term.pattern) | term.term]
       AS matched_terms
WHERE size(matched_terms) >= $candidate_min_matches
WITH event, matched_terms,
     toFloat(size(matched_terms)) / size($candidate_terms) AS match_coverage,
     size([phrase IN $candidate_phrases
           WHERE any(text IN candidate_texts WHERE text =~ phrase)]) AS phrase_matches
WITH event, matched_terms, match_coverage,
     match_coverage + CASE WHEN size($candidate_phrases) = 0 THEN 0.0
       ELSE 0.2 * phrase_matches / size($candidate_phrases) END AS candidate_score
"""
