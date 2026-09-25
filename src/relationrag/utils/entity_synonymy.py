import logging
import re
from collections import defaultdict
from typing import List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

ARTICLES = {"the", "a", "an"}
MIN_ALNUM_LEN = 4


def normalize_entity_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t and t not in ARTICLES]
    return " ".join(tokens)


def tokenize_entity_name(name: str) -> List[str]:
    normalized = normalize_entity_name(name)
    return normalized.split() if normalized else []


def _alnum_len(text: str) -> int:
    return len(re.sub(r"[^a-z0-9]", "", text))


def are_entity_synonyms(left: str, right: str) -> bool:
    """True only for identity/alias, not thematic similarity."""
    if left == right:
        return False

    left_norm = normalize_entity_name(left)
    right_norm = normalize_entity_name(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_tokens = left_norm.split()
    right_tokens = right_norm.split()
    if not left_tokens or not right_tokens:
        return False

    if len(left_norm) <= len(right_norm):
        shorter_tokens, longer_tokens = left_tokens, right_tokens
        shorter_norm = left_norm
    else:
        shorter_tokens, longer_tokens = right_tokens, left_tokens
        shorter_norm = right_norm

    if _alnum_len(shorter_norm) < MIN_ALNUM_LEN:
        return False
    if set(shorter_tokens) == set(longer_tokens):
        return False
    return set(shorter_tokens).issubset(set(longer_tokens))


def find_synonym_pairs(entities: Sequence[str]) -> List[Tuple[str, str]]:
    """Return unique unordered alias pairs among entity surface forms."""
    unique_entities = list(dict.fromkeys(e for e in entities if e.strip()))
    token_to_entities = defaultdict(list)
    normalized_tokens = {}

    for entity in unique_entities:
        tokens = tokenize_entity_name(entity)
        normalized_tokens[entity] = tokens
        for token in set(tokens):
            token_to_entities[token].append(entity)

    pairs: Set[Tuple[str, str]] = set()
    for entity in unique_entities:
        tokens = normalized_tokens[entity]
        if not tokens:
            continue
        candidate_sets = [token_to_entities[token] for token in tokens]
        if not candidate_sets:
            continue
        candidates = set(candidate_sets[0])
        for extra in candidate_sets[1:]:
            candidates.intersection_update(extra)

        for other in candidates:
            if other == entity:
                continue
            if are_entity_synonyms(entity, other):
                pairs.add(tuple(sorted((entity, other))))

    logger.info(f"Found {len(pairs)} entity synonym pairs from {len(unique_entities)} entities.")
    return list(pairs)
