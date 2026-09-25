import re
from typing import Any, Dict, List, Optional, Set

SYMMETRIC_TYPE_STEMS = {
    "contradict",
    "contrast",
    "compare",
    "oppose",
    "conflict",
}


def sanitize_relation_type(raw: Any) -> Optional[str]:
    """Normalize a semi-open P→P type to snake_case with 1–3 tokens."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    tokens = [t for t in s.split("_") if t]
    if not tokens or len(tokens) > 3:
        return None
    return "_".join(tokens)


def light_stem_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def is_symmetric_relation(rel_type: str, flagged: bool = False) -> bool:
    if flagged:
        return True
    stems = {light_stem_token(part) for part in rel_type.split("_")}
    return bool(stems & SYMMETRIC_TYPE_STEMS)


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def sanitize_proposition_relations(
    relations: Any,
    num_propositions: int,
) -> List[Dict[str, Any]]:
    """Keep only well-formed intra-list P→P relations."""
    if not isinstance(relations, list) or num_propositions <= 0:
        return []

    cleaned: List[Dict[str, Any]] = []
    seen: Set[tuple] = set()

    for rel in relations:
        if not isinstance(rel, dict):
            continue
        src = _as_int(rel.get("src"))
        dst = _as_int(rel.get("dst"))
        rel_type = sanitize_relation_type(rel.get("type"))
        if src is None or dst is None or rel_type is None:
            continue
        if src == dst:
            continue
        if not (0 <= src < num_propositions and 0 <= dst < num_propositions):
            continue
        flagged = bool(rel.get("symmetric", False))
        symmetric = is_symmetric_relation(rel_type, flagged=flagged)
        key = (src, dst, rel_type)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "src": src,
            "dst": dst,
            "type": rel_type,
            "symmetric": symmetric,
        })

    return cleaned
