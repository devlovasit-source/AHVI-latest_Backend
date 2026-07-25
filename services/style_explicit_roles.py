"""Explicit requested-role contract for Style boards.

When a user names garment roles in the prompt ("a dinner outfit with a dress,
shoes, and a bag") those roles are HARD constraints: the final board must carry
every one of them, or the pipeline must return a typed failure. Before this
module the only completeness rule was the generic top+bottom+footwear gate in
``brain.engines.outfit_quality_guard``, so explicitly requested outerwear / bag /
dress were silently dropped.

Two responsibilities:
  1. phrase-aware extraction of requested roles from the query;
  2. validation + one deterministic repair pass over a candidate pool.

Nothing here fetches. It is pure over the items/pool it is given, so it is safe
to call from the curation path.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("ahvi.style_explicit_roles")

# Canonical explicit roles. "bag" is tracked separately from the generic
# "accessory" role because users request it by name and the accessory curator
# used to drop it for date/dinner occasions.
ROLE_TOP = "top"
ROLE_BOTTOM = "bottom"
ROLE_DRESS = "dress"
ROLE_OUTERWEAR = "outerwear"
ROLE_FOOTWEAR = "footwear"
ROLE_BAG = "bag"
ROLE_ACCESSORY = "accessory"

EXPLICIT_ROLES = (
    ROLE_TOP,
    ROLE_BOTTOM,
    ROLE_DRESS,
    ROLE_OUTERWEAR,
    ROLE_FOOTWEAR,
    ROLE_BAG,
    ROLE_ACCESSORY,
)

# Phrases where "dress" is an ADJECTIVE, not a dress garment. They are consumed
# before role scanning so "dress shoes and a bag" never yields a dress.
_MODIFIER_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("dress shoes", ROLE_FOOTWEAR),
    ("dress shoe", ROLE_FOOTWEAR),
    ("dress boots", ROLE_FOOTWEAR),
    ("dress boot", ROLE_FOOTWEAR),
    ("dress sneakers", ROLE_FOOTWEAR),
    ("dress loafers", ROLE_FOOTWEAR),
    ("dress sandals", ROLE_FOOTWEAR),
    ("dress shirt", ROLE_TOP),
    ("dress shirts", ROLE_TOP),
    ("dress top", ROLE_TOP),
    ("dress pants", ROLE_BOTTOM),
    ("dress pant", ROLE_BOTTOM),
    ("dress trousers", ROLE_BOTTOM),
    ("dress socks", ROLE_ACCESSORY),
    ("dress watch", ROLE_ACCESSORY),
    ("dress belt", ROLE_ACCESSORY),
)

# Role keywords scanned with word boundaries after modifier phrases are removed.
_ROLE_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (ROLE_DRESS, ("dress", "dresses", "gown", "gowns", "saree", "sari", "lehenga", "jumpsuit")),
    (ROLE_OUTERWEAR, ("outerwear", "jacket", "jackets", "blazer", "blazers", "coat", "coats", "overcoat", "trench", "cardigan")),
    (ROLE_FOOTWEAR, ("shoes", "shoe", "footwear", "sneakers", "sneaker", "heels", "heel", "boots", "boot", "loafers", "loafer", "sandals", "sandal", "mojaris", "flats")),
    (ROLE_BAG, ("bag", "bags", "handbag", "purse", "tote", "clutch", "backpack")),
    (ROLE_BOTTOM, ("trousers", "trouser", "pants", "jeans", "skirt", "skirts", "shorts", "chinos", "chino", "joggers", "bottoms", "bottom", "palazzo", "churidar")),
    (ROLE_TOP, ("top", "tops", "shirt", "shirts", "tshirt", "t-shirt", "tee", "blouse", "kurta", "kurti", "sweater", "hoodie", "polo", "tunic")),
    (ROLE_ACCESSORY, ("accessory", "accessories", "watch", "belt", "scarf", "jewellery", "jewelry", "necklace", "bracelet", "earrings", "sunglasses")),
)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def extract_requested_roles(query: Any) -> List[str]:
    """Phrase-aware explicit garment roles named in [query].

    "dress, shoes and a bag"  -> [dress, footwear, bag]
    "dress shoes and a bag"   -> [footwear, bag]        (no dress garment)
    "suggest a complete outfit" -> []                   (nothing explicit)

    Order follows EXPLICIT_ROLES so the result is deterministic.
    """
    text = _norm(query)
    if not text:
        return []

    found: set = set()
    # 1. Consume adjective phrases first ("dress shoes" -> footwear only).
    for phrase, role in _MODIFIER_PHRASES:
        if phrase in text:
            found.add(role)
            text = text.replace(phrase, " ")

    # 2. Scan what remains for standalone role nouns.
    for role, words in _ROLE_KEYWORDS:
        for word in words:
            if re.search(r"\b" + re.escape(word) + r"\b", text):
                found.add(role)
                break

    return [role for role in EXPLICIT_ROLES if role in found]


# Item vocabulary. Owned here (not delegated) because the outfit guard's role
# map has no "outerwear" role and does not know heels / blazer / pumps — it only
# needs top/bottom/footwear for the generic completeness gate. Order matters:
# bag and outerwear are tested before top/bottom so a blazer is never a "top".
_ITEM_ROLE_TOKENS: Tuple[Tuple[str, set], ...] = (
    (ROLE_BAG, {"bag", "bags", "handbag", "handbags", "purse", "tote", "clutch", "backpack", "sling"}),
    (ROLE_OUTERWEAR, {"outerwear", "jacket", "jackets", "blazer", "blazers", "coat", "coats", "overcoat", "trench", "cardigan", "shrug"}),
    (ROLE_FOOTWEAR, {"footwear", "shoe", "shoes", "heel", "heels", "pump", "pumps", "boot", "boots", "loafer", "loafers", "sneaker", "sneakers", "sandal", "sandals", "slider", "sliders", "mojari", "mojaris", "flats", "trainers", "brogues", "oxfords", "stiletto", "stilettos"}),
    (ROLE_DRESS, {"dress", "dresses", "gown", "gowns", "saree", "sari", "lehenga", "jumpsuit", "onepiece"}),
    (ROLE_BOTTOM, {"bottom", "bottoms", "trouser", "trousers", "pant", "pants", "jeans", "denim", "skirt", "skirts", "shorts", "chino", "chinos", "jogger", "joggers", "palazzo", "churidar", "leggings"}),
    (ROLE_TOP, {"top", "tops", "shirt", "shirts", "tshirt", "tee", "tees", "blouse", "kurta", "kurti", "polo", "hoodie", "sweater", "sweatshirt", "tunic", "tank", "cami"}),
    (ROLE_ACCESSORY, {"accessory", "accessories", "watch", "watches", "belt", "belts", "scarf", "scarves", "jewelry", "jewellery", "necklace", "bracelet", "ring", "rings", "earring", "earrings", "sunglasses", "eyewear", "cap", "caps", "hat", "hats"}),
)


def item_explicit_role(item: Dict[str, Any]) -> str:
    """Explicit role of a single item. Same role vocabulary as the rest of the
    Style pipeline, plus: a bag/purse/tote/clutch resolves to "bag" rather than
    collapsing into the generic "accessory" bucket, and outerwear is a role of
    its own (the outfit guard has no outerwear concept)."""
    if not isinstance(item, dict):
        return ""
    blob = " ".join(
        _norm(item.get(key))
        for key in ("name", "label", "title", "category", "type", "sub_category", "subcategory", "accessory_type", "slot", "role")
        if item.get(key)
    )
    # "dress shoes" / "dress shirt" in an item name must not read as a dress.
    for phrase, _role in _MODIFIER_PHRASES:
        blob = blob.replace(phrase, " ")
    tokens = set(re.findall(r"[a-z0-9\-]+", blob))
    for role, words in _ITEM_ROLE_TOKENS:
        if tokens & words:
            return role
    try:
        from brain.engines.outfit_quality_guard import _role_from_item

        role = _role_from_item(item)
    except Exception:  # noqa: BLE001 - never break curation on import issues
        role = ""
    return role or ""


def board_explicit_roles(items: Any) -> List[str]:
    """Explicit roles present on a board, deterministic order."""
    present: set = set()
    for item in items or []:
        role = item_explicit_role(item)
        if role:
            present.add(role)
    # A bag also satisfies a generic "accessory" request.
    if ROLE_BAG in present:
        present.add(ROLE_ACCESSORY)
    return [role for role in EXPLICIT_ROLES if role in present]


def missing_explicit_roles(items: Any, required_roles: Any) -> List[str]:
    """Required roles absent from [items]. Dress and top/bottom never substitute
    for one another: if the user asked for a dress, a top+bottom board is a
    miss, and vice versa."""
    present = set(board_explicit_roles(items))
    required = [r for r in (required_roles or []) if r in EXPLICIT_ROLES]
    return [role for role in EXPLICIT_ROLES if role in required and role not in present]


def _candidate_safe_for_occasion(item: Dict[str, Any], occasion: Any) -> bool:
    """Reuse the existing occasion policy for repair candidates: a refined
    dinner/evening board must not gain camouflage/cargo pieces or athletic
    trainers. Falls open on import failure so repair still works."""
    try:
        from brain.engines.outfit_quality_guard import (
            SMART_OCCASIONS,
            _is_athletic_footwear,
            _is_relaxed_footwear,
            normalize_occasion,
        )
    except Exception:  # noqa: BLE001
        return True

    occ = normalize_occasion(occasion)
    refined = occ in SMART_OCCASIONS or occ in {
        "date_night",
        "client_dinner",
        "team_dinner",
        "dinner",
        "casual_dinner",
    }
    if not refined:
        return True

    blob = " ".join(
        _norm(item.get(key))
        for key in ("name", "label", "title", "category", "type", "sub_category", "subcategory", "pattern", "print")
        if item.get(key)
    )
    if any(term in blob for term in ("camo", "camouflage", "cargo")):
        return False
    if _is_athletic_footwear(item) or _is_relaxed_footwear(item):
        return False
    # Sporty framing the guard's athletic list does not spell out ("running
    # sneakers", "gym trainers"). Plain / white / leather sneakers stay eligible,
    # matching DATE_NIGHT_FOOTWEAR_OK.
    if item_explicit_role(item) == ROLE_FOOTWEAR and any(
        term in blob
        for term in ("running", "gym", "sport", "athletic", "trainer", "track", "jogging", "basketball", "tennis")
    ):
        return False
    return True


def _source_ok(item: Dict[str, Any], source_policy: Any) -> bool:
    try:
        from brain.engines.outfit_quality_guard import _source_ok_for_policy

        return bool(_source_ok_for_policy(item, str(source_policy or "")))
    except Exception:  # noqa: BLE001
        return True


def _item_key(item: Dict[str, Any]) -> str:
    try:
        from brain.engines.outfit_quality_guard import _stable_item_key

        return str(_stable_item_key(item))
    except Exception:  # noqa: BLE001
        return _norm(item.get("id") or item.get("item_id") or item.get("name"))


def enforce_explicit_roles(
    card: Dict[str, Any],
    required_roles: Any,
    *,
    candidate_pool: Any = None,
    source_policy: str = "",
    occasion: Any = "",
) -> Tuple[Dict[str, Any] | None, str, List[str]]:
    """Validate + one deterministic repair pass.

    Returns ``(card|None, status, missing_roles)`` where status is one of
    ``satisfied`` / ``repaired`` / ``missing_explicit_roles``.

    Valid items are always preserved; a missing role is only ever filled by a
    candidate of THAT role (never swapped for an unrelated accessory) that also
    passes the existing source-policy and occasion rules.
    """
    required = [r for r in (required_roles or []) if r in EXPLICIT_ROLES]
    if not isinstance(card, dict):
        return None, "missing_explicit_roles", required
    if not required:
        return card, "satisfied", []

    items = [i for i in (card.get("items") or []) if isinstance(i, dict)]
    missing = missing_explicit_roles(items, required)
    if not missing:
        return card, "satisfied", []

    pool = [c for c in (candidate_pool or []) if isinstance(c, dict)]
    used = {_item_key(i) for i in items}
    repaired = list(items)
    for role in missing:
        picked = None
        for cand in pool:
            if item_explicit_role(cand) != role:
                continue
            if _item_key(cand) in used:
                continue
            if not _source_ok(cand, source_policy):
                continue
            if not _candidate_safe_for_occasion(cand, occasion):
                continue
            picked = cand
            break
        if picked is None:
            still = missing_explicit_roles(repaired, required)
            return None, "missing_explicit_roles", still
        repaired.append(picked)
        used.add(_item_key(picked))

    still = missing_explicit_roles(repaired, required)
    if still:
        return None, "missing_explicit_roles", still
    out = dict(card)
    out["items"] = repaired
    out["item_count"] = len(repaired)
    return out, "repaired", []


__all__ = [
    "EXPLICIT_ROLES",
    "extract_requested_roles",
    "item_explicit_role",
    "board_explicit_roles",
    "missing_explicit_roles",
    "enforce_explicit_roles",
]
