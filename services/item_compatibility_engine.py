"""
services/item_compatibility_engine.py
Compatibility Intelligence Engine V1 for AHVI 'Works Well With'.

Determines the strongest complementary wardrobe pieces for an anchor item based on:
1. Canonical role detection & complement matrix
2. Hard incompatibility filters (same-item, deleted, extreme formality gap, occasion conflict, non-fashion, duplicates by ID/image/hash)
3. Calibrated compatibility scoring (neutral missing metadata, zero score inflation)
4. Disliked / Not-for-me penalties (-25.0)
5. Optimal role-diversity selection pass (MAX_PER_ROLE = 2)
6. Response minimization (clean contract without raw item payloads)
7. Explainability & reason codes
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ahvi.services.item_compatibility")

MAX_PER_ROLE = 2
MIN_COMPATIBILITY_SCORE = 55.0
DEFAULT_MAX_RESULTS = 5

# Non-fashion tokens requiring exact word boundary matching so "box" does NOT reject "boxy shirt"
_NON_FASHION_EXACT_PATTERNS = [
    r'\bcharger\b', r'\bcable\b', r'\badapter\b', r'\bbottle\b', r'\bphone\b',
    r'\bremote\b', r'\bmouse\b', r'\bkeyboard\b', r'\blaptop\b', r'\bearbud\b',
    r'\bheadphone\b', r'\bairpod\b', r'\bpower\s*bank\b', r'\bplug\b', r'\bwire\b',
    r'\bbattery\b', r'\bspeaker\b', r'\bcamera\b', r'\bmug\b', r'\bcup\b',
    r'\bpen\b', r'\bbook\b', r'\bbox\b', r'\bmystery\s*object\b', r'\blaptop\s*charger\b',
    r'\bmystery\b', r'\bobject\b'
]

# Sport / swim items that should not be paired into general fashion looks.
_SPORT_SWIM_PATTERNS = [
    r'\bgoggle\b', r'\bwetsuit\b', r'\bsnorkel\b', r'\bflipper\b', r'\bcleat\b',
    r'\bshin\s*guard\b', r'\bhelmet\b', r'\bski\b', r'\bsnowboard\b',
    r'\blife\s*jacket\b', r'\blife\s*vest\b', r'\bboxing\s*glove\b'
]

# Formality level values
_FORMALITY_LEVELS = {
    "athletic": 0,
    "gym": 0,
    "sport": 0,
    "active": 0,
    "casual": 1,
    "daily": 1,
    "weekend": 1,
    "smart_casual": 2,
    "polished": 2,
    "workwear": 2,
    "business": 3,
    "formal": 3,
    "festive": 4,
    "ethnic_formal": 4,
    "wedding": 4,
}

PREFERRED_COMPLEMENTS = {
    "top": ["bottom", "footwear", "outerwear", "accessory", "bag"],
    "bottom": ["top", "footwear", "outerwear", "accessory", "bag"],
    "footwear": ["bottom", "top", "dress", "ethnicwear", "outerwear"],
    "outerwear": ["top", "bottom", "footwear", "accessory"],
    "dress": ["footwear", "outerwear", "bag", "accessory", "jewellery"],
    "ethnicwear": ["footwear", "jewellery", "bag", "outerwear", "bottom"],
    "accessory": ["top", "dress", "bottom", "footwear", "bag"],
    "bag": ["dress", "top", "bottom", "footwear", "accessory"],
    "jewellery": ["dress", "ethnicwear", "top", "footwear", "bag"],
}

# Explicit Classic Perfect Color Harmony Pairs (Score = 20.0)
_PERFECT_COLOR_HARMONY_PAIRS = {
    frozenset({"grey", "beige"}), frozenset({"grey", "black"}), frozenset({"grey", "navy"}),
    frozenset({"grey", "olive"}), frozenset({"grey", "brown"}), frozenset({"grey", "white"}),
    frozenset({"brown", "beige"}), frozenset({"brown", "cream"}), frozenset({"brown", "navy"}),
    frozenset({"brown", "olive"}), frozenset({"brown", "tan"}), frozenset({"brown", "white"}),
    frozenset({"white", "blue"}), frozenset({"white", "black"}), frozenset({"white", "navy"}),
    frozenset({"white", "grey"}), frozenset({"white", "beige"}), frozenset({"white", "cream"}),
    frozenset({"black", "cream"}), frozenset({"black", "grey"}), frozenset({"black", "white"}),
    frozenset({"black", "burgundy"}), frozenset({"black", "beige"}), frozenset({"black", "tan"}),
    frozenset({"olive", "cream"}), frozenset({"olive", "beige"}), frozenset({"olive", "white"}),
    frozenset({"olive", "navy"}), frozenset({"olive", "brown"}), frozenset({"navy", "beige"}),
    frozenset({"navy", "tan"}), frozenset({"navy", "cream"}), frozenset({"navy", "khaki"}),
}

_NEUTRAL_COLORS = {
    "white", "black", "grey", "gray", "cream", "beige", "navy", "tan", "khaki",
    "brown", "charcoal", "nude", "ivory", "off-white", "off white", "stone"
}
_EARTH_COLORS = {
    "olive", "brown", "taupe", "sand", "tan", "cream", "beige", "rust",
    "terracotta", "khaki", "chocolate", "espresso", "mocha", "coffee", "camel"
}
_COOL_COLORS = {"blue", "navy", "grey", "gray", "teal", "slate", "indigo"}
_WARM_COLORS = {"burgundy", "rust", "gold", "mustard", "maroon", "coral", "terracotta"}
_METALLIC_COLORS = {"gold", "silver", "metallic", "bronze", "copper"}


def _as_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v).lower().strip() for v in val if str(v or "").strip()]
    if isinstance(val, (tuple, set)):
        return [str(v).lower().strip() for v in val if str(v or "").strip()]
    if isinstance(val, str) and val.strip():
        return [str(v).lower().strip() for v in val.split(",") if str(v or "").strip()]
    return []


def _text(item: Dict[str, Any]) -> str:
    parts = [
        str(item.get(k) or "")
        for k in ("name", "title", "label", "category", "main_category", "sub_category", "subcategory", "type")
    ]
    raw_tags = item.get("tags") or item.get("style_tags") or []
    if isinstance(raw_tags, list):
        parts.extend([str(t) for t in raw_tags if t])
    return " ".join(parts).lower().strip()


def _is_non_fashion(blob: str) -> bool:
    """Token-safe matching: rejects 'laptop charger', 'cardboard box' but keeps 'boxy shirt' and 'box pleat skirt'."""
    low = blob.lower()
    for p in _NON_FASHION_EXACT_PATTERNS:
        if p == r'\bbox\b' and any(f in low for f in ("shirt", "skirt", "pant", "trouser", "dress", "top", "pleat", "pleated", "tailored")):
            continue
        if re.search(p, low):
            return True
    return False


def _is_sport_swim(blob: str) -> bool:
    for p in _SPORT_SWIM_PATTERNS:
        if re.search(p, blob):
            return True
    return False


def canonical_role(item: Dict[str, Any]) -> str:
    """Detect canonical role for item."""
    if not isinstance(item, dict):
        return "unknown"
    
    assigned = str(item.get("role") or "").lower().strip()
    if assigned in PREFERRED_COMPLEMENTS:
        return assigned

    blob = _text(item)
    if not blob:
        return "unknown"

    if _is_non_fashion(blob):
        return "unknown"
    if _is_sport_swim(blob):
        return "unknown"

    if any(t in blob for t in ("saree", "sari", "lehenga", "sherwani", "kurta", "kurti", "anarkali", "dupatta", "indo-western", "ethnicwear", "ethnic")):
        return "ethnicwear"
    if any(t in blob for t in ("dress", "gown", "frock", "jumpsuit", "one-piece", "one piece", "kaftan")):
        return "dress"
    if any(t in blob for t in ("jewellery", "jewelry", "ring", "necklace", "bracelet", "earring", "pendant", "bangle")):
        return "jewellery"
    if any(t in blob for t in ("bag", "tote", "clutch", "handbag", "purse", "backpack", "crossbody")):
        return "bag"
    if any(t in blob for t in ("watch", "belt", "sunglasses", "sunglass", "scarf", "hat", "cap", "accessory", "accessories")):
        return "accessory"
    if any(t in blob for t in ("shoe", "shoes", "sneaker", "sneakers", "loafer", "loafers", "boot", "boots", "sandal", "sandals", "heel", "heels", "flat", "flats", "jutti", "juttis", "footwear", "oxford", "derby", "mule", "pump")):
        return "footwear"
    if any(t in blob for t in ("bottom", "bottoms", "trouser", "trousers", "pant", "pants", "jean", "jeans", "chino", "chinos", "short", "shorts", "skirt", "skirts", "legging", "leggings")):
        return "bottom"
    if any(t in blob for t in ("outerwear", "blazer", "jacket", "coat", "overcoat", "cardigan", "trench", "suit")):
        return "outerwear"
    if any(t in blob for t in ("top", "tops", "shirt", "shirts", "tee", "tshirt", "t-shirt", "polo", "blouse", "hoodie", "sweater", "knit", "overshirt")):
        return "top"

    return "unknown"


def _item_id(item: Dict[str, Any]) -> str:
    return str(item.get("id") or item.get("item_id") or item.get("garment_id") or item.get("$id") or "").strip()


def _cand_dup_key(item: Dict[str, Any]) -> str:
    """Multi-key deduplication (Pixel/Content hash priority, Image URL fallback, Item ID fallback)."""
    pixel_hash = str(item.get("pixel_hash") or item.get("content_hash") or item.get("hash") or "").strip().lower()
    if pixel_hash:
        return f"hash:{pixel_hash}"
    img = (
        item.get("normalized_url") or item.get("normalizedUrl")
        or item.get("image_url") or item.get("imageUrl")
        or item.get("masked_url") or ""
    )
    img_clean = str(img).strip().lower()
    if img_clean:
        return f"img:{img_clean}"
    c_id = _item_id(item)
    return f"id:{c_id}" if c_id else ""


def _extract_image_url(item: Dict[str, Any]) -> str:
    return str(
        item.get("normalized_url") or item.get("normalizedUrl")
        or item.get("image_url") or item.get("imageUrl")
        or item.get("masked_url") or item.get("maskedUrl") or ""
    ).strip()


def _formality_val(item: Dict[str, Any]) -> int:
    f = str(item.get("formality") or "").lower().strip()
    if f in _FORMALITY_LEVELS:
        return _FORMALITY_LEVELS[f]
    
    blob = _text(item)
    if any(x in blob for x in ("athletic", "gym", "running", "sport", "workout")):
        return 0
    if any(x in blob for x in ("formal", "blazer", "suit", "oxford", "derby", "evening")):
        return 3
    if any(x in blob for x in ("festive", "kurta", "sherwani", "lehenga", "jutti", "wedding", "sangeet")):
        return 4
    if any(x in blob for x in ("smart", "loafer", "chino", "trouser", "polo")):
        return 2
    return 1


def _color_name(item: Dict[str, Any]) -> str:
    c = item.get("color") or item.get("colour") or item.get("color_name") or ""
    if isinstance(c, list) and c:
        return str(c[0]).lower().strip()
    return str(c).lower().strip()


def _is_hard_incompatible(
    anchor: Dict[str, Any],
    candidate: Dict[str, Any],
    anchor_role: str,
    cand_role: str,
    occasion: Optional[str] = None
) -> bool:
    if cand_role == "unknown" or anchor_role == "unknown":
        return True

    # Same item
    a_id = _item_id(anchor)
    c_id = _item_id(candidate)
    if a_id and c_id and a_id == c_id:
        return True

    # Deleted / inactive
    if candidate.get("deleted") is True or str(candidate.get("status", "")).lower() in {"deleted", "inactive"}:
        return True

    # Same role (except accessories/jewellery)
    if anchor_role == cand_role and anchor_role not in {"accessory", "jewellery"}:
        return True

    # Formality Conflict: gap >= 3 or athletic (0) vs smart_casual/formal/festive (>= 2)
    a_form = _formality_val(anchor)
    c_form = _formality_val(candidate)
    if abs(a_form - c_form) >= 3:
        return True
    if (a_form == 0 and c_form >= 2) or (c_form == 0 and a_form >= 2):
        return True

    # Occasion conflict. Handled via _as_list for strings or lists.
    avoid_list = [x.replace(" ", "_") for x in _as_list(candidate.get("avoid_for"))]
    if occasion:
        occ_norm = str(occasion).lower().strip().replace(" ", "_")
        if occ_norm in avoid_list:
            return True

    return False


def _score_color_harmony(color_a: str, color_b: str) -> float:
    """Structured color harmony score (0.0 if missing, strictly neutral)."""
    if not color_a or not color_b:
        return 0.0  # Missing metadata is NEUTRAL (no bonus score)

    ca = color_a.lower()
    cb = color_b.lower()

    if ca == cb:
        return 15.0

    pair = frozenset({ca, cb})
    if pair in _PERFECT_COLOR_HARMONY_PAIRS:
        return 20.0
    if ca in _NEUTRAL_COLORS or cb in _NEUTRAL_COLORS:
        return 17.0
    if ca in _EARTH_COLORS and cb in _EARTH_COLORS:
        return 18.0
    if ca in _COOL_COLORS and cb in _COOL_COLORS:
        return 17.0
    if ca in _WARM_COLORS and cb in _WARM_COLORS:
        return 17.0
    if ca in _METALLIC_COLORS or cb in _METALLIC_COLORS:
        return 20.0

    return 14.0


class ItemCompatibilityEngine:
    @classmethod
    def rank(
        cls,
        anchor: Dict[str, Any],
        wardrobe: List[Dict[str, Any]],
        occasion: Optional[str] = None,
        max_results: int = DEFAULT_MAX_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate wardrobe items for anchor item compatibility.
        Returns top 3-5 candidates formatted with item_id, role, match_score, match_level, reason_codes.
        """
        if not anchor or not wardrobe:
            return []

        # K4: Authoritative anchor resolution from backend wardrobe if anchor lacks full metadata
        aid = _item_id(anchor)
        if aid:
            for w in wardrobe:
                if isinstance(w, dict) and _item_id(w) == aid:
                    anchor = {**w, **anchor}
                    break

        anchor_role = canonical_role(anchor)
        if anchor_role == "unknown":
            return []

        preferred_roles = PREFERRED_COMPLEMENTS.get(anchor_role, [])
        a_form = _formality_val(anchor)
        color_a = _color_name(anchor)
        a_tags = set(_as_list(anchor.get("tags")) + _as_list(anchor.get("style_tags")) + _as_list(anchor.get("archetype")))
        a_seasons = set(_as_list(anchor.get("season")) + _as_list(anchor.get("season_affinity")))

        best_candidates_by_key: Dict[str, Dict[str, Any]] = {}

        for cand in wardrobe:
            if not isinstance(cand, dict):
                continue

            cand_role = canonical_role(cand)

            if _is_hard_incompatible(anchor, cand, anchor_role, cand_role, occasion=occasion):
                continue

            reasons = []

            # 1. Role Complement Score (0–30)
            role_score = 0.0
            if cand_role in preferred_roles:
                idx = preferred_roles.index(cand_role)
                role_score = 30.0 if idx < 2 else 25.0
                reasons.append("ROLE_COMPLEMENT")

            # 2. Structured Color Harmony Score (0–20, 0.0 if missing)
            color_b = _color_name(cand)
            color_score = _score_color_harmony(color_a, color_b)
            if color_score >= 17.0:
                reasons.append("COLOR_HARMONY")

            # 3. Formality Compatibility Score (0–15, 5.0 baseline)
            c_form = _formality_val(cand)
            form_diff = abs(a_form - c_form)
            if form_diff == 0:
                form_score = 15.0
                reasons.append("FORMALITY_MATCH")
            elif form_diff == 1:
                form_score = 10.0
                reasons.append("FORMALITY_MATCH")
            else:
                form_score = 5.0

            # 4. Occasion Compatibility (0–15, 0.0 if no match)
            occ_score = 0.0
            if occasion:
                occ_norm = str(occasion).lower().strip().replace(" ", "_")
                cand_occs = [str(o).lower().strip().replace(" ", "_") for o in _as_list(cand.get("occasions"))]
                if occ_norm in cand_occs:
                    occ_score = 15.0
                    reasons.append("OCCASION_MATCH")

            # 5. Tag / Archetype Style Match (0–10, 0.0 if missing)
            c_tags = set(_as_list(cand.get("tags")) + _as_list(cand.get("style_tags")) + _as_list(cand.get("archetype")))
            shared_tags = a_tags & c_tags
            if shared_tags:
                style_score = min(10.0, 5.0 + len(shared_tags) * 2.5)
                reasons.append("STYLE_MATCH")
            else:
                style_score = 0.0

            # 6. Material / Season Match (0–5, 0.0 if missing)
            c_seasons = set(_as_list(cand.get("season")) + _as_list(cand.get("season_affinity")))
            if a_seasons and c_seasons:
                if a_seasons & c_seasons or "all_season" in a_seasons or "all_season" in c_seasons:
                    mat_score = 5.0
                    reasons.append("SEASON_MATCH")
                else:
                    mat_score = 1.0
            else:
                mat_score = 0.0

            # 7. User Affinity (0–5, 0.0 if default)
            user_score = 5.0 if cand.get("liked") or cand.get("isLiked") else 0.0

            # Penalties
            penalty = 0.0
            if form_diff == 2:
                penalty += 15.0

            # Disliked / Not-for-me Penalty (-25.0)
            if (
                cand.get("disliked") is True
                or cand.get("not_for_me") is True
                or cand.get("isDisliked") is True
                or str(cand.get("status", "")).lower() == "disliked"
                or "disliked" in [str(t).lower() for t in (cand.get("tags") or [])]
            ):
                penalty += 25.0

            raw_score = (role_score + color_score + form_score + occ_score + style_score + mat_score + user_score) - penalty
            final_score = max(0.0, min(100.0, raw_score))

            # Threshold Filter: Must reach >= 55.0 to be eligible
            if final_score < MIN_COMPATIBILITY_SCORE:
                continue

            if final_score >= 85.0:
                level = "Excellent match"
            elif final_score >= 70.0:
                level = "Strong match"
            else:
                level = "Possible match"

            # K8 Response Minimization — bounded response payload
            candidate_payload = {
                "item_id": _item_id(cand),
                "name": str(cand.get("name") or cand.get("label") or "Item"),
                "category": str(cand.get("category") or cand.get("main_category") or ""),
                "role": cand_role,
                "image_url": _extract_image_url(cand),
                "match_score": int(round(final_score)),
                "match_level": level,
                "reason_codes": list(dict.fromkeys(reasons)),
            }

            # K7 Deduplication keeping HIGHEST scoring duplicate candidate
            dup_key = _cand_dup_key(cand)
            if dup_key:
                if dup_key not in best_candidates_by_key or candidate_payload["match_score"] > best_candidates_by_key[dup_key]["match_score"]:
                    best_candidates_by_key[dup_key] = candidate_payload

        scored_candidates = list(best_candidates_by_key.values())
        if not scored_candidates:
            return []

        # --- K9 Enhanced Role-Diversity Selection Pass (MAX_PER_ROLE = 2) ---
        by_role: Dict[str, List[Dict[str, Any]]] = {}
        for cand in sorted(scored_candidates, key=lambda x: x["match_score"], reverse=True):
            by_role.setdefault(cand["role"], []).append(cand)

        reranked = []
        role_counts: Dict[str, int] = {}

        # First pass: Pick 1 highest-scoring candidate from each preferred role in preference order
        for p_role in preferred_roles:
            if p_role in by_role and by_role[p_role]:
                pick = by_role[p_role].pop(0)
                reranked.append(pick)
                role_counts[p_role] = 1
                if len(reranked) >= max_results:
                    break

        # Second pass: Fill remaining slots up to MAX_PER_ROLE = 2 from remaining candidates sorted by score
        if len(reranked) < max_results:
            remaining = []
            for r_list in by_role.values():
                remaining.extend(r_list)
            remaining.sort(key=lambda x: x["match_score"], reverse=True)
            for cand in remaining:
                r = cand["role"]
                if role_counts.get(r, 0) < MAX_PER_ROLE:
                    reranked.append(cand)
                    role_counts[r] = role_counts.get(r, 0) + 1
                if len(reranked) >= max_results:
                    break

        return reranked[:max_results]
