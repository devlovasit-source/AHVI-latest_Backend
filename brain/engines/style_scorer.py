import logging
import re
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger("ahvi.style_scorer")

from services.wardrobe_intelligence_service import (
    _style_meta,
    normalize_occasion as normalize_style_occasion,
    score_item_for_occasion,
)


# ============================================================
# FINAL STYLE SAFETY GATE — PER-OCCASION CONFIDENCE THRESHOLDS
# ============================================================
# Used by the response assembly layer to decide whether an outfit is
# strong enough to show as a confident recommendation. Below the
# threshold we emit a weak_match response and let the user pick:
# "Show closest option", "Try another occasion", etc.

OCCASION_CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    # High-stakes occasions need a stronger fit.
    "office": 0.72,
    "business": 0.72,
    "interview": 0.72,
    "date": 0.72,
    "date_night": 0.72,
    "wedding": 0.72,
    "festive": 0.72,

    # Forgiving day-to-day occasions tolerate a softer match.
    "casual": 0.62,
    "travel": 0.62,
    "airport": 0.62,

    # Mid bar.
    "beach": 0.65,
    "party": 0.65,
    "gym": 0.65,
    "workout": 0.65,
    "dinner": 0.65,
}

_DEFAULT_CONFIDENCE_THRESHOLD = 0.68


def occasion_confidence_threshold(occasion: Any) -> float:
    """Return the min occasion_compatibility_score we'll show as
    a confident recommendation. Used by the safety gate.
    """
    key = str(occasion or "").strip().lower().replace("-", "_")
    if key in OCCASION_CONFIDENCE_THRESHOLDS:
        return OCCASION_CONFIDENCE_THRESHOLDS[key]
    normalized = normalize_occasion(key) if key else ""
    if normalized in OCCASION_CONFIDENCE_THRESHOLDS:
        return OCCASION_CONFIDENCE_THRESHOLDS[normalized]
    return _DEFAULT_CONFIDENCE_THRESHOLD


# ============================================================
# GLOBAL OCCASION COMPATIBILITY MATRIX
# ============================================================
# One source of truth for "what reads right for this occasion".
# Consumed by score_occasion_compatibility() and the quality guard.
# Per-occasion lists are intentionally short and high-signal so a
# single matching token can move the score decisively. All terms are
# lowercase substring matches against the item/outfit blob.

OCCASION_COMPATIBILITY_RULES: Dict[str, Dict[str, Any]] = {
    "beach": {
        "boost_terms": [
            "linen", "cotton", "breathable", "relaxed", "shorts", "sandals",
            "slides", "espadrille", "espadrilles", "camp collar", "tank",
            "tote", "open collar", "resort", "white", "pastel", "sky blue",
            "swim", "boardshort", "tropical", "hawaiian", "floral",
            "printed", "patterned", "resort print", "vacation print",
            "open shirt", "short sleeve shirt",
        ],
        "hard_penalty_terms": [
            "satin", "shiny", "glossy", "sequin", "formal shirt",
            "dress shirt", "blazer", "wool", "leather shoes", "oxford",
            "derby", "monk strap", "dress pants", "business",
        ],
        "reject_if_terms": [
            "blazer", "wool", "tuxedo", "formal suit",
        ],
        "preferred_formality": "casual",
        "min_required_score": 0.45,
    },
    "office": {
        "boost_terms": [
            "plain shirt", "muted", "button", "trouser", "chino", "loafer",
            "leather sneaker", "minimal sneaker", "blazer", "structured",
            "smart casual", "navy", "grey", "white", "black", "watch",
            "belt", "minimal", "clean",
        ],
        "hard_penalty_terms": [
            "flip flop", "flip-flop", "swimwear", "gym shorts", "running shorts",
            "shiny", "satin", "glossy", "sequin", "loud print",
            "loud pattern", "festive", "embroidered", "party shoes",
            "red loafers", "neon", "bright orange", "bright yellow",
            "burgundy loafers", "red shoes", "ornate", "party shirt",
            "tropical", "hawaiian", "floral print", "beachwear", "tank",
            "crop top", "shorts", "open shirt",
        ],
        "reject_if_terms": [
            "swimwear", "swim trunks", "flip flop", "flip-flop",
        ],
        "preferred_formality": "smart_casual",
        "min_required_score": 0.5,
    },
    "date": {
        "boost_terms": [
            "polished", "fitted", "intentional", "loafer", "chelsea boot",
            "watch", "leather sneaker", "knit", "evening", "minimal",
            "burgundy", "cream", "navy",
        ],
        "hard_penalty_terms": [
            "gym", "track", "athletic", "running", "flip flop",
            "oversized hoodie", "cargo", "boardroom", "client",
            "professional", "corporate", "shorts", "running shorts",
            "gym shorts", "sliders", "slides", "slippers", "flip-flop",
        ],
        "reject_if_terms": [],
        "preferred_formality": "smart_casual",
        "min_required_score": 0.45,
    },
    "party": {
        "boost_terms": [
            "print", "pattern", "shirt", "dress", "boots", "leather sneaker",
            "statement", "black", "burgundy", "silver", "gold accent",
        ],
        "hard_penalty_terms": [
            "office", "corporate", "workwear", "gym", "running shoes",
        ],
        "reject_if_terms": [],
        "preferred_formality": "social",
        "min_required_score": 0.45,
    },
    "gym": {
        "boost_terms": [
            "performance", "stretch", "breathable", "training", "sneaker",
            "running shoe", "athletic", "tech", "moisture wicking",
            "joggers", "shorts", "tee", "tank",
        ],
        "hard_penalty_terms": [
            "jeans", "denim", "formal shirt", "blazer", "loafer",
            "leather", "dress shoes", "jewelry",
        ],
        "reject_if_terms": [
            "blazer", "tuxedo", "formal suit",
        ],
        "preferred_formality": "athletic",
        "min_required_score": 0.55,
    },
    "workout": {
        "boost_terms": [
            "performance", "stretch", "breathable", "training", "sneaker",
            "running shoe", "athletic", "tech", "joggers", "shorts", "tee",
        ],
        "hard_penalty_terms": [
            "jeans", "denim", "formal shirt", "blazer", "loafer",
            "leather", "dress shoes",
        ],
        "reject_if_terms": [],
        "preferred_formality": "athletic",
        "min_required_score": 0.55,
    },
    "wedding": {
        "boost_terms": [
            "kurta", "sherwani", "saree", "lehenga", "suit", "blazer",
            "dress shoes", "oxford", "loafer", "silk", "satin",
            "embroidered", "festive", "elevated",
        ],
        "hard_penalty_terms": [
            "basic tee", "running shoes", "gym", "shorts", "beach", "tank",
            "flip flop", "flip-flop",
        ],
        "reject_if_terms": [],
        "preferred_formality": "formal",
        "min_required_score": 0.55,
    },
    "festive": {
        "boost_terms": [
            "embroidered", "silk", "satin", "elevated", "kurta",
            "saree", "dress shoes", "ethnic",
        ],
        "hard_penalty_terms": [
            "gym", "running shoes", "athletic", "track pants",
        ],
        "reject_if_terms": [],
        "preferred_formality": "smart",
        "min_required_score": 0.5,
    },
    "dinner": {
        "boost_terms": [
            "polished", "shirt", "trouser", "loafer", "leather sneaker",
            "blazer", "knit",
        ],
        "hard_penalty_terms": [
            "gym shorts", "flip flop", "tank", "athletic shoes",
        ],
        "reject_if_terms": [],
        "preferred_formality": "smart_casual",
        "min_required_score": 0.45,
    },
    "travel": {
        "boost_terms": [
            "comfort", "layers", "sneaker", "breathable", "stretch",
            "wrinkle-resistant", "tee", "joggers", "tote", "backpack",
            "hoodie", "easy",
        ],
        "hard_penalty_terms": [
            "stiff", "tuxedo", "tight formal", "patent leather", "stilettos",
            "very heavy formalwear",
        ],
        "reject_if_terms": [],
        "preferred_formality": "casual",
        "min_required_score": 0.45,
    },
    "airport": {
        "boost_terms": [
            "comfort", "layers", "sneaker", "breathable", "stretch",
            "joggers", "hoodie", "tee", "tote",
        ],
        "hard_penalty_terms": [
            "stilettos", "heels", "tuxedo", "patent leather", "stiff formal",
        ],
        "reject_if_terms": [],
        "preferred_formality": "casual",
        "min_required_score": 0.45,
    },
    "business": {
        "boost_terms": [
            "blazer", "suit", "dress shirt", "trouser", "loafer", "oxford",
            "derby", "watch", "tie",
        ],
        "hard_penalty_terms": [
            "flip flop", "gym shorts", "tank", "beach", "running shoes",
            "swim", "shorts",
        ],
        "reject_if_terms": [
            "swim", "flip flop",
        ],
        "preferred_formality": "formal",
        "min_required_score": 0.5,
    },
    "casual": {
        "boost_terms": [
            "tee", "shirt", "jeans", "chino", "sneaker", "hoodie",
            "relaxed", "easy",
        ],
        "hard_penalty_terms": [
            "tuxedo", "ball gown", "formal evening",
        ],
        "reject_if_terms": [],
        "preferred_formality": "casual",
        "min_required_score": 0.4,
    },
    "rain": {
        "boost_terms": [
            "waterproof", "boot", "chelsea boot", "leather boot", "jacket",
            "rain", "synthetic", "dark sneaker",
        ],
        "hard_penalty_terms": [
            "suede", "canvas", "white sneaker", "cream sneaker", "slipper",
            "sandals", "open shoe",
        ],
        "reject_if_terms": [],
        "preferred_formality": None,
        "min_required_score": 0.4,
    },
    "winter": {
        "boost_terms": [
            "wool", "cashmere", "knit", "coat", "boot", "layered",
            "scarf", "thermal",
        ],
        "hard_penalty_terms": [
            "tank", "shorts", "linen", "open sandal", "swim",
        ],
        "reject_if_terms": [],
        "preferred_formality": None,
        "min_required_score": 0.4,
    },
    "summer": {
        "boost_terms": [
            "linen", "cotton", "breathable", "shorts", "tee", "sandal",
            "white", "pastel", "light",
        ],
        "hard_penalty_terms": [
            "wool", "heavy coat", "thermal", "fur",
        ],
        "reject_if_terms": [],
        "preferred_formality": None,
        "min_required_score": 0.4,
    },
}


# Alias normalization for prompts the FE / orchestrator may pass.
_OCCASION_ALIASES: Dict[str, str] = {
    "casual beach walk": "beach",
    "beach vacation": "beach",
    "beach day": "beach",
    "poolside": "beach",
    "poolside day": "beach",
    "resort dinner": "beach",
    "client meeting": "office",
    "office meeting": "office",
    "casual office day": "office",
    "office wear": "office",
    "boardroom": "office",
    "corporate": "office",
    "business meeting": "business",
    "date night": "date",
    "dinner date": "date",
    "coffee date": "date",
    "movie date": "date",
    "date_night": "date",
    "date night": "date",
    "house_party": "party",
    "house party": "party",
    "rave": "party",
    "rave party": "party",
    "club night": "party",
    "night out": "party",
    "dinner party": "party",
    "capsule wardrobe": "capsule",
    "wardrobe essentials": "capsule",
    "core wardrobe": "capsule",
    "minimalist wardrobe": "capsule",
    "airport outfit": "airport",
    "airport look": "airport",
    "road trip": "travel",
    "day trip": "travel",
    "overnight trip": "travel",
    "strength training": "workout",
    "cardio": "workout",
    "yoga": "workout",
    "gym": "workout",
    "indian wedding": "wedding",
    "western wedding": "wedding",
    "reception": "wedding",
}


def _resolve_occasion(context: Dict[str, Any]) -> str:
    """Pick an occasion from any of the keys the orchestrator may send."""
    candidates = [
        context.get("occasion"),
        context.get("interpreted_occasion"),
        context.get("occasion_slug"),
        context.get("event_type"),
        context.get("style_intent"),
        context.get("resolved_prompt"),
        context.get("prompt"),
    ]
    for raw in candidates:
        if not raw:
            continue
        text = str(raw).strip().lower()
        if not text:
            continue
        # Try alias first (more specific phrases win).
        for alias, target in _OCCASION_ALIASES.items():
            if alias in text:
                return target
        normalized = normalize_occasion(text)
        if normalized:
            return normalized
    return ""


def resolve_occasion_profile(occasion: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return normalized occasion rules used by scoring and guard layers."""
    ctx = dict(context or {})
    if occasion:
        ctx.setdefault("occasion", occasion)
    resolved = _resolve_occasion(ctx)
    rules = OCCASION_COMPATIBILITY_RULES.get(resolved, {})
    logger.info(
        "occasion_profile.selected occasion=%s has_rules=%s preferred_formality=%s",
        resolved,
        bool(rules),
        rules.get("preferred_formality"),
    )
    return {
        "occasion": resolved,
        "preferred_formality": rules.get("preferred_formality"),
        "min_required_score": float(rules.get("min_required_score") or 0.45),
        "confidence_threshold": occasion_confidence_threshold(resolved),
        "has_rules": bool(rules),
    }


def _item_ids(items: List[Dict[str, Any]]) -> set:
    out = set()
    for it in items or []:
        if isinstance(it, dict):
            v = str(it.get("$id") or it.get("id") or it.get("item_id") or it.get("name") or "").strip().lower()
            if v:
                out.add(v)
    return out


def _memory_breakdown(
    items: List[Dict[str, Any]], context: Dict[str, Any]
) -> "Tuple[Dict[str, float], List[str]]":
    """Explicit wear-history ranking signals. Reads optional context keys
    (recently_worn_ids, underworn_ids, saved_item_ids, wear_counts); returns
    neutral 0.0 fields + reasons. Never errors on missing data."""
    ctx = context if isinstance(context, dict) else {}
    ids = _item_ids(items)
    # These 3 sum into the total score (line ~1094 sums breakdown.values()).
    fields: Dict[str, float] = {
        "saved_board_affinity": 0.0,
        "recent_repeat_penalty": 0.0,
        "underworn_boost": 0.0,
    }
    reasons: List[str] = []
    if not ids:
        return fields, reasons

    def _idset(key: str) -> set:
        return {str(x).strip().lower() for x in (ctx.get(key) or []) if str(x).strip()}

    recent = _idset("recently_worn_ids")
    underworn = _idset("underworn_ids")
    saved = _idset("saved_item_ids")

    if recent and ids & recent:
        fields["recent_repeat_penalty"] = -1.5 * len(ids & recent)
        reasons.append("recently worn — offering a fresher option")
    if underworn and ids & underworn:
        fields["underworn_boost"] = 1.2 * len(ids & underworn)
        reasons.append("brings an under-worn piece back into rotation")
    if saved and ids & saved:
        fields["saved_board_affinity"] = 1.0 * len(ids & saved)
        reasons.append("echoes a look you saved")
    # memory_freshness is a display-only composite — NOT summed into the score.
    freshness = round(fields["underworn_boost"] + fields["recent_repeat_penalty"], 3)
    if any(v for v in fields.values()):
        logger.info(
            "AHVI_MEMORY_SCORER_APPLIED repeat=%.1f underworn=%.1f saved=%.1f freshness=%.1f",
            fields["recent_repeat_penalty"], fields["underworn_boost"],
            fields["saved_board_affinity"], freshness,
        )
    return fields, reasons


def _outfit_blob(outfit: Dict[str, Any]) -> str:
    """Flatten outfit + every item into a single lower-case blob."""
    parts: List[str] = []
    for key in (
        "title",
        "vibe",
        "aesthetic",
        "style_direction",
        "explanation",
        "occasion",
        "intent",
        "use_case",
        "scenario",
    ):
        val = outfit.get(key)
        if val:
            parts.append(str(val))
    for slot in ("master_piece", "top", "bottom", "dress", "shoes", "footwear", "outerwear"):
        item = outfit.get(slot)
        if isinstance(item, dict):
            parts.append(_item_blob(item))
    for item in outfit.get("items") or []:
        if not isinstance(item, dict):
            continue
        parts.append(_item_blob(item))
    return " ".join(parts).lower()


def score_occasion_compatibility(
    outfit: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """Score how well an outfit fits the inferred occasion.

    Returns:
      {
        "score":  float in [0, 1]  (1 == great fit)
        "raw_score": float          (signed, can be negative)
        "boosts":   [str]
        "penalties":[str]
        "reject":   bool
        "reason":   str
        "occasion": str
      }
    """
    profile = resolve_occasion_profile(None, context)
    occasion = str(profile.get("occasion") or "")
    if not occasion or occasion not in OCCASION_COMPATIBILITY_RULES:
        # Unknown occasion → don't penalize, but report neutral.
        return {
            "score": 0.5,
            "raw_score": 0.0,
            "boosts": [],
            "penalties": [],
            "reject": False,
            "reason": "occasion_unknown" if not occasion else f"no_rules_for_{occasion}",
            "occasion": occasion,
            "profile": profile,
            "min_required_score": float(profile.get("min_required_score") or 0.45),
        }

    rules = OCCASION_COMPATIBILITY_RULES[occasion]
    blob = _outfit_blob(outfit)

    boosts: List[str] = []
    penalties: List[str] = []
    raw = 0.0

    for term in rules.get("boost_terms", []):
        if term and term in blob:
            boosts.append(term)
            raw += 1.0
    for term in rules.get("hard_penalty_terms", []):
        if term and term in blob:
            penalties.append(term)
            raw -= 1.5

    # Reject when a hard reject term is present AND no boost balances it.
    reject = False
    for term in rules.get("reject_if_terms", []):
        if term and term in blob:
            reject = True
            penalties.append(f"reject_term:{term}")
            break

    # Normalize to [0, 1] using a soft tanh-like curve: each ±1 point
    # nudges score ~0.1 around the 0.5 midpoint.
    soft = 0.5 + 0.1 * raw
    score = max(0.0, min(1.0, soft))

    min_required = float(rules.get("min_required_score") or 0.45)
    reason = ""
    if reject:
        reason = f"hard_reject:{penalties[-1] if penalties else 'forbidden_term'}"
    elif score < min_required:
        reason = f"below_min:{score:.2f}<{min_required:.2f}"
    elif boosts:
        reason = f"fits:{boosts[0]}"

    return {
        "score": score,
        "raw_score": raw,
        "boosts": boosts,
        "penalties": penalties,
        "reject": reject or score < min_required - 0.15,
        "reason": reason,
        "occasion": occasion,
        "profile": profile,
        "min_required_score": min_required,
    }


def score_weather_compatibility(
    outfit: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """Score weather realism using the same compatibility engine."""
    text = str(
        (context or {}).get("weather")
        or (context or {}).get("condition")
        or (context or {}).get("season")
        or ""
    ).strip().lower()
    weather_key = ""
    if any(w in text for w in ("rain", "monsoon", "wet")):
        weather_key = "rain"
    elif any(w in text for w in ("winter", "cold", "chilly")):
        weather_key = "winter"
    elif any(w in text for w in ("summer", "hot", "humid", "warm")):
        weather_key = "summer"
    if not weather_key:
        return {
            "score": 0.5,
            "raw_score": 0.0,
            "boosts": [],
            "penalties": [],
            "reject": False,
            "reason": "weather_unknown",
            "weather": text,
        }
    result = score_occasion_compatibility(outfit, {"occasion": weather_key})
    return {
        **result,
        "weather": weather_key,
        "reason": result.get("reason") or f"weather:{weather_key}",
    }




from brain.engines.color_normalizer import color_normalizer
from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.styling.palette_engine import palette_engine
from services.embedding_service import encode_metadata

try:
    from brain.engines.memory_scorer import memory_scorer
except Exception:  # pragma: no cover - optional Qdrant/dependency path
    memory_scorer = None


def _item_blob(item: dict) -> str:
    return " ".join(
        str(item.get(k, "") or "")
        for k in [
            "name",
            "category",
            "subcategory",
            "sub_category",
            "color",
            "material",
            "fabric",
            "pattern",
            "style_tags",
            "details",
        ]
    ).lower()


def normalize_occasion(occasion: Any) -> str:
    meta_occasion = normalize_style_occasion(occasion)
    text = str(occasion or "").strip().lower().replace("-", "_")
    readable = text.replace("_", " ")
    tokens = set(re.sub(r"[^a-z0-9]+", " ", readable).split())
    if any(w in readable for w in ["capsule wardrobe", "wardrobe essentials", "core wardrobe", "minimalist wardrobe"]):
        return "capsule"
    if (
        any(w in readable for w in ["workout", "gym", "fitness", "strength training"])
        or tokens.intersection({"training", "yoga", "running", "cardio", "pilates"})
    ):
        return "workout"
    if "night out" in readable and not any(w in readable for w in ["date", "dinner"]):
        return "party"
    if any(w in text for w in ["date_night", "date night", "date", "dinner", "tonight"]):
        return "date_night"
    if any(w in text for w in ["beach", "pool", "seaside", "coastal", "resort"]):
        return "beach"
    if (
        any(w in text for w in ["office", "corporate_office", "smart_casual_office", "meeting", "client", "boardroom"])
        or "work" in tokens
    ):
        return "office"
    if "brunch" in text:
        return "brunch"
    if "rave" in text or "club" in text:
        return "party"
    if "cocktail" in text:
        return "cocktail"
    if any(w in text for w in ["party", "house_party", "after_hours", "night out"]):
        return "party"
    if any(w in text for w in ["travel", "airport", "flight", "vacation", "trip"]):
        return "travel"
    if any(w in text for w in ["wedding", "reception", "ceremony", "event"]):
        return "wedding"
    if any(w in text for w in ["casual", "daily", "today", "weekend"]):
        return "casual"
    return meta_occasion or text


def occasion_item_score(item: dict, occasion: str) -> float:
    blob = _item_blob(item)
    score = float(score_item_for_occasion(item, occasion))
    occasion = normalize_occasion(occasion)

    if occasion == "date_night":
        if any(w in blob for w in ["boardroom", "office", "corporate", "workwear"]):
            score -= 0.45
        if any(w in blob for w in ["statement", "print", "watch", "loafers", "evening"]):
            score += 0.18
        if any(w in blob for w in ["slipper", "gym", "running", "shorts"]):
            score -= 0.55

    elif occasion == "beach":
        if any(
            w in blob
            for w in [
                "linen",
                "cotton",
                "shorts",
                "sandals",
                "slides",
                "sunglasses",
                "tote",
                "camp collar",
                "tropical",
                "hawaiian",
                "floral",
                "printed",
                "patterned",
                "resort print",
                "vacation print",
                "open shirt",
            ]
        ):
            score += 0.45
        if any(w in blob for w in ["black trousers", "loafers", "dress shoes", "blazer", "formal", "office"]):
            score -= 0.65

    elif occasion == "office":
        if any(w in blob for w in ["shirt", "trousers", "loafers", "belt", "watch", "blazer"]):
            score += 0.20
        if any(
            w in blob
            for w in [
                "slipper", "slides", "beach", "gym shorts", "running shorts",
                "red loafers", "burgundy loafers", "red shoes", "embroidered",
                "festive", "party shirt", "tropical", "hawaiian", "floral",
                "loud print", "loud pattern", "satin", "shiny", "glossy",
            ]
        ):
            score -= 0.65

    return score


def _ahvi_agent_item_blob(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return " ".join(
        str(item.get(k) or "").lower()
        for k in (
            "name",
            "title",
            "label",
            "category",
            "sub_category",
            "subcategory",
            "type",
            "slot",
            "role",
            "material",
            "fabric",
            "color",
            "dominant_color",
            "style",
        )
    )


_AHVI_FORMALITY_TO_INT = {
    "very_low": 1,
    "low": 2,
    "mid": 3,
    "high": 4,
    "very_high": 5,
    "formal": 5,
    "casual": 2,
}


def _ahvi_formality_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        pass
    return _AHVI_FORMALITY_TO_INT.get(str(value).strip().lower())


def _ahvi_agent_score_adjustment(
    items: List[Dict[str, Any]], agent: Dict[str, Any]
) -> tuple[float, List[str]]:
    """Return (delta, reasons) for agent-provided style signals.

    The delta is intentionally small (bounded ±~3) so the agent shapes
    ranking without overwhelming the existing scoring breakdown.
    """
    delta = 0.0
    reasons: List[str] = []
    if not items or not isinstance(agent, dict):
        return 0.0, reasons

    style_direction = str(agent.get("style_direction") or "").strip().lower()
    avoid_items = [str(t).strip().lower() for t in (agent.get("avoid_items") or []) if str(t).strip()]
    palette_direction = [str(t).strip().lower() for t in (agent.get("palette_direction") or []) if str(t).strip()]
    agent_formality = _ahvi_formality_int(agent.get("formality"))

    accessory_blobs: List[str] = []
    accessory_types: List[str] = []
    for item in items:
        blob = _ahvi_agent_item_blob(item)
        if not blob:
            continue
        if style_direction and style_direction.replace("_", " ") in blob.replace("_", " "):
            delta += 0.4
            reasons.append("agent style_direction match")
        if any(term in blob for term in avoid_items):
            delta -= 1.5
            reasons.append("agent avoid_items penalty")
        if palette_direction:
            color = str(item.get("color") or item.get("dominant_color") or "").lower()
            if color and any(p in color for p in palette_direction):
                delta += 0.3
                reasons.append("agent palette aligned")
        if agent_formality is not None:
            item_form = _ahvi_formality_int(item.get("formality"))
            if item_form is not None and abs(item_form - agent_formality) >= 2:
                delta -= 0.6
                reasons.append("agent formality mismatch")
        category = str(
            item.get("category") or item.get("sub_category") or item.get("type") or ""
        ).lower()
        if any(k in category for k in ("watch", "belt", "ring", "necklace", "bracelet", "bag", "cap", "hat", "scarf", "sunglass")):
            accessory_blobs.append(blob)
            accessory_types.append(category)

    # Accessory duplication penalty (e.g. two watches, two belts).
    if len(accessory_types) != len(set(accessory_types)) and accessory_types:
        delta -= 0.8
        reasons.append("agent accessory duplication")

    # Bound the delta to avoid dominating the scoring breakdown.
    if delta > 3.0:
        delta = 3.0
    if delta < -3.0:
        delta = -3.0
    return delta, reasons


class UnifiedStyleScorer:
    """
    Single scoring authority for outfit candidates.

    This scorer is intentionally read-only: it scores the supplied items and
    returns reasons/breakdown/warnings, but never swaps or mutates garments.
    """

    def score_outfit(
        self,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not items:
            return {
                "score": 0.0,
                "label": "Weak",
                "reasons": [],
                "breakdown": {},
                "rejection_warnings": ["missing outfit items"],
            }

        context = context or {}
        style_dna = context.get("style_dna", {}) or {}
        refinement = context.get("refinement")
        session = context.get("session", {}).get("derived", {})

        confidence = float(style_dna.get("confidence", 0.5) or 0.5)
        exploration_factor = max(0.0, 1.0 - confidence)

        reasons: List[str] = []
        warnings: List[str] = []
        breakdown = {
            "style_graph": 0.0,
            "palette": 0.0,
            "rules": 0.0,
            "aesthetic_balance": 0.0,
            "style_dna": 0.0,
            "memory": 0.0,
            "session": 0.0,
            "refinement": 0.0,
            "exploration": 0.0,
            "footwear_occasion": 0.0,
            "occasion_item": 0.0,
            "occasion_compatibility": 0.0,
            "weather_compatibility": 0.0,
            "metadata_richness": 0.0,
        }

        rules = style_engine.get_scoring_rules(style_dna, context)
        palette = palette_engine.select_palette(
            {
                "event": context.get("occasion"),
                "microtheme": style_dna.get("primary_aesthetic"),
            }
        )
        palette_colors = [
            color_normalizer.normalize(c) for c in palette.get("hex", []) if c
        ]

        graph_score = self._graph_score(items, graph or {})
        breakdown["style_graph"] = graph_score
        if graph_score > 1:
            # Internal-only reason. The generic 'items pair well' copy
            # leaked into editorial explanations and read as a
            # fake-positive when occasion score was actually weak.
            reasons.append("graph_pairing_strong")

        outfit_view = {"items": items}
        occasion_result = score_occasion_compatibility(outfit_view, context)
        weather_result = score_weather_compatibility(outfit_view, context)
        breakdown["occasion_compatibility"] = (
            (float(occasion_result.get("score") or 0.5) - 0.5) * 8.0
        )
        breakdown["weather_compatibility"] = (
            (float(weather_result.get("score") or 0.5) - 0.5) * 3.0
        )
        if occasion_result.get("boosts"):
            reasons.append(f"occasion_fit:{occasion_result['boosts'][0]}")
        if occasion_result.get("penalties"):
            warnings.extend(
                f"occasion_penalty:{p}" for p in occasion_result["penalties"][:3]
            )
        if occasion_result.get("reject"):
            warnings.append(f"occasion_reject:{occasion_result.get('reason') or 'low_fit'}")

        for item in items:
            color = color_normalizer.normalize(item.get("color") or item.get("color_code"))
            item_type = str(item.get("type") or item.get("sub_category") or "").lower()
            occasion_text = (
                context.get("occasion")
                or context.get("intent")
                or context.get("query")
                or context.get("user_query")
                or ""
            )
            breakdown["occasion_item"] += occasion_item_score(item, occasion_text)

            try:
                rule_delta, rule_reasons = style_engine.score_item_rule_adjustment(
                    item, rules
                )
                breakdown["rules"] += rule_delta
                reasons.extend(rule_reasons)
                if rule_delta <= -10:
                    warnings.extend(rule_reasons)
            except Exception:
                pass

            if color in palette_colors:
                breakdown["palette"] += 1.0
                reasons.append("palette aligned")
            elif self._is_neutral(color):
                breakdown["palette"] += 0.4

            if color in rules.get("preferred_colors", []):
                breakdown["palette"] += 0.6

            if item_type in rules.get("avoided_items", []):
                breakdown["rules"] -= 2.0
                reasons.append("conflicts with style")
                warnings.append(f"conflicts with style: {item_type}")

        footwear_delta, footwear_reasons, footwear_warnings = (
            self._footwear_occasion_score(items, context)
        )
        breakdown["footwear_occasion"] = footwear_delta
        reasons.extend(footwear_reasons)
        warnings.extend(footwear_warnings)

        aesthetic_score = self._aesthetic_score(items)
        breakdown["aesthetic_balance"] = aesthetic_score
        if aesthetic_score > 0.7:
            reasons.append("clean aesthetic balance")

        dna_score = self._dna_score(items, style_dna)
        breakdown["style_dna"] = dna_score * (0.5 + confidence)
        if dna_score > 0:
            reasons.append("matches your style")

        vector = self._build_outfit_embedding(items)
        if vector and memory_scorer is not None:
            try:
                memory_score = float(memory_scorer.score(vector, context) or 0.0)
            except Exception:
                memory_score = 0.0
            breakdown["memory"] = memory_score
            if memory_score > 0:
                reasons.append("aligned with your past choices")

        # Explicit wear-history signals (P1). Derived from context when the
        # caller supplies wear data; neutral (0.0) otherwise so a missing
        # history never distorts ranking.
        try:
            mem_fields, mem_reasons = _memory_breakdown(items, context)
        except Exception:
            mem_fields, mem_reasons = {}, []
        if mem_fields:
            breakdown.update(mem_fields)
            reasons.extend(mem_reasons)

        dominant = session.get("dominant_refinement")
        if dominant:
            breakdown["session"] = 0.6
            reasons.append(f"fits your current {dominant} preference")

        if refinement:
            refine_score = self._refinement_score(items, refinement)
            breakdown["refinement"] = refine_score
            if refine_score > 0:
                reasons.append(f"refined for {refinement}")

        breakdown["exploration"] = self._exploration_boost(items, exploration_factor)

        # ---- AHVI Style Orchestrator agent signals ----
        agent_payload = context.get("agent_orchestration") if isinstance(context, dict) else None
        if isinstance(agent_payload, dict):
            try:
                agent_delta, agent_reasons = _ahvi_agent_score_adjustment(items, agent_payload)
            except Exception:
                agent_delta, agent_reasons = 0.0, []
            if agent_delta:
                breakdown["agent_orchestration"] = round(float(agent_delta), 3)
                reasons.extend(agent_reasons)

        # ---- Metadata Validator richness signals as RANKING boosts ----
        # The quality guard already uses these to reject/penalize unsafe
        # combos. Surface them in the score breakdown too so good
        # capsule / client / date items rise to the top before
        # diversity-selection trims the pool.
        try:
            occ_text = (
                str(context.get("occasion") or "").lower()
                + " "
                + str(context.get("query") or context.get("user_query") or "").lower()
            )
            md_delta = 0.0
            md_reasons: List[str] = []
            for it in items:
                meta = it.get("style_metadata") if isinstance(it, dict) else None
                if not isinstance(meta, dict):
                    continue
                try:
                    pro = float(meta.get("professionalism_score") or 0.0)
                    cm = float(meta.get("client_meeting_score") or 0.0)
                    br = float(meta.get("boardroom_score") or 0.0)
                    cap = float(meta.get("capsule_score") or 0.0)
                    ver = float(meta.get("versatility_score") or 0.0)
                    dn = float(meta.get("date_night_score") or 0.0)
                except Exception:
                    continue
                noise = str(meta.get("visual_noise") or "").strip().lower()
                stmt = str(meta.get("statement_level") or "").strip().lower()
                professional_context = any(
                    marker in occ_text
                    for marker in ("client", "meeting", "boardroom", "office", "business", "interview")
                )
                capsule_context = "capsule" in occ_text or "essentials" in occ_text
                date_context = "date" in occ_text or "dinner" in occ_text or "evening" in occ_text
                if professional_context:
                    if pro:
                        md_delta += (pro - 0.5) * 1.5
                    if cm:
                        md_delta += (cm - 0.5) * 2.2
                        md_reasons.append("metadata.client_meeting_score boost")
                    if br:
                        md_delta += (br - 0.5) * 1.8
                    if noise == "high":
                        md_delta -= 1.25
                    if stmt in {"statement", "risky"}:
                        md_delta -= 1.25
                if capsule_context:
                    if cap:
                        md_delta += (cap - 0.5) * 3.0
                        md_reasons.append("metadata.capsule_score boost")
                    if ver:
                        md_delta += (ver - 0.5) * 1.5
                    if noise == "high":
                        md_delta -= 1.0
                    if stmt in {"statement", "risky"}:
                        md_delta -= 1.5
                if date_context:
                    if dn:
                        md_delta += (dn - 0.5) * 2.2
                        md_reasons.append("metadata.date_night_score boost")
            if md_delta:
                # Clamp so metadata can shift ranking but not dominate.
                md_delta = max(-3.0, min(3.0, md_delta))
                breakdown["metadata_richness"] = round(float(md_delta), 3)
                reasons.extend(md_reasons[:3])
                logger.info(
                    "metadata_rank.applied delta=%.3f occasion_text=%r reasons=%s",
                    md_delta,
                    occ_text[:120],
                    md_reasons[:3],
                )
        except Exception:
            pass

        raw_score = sum(float(v or 0.0) for v in breakdown.values())
        score = max(0.0, min(raw_score, 10.0))

        return {
            "score": round(score, 3),
            "label": self._label(score),
            "reasons": list(dict.fromkeys(reasons))[:3],
            "breakdown": {
                key: round(float(value or 0.0), 3)
                for key, value in breakdown.items()
            },
            "rejection_warnings": list(dict.fromkeys(warnings))[:5],
            "occasion_profile": occasion_result.get("profile") or {},
            "occasion_compatibility": occasion_result,
            "occasion_compatibility_score": occasion_result.get("score"),
            "occasion_penalties": occasion_result.get("penalties", []),
            "occasion_reject": occasion_result.get("reject", False),
            "weather_compatibility": weather_result,
        }

    def _graph_score(self, items: List[Dict[str, Any]], graph: Dict[str, Any]) -> float:
        score = 0.0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_id = items[i].get("id")
                b_id = items[j].get("id")
                if a_id and b_id:
                    score += style_graph_engine.pair_weight(graph, a_id, b_id)
        return score

    def _item_text(self, item: Dict[str, Any]) -> str:
        if not isinstance(item, dict):
            return ""
        fields = (
            "name",
            "label",
            "title",
            "type",
            "category",
            "sub_category",
            "subcategory",
            "role",
            "slot",
            "style",
            "material",
            "fabric",
            "color",
        )
        return " ".join(str(item.get(k) or "").lower() for k in fields)

    def _item_role(self, item: Dict[str, Any]) -> str:
        text = self._item_text(item)
        if any(
            x in text
            for x in [
                "shoe",
                "sneaker",
                "loafer",
                "boot",
                "sandal",
                "slipper",
                "slider",
            ]
        ):
            return "footwear"
        if any(x in text for x in ["shirt", "tee", "tshirt", "top", "polo", "hoodie", "kurta"]):
            return "top"
        if any(x in text for x in ["pant", "trouser", "jean", "chino", "shorts", "skirt"]):
            return "bottom"
        if any(x in text for x in ["dress", "saree", "lehenga", "gown"]):
            return "dress"
        return ""

    def _context_intent(self, context: Dict[str, Any]) -> str:
        text = " ".join(
            str(context.get(k) or "").lower()
            for k in ("occasion", "intent", "query", "user_query")
        )
        normalized = normalize_occasion(text)
        if normalized:
            return normalized
        if any(x in text for x in ["office", "meeting", "client", "work"]):
            return "office"
        if any(x in text for x in ["date", "dinner", "night"]):
            return "date_night"
        if any(x in text for x in ["party", "club", "after-hours"]):
            return "party"
        return "daily"

    def _footwear_occasion_score(
        self, items: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        intent = self._context_intent(context or {})
        footwear = next((i for i in items if self._item_role(i) == "footwear"), None)
        text = self._item_text(footwear or {})
        if not footwear:
            return -4.0, [], ["missing footwear"]

        relaxed = any(x in text for x in ["slipper", "slider", "slides", "flip", "crocs", "birkenstock", "sandal"])
        athletic = any(x in text for x in ["running", "gym", "training", "sports", "chunky", "runner", "hiking", "trail"])
        polished = any(
            x in text
            for x in [
                "loafer",
                "formal",
                "leather",
                "chelsea",
                "boot",
                "clean sneaker",
                "white sneaker",
                "minimal sneaker",
            ]
        )

        if intent == "office":
            if relaxed or athletic:
                return -5.0, ["footwear weak for office"], ["office footwear mismatch"]
            if polished:
                return 1.8, ["office-ready footwear"], []
        if intent == "date_night":
            if relaxed or athletic:
                return -5.0, ["footwear weak for date night"], ["date footwear mismatch"]
            if polished:
                return 1.5, ["polished footwear"], []
        if intent == "party" and polished:
            return 0.7, ["strong footwear finish"], []
        return 0.0, [], []

    def _dna_score(self, items: List[Dict[str, Any]], dna: Dict[str, Any]) -> float:
        if not dna:
            return 0.0
        score = 0.0
        preferred_styles = {str(x).lower() for x in dna.get("preferred_styles", [])}
        preferred_colors = {str(x).lower() for x in dna.get("preferred_colors", [])}
        for item in items:
            style = str(item.get("style") or "").lower()
            color = str(item.get("color") or item.get("color_code") or "").lower()
            if style and style in preferred_styles:
                score += 0.6
            if color and color in preferred_colors:
                score += 0.5
        return score

    def _refinement_score(self, items: List[Dict[str, Any]], refinement: str) -> float:
        score = 0.0
        for item in items:
            style = str(item.get("style") or "").lower()
            pattern = str(item.get("pattern") or "").lower()
            if refinement == "sharp" and style in {"formal", "structured"}:
                score += 0.5
            if refinement == "relaxed" and style in {"casual", "loose"}:
                score += 0.5
            if refinement == "minimal" and pattern in {"solid", "plain"}:
                score += 0.4
        return score

    def _build_outfit_embedding(self, items: List[Dict[str, Any]]) -> List[float]:
        text = " ".join(
            f"{i.get('type','')} {i.get('sub_category','')} {i.get('color','')} {i.get('style','')}"
            for i in items
            if isinstance(i, dict)
        )
        try:
            return encode_metadata({"text": text}) or []
        except Exception:
            return []

    def _exploration_boost(self, items: List[Dict[str, Any]], factor: float) -> float:
        if factor <= 0:
            return 0.0
        colors = [i.get("color") or i.get("color_code") for i in items if i.get("color") or i.get("color_code")]
        styles = [i.get("style") for i in items if i.get("style")]
        score = 0.0
        if len(set(colors)) >= 3:
            score += 0.5 * factor
        if len(set(styles)) >= 2:
            score += 0.4 * factor
        return score

    def _aesthetic_score(self, items: List[Dict[str, Any]]) -> float:
        colors = [
            color_normalizer.normalize(i.get("color") or i.get("color_code"))
            for i in items
            if i.get("color") or i.get("color_code")
        ]
        unique = len(set(colors))
        if unique == 1:
            return 1.0
        if unique == 2:
            return 0.7
        if unique >= 3:
            return 0.5
        return 0.0

    def _label(self, score: float) -> str:
        if score >= 6:
            return "Excellent"
        if score >= 4:
            return "Strong"
        if score >= 2:
            return "Good"
        return "Basic"

    def _is_neutral(self, color: str) -> bool:
        return color in {"black", "white", "grey", "gray", "beige", "navy", "cream"}


style_scorer = UnifiedStyleScorer()
