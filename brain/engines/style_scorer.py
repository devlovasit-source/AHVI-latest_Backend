from typing import Any, Dict, List, Tuple, Optional

from services.wardrobe_intelligence_service import (
    _style_meta,
    normalize_occasion as normalize_style_occasion,
    score_item_for_occasion,
)


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
            "swim", "boardshort",
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
            "shirt", "button", "trouser", "chino", "loafer", "leather sneaker",
            "blazer", "structured", "smart casual", "muted", "navy", "grey",
            "white", "black", "watch", "belt",
        ],
        "hard_penalty_terms": [
            "flip flop", "flip-flop", "swimwear", "gym shorts", "running shorts",
            "shiny sequin", "beachwear", "tank", "crop top", "shorts",
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
            "professional", "corporate",
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
    "club night": "party",
    "dinner party": "party",
    "airport outfit": "airport",
    "airport look": "airport",
    "road trip": "travel",
    "day trip": "travel",
    "overnight trip": "travel",
    "strength training": "gym",
    "cardio": "gym",
    "yoga": "gym",
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


def _outfit_blob(outfit: Dict[str, Any]) -> str:
    """Flatten outfit + every item into a single lower-case blob."""
    parts: List[str] = []
    for key in ("title", "vibe", "aesthetic", "style_direction", "explanation"):
        val = outfit.get(key)
        if val:
            parts.append(str(val))
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
    occasion = _resolve_occasion(context)
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
        "min_required_score": min_required,
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
    if any(w in text for w in ["date_night", "date night", "date", "dinner", "tonight"]):
        return "date_night"
    if any(w in text for w in ["beach", "pool", "seaside", "coastal", "resort"]):
        return "beach"
    if any(w in text for w in ["office", "corporate_office", "smart_casual_office", "work", "meeting", "client", "boardroom"]):
        return "office"
    if "brunch" in text:
        return "brunch"
    if "rave" in text or "club" in text:
        return "rave"
    if "cocktail" in text:
        return "cocktail"
    if any(w in text for w in ["party", "house_party", "after_hours", "night out"]):
        return "house_party"
    if any(w in text for w in ["travel", "airport", "flight", "vacation", "trip"]):
        return "travel"
    if any(w in text for w in ["workout", "gym", "fitness", "training", "yoga", "running"]):
        return "workout"
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
            score -= 0.35

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
            ]
        ):
            score += 0.35
        if any(w in blob for w in ["black trousers", "loafers", "dress shoes", "blazer", "formal", "office"]):
            score -= 0.65

    elif occasion == "office":
        if any(w in blob for w in ["shirt", "trousers", "loafers", "belt", "watch", "blazer"]):
            score += 0.20
        if any(w in blob for w in ["slipper", "slides", "beach", "gym shorts", "running shorts"]):
            score -= 0.45

    return score


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
            reasons.append("items pair well together")

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
