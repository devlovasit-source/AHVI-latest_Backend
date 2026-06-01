from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _tokens(value: Any) -> set[str]:
    import re

    return set(re.sub(r"[^a-z0-9]+", " ", _text(value)).split())


def _item_blob(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    parts: List[str] = []
    for key in (
        "name",
        "label",
        "title",
        "category",
        "sub_category",
        "subcategory",
        "type",
        "style",
        "style_tags",
        "tags",
        "pattern",
        "color",
        "color_name",
        "material",
        "fabric",
        "season",
        "occasions",
    ):
        value = item.get(key)
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(v) for v in value)
        else:
            parts.append(str(value or ""))
    return " ".join(parts).lower()


def _role(item: Dict[str, Any]) -> str:
    blob = _item_blob(item)
    tokens = _tokens(blob)
    if tokens.intersection(
        {
            "shoe",
            "shoes",
            "sneaker",
            "sneakers",
            "loafer",
            "loafers",
            "boot",
            "boots",
            "sandal",
            "sandals",
            "slide",
            "slides",
            "slider",
            "sliders",
            "slipper",
            "slippers",
            "footwear",
            "espadrille",
            "espadrilles",
        }
    ):
        return "footwear"
    if tokens.intersection({"dress", "gown", "jumpsuit"}):
        return "dress"
    if tokens.intersection({"pant", "pants", "trouser", "trousers", "jeans", "shorts", "skirt", "bottom", "bottoms"}):
        return "bottom"
    if tokens.intersection({"shirt", "tee", "tshirt", "t", "top", "polo", "blouse", "kurta", "sweater", "hoodie"}):
        return "top"
    if tokens.intersection({"watch", "ring", "bracelet", "necklace", "belt", "bag", "tote", "sunglasses", "eyewear", "cap", "hat"}):
        return "accessory"
    return _text(item.get("role") or item.get("category") or "unknown")


OCCASION_STYLE_RULES: Dict[str, Dict[str, Any]] = {
    "beach": {
        "mood": "airy, relaxed, breathable, coastal",
        "resolved_brief": "coastal casual, breathable, sand-friendly",
        "style_direction": "coastal_casual",
        "formality_target": 1.2,
        "footwear_energy": "sand-friendly",
        "prefer_keywords": [
            "linen",
            "cotton",
            "shorts",
            "camp collar",
            "resort",
            "vacation",
            "slides",
            "slide",
            "sandals",
            "sandal",
            "espadrilles",
            "sunglasses",
            "tote",
            "lightweight",
            "breathable",
        ],
        "avoid_keywords": [
            "formal",
            "office",
            "black trousers",
            "dress shoes",
            "loafers",
            "loafer",
            "blazer",
            "suit",
            "corporate",
            "oxford",
            "derby",
        ],
        "preferred_colors": ["white", "cream", "beige", "sand", "sky blue", "blue", "mint", "ecru"],
        "avoid_colors": ["black", "charcoal"],
        "preferred_materials": ["linen", "cotton", "canvas", "terry"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [
            ["black trousers", "loafers"],
            ["black pants", "loafers"],
            ["blazer", "sandals"],
            ["formal shirt", "dress shoes"],
            ["office shirt", "loafers"],
        ],
        "fallback_message": "I don't see enough beach-ready pieces yet.",
        "missing_recommendations": [
            {"label": "Linen shirt", "reason": "Adds breathable coastal texture", "cta": "Find this"},
            {"label": "Relaxed shorts", "reason": "Keeps the silhouette sand-friendly", "cta": "Find this"},
            {"label": "Slides or sandals", "reason": "Grounds the look without office energy", "cta": "Find this"},
            {"label": "Canvas tote", "reason": "Adds practical beach utility", "cta": "Find this"},
        ],
    },
    "office": {
        "mood": "clean, credible, professional",
        "resolved_brief": "smart office polish, structured but wearable",
        "style_direction": "smart_casual_office",
        "formality_target": 3.6,
        "footwear_energy": "polished",
        "prefer_keywords": ["shirt", "button", "trouser", "chino", "loafer", "leather sneaker", "belt", "watch"],
        "avoid_keywords": [
            "beach",
            "slipper",
            "slippers",
            "slides",
            "slide",
            "sandal",
            "sandals",
            "birkenstock",
            "espadrille",
            "espadrilles",
            "cap",
            "gym",
            "running",
            "embroidered",
            "embroidery",
            "sequin",
            "sequins",
            "festive",
            "ethnic",
            "kurta",
            "lehenga",
            "saree",
            "dhoti",
            "sherwani",
        ],
        "preferred_colors": ["white", "blue", "navy", "grey", "black", "cream"],
        "avoid_colors": [],
        "preferred_materials": ["cotton", "wool", "leather"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [
            ["cap", "office"],
            ["slippers", "shirt"],
            ["birkenstock", "trousers"],
            ["sandals", "shirt"],
            ["sandal", "trouser"],
            ["embroidered", "trouser"],
            ["embroidered", "chino"],
            ["festive", "trouser"],
            ["kurta", "trouser"],
        ],
        "fallback_message": "I don't see enough office-ready pieces yet.",
        "missing_recommendations": [
            {"label": "Crisp shirt", "reason": "Gives office structure immediately", "cta": "Find this"},
            {"label": "Tailored trouser", "reason": "Creates a credible base", "cta": "Find this"},
            {"label": "Loafers", "reason": "Adds polish without over-dressing", "cta": "Find this"},
        ],
    },
    "date": {
        "mood": "polished, approachable, evening-aware",
        "resolved_brief": "late-evening polished casual",
        "style_direction": "polished_social",
        "formality_target": 3.1,
        "footwear_energy": "polished",
        "prefer_keywords": ["shirt", "knit", "dress", "loafer", "boot", "minimal sneaker", "watch"],
        "avoid_keywords": ["slipper", "slides", "birkenstock", "gym", "running", "wrinkled"],
        "preferred_colors": ["black", "cream", "white", "navy", "burgundy", "brown"],
        "avoid_colors": [],
        "preferred_materials": ["cotton", "leather", "suede", "silk", "knit"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [["slippers", "date"], ["birkenstock", "dinner"], ["running shoes", "dinner"]],
        "fallback_message": "I don't see enough date-ready pieces yet.",
        "missing_recommendations": [
            {"label": "Polished shirt", "reason": "Keeps the mood intentional", "cta": "Find this"},
            {"label": "Clean loafers", "reason": "Sharpens the register after dark", "cta": "Find this"},
        ],
    },
    "party": {
        "mood": "social, memorable, edited",
        "resolved_brief": "after-hours social, expressive but controlled",
        "style_direction": "statement_social",
        "formality_target": 2.8,
        "footwear_energy": "social",
        "prefer_keywords": ["print", "pattern", "black", "shirt", "dress", "boot", "sneaker", "jewelry"],
        "avoid_keywords": ["office", "corporate", "sloppy"],
        "preferred_colors": ["black", "white", "red", "burgundy", "silver", "gold"],
        "avoid_colors": [],
        "preferred_materials": ["leather", "satin", "cotton", "denim"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [],
        "fallback_message": "I don't see enough party-ready pieces yet.",
        "missing_recommendations": [
            {"label": "Statement shirt", "reason": "Creates social energy quickly", "cta": "Find this"},
            {"label": "After-dark footwear", "reason": "Keeps the outfit from reading daytime", "cta": "Find this"},
        ],
    },
    "brunch": {
        "mood": "light, daytime, relaxed polish",
        "resolved_brief": "daytime brunch, easy polish, movement friendly",
        "style_direction": "daytime_polish",
        "formality_target": 2.2,
        "footwear_energy": "easy polish",
        "prefer_keywords": ["linen", "cotton", "shirt", "dress", "jeans", "chino", "sneaker", "sandal"],
        "avoid_keywords": ["formal", "corporate", "heavy", "gown"],
        "preferred_colors": ["white", "cream", "blue", "green", "beige", "pink"],
        "avoid_colors": [],
        "preferred_materials": ["cotton", "linen", "denim"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [],
        "fallback_message": "I don't see enough brunch-ready pieces yet.",
        "missing_recommendations": [
            {"label": "Light shirt", "reason": "Keeps brunch relaxed but styled", "cta": "Find this"},
            {"label": "Clean sneakers", "reason": "Works if the day continues into a walk", "cta": "Find this"},
        ],
    },
    "travel": {
        "mood": "comfortable, practical, clean",
        "resolved_brief": "travel comfort polish, movement friendly",
        "style_direction": "comfort_polish",
        "formality_target": 2.0,
        "footwear_energy": "walkable",
        "prefer_keywords": ["sneaker", "jogger", "chino", "tee", "layer", "jacket", "hoodie", "tote", "backpack"],
        "avoid_keywords": ["heel", "formal", "slipper"],
        "preferred_colors": ["black", "grey", "navy", "cream", "olive"],
        "avoid_colors": [],
        "preferred_materials": ["cotton", "denim", "knit", "fleece"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [["heels", "airport"], ["slippers", "travel"]],
        "fallback_message": "I don't see enough travel-ready pieces yet.",
        "missing_recommendations": [
            {"label": "Walkable sneakers", "reason": "Protects comfort through transit", "cta": "Find this"},
            {"label": "Light layer", "reason": "Handles AC and weather shifts", "cta": "Find this"},
        ],
    },
    "workout": {
        "mood": "functional, breathable, movement-first",
        "resolved_brief": "workout-ready, breathable, movement friendly",
        "style_direction": "training_functional",
        "formality_target": 1.0,
        "footwear_energy": "training",
        "prefer_keywords": ["training", "gym", "tee", "shorts", "leggings", "jogger", "running shoes", "sports"],
        "avoid_keywords": ["loafer", "formal", "blazer", "dress shoe"],
        "preferred_colors": ["black", "grey", "navy", "white"],
        "avoid_colors": [],
        "preferred_materials": ["polyester", "cotton", "mesh", "stretch"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [["loafers", "workout"], ["dress shoes", "gym"]],
        "fallback_message": "I don't see enough workout-ready pieces yet.",
        "missing_recommendations": [
            {"label": "Training tee", "reason": "Keeps movement breathable", "cta": "Find this"},
            {"label": "Training shoes", "reason": "Supports the workout safely", "cta": "Find this"},
        ],
    },
    "wedding": {
        "mood": "respectful, refined, occasion-aware",
        "resolved_brief": "formal event, respectful polish",
        "style_direction": "formal_event",
        "formality_target": 4.0,
        "footwear_energy": "formal polish",
        "prefer_keywords": ["saree", "kurta", "suit", "blazer", "dress", "gown", "loafer", "oxford", "heel"],
        "avoid_keywords": ["tee", "shorts", "slipper", "running", "gym"],
        "preferred_colors": ["black", "navy", "cream", "gold", "burgundy", "green"],
        "avoid_colors": [],
        "preferred_materials": ["silk", "satin", "wool", "leather"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [["sneakers", "wedding"], ["slippers", "wedding"], ["shorts", "ceremony"]],
        "fallback_message": "I don't see enough event-ready pieces yet.",
        "missing_recommendations": [
            {"label": "Formal outfit", "reason": "Respects the occasion", "cta": "Find this"},
            {"label": "Polished footwear", "reason": "Finishes the look properly", "cta": "Find this"},
        ],
    },
    "temple_modest": {
        "mood": "modest, respectful, soft, traditional",
        "resolved_brief": "modest temple-ready, covered shoulders, soft palette",
        "style_direction": "modest_traditional",
        "formality_target": 3.0,
        "footwear_energy": "soft, easy off",
        "prefer_keywords": [
            "saree",
            "sari",
            "kurta",
            "kurti",
            "salwar",
            "dupatta",
            "anarkali",
            "long skirt",
            "modest",
            "covered",
            "linen",
            "cotton",
            "silk",
            "sandals",
        ],
        "avoid_keywords": [
            "strapless",
            "cami",
            "camisole",
            "corset",
            "crop",
            "crop top",
            "shorts",
            "mini",
            "miniskirt",
            "club",
            "party",
            "deep neck",
            "low neck",
            "backless",
            "sleeveless",
            "bralette",
            "boots",
            "combat",
            "leather pants",
        ],
        "preferred_colors": ["white", "cream", "beige", "pastel", "soft pink", "mint", "lavender", "sand", "ivory"],
        "avoid_colors": ["neon", "fluorescent"],
        "preferred_materials": ["cotton", "silk", "linen", "chiffon"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [
            ["crop", "temple"],
            ["shorts", "temple"],
            ["strapless", "temple"],
            ["cami", "temple"],
        ],
        "fallback_message": "I don't see enough temple-ready modest pieces yet.",
        "missing_recommendations": [
            {"label": "Modest top or kurta", "reason": "Keeps the silhouette respectful", "cta": "Find this"},
            {"label": "Long skirt or trouser", "reason": "Covers and reads traditional", "cta": "Find this"},
            {"label": "Soft sandals", "reason": "Easy on/off for temple etiquette", "cta": "Find this"},
        ],
    },
    "casual": {
        "mood": "easy, clean, everyday",
        "resolved_brief": "clean daily, comfortable but intentional",
        "style_direction": "clean_daily",
        "formality_target": 2.0,
        "footwear_energy": "casual",
        "prefer_keywords": ["tee", "shirt", "jeans", "chino", "sneaker", "sandals"],
        "avoid_keywords": ["gown", "corporate"],
        "preferred_colors": [],
        "avoid_colors": [],
        "preferred_materials": ["cotton", "denim"],
        "required_slots": ["top_or_dress", "bottom_or_dress", "footwear"],
        "forbidden_pairings": [],
        "fallback_message": "I don't see enough clean everyday pieces yet.",
        "missing_recommendations": [
            {"label": "Clean tee", "reason": "Creates an easy base", "cta": "Find this"},
            {"label": "Versatile sneakers", "reason": "Works across daily plans", "cta": "Find this"},
        ],
    },
}


ALIASES = {
    "casual_outing": "casual",
    "daily": "casual",
    "date_night": "date",
    "date night": "date",
    "everyday": "casual",
    "event": "wedding",
    "formal_event": "wedding",
    "swimming": "beach",
    "swim": "beach",
    "pool": "beach",
    "temple": "temple_modest",
    "mandir": "temple_modest",
    "pooja": "temple_modest",
    "puja": "temple_modest",
    "religious": "temple_modest",
}


def get_occasion_rule(occasion: str) -> Dict[str, Any]:
    key = ALIASES.get(_text(occasion), _text(occasion))
    return dict(OCCASION_STYLE_RULES.get(key) or OCCASION_STYLE_RULES["casual"])


def score_item_for_occasion(item: Dict[str, Any], occasion_rule: Dict[str, Any]) -> float:
    blob = _item_blob(item)
    score = 0.0
    for keyword in occasion_rule.get("prefer_keywords") or []:
        if _text(keyword) in blob:
            score += 2.0
    for keyword in occasion_rule.get("avoid_keywords") or []:
        if _text(keyword) in blob:
            score -= 3.0
    for color in occasion_rule.get("preferred_colors") or []:
        if _text(color) and _text(color) in blob:
            score += 0.8
    for color in occasion_rule.get("avoid_colors") or []:
        if _text(color) and _text(color) in blob:
            score -= 1.8
    for material in occasion_rule.get("preferred_materials") or []:
        if _text(material) and _text(material) in blob:
            score += 1.2
    try:
        formality = float(item.get("formality") or item.get("formality_score") or 2.5)
        target = float(occasion_rule.get("formality_target") or 2.5)
        score -= abs(formality - target) * 0.6
    except Exception:
        pass
    return score


def board_text(card: Dict[str, Any]) -> str:
    return " ".join(_item_blob(item) for item in (card.get("items") or []) if isinstance(item, dict))


def forbidden_pairing_reason(card: Dict[str, Any], occasion_rule: Dict[str, Any]) -> str:
    text = board_text(card)
    for pair in occasion_rule.get("forbidden_pairings") or []:
        terms = [_text(term) for term in pair if _text(term)]
        if terms and all(term in text for term in terms):
            return "forbidden_pairing:" + "+".join(terms)
    return ""


def reject_board_for_occasion(card: Dict[str, Any], occasion: str, occasion_rule: Dict[str, Any] | None = None) -> str:
    rule = occasion_rule or get_occasion_rule(occasion)
    reason = forbidden_pairing_reason(card, rule)
    if reason:
        return reason
    text = board_text(card)
    key = ALIASES.get(_text(occasion), _text(occasion))
    if key == "beach":
        beach_blocks = (
            "black trousers",
            "black pants",
            "loafers",
            "dress shoes",
            "oxford",
            "derby",
            "blazer",
            "suit",
            "charcoal",
        )
        for block in beach_blocks:
            if block in text:
                return f"occasion_mismatch:{block}"
    if key in {"office", "date", "wedding"}:
        if any(term in text for term in ("slipper", "slippers", "slide", "slides", "birkenstock")):
            return "relaxed_footwear_blocked_for_occasion"
    if key == "office":
        if any(term in text for term in ("sandal", "sandals", "espadrille", "espadrilles", "flip flop", "flip-flop")):
            return "casual_footwear_blocked_for_office"
        if any(term in text for term in ("embroidered", "embroidery", "sequin", "festive")):
            return "festive_top_blocked_for_office"
        if any(term in text for term in ("strapless", "cami", "camisole", "corset", "bralette", "crop top", " crop ")):
            return "revealing_top_blocked_for_office"
        if any(term in text for term in (" cap ", "baseball cap", "bucket hat")):
            return "headwear_blocked_for_polished_occasion"
    if key == "temple_modest":
        if any(term in text for term in ("strapless", "cami", "camisole", "corset", "bralette", "crop top", " crop ", "deep neck", "low neck", "backless")):
            return "revealing_top_blocked_for_temple"
        if any(term in text for term in (" mini ", "mini skirt", "miniskirt", "shorts ")):
            return "short_bottom_blocked_for_temple"
        if any(term in text for term in ("combat boots", "chunky boots")):
            return "heavy_footwear_blocked_for_temple"
    return ""


def wardrobe_slot_scores(wardrobe: List[Dict[str, Any]], occasion_rule: Dict[str, Any]) -> Dict[str, List[Tuple[float, Dict[str, Any]]]]:
    slots = {"top": [], "bottom": [], "dress": [], "footwear": [], "accessory": []}
    for item in wardrobe or []:
        if not isinstance(item, dict):
            continue
        role = _role(item)
        score = score_item_for_occasion(item, occasion_rule)
        if role in slots:
            slots[role].append((score, item))
    for key in slots:
        slots[key].sort(key=lambda row: row[0], reverse=True)
    return slots


def detect_wardrobe_gap(
    wardrobe: List[Dict[str, Any]],
    occasion: str,
    occasion_rule: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rule = occasion_rule or get_occasion_rule(occasion)
    slots = wardrobe_slot_scores(wardrobe, rule)
    threshold = 1.0 if _text(occasion) == "beach" else 0.0
    has_top = bool(slots["top"] and slots["top"][0][0] >= threshold)
    has_bottom = bool(slots["bottom"] and slots["bottom"][0][0] >= threshold)
    has_dress = bool(slots["dress"] and slots["dress"][0][0] >= threshold)
    has_footwear = bool(slots["footwear"] and slots["footwear"][0][0] >= threshold)
    missing: List[Dict[str, str]] = []
    recommendations = list(rule.get("missing_recommendations") or [])
    if not (has_top or has_dress):
        missing.append(recommendations[0] if recommendations else {"label": "Occasion-ready top", "reason": "Starts the outfit in the right mood", "cta": "Find this"})
    if not (has_bottom or has_dress):
        missing.append(recommendations[1] if len(recommendations) > 1 else {"label": "Occasion-ready bottom", "reason": "Keeps the silhouette authentic", "cta": "Find this"})
    if not has_footwear:
        footwear_rec = next((r for r in recommendations if any(k in _text(r.get("label")) for k in ("shoe", "loafer", "sandal", "slide", "footwear", "sneaker"))), None)
        missing.append(footwear_rec or {"label": "Right footwear", "reason": "Footwear controls the occasion register", "cta": "Find this"})
    if len(missing) < 2 and recommendations:
        for rec in recommendations:
            if rec not in missing:
                missing.append(rec)
            if len(missing) >= 2:
                break
    missing = missing[:4]
    sufficient = (has_dress or (has_top and has_bottom)) and has_footwear
    return {
        "has_enough": bool(sufficient),
        "missing_items": missing,
        "closest_safe_brief": "clean daily smart casual" if _text(occasion) == "beach" else "clean daily",
        "rule": rule,
        "slot_scores": {key: [round(score, 3) for score, _ in values[:3]] for key, values in slots.items()},
    }
