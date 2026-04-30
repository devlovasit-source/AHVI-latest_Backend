import logging
from typing import Dict, Any

from services.ai_gateway import generate_text, parse_json_object

logger = logging.getLogger("ahvi.intent_engine")


INTENT_PROMPT = """
You are an intent classification engine for an AI stylist and organizer app.

Return ONLY JSON.

Schema:
{
  "intent": "daily_dependency | daily_outfit | occasion_outfit | explore_styles | wardrobe_query | try_on | organize_hub | plan_pack | general",
  "slots": {
    "occasion": "string or null",
    "style": "string or null",
    "vibe": "string or null",
    "time": "morning | midday | afternoon | evening | night | null",
    "module": "life_boards | meal_planner | medicines | bills | calendar | workout | skincare | contacts | life_goals | null"
  },
  "confidence": 0.0-1.0
}

Rules:
- "morning plan / daily cards / today plan / tomorrow preview" -> daily_dependency
- "what should I wear today" -> daily_outfit
- wedding/party/event -> occasion_outfit
- "show styles / casual / trending" -> explore_styles
- "how many tops do I have / count my wardrobe items" -> wardrobe_query
- "try this / try on" -> try_on
- "organize / life planner / bills / medicines / calendar / workout / skincare / contacts / goals" -> organize_hub
- "plan trip / pack for travel / wedding checklist / business travel packing" -> plan_pack
- Fill slots if clearly mentioned
- If unsure -> general

User:
"""


def _safe_parse(text: str) -> Dict[str, Any]:
    try:
        return parse_json_object(text)
    except Exception:
        return {"intent": "general", "slots": {}, "confidence": 0.3}


_ALLOWED_INTENTS = {
    "daily_dependency",
    "daily_outfit",
    "occasion_outfit",
    "explore_styles",
    "wardrobe_query",
    "try_on",
    "organize_hub",
    "plan_pack",
    "general",
}

_ALLOWED_TIMES = {"morning", "midday", "afternoon", "evening", "night"}

_ALLOWED_MODULES = {
    "life_boards",
    "meal_planner",
    "medicines",
    "bills",
    "calendar",
    "workout",
    "skincare",
    "contacts",
    "life_goals",
}


def _norm_text(value: Any, *, max_len: int = 64) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def _norm_key(value: Any) -> str:
    text = _norm_text(value, max_len=64).lower()
    text = text.replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _normalize_slots(raw: Any) -> Dict[str, Any]:
    slots = dict(raw) if isinstance(raw, dict) else {}
    out: Dict[str, Any] = {}

    occasion = _norm_key(slots.get("occasion"))
    occasion_map = {
        "date": "date_night",
        "date_night": "date_night",
        "datenight": "date_night",
        "office": "office",
        "work": "office",
        "workwear": "office",
        "wedding": "wedding",
        "party": "party",
        "travel": "travel",
        "gym": "gym",
        "workout": "gym",
        "fitness": "gym",
        "casual": "casual",
        "formal": "formal",
        "event": "event",
    }
    if occasion:
        out["occasion"] = occasion_map.get(occasion, occasion)

    time_slot = _norm_key(slots.get("time"))
    if time_slot in _ALLOWED_TIMES:
        out["time"] = time_slot

    module = _norm_key(slots.get("module"))
    module_map = {
        "life_board": "life_boards",
        "life_boards": "life_boards",
        "meals": "meal_planner",
        "meal": "meal_planner",
        "diet": "meal_planner",
        "nutrition": "meal_planner",
        "medicine": "medicines",
        "meds": "medicines",
        "bills": "bills",
        "bill": "bills",
        "calendar": "calendar",
        "schedule": "calendar",
        "workout": "workout",
        "fitness": "workout",
        "gym": "workout",
        "skincare": "skincare",
        "contacts": "contacts",
        "goals": "life_goals",
        "life_goals": "life_goals",
    }
    if module:
        canonical = module_map.get(module, module)
        if canonical in _ALLOWED_MODULES:
            out["module"] = canonical

    style = _norm_text(slots.get("style"), max_len=48).lower()
    if style:
        out["style"] = style

    vibe = _norm_text(slots.get("vibe"), max_len=48).lower()
    if vibe:
        out["vibe"] = vibe

    return out


def _validate_intent_row(row: Any, *, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce over-trust in the LLM by:
    - allowlist intents
    - clamping confidence
    - normalizing slots
    - requiring minimum confidence for non-general intents
    - cross-checking against deterministic heuristic intent when confidence is low
    """
    base = dict(row) if isinstance(row, dict) else {}
    intent = _norm_key(base.get("intent")) or "general"
    if intent not in _ALLOWED_INTENTS:
        intent = "general"

    try:
        conf = float(base.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(conf, 1.0))

    slots = _normalize_slots(base.get("slots"))

    # If model claims a specific intent but isn't confident, fall back.
    if intent != "general" and conf < 0.55:
        return {**fallback, "slots": {**_normalize_slots(fallback.get("slots")), **slots}}

    # If model disagrees with heuristic and isn't very confident, prefer heuristic.
    heuristic_intent = _norm_key(fallback.get("intent")) or "general"
    if heuristic_intent in _ALLOWED_INTENTS and intent != heuristic_intent and conf < 0.75:
        merged_slots = {**_normalize_slots(base.get("slots")), **_normalize_slots(fallback.get("slots"))}
        return {"intent": heuristic_intent, "slots": merged_slots, "confidence": max(float(fallback.get("confidence", 0.0) or 0.0), conf)}

    return {"intent": intent, "slots": slots, "confidence": conf}


def _fallback_intent(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    slots: Dict[str, Any] = {}

    if "office" in t or "work" in t:
        slots["occasion"] = "office"
    elif "party" in t:
        slots["occasion"] = "party"
    elif "wedding" in t:
        slots["occasion"] = "wedding"
    elif "date" in t:
        slots["occasion"] = "date_night"

    if "morning" in t:
        slots["time"] = "morning"
    elif "midday" in t or "noon" in t:
        slots["time"] = "midday"
    elif "afternoon" in t:
        slots["time"] = "afternoon"
    elif "evening" in t:
        slots["time"] = "evening"
    elif "night" in t:
        slots["time"] = "night"

    if any(x in t for x in ["wear", "outfit", "style me", "recommend look", "what should i wear", "dress me"]):
        return {"intent": "daily_outfit", "slots": slots, "confidence": 0.68}

    daily_words = [
        "daily plan", "daily cards", "morning flow", "midday flow", "afternoon flow",
        "evening flow", "night flow", "tomorrow preview", "day planner", "daily dependency",
    ]
    if any(x in t for x in daily_words):
        return {"intent": "daily_dependency", "slots": slots, "confidence": 0.8}

    if any(x in t for x in ["wedding", "party", "event"]):
        return {"intent": "occasion_outfit", "slots": slots or {"occasion": "event"}, "confidence": 0.7}

    if any(x in t for x in ["try", "try on", "virtual try", "preview this"]):
        return {"intent": "try_on", "slots": slots, "confidence": 0.72}

    if any(x in t for x in ["trend", "style ideas", "inspiration", "new styles"]):
        return {"intent": "explore_styles", "slots": slots, "confidence": 0.62}

    count_words = ["how many", "count", "number of", "total", "do i have"]
    wardrobe_words = [
        "wardrobe", "closet", "outfit", "outfits", "tops", "top", "shirts", "shirt",
        "tshirt", "t-shirt", "pants", "trousers", "jeans", "bottoms", "shoes",
        "footwear", "dress", "dresses", "accessories", "jewelry", "bags", "bag",
    ]
    if any(x in t for x in count_words) and any(x in t for x in wardrobe_words):
        return {"intent": "wardrobe_query", "slots": slots, "confidence": 0.8}

    organize_words = [
        "organize", "life board", "meal planner", "meal", "diet", "nutrition",
        "medicine", "meds", "bills", "calendar", "workout", "fitness", "gym",
        "skincare", "contacts", "life goals", "goals"
    ]
    if any(x in t for x in organize_words):
        if "life board" in t:
            slots["module"] = "life_boards"
        elif "meal" in t or "diet" in t or "nutrition" in t:
            slots["module"] = "meal_planner"
        elif "med" in t:
            slots["module"] = "medicines"
        elif "bill" in t:
            slots["module"] = "bills"
        elif "calendar" in t:
            slots["module"] = "calendar"
        elif "workout" in t or "fitness" in t or "gym" in t:
            slots["module"] = "workout"
        elif "skin" in t:
            slots["module"] = "skincare"
        elif "contact" in t:
            slots["module"] = "contacts"
        elif "goal" in t:
            slots["module"] = "life_goals"
        return {"intent": "organize_hub", "slots": slots, "confidence": 0.75}

    plan_pack_words = [
        "plan trip", "trip plan", "travel plan", "packing list", "pack for",
        "pack my", "business travel", "wedding checklist", "checklist for trip",
        "goa trip", "vacation packing"
    ]
    if any(x in t for x in plan_pack_words):
        return {"intent": "plan_pack", "slots": slots, "confidence": 0.78}

    return {"intent": "general", "slots": slots, "confidence": 0.4}


def detect_intent(user_text: str, history=None, model: str | None = None) -> Dict[str, Any]:
    if not user_text:
        return {"intent": "general", "slots": {}, "confidence": 0.0}

    # Fast deterministic path first; avoids unnecessary model latency for obvious intents.
    fallback = _fallback_intent(user_text)
    if float(fallback.get("confidence", 0.0)) >= 0.75:
        return _validate_intent_row(fallback, fallback=fallback)

    prompt = INTENT_PROMPT + user_text
    try:
        response = generate_text(
            prompt,
            options={"temperature": 0.2, "num_predict": 200},
            usecase="intent",
            model=model,
        )
    except Exception:
        return _validate_intent_row(fallback, fallback=fallback)

    parsed = _safe_parse(response)
    parsed = _validate_intent_row(parsed, fallback=fallback)

    if history:
        last = history[-1] if history else {}
        if parsed.get("intent") == "general" and last.get("intent"):
            last_intent = _norm_key(last.get("intent"))
            if last_intent in _ALLOWED_INTENTS and last_intent != "general":
                parsed["intent"] = last_intent
                parsed["confidence"] = max(float(parsed.get("confidence", 0.0) or 0.0), 0.6)

    return parsed
