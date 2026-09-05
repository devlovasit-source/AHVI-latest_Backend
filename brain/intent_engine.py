import logging
import re
from typing import Dict, Any

from services.ai_gateway import generate_text, parse_json_object
from services.style_item_contract import has_garment_word
from services.stylist_knowledge_service import (
    COLOR_BODY_ADVICE,
    SHOPPING_ASSIST,
    STYLE_ADVICE,
    STYLE_EDUCATION,
    STYLE_PAIRING,
    WARDROBE_STYLE,
    classify_style_mode,
)

logger = logging.getLogger("ahvi.intent_engine")


# Words that name a topic AHVI could plausibly act on in more than one way
# (Style, Diet, or Planning) -- grouped so one clarification question/chip
# set can serve every noun in the group instead of one-off phrasing per word.
_AMBIGUOUS_CONTEXT_GROUPS = {
    "meal": {
        "nouns": ("dinner", "lunch", "breakfast"),
        "question": "are you thinking about what to eat, what to wear, or planning the evening",
        "chips": (("What to eat", "what should i eat for {noun}"),
                  ("What to wear", "what should i wear for {noun}"),
                  ("Plan my day", "plan my day")),
    },
    "event": {
        "nouns": ("wedding", "party"),
        "question": "do you want help with your outfit, preparation, or plans",
        "chips": (("Style me", "style me for the {noun}"),
                  ("Help me prepare", "help me prepare for the {noun}"),
                  ("Plan / checklist", "create a checklist for the {noun}")),
    },
    "work": {
        "nouns": ("work", "office", "meeting"),
        "question": "are you thinking outfit, schedule, or something else",
        "chips": (("What to wear", "what should i wear to {noun}"),
                  ("Plan my day", "plan my day")),
    },
    "date": {
        "nouns": ("date",),
        "question": "are you thinking what to wear or how to plan the evening",
        "chips": (("What to wear", "what should i wear for a {noun}"),
                  ("Plan the evening", "plan my evening")),
    },
    "travel": {
        "nouns": ("vacation",),
        "question": "are you thinking what to pack or how to plan the trip",
        "chips": (("What to wear", "what should i wear for {noun}"),
                  ("Plan my trip", "create a checklist for my trip")),
    },
}
_AMBIGUOUS_CONTEXT_NOUNS = {
    noun: group for group in _AMBIGUOUS_CONTEXT_GROUPS.values() for noun in group["nouns"]
}
_AMBIGUOUS_TIME_QUALIFIERS = (
    "today", "tonight", "tomorrow", "this evening", "this morning",
    "this weekend", "next weekend", "this week", "next week",
)
_AMBIGUOUS_STOPWORDS = {"the", "a", "an", "my"}


def detect_action_ambiguity(text: str) -> Dict[str, Any] | None:
    """A bare context noun ("Dinner tonight") names a TOPIC, not a request --
    it is genuinely ambiguous across Style/Diet/Planning and should ask one
    clarifying question rather than guess a module or answer as if it were
    an opinion. Distinguishing signal: after removing the context noun, any
    time qualifier, and a couple of stopwords, nothing is left -- there is
    no verb, no opinion, no other content word. A real sentence ("Dinner was
    terrible") always leaves something behind ("was", "terrible") and must
    NOT be treated as ambiguous.
    """
    normalized = " ".join(str(text or "").lower().replace("'", "").split())
    if not normalized:
        return None
    # Word-boundary matched: short nouns like "work"/"date" would otherwise
    # false-match inside ordinary words ("homework", "update", "candidate").
    matched_noun = next(
        (n for n in _AMBIGUOUS_CONTEXT_NOUNS if re.search(rf"\b{n}\b", normalized)),
        None,
    )
    if not matched_noun:
        return None
    remainder = re.sub(rf"\b{matched_noun}\b", " ", normalized, count=1)
    for phrase in _AMBIGUOUS_TIME_QUALIFIERS:
        remainder = remainder.replace(phrase, " ", 1)
    leftover_tokens = [tok for tok in remainder.split() if tok not in _AMBIGUOUS_STOPWORDS]
    if leftover_tokens:
        return None
    group = _AMBIGUOUS_CONTEXT_NOUNS[matched_noun]
    phrase = " ".join(word.capitalize() if i == 0 else word for i, word in enumerate(normalized.split()))
    return {
        "noun": matched_noun,
        "message": f"{phrase} — {group['question']}?",
        "chips": [
            {"label": label, "value": template.format(noun=matched_noun)}
            for label, template in group["chips"]
        ],
    }


# Bare meal-time nouns (breakfast/lunch/dinner) name the TOPIC of a sentence,
# not what the user wants AHVI to do with it -- "Dinner was terrible" is
# conversation, not a meal-planning request. Diet only admits on those nouns
# when a genuine action/planning word co-occurs, or when
# "diet"/"nutrition"/"calories" are named directly (those ARE the ask, not
# just a topic word). Shared by every deterministic diet-admission
# chokepoint (brain/intent_engine.py's own fallback, and routers/chat.py's
# separate visual-board fast path) so the rule lives in exactly one place.
_DIET_ACTION_WORDS = (
    "plan", "suggest", "recommend", "create", "build", "give me",
    "help me", "healthy", "high protein", "high-protein", "low carb",
    "low-carb",
)
# Word-boundary matched: "eat" is short enough to false-match inside
# ordinary words ("create", "repeat", "wheat") as a plain substring.
_DIET_CONTEXT_WORDS = ("breakfast", "lunch", "dinner", "eat", "eating")


def has_diet_action_intent(text: str) -> bool:
    t = str(text or "").lower()
    if any(w in t for w in ("diet", "nutrition", "calories")):
        return True
    return any(action in t for action in _DIET_ACTION_WORDS) and any(
        re.search(rf"\b{context}\b", t) for context in _DIET_CONTEXT_WORDS
    )


INTENT_PROMPT = """
You are an intent classification engine for AHVI, a personal assistant for
Style, Planning, Preparation, Wardrobe, Meals, Fitness, and daily life.

Return ONLY valid JSON.

Schema:
{
  "intent": "daily_dependency | daily_outfit | occasion_outfit | explore_styles | wardrobe_query | try_on | organize_hub | plan_pack | style_advice | style_pairing | wardrobe_style | shopping_assist | style_education | color_body_advice | general",
  "slots": {
    "occasion": "string or null",
    "style": "string or null",
    "vibe": "string or null",
    "time": "morning | midday | afternoon | evening | night | null",
    "module": "life_boards | meal_planner | medicines | bills | calendar | workout | skincare | contacts | life_goals | null"
  },
  "confidence": 0.0-1.0
}

CORE PRINCIPLE:

INTENT answers:
"What does the user want AHVI to do?"

SLOTS answer:
"What is the situation or context about?"

Context words may populate slots, but MUST NOT by themselves determine intent.

Words such as:
work, office, dinner, lunch, breakfast, wedding, party, meeting,
date, vacation, travel, gym, event

may describe context.

Do NOT infer a module only because one of these words is present.

==================================================
GENERAL / CONVERSATIONAL
==================================================

Conversational, emotional, descriptive, retrospective, or opinion statements
should normally be classified as general unless the user also expresses a
clear actionable request.

Examples:

"Work was horrible today"
-> general

"Dinner was terrible"
-> general

"The wedding was exhausting"
-> general

"The party was boring"
-> general

"I hate going to the office"
-> general

"Lunch was amazing"
-> general

"That meeting went badly"
-> general

"Vacation was exhausting"
-> general

==================================================
AMBIGUOUS CONTEXT
==================================================

If the user provides context but does not state what they want AHVI to do,
do NOT guess a module.

Examples:

"Dinner tonight"
-> general

"Wedding tomorrow"
-> general

"Work tomorrow"
-> general

"Party tonight"
-> general

"Lunch tomorrow"
-> general

"Vacation next week"
-> general

"Meeting tomorrow"
-> general

"Date tonight"
-> general

These may be handled by a downstream clarification layer.

You may still populate relevant slots when clearly inferable.

Example:

"Wedding tomorrow"
-> intent=general
-> slots.occasion="wedding"

Do not force Style, Diet, or Planning merely from the context.

==================================================
STYLE
==================================================

Route to Style only when the user clearly asks for clothing, styling,
outfit, dressing, pairing, appearance, or wardrobe help.

Examples:

"What should I wear today?"
-> daily_outfit

"What should I wear to work?"
-> style_advice

"What should I wear for dinner tonight?"
-> style_advice

"Style me for a wedding"
-> occasion_outfit

"Give me an outfit for the party"
-> occasion_outfit

"Help me dress for dinner"
-> style_advice

"What goes with these trousers?"
-> style_pairing

"How do I style this shirt?"
-> style_pairing

"Use my wardrobe for dinner"
-> wardrobe_style

"Build a look from my closet"
-> wardrobe_style

"What colors suit my skin tone?"
-> color_body_advice

"What is smart casual?"
-> style_education

"Show me casual styles"
-> explore_styles

"Show me trending styles"
-> explore_styles

IMPORTANT:

Do NOT classify bare occasion/context nouns as Style.

Examples:

"Wedding tomorrow"
-> general

"Office tomorrow"
-> general

"Party tonight"
-> general

"Casual"
-> general unless the wording clearly requests style exploration

==================================================
MEALS / DIET
==================================================

Route to the meal planner only when the user clearly asks for meal, diet,
nutrition, food-planning, or eating guidance.

Examples:

"What should I eat tonight?"
-> organize_hub
-> slots.module="meal_planner"

"Plan my dinner"
-> organize_hub
-> slots.module="meal_planner"

"Create a meal plan"
-> organize_hub
-> slots.module="meal_planner"

"Suggest a healthy lunch"
-> organize_hub
-> slots.module="meal_planner"

"Give me a high-protein breakfast"
-> organize_hub
-> slots.module="meal_planner"

"Help me with my diet"
-> organize_hub
-> slots.module="meal_planner"

IMPORTANT:

Do NOT infer meal_planner from meal names alone.

Examples:

"Dinner was terrible"
-> general

"Lunch was awful"
-> general

"Breakfast was amazing"
-> general

"Dinner tonight"
-> general

"Lunch tomorrow"
-> general

==================================================
PLANNING / CALENDAR
==================================================

Route to Planning or Calendar only when the user clearly asks AHVI to plan,
schedule, organize, add, remind, or create an event.

Examples:

"Plan my day tomorrow"
-> organize_hub
-> slots.module="calendar"

"Schedule my work day"
-> organize_hub
-> slots.module="calendar"

"Add a meeting tomorrow at 4 PM"
-> organize_hub
-> slots.module="calendar"

"Remind me about my appointment"
-> organize_hub
-> slots.module="calendar"

IMPORTANT:

Do NOT infer Planning merely from a date, time, meeting, work, or event noun.

Examples:

"Meeting tomorrow"
-> general

"Work tomorrow"
-> general

"Party tonight"
-> general

==================================================
PLAN / PACK / PREPARATION
==================================================

Use plan_pack when the user clearly asks to prepare, pack, or create a
checklist for travel or an event.

Examples:

"Pack for the wedding"
-> plan_pack

"Create a checklist for my trip"
-> plan_pack

"Help me prepare for a business trip"
-> plan_pack

Do NOT classify "Wedding tomorrow" or "Vacation next week" as plan_pack
unless preparation or packing intent is explicit.

==================================================
OTHER ROUTING RULES
==================================================

- "morning plan", "daily cards", "today plan", "tomorrow preview"
  -> daily_dependency

- Explicit wardrobe count/query:
  "How many tops do I have?"
  "Count my wardrobe items"
  -> wardrobe_query

- Explicit try-on:
  "Try this on"
  "Show me how this would look on me"
  -> try_on

- Explicit shopping or missing-piece requests:
  "What should I buy?"
  "Recommend shoes for this look"
  "What am I missing?"
  -> shopping_assist

- Explicit organization requests for medicines, bills, workout, skincare,
  contacts, or goals
  -> organize_hub with the appropriate module

==================================================
FOLLOW-UP CONTEXT
==================================================

When history clearly establishes an active module or task, preserve that context
for natural follow-ups even when the new message does not repeat domain words.

Examples:

Previous:
"Plan my dinner"

Follow-up:
"Make it lighter"
-> preserve meal-planner context

Previous:
"What should I wear for dinner?"

Follow-up:
"Make it less formal"
-> preserve Style context

Previous:
"Plan my day tomorrow"

Follow-up:
"Add my meeting at 4"
-> preserve Planning context

Do not apply a fresh context-noun guess when an active module is already clear.

==================================================
FINAL SAFETY RULE
==================================================

Do not infer Style merely from an occasion.
Do not infer Diet merely from a meal name.
Do not infer Planning merely from a date/time or event noun.

If the requested action is unclear:
-> intent=general

If unsure:
-> intent=general

Fill slots only when they are clearly supported by the text.

Return ONLY valid JSON.

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
    "style_advice",
    "style_pairing",
    "style_education",
    "color_body_advice",
    "body_proportion_advice",
    "color_advice",
    "occasion_advice",
    "wardrobe_style",
    "shopping_assist",
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

    action = _norm_key(slots.get("action"))
    if action in {"create_event"}:
        out["action"] = action

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
        return {
            **fallback,
            "slots": {**_normalize_slots(fallback.get("slots")), **slots},
        }

    # If model disagrees with heuristic and isn't very confident, prefer heuristic.
    heuristic_intent = _norm_key(fallback.get("intent")) or "general"
    if heuristic_intent in {
        STYLE_ADVICE,
        STYLE_EDUCATION,
        COLOR_BODY_ADVICE,
        SHOPPING_ASSIST,
        WARDROBE_STYLE,
    } and intent != heuristic_intent:
        merged_slots = {
            **_normalize_slots(base.get("slots")),
            **_normalize_slots(fallback.get("slots")),
        }
        return {
            "intent": heuristic_intent,
            "slots": merged_slots,
            "confidence": max(float(fallback.get("confidence", 0.0) or 0.0), conf),
        }

    if (
        heuristic_intent in _ALLOWED_INTENTS
        and intent != heuristic_intent
        and conf < 0.75
    ):
        merged_slots = {
            **_normalize_slots(base.get("slots")),
            **_normalize_slots(fallback.get("slots")),
        }
        return {
            "intent": heuristic_intent,
            "slots": merged_slots,
            "confidence": max(float(fallback.get("confidence", 0.0) or 0.0), conf),
        }

    if heuristic_intent == intent:
        slots = {
            **_normalize_slots(fallback.get("slots")),
            **slots,
        }

    return {"intent": intent, "slots": slots, "confidence": conf}


def _fallback_intent(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    slots: Dict[str, Any] = {}
    normalized = " ".join(t.replace("'", "").split())

    greeting_phrases = {
        "hi",
        "hello",
        "hey",
        "hi ahvi",
        "hello ahvi",
        "hey ahvi",
        "good morning",
        "good evening",
        "good afternoon",
    }
    if normalized in greeting_phrases:
        return {
            "intent": "general",
            "slots": {"sub_intent": "greeting"},
            "confidence": 0.96,
        }

    help_identity_phrases = {
        "what can you do",
        "who are you",
        "what are you",
        "what is ahvi",
        "tell me about yourself",
        "how can you help",
        "help",
        "what do you do",
        "what are your features",
        "what can ahvi do",
    }
    if normalized in help_identity_phrases:
        return {
            "intent": "general",
            "slots": {"sub_intent": "help_identity"},
            "confidence": 0.95,
        }

    small_talk_phrases = {
        "how are you",
        "thanks",
        "thank you",
    }
    if normalized in small_talk_phrases:
        return {
            "intent": "general",
            "slots": {"sub_intent": "small_talk"},
            "confidence": 0.95,
        }

    def _has_any(*phrases: str) -> bool:
        return any(phrase in t or phrase in normalized for phrase in phrases)

    style_mode = classify_style_mode(normalized)
    if style_mode and style_mode != STYLE_ADVICE:
        confidence = 0.95 if style_mode == WARDROBE_STYLE else 0.93
        if style_mode == SHOPPING_ASSIST:
            confidence = 0.93
        elif style_mode in {COLOR_BODY_ADVICE, STYLE_EDUCATION}:
            confidence = 0.94
        return {"intent": style_mode, "slots": slots, "confidence": confidence}

    style_words = {"outfit", "wear", "style", "look", "looks", "wardrobe"}
    has_style_word = any(word in normalized.split() for word in style_words)
    calendar_event_phrases = (
        "appointment",
        "doctor",
        "dentist",
        "meeting with",
        "call with",
        "call at",
        "interview",
        "remind me",
        "schedule meeting",
        "tomorrow at",
        "today at",
        "monday at",
        "tuesday at",
        "wednesday at",
        "thursday at",
        "friday at",
        "saturday at",
        "sunday at",
    )
    explicit_event_phrase = _has_any(
        "appointment",
        "doctor",
        "dentist",
        "meeting with",
        "call with",
        "call at",
        "interview",
        "remind me",
        "schedule meeting",
    )
    timed_event_phrase = _has_any(
        "tomorrow at",
        "today at",
        "monday at",
        "tuesday at",
        "wednesday at",
        "thursday at",
        "friday at",
        "saturday at",
        "sunday at",
    )
    if _has_any(*calendar_event_phrases) and (explicit_event_phrase or (timed_event_phrase and not has_style_word)):
        slots["module"] = "calendar"
        slots["action"] = "create_event"
        return {"intent": "organize_hub", "slots": slots, "confidence": 0.95}

    planner_phrases = (
        "plan my day",
        "organize my day",
        "schedule my day",
        "today plan",
        "today schedule",
        "plan today",
        "prep my day",
    )
    if _has_any(*planner_phrases):
        slots["module"] = "calendar"
        return {"intent": "organize_hub", "slots": slots, "confidence": 0.9}

    style_priority_phrases = (
        "office outfit",
        "client meeting",
        "date night",
        "party look",
        "casual dinner",
        "style me",
    )
    if not style_mode and _has_any(*style_priority_phrases):
        if "client meeting" in normalized or "office" in normalized:
            slots["occasion"] = "office"
        elif "date" in normalized:
            slots["occasion"] = "date_night"
        elif "party" in normalized:
            slots["occasion"] = "party"
        elif "dinner" in normalized:
            slots["occasion"] = "date_night"
        return {"intent": "occasion_outfit", "slots": slots, "confidence": 0.9}

    diet_phrases = (
        "eat today",
        "what should i eat",
        "what to eat today",
        "food today",
        "meal today",
        "diet today",
        "meal plan",
    )
    if style_mode != STYLE_ADVICE and (_has_any(*diet_phrases) or has_diet_action_intent(t)):
        slots["module"] = "meal_planner"
        return {"intent": "organize_hub", "slots": slots, "confidence": 0.9}

    workout_phrases = (
        "workout today",
        "today workout",
        "gym today",
        "exercise today",
        "fitness today",
    )
    if _has_any(*workout_phrases):
        slots["module"] = "workout"
        return {"intent": "organize_hub", "slots": slots, "confidence": 0.9}

    early_module_hits = [
        (
            "meal_planner",
            (
                "today's meals",
                "todays meals",
                "today meals",
                "my meals",
                "meal plan",
                "meal planner",
                "what should i eat",
            ),
        ),
        (
            "workout",
            (
                "today's workout",
                "todays workout",
                "today workout",
                "my workout",
                "workout today",
                "fitness today",
            ),
        ),
        (
            "skincare",
            (
                "morning skincare",
                "night skincare",
                "evening skincare",
                "skincare routine",
                "my skincare",
                "morning routine",
                "evening routine",
                "night routine",
                "create routine",
                "open skincare",
            ),
        ),
        (
            "bills",
            (
                "pending bills",
                "my bills",
                "unpaid bills",
                "bills due",
                "today's bills",
                "todays bills",
            ),
        ),
        (
            "medicines",
            (
                "my medicines",
                "my medicine",
                "my meds",
                "today's medicines",
                "todays medicines",
                "today medicines",
                "medication list",
            ),
        ),
        (
            "calendar",
            (
                "today's events",
                "todays events",
                "today events",
                "upcoming events",
                "my events",
                "my schedule",
                "today's schedule",
                "todays schedule",
            ),
        ),
    ]
    for module, phrases in early_module_hits:
        if _has_any(*phrases):
            slots["module"] = module
            return {"intent": "organize_hub", "slots": slots, "confidence": 0.84}

    style_priority_phrases = (
        "client meeting outfit",
    )
    if _has_any(*style_priority_phrases):
        if "client meeting" in normalized or "office" in normalized:
            slots["occasion"] = "office"
        elif "date" in normalized:
            slots["occasion"] = "date_night"
        elif "party" in normalized:
            slots["occasion"] = "party"
        elif "dinner" in normalized:
            slots["occasion"] = "date_night"
        return {"intent": "occasion_outfit", "slots": slots, "confidence": 0.9}

    if style_mode == STYLE_ADVICE:
        return {"intent": STYLE_ADVICE, "slots": slots, "confidence": 0.9}

    plan_pack_words = [
        "plan trip",
        "trip plan",
        "travel plan",
        "packing list",
        "pack for",
        "pack my",
        "business travel",
        "wedding checklist",
        "checklist for trip",
        "goa trip",
        "vacation packing",
        "carry-on",
        "carry on",
        "camping",
        "prep for camping",
        "birthday party",
        "plan a birthday",
        "plan birthday",
    ]
    if any(x in t for x in plan_pack_words):
        return {"intent": "plan_pack", "slots": slots, "confidence": 0.82}

    module_hits = [
        (
            "meal_planner",
            (
                "today's meals",
                "todays meals",
                "today meals",
                "my meals",
                "meal plan",
                "meal planner",
                "what should i eat",
            ),
        ),
        (
            "workout",
            (
                "today's workout",
                "todays workout",
                "today workout",
                "my workout",
                "workout today",
                "fitness today",
            ),
        ),
        (
            "skincare",
            (
                "morning skincare",
                "night skincare",
                "evening skincare",
                "skincare routine",
                "my skincare",
                "morning routine",
                "evening routine",
                "night routine",
                "create routine",
                "open skincare",
            ),
        ),
        (
            "bills",
            (
                "pending bills",
                "my bills",
                "unpaid bills",
                "bills due",
                "today's bills",
                "todays bills",
            ),
        ),
        (
            "medicines",
            (
                "my medicines",
                "my medicine",
                "my meds",
                "today's medicines",
                "todays medicines",
                "today medicines",
                "medication list",
            ),
        ),
        (
            "calendar",
            (
                "today's events",
                "todays events",
                "today events",
                "upcoming events",
                "my events",
                "my schedule",
                "today's schedule",
                "todays schedule",
            ),
        ),
    ]
    for module, phrases in module_hits:
        if _has_any(*phrases):
            slots["module"] = module
            return {"intent": "organize_hub", "slots": slots, "confidence": 0.84}

    # Specific occasions must beat generic "work" keyword so Gym/Party/etc.
    # calendar chats don't get hijacked when user types "work outfit"
    # inside a non-work session.
    if "gym" in t or "workout" in t or "fitness" in t:
        slots["occasion"] = "gym"
    elif "party" in t:
        slots["occasion"] = "party"
    elif "wedding" in t:
        slots["occasion"] = "wedding"
    elif "date night" in t or "date" in t:
        slots["occasion"] = "date_night"
    elif "shopping" in t:
        slots["occasion"] = "shopping"
    elif "study" in t:
        slots["occasion"] = "study"
    elif "travel" in t or "trip" in t:
        slots["occasion"] = "travel"
    elif "office" in t or "work" in t:
        slots["occasion"] = "office"

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

    if _has_any(
        "what to pair with",
        "what pairs with",
        "what pairs best with",
        "what pairs",
        "what works with",
        "what goes with",
        "how to style",
        "how do i style",
        "how do i wear",
        "ways to wear",
        "ways to style",
        "how can i wear",
        "what matches",
        "what should i wear with",
        "style this",
        "pair this with",
        "pair with",
        "wear with",
    ):
        return {"intent": STYLE_PAIRING, "slots": slots, "confidence": 0.9}

    if _has_any("what should i wear", "what to wear", "what do i wear"):
        return {"intent": STYLE_ADVICE, "slots": slots, "confidence": 0.86}

    if any(
        x in t
        for x in [
            "wear",
            "outfit",
            "style me",
            "recommend look",
            "what should i wear",
            "dress me",
            # "I need something for X" / "looking for ... look" patterns.
            "need something for",
            "looking for a look",
            "need a look",
        ]
    ):
        return {"intent": "daily_outfit", "slots": slots, "confidence": 0.82}

    # Catch occasion-led phrasing without an explicit garment word: "for a
    # business meeting", "for date night", "for office", "for a wedding".
    _occasion_keywords = (
        "business meeting",
        "client meeting",
        "office meeting",
        "interview",
        "date night",
        "first date",
        "office",
        "wedding",
        "party",
        "brunch",
        "dinner",
        "club",
    )
    if any(kw in t for kw in _occasion_keywords) and any(
        verb in t
        for verb in ("need", "want", "suggest", "give me", "i have", "ideas")
    ):
        return {"intent": "daily_outfit", "slots": slots, "confidence": 0.8}

    daily_words = [
        "daily plan",
        "daily cards",
        "morning flow",
        "midday flow",
        "afternoon flow",
        "evening flow",
        "night flow",
        "tomorrow preview",
        "day planner",
        "daily dependency",
    ]
    if any(x in t for x in daily_words):
        return {"intent": "daily_dependency", "slots": slots, "confidence": 0.8}

    if any(x in t for x in ["try", "try on", "virtual try", "preview this"]):
        return {"intent": "try_on", "slots": slots, "confidence": 0.72}

    if any(x in t for x in ["trend", "style ideas", "inspiration", "new styles"]):
        return {"intent": "explore_styles", "slots": slots, "confidence": 0.62}

    count_words = ["how many", "count", "number of", "total", "do i have"]
    # Canonical garment vocabulary (single source: services.style_item_contract),
    # token-aware so "spring events"/"captions" never false-match "ring"/"cap".
    if any(x in t for x in count_words) and has_garment_word(
        t, extra_words=("wardrobe", "closet", "outfit", "outfits")
    ):
        return {"intent": "wardrobe_query", "slots": slots, "confidence": 0.8}

    organize_words = [
        "organize",
        "life board",
        "meal planner",
        "diet",
        "nutrition",
        "medicine",
        "meds",
        "bills",
        "calendar",
        "workout",
        "fitness",
        "gym",
        "skincare",
        "contacts",
        "life goals",
        "goals",
    ]
    # Bare "meal" is a topic word, not an ask (see has_diet_action_intent
    # above) -- it only admits Diet here alongside "meal planner"/"diet"/
    # "nutrition" or a genuine action word, matching the diet_phrases guard.
    if any(x in t for x in organize_words) or has_diet_action_intent(t):
        if "life board" in t:
            slots["module"] = "life_boards"
        elif "meal planner" in t or "diet" in t or "nutrition" in t or has_diet_action_intent(t):
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

    return {"intent": "general", "slots": slots, "confidence": 0.4}


def detect_intent(
    user_text: str, history=None, model: str | None = None
) -> Dict[str, Any]:
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
        _sub_intent = (parsed.get("slots") or {}).get("sub_intent")
        # Never let a previous style intent override a fresh help/identity
        # request — "who are you" after an outfit chat must stay general.
        if (
            parsed.get("intent") == "general"
            and _sub_intent != "help_identity"
            and last.get("intent")
        ):
            last_intent = _norm_key(last.get("intent"))
            if last_intent in _ALLOWED_INTENTS and last_intent != "general":
                parsed["intent"] = last_intent
                parsed["confidence"] = max(
                    float(parsed.get("confidence", 0.0) or 0.0), 0.6
                )

    return parsed
