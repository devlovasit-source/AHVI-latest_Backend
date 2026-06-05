from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger("ahvi.style_reasoning")

from brain.tone.tone_engine import tone_engine
from prompts.core_prompts import AHVI_SYSTEM_PROMPT
from prompts.styling_prompts import OCCASION_INTERPRETER_PROMPT
from services.ai_gateway import generate_text, parse_json_object
from services.stylist_knowledge_service import (
    COLOR_BODY_ADVICE,
    SHOPPING_ASSIST,
    STYLE_ADVICE,
    STYLE_EDUCATION,
    STYLE_PAIRING,
    WARDROBE_STYLE,
    classify_style_mode,
)

GENERAL = "general"
VISUAL_INSPIRATION = "visual_inspiration"

_STYLE_REASONING_MODES = {
    GENERAL,
    STYLE_ADVICE,
    VISUAL_INSPIRATION,
    WARDROBE_STYLE,
    SHOPPING_ASSIST,
    STYLE_EDUCATION,
    COLOR_BODY_ADVICE,
    STYLE_PAIRING,
}

_GEMINI_MODES = {
    STYLE_ADVICE,
    VISUAL_INSPIRATION,
    SHOPPING_ASSIST,
    STYLE_EDUCATION,
    COLOR_BODY_ADVICE,
    STYLE_PAIRING,
}


def _norm(value: Any) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


_RECURSIVE_PREFIXES = (
    "show visual inspiration for:",
    "show visual inspiration for",
    "use my wardrobe for:",
    "use my wardrobe for",
    "find missing pieces for:",
    "find missing pieces for",
    "show shopping ideas for:",
    "show shopping ideas for",
)


def _clean_recursive_prompt(query: str) -> str:
    """Strip stacked action prefixes so the base occasion survives instead of
    polluting the prompt as "show visual inspiration for: show visual
    inspiration for: coffee date". Keeps only the trailing real intent."""
    text = str(query or "").strip()
    changed = True
    guard = 0
    while changed and guard < 6:
        changed = False
        guard += 1
        low = text.lower()
        for pref in _RECURSIVE_PREFIXES:
            if low.startswith(pref):
                text = text[len(pref):].strip(" :·-")
                changed = True
                break
        # Collapse an internal " · " chain to its last meaningful segment.
        if " · " in text:
            tail = text.split(" · ")[-1].strip()
            if tail:
                text = tail
                changed = True
    return text or str(query or "").strip()


def _intent_name(intent: dict | str | None) -> str:
    if isinstance(intent, dict):
        return _norm(intent.get("intent"))
    return _norm(intent)


def _confidence(intent: dict | str | None, fallback: float) -> float:
    if not isinstance(intent, dict):
        return fallback
    try:
        return max(0.0, min(1.0, float(intent.get("confidence", fallback))))
    except Exception:
        return fallback


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


_PAIRING_TRIGGERS = (
    "what to pair with",
    "what goes with",
    "how do i style",
    "how to style",
    "ways to wear",
    "ways to style",
    "how can i wear",
    "what matches",
    "style this",
    "pair this with",
    "pair with",
)

_ANCHOR_COLORS = (
    "white", "black", "grey", "gray", "navy", "blue", "brown", "tan",
    "beige", "cream", "olive", "green", "red", "burgundy", "pink",
    "yellow", "gold", "silver", "charcoal",
)

_ANCHOR_CATEGORIES = {
    "shirt": ("shirt", "shirts", "button down", "button-down", "oxford"),
    "footwear": ("loafers", "loafer", "sneakers", "sneaker", "shoes", "boots", "boot", "sandals", "sandal"),
    "bottom": ("trousers", "trouser", "pants", "pant", "jeans", "denim", "chinos", "chino", "shorts"),
    "outerwear": ("blazer", "jacket", "coat", "overshirt"),
    "top": ("tee", "t shirt", "tshirt", "polo", "knit", "sweater", "hoodie", "top"),
    "dress": ("dress", "gown", "jumpsuit"),
    "ethnic": ("kurta", "saree", "sherwani", "lehenga"),
}


def _extract_pairing_anchor(query: str) -> Dict[str, str]:
    q = _norm(_clean_recursive_prompt(query))
    anchor = q
    for trigger in _PAIRING_TRIGGERS:
        if trigger in q:
            tail = q.split(trigger, 1)[1].strip()
            if tail:
                anchor = tail
                break
    anchor = re.sub(r"^(a|an|the|my|this|these|those)\s+", "", anchor).strip()
    anchor = re.sub(r"\b(casually|formally|well|better|today|outfit|look)\b", " ", anchor)
    anchor = re.sub(r"\s+", " ", anchor).strip()
    color = next((c for c in _ANCHOR_COLORS if re.search(rf"\b{re.escape(c)}\b", anchor)), "")
    if color == "gray":
        color = "grey"
    category = ""
    for cat, terms in _ANCHOR_CATEGORIES.items():
        if any(re.search(rf"\b{re.escape(term)}\b", anchor) for term in terms):
            category = cat
            break
    name = anchor or "the item"
    if color and category and category not in name:
        pass
    logger.info(
        "AHVI_STYLE_PAIRING_ANCHOR name=%r category=%s color=%s",
        name,
        category,
        color,
    )
    return {"name": name, "category": category, "color": color}


def _safe_list(value: Any, *, limit: int = 8) -> List[str]:
    if isinstance(value, list):
        out = [str(x or "").strip() for x in value if str(x or "").strip()]
    elif str(value or "").strip():
        out = [str(value).strip()]
    else:
        out = []
    return out[:limit]


def _occasion_category(query: str) -> tuple[str | None, str | None, str | None, str | None]:
    q = _norm(query)

    sensitive = (
        "funeral",
        "memorial",
        "condolence",
        "wake",
        "church",
        "temple",
        "mosque",
        "prayer",
        "religious",
        "ceremony",
        "traditional",
    )
    work = (
        "office",
        "work",
        "client",
        "pitch",
        "meeting",
        "interview",
        "conference",
        "presentation",
        "business",
    )
    social = (
        "date",
        "coffee",
        "dinner",
        "drinks",
        "party",
        "wedding",
        "brunch",
        "lunch",
        "birthday",
        "reception",
        "festival",
    )

    has_sensitive = _has_any(q, sensitive)
    has_work = _has_any(q, work)
    has_social = _has_any(q, social)

    if has_work and has_social:
        return ("hybrid_occasion", "polished", "smart-to-social", "hybrid occasion")
    if has_sensitive:
        return ("sensitive_occasion", "respectful", "polished", "sensitive occasion")
    if has_work:
        return ("work_occasion", "competent", "smart", "work occasion")
    if has_social:
        return ("social_occasion", "warm", "smart casual", "social occasion")
    if _has_any(q, ("beach", "travel", "airport", "vacation", "trip")):
        return ("travel_occasion", "practical", "casual", "travel occasion")
    return ("custom_occasion", "considered", "context-aware", "custom occasion")


def _fallback_cta(query: str) -> List[Dict[str, str]]:
    return [
        {"label": "Use my wardrobe", "value": f"Use my wardrobe for: {query}"},
        {"label": "Show visual inspiration", "value": f"Show visual inspiration for: {query}"},
        {"label": "Find missing pieces", "value": f"Show shopping ideas for: {query}"},
    ]


def _reason_for_mode(mode: str, category: str | None) -> str:
    if mode == WARDROBE_STYLE:
        return "wardrobe_request"
    if mode == VISUAL_INSPIRATION:
        return "visual_inspiration_request"
    if mode == SHOPPING_ASSIST:
        return "shopping_request"
    if mode == COLOR_BODY_ADVICE:
        return "body_color_advice"
    if mode == STYLE_EDUCATION:
        return "style_education"
    return category or "style_advice"


def _fallback_goal(mode: str, category: str | None) -> str:
    if mode == VISUAL_INSPIRATION:
        return "Turn the occasion into three clear visual directions."
    if mode == COLOR_BODY_ADVICE:
        return "Find colors and proportions that flatter without feeling overworked."
    if mode == STYLE_EDUCATION:
        return "Explain the style idea in a practical, usable way."
    if mode == SHOPPING_ASSIST:
        return "Identify the missing piece without pushing a full new outfit."
    if category == "sensitive_occasion":
        return "Look respectful, calm, and quietly put together."
    if category == "hybrid_occasion":
        return "Look professional first, with an easy shift into a social setting."
    if category == "work_occasion":
        return "Look credible first, then add just enough personality."
    if category == "social_occasion":
        return "Look approachable, confident, and comfortable in motion."
    return "Create a context-aware outfit direction that feels natural and intentional."


def _fallback_atmosphere(category: str | None) -> str:
    return {
        "sensitive_occasion": "respectful and understated",
        "hybrid_occasion": "professional first, social second",
        "work_occasion": "polished and precise",
        "social_occasion": "easy, warm, and lightly styled",
        "travel_occasion": "practical, relaxed, and prepared",
    }.get(category or "", "considered and context-aware")


def _fallback_impression(category: str | None) -> str:
    return {
        "sensitive_occasion": "understated and considerate",
        "hybrid_occasion": "competent but not stiff",
        "work_occasion": "credible and composed",
        "social_occasion": "intentional but relaxed",
        "travel_occasion": "easy and prepared",
    }.get(category or "", "considered and self-assured")


def _fallback_missing_piece(query: str, category: str | None) -> str:
    if category == "sensitive_occasion":
        return (
            "A pair of clean, closed leather shoes would anchor this and quietly "
            "carry across other formal or respectful settings."
        )
    if category in {"work_occasion", "hybrid_occasion"}:
        return (
            "A well-cut neutral blazer would do the most work here — it sharpens "
            "the look for the room and still relaxes for drinks afterward."
        )
    if category == "social_occasion":
        return (
            "A brown suede loafer would elevate this and earn its place across "
            "coffee dates, weekend dinners, and smart-casual office days."
        )
    if category == "travel_occasion":
        return (
            "One light structured layer would lift the outfit from purely "
            "practical to put-together without adding bulk."
        )
    return (
        "One refined pair of shoes is the piece that would shift this from fine "
        "to intentional, and it would carry across several other settings too."
    )


def _fallback_emotion(category: str | None) -> str:
    if category in {"work_occasion", "hybrid_occasion"}:
        return "professional"
    if category == "social_occasion":
        return "social"
    if category == "sensitive_occasion":
        return "vulnerable"
    return "neutral"


def _fallback_advice(query: str, mode: str, category: str | None) -> str:
    context = query.strip() or "this"
    if mode == VISUAL_INSPIRATION:
        return "I would frame this as three different directions so you can pick the mood before choosing pieces."
    if mode == COLOR_BODY_ADVICE:
        return (
            "For warm olive skin, stay close to colors with warmth and depth: olive, tobacco, cream, rust, "
            "warm navy, and deep teal. Very icy pastels or flat greys can work, but they need a warmer layer "
            "near the face."
        )
    if mode == STYLE_EDUCATION:
        return (
            "Think of the dress code as a signal, not a costume. Match the room first, then use fit, texture, "
            "and one deliberate detail to make it feel like you."
        )
    if mode == SHOPPING_ASSIST:
        return (
            "I would solve the missing piece first, not rebuild the whole outfit. Start with the shoe or layer "
            "that changes the formality, then choose a color that already works with what you own."
        )
    if category == "sensitive_occasion":
        return (
            "For this, keep the outfit quiet, respectful, and easy to sit or stand in for a while. Choose muted "
            "or deeper colors, clean lines, closed footwear, and minimal accessories so the clothes support the "
            "moment instead of asking for attention."
        )
    if category == "hybrid_occasion":
        return (
            "Dress for the professional room first, then build in one small switch for later. A sharp base, clean "
            "shoes, and a layer you can remove or soften after work will carry you from presentation to drinks "
            "without looking like two different outfits."
        )
    if category == "work_occasion":
        return (
            "Lead with polish: a sharper base, clean footwear, and one controlled detail. If the day moves into "
            "something social, use a layer or accessory you can soften after work."
        )
    if category == "social_occasion":
        return (
            "Keep it relaxed but intentional: one clean hero piece, comfortable footwear, and a palette that feels "
            "warm rather than loud. You want to look considered, not over-planned."
        )
    return (
        f"For {context}, read the setting first: formality, venue, culture, weather, and how much movement the "
        "day needs. Then build a clean base, choose footwear that fits the room, and keep accessories restrained."
    )


def _fallback_visual_directions(mode: str, category: str | None) -> List[Dict[str, Any]]:
    if category == "sensitive_occasion":
        return [
            {
                "title": "Quiet Formal",
                "description": "A dark, clean base with closed shoes and almost no shine.",
                "palette": ["black", "charcoal", "deep navy"],
                "pieces": ["plain shirt or kurta", "tailored trousers", "closed shoes"],
                "style_note": "Respectful, restrained, and appropriate for a serious setting.",
            },
            {
                "title": "Soft Traditional",
                "description": "Modest coverage, muted color, and a gentle fabric texture.",
                "palette": ["deep grey", "ink blue", "soft white"],
                "pieces": ["modest top", "straight trousers", "simple layer"],
                "style_note": "Keeps cultural sensitivity and comfort in balance.",
            },
            {
                "title": "Minimal Polished",
                "description": "A tonal outfit with structure but no loud details.",
                "palette": ["navy", "stone", "black"],
                "pieces": ["structured shirt", "clean bottom", "low-profile footwear"],
                "style_note": "Polished without looking dressed for attention.",
            },
        ]
    if category == "hybrid_occasion":
        return [
            {
                "title": "Presentation Base",
                "description": "Sharp shirt, tailored bottom, and serious shoes for the first room.",
                "palette": ["navy", "white", "charcoal"],
                "pieces": ["crisp shirt", "tailored trousers", "loafers"],
                "style_note": "Keep the authority in the base layer.",
            },
            {
                "title": "After-Work Softening",
                "description": "A removable layer or open collar that relaxes the same outfit.",
                "palette": ["ink", "taupe", "cream"],
                "pieces": ["light blazer", "soft knit", "sleek belt"],
                "style_note": "The transition should feel effortless, not like a costume change.",
            },
            {
                "title": "One Social Detail",
                "description": "A textured accessory or richer color that reads well in evening light.",
                "palette": ["slate", "burgundy", "black"],
                "pieces": ["tonal shirt", "structured trousers", "subtle accessory"],
                "style_note": "One relaxed detail is enough after a client-facing day.",
            },
        ]
    if category == "work_occasion":
        return [
            {
                "title": "Sharp Day Base",
                "description": "A crisp top, tailored bottom, and professional footwear.",
                "palette": ["navy", "white", "charcoal"],
                "pieces": ["crisp shirt", "tailored trousers", "loafers"],
                "style_note": "Credible first, with a clean line from meeting to commute.",
            },
            {
                "title": "Desk To Drinks",
                "description": "Work polish with one softer evening detail.",
                "palette": ["ink", "taupe", "cream"],
                "pieces": ["lightweight blazer", "knit or shirt", "sleek shoes"],
                "style_note": "Remove the layer or open the collar after work to relax it.",
            },
            {
                "title": "Quiet Authority",
                "description": "Tonal dressing with minimal accessories and stronger shoes.",
                "palette": ["slate", "black", "steel blue"],
                "pieces": ["tonal top", "structured bottom", "watch or belt"],
                "style_note": "Precise, not flashy.",
            },
        ]
    if mode == COLOR_BODY_ADVICE:
        return [
            {
                "title": "Warm Depth",
                "description": "Warm, earthy colors placed near the face.",
                "palette": ["olive", "cream", "rust"],
                "pieces": ["olive shirt", "cream layer", "brown footwear"],
                "style_note": "Adds warmth without washing out olive undertones.",
            },
            {
                "title": "Grounded Contrast",
                "description": "A deeper neutral base with one warm accent.",
                "palette": ["warm navy", "camel", "ivory"],
                "pieces": ["navy base", "camel layer", "ivory top"],
                "style_note": "Keeps contrast clean but not harsh.",
            },
            {
                "title": "Muted Statement",
                "description": "A controlled color moment with soft neutrals around it.",
                "palette": ["teal", "stone", "tobacco"],
                "pieces": ["teal top", "stone trouser", "tobacco shoe"],
                "style_note": "Color reads intentional rather than loud.",
            },
        ]
    return [
        {
            "title": "Relaxed Oxford",
            "description": "Oxford shirt, clean denim, and easy footwear.",
            "palette": ["navy", "white", "tan"],
            "pieces": ["Oxford shirt", "dark denim", "clean sneakers"],
            "style_note": "Approachable and tidy without feeling formal.",
        },
        {
            "title": "Knit Polo Polish",
            "description": "A knit top with sharper trousers and refined shoes.",
            "palette": ["cream", "olive", "brown"],
            "pieces": ["knit polo", "straight trousers", "loafers"],
            "style_note": "Soft texture makes the polish feel relaxed.",
        },
        {
            "title": "Soft Layered Casual",
            "description": "Light layer over a simple base with grounded footwear.",
            "palette": ["stone", "blue", "charcoal"],
            "pieces": ["light jacket", "plain tee", "chinos"],
            "style_note": "Useful when the setting might shift.",
        },
    ]


def _coerce_mode(query: str, intent: dict | str | None, context: dict | None) -> str:
    """Resolve the style mode with a HARD precedence so a wardrobe action
    never loses to a visual-inspiration phrase that happens to sit in the
    same string (e.g. chip value
    "Use my wardrobe for: show visual inspiration for coffee date").

        use_wardrobe > find_missing_pieces > visual_inspiration > style_advice
    """
    ctx = context if isinstance(context, dict) else {}
    style_action = _norm(ctx.get("style_action"))
    next_action = _norm(ctx.get("next_action"))
    module_context = str(ctx.get("module_context") or ctx.get("module") or "")
    # _norm strips underscores ("style_advice" -> "style advice"); restore them
    # so the intent name matches the mode constants.
    intent_value = _intent_name(intent).replace(" ", "_")
    q = _norm(query)
    action_blob = f"{q} {style_action} {next_action}"

    # 1) use_wardrobe — highest precedence.
    if (
        module_context.lower() in {"wardrobe", "closet"}
        or _has_any(action_blob, ("use_wardrobe", "use my wardrobe", "use wardrobe", "from my wardrobe", "with my wardrobe"))
    ):
        logger.info("AHVI_STYLE_ROUTE_FORCED mode=wardrobe_style reason=use_wardrobe")
        return WARDROBE_STYLE

    # 2) find_missing_pieces.
    if _has_any(action_blob, ("find_missing_pieces", "find missing pieces", "missing piece", "missing pieces", "what should i buy", "shopping ideas", "complete the look", "complete this look", "find this", "find similar", "shop this", "buy similar")):
        logger.info("AHVI_STYLE_ROUTE_FORCED mode=missing_pieces reason=find_missing_pieces")
        return SHOPPING_ASSIST

    # 3) visual_inspiration.
    if _has_any(action_blob, ("show visual inspiration", "visual inspiration", "generate moodboard", "show moodboard", "moodboard for")):
        return VISUAL_INSPIRATION

    # 4) explicit intent, else classify (style_advice default).
    if intent_value in _STYLE_REASONING_MODES:
        return intent_value
    return (
        classify_style_mode(
            query,
            module_context=module_context,
            style_action=str(ctx.get("style_action") or ""),
        )
        or GENERAL
    )


def _build_reasoning_prompt(
    *,
    query: str,
    mode: str,
    category: str | None,
    user_profile: dict,
    context: dict,
    policy: dict | None = None,
    style_ctx: dict | None = None,
    persona: dict | None = None,
    archetypes: list | None = None,
) -> str:
    policy = policy or {}
    style_ctx = style_ctx or {}
    persona = persona or {}
    archetypes = archetypes or []
    anchor = _extract_pairing_anchor(query) if mode == STYLE_PAIRING else {}
    if mode == STYLE_PAIRING:
        import json as _json

        gender = str(persona.get("gender_profile") or "unknown")
        archetype_names = [a.get("name") for a in archetypes if isinstance(a, dict)]
        _arch_compact = [
            {
                "name": a.get("name"),
                "impression": a.get("impression"),
                "preferred_items": a.get("preferred_items"),
                "avoid_items": a.get("avoid_items"),
                "palette": a.get("palette"),
            }
            for a in archetypes
            if isinstance(a, dict)
        ]
        _arch_json = _json.dumps(_arch_compact, ensure_ascii=False)
        _persona_json = _json.dumps(persona, ensure_ascii=False)
        return f"""
{AHVI_SYSTEM_PROMPT}

You are AHVI's PERSONAL stylist (not a generic fashion encyclopedia). The user
is asking an open-ended pairing question. Ground every route in their persona +
the selected archetypes below.

Persona context (obey — never assume beyond it):
{_persona_json}

Selected archetypes (build routes ONLY from these — do not invent others):
{_arch_json}

Return ONLY valid JSON matching this schema:
{{
  "mode": "style_pairing",
  "anchor_item": {{"name": string, "category": string, "color": string}},
  "stylist_reasoning": string,
  "pairing_routes": [
    {{
      "title": string,
      "archetype": string,
      "impression_created": string,
      "use_case": string,
      "strategy": string,
      "items": [string],
      "palette": [string],
      "why_it_works": string,
      "avoid": [string],
      "styling_tip": string,
      "persona_fit_reason": string
    }}
  ],
  "what_to_avoid": [string],
  "next_actions": ["Use my wardrobe", "Show visual inspiration", "Find missing pieces"],
  "follow_up_question": string|null,
  "confidence": float
}}

Rules:
- Return 4-5 pairing_routes, each mapped to a DIFFERENT selected archetype
  (set route.archetype to that archetype's exact name). Use evocative titles,
  never "Option 1" / "Casual Look".
- Persona gender = {gender}. If male: NEVER suggest skirts, dresses, camisoles,
  heels, or feminine-only silhouettes (unless the user explicitly asked). If
  female: feminine routes allowed, still respect style DNA. If unknown: keep
  items gender-neutral (trousers, denim, chinos, shirts, polos, knitwear,
  overshirts, jackets, sneakers, loafers, boots).
- Do not mention gender unless relevant. Never assume beyond persona context.
- persona_fit_reason: one line on why this route suits THIS user.
- Include avoid guidance per route. Do not generate wardrobe boards, images, or
  shopping links. Do not sound like a textbook.
- Allowed archetype names: {archetype_names}

Known deterministic mode: style_pairing
Detected anchor_item: {anchor}
User query: {_clean_recursive_prompt(query)}
"""
    return f"""
{AHVI_SYSTEM_PROMPT}

{OCCASION_INTERPRETER_PROMPT}

You are AHVI's senior stylist — a real human stylist thinking out loud, not a
fashion database listing templates.

Before recommending any clothing, decide in this order:
1. What impression should the user create in this exact moment?
2. What social outcome actually matters here?
3. What styling strategy best fits the moment?
4. What styling risk should be avoided, and why?
5. What atmosphere should the outfit communicate?
6. What single missing piece would most improve this direction?

Lead with the opinion and the reasoning. The outfit directions only support
that reasoning — they never replace it.

Return ONLY valid JSON matching this schema:
{{
  "mode": "style_advice | visual_inspiration | color_body_advice | style_education | shopping_assist",
  "occasion": string|null,
  "goal": string,
  "impression": string,
  "atmosphere": string,
  "confidence_strategy": string,
  "emotion_state": "neutral | excited | frustrated | vulnerable | professional | social",
  "stylist_reasoning": string,
  "what_to_avoid": [string],
  "missing_piece_reasoning": string,
  "visual_directions": [
    {{
      "title": string,
      "strategy": string,
      "description": string,
      "palette": [string],
      "pieces": [string],
      "why_it_works": string,
      "board_brief": {{"hero": string, "support": string, "footwear": string, "accent": string}},
      "style_note": string
    }}
  ],
  "missing_piece": {{"name": string, "category": string, "reason": string, "unlocks": [string]}},
  "visual_inspiration_board": {{"title": string, "aesthetic": string, "mood": string, "palette": [string], "hero_piece": string, "silhouette": string, "styling_notes": string}},
  "follow_up_question": string|null,
  "confidence": float
}}

confidence_strategy = one line on how to make the user feel confident in this
specific moment (what to lean into, what to reassure).
board_brief = a compact brief the board renderer can visualize: which piece is
the hero, what supports it, the footwear, and the one accent.

Writing rules for stylist_reasoning:
- Speak like a stylist explaining a decision. Use phrasing such as
  "This works because...", "I would avoid...", "The priority here is...".
- Explain the social strategy first, then the clothing logic.
- KEEP IT SHORT: 35-60 words for a normal occasion, max 70 for multi-event.
  No markdown, no headings, no "**Core:**" / "**Presentation Layer:**" labels,
  no bullet lists. One tight paragraph.

Wardrobe grounding: if the style context shows wardrobe_available with items,
do NOT describe a garment as owned ("wear your suit") unless that garment
appears in the wardrobe list. Any garment the user does not own goes ONLY in
missing_piece, clearly marked as a suggestion to acquire — never implied owned.

Obey the policy.outfit_validation_principles: if the occasion and an item
clash (shiny/formal for coffee, office polish forced into date night, formal
shirt for a game), do NOT overconfidently recommend it — name the mismatch
softly in what_to_avoid and stylist_reasoning, then offer one correction.

Memory: if style_ctx.memory.recently_worn is non-empty, you MAY add ONE short
clause about rotating in a fresher option (e.g. "since you wore the navy shirt
recently, I'm rotating in something fresher"). Do NOT over-explain. If
style_ctx.memory is null/absent, NEVER invent or mention wear history.

Personalization: if style_ctx.style_dna is present, ground the reasoning in it
naturally ("your wardrobe leans relaxed-modern", lean into preferred_colors /
preferred_silhouettes, respect avoided_colors + avoid_style_keywords). If
style_dna is absent or empty, do NOT invent personal taste — stay occasion-led.

Obey the policy.wardrobe_management_principles for weak/empty wardrobes: say
what the wardrobe leans toward first (e.g. office/casual), acknowledge what CAN
be built, then frame gaps as one or two occasion-specific anchors to ADD
(e.g. "a relaxed evening shirt and a softer shoe") — never a long missing list,
never a blunt "I don't see options".

When visual_inspiration: shape visual_inspiration_board from
policy.mood_board_contract (pick aesthetic from its aesthetic_taxonomy, mood +
keywords from its emotion_mapping) and end with a clear next action +
missing_piece per policy.inspiration_board_contract.

For visual_inspiration mode also fill visual_inspiration_board:
{{"title","aesthetic","mood","palette":[],"hero_piece","silhouette","styling_notes"}}.

Each visual_direction.why_it_works must explain the STYLING LOGIC, not just
restate the pieces. Bad: "Oxford shirt with denim." Good: "The shirt creates
structure while the denim keeps it approachable."

missing_piece_reasoning must justify ONE piece and where else it earns its
place. Bad: "Brown loafers." Good: "A brown suede loafer would elevate this
and carry across coffee dates, weekend dinners, and smart-casual office days."

Ban this generic filler unless genuinely unavoidable:
"balanced silhouette", "color harmony", "approachable and tidy",
"elevated aesthetic", "perfect for". Replace with a real reason.

Hard rules:
- Do not generate image prompts or real images.
- For style_advice and visual_inspiration, return exactly 3 visual_directions.
- Each direction must differ by mood, silhouette, palette, or formality.
- For "Show visual inspiration", make the cards the main response.
- Do not open with "Here are styling principles". Do not sound like a textbook.
- Different occasions MUST produce clearly different goal/impression/avoid.
  Christian funeral: respectful presence, understated, no bright/flashy.
  Coffee date: approachable confidence, intentional not corporate.
  Client presentation + drinks: credibility first then social ease,
  avoid full formal that feels awkward later (transitional dressing).
  Wedding guest: celebratory restraint, festive without competing, no bridal.
  Beach dinner: relaxed evening polish, no swimwear/flip-flops after sunset.
- Wardrobe styling is not allowed in this response.
- MULTI-EVENT / TRANSITION: if style context has sub_occasions or a
  style_strategy, the user is dressing for MORE THAN ONE event in sequence.
  Do NOT collapse it to a single occasion (never "date night" for a game +
  dinner). Reason about the transition: dress for the first event's needs
  (e.g. comfort + movement for a game), then make the later event feel
  intentional without a full outfit change. goal/impression must reflect the
  transition, and what_to_avoid should flag anything that fails either event.

Style policy (compact — obey, do not echo verbatim):
{policy}

Style context (compact — the user's real situation):
{style_ctx}

Known deterministic mode: {mode}
Detected category: {category or "unknown"}
User query: {query}
"""


def _normalize_direction(value: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(value) if isinstance(value, dict) else {}
    return {
        "title": str(item.get("title") or fallback.get("title") or "Style Direction").strip(),
        "strategy": str(item.get("strategy") or fallback.get("strategy") or "").strip(),
        "description": str(item.get("description") or fallback.get("description") or "").strip(),
        "palette": _safe_list(item.get("palette") or fallback.get("palette"), limit=5),
        "pieces": _safe_list(item.get("pieces") or fallback.get("pieces"), limit=6),
        "why_it_works": str(
            item.get("why_it_works") or fallback.get("why_it_works") or ""
        ).strip(),
        "board_brief": item.get("board_brief") if isinstance(item.get("board_brief"), dict) else {},
        "style_note": str(item.get("style_note") or fallback.get("style_note") or "").strip(),
    }


def _ensure_direction_logic(direction: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee why_it_works + strategy so even fallback cards read like a
    stylist explaining logic, not a database listing pieces."""
    pieces = direction.get("pieces") or []
    title = str(direction.get("title") or "this direction").strip()
    if not direction.get("why_it_works"):
        if len(pieces) >= 2:
            direction["why_it_works"] = (
                f"The {str(pieces[0]).lower()} sets the structure while the "
                f"{str(pieces[1]).lower()} keeps it easy — so the look reads "
                "intentional without trying too hard."
            )
        else:
            direction["why_it_works"] = (
                f"{title} keeps one clear focal point so the outfit feels "
                "deliberate rather than busy."
            )
    if not direction.get("strategy"):
        direction["strategy"] = str(direction.get("style_note") or "").strip()
    return direction


def _normalize_visual_directions(value: Any, mode: str, category: str | None) -> List[Dict[str, Any]]:
    fallbacks = _fallback_visual_directions(mode, category)
    rows = value if isinstance(value, list) else []
    return [
        _ensure_direction_logic(
            _normalize_direction(rows[idx] if idx < len(rows) else {}, fallbacks[idx])
        )
        for idx in range(3)
    ]


def _fallback_pairing_routes(anchor: Dict[str, str]) -> List[Dict[str, Any]]:
    name = str(anchor.get("name") or "the item").strip()
    category = str(anchor.get("category") or "").strip()
    color = str(anchor.get("color") or "").strip()
    base = name if name != "the item" else "this piece"
    if category == "footwear" or any(x in name for x in ("loafer", "sneaker", "shoe", "boot")):
        return [
            {
                "title": "Smart Casual",
                "use_case": "office-adjacent days",
                "strategy": "Use the shoes to sharpen relaxed separates.",
                "items": [base, "button-down shirt", "chinos", "simple watch"],
                "palette": [color or "black", "white", "navy", "tan"],
                "why_it_works": "The footwear adds polish, while chinos keep it from becoming fully formal.",
                "avoid": ["Overly shiny belts", "matching everything in the same dark tone"],
                "styling_tip": "Let the trouser hem sit cleanly on the shoe.",
            },
            {
                "title": "Office Clean",
                "use_case": "meetings and workdays",
                "strategy": "Lean into structure without going stiff.",
                "items": [base, "crisp shirt", "tailored trousers", "belt"],
                "palette": [color or "black", "blue", "grey"],
                "why_it_works": "A structured base makes the footwear feel intentional and credible.",
                "avoid": ["Shorts", "athletic socks", "loud patterned trousers"],
                "styling_tip": "Match the belt mood, not necessarily the exact color.",
            },
            {
                "title": "Evening Minimal",
                "use_case": "dinner or drinks",
                "strategy": "Keep the outfit tonal and let texture do the work.",
                "items": [base, "dark shirt", "straight trousers"],
                "palette": [color or "black", "charcoal", "cream"],
                "why_it_works": "The darker palette gives evening polish without needing a statement piece.",
                "avoid": ["Corporate blazer unless the setting is formal"],
                "styling_tip": "Open the collar slightly to soften the polish.",
            },
            {
                "title": "Weekend Neat",
                "use_case": "casual plans",
                "strategy": "Use one clean top so the shoes do not feel overdressed.",
                "items": [base, "plain tee or polo", "denim", "light overshirt"],
                "palette": [color or "black", "stone", "blue"],
                "why_it_works": "Casual layers make polished footwear feel relaxed and wearable.",
                "avoid": ["Gym shorts", "wrinkled oversized tees"],
                "styling_tip": "Choose denim with a clean wash rather than heavy distressing.",
            },
        ]
    if "blazer" in name or category == "outerwear":
        return [
            {
                "title": "T-Shirt Tailoring",
                "use_case": "casual dinner",
                "strategy": "Break the blazer's corporate signal with a clean tee.",
                "items": [base, "plain tee", "straight denim", "minimal sneakers"],
                "palette": [color or "navy", "white", "blue"],
                "why_it_works": "The tee and sneakers relax the blazer while the jacket keeps shape.",
                "avoid": ["Dress shirt plus formal trousers if you want casual"],
                "styling_tip": "Keep the tee neckline clean and the blazer unbuttoned.",
            },
            {
                "title": "Knit Softness",
                "use_case": "smart casual weekends",
                "strategy": "Swap office shirting for soft texture.",
                "items": [base, "fine knit", "chinos", "suede loafers"],
                "palette": [color or "navy", "cream", "brown"],
                "why_it_works": "Knitwear makes the blazer feel warm and easy instead of corporate.",
                "avoid": ["Tie", "stiff dress shoes"],
                "styling_tip": "Use a thinner knit so the shoulder line stays smooth.",
            },
            {
                "title": "Denim Contrast",
                "use_case": "creative casual",
                "strategy": "Let denim pull the blazer down a notch.",
                "items": [base, "casual shirt", "dark denim", "clean sneakers"],
                "palette": [color or "charcoal", "blue", "white"],
                "why_it_works": "Denim adds ease while the blazer keeps the outfit intentional.",
                "avoid": ["Ripped denim with a formal blazer"],
                "styling_tip": "Keep the denim straight or slim, not baggy.",
            },
            {
                "title": "Summer Relaxed",
                "use_case": "warm evenings",
                "strategy": "Pair the blazer with breathable pieces.",
                "items": [base, "linen shirt", "light chinos", "loafers"],
                "palette": [color or "navy", "ecru", "tan"],
                "why_it_works": "Light fabrics stop the blazer from feeling too serious.",
                "avoid": ["Heavy wool trousers"],
                "styling_tip": "Roll sleeves only if the blazer fabric is relaxed enough.",
            },
        ]
    return [
        {
            "title": "Smart Casual",
            "use_case": "office-adjacent or polished daily",
            "strategy": "Use clean structure around the anchor.",
            "items": [base, "chinos or tailored trousers", "loafers or minimal sneakers", "simple watch"],
            "palette": [color or "white", "navy", "tan"],
            "why_it_works": "The base feels intentional without pushing the outfit into full formal.",
            "avoid": ["Too many statement accessories", "overly shiny shoes"],
            "styling_tip": "Keep one piece relaxed so the outfit stays modern.",
        },
        {
            "title": "Business Casual",
            "use_case": "meetings and workdays",
            "strategy": "Add sharper bottoms and restrained footwear.",
            "items": [base, "structured trousers", "belt", "loafers"],
            "palette": [color or "white", "grey", "brown"],
            "why_it_works": "The structured pieces make the anchor read professional rather than plain.",
            "avoid": ["Loud prints near the anchor"],
            "styling_tip": "Tuck only if the trouser waistband looks clean.",
        },
        {
            "title": "Weekend Clean",
            "use_case": "coffee, errands, casual lunch",
            "strategy": "Relax the anchor with denim and easy footwear.",
            "items": [base, "straight denim", "clean sneakers", "light overshirt"],
            "palette": [color or "white", "blue", "stone"],
            "why_it_works": "Denim makes the anchor feel approachable while the clean shoe keeps polish.",
            "avoid": ["Distressed denim if the anchor is already crisp"],
            "styling_tip": "Leave a little ease in the fit.",
        },
        {
            "title": "Evening Minimal",
            "use_case": "dinner or drinks",
            "strategy": "Use contrast and a darker base.",
            "items": [base, "dark trousers", "sleek shoes", "minimal accessory"],
            "palette": [color or "white", "black", "charcoal"],
            "why_it_works": "The darker pieces make the anchor feel deliberate and evening-ready.",
            "avoid": ["Office-heavy layering"],
            "styling_tip": "Keep accessories quiet so the contrast does the work.",
        },
        {
            "title": "Summer Relaxed",
            "use_case": "warm days or vacations",
            "strategy": "Pair the anchor with breathable textures.",
            "items": [base, "linen or cotton bottom", "sandals or canvas sneakers"],
            "palette": [color or "white", "ecru", "olive"],
            "why_it_works": "Lighter texture keeps the anchor fresh and relaxed.",
            "avoid": ["Heavy formal trousers in heat"],
            "styling_tip": "Use softer colors if the fabric is crisp.",
        },
    ]


def _normalize_pairing_routes(value: Any, anchor: Dict[str, str], gender: str = "unknown") -> List[Dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    fallbacks = _fallback_pairing_routes(anchor)
    normalized: List[Dict[str, Any]] = []
    total_removed = 0
    for idx in range(min(5, max(4, len(rows), len(fallbacks)))):
        source = rows[idx] if idx < len(rows) and isinstance(rows[idx], dict) else {}
        fb = fallbacks[idx % len(fallbacks)]
        items = _safe_list(source.get("items") or fb.get("items"), limit=8)
        avoid = _safe_list(source.get("avoid") or fb.get("avoid"), limit=5)
        # Deterministic persona safety net: strip feminine-only items for a male
        # persona even if Gemini slipped one in.
        try:
            from services.stylist_knowledge_service import filter_items_for_persona

            items, removed = filter_items_for_persona(items, gender)
            if removed:
                total_removed += len(removed)
                avoid = (avoid + removed)[:6]
        except Exception:  # noqa: BLE001
            pass
        route = {
            "title": str(source.get("title") or fb.get("title") or "Pairing Route").strip(),
            "archetype": str(source.get("archetype") or "").strip(),
            "impression_created": str(source.get("impression_created") or "").strip(),
            "use_case": str(source.get("use_case") or source.get("useCase") or fb.get("use_case") or "").strip(),
            "strategy": str(source.get("strategy") or fb.get("strategy") or "").strip(),
            "items": items,
            "palette": _safe_list(source.get("palette") or fb.get("palette"), limit=6),
            "why_it_works": str(source.get("why_it_works") or source.get("whyItWorks") or fb.get("why_it_works") or "").strip(),
            "avoid": avoid,
            "styling_tip": str(source.get("styling_tip") or source.get("style_note") or source.get("styleNote") or fb.get("styling_tip") or "").strip(),
            "persona_fit_reason": str(source.get("persona_fit_reason") or "").strip(),
        }
        normalized.append(route)
    if total_removed:
        logger.info("AHVI_PAIRING_PERSONA_FILTER_APPLIED gender=%s removed=%d", gender, total_removed)
    logger.info("AHVI_PAIRING_ROUTES_BUILT count=%d titles=%s", len(normalized), [r["title"] for r in normalized])
    return normalized[:5]


def _pairing_routes_as_visual_directions(routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "title": route.get("title"),
            "strategy": route.get("strategy"),
            "description": route.get("use_case"),
            "palette": route.get("palette") if isinstance(route.get("palette"), list) else [],
            "pieces": route.get("items") if isinstance(route.get("items"), list) else [],
            "why_it_works": route.get("why_it_works"),
            "style_note": route.get("styling_tip"),
            "use_case": route.get("use_case"),
            "avoid": route.get("avoid") if isinstance(route.get("avoid"), list) else [],
        }
        for route in routes
    ]


def _gemini_reasoning(
    *,
    query: str,
    mode: str,
    category: str | None,
    user_profile: dict,
    context: dict,
) -> Dict[str, Any]:
    # Compact policy + style context (never the full rule libraries).
    policy: Dict[str, Any] = {}
    style_ctx: Dict[str, Any] = {}
    try:
        from brain.config_loader import get_style_policy_context

        policy = get_style_policy_context(
            intent=mode, occasion=str(context.get("occasion") or category or ""), mode=mode
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ahvi.style.policy_failed err=%s", str(exc)[:140])
    try:
        from services.style_context_service import build_style_context, compact_context_for_prompt

        full_ctx = build_style_context(
            query=query,
            occasion=str(context.get("occasion") or category or "") or None,
            mode=mode if mode in {"style_advice", "visual_inspiration", "wardrobe_style", "missing_pieces"} else "style_advice",
            wardrobe_items=context.get("wardrobe") or context.get("wardrobe_items"),
            weather=context.get("weather") or context.get("weather_context"),
            event_context=context.get("event_context"),
            user_profile=user_profile,
            last_style_context=context.get("last_style_context"),
            user_id=str(context.get("user_id") or ""),
        )
        style_ctx = compact_context_for_prompt(full_ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ahvi.style.context_failed err=%s", str(exc)[:140])

    # Wire the 4 intelligence configs as compact principle slices into the
    # policy block (outfit_validation + wardrobe_management always; mood +
    # inspiration contracts only for visual_inspiration).
    try:
        from brain import config_loader as _cfg

        policy["outfit_validation_principles"] = _cfg.get_outfit_validation_principles()
        policy["wardrobe_management_principles"] = _cfg.get_wardrobe_management_principles()
        policy["personalization_principles"] = _cfg.get_personalization_principles()
        policy["visual_response_principles"] = _cfg.get_visual_response_principles()
        policy["decision_principles"] = _cfg.get_decision_principles()
        if mode == VISUAL_INSPIRATION:
            policy["mood_board_contract"] = _cfg.get_mood_board_contract()
            policy["inspiration_board_contract"] = _cfg.get_inspiration_board_contract()
        _slices = [k for k in (
            "outfit_validation_principles", "wardrobe_management_principles",
            "personalization_principles", "visual_response_principles",
            "decision_principles", "mood_board_contract", "inspiration_board_contract",
        ) if k in policy]
        logger.info("AHVI_CONFIG_USAGE_AUDIT mode=%s slices=%s", mode, _slices)
        logger.info("AHVI_CONFIG_SLICES_USED count=%d slices=%s", len(_slices), _slices)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ahvi.style.config_slices_failed err=%s", str(exc)[:140])

    # Persona + archetype selection for the pairing path (personal stylist).
    persona = {}
    selected_archetypes = []
    if mode == STYLE_PAIRING:
        try:
            from services.style_context_service import build_pairing_persona
            from services.stylist_knowledge_service import select_archetypes

            _uprof = user_profile if isinstance(user_profile, dict) else {}
            persona = build_pairing_persona(
                user_profile=_uprof,
                style_dna=_uprof.get("style_dna") or _uprof.get("styleDNA"),
                wardrobe_summary=(style_ctx or {}).get("wardrobe_summary"),
            )
            _anchor = _extract_pairing_anchor(query)
            selected_archetypes = select_archetypes(
                anchor=_anchor,
                occasion=str(context.get("occasion") or category or ""),
                style_keywords=persona.get("style_dna") or [],
            )
            logger.info(
                "AHVI_PERSONAL_STYLIST_CONTEXT_BUILT gender=%s archetypes=%s",
                persona.get("gender_profile"), [a.get("name") for a in selected_archetypes],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ahvi.pairing_persona_failed err=%s", str(exc)[:140])

    prompt = _build_reasoning_prompt(
        query=_clean_recursive_prompt(query),
        mode=mode,
        category=category,
        user_profile=user_profile,
        context=context,
        policy=policy,
        style_ctx=style_ctx,
        persona=persona,
        archetypes=selected_archetypes,
    )
    raw = generate_text(
        prompt,
        options={"temperature": 0.45, "max_output_tokens": 1600},
        user_profile=user_profile,
        signals={"context_mode": "style_reasoning", "style_mode": mode},
        usecase="style_reasoning",
    )
    logger.info("AHVI_STYLE_GEMINI_RAW_LEN usecase=style_reasoning len=%d", len(str(raw or "")))
    parsed = parse_json_object(raw)
    if isinstance(parsed, dict):
        logger.info(
            "AHVI_STYLE_GEMINI_PARSED_KEYS keys=%s",
            ",".join(sorted(parsed.keys()))[:240],
        )
    return parsed


def _build_missing_piece(payload: Dict[str, Any], reasoning_text: str) -> Dict[str, Any] | None:
    mp = payload.get("missing_piece")
    if isinstance(mp, dict) and str(mp.get("name") or "").strip():
        return {
            "name": str(mp.get("name") or "").strip(),
            "category": str(mp.get("category") or "").strip(),
            "reason": str(mp.get("reason") or reasoning_text or "").strip(),
            "unlocks": [str(u).strip() for u in (mp.get("unlocks") or []) if str(u).strip()][:6],
        }
    return None


def _build_visual_inspiration_board(
    payload: Dict[str, Any],
    visual_directions: List[Dict[str, Any]],
    goal: str,
    impression: str,
    missing_piece: Dict[str, Any] | None,
    query: str,
) -> Dict[str, Any]:
    """Premium visual-inspiration metadata block. Builds an image_prompt for a
    future generation step, but does NOT generate images yet."""
    direct = payload.get("visual_inspiration_board")
    direct = direct if isinstance(direct, dict) else {}
    first = visual_directions[0] if visual_directions else {}
    palette = first.get("palette") if isinstance(first.get("palette"), list) else []
    pieces = first.get("pieces") if isinstance(first.get("pieces"), list) else []
    board = {
        "type": "visual_inspiration_board",
        "title": str(direct.get("title") or first.get("title") or "Style Inspiration").strip(),
        "aesthetic": str(direct.get("aesthetic") or first.get("strategy") or "").strip(),
        "mood": str(direct.get("mood") or impression or "").strip(),
        "palette": [str(p).strip() for p in (direct.get("palette") or palette) if str(p).strip()][:6],
        "hero_piece": str(direct.get("hero_piece") or (pieces[0] if pieces else "")).strip(),
        "silhouette": str(direct.get("silhouette") or "").strip(),
        "styling_notes": str(
            direct.get("styling_notes") or first.get("why_it_works") or first.get("style_note") or goal
        ).strip(),
        "missing_piece": missing_piece,
    }
    board["image_prompt"] = _build_inspiration_image_prompt(board, query)
    board["inspiration_image_url"] = ""
    board["image_status"] = "not_generated"
    return board


def _build_inspiration_image_prompt(board: Dict[str, Any], query: str) -> str:
    """Editorial moodboard image prompt from the inspiration metadata.
    Generation is wired later (Imagen/Flux) — this only builds the prompt."""
    occ = _clean_recursive_prompt(query).strip() or "this occasion"
    parts = [f"Editorial fashion moodboard for {occ}."]
    if board.get("aesthetic"):
        parts.append(f"Aesthetic: {board['aesthetic']}.")
    if board.get("mood"):
        parts.append(f"Mood: {board['mood']}.")
    if board.get("palette"):
        parts.append(f"Palette: {', '.join(board['palette'])}.")
    if board.get("hero_piece"):
        parts.append(f"Hero piece: {board['hero_piece']}.")
    if board.get("silhouette"):
        parts.append(f"Silhouette: {board['silhouette']}.")
    parts.append("No faces. No text. Pinterest-style board.")
    return " ".join(parts)


def _compact_reasoning(text: str, *, multi_event: bool = False) -> str:
    """Trim stylist_reasoning for the UI: strip markdown headings/bold, collapse
    whitespace, cap at ~60 words (70 for multi-event) on a sentence boundary."""
    raw = str(text or "").strip()
    if not raw:
        return raw
    # Strip markdown headings, bold labels like "**Core:**", bullets.
    raw = re.sub(r"\*\*[^*]+\*\*:?", "", raw)
    raw = re.sub(r"(?m)^#{1,6}\s*", "", raw)
    raw = re.sub(r"(?m)^\s*[-*•]\s*", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" :-•")
    cap = 70 if multi_event else 60
    words = raw.split()
    if len(words) <= cap:
        compacted = raw
    else:
        clipped = " ".join(words[:cap])
        # End on the last sentence boundary if one exists in range.
        m = list(re.finditer(r"[.!?]", clipped))
        compacted = clipped[: m[-1].end()] if m else clipped.rstrip(",;: ") + "."
    logger.info(
        "AHVI_REASONING_COMPACTED words_in=%d words_out=%d multi_event=%s",
        len(words), len(compacted.split()), multi_event,
    )
    return compacted


def _coerce_emotion(value: Any, category: str | None) -> str:
    emotion = _norm(value)
    if emotion in {"neutral", "excited", "frustrated", "vulnerable", "professional", "social"}:
        return emotion
    return _fallback_emotion(category)


def _coerce_ai_mode(value: Any, fallback: str) -> str:
    mode = _norm(value).replace(" ", "_")
    if mode in _GEMINI_MODES:
        return mode
    return fallback if fallback in _GEMINI_MODES else STYLE_ADVICE


def _build_response(
    *,
    query: str,
    mode: str,
    category: str | None,
    tone: str | None,
    formality: str | None,
    occasion: str | None,
    confidence: float,
    ai_payload: Dict[str, Any] | None,
    user_profile: dict,
    context: dict,
) -> Dict[str, Any]:
    payload = ai_payload if isinstance(ai_payload, dict) else {}
    final_mode = mode if mode in _GEMINI_MODES else _coerce_ai_mode(payload.get("mode"), mode)
    pairing_anchor = _extract_pairing_anchor(query) if final_mode == STYLE_PAIRING else {}
    pairing_gender = "unknown"
    if final_mode == STYLE_PAIRING:
        raw_anchor = payload.get("anchor_item") if isinstance(payload.get("anchor_item"), dict) else {}
        pairing_anchor = {
            "name": str(raw_anchor.get("name") or pairing_anchor.get("name") or "").strip(),
            "category": str(raw_anchor.get("category") or pairing_anchor.get("category") or "").strip(),
            "color": str(raw_anchor.get("color") or pairing_anchor.get("color") or "").strip(),
        }
        try:
            from services.style_context_service import _resolve_gender

            pairing_gender = _resolve_gender(user_profile if isinstance(user_profile, dict) else {})
        except Exception:  # noqa: BLE001
            pairing_gender = "unknown"
    goal = str(payload.get("goal") or _fallback_goal(final_mode, category)).strip()
    impression = str(payload.get("impression") or _fallback_impression(category)).strip()
    atmosphere = str(payload.get("atmosphere") or _fallback_atmosphere(category)).strip()
    emotion_state = _coerce_emotion(payload.get("emotion_state"), category)
    # stylist_reasoning leads. Accept legacy stylist_advice as alias so older
    # Gemini responses and tests keep working.
    raw_advice = str(
        payload.get("stylist_reasoning")
        or payload.get("stylist_advice")
        or _fallback_advice(query, final_mode, category)
    ).strip()
    missing_piece_reasoning = str(
        payload.get("missing_piece_reasoning") or _fallback_missing_piece(query, category)
    ).strip()
    confidence_strategy = str(
        payload.get("confidence_strategy")
        or "Lean into what already fits well and keep one deliberate detail — "
        "confidence reads as ease, not effort."
    ).strip()
    polished_advice = tone_engine.apply(
        raw_advice,
        user_profile=user_profile,
        signals={"mode": final_mode, "emotion_state": emotion_state},
        context=context,
    )
    is_multi_event = bool((context or {}).get("multi_event")) or (context or {}).get("occasion") == "multi_event"
    polished_advice = _compact_reasoning(polished_advice, multi_event=is_multi_event)
    follow_up = str(payload.get("follow_up_question") or "").strip() or None
    pairing_routes: List[Dict[str, Any]] = []
    if final_mode == STYLE_PAIRING:
        pairing_routes = _normalize_pairing_routes(payload.get("pairing_routes"), pairing_anchor, pairing_gender)
        visual_directions = _pairing_routes_as_visual_directions(pairing_routes)
    else:
        visual_directions = _normalize_visual_directions(
            payload.get("visual_directions"),
            final_mode,
            category,
        )
    try:
        final_confidence = max(0.0, min(1.0, float(payload.get("confidence", confidence))))
    except Exception:
        final_confidence = confidence

    what_to_avoid = _safe_list(payload.get("what_to_avoid"), limit=6)
    missing_piece = _build_missing_piece(payload, missing_piece_reasoning)
    visual_inspiration_board = None
    if final_mode == VISUAL_INSPIRATION:
        visual_inspiration_board = _build_visual_inspiration_board(
            payload, visual_directions, goal, impression, missing_piece, query
        )

    return {
        "mode": final_mode,
        "occasion": str(payload.get("occasion") or occasion or "").strip() or None,
        "tone": tone,
        "formality": formality,
        "should_use_wardrobe": False,
        "should_generate_board": False,
        "advice": polished_advice,
        "stylist_reasoning": polished_advice,
        "goal": goal,
        "impression": impression,
        "atmosphere": atmosphere,
        "confidence_strategy": confidence_strategy,
        "missing_piece_reasoning": missing_piece_reasoning,
        "missing_piece": missing_piece,
        "visual_inspiration_board": visual_inspiration_board,
        "anchor_item": pairing_anchor or None,
        "pairing_routes": pairing_routes,
        "follow_up_question": follow_up,
        "cta": (
            [
                {"label": "Use my wardrobe", "value": f"Use my wardrobe for: {query}"},
                {"label": "Show visual inspiration", "value": f"Show visual inspiration for: {query}"},
                {"label": "Find missing pieces", "value": f"Show shopping ideas for: {query}"},
            ]
            if final_mode == STYLE_PAIRING
            else _fallback_cta(query)
        ),
        "visual_directions": visual_directions,
        "what_to_avoid": what_to_avoid,
        "meta": {
            "source": "style_reasoning_engine",
            "reason": _reason_for_mode(final_mode, category),
            "goal": goal,
            "impression": impression,
            "atmosphere": atmosphere,
            "missing_piece_reasoning": missing_piece_reasoning,
            "emotion_state": emotion_state,
            "confidence": final_confidence,
            "anchor_item": pairing_anchor or None,
        },
    }


def reason(
    query: str,
    intent: dict | str | None = None,
    user_profile: dict | None = None,
    context: dict | None = None,
    wardrobe_summary: dict | None = None,
    history: list | None = None,
) -> Dict[str, Any]:
    del wardrobe_summary, history
    safe_query = str(query or "").strip()
    safe_profile = user_profile if isinstance(user_profile, dict) else {}
    safe_context = context if isinstance(context, dict) else {}
    mode = _coerce_mode(safe_query, intent, safe_context)
    category, tone, formality, occasion = _occasion_category(safe_query)
    confidence = _confidence(intent, 0.9 if mode != GENERAL else 0.55)
    if mode == STYLE_PAIRING:
        logger.info("AHVI_STYLE_PAIRING_ROUTE query=%r", safe_query[:120])
        logger.info(
            "AHVI_PAIRING_FLOW_ORDER step=general_suggestions auto_wardrobe=False auto_visual=False ctas=%s",
            ["Show visual inspiration", "Use my wardrobe", "Find missing pieces"],
        )

    if mode == WARDROBE_STYLE:
        return {
            "mode": WARDROBE_STYLE,
            "occasion": occasion,
            "tone": tone,
            "formality": formality,
            "should_use_wardrobe": True,
            "should_generate_board": True,
            "advice": "",
            "follow_up_question": None,
            "cta": _fallback_cta(safe_query),
            "visual_directions": [],
            "meta": {
                "source": "style_reasoning_engine",
                "reason": _reason_for_mode(mode, category),
                "goal": "Build the look from the user's wardrobe.",
                "atmosphere": _fallback_atmosphere(category),
                "emotion_state": _fallback_emotion(category),
                "confidence": confidence,
            },
        }

    if mode == GENERAL:
        return {
            "mode": GENERAL,
            "occasion": None,
            "tone": None,
            "formality": None,
            "should_use_wardrobe": False,
            "should_generate_board": False,
            "advice": "",
            "follow_up_question": None,
            "cta": [],
            "visual_directions": [],
            "meta": {
                "source": "style_reasoning_engine",
                "reason": "not_style_request",
                "goal": "",
                "atmosphere": "",
                "emotion_state": "neutral",
                "confidence": confidence,
            },
        }

    ai_payload: Dict[str, Any] | None = None
    try:
        ai_payload = _gemini_reasoning(
            query=safe_query,
            mode=mode,
            category=category,
            user_profile=safe_profile,
            context=safe_context,
        )
    except Exception as exc:
        logger.warning(
            "ahvi.style_reasoning_gemini_failed mode=%s err=%s", mode, repr(exc)[:200]
        )
        ai_payload = None

    built = _build_response(
        query=safe_query,
        mode=mode,
        category=category,
        tone=tone,
        formality=formality,
        occasion=occasion,
        confidence=confidence,
        ai_payload=ai_payload,
        user_profile=safe_profile,
        context=safe_context,
    )
    # Wardrobe grounding: when a wardrobe is available, any suggested garment
    # that is not owned must live in missing_piece (the prompt enforces this);
    # log that grounding was active so we can audit owned-vs-suggested.
    _wardrobe = safe_context.get("wardrobe") or safe_context.get("wardrobe_items")
    if isinstance(_wardrobe, list) and _wardrobe:
        logger.info(
            "AHVI_WARDROBE_GROUNDING_APPLIED wardrobe_items=%d has_missing_piece=%s",
            len(_wardrobe),
            bool(built.get("missing_piece")),
        )
    return built


class _StyleReasoningEngine:
    def reason(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return reason(*args, **kwargs)


style_reasoning_engine = _StyleReasoningEngine()
