from __future__ import annotations

import re
from typing import Any, Dict, List

from brain.tone.tone_engine import tone_engine
from prompts.core_prompts import AHVI_SYSTEM_PROMPT
from prompts.styling_prompts import OCCASION_INTERPRETER_PROMPT
from services.ai_gateway import generate_text, parse_json_object
from services.stylist_knowledge_service import (
    COLOR_BODY_ADVICE,
    SHOPPING_ASSIST,
    STYLE_ADVICE,
    STYLE_EDUCATION,
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
}

_GEMINI_MODES = {
    STYLE_ADVICE,
    VISUAL_INSPIRATION,
    SHOPPING_ASSIST,
    STYLE_EDUCATION,
    COLOR_BODY_ADVICE,
}


def _norm(value: Any) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


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
    ctx = context if isinstance(context, dict) else {}
    style_action = str(ctx.get("style_action") or "")
    module_context = str(ctx.get("module_context") or ctx.get("module") or "")
    intent_value = _intent_name(intent)
    q = _norm(query)
    if _has_any(
        q,
        (
            "show visual inspiration",
            "visual inspiration",
            "generate moodboard",
            "show moodboard",
            "moodboard for",
        ),
    ):
        return VISUAL_INSPIRATION
    if intent_value in _STYLE_REASONING_MODES:
        return intent_value
    return (
        classify_style_mode(
            query,
            module_context=module_context,
            style_action=style_action,
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
) -> str:
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
      "style_note": string
    }}
  ],
  "follow_up_question": string|null,
  "confidence": float
}}

Writing rules for stylist_reasoning:
- Speak like a stylist explaining a decision. Use phrasing such as
  "This works because...", "I would avoid...", "The priority here is...",
  "The risk is...", "This creates...".
- Explain the social strategy first, then the clothing logic.
- 2-4 sentences. Specific to THIS occasion. Never reusable boilerplate.

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

Known deterministic mode: {mode}
Detected category: {category or "unknown"}
User query: {query}
User profile: {user_profile}
Context: {context}
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


def _gemini_reasoning(
    *,
    query: str,
    mode: str,
    category: str | None,
    user_profile: dict,
    context: dict,
) -> Dict[str, Any]:
    prompt = _build_reasoning_prompt(
        query=query,
        mode=mode,
        category=category,
        user_profile=user_profile,
        context=context,
    )
    raw = generate_text(
        prompt,
        options={"temperature": 0.45, "max_output_tokens": 1600},
        user_profile=user_profile,
        signals={"context_mode": "style_reasoning", "style_mode": mode},
        usecase="style_reasoning",
    )
    return parse_json_object(raw)


def _coerce_emotion(value: Any, category: str | None) -> str:
    emotion = _norm(value)
    if emotion in {"neutral", "excited", "frustrated", "vulnerable", "professional", "social"}:
        return emotion
    return _fallback_emotion(category)


def _coerce_ai_mode(value: Any, fallback: str) -> str:
    mode = _norm(value)
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
    final_mode = _coerce_ai_mode(payload.get("mode"), mode)
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
    polished_advice = tone_engine.apply(
        raw_advice,
        user_profile=user_profile,
        signals={"mode": final_mode, "emotion_state": emotion_state},
        context=context,
    )
    follow_up = str(payload.get("follow_up_question") or "").strip() or None
    visual_directions = _normalize_visual_directions(
        payload.get("visual_directions"),
        final_mode,
        category,
    )
    try:
        final_confidence = max(0.0, min(1.0, float(payload.get("confidence", confidence))))
    except Exception:
        final_confidence = confidence

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
        "missing_piece_reasoning": missing_piece_reasoning,
        "follow_up_question": follow_up,
        "cta": _fallback_cta(query),
        "visual_directions": visual_directions,
        "what_to_avoid": _safe_list(payload.get("what_to_avoid"), limit=6),
        "meta": {
            "source": "style_reasoning_engine",
            "reason": _reason_for_mode(final_mode, category),
            "goal": goal,
            "impression": impression,
            "atmosphere": atmosphere,
            "missing_piece_reasoning": missing_piece_reasoning,
            "emotion_state": emotion_state,
            "confidence": final_confidence,
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
    except Exception:
        ai_payload = None

    return _build_response(
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


class _StyleReasoningEngine:
    def reason(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return reason(*args, **kwargs)


style_reasoning_engine = _StyleReasoningEngine()
