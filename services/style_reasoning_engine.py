from __future__ import annotations

import re
from typing import Any, Dict, List

from services.stylist_knowledge_service import (
    COLOR_BODY_ADVICE,
    SHOPPING_ASSIST,
    STYLE_ADVICE,
    STYLE_EDUCATION,
    WARDROBE_STYLE,
    build_stylist_advice_response,
    classify_style_mode,
)


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


def _occasion_category(query: str) -> tuple[str | None, str | None, str | None, str]:
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
        "party",
        "wedding",
        "brunch",
        "lunch",
        "birthday",
        "reception",
        "festival",
    )

    if _has_any(q, sensitive):
        return ("sensitive_occasion", "respectful", "polished", "sensitive occasion")
    if _has_any(q, work):
        return ("work_occasion", "competent", "smart", "work occasion")
    if _has_any(q, social):
        return ("social_occasion", "warm", "smart casual", "social occasion")
    if _has_any(q, ("beach", "travel", "airport", "vacation", "trip")):
        return ("travel_occasion", "practical", "casual", "travel occasion")
    return ("custom_occasion", "considered", "context-aware", "custom occasion")


def _fallback_cta(query: str) -> List[Dict[str, str]]:
    return [
        {"label": "Use my wardrobe", "value": f"Use my wardrobe for: {query}"},
        {"label": "Show visual inspiration", "value": f"Generate moodboard for: {query}"},
        {"label": "Find missing pieces", "value": f"Show shopping ideas for: {query}"},
    ]


def _reason_for_mode(mode: str, category: str | None) -> str:
    if mode == WARDROBE_STYLE:
        return "wardrobe_request"
    if mode == SHOPPING_ASSIST:
        return "shopping_request"
    if mode == COLOR_BODY_ADVICE:
        return "body_color_advice"
    if mode == STYLE_EDUCATION:
        return "style_education"
    return category or "style_advice"


def _advice_for_mode(query: str, mode: str) -> str:
    response = build_stylist_advice_response(query=query, mode=mode)
    return str(
        response.get("message_text")
        or response.get("response")
        or response.get("text")
        or ""
    ).strip()


def _coerce_mode(query: str, intent: dict | str | None, context: dict | None) -> str:
    ctx = context if isinstance(context, dict) else {}
    style_action = str(ctx.get("style_action") or "")
    module_context = str(ctx.get("module_context") or ctx.get("module") or "")
    intent_value = _intent_name(intent)
    if intent_value in {
        STYLE_ADVICE,
        WARDROBE_STYLE,
        SHOPPING_ASSIST,
        STYLE_EDUCATION,
        COLOR_BODY_ADVICE,
    }:
        return intent_value
    return classify_style_mode(
        query,
        module_context=module_context,
        style_action=style_action,
    ) or "general"


def reason(
    query: str,
    intent: dict | str | None = None,
    user_profile: dict | None = None,
    context: dict | None = None,
    wardrobe_summary: dict | None = None,
    history: list | None = None,
) -> Dict[str, Any]:
    del user_profile, wardrobe_summary, history
    safe_query = str(query or "").strip()
    mode = _coerce_mode(safe_query, intent, context)
    category, tone, formality, occasion = _occasion_category(safe_query)
    confidence = _confidence(intent, 0.9 if mode != "general" else 0.55)

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
            "meta": {
                "source": "style_reasoning_engine",
                "reason": _reason_for_mode(mode, category),
                "category": category,
                "confidence": confidence,
            },
        }

    if mode == "general":
        return {
            "mode": "general",
            "occasion": None,
            "tone": None,
            "formality": None,
            "should_use_wardrobe": False,
            "should_generate_board": False,
            "advice": "",
            "follow_up_question": None,
            "cta": [],
            "meta": {
                "source": "style_reasoning_engine",
                "reason": "not_style_request",
                "confidence": confidence,
            },
        }

    advice = _advice_for_mode(safe_query, mode)
    return {
        "mode": mode,
        "occasion": occasion,
        "tone": tone,
        "formality": formality,
        "should_use_wardrobe": False,
        "should_generate_board": False,
        "advice": advice,
        "follow_up_question": None,
        "cta": _fallback_cta(safe_query),
        "meta": {
            "source": "style_reasoning_engine",
            "reason": _reason_for_mode(mode, category),
            "category": category,
            "confidence": confidence,
        },
    }


class _StyleReasoningEngine:
    def reason(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return reason(*args, **kwargs)


style_reasoning_engine = _StyleReasoningEngine()
