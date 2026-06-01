from __future__ import annotations

from typing import Any, Dict, List, Optional

from brain.engines.occasion_style_rules import get_occasion_rule


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _has_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def detect_occasion(query: str) -> str:
    q = _text(query)
    if _has_any(q, ("temple", "mandir", "pooja", "puja", "religious", "shrine", "darshan")):
        return "temple_modest"
    if _has_any(q, ("swim", "swimming", "swimwear", "swimsuit", "pool")):
        return "swimming"
    if _has_any(q, ("beach", "seaside", "coastal", "sand-friendly", "sand friendly")):
        return "beach"
    if _has_any(q, ("gym", "workout", "fitness", "training", "yoga", "run ", "running")):
        return "workout"
    if _has_any(q, ("brunch",)):
        return "brunch"
    if _has_any(
        q,
        (
            "office",
            "work",
            "meeting",
            "client",
            "business",
            "interview",
            "corporate",
            "boardroom",
            "presentation",
            "formal work",
        ),
    ):
        return "office"
    if _has_any(q, ("date", "dinner", "tonight", "evening")):
        return "date_night"
    if _has_any(q, ("party", "club", "rave", "after-hours", "night out", "cocktail", "pub", "birthday")):
        return "party"
    if _has_any(q, ("travel", "airport", "flight", "vacation", "trip")):
        return "travel"
    if _has_any(q, ("wedding", "reception", "ceremony", "sangeet", "formal event", "event")):
        return "wedding"
    if _has_any(q, ("casual", "weekend", "sunday", "home", "errand", "coffee", "resort")):
        return "casual"
    # Generic "today"/"daily"/no signal → daily (aliased to casual rule)
    return "daily"


def _chips(*labels_and_values: tuple[str, str]) -> List[Dict[str, str]]:
    return [{"label": label, "value": value} for label, value in labels_and_values][:3]


def interpret_occasion_context(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = dict(context or {})
    q = _text(query)
    kind = detect_occasion(query)
    rule = get_occasion_rule(kind)
    weather = _text(ctx.get("weather") or ctx.get("condition"))
    time_of_day = _text(ctx.get("time_of_day") or ctx.get("hour"))

    confidence = "high"
    ask_user = False
    reason = "context_sufficient"
    chips: List[Dict[str, str]] = []

    if kind == "temple_modest":
        brief = "modest, respectful, covered shoulders, soft palette, traditional ease"
    elif kind == "workout":
        brief = "workout-ready, breathable, movement friendly"
    elif kind == "brunch":
        brief = "daytime brunch, easy polish, movement friendly"
    elif kind == "office":
        if _has_any(q, ("client", "presentation", "meeting", "corporate", "interview", "boardroom")):
            brief = "client-facing office, polish required, smart footwear"
        elif _has_any(q, ("creative", "studio", "agency")):
            brief = "creative office, expressive but controlled"
        elif _has_any(q, ("startup", "start-up", "friday", "casual office")):
            brief = "relaxed office, intentional smart casual"
        else:
            brief = "smart casual office, clean structure, credible footwear"
    elif kind in {"date", "date_night"}:
        if _has_any(q, ("restaurant", "candle", "dinner date")):
            brief = "polished dinner, low light, intentional but not overworked"
        elif _has_any(q, ("coffee", "day", "afternoon", "walk")):
            brief = "daytime date, easy polish, movement friendly"
        elif _has_any(q, ("rooftop", "drinks", "bar")):
            brief = "rooftop casual, warm evening, relaxed polish"
        else:
            brief = "polished date, social ease, clean footwear"
            ask_user = "date" in q and not _has_any(q, ("today", "now", "quick", "suggest"))
            confidence = "medium" if ask_user else "high"
            reason = "date_venue_changes_formality" if ask_user else reason
            chips = _chips(
                ("Polished dinner", "date_polished_dinner"),
                ("Rooftop easy", "date_rooftop_easy"),
                ("Daytime quiet", "date_daytime_quiet"),
            )
    elif kind == "beach":
        brief = "beach ready, sand-friendly, relaxed resort casual"
        confidence = "high"
        reason = "beach_context_detected"
    elif kind == "swimming":
        brief = "swim-ready, pool appropriate, quick-dry, easy layers"
        confidence = "high"
        reason = "swimming_context_detected"
    elif kind == "party":
        if _has_any(q, ("rave", "club", "edm", "festival")):
            brief = "rave party, after-hours energy, movement friendly, expressive edge"
        elif _has_any(q, ("cocktail", "lounge", "bar", "drinks")):
            brief = "cocktail party, sharp social polish, after-dark restraint"
        elif _has_any(q, ("house", "birthday", "friends")):
            brief = "house party, relaxed social polish, memorable but easy"
        else:
            brief = "after-hours social, sharper contrast, memorable but edited"
            ask_user = True
            confidence = "medium"
            reason = "party_atmosphere_changes_formality"
            chips = _chips(
                ("Rave energy", "party_rave_energy"),
                ("Cocktail sharp", "party_cocktail_sharp"),
                ("House easy", "party_house_easy"),
            )
    elif kind == "travel":
        brief = "travel comfort polish, movement friendly, weather aware"
    elif kind == "wedding":
        brief = "formal event, respectful polish, ceremony-aware restraint"
        if not _has_any(q, ("reception", "wedding", "formal", "traditional", "ceremony")):
            ask_user = True
            confidence = "medium"
            reason = "event_formality_or_cultural_context"
            chips = _chips(
                ("Ceremony refined", "event_ceremony_refined"),
                ("Reception polish", "event_reception_polish"),
                ("Statement occasion", "event_statement_occasion"),
            )
    elif kind == "casual":
        brief = "clean daily, comfortable but intentional"
    elif kind == "daily":
        brief = "smart daily, smart casual, clean casual, easy and intentional"
    else:
        brief = "elevated daily, smart casual, repeat-aware"

    if "rain" in weather:
        brief = f"{brief}, rain-ready"
    elif _has_any(weather, ("hot", "summer", "humid")):
        brief = f"{brief}, heat-conscious"
    elif _has_any(weather, ("cold", "winter")):
        brief = f"{brief}, layered"
    if time_of_day and kind in {"date", "date_night", "party"}:
        brief = f"{brief}, {time_of_day}"

    return {
        "occasion": kind,
        "resolved_brief": brief,
        "confidence": confidence,
        "style_direction": str(rule.get("style_direction") or ""),
        "formality_target": float(rule.get("formality_target") or 2.5),
        "footwear_energy": str(rule.get("footwear_energy") or ""),
        "should_ask": bool(ask_user),
        "ask_user": bool(ask_user),
        "question": "Which brief is closer?" if ask_user else "",
        "chips": chips,
        "board_generation_notes": {
            "occasion_kind": kind,
            "style_direction": str(rule.get("style_direction") or ""),
            "reason": reason,
            "max_questions": 3,
            "mood": str(rule.get("mood") or ""),
        },
    }
