"""Board storyteller — produces compact, premium board story objects from
existing outfit intelligence.

This module never replaces the outfit pipeline or scoring; it consumes the
fields already produced by them (`score_meta`, `score_breakdown`,
`unified_style.reasons`, `style_metadata`, `agent_orchestration`,
`style_direction`, `palette_direction`, `occasion`, `weather`, `ml_score`,
`rank_score`) and shapes them into a small, UI-safe story dict.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from brain.intelligence.bank_snippets import (
        color_harmony_snippet,
        print_pattern_snippet,
        silhouette_snippet,
        weather_overlay_snippet,
    )
except Exception:  # pragma: no cover
    color_harmony_snippet = lambda *_a, **_k: ""  # noqa: E731
    print_pattern_snippet = lambda *_a, **_k: ""  # noqa: E731
    silhouette_snippet = lambda *_a, **_k: ""  # noqa: E731
    weather_overlay_snippet = lambda *_a, **_k: ""  # noqa: E731

try:
    from brain.tone.tone_engine import tone_engine
except Exception:  # pragma: no cover
    tone_engine = None

try:
    from brain.response_validator import polish_final_text
except Exception:  # pragma: no cover
    def polish_final_text(text: str, *, fallback: str = "") -> str:
        return (text or fallback or "").strip()


logger = logging.getLogger("ahvi.board_storyteller")


# ---------------------------------------------------------------------------
# Role tables
# ---------------------------------------------------------------------------

_DEFAULT_ROLES = ["Safest polished option", "Softer alternate", "Bolder style move"]

_ROLE_TABLES: Dict[str, List[str]] = {
    "office": [
        "Safest polished option",
        "More relaxed office option",
        "Sharper authority option",
    ],
    "business": [
        "Safest polished option",
        "More relaxed office option",
        "Sharper authority option",
    ],
    "client_meeting": [
        "Safest polished option",
        "More relaxed office option",
        "Sharper authority option",
    ],
    "date": [
        "Most effortless option",
        "Softer date-night option",
        "Stronger evening move",
    ],
    "date_night": [
        "Most effortless option",
        "Softer date-night option",
        "Stronger evening move",
    ],
    "workout": [
        "Clean performance option",
        "Comfort-first option",
        "Outdoor-ready option",
    ],
    "gym": [
        "Clean performance option",
        "Comfort-first option",
        "Outdoor-ready option",
    ],
    "travel": [
        "Long-day comfort option",
        "Polished transit option",
        "Layered travel option",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _occ_key(value: Any) -> str:
    raw = _text(value).lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return ""
    if any(k in raw for k in ("client", "meeting", "boardroom", "office", "business", "interview")):
        if "client" in raw or "meeting" in raw:
            return "client_meeting"
        if "business" in raw or "boardroom" in raw or "office" in raw or "interview" in raw:
            return "office"
    if any(k in raw for k in ("date", "dinner")):
        return "date_night"
    if any(k in raw for k in ("gym", "workout", "training")):
        return "workout"
    if any(k in raw for k in ("travel", "trip", "flight", "airport")):
        return "travel"
    if any(k in raw for k in ("wedding", "ceremony", "reception", "gala", "formal")):
        return "formal_event"
    return raw


def _role_for(occasion_key: str, rank_index: int) -> str:
    table = _ROLE_TABLES.get(occasion_key) or _DEFAULT_ROLES
    if not table:
        return _DEFAULT_ROLES[0]
    idx = max(0, min(rank_index, len(table) - 1))
    return table[idx]


def _truncate_sentence(text: str, max_chars: int = 140) -> str:
    """Keep one sentence, soft-capped — no mid-word cuts."""
    t = _text(text)
    if not t:
        return ""
    # Take first sentence.
    m = re.search(r"[.!?…]['\")\]]?\s", t + " ")
    one = t[: m.end()].strip() if m else t
    if len(one) <= max_chars:
        return one
    cut = one[:max_chars].rsplit(" ", 1)[0].rstrip(",;:- ")
    if not cut.endswith((".", "!", "?", "…")):
        cut += "."
    return cut


def _headline_words(text: str, max_words: int = 4) -> str:
    t = _text(text).strip(".!? ").title()
    words = [w for w in re.split(r"\s+", t) if w]
    if not words:
        return ""
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------

def _agent_payload(context: Dict[str, Any]) -> Dict[str, Any]:
    return _dict(context.get("agent_orchestration"))


def _resolve_occasion(board: Dict[str, Any], outfit: Dict[str, Any], context: Dict[str, Any]) -> str:
    return _text(
        board.get("occasion")
        or outfit.get("occasion")
        or context.get("occasion")
        or _agent_payload(context).get("occasion")
    )


def _style_direction(board: Dict[str, Any], outfit: Dict[str, Any], context: Dict[str, Any]) -> str:
    return _text(
        board.get("style_direction")
        or outfit.get("style_direction")
        or context.get("style_direction")
        or _agent_payload(context).get("style_direction")
    )


def _palette_direction(board: Dict[str, Any], outfit: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
    for src in (board, outfit, context, _agent_payload(context)):
        val = src.get("palette_direction") if isinstance(src, dict) else None
        if isinstance(val, list) and val:
            return [str(v).strip() for v in val if str(v).strip()]
    return []


def _score_reasons(outfit: Dict[str, Any]) -> List[str]:
    meta = _dict(outfit.get("score_meta"))
    reasons = meta.get("reasons") if isinstance(meta.get("reasons"), list) else []
    if not reasons:
        unified = _dict(outfit.get("unified_style"))
        reasons = unified.get("reasons") if isinstance(unified.get("reasons"), list) else []
    return [str(r).strip() for r in (reasons or []) if str(r).strip()]


def _first_item_style_meta(outfit: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the richest style_metadata from any item in the board."""
    items = outfit.get("items") if isinstance(outfit.get("items"), list) else []
    for item in items or []:
        if isinstance(item, dict) and isinstance(item.get("style_metadata"), dict):
            return item["style_metadata"]
    return {}


# ---------------------------------------------------------------------------
# Story composition
# ---------------------------------------------------------------------------

_OCCASION_FIT_LINES = {
    "office": "Strong for office and boardroom settings because nothing reads casual.",
    "client_meeting": "Strong for client-facing settings because nothing reads too casual.",
    "date_night": "Reads thoughtful for dinner without trying too hard.",
    "workout": "Built for movement without losing intention.",
    "travel": "Holds up across a long day of transit without looking tired.",
    "formal_event": "Respects the dress code while staying composed.",
}

_TIP_LINES = {
    "office": "Keep accessories restrained — one strong finisher is enough.",
    "client_meeting": "Keep accessories restrained — one strong finisher is enough.",
    "date_night": "Leave one button undone before you leave.",
    "workout": "Layer in something breathable if the room runs warm.",
    "travel": "Pack a single neutral layer for transit shifts.",
    "formal_event": "Let the footwear carry the polish.",
}

_HEADLINES_BY_DIRECTION = {
    "smart_casual_office": "Sharp Daily Edit",
    "corporate_office": "Minimal Executive",
    "creative_office": "Creative Polish",
    "friday_office": "Quiet Friday",
    "startup_office": "Easy Office",
    "date_night": "Evening Ease",
    "coastal_casual": "Coastal Ease",
    "training_functional": "Clean Performance",
    "daytime_polish": "Daytime Polish",
    "formal_event": "Composed Formal",
    "party": "After-Hours Edit",
    "comfort_polish": "Travel Polish",
    "clean_daily": "Clean Daily",
    "elevated_basics": "Elevated Basics",
}

_HEADLINES_BY_OCCASION = {
    "office": "Polished Daily",
    "client_meeting": "Minimal Executive",
    "date_night": "Soft Statement",
    "workout": "Clean Performance",
    "travel": "Comfort Polish",
    "formal_event": "Composed Formal",
}


def _build_headline(occasion_key: str, style_direction: str, fallback_title: str) -> str:
    text = (
        _HEADLINES_BY_DIRECTION.get(style_direction.lower())
        or _HEADLINES_BY_OCCASION.get(occasion_key)
        or fallback_title
        or "Considered Look"
    )
    return _headline_words(text, max_words=4)


def fallback_title_and_why(occasion: str, rank_index: int = 0) -> "tuple[str, str]":
    """Public helper for the curation layer: occasion-aware title + why copy,
    used when Gemini curation returns missing/weak/generic text. Never returns
    'Considered Look' or a generic item-list line."""
    occ_key = _occ_key(occasion)
    headline = (
        _HEADLINES_BY_OCCASION.get(occ_key)
        or _build_summary(occ_key, {}).split(",")[0].strip().title()
        or "Quietly Intentional"
    )
    if headline.lower() == "considered look":
        headline = "Quietly Intentional"
    why = ""
    templates = _OCCASION_WHY_TEMPLATES.get(occ_key)
    if templates:
        why = templates[rank_index % len(templates)]
    if not why:
        why = _build_summary(occ_key, {})
    return _headline_words(headline, max_words=4), why


def _build_summary(occasion_key: str, style_meta: Dict[str, Any]) -> str:
    formality = str(style_meta.get("formality") or "").lower()
    if occasion_key in {"client_meeting", "office"} or formality in {"high", "business_casual", "smart_casual"}:
        return "Sharp, composed, and ready for the room."
    if occasion_key == "date_night":
        return "Soft, intentional, and easy to wear."
    if occasion_key == "workout":
        return "Built to move without looking thrown together."
    if occasion_key == "travel":
        return "Comfortable but never sloppy across a long day."
    if occasion_key == "formal_event":
        return "Quietly formal without overworking the look."
    return "Clean, intentional, and easy to wear."


_OCCASION_WHY_TEMPLATES: Dict[str, List[str]] = {
    "client_meeting": [
        "The tailored bottom and clean footwear keep the look client-facing and polished while staying approachable.",
        "The composed top + structured bottom combination reads intentional in a boardroom without trying too hard.",
        "Clean lines and a restrained palette let the work speak — nothing on the body competes with the conversation.",
    ],
    "office": [
        "The shirt + trouser register stays smart-casual without leaning corporate-stiff.",
        "Footwear and palette keep this credible for the office while remaining easy to move in.",
        "Composed proportions read intentional from across a room without feeling overdressed.",
    ],
    "date_night": [
        "Soft texture on top and a darker bottom register make this read intentional after dark.",
        "The relaxed silhouette + cleaner footwear gives an effortless evening polish.",
        "Restrained palette + one quiet accessory finish keeps the focus on you, not the outfit.",
    ],
    "party": [
        "The statement piece is paired with quieter neutrals so the look has a hero without competing.",
        "Sharper bottom + relaxed top + considered footwear gives the social-edited finish.",
    ],
    "workout": [
        "Activewear pairing keeps movement easy with no formal pieces fighting the brief.",
        "Performance-led top + bottom combo with the right footwear for the session.",
    ],
    "travel": [
        "Comfortable layers + composed footwear handle a long day across transit without looking thrown together.",
        "The neutral palette reads polished on arrival and forgiving en route.",
    ],
    "beach": [
        "Breathable top + relaxed bottom + sand-friendly footwear stay coastal without effort.",
        "Light fabric + neutral palette reads resort-easy in heat.",
    ],
    "swimming": [
        "Swim-ready essentials only — nothing here will fight the pool.",
        "Easy slip-on footwear and a swim-appropriate top keep the brief honest.",
    ],
    "wedding": [
        "Formal-event pairing with respectful tailoring + clean footwear for a ceremony register.",
        "Composed silhouette + a restrained accessory finish keeps the focus on the moment.",
    ],
    "formal_event": [
        "Tailored top + structured bottom + formal footwear keeps the register correct for the event.",
        "Quiet palette + considered accessory keeps the look composed without overstating.",
    ],
    "capsule": [
        "These pieces form a versatile foundation — each can recombine across multiple briefs without leaning into novelty.",
        "Neutral palette + clean cuts mean every item earns repeat wear instead of single-occasion use.",
    ],
    "casual": [
        "Easy top + relaxed bottom + clean footwear delivers a polished daily without overworking.",
    ],
    "daily": [
        "Clean smart-casual base that handles whatever the day asks of it.",
    ],
}


def _build_why(
    outfit: Dict[str, Any],
    score_reasons: List[str],
    palette: List[str],
    style_direction: str,
    weather: str,
    occasion_key: str = "",
) -> str:
    # Occasion-aware editorial template wins when the brief is known —
    # avoids the legacy "carries the personality" / "keeps it from
    # turning loud" generic copy.
    occ_key = (occasion_key or "").lower()
    if occ_key in _OCCASION_WHY_TEMPLATES:
        templates = _OCCASION_WHY_TEMPLATES[occ_key]
        stable_key = _text(outfit.get("combo_id") or outfit.get("id") or "board")
        idx = (sum(ord(c) for c in stable_key) if stable_key else 0) % len(templates)
        return templates[idx]

    breakdown = _dict(outfit.get("score_breakdown"))
    try:
        color_hint = float(breakdown.get("color_intelligence") or breakdown.get("palette") or 0.0)
    except Exception:
        color_hint = 0.0
    try:
        silhouette_hint = float(breakdown.get("style_graph") or breakdown.get("aesthetic_balance") or 0.0)
    except Exception:
        silhouette_hint = 0.0
    stable_key = _text(outfit.get("combo_id") or outfit.get("id") or "board")

    bank = " ".join(
        x
        for x in (
            silhouette_snippet(silhouette_hint, key=stable_key),
            color_harmony_snippet(color_hint, key=stable_key),
        )
        if x
    ).strip()

    if bank and "personality" not in bank.lower() and "loud" not in bank.lower():
        return _truncate_sentence(bank)

    if "palette aligned" in score_reasons:
        return "The palette holds together and lets the silhouette carry the look."
    if "clean aesthetic balance" in score_reasons:
        return "The proportions stay balanced so the look reads intentional."
    if style_direction:
        return f"The look leans into {style_direction.replace('_', ' ')} without forcing it."
    if palette:
        return f"The {palette[0]} thread keeps the palette quiet and considered."
    return "The pieces sit in the same register, so nothing competes."


def _build_personal_note(context: Dict[str, Any], style_meta: Dict[str, Any]) -> str:
    profile = _dict(context.get("user_profile"))
    prefs = profile.get("stylePreferences") or profile.get("style_preferences") or []
    if isinstance(prefs, str):
        prefs = [p.strip() for p in prefs.split(",") if p.strip()]
    if prefs:
        first = str(prefs[0]).strip().lower()
        if first:
            return f"This stays close to your {first} preferences without leaning predictable."
    formality = str(style_meta.get("formality") or "").lower()
    if formality:
        return f"This matches the {formality.replace('_', ' ')} register you usually pick."
    return "This stays close to the cleaner choices you tend to favor."


def _build_occasion_fit(occasion_key: str, weather: str) -> str:
    base = _OCCASION_FIT_LINES.get(occasion_key, "Reads intentional for the day ahead.")
    if weather:
        overlay = weather_overlay_snippet(weather, key=occasion_key or "board")
        if overlay:
            return _truncate_sentence(f"{base} {overlay}")
    return base


def _build_tip(occasion_key: str, palette: List[str]) -> str:
    base = _TIP_LINES.get(occasion_key)
    if base:
        return base
    if palette:
        return f"Hold the palette in the {palette[0]} family — one accent is enough."
    return "Keep one piece quiet so the strongest piece can lead."


# ---------------------------------------------------------------------------
# Tone + polish pipeline
# ---------------------------------------------------------------------------

_STORY_FIELDS = ("headline", "summary", "why", "personal_note", "occasion_fit", "tip", "role")


def _refine_story_strings(story: Dict[str, str], context: Dict[str, Any]) -> Dict[str, str]:
    out = dict(story)
    user_profile = _dict(context.get("user_profile"))
    signals = _dict(context.get("signals"))
    for key in _STORY_FIELDS:
        raw = _text(out.get(key))
        if not raw:
            out[key] = ""
            continue
        toned = raw
        if tone_engine is not None and key != "role" and key != "headline":
            try:
                toned = tone_engine.apply(raw, user_profile=user_profile, signals=signals)
            except Exception:
                toned = raw
        polished = polish_final_text(toned, fallback=raw)
        if key in {"headline", "role"}:
            polished = polished.rstrip(".")
        out[key] = polished
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BoardStoryteller:
    """Builds a compact `story` dict for every style board.

    Stateless; safe to use as a module-level singleton.
    """

    def enrich_boards(
        self,
        boards: List[Dict[str, Any]],
        outfits: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(boards, list):
            return boards
        ctx = _dict(context)
        outfits_list = outfits if isinstance(outfits, list) else []

        enriched: List[Dict[str, Any]] = []
        for idx, raw_board in enumerate(boards):
            if not isinstance(raw_board, dict):
                enriched.append(raw_board)
                continue
            board = dict(raw_board)
            matched_outfit = self._match_outfit(board, outfits_list, idx)
            story = self._build_story(board, matched_outfit, ctx, rank_index=idx)
            board["story"] = story
            # Backward-compatible mirrors so older clients keep rendering.
            board.setdefault("why_it_works", story.get("why") or "")
            board.setdefault("explanation", story.get("summary") or "")
            board.setdefault("styling_tip", story.get("tip") or "")
            enriched.append(board)
        return enriched

    # -- internals -----------------------------------------------------------

    def _match_outfit(
        self, board: Dict[str, Any], outfits: List[Dict[str, Any]], idx: int
    ) -> Dict[str, Any]:
        board_id = _text(board.get("id") or board.get("combo_id") or board.get("board_id"))
        if board_id:
            for outfit in outfits:
                if not isinstance(outfit, dict):
                    continue
                oid = _text(outfit.get("id") or outfit.get("combo_id"))
                if oid and oid == board_id:
                    return outfit
        if 0 <= idx < len(outfits) and isinstance(outfits[idx], dict):
            return outfits[idx]
        return _dict(board)

    def _build_story(
        self,
        board: Dict[str, Any],
        outfit: Dict[str, Any],
        context: Dict[str, Any],
        *,
        rank_index: int,
    ) -> Dict[str, str]:
        occasion_raw = _resolve_occasion(board, outfit, context)
        occasion_key = _occ_key(occasion_raw)
        style_direction = _style_direction(board, outfit, context)
        palette = _palette_direction(board, outfit, context)
        weather = _text(context.get("weather") or _dict(context.get("signals")).get("weather_mode"))
        score_reasons = _score_reasons(outfit)
        style_meta = _first_item_style_meta(outfit)

        story = {
            "headline": _build_headline(
                occasion_key,
                style_direction,
                fallback_title=_text(board.get("title")),
            ),
            "summary": _build_summary(occasion_key, style_meta),
            "why": _build_why(outfit, score_reasons, palette, style_direction, weather, occasion_key=occasion_key),
            "personal_note": _build_personal_note(context, style_meta),
            "occasion_fit": _build_occasion_fit(occasion_key, weather),
            "tip": _build_tip(occasion_key, palette),
            "role": _role_for(occasion_key, rank_index),
        }

        story = _refine_story_strings(story, context)
        try:
            logger.debug(
                "ahvi.board_story.built rank=%d occasion=%s headline=%r",
                rank_index,
                occasion_key,
                story.get("headline"),
            )
        except Exception:
            pass
        return story


board_storyteller = BoardStoryteller()


def enrich_boards(
    boards: List[Dict[str, Any]],
    outfits: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Module-level convenience wrapper around the singleton."""
    return board_storyteller.enrich_boards(boards, outfits, context)


__all__ = ["BoardStoryteller", "board_storyteller", "enrich_boards"]
