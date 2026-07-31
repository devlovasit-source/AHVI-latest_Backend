from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import asyncio
import os
import logging
import time
import hashlib
import concurrent.futures
import threading
import re
import json
from datetime import datetime, timedelta, timezone

from deep_translator import GoogleTranslator

try:
    from worker import run_heavy_audio_task
except Exception:
    run_heavy_audio_task = None

from brain.orchestrator import ahvi_orchestrator
from brain.intent_engine import detect_intent
from brain.plan_pack_flow import build_plan_pack_response
from brain.tone.tone_engine import tone_engine
from brain.outfit_pipeline import save_feedback
from services.appwrite_proxy import AppwriteProxy
from services.llm_service import chat_completion
from services.style_flow_service import (
    STYLE_ACTION_CHIPS,
    build_style_flow_response,
    card_signature as style_card_signature,
    curate_wardrobe_boards,
    finalize_style_response_payload,
    interpret_occasion,
    _build_composition_brief,
)
from services.module_chat_service import handle_module_chat
from services.style_reasoning_engine import VISUAL_INSPIRATION, style_reasoning_engine
from services.beta_style_bridge import (
    decorate_style_response as decorate_beta_style_response,
    engine_dispatch as beta_style_engine_dispatch,
    interpret_style_followup,
    normalize_style_state,
    refine_style_response as beta_refine_style_response,
    visual_items_from_response as beta_visual_items_from_response,
    visual_intelligence as beta_visual_intelligence,
)
from services.stylist_knowledge_service import (
    COLOR_BODY_ADVICE,
    SHOPPING_ASSIST,
    STYLE_ADVICE,
    STYLE_EDUCATION,
    STYLE_PAIRING,
    STYLE_MODES,
    WARDROBE_STYLE,
    classify_style_mode,
    is_wardrobe_style_request,
)

try:
    from services.job_tracker import job_tracker
except Exception:
    job_tracker = None
from services.task_queue import enqueue_task

# Ã°Å¸â€Â¥ NEW
from services.weather_service import get_hourly_weather
from services.location_weather_context import resolve_location_weather_context

router = APIRouter()
logger = logging.getLogger("ahvi.routers.chat")

_CHAT_CACHE_MAX_ITEMS = max(64, int(os.getenv("CHAT_CACHE_MAX_ITEMS", "512")))
_CHAT_CACHE_TTL_SECONDS = max(15, int(os.getenv("CHAT_CACHE_TTL_SECONDS", "60")))
_WEATHER_CACHE_MAX_ITEMS = max(32, int(os.getenv("WEATHER_CACHE_MAX_ITEMS", "256")))
_WEATHER_CACHE_TTL_SECONDS = max(60, int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "900")))
_ORCH_TIMEOUT_SECONDS = max(
    2,
    int(
        os.getenv("ORCHESTRATOR_TIMEOUT_SECONDS")
        or os.getenv("CHAT_ORCHESTRATOR_TIMEOUT_SECONDS")
        or "45"
    ),
)
_ORCHESTRATOR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    # Default reduced from 8 -> 3 to avoid oversubscribing 1-vCPU Cloud Run
    # instances under default concurrency=80. Override via env if needed.
    max_workers=max(2, int(os.getenv("CHAT_ORCHESTRATOR_MAX_WORKERS", "3"))),
    thread_name_prefix="chat-orch",
)
_CHAT_INCLUDE_BASE64_ALLOWED = str(
    os.getenv("AHVI_CHAT_INCLUDE_BASE64_ALLOWED", "")
).lower() in {"1", "true", "yes"}


class _TTLLRUCache:
    """Thread-safe TTL+LRU cache. O(1) get/set, lazy expiry on hit, bounded size."""

    def __init__(self, max_items: int, ttl_seconds: int):
        self._max = int(max_items)
        self._ttl = int(ttl_seconds)
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            row = self._data.get(key)
            if row is None:
                return None
            expires_at, value = row
            if now >= expires_at:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        expires_at = time.time() + self._ttl
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_CHAT_CACHE = _TTLLRUCache(_CHAT_CACHE_MAX_ITEMS, _CHAT_CACHE_TTL_SECONDS)
_WEATHER_CACHE = _TTLLRUCache(_WEATHER_CACHE_MAX_ITEMS, _WEATHER_CACHE_TTL_SECONDS)
_STYLE_CONTEXT_CACHE = _TTLLRUCache(512, 15 * 60)


def lightweight_chat(text: str, *, user_profile: dict | None = None) -> str:
    """Lightweight fallback responses MUST pass through tone + polish so the
    premium AHVI voice is not bypassed by hardcoded copy."""
    prompt = str(text or "").strip()
    lower = prompt.lower()
    if not prompt:
        raw = "What is on your mind today?"
    elif "joke" in lower:
        raw = "A small one: the shirt got promoted because it had outstanding style."
    elif "how are you" in lower or lower in {"hi", "hello", "hey"}:
        raw = "I’m here. Ask me for an outfit, a capsule, or talk it through."
    else:
        raw = "I can help with style, planning, and wardrobe. Tell me what you want to solve."

    try:
        from brain.tone.tone_engine import tone_engine as _tone
        from brain.response_validator import polish_final_text as _polish

        toned = _tone.apply(
            raw,
            user_profile=user_profile or {},
            signals={"context_mode": "professional", "emotion_state": "neutral"},
        )
        polished = _polish(toned, fallback=raw)
        try:
            logger.info("ahvi.response.router_bypass_prevented path=lightweight_chat")
        except Exception:
            pass
        return polished
    except Exception:
        return raw


def _wardrobe_hash(wardrobe: Any) -> str:
    if not isinstance(wardrobe, list):
        return "no_wardrobe"
    rows: List[str] = []
    for item in wardrobe:
        if not isinstance(item, dict):
            continue
        rows.append(
            "|".join(
                str(item.get(k) or "").strip().lower()
                for k in ("id", "$id", "name", "category", "sub_category", "subcategory", "color")
            )
        )
    return hashlib.sha1("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def _cache_key(text, user_id, *, module_context: str = "", wardrobe: Any = None, occasion: str = "", weather: Any = None):
    parts = [
        str(user_id or "").strip(),
        str(module_context or "").strip().lower(),
        str(text or "").strip().lower(),
        str(occasion or "").strip().lower(),
        str(weather or "").strip().lower(),
        _wardrobe_hash(wardrobe),
    ]
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()


def _structured_error_response(
    *,
    code: str,
    message: str,
    status_type: str = "error",
    details: Optional[Dict[str, Any]] = None,
    chips: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    safe_message = str(message or "AHVI could not complete this request.").strip()
    return {
        "ok": False,
        "success": False,
        "type": status_type,
        "message": {"role": "assistant", "content": safe_message},
        "message_text": safe_message,
        "response": safe_message,
        "cards": [],
        "style_boards": [],
        "chips": chips or [],
        "board_ids": "",
        "data": {"outfits": [], "rendered_boards": [], "error": {"code": code, "message": safe_message, "details": details or {}}},
        "error": {"code": code, "message": safe_message, "details": details or {}},
        "meta": {"error_code": code, **(details or {})},
        "audio_job_id": "offline",
    }


_OCCASION_CLARIFICATION_CHIPS: Dict[str, List[Dict[str, str]]] = {
    "beach": [
        {"label": "Beach vacation", "value": "Beach vacation"},
        {"label": "Poolside day", "value": "Poolside day"},
        {"label": "Resort dinner", "value": "Resort dinner"},
        {"label": "Casual beach walk", "value": "Casual beach walk"},
        {"label": "Use my wardrobe", "value": "Use my wardrobe"},
    ],
    "office": [
        {"label": "Office meeting", "value": "Office meeting"},
        {"label": "Casual office day", "value": "Casual office day"},
        {"label": "Client meeting", "value": "Client meeting"},
        {"label": "Use my wardrobe", "value": "Use my wardrobe"},
    ],
    "party": [
        {"label": "House party", "value": "House party"},
        {"label": "Club night", "value": "Club night"},
        {"label": "Dinner party", "value": "Dinner party"},
        {"label": "Use my wardrobe", "value": "Use my wardrobe"},
    ],
    "date": [
        {"label": "Dinner date", "value": "Dinner date"},
        {"label": "Coffee date", "value": "Coffee date"},
        {"label": "Movie date", "value": "Movie date"},
        {"label": "Use my wardrobe", "value": "Use my wardrobe"},
    ],
    "travel": [
        {"label": "Airport outfit", "value": "Airport outfit"},
        {"label": "Road trip", "value": "Road trip"},
        {"label": "Day trip", "value": "Day trip"},
        {"label": "Overnight trip", "value": "Overnight trip"},
        {"label": "Use my wardrobe", "value": "Use my wardrobe"},
    ],
    "gym": [
        {"label": "Strength training", "value": "Strength training"},
        {"label": "Cardio", "value": "Cardio"},
        {"label": "Yoga", "value": "Yoga"},
        {"label": "Use my wardrobe", "value": "Use my wardrobe"},
    ],
    "workout": [
        {"label": "Strength training", "value": "Strength training"},
        {"label": "Cardio", "value": "Cardio"},
        {"label": "Yoga", "value": "Yoga"},
        {"label": "Use my wardrobe", "value": "Use my wardrobe"},
    ],
    "wedding": [
        {"label": "Indian wedding", "value": "Indian wedding"},
        {"label": "Western wedding", "value": "Western wedding"},
        {"label": "Reception", "value": "Reception"},
        {"label": "Use my wardrobe", "value": "Use my wardrobe"},
    ],
}


def _clarification_chips_for_occasion(occasion: Optional[str]) -> List[Dict[str, str]]:
    """Pick a set of clarification chips tuned to the inferred occasion.

    Falls back to a broad set when the occasion is unknown so the user
    still has something useful to tap.
    """
    key = str(occasion or "").lower().strip()
    if key in _OCCASION_CLARIFICATION_CHIPS:
        return list(_OCCASION_CLARIFICATION_CHIPS[key])
    return [
        {"label": "Office", "value": "Office outfit"},
        {"label": "Casual", "value": "Casual outfit"},
        {"label": "Date", "value": "Date outfit tonight"},
        {"label": "Party", "value": "Party outfit tonight"},
        {"label": "Travel", "value": "Airport travel outfit"},
        {"label": "Use my wardrobe", "value": "Use my wardrobe"},
    ]


def _style_clarification_response(query: str, interpretation: Dict[str, Any]) -> Dict[str, Any]:
    occasion = (
        (interpretation.get("board_generation_notes") or {}).get("occasion_kind")
        or interpretation.get("occasion")
        or interpretation.get("interpreted_occasion")
    )
    # Prefer interpretation-provided chips, otherwise use occasion-specific set.
    interp_chips = interpretation.get("chips") if isinstance(interpretation.get("chips"), list) else []
    chips = interp_chips or _clarification_chips_for_occasion(occasion)

    # Pre-merge each chip's value with the original prompt so a chip
    # tap in the FE retransmits the full intent ("beach wear · Beach
    # vacation") and won't re-trigger the same clarification loop.
    original = str(query or "").strip()
    if original:
        merged: List[Dict[str, str]] = []
        for chip in chips:
            if not isinstance(chip, dict):
                continue
            label = str(chip.get("label") or chip.get("value") or "").strip()
            raw_value = str(chip.get("value") or label).strip()
            if not label or not raw_value:
                continue
            already_merged = " · " in raw_value or raw_value.lower().startswith(original.lower())
            value = raw_value if already_merged else f"{original} · {raw_value}"
            merged.append({"label": label, "value": value})
        if merged:
            chips = merged

    occasion_label = str(occasion or "").strip()
    if occasion_label:
        pretty = occasion_label.replace("_", " ").title()
        message = (
            f"{pretty} — got it. What are you dressing for?"
        )
    else:
        message = (
            "Got it. What are you dressing for today? Pick one of these or add a detail like weather, time, or vibe."
        )
    return {
        "success": True,
        "ok": True,
        "type": "clarification",
        "intent": "style",
        "message": {"role": "assistant", "content": message},
        "message_text": message,
        "response": message,
        "text": message,
        "cards": [],
        "style_boards": [],
        "chips": chips,
        "board_ids": "",
        "data": {
            "outfits": [],
            "rendered_boards": [],
            "intent": "style",
            "requires_clarification": True,
            "original_prompt": query,
            "interpreted_occasion": occasion_label or None,
            "clarification": {
                "prompt": query,
                "questions": ["occasion", "weather/timing", "mood/style", "comfort/dress code"],
            },
        },
        "meta": {
            "mode": "style_intent_clarification",
            "intent_status": "clarify",
            "occasion_interpretation": interpretation,
        },
        "audio_job_id": "offline",
    }


def _wardrobe_reality_explanation(
    missing_block: Dict[str, Any], wardrobe: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Phase 5: 'you already own X / adding Y unlocks archetypes Z'. Uses the
    archetype library to compute what the missing piece unlocks. No shopping."""
    try:
        from services.stylist_knowledge_service import ARCHETYPE_LIBRARY
    except Exception:  # noqa: BLE001
        ARCHETYPE_LIBRARY = []
    owned = [
        str((it.get("name") if isinstance(it, dict) else it) or "").strip()
        for it in (wardrobe or [])
    ]
    owned = [o for o in owned if o][:8]
    missing_names = [
        str(m.get("name") or "").strip()
        for m in missing_block.get("missing_items", [])
        if str(m.get("name") or "").strip()
    ]
    # Archetypes whose preferred_items include the missing piece -> "unlocks".
    unlocks: List[str] = []
    miss_blob = " ".join(missing_names).lower()
    for arch in ARCHETYPE_LIBRARY:
        pref = " ".join(str(x) for x in (arch.get("preferred_items") or [])).lower()
        if any(tok in pref for m in missing_names for tok in m.lower().split() if len(tok) > 3):
            unlocks.append(arch.get("name"))
    unlocks = list(dict.fromkeys([u for u in unlocks if u]))[:4]
    return {
        "owned_items": owned,
        "adding_items": missing_names[:3],
        "unlocks_archetypes": unlocks,
    }


def _style_reasoning_chat_response(
    reasoning: Dict[str, Any],
    query: str,
    module_context: str = "",
    wardrobe: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    message = str(reasoning.get("advice") or "").strip()
    chips = reasoning.get("cta") if isinstance(reasoning.get("cta"), list) else []
    mode = str(reasoning.get("mode") or "style_advice")
    visual_directions = reasoning.get("visual_directions")
    if not isinstance(visual_directions, list):
        visual_directions = []

    is_missing_pieces = mode in {"shopping_assist", "missing_pieces"}
    is_visual_inspiration = mode == "visual_inspiration"

    # Phase 7: missing-piece intelligence block for shopping/missing-piece mode.
    missing_block: Dict[str, Any] = {}
    if is_missing_pieces:
        try:
            from services.style_context_service import (
                build_style_context,
                build_missing_piece_intelligence,
            )

            _ctx = build_style_context(
                query=query, mode="missing_pieces", wardrobe_items=wardrobe or []
            )
            _mp = reasoning.get("missing_piece") if isinstance(reasoning.get("missing_piece"), dict) else None
            _reason = str(reasoning.get("missing_piece_reasoning") or "").strip()
            _missing = (
                [_mp]
                if _mp
                else ([{"name": "", "category": "", "reason": _reason, "unlocks": []}] if _reason else [])
            )
            missing_block = build_missing_piece_intelligence(
                wardrobe_summary=_ctx.get("wardrobe_summary", {}),
                missing_items=_missing,
            )
            logger.info(
                "AHVI_MISSING_PIECES_ROUTE missing_items=%d", len(missing_block.get("missing_items") or [])
            )
        except Exception:  # noqa: BLE001
            missing_block = {}

    # P1: visual inspiration metadata block (no image generation).
    visual_board = reasoning.get("visual_inspiration_board") if is_visual_inspiration else None
    if is_visual_inspiration:
        logger.info("AHVI_VISUAL_INSPIRATION_ROUTE title=%r", (visual_board or {}).get("title"))

    # Missing-pieces must NOT also dump visual-direction cards.
    visual_cards = [] if is_missing_pieces else [
        {
            "type": "visual_direction",
            "archetype": str(item.get("archetype") or item.get("direction_name") or item.get("title") or ""),
            "direction_name": str(item.get("direction_name") or ""),
            "title": str(item.get("title") or "Style Direction"),
            "subtitle": str(item.get("subtitle") or item.get("style_direction") or ""),
            "impression": str(item.get("impression") or ""),
            "strategy": str(item.get("strategy") or ""),
            "description": str(item.get("description") or ""),
            "hero_piece": str(item.get("hero_piece") or item.get("heroPiece") or ""),
            "hero_piece_reasoning": str(item.get("hero_piece_reasoning") or item.get("heroPieceReasoning") or ""),
            "palette": item.get("palette") if isinstance(item.get("palette"), list) else [],
            "colors": item.get("colors") if isinstance(item.get("colors"), list) else (item.get("palette") if isinstance(item.get("palette"), list) else []),
            "pieces": item.get("pieces") if isinstance(item.get("pieces"), list) else [],
            "items": item.get("items") if isinstance(item.get("items"), list) else (item.get("pieces") if isinstance(item.get("pieces"), list) else []),
            "why_it_works": str(item.get("why_it_works") or ""),
            "why_this_works": str(item.get("why_this_works") or item.get("why_it_works") or ""),
            "style_note": str(item.get("style_note") or ""),
            "styling_tip": str(item.get("styling_tip") or item.get("style_note") or "")[:80],
            "use_case": str(item.get("use_case") or ""),
            "avoid": item.get("avoid") if isinstance(item.get("avoid"), list) else [],
            "archetype_reasoning": str(item.get("archetype_reasoning") or ""),
            "dna_alignment": item.get("dna_alignment"),
            "wardrobe_alignment": item.get("wardrobe_alignment"),
            "style_dna_alignment": item.get("style_dna_alignment") or item.get("dna_alignment"),
            "persona_fit_reason": item.get("persona_fit_reason"),
            # Wardrobe-match pill: the reasoning engine sets wardrobe_match_pct
            # on each direction; forward it so the frontend can render the pill.
            "wardrobe_match_pct": item.get("wardrobe_match_pct"),
            "image_url": str(item.get("image_url") or item.get("imageUrl") or ""),
            "asset_id": str(item.get("asset_id") or ""),
            "complete_the_look": item.get("complete_the_look") if isinstance(item.get("complete_the_look"), list) else [],
            # Itemized, role-tagged, image-bearing pieces — the frontend renders
            # the visual board from board_items; without it the catalog/visual-
            # inspiration directions showed as a checklist instead of a board.
            "board_items": item.get("board_items") if isinstance(item.get("board_items"), list) else [],
            # Additive styling-intent brief for the frontend board renderer.
            "composition_brief": _build_composition_brief(
                item.get("board_items") if isinstance(item.get("board_items"), list)
                else (item.get("pieces") if isinstance(item.get("pieces"), list)
                      else (item.get("items") if isinstance(item.get("items"), list) else [])),
                str(reasoning.get("occasion") or reasoning.get("formality") or item.get("use_case") or ""),
            ),
        }
        for item in visual_directions
        if isinstance(item, dict)
    ]
    summary_cards = [
        {
            "type": "style_reasoning",
            "title": "Stylist Advice",
            "subtitle": message.split("\n", 1)[0] if message else "Style guidance",
            "mode": mode,
            "occasion": reasoning.get("occasion"),
            "tone": reasoning.get("tone"),
            "formality": reasoning.get("formality"),
        }
    ] if message else []

    # Phase 4: a visible "why this fits YOU" stylist-reasoning block, built from
    # the top route's archetype + alignment fields.
    stylist_reasoning_block: Dict[str, Any] = {}
    _top = visual_directions[0] if visual_directions else {}
    if isinstance(_top, dict) and (_top.get("archetype") or _top.get("archetype_reasoning")):
        stylist_reasoning_block = {
            "type": "stylist_reasoning",
            "archetype": str(_top.get("archetype") or _top.get("title") or "").strip(),
            "why_this_fits_you": str(
                _top.get("archetype_reasoning")
                or _top.get("why_this_works")
                or _top.get("why_it_works") or ""
            ).strip(),
            "dna_alignment": str(_top.get("dna_alignment") or "").strip(),
            "wardrobe_alignment": str(_top.get("wardrobe_alignment") or "").strip(),
        }
        if stylist_reasoning_block.get("archetype"):
            logger.info(
                "AHVI_STYLIST_REASONING_RENDERED archetype=%r",
                stylist_reasoning_block["archetype"],
            )

    # Phase 5: wardrobe-reality explanation — turn the missing-piece block into
    # "you own X / adding Y unlocks Z" using the archetype library.
    if missing_block.get("missing_items"):
        try:
            _wr = _wardrobe_reality_explanation(missing_block, wardrobe or [])
            if _wr:
                missing_block = {**missing_block, **_wr}
        except Exception:  # noqa: BLE001
            pass
    return {
        "success": True,
        "ok": True,
        "type": "stylist_advice",
        "intent": mode,
        "message": {"role": "assistant", "content": message},
        "message_text": message,
        "response": message,
        "text": message,
        "cards": summary_cards + visual_cards,
        # Expose the visual-direction boards under style_boards so the frontend
        # board extractor renders them (it reads style_boards/data.outfits, not
        # `cards`). Empty for non-visual modes since visual_cards is empty then.
        "style_boards": visual_cards,
        "chips": chips,
        "board_ids": "",
        "data": {
            "intent": mode,
            "style_mode": mode,
            "style_reasoning": reasoning,
            "stylist_mode": True,
            "cta_actions": chips,
            "visual_directions": visual_directions,
            "pairing_routes": reasoning.get("pairing_routes") if isinstance(reasoning.get("pairing_routes"), list) else [],
            "transition_plan": reasoning.get("transition_plan") if isinstance(reasoning.get("transition_plan"), dict) else None,
            "is_transition": bool(reasoning.get("is_transition")),
            "anchor_item": reasoning.get("anchor_item") if isinstance(reasoning.get("anchor_item"), dict) else None,
            "archetype_reasoning": reasoning.get("archetype_reasoning"),
            "dna_alignment": reasoning.get("dna_alignment"),
            "wardrobe_alignment": reasoning.get("wardrobe_alignment"),
            # Persisted style session — FE echoes this back in current_memory so
            # follow-ups (use wardrobe / find missing / visual) keep the anchor.
            "last_style_context": reasoning.get("last_style_context") or None,
            "goal": reasoning.get("goal"),
            "impression": reasoning.get("impression"),
            "confidence_strategy": reasoning.get("confidence_strategy"),
            "missing_piece_intelligence": missing_block or None,
            "visual_inspiration_board": visual_board or None,
        },
        "blocks": (
            ([reasoning["advice_block"]]
                if isinstance(reasoning.get("advice_block"), dict) else [])
            + ([{"type": "transition_plan", **reasoning["transition_plan"]}]
                if isinstance(reasoning.get("transition_plan"), dict) else [])
            + ([stylist_reasoning_block] if stylist_reasoning_block else [])
            + ([missing_block] if missing_block.get("missing_items") else [])
            + ([visual_board] if isinstance(visual_board, dict) and visual_board else [])
        ),
        "meta": {
            **(reasoning.get("meta") if isinstance(reasoning.get("meta"), dict) else {}),
            "mode": "style_reasoning",
            "style_mode": mode,
            "intent": mode,
            "module_context": module_context or "chat",
            "wardrobe_lookup": False,
            "original_prompt": query,
        },
        "audio_job_id": "offline",
    }


_VAGUE_STYLE_LITERALS = {
    "outfit for today",
    "suggest outfit for today",
    "suggest an outfit for today",
    "style me",
    "what should i wear",
    "what to wear",
    "outfit",
    "daily wear",
    "today outfit",
}


def _is_greeting(text: str) -> bool:
    q = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    return q in {
        "hi",
        "hello",
        "hey",
        "hi ahvi",
        "hello ahvi",
        "hey ahvi",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
    }


def _is_help_identity_request(text: str) -> bool:
    q = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    return q in {
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


_STYLE_PRIORITY_WORDS = (
    "wear", "outfit", "outfits", "style", "dress", "dressed", "look", "looks",
    "pair", "pairing", "transition", "avoid", "improve", "make it look",
    "how do i wear", "how do i style", "what to wear", "what should i wear",
    "what goes with", "ways to style", "how can i wear", "how do i dress",
)
_EXPLICIT_FOOD_WORDS = (
    "meal plan", "meal-plan", "what to eat", "what should i eat", "calories",
    "calorie", "recipe", "protein", "nutrition", "food", "meal idea", "meal ideas",
    "diet plan", "eat", "snack", "carb", "macros", "grocery",
)


def _is_explicit_food_intent(text: str) -> bool:
    q = re.sub(r"\s+", " ", str(text or "").lower())
    return any(w in q for w in _EXPLICIT_FOOD_WORDS)


def _is_style_priority_query(text: str) -> bool:
    """Style wins over diet/calendar when the user clearly asks about clothing,
    unless they explicitly ask about food/eating."""
    q = re.sub(r"\s+", " ", str(text or "").lower())
    if _is_explicit_food_intent(q):
        return False
    return any(w in q for w in _STYLE_PRIORITY_WORDS)


def _is_find_this_request(text: str) -> bool:
    q = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return (
        q.startswith("find this")
        or q.startswith("find similar")
        or q.startswith("shop this")
        or q.startswith("buy similar")
        or "find this:" in q
    )


_FIND_THIS_COLOR_WORDS = {
    "black", "white", "brown", "tan", "navy", "blue", "grey", "gray", "cream",
    "beige", "olive", "green", "burgundy", "maroon", "charcoal", "silver",
    "gold", "ivory", "stone", "khaki", "dark", "light",
}

_FIND_THIS_CATEGORY_RULES = [
    ("footwear", "loafers", ("loafer", "loafers", "penny loafer", "penny loafers")),
    ("footwear", "sneakers", ("sneaker", "sneakers", "trainer", "trainers")),
    ("footwear", "boots", ("boot", "boots", "chelsea boot", "chelsea boots")),
    ("footwear", "formal shoes", ("derby", "derbies", "oxford shoe", "formal shoe", "formal shoes")),
    ("outerwear", "blazer", ("blazer", "sport coat", "jacket")),
    ("outerwear", "overshirt", ("overshirt", "shacket", "utility layer")),
    ("top", "shirt", ("shirt", "button down", "button-down", "oxford")),
    ("top", "knitwear", ("knit", "sweater", "jumper", "polo")),
    ("bottom", "jeans", ("jean", "jeans", "denim")),
    ("bottom", "trousers", ("trouser", "trousers", "pants", "chinos", "chino")),
    ("accessory", "watch", ("watch",)),
    ("accessory", "belt", ("belt",)),
    ("accessory", "bag", ("bag", "tote", "sling", "pouch")),
    ("accessory", "jewelry", ("earring", "earrings", "necklace", "bracelet", "ring")),
]


def _find_this_extract_item(raw: str) -> str:
    item = str(raw or "").strip()
    for pref in (
        "find this:",
        "find this",
        "find similar to",
        "find similar",
        "shop this",
        "buy similar to",
        "buy similar",
    ):
        if item.lower().startswith(pref):
            item = item[len(pref):].strip(" :-")
            break
    if "find this:" in item.lower():
        item = item.lower().split("find this:", 1)[1].strip()
    # Strip structured suffix; it is parsed separately.
    item = re.split(r"\s+\|\s+", item, maxsplit=1)[0].strip()
    return item or "this piece"


def _find_this_parse_meta(raw: str) -> Dict[str, str]:
    text = str(raw or "")
    meta: Dict[str, str] = {}
    for part in re.split(r"\s+\|\s+", text):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")
        value = value.strip()
        if key and value:
            meta[key] = value
    return meta


def _find_this_infer_category(item_name: str, meta: Dict[str, str]) -> tuple[str, str]:
    category = str(meta.get("category") or "").strip()
    subcategory = str(meta.get("subcategory") or meta.get("sub_category") or "").strip()
    if category and subcategory:
        return category, subcategory
    name = str(item_name or "").lower()
    for cat, sub, markers in _FIND_THIS_CATEGORY_RULES:
        if any(marker in name for marker in markers):
            return category or cat, subcategory or sub
    return category or "fashion", subcategory or ""


def _find_this_infer_color(item_name: str, meta: Dict[str, str]) -> str:
    if meta.get("color"):
        return meta["color"]
    tokens = re.findall(r"[a-z0-9]+", str(item_name or "").lower())
    colors: List[str] = []
    for token in tokens:
        if token in _FIND_THIS_COLOR_WORDS:
            colors.append(token)
    # Preserve useful compounds like "dark brown".
    if len(colors) >= 2 and colors[0] in {"dark", "light"}:
        return f"{colors[0]} {colors[1]}"
    return colors[0] if colors else ""


def _find_this_profile_gender(user_profile: Optional[Dict[str, Any]]) -> str:
    if not isinstance(user_profile, dict):
        return ""
    for key in ("style_gender", "gender", "preferred_gender", "target_gender"):
        value = str(user_profile.get(key) or "").strip().lower()
        if value in {"male", "man", "men", "mens", "masculine"}:
            return "men"
        if value in {"female", "woman", "women", "womens", "feminine"}:
            return "women"
    return ""


def _find_this_grounding(query: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = str(query or "").strip()
    meta = _find_this_parse_meta(raw)
    item_name = meta.get("item_name") or meta.get("name") or _find_this_extract_item(raw)
    category, subcategory = _find_this_infer_category(item_name, meta)
    color = _find_this_infer_color(item_name, meta)
    occasion = str(meta.get("occasion") or "").strip()
    archetype = str(meta.get("archetype") or meta.get("style_archetype") or "").strip()
    gender = str(meta.get("gender") or _find_this_profile_gender(user_profile)).strip()
    item_has_color = bool(color and color.lower() in item_name.lower())
    search_parts = ([] if item_has_color else [color]) + [item_name, gender]
    search_query = " ".join(part for part in search_parts if part).strip()
    search_query = re.sub(r"\s+", " ", search_query)
    grounding = {
        "item_name": item_name,
        "category": category,
        "subcategory": subcategory,
        "color": color,
        "occasion": occasion,
        "archetype": archetype,
        "gender": gender,
        "search_query": search_query or item_name,
    }
    logger.info(
        "AHVI_FIND_THIS_GROUNDED item=%r category=%s subcategory=%s color=%s archetype=%r occasion=%r gender=%s",
        item_name,
        category,
        subcategory,
        color,
        archetype,
        occasion,
        gender,
    )
    return grounding


def _shopping_intent_response(query: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Safe placeholder for the Find This CTA until product search ships.
    Routes the intent (shopping_assist) without falling back to generic
    style advice."""
    grounding = _find_this_grounding(query, user_profile=user_profile)
    item = grounding["item_name"]
    block = {
        "type": "shopping_intent",
        "query": grounding["search_query"],
        **grounding,
        "message": f"I'll help you find similar options for {item}.",
        "status": "pending_catalog",
    }
    logger.info("AHVI_FIND_THIS_ROUTE item=%r search_query=%r status=pending_catalog", item, grounding["search_query"])
    msg = block["message"]
    return {
        "success": True,
        "ok": True,
        "type": "shopping_intent",
        "intent": "shopping_assist",
        "message": {"role": "assistant", "content": msg},
        "message_text": msg,
        "response": msg,
        "text": msg,
        "cards": [],
        "style_boards": [],
        "blocks": [block],
        "chips": [],
        "board_ids": "",
        "data": {"shopping_intent": block, "find_this": grounding, "intent": "shopping_assist"},
        "meta": {"mode": "shopping_intent", "status": "pending_catalog", "find_this_grounded": True},
        "audio_job_id": "offline",
    }


def _is_small_talk(text: str) -> bool:
    q = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    return q in {"how are you", "thanks", "thank you"}


def _ahvi_greeting_response(module_context: str = ""):
    return {
        "success": True,
        "message": "Hi, I’m here. What would you like help with today?",
        "message_text": "Hi, I’m here. What would you like help with today?",
        "response": "Hi, I’m here. What would you like help with today?",
        "board": "general",
        "type": "text",
        "cards": [],
        "style_boards": [],
        "chips": [
            {"label": "Today's Outfit", "value": "Outfit for today"},
            {"label": "Office Look", "value": "Office outfit"},
            {"label": "Plan My Day", "value": "Plan my day"},
            {"label": "Workout", "value": "Today's workout"},
        ],
        "data": {},
        "meta": {
            "mode": "greeting_bypass",
            "module_context": module_context or "chat",
        },
        "audio_job_id": "offline",
    }


def _ahvi_help_identity_response(text: str, module_context: str = ""):
    q = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    if q in {"who are you", "what are you", "what is ahvi", "tell me about yourself"}:
        message = (
            "I'm AHVI, your personal assistant for Style, Planning, and Preparation. "
            "I help you get dressed, organize your day, and prepare for what’s coming next."
        )
        chips = [
            {"label": "Today's Outfit", "value": "Outfit for today"},
            {"label": "Plan My Day", "value": "Plan my day"},
            {"label": "Meals", "value": "Eat today"},
            {"label": "Workout", "value": "Today's workout"},
        ]
    else:
        message = (
            "I can help with Style, Planning, and Preparation. I can create outfits from your wardrobe, "
            "plan your day, suggest meals and workouts, organize routines, and help you prepare for events or trips."
        )
        chips = [
            {"label": "Today's Outfit", "value": "Outfit for today"},
            {"label": "Plan My Day", "value": "Plan my day"},
            {"label": "Eat Today", "value": "Eat today"},
            {"label": "Workout", "value": "Today's workout"},
        ]
    return {
        "success": True,
        "message": message,
        "message_text": message,
        "response": message,
        "board": "general",
        "type": "text",
        "cards": [],
        "style_boards": [],
        "chips": chips,
        "data": {},
        "meta": {
            "mode": "help_identity_bypass",
            "module_context": module_context or "chat",
        },
        "audio_job_id": "offline",
    }


def _ahvi_small_talk_response(module_context: str = ""):
    message = (
        "Ready to help. Are we styling an outfit, planning your day, or preparing for something upcoming?"
    )
    return {
        "success": True,
        "message": message,
        "message_text": message,
        "response": message,
        "board": "general",
        "type": "text",
        "cards": [],
        "style_boards": [],
        "chips": [
            {"label": "Style Me", "value": "Style me"},
            {"label": "Plan My Day", "value": "Plan my day"},
            {"label": "Eat Today", "value": "Eat today"},
            {"label": "Workout", "value": "Today's workout"},
        ],
        "data": {},
        "meta": {
            "mode": "small_talk_bypass",
            "module_context": module_context or "chat",
        },
        "audio_job_id": "offline",
    }


def _organize_domain_for_module(module: str) -> str:
    module_key = str(module or "").strip().lower().replace("-", "_")
    return {
        "meal_planner": "diet",
        "meals": "diet",
        "meal": "diet",
        "diet": "diet",
        "workout": "fitness",
        "fitness": "fitness",
        "calendar": "calendar",
        "planner": "calendar",
        "plan": "calendar",
        "planning": "calendar",
        "skincare": "skincare",
        "medicines": "medi",
        "medicine": "medi",
        "meds": "medi",
        "bills": "bills",
    }.get(module_key, module_key or "chat")


def _run_module_chat_sync(payload: Dict[str, Any], user_id: str = "") -> Dict[str, Any]:
    return asyncio.run(handle_module_chat(payload, user_id=user_id))


# Broad fashion words that, when used alone or in a 1-4 word prompt,
# need clarification before we burn 10s of orchestrator time.
_BROAD_FASHION_TOKENS = {
    "wear",
    "outfit",
    "style",
    "look",
    "beach",
    "office",
    "party",
    "date",
    "travel",
    "gym",
    "wedding",
    "casual",
    "formal",
    "vacation",
    "workout",
    "dinner",
    "brunch",
}

# Signals that mean the prompt already has enough context. If ANY of
# these appear we let the orchestrator run.
_SPECIFICITY_TOKENS = {
    # weather / timing
    "rain", "rainy", "cold", "hot", "humid", "summer", "winter", "monsoon",
    "tomorrow", "today", "tonight", "morning", "evening", "afternoon",
    "weekend", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    # wardrobe / item anchors
    "my wardrobe", "wardrobe", "use my", "with my", "from my", "linen",
    "denim", "shirt", "trouser", "jeans", "skirt", "saree", "kurta",
    "blazer", "dress", "shoes", "sneakers", "loafers", "white", "black",
    "blue", "red", "green", "pink", "navy", "cream", "beige",
    # event detail / venue
    "rooftop", "candle", "ceremony", "reception", "interview", "meeting",
    "boardroom", "client", "presentation", "airport", "trip", "hike",
    # gender / body / preference
    "men", "women", "male", "female", "petite", "tall", "curvy", "modest",
    "ethnic", "western",
}


def _is_vague_style_prompt(text: str) -> bool:
    """Legacy literal check; kept as a fast pre-pass."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized in _VAGUE_STYLE_LITERALS


def _needs_style_clarification(
    prompt: str, interpreted_occasion: Optional[str] = None
) -> bool:
    """Decide whether a style prompt is too vague to burn orchestrator time.

    Returns True when the prompt is short (1-4 words), contains a broad
    fashion term, and lacks any specificity signal (weather/time/anchor
    item/event detail). Empty / non-style prompts return False so they
    don't accidentally divert non-style flows.
    """
    text = str(prompt or "").strip().lower()
    if not text:
        return False

    if _is_vague_style_prompt(text):
        return True

    normalized = re.sub(r"[^a-z0-9\s]", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = normalized.split() if normalized else []
    if not tokens:
        return False

    if str(interpreted_occasion or "").strip():
        return False

    # Only short prompts are candidates for clarification. Anything 5+
    # words usually has enough signal; let the orchestrator decide.
    if len(tokens) > 4:
        return False

    has_broad = (
        any(t in _BROAD_FASHION_TOKENS for t in tokens)
        or any(b in normalized for b in _BROAD_FASHION_TOKENS)
    )
    if not has_broad:
        return False

    # Multi-word phrases too (e.g. "use my wardrobe").
    has_specificity = (
        any(t in _SPECIFICITY_TOKENS for t in tokens)
        or any(s in normalized for s in _SPECIFICITY_TOKENS)
    )
    if has_specificity:
        return False

    return True


def _weather_cache_key(lat: Any, lon: Any) -> str:
    return f"{float(lat):.4f}:{float(lon):.4f}"


def _get_weather_cached(lat: Any, lon: Any) -> Dict[str, Any]:
    key = _weather_cache_key(lat, lon)
    cached = _WEATHER_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    weather = get_hourly_weather(lat=float(lat), lon=float(lon))
    _WEATHER_CACHE.set(key, weather)
    return weather


def shutdown_chat_resources(wait_seconds: float = 5.0) -> None:
    """Called from app shutdown to drain in-flight orchestrator work."""
    try:
        _ORCHESTRATOR_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    except TypeError:
        # cancel_futures kw added in 3.9; fallback for unusual runtimes
        _ORCHESTRATOR_EXECUTOR.shutdown(wait=True)
    except Exception:
        logger.exception("orchestrator executor shutdown failed")


def _build_history(messages: List["Message"]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for msg in messages[-8:]:
        role = str(getattr(msg, "role", "user")).lower()
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue
        history.append({"role": role, "text": content[:500]})
    return history


def _normalize_memory_history(events: Any, max_items: int = 12) -> List[Dict[str, Any]]:
    if not isinstance(events, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for event in events[-max_items:]:
        if not isinstance(event, dict):
            continue
        row: Dict[str, Any] = {}
        if event.get("intent"):
            row["intent"] = str(event.get("intent"))[:80]
        if isinstance(event.get("slots"), dict):
            row["slots"] = event.get("slots")
        if event.get("role"):
            row["role"] = str(event.get("role"))[:32]
        if event.get("text"):
            row["text"] = str(event.get("text"))[:500]
        if row:
            normalized.append(row)
    return normalized


def _is_fast_wardrobe_count_query(text: str) -> bool:
    lowered = str(text or "").lower()
    count_words = ["how many", "count", "number of", "total", "do i have"]
    wardrobe_words = [
        "wardrobe",
        "closet",
        "outfit",
        "outfits",
        "tops",
        "top",
        "shirts",
        "shirt",
        "pants",
        "trousers",
        "jeans",
        "bottoms",
        "shoes",
        "footwear",
        "dress",
        "dresses",
        "accessories",
        "jewelry",
        "bags",
        "bag",
    ]
    return any(k in lowered for k in count_words) and any(
        k in lowered for k in wardrobe_words
    )


_CHAT_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _chat_tokens(value: Any) -> List[str]:
    return _CHAT_TOKEN_RE.sub(" ", str(value or "").lower()).strip().split()


def _chat_has_any(tokens: List[str], words: List[str]) -> bool:
    return any(word in tokens for word in words)


def _infer_chat_category(item: Dict[str, Any]) -> str:
    # Delegate to the shared taxonomy. Behavior unchanged.
    from services.category_taxonomy import categorize_for_chat

    return categorize_for_chat(item)


def _fast_wardrobe_count_response(user_id: str, query_text: str) -> Dict[str, Any]:
    # Paginate fully so totals are accurate for wardrobes >100 items.
    docs: List[Dict[str, Any]] = []
    try:
        proxy = AppwriteProxy()
        page_size = 100
        offset = 0
        while True:
            page = proxy.list_documents(
                "outfits", user_id=user_id, limit=page_size, offset=offset
            )
            if not page:
                break
            docs.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
            if offset >= 5000:  # safety cap
                break
    except Exception:
        logger.warning("fast_wardrobe_count_response fetch failed", exc_info=True)
        docs = []

    counts = {"tops": 0, "bottoms": 0, "shoes": 0, "dresses": 0, "accessories": 0}
    for d in docs:
        cat = _infer_chat_category(d)

        if cat in {"Tops", "Outerwear"}:
            counts["tops"] += 1
        elif cat == "Bottoms":
            counts["bottoms"] += 1
        elif cat == "Footwear":
            counts["shoes"] += 1
        elif cat == "Dresses":
            counts["dresses"] += 1
        else:
            counts["accessories"] += 1

    lowered = str(query_text or "").lower()
    if any(
        k in lowered for k in ["top", "tops", "shirt", "shirts", "blouse", "blouses"]
    ):
        message = f"You have {counts['tops']} tops in your wardrobe."
    elif any(
        k in lowered
        for k in [
            "bottom",
            "bottoms",
            "pant",
            "pants",
            "trouser",
            "trousers",
            "jean",
            "jeans",
        ]
    ):
        message = f"You have {counts['bottoms']} bottoms in your wardrobe."
    elif any(
        k in lowered for k in ["shoe", "shoes", "footwear", "sneaker", "sneakers"]
    ):
        message = f"You have {counts['shoes']} shoes in your wardrobe."
    else:
        total = len(docs)
        message = (
            f"You currently have {total} items: {counts['tops']} tops, {counts['bottoms']} bottoms, "
            f"{counts['shoes']} shoes, {counts['dresses']} dresses, and {counts['accessories']} accessories."
        )

    return {
        "success": True,
        "message": message,
        "board": "wardrobe",
        "type": "stats",
        "cards": [
            {"id": "tops", "title": "Tops", "kind": "stat", "value": counts["tops"]},
            {
                "id": "bottoms",
                "title": "Bottoms",
                "kind": "stat",
                "value": counts["bottoms"],
            },
            {"id": "shoes", "title": "Shoes", "kind": "stat", "value": counts["shoes"]},
            {
                "id": "dresses",
                "title": "Dresses",
                "kind": "stat",
                "value": counts["dresses"],
            },
            {
                "id": "accessories",
                "title": "Accessories",
                "kind": "stat",
                "value": counts["accessories"],
            },
        ],
        "data": {"counts": counts, "total_items": len(docs)},
        "meta": {"intent": "wardrobe_query", "domain": "wardrobe", "fast_path": True},
        "audio_job_id": "offline",
    }


# ================= AHVI CLEAN CHAT STYLE V2 BEGIN =================
# One clean style adapter for chat.
# Source of truth: brain.outfit_pipeline.get_daily_outfits
# Purpose:
# - remove duplicate _demo_style_board_payload definitions
# - sanitize cards before they reach Flutter
# - one watch max
# - category/role/slot guaranteed for top, bottom, footwear, accessory
# - fallback still works if orchestrator times out

_AHVI_MALE_STYLE_GENDERS = {"m", "male", "man", "men", "mens", "boy"}
_AHVI_FEMALE_STYLE_GENDERS = {
    "f",
    "female",
    "woman",
    "women",
    "womens",
    "girl",
    "ladies",
}
_AHVI_UNISEX_STYLE_GENDERS = {"unisex", "neutral", "genderless", "any"}

_AHVI_FEMININE_ONLY_GARMENTS = {
    "saree",
    "sari",
    "lehenga",
    "gown",
    "skirt",
    "skirts",
    "blouse",
    "kurti",
}
_AHVI_MALE_TRADITIONAL_GARMENTS = {"sherwani", "achkan"}

_AHVI_EXPLICIT_FEMININE_REQUEST = {
    "saree",
    "sari",
    "lehenga",
    "gown",
    "skirt",
    "skirts",
    "female",
    "women",
    "woman",
    "ladies",
    "feminine",
}


def _ahvi_coerce_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            import json as _json

            parsed = _json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _ahvi_normalize_style_gender(value):
    raw = str(value or "").strip().lower()
    if raw in _AHVI_MALE_STYLE_GENDERS:
        return "male"
    if raw in _AHVI_FEMALE_STYLE_GENDERS:
        return "female"
    if raw in _AHVI_UNISEX_STYLE_GENDERS:
        return "unisex"
    return ""


def _ahvi_profile_style_gender(profile):
    profile = profile or {}
    candidates = [
        profile.get("style_gender"),
        profile.get("gender"),
        profile.get("preferred_gender"),
        profile.get("target_gender"),
    ]

    for key in ("preferences", "style_preferences", "stylePreference", "profile"):
        nested = _ahvi_coerce_dict(profile.get(key))
        candidates.extend(
            [
                nested.get("style_gender"),
                nested.get("gender"),
                nested.get("preferred_gender"),
                nested.get("target_gender"),
            ]
        )

    for value in candidates:
        gender = _ahvi_normalize_style_gender(value)
        if gender:
            return gender

    return "unisex"


def _ahvi_resolve_effective_user_profile(user_id, request_profile=None):
    request_profile = request_profile if isinstance(request_profile, dict) else {}
    try:
        from services.data_access_service import get_user_profile, merge_user_profiles

        stored = get_user_profile(user_id=str(user_id or "").strip())
        merged = merge_user_profiles(stored, request_profile)
        merged.setdefault("user_id", user_id)
        return merged
    except Exception:
        out = dict(request_profile)
        out.setdefault("user_id", user_id)
        return out


def _ahvi_query_allows_feminine_item(query_text):
    tokens = set(_chat_tokens(query_text))
    return bool(tokens.intersection(_AHVI_EXPLICIT_FEMININE_REQUEST))


def _ahvi_item_tokens(item):
    blob = " ".join(
        str(item.get(k, "") or "")
        for k in (
            "slot",
            "role",
            "type",
            "category",
            "cat",
            "category_group",
            "sub_category",
            "subcategory",
            "subCategory",
            "name",
            "label",
            "description",
            "gender",
            "style_gender",
            "target_gender",
            "audience",
            "department",
            "intended_for",
            "wearer",
        )
    )
    return set(_chat_tokens(blob))


def _ahvi_item_allowed_for_user_profile(item, user_profile=None, query_text=""):
    if not isinstance(item, dict):
        return False

    target_gender = _ahvi_profile_style_gender(user_profile or {})

    if target_gender != "male":
        return True

    if _ahvi_query_allows_feminine_item(query_text):
        return True

    tokens = _ahvi_item_tokens(item)

    audience_blob = " ".join(
        str(item.get(k, "") or "")
        for k in (
            "gender",
            "style_gender",
            "target_gender",
            "audience",
            "department",
            "intended_for",
            "wearer",
        )
    )
    audience_tokens = set(_chat_tokens(audience_blob))

    if audience_tokens.intersection(_AHVI_FEMALE_STYLE_GENDERS):
        return False

    if tokens.intersection(_AHVI_FEMININE_ONLY_GARMENTS):
        return False

    if _infer_chat_category(item) == "Dresses" and not tokens.intersection(
        _AHVI_MALE_TRADITIONAL_GARMENTS
    ):
        return False

    return True


def _fetch_wardrobe_for_style(
    user_id: str, request_wardrobe: Any
) -> List[Dict[str, Any]]:
    if isinstance(request_wardrobe, list):
        items = [dict(i) for i in request_wardrobe if isinstance(i, dict)]
    else:
        try:
            docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=100)
            if isinstance(docs, dict):
                rows = docs.get("documents") or docs.get("items") or []
            else:
                rows = docs or []
            items = [dict(i) for i in rows if isinstance(i, dict)]
        except Exception as exc:
            logger.warning("style wardrobe fetch failed user_id=%s error=%s", user_id, exc)
            items = []

    # AHVI Metadata Validator enrichment — merge parsed style_metadata onto
    # each outfit row when available. Failures must not break the style flow.
    try:
        from services.agent_metadata_validator import (
            fetch_style_metadata_docs_for_user,
            merge_style_metadata_into_wardrobe_items,
        )

        meta_docs = fetch_style_metadata_docs_for_user(user_id)
        if meta_docs:
            items = merge_style_metadata_into_wardrobe_items(items, meta_docs)
    except Exception as exc:
        logger.debug("style wardrobe metadata merge failed user_id=%s err=%s", user_id, exc)
    return items


def _ahvi_style_occasion(query_text):
    q = str(query_text or "").lower()
    # Multi-event / transition prompts must win before generic mapping so
    # "basketball game ... then team dinner" is not flattened to date night.
    try:
        from services.style_context_service import detect_multi_event

        if detect_multi_event(query_text):
            return "multi_event"
    except Exception:  # noqa: BLE001
        pass
    if any(k in q for k in ["swim", "swimming", "swimwear", "swimsuit", "pool"]):
        return "swimming"
    if any(k in q for k in ["beach", "seaside", "coastal", "resort"]):
        return "beach"
    if any(k in q for k in ["gym", "workout", "fitness", "training", "yoga"]):
        return "workout"
    if any(
        k in q
        for k in [
            # Indian wedding + festive ceremonies → wedding occasion so the
            # ethnic/festive asset boost + men's festive text guard fire.
            "wedding", "marriage", "matrimony", "reception", "ceremony",
            "sangeet", "haldi", "mehendi", "mehndi", "engagement", "shaadi",
            "baraat", "roka", "nikah", "varmala", "diwali", "eid", "navratri",
            "holi", "pongal", "onam", "festive", "festival",
        ]
    ):
        return "wedding"
    if any(k in q for k in ["funeral", "memorial", "condolence", "wake"]):
        return "funeral"
    if any(k in q for k in ["temple", "mandir", "puja", "pooja", "darshan", "shrine"]):
        return "temple_modest"
    if any(k in q for k in ["date", "dinner", "night"]):
        return "date night"
    # Office ONLY on explicit work signals — never as a generic-daily fallback.
    if any(
        k in q
        for k in [
            "office",
            "meeting",
            "work",
            "client",
            "presentation",
            "corporate",
            "business",
            "interview",
        ]
    ):
        return "office"
    if any(k in q for k in ["party", "club", "night out", "rave"]):
        return "party"
    if any(k in q for k in ["travel", "airport", "trip"]):
        return "travel"
    if any(
        k in q
        for k in ["coffee", "casual", "outing", "weekend", "street", "sport", "outdoor"]
    ):
        return "casual outing"

    # Generic daily style ("what should I wear today", "suggest an outfit")
    # routes to daily_style/today — NOT office.
    logger.info("AHVI_DAILY_STYLE_ROUTE occasion=today prompt=%r", str(query_text)[:60])
    return "today"


def _daily_wear_style_tips_payload(query_text: str, user_id: str) -> Dict[str, Any] | None:
    """Fast path for Daily Wear's Ask AHVI sheet.

    Daily Wear sends a style-advice prompt with the current outfit/weather
    embedded in text. That flow needs quick advice, not the full board
    generator. Keeping it here avoids the slow style-board path and prevents
    the frontend from sitting on the typing indicator for simple tips.
    """
    raw = str(query_text or "")
    q = raw.lower()
    if "current outfit:" not in q and "weather:" not in q:
        return None
    if not any(k in q for k in ("style tip", "style tips", "give me style", "tips")):
        return None

    def _line_after(label: str) -> str:
        pattern = re.compile(rf"{re.escape(label)}\s*:\s*([^\n]+)", re.IGNORECASE)
        match = pattern.search(raw)
        return (match.group(1).strip(" .") if match else "").strip()

    outfit_line = _line_after("Current outfit")
    weather_line = _line_after("Weather")
    outfit_name = (outfit_line.split(" - ", 1)[0] if outfit_line else "this look").strip()

    heat_note = ""
    if any(k in q for k in ("very hot", "hot", "40°", "40c", "39°", "38°")):
        heat_note = " Since it is very hot, keep the styling breathable: avoid heavy layers, thick socks, and dark heat-trapping extras."
    weather_note = f" Weather context: {weather_line}." if weather_line else ""
    message = (
        f"{outfit_name} already reads relaxed and warm-weather friendly."
        f"{weather_note}{heat_note} For a sharper finish, keep the linen/breezy base, roll sleeves neatly, choose light footwear, and add one practical accessory like sunglasses, a slim watch, or a tote. If this is for evening, swap to cleaner shoes and keep the palette calm."
    )
    logger.info(
        "daily_wear.style_tips.fast_response user_id=%s outfit=%r weather=%r",
        user_id,
        outfit_name,
        weather_line,
    )
    return {
        "success": True,
        "type": "style_advice",
        "module": "style",
        "domain": "style",
        "intent": "daily_wear_style_tips",
        "message": message,
        "message_text": message,
        "response": message,
        "chips": ["Make it cooler", "More polished", "Swap footwear", "Show alternatives"],
        "quick_actions": ["Make it cooler", "More polished", "Swap footwear", "Show alternatives"],
        "cards": [],
        "style_boards": [],
        "data": {},
        "meta": {"fast_daily_wear_style_tips": True},
    }


def _ahvi_style_image(item):
    if not isinstance(item, dict):
        return ""

    # Prefer the cleanest asset for style boards.
    # normalized_url is the 1024x1024 transparent PNG created by the backend
    # image normalizer. raw_url is intentionally last for debugging/fallback only.
    candidates = [
        item.get("normalized_url"),
        item.get("normalizedUrl"),
        item.get("normalized_image_url"),
        item.get("normalizedImageUrl"),
        item.get("transparent_url"),
        item.get("transparentUrl"),
        item.get("processed_url"),
        item.get("processedUrl"),
        item.get("png_url"),
        item.get("pngUrl"),
        item.get("cutout_url"),
        item.get("cutoutUrl"),
        item.get("masked_url"),
        item.get("maskedUrl"),
        item.get("masked_image_url"),
        item.get("maskedImageUrl"),
        item.get("image_url"),
        item.get("imageUrl"),
        item.get("url"),
        item.get("image"),
        item.get("raw_url"),
        item.get("rawUrl"),
        item.get("raw_image_url"),
        item.get("rawImageUrl"),
    ]

    for value in candidates:
        url = str(value or "").strip()
        if url and url.lower() not in {"null", "none", "undefined"}:
            return url

    return ""


def _ahvi_style_blob(item):
    if not isinstance(item, dict):
        return ""
    return " ".join(
        str(item.get(k, "") or "")
        for k in (
            "role",
            "slot",
            "type",
            "category",
            "cat",
            "category_group",
            "sub_category",
            "subcategory",
            "subCategory",
            "name",
            "label",
            "description",
            "pattern",
            "color",
            "color_name",
        )
    ).lower()


def _ahvi_style_key(item):
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("$id")
        or item.get("id")
        or item.get("item_id")
        or item.get("itemId")
        or item.get("image_id")
        or item.get("name")
        or item.get("label")
        or id(item)
    ).lower()


def _ahvi_style_role(item):
    tokens = set(_chat_tokens(_ahvi_style_blob(item)))

    if tokens.intersection(
        {
            "shoe",
            "shoes",
            "sneaker",
            "sneakers",
            "boot",
            "boots",
            "heel",
            "heels",
            "sandal",
            "sandals",
            "loafer",
            "loafers",
            "slipper",
            "slippers",
            "slider",
            "sliders",
            "footwear",
        }
    ):
        return "footwear"

    if tokens.intersection(
        {
            "watch",
            "watches",
            "belt",
            "belts",
            "cap",
            "caps",
            "hat",
            "hats",
            "sunglass",
            "sunglasses",
            "eyewear",
            "glasses",
            "bag",
            "bags",
            "purse",
            "handbag",
            "clutch",
            "tote",
            "jewelry",
            "jewellery",
            "ring",
            "rings",
            "necklace",
            "necklaces",
            "bracelet",
            "bracelets",
            "earring",
            "earrings",
            "scarf",
            "scarves",
            "accessory",
            "accessories",
        }
    ):
        return "accessory"

    if tokens.intersection(
        {"dress", "dresses", "saree", "sari", "lehenga", "gown", "jumpsuit", "sherwani"}
    ):
        return "dress"

    # Tops before bottoms: short-sleeved shirt must not become shorts.
    if tokens.intersection(
        {
            "top",
            "tops",
            "shirt",
            "shirts",
            "tee",
            "tshirt",
            "tshirts",
            "polo",
            "polos",
            "jacket",
            "blazer",
            "sweater",
            "hoodie",
            "kurta",
            "kurti",
            "tunic",
            "tunics",
        }
    ):
        return "top"

    if tokens.intersection(
        {
            "bottom",
            "bottoms",
            "pant",
            "pants",
            "trouser",
            "trousers",
            "jean",
            "jeans",
            "shorts",
            "skirt",
            "skirts",
            "chino",
            "chinos",
        }
    ):
        return "bottom"

    return "unknown"


def _ahvi_style_names(items):
    return [
        str(
            (i or {}).get("name")
            or (i or {}).get("label")
            or (i or {}).get("category")
            or ""
        )
        for i in items or []
        if isinstance(i, dict)
    ]


def _ahvi_router_style_fallback_enabled():
    return os.getenv("AHVI_ENABLE_ROUTER_STYLE_FALLBACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ahvi_style_action_chips() -> List[str]:
    return list(STYLE_ACTION_CHIPS)


def _normalize_action_key(value: Any) -> str:
    q = re.sub(r"[^a-z0-9\s_]", " ", str(value or "").lower())
    q = re.sub(r"\s+", " ", q).strip().replace(" ", "_")
    return q


def _is_use_wardrobe_action(*, action: Any = "", prompt: str = "") -> bool:
    action_key = _normalize_action_key(action)
    q = re.sub(r"[^a-z0-9\s]", " ", str(prompt or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    # Substring match (not just prefix): "office outfit using my wardrobe" has
    # the wardrobe cue mid-sentence. Includes the gerund "using my wardrobe".
    # (Lost-in-merge regression; restored.)
    _wardrobe_phrases = (
        "use my wardrobe",
        "use wardrobe",
        "using my wardrobe",
        "using wardrobe",
        "show wardrobe matches",
        "build from my wardrobe",
        "from my wardrobe",
        "with my wardrobe",
        "with my clothes",
        "from my closet",
        "in my wardrobe",
        "my wardrobe",
    )
    return action_key in {
        "use_wardrobe",
        "use_my_wardrobe",
        "show_wardrobe_matches",
        "build_from_my_wardrobe",
        "from_my_wardrobe",
        "with_my_clothes",
        "from_my_closet",
    } or any(phrase in q for phrase in _wardrobe_phrases)


def _wardrobe_action_prompt(prompt: str) -> str:
    text = str(prompt or "").strip()
    cleaned = re.sub(
        r"^\s*(use\s+my\s+wardrobe|use\s+wardrobe|show\s+wardrobe\s+matches|build\s+from\s+my\s+wardrobe|from\s+my\s+wardrobe|with\s+my\s+clothes|from\s+my\s+closet)\s*(for|with)?\s*:?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"^\s*(show\s+visual\s+inspiration|visual\s+inspiration|show\s+me\s+visual\s+inspiration)\s*(for)?\s*:?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    base = cleaned or text or "today"
    if re.search(r"\b(use my wardrobe|from my wardrobe|with my clothes|from my closet)\b", base, re.I):
        return base
    return f"Use my wardrobe for {base}"


def _style_default_visual_inspiration_enabled() -> bool:
    return os.getenv("STYLE_DEFAULT_VISUAL_INSPIRATION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _should_default_visual_inspiration(
    query: str,
    *,
    intent: str = "",
    module_context: str = "",
    style_action: str = "",
    multi_event: Optional[Dict[str, Any]] = None,
) -> bool:
    if not _style_default_visual_inspiration_enabled():
        return False

    module = str(module_context or "").strip().lower()
    action_key = _normalize_action_key(style_action)
    if (
        module in {"wardrobe", "closet"}
        or _is_use_wardrobe_action(action=style_action, prompt=query)
        or is_wardrobe_style_request(query, module_context=module)
    ):
        return False

    if action_key in {
        "more_options",
        "more_looks",
        "next_best_options",
        "show_closest_option",
        "closest_option",
        "retry",
        "find_missing_pieces",
        "find_missing_piece",
        "shopping_assist",
    }:
        return False

    resolved_mode = str(intent or "").strip().lower()
    if resolved_mode not in STYLE_MODES:
        resolved_mode = classify_style_mode(
            query,
            module_context=module,
            style_action=style_action,
        )
    if resolved_mode in {
        WARDROBE_STYLE,
        SHOPPING_ASSIST,
        COLOR_BODY_ADVICE,
        STYLE_EDUCATION,
        STYLE_PAIRING,
        "body_proportion_advice",
        "color_advice",
        "occasion_advice",
    }:
        return False

    if multi_event:
        return True
    if resolved_mode in {
        STYLE_ADVICE,
        VISUAL_INSPIRATION,
        "daily_outfit",
        "occasion_outfit",
        "explore_styles",
    }:
        return True

    occasion = _ahvi_style_occasion(query)
    return (
        module in {"style", "daily_wear"}
        and (
            _is_explicit_style_request(query, module)
            or _is_style_priority_query(query)
            or occasion != "today"
        )
    )


def _style_source_policy(cards: Any) -> str:
    """First non-empty source_policy across candidate cards (defaults empty)."""
    for card in cards or []:
        if isinstance(card, dict) and card.get("source_policy"):
            return str(card.get("source_policy"))
    return ""


def _style_final_roles(cards: Any) -> list:
    """Union of explicit roles present across the final cards (deterministic)."""
    try:
        from services.style_explicit_roles import board_explicit_roles

        roles: set = set()
        for card in cards or []:
            if isinstance(card, dict):
                roles.update(board_explicit_roles(card.get("items")))
        from services.style_explicit_roles import EXPLICIT_ROLES

        return [r for r in EXPLICIT_ROLES if r in roles]
    except Exception:  # noqa: BLE001
        return []


def _emit_style_outcome_trace(
    *,
    user_id: Any,
    intent: str,
    occasion: str,
    source_policy: str,
    requested_roles: list,
    required_roles: list,
    final_cards: Any,
    missing_roles: list,
    repair_attempted: bool,
    validation_result: str,
) -> None:
    """One stable structured Style decision trace. No tokens / secrets / full
    profile — only a short stable hash of the user id."""
    try:
        import hashlib

        uid = hashlib.sha256(str(user_id or "").encode("utf-8")).hexdigest()[:12]
        logger.info(
            "AHVI_STYLE_OUTCOME_TRACE user=%s intent=%s occasion=%s source_policy=%s "
            "requested_roles=%s required_roles=%s final_roles=%s missing_roles=%s "
            "repair_attempted=%s fallback_used=%s validation_result=%s",
            uid,
            intent,
            occasion,
            source_policy,
            list(requested_roles or []),
            list(required_roles or []),
            _style_final_roles(final_cards),
            list(missing_roles or []),
            bool(repair_attempted),
            bool(missing_roles),
            validation_result,
        )
    except Exception:  # noqa: BLE001 - tracing must never break the response
        pass


_WARDROBE_ONLY_PHRASES = (
    "use my wardrobe",
    "using my wardrobe",
    "from my wardrobe",
    "only my wardrobe",
    "wardrobe only",
    "wardrobe-only",
    "use only my wardrobe",
    "use what i own",
    "only items i own",
    "from my closet",
    "using my closet",
    "no inspiration",
)


def _canonical_source_policy(query: Any, *, action: Any = None) -> str:
    """Single source of truth for source policy, derived from EXPLICIT user
    language only. "using my wardrobe" -> wardrobe; a use-wardrobe action ->
    wardrobe; otherwise "" (unspecified — never silently forced to wardrobe).

    Different Style paths used to each derive this independently (some stamped
    wardrobe with no request, some left it blank despite "using my wardrobe"),
    so the trace disagreed. Every path now calls this."""
    text = re.sub(r"\s+", " ", str(query or "").strip().lower())
    if any(phrase in text for phrase in _WARDROBE_ONLY_PHRASES):
        return "wardrobe"
    act = str(action or "").strip().lower()
    if act in {"use_wardrobe", "wardrobe", "wardrobe_only"}:
        return "wardrobe"
    return ""


def _style_response_cards(response: Dict[str, Any]) -> list:
    """The card list a style response actually renders (cards first, then the
    style_boards mirror)."""
    for key in ("cards", "style_boards"):
        value = response.get(key)
        if isinstance(value, list) and value:
            return [c for c in value if isinstance(c, dict)]
    return []


_COMPLETE_OUTFIT_CTA_PHRASES = (
    "suggest a complete outfit",
    "create a complete outfit",
    "build a complete outfit",
    "complete outfit for me",
    "complete outfit for today",
    "full look for today",
    "full outfit for today",
    "give me a full look",
    "what should i wear today",
    "what should i wear",
    "style me for today",
    "style me today",
    "dress me for today",
    "outfit for me today",
    "outfit for today",
)


def _is_complete_outfit_cta(text: Any) -> bool:
    """The predefined default CTA(s) are complete-outfit GENERATION requests, not
    generic style advice. Match phrase-first so a bare 'complete outfit' intent
    routes to the board generator + universal completeness gate."""
    t = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not t:
        return False
    if any(p in t for p in _COMPLETE_OUTFIT_CTA_PHRASES):
        return True
    # "suggest / create / build ... a complete outfit ..." (words between verb
    # and the noun phrase).
    if re.search(r"(suggest|create|build|make|give|show).*complete outfit", t):
        return True
    return False


def _is_alternative_look_request(text: Any) -> bool:
    """An "another look" / "shuffle this look" request is a COMPLETE-outfit
    regeneration, not generic stylist advice. Route it to the board generator +
    completeness gate so the alternative is a real outfit, never a
    bottom+footwear+accessory stub."""
    t = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not t:
        return False
    return (
        "another look" in t
        or "more looks" in t
        or "show me another" in t
        or "different look" in t
        or "shuffle this look" in t
        or t.startswith("shuffle")
    )


def _is_generate_style_board_request(text: Any) -> bool:
    """Any prompt that must produce a complete Style board: the default CTA or an
    alternative-look regeneration."""
    return _is_complete_outfit_cta(text) or _is_alternative_look_request(text)


def _pool_gender_ok(item: Dict[str, Any], gender: str) -> bool:
    """Keep opposite-gender assets out of a gendered final board (repair pool),
    unless gender is unknown. Reuses simple token signals; never the full asset
    metadata pipeline."""
    g = str(gender or "").strip().lower()
    if not g or not isinstance(item, dict):
        return True
    blob = " ".join(
        str(item.get(k) or "").lower()
        for k in ("name", "category", "sub_category", "subcategory", "gender", "style_gender", "tags")
    )
    male = g.startswith("m") or g in {"man", "men", "male"}
    female = g.startswith("f") or g in {"woman", "women", "female", "lady", "ladies"}
    if male and any(w in blob for w in ("women", "woman", "female", "ladies", "girl")):
        return False
    if female and any(w in blob for w in ("mens", "men's", " male ", " man ")):
        return False
    return True


def _generic_core_missing(items: Any) -> list:
    """Core slots a board still needs to be a real outfit: footwear plus either a
    dress or (top AND bottom). Accessories never fill a core slot. Uses the
    shared style role vocabulary so heels / pumps / blazers resolve correctly."""
    try:
        from services.style_explicit_roles import board_explicit_roles
    except Exception:  # noqa: BLE001
        return []
    roles = set(board_explicit_roles(items))
    missing: list = []
    if "footwear" not in roles:
        missing.append("footwear")
    if "dress" not in roles:
        for role in ("top", "bottom"):
            if role not in roles:
                missing.append(role)
    return missing


def _enforce_generic_completeness(
    cards: list,
    *,
    wardrobe: Any = None,
    source_policy: str = "",
    gender: str = "",
) -> tuple:
    """Every board must be a real outfit (top+bottom+footwear OR dress+footwear);
    accessories never fill a core slot. Repairs once from sibling-card items +
    wardrobe (sibling first so gender/occasion stay correct), else drops the
    card. Returns (complete_cards, missing_core_slots)."""
    try:
        from services.style_explicit_roles import _item_key, item_explicit_role
        from services.style_explicit_roles import _source_ok as _src_ok
    except Exception:  # noqa: BLE001 - never break the response
        return list(cards or []), []

    sibling_items: list = []
    for card in cards or []:
        if isinstance(card, dict):
            for item in card.get("items") or []:
                if isinstance(item, dict):
                    sibling_items.append(item)
    pool = list(sibling_items)
    if isinstance(wardrobe, list):
        pool.extend(w for w in wardrobe if isinstance(w, dict))
    if gender:
        pool = [p for p in pool if _pool_gender_ok(p, gender)]

    complete: list = []
    missing: set = set()
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        need = _generic_core_missing(card.get("items"))
        if not need:
            complete.append(card)
            continue
        items = [i for i in (card.get("items") or []) if isinstance(i, dict)]
        used = {_item_key(i) for i in items}
        repaired = list(items)
        ok = True
        for role in need:
            picked = None
            for cand in pool:
                if item_explicit_role(cand) != role:
                    continue
                if _item_key(cand) in used:
                    continue
                if not _src_ok(cand, source_policy):
                    continue
                picked = cand
                break
            if picked is None:
                ok = False
                missing.add(role)
                continue
            repaired.append(picked)
            used.add(_item_key(picked))
        if ok and not _generic_core_missing(repaired):
            out = dict(card)
            out["items"] = repaired
            out["item_count"] = len(repaired)
            complete.append(out)
        else:
            for role in _generic_core_missing(card.get("items")):
                missing.add(role)
    return complete, sorted(missing)


def _board_provenance_policy(items: Any) -> str:
    """Source policy from item PROVENANCE (where items came from) for the board
    contract — distinct from the user-language canonical policy. All-wardrobe ->
    wardrobe; all-catalog -> catalog; else mixed."""
    srcs = set()
    for item in items or []:
        if isinstance(item, dict):
            src = str(item.get("source") or item.get("provenance") or item.get("origin") or "").strip().lower()
            if src:
                srcs.add(src)
    if not srcs:
        return "wardrobe"
    if srcs <= {"wardrobe", "owned", "closet", "uploaded"}:
        return "wardrobe"
    if srcs <= {"catalog", "commerce"}:
        return "catalog"
    return "mixed"


# Deterministic normalized flat-lay positions per role so every item satisfies
# the frontend's StyleBoardState.supportsShuffle (which needs a usable x/y/w/h on
# every item, in addition to the board-level contract). Kept close to the FE
# editorial role template so the rendered layout stays sane.
_ROLE_POSITIONS = {
    "top": (0.15, 0.06, 0.60, 0.44),
    "outerwear": (0.05, 0.05, 0.55, 0.55),
    "bottom": (0.45, 0.40, 0.50, 0.50),
    "dress": (0.18, 0.05, 0.60, 0.72),
    "footwear": (0.10, 0.72, 0.44, 0.22),
    "bag": (0.68, 0.55, 0.24, 0.24),
    "accessory": (0.70, 0.30, 0.20, 0.20),
    "unknown": (0.30, 0.35, 0.36, 0.36),
}


def _ensure_item_positions(items: Any) -> None:
    """Give every board item a usable normalized position (x/y/width/height) when
    it lacks one, so the locked Shuffle flow is available. Never overwrites a
    position the pipeline already supplied."""
    try:
        from services.style_explicit_roles import item_explicit_role
    except Exception:  # noqa: BLE001
        item_explicit_role = None  # type: ignore
    for idx, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        pos = item.get("position")
        has_nested = isinstance(pos, dict) and all(
            pos.get(k) is not None for k in ("x", "y", "width", "height")
        )
        has_flat = all(item.get(k) is not None for k in ("x", "y", "width", "height"))
        if has_nested or has_flat:
            continue
        role = ""
        if item_explicit_role is not None:
            try:
                role = item_explicit_role(item)
            except Exception:  # noqa: BLE001
                role = ""
        x, y, w, h = _ROLE_POSITIONS.get(role or "unknown", _ROLE_POSITIONS["unknown"])
        # Nudge duplicates of the same role so two accessories never fully stack.
        offset = (idx % 3) * 0.02
        item["position"] = {
            "x": round(min(0.95 - w, x + offset), 4),
            "y": round(y, 4),
            "width": w,
            "height": h,
            "z": idx,
        }


def _board_signature(board: Any) -> str:
    if not isinstance(board, dict):
        return ""
    items = board.get("items") or []
    return "|".join(
        sorted(
            str(i.get("item_id") or i.get("id") or i.get("$id") or i.get("name") or "")
            for i in items
            if isinstance(i, dict)
        )
    )


def _propagate_board_contract(
    response: Dict[str, Any], *, occasion: str, source_policy: str
) -> None:
    """Stamp the SAME contract (board_id / revision / source_policy / occasion)
    and usable item positions onto EVERY representation of a board that Flutter
    might consume, not just `cards`. All aliases of one board share its board_id.
    """
    cards = [c for c in (response.get("cards") or []) if isinstance(c, dict)]
    canonical: Dict[str, Dict[str, Any]] = {}
    for card in cards:
        _ensure_item_positions(card.get("items"))
        if card.get("board_id"):
            canonical[_board_signature(card)] = {
                "board_id": card.get("board_id"),
                "revision": card.get("revision"),
                "source_policy": card.get("source_policy"),
                "occasion": card.get("occasion"),
            }

    data = response.get("data") if isinstance(response.get("data"), dict) else None
    collections = [
        ("style_boards", response),
        ("rendered_boards", response),
    ]
    if data is not None:
        collections.append(("outfits", data))
        collections.append(("rendered_boards", data))

    for key, holder in collections:
        col = holder.get(key)
        if not isinstance(col, list) or not col:
            continue
        for i, board in enumerate(col):
            if not isinstance(board, dict):
                continue
            contract = canonical.get(_board_signature(board))
            if contract:
                board.update({k: v for k, v in contract.items() if v is not None})
            else:
                _stamp_board_contract(
                    board, occasion=occasion, index=i, source_policy=source_policy
                )
            _ensure_item_positions(board.get("items"))


def _stamp_board_contract(
    card: Dict[str, Any], *, occasion: str, index: int = 0, source_policy: str = ""
) -> Dict[str, Any]:
    """Stamp stable presentation metadata on an unregistered chat board."""
    if not isinstance(card, dict):
        return card
    import hashlib

    items = [i for i in (card.get("items") or []) if isinstance(i, dict)]
    key = "|".join(
        sorted(
            str(i.get("item_id") or i.get("id") or i.get("$id") or i.get("name") or "")
            for i in items
        )
    ) + "|" + str(occasion or "") + "|" + str(index)
    board_id = str(card.get("board_id") or "").strip() or (
        "board_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    )
    try:
        revision = int(card.get("revision"))
    except (TypeError, ValueError):
        revision = 0
    if revision < 1:
        revision = 1
    policy = str(card.get("source_policy") or "").strip() or source_policy or _board_provenance_policy(items)
    card["board_id"] = board_id
    card["revision"] = revision
    card["source_policy"] = policy
    card["occasion"] = card.get("occasion") or occasion or ""
    # Generic chat boards are not persisted by the canonical board-state
    # service. A stable presentation id must never imply mutation capability.
    card["interaction_mode"] = "recommendation"
    card["shuffle_available"] = False
    card["can_shuffle"] = False
    return card


def _stamp_response_contract(
    response: Dict[str, Any], *, occasion: str, source_policy: str
) -> None:
    """Stamp presentation metadata + item positions on the
    response's existing cards and mirror to every alias. Used when the wardrobe
    adapter already enforced roles (meta.style_compliance_gated)."""
    cards = [c for c in _style_response_cards(response) if isinstance(c, dict)]
    if not cards:
        return
    for i, card in enumerate(cards):
        _stamp_board_contract(card, occasion=occasion, index=i, source_policy=source_policy)
        _ensure_item_positions(card.get("items"))
    response["cards"] = cards
    response["style_boards"] = cards
    response["rendered_boards"] = cards
    _data = response.get("data") if isinstance(response.get("data"), dict) else {}
    _data["outfits"] = cards
    _data["rendered_boards"] = cards
    response["data"] = _data
    _propagate_board_contract(response, occasion=occasion, source_policy=source_policy)
    bids = [str(c.get("board_id") or "") for c in cards if c.get("board_id")]
    if bids:
        response["board_ids"] = ",".join(bids)


def _apply_style_compliance_gate(
    response: Dict[str, Any],
    *,
    query: str,
    user_id: Any,
    wardrobe: Any = None,
    action: Any = None,
    default_cta: bool = False,
) -> Dict[str, Any]:
    """Universal explicit-role gate. Runs AFTER every Style path converges on the
    serializer, so premium / editorial / multi-board / visual routes can no
    longer ship boards that ignore explicitly requested garment roles.

    Validates EVERY returned card, repairs once from wardrobe + sibling-card
    items, drops non-compliant cards, and never falls back to the original
    unvalidated list: if nothing survives it returns the typed
    missing_explicit_roles result. Also stamps the canonical source_policy and
    emits exactly one AHVI_STYLE_OUTCOME_TRACE."""
    if not isinstance(response, dict):
        return response
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    # The wardrobe adapter enforces roles and sets style_compliance_gated, but it
    # never stamps board presentation metadata — so stamp it here (idempotent:
    # _stamp_board_contract keeps any board_id already present) before returning,
    # else /api/text adapter boards ship without board_id/revision/source_policy
    # and the app can render a stable recommendation board without Shuffle.
    if meta.get("style_compliance_gated"):
        _stamp_response_contract(
            response,
            occasion=str(meta.get("occasion") or _ahvi_style_occasion(query) or "").strip(),
            source_policy=(
                _canonical_source_policy(query, action=action)
                or str(meta.get("source_policy") or "")
            ),
        )
        return response

    source_policy = _canonical_source_policy(query, action=action)
    cards = _style_response_cards(response)
    if not cards:
        # Non-board style reply (advice, clarification): only fix source policy.
        if source_policy:
            response["source_policy"] = source_policy
            response["meta"] = {**meta, "source_policy": source_policy}
        return response

    try:
        from services.style_flow_service import _enforce_explicit_roles_on_cards
        from services.style_explicit_roles import extract_requested_roles
    except Exception:  # noqa: BLE001 - gate must never break the response
        return response

    occasion = str(meta.get("occasion") or _ahvi_style_occasion(query) or "").strip()
    requested = list(extract_requested_roles(query))
    enforcement: Dict[str, Any] = {}
    pool = wardrobe if isinstance(wardrobe, list) else None
    kept = _enforce_explicit_roles_on_cards(
        cards,
        query=query,
        occasion=occasion,
        candidate_pool=pool,
        enforcement=enforcement,
    )

    if requested and not kept and enforcement.get("status") == "missing_explicit_roles":
        missing = list(enforcement.get("missing_roles") or [])
        available = list(enforcement.get("available_roles") or [])
        _emit_style_outcome_trace(
            user_id=user_id,
            intent=str(meta.get("intent") or "style"),
            occasion=occasion,
            source_policy=source_policy or "wardrobe",
            requested_roles=requested,
            required_roles=list(enforcement.get("required_explicit_roles") or requested),
            final_cards=[],
            missing_roles=missing,
            repair_attempted=bool(enforcement.get("repair_attempted")),
            validation_result="missing_explicit_roles",
        )
        _missing_label = ", ".join(missing or requested)
        return {
            "success": False,
            "reason": "missing_explicit_roles",
            "message": (
                "I couldn't complete this exact "
                + (source_policy or "wardrobe")
                + "-only look because your wardrobe doesn't currently contain a "
                + "suitable " + _missing_label + ". I can create a catalog-inspired "
                + "option or suggest the missing piece."
            ),
            "board": "style",
            "type": "missing_explicit_roles",
            "cards": [],
            "style_boards": [],
            "board_ids": "",
            "source_policy": source_policy or "wardrobe",
            "can_offer_catalog_option": True,
            "requested_roles": requested,
            "missing_roles": missing,
            "available_roles": available,
            "repair_attempted": bool(enforcement.get("repair_attempted")),
            "data": {"outfits": [], "rendered_boards": [], "board_item_ids": []},
            "meta": {
                **meta,
                "status": "missing_explicit_roles",
                "reason": "missing_explicit_roles",
                "requested_roles": requested,
                "missing_roles": missing,
                "available_roles": available,
                "repair_attempted": bool(enforcement.get("repair_attempted")),
                "source_policy": source_policy or "wardrobe",
                "can_offer_catalog_option": True,
                "style_compliance_gated": True,
            },
        }

    # Universal generic completeness: with or without explicit roles, every board
    # must be a real outfit (top+bottom+footwear OR dress+footwear). Accessories
    # never fill a core slot. Catches truncation-fallback / premium / visual
    # boards that shipped e.g. bottom+footwear+accessory.
    base_cards = kept if kept else cards
    # Generic completeness is enforced for complete-outfit CTA prompts only, so
    # advice / visual-inspiration / wardrobe-action stubs keep their behaviour.
    is_cta = bool(default_cta) or _is_generate_style_board_request(query)
    if is_cta:
        complete_cards, generic_missing = _enforce_generic_completeness(
            base_cards,
            wardrobe=pool,
            source_policy=source_policy,
            gender=str(meta.get("style_gender") or "").strip().lower(),
        )
    else:
        complete_cards, generic_missing = base_cards, []
    # Default CTA stays compact: at most 2 boards.
    if (default_cta or bool(meta.get("default_cta"))) and len(complete_cards) > 2:
        complete_cards = complete_cards[:2]

    if is_cta and not complete_cards:
        # Nothing is a complete outfit -> typed failure, never ship an incomplete
        # board labelled as a success, and never restore the original cards.
        _emit_style_outcome_trace(
            user_id=user_id,
            intent=str(meta.get("intent") or "style"),
            occasion=occasion,
            source_policy=source_policy,
            requested_roles=requested,
            required_roles=requested,
            final_cards=[],
            missing_roles=generic_missing or ["top", "bottom", "footwear"],
            repair_attempted=True,
            validation_result="no_complete_outfit",
        )
        return {
            "success": False,
            "reason": "no_complete_outfit",
            "message": (
                "I couldn't build a complete outfit yet — I need at least a top, "
                "a bottom and footwear (or a dress and footwear) to work with."
            ),
            "board": "style",
            "type": "no_complete_outfit",
            "cards": [],
            "style_boards": [],
            "board_ids": "",
            "source_policy": source_policy or "",
            "can_offer_catalog_option": True,
            "missing_roles": generic_missing or ["top", "bottom", "footwear"],
            "repair_attempted": True,
            "data": {"outfits": [], "rendered_boards": [], "board_item_ids": []},
            "meta": {
                **meta,
                "status": "no_complete_outfit",
                "reason": "no_complete_outfit",
                "missing_roles": generic_missing or ["top", "bottom", "footwear"],
                "source_policy": source_policy or "",
                "style_compliance_gated": True,
            },
        }

    # Stamp stable presentation metadata on every generic chat card. These cards
    # are not registered board state, so Shuffle remains explicitly unavailable.
    contract_policy = source_policy or str(meta.get("source_policy") or "")
    for i, card in enumerate(complete_cards):
        _stamp_board_contract(card, occasion=occasion, index=i, source_policy=contract_policy)
        _ensure_item_positions(card.get("items"))

    response["cards"] = complete_cards
    # Mirror the validated + stamped boards into every alias Flutter reads, so the
    # contract reaches style_boards / rendered_boards / data.* and not just cards.
    response["style_boards"] = complete_cards
    response["rendered_boards"] = complete_cards
    _data = response.get("data")
    if not isinstance(_data, dict):
        _data = {}
    _data["outfits"] = complete_cards
    _data["rendered_boards"] = complete_cards
    response["data"] = _data
    # Belt-and-suspenders: re-stamp any board object still present in an alias
    # that was not replaced above (keeps all representations consistent).
    _propagate_board_contract(response, occasion=occasion, source_policy=contract_policy)
    board_ids = [str(c.get("board_id") or "") for c in complete_cards if c.get("board_id")]
    if board_ids:
        response["board_ids"] = ",".join(board_ids)
    if source_policy:
        response["source_policy"] = source_policy
    response["meta"] = {
        **meta,
        "source_policy": source_policy or meta.get("source_policy") or "",
        "requested_roles": requested,
        "board_count": len(complete_cards),
        "board_ids": board_ids,
        "style_compliance_gated": True,
    }
    _emit_style_outcome_trace(
        user_id=user_id,
        intent=str(meta.get("intent") or "style"),
        occasion=occasion,
        source_policy=source_policy,
        requested_roles=requested,
        required_roles=requested,
        final_cards=complete_cards,
        missing_roles=[],
        repair_attempted=bool(enforcement.get("repair_attempted")),
        validation_result="satisfied",
    )
    return response


def _style_curation_brief(query_text: str, occasion: str) -> Dict[str, Any]:
    """Compact stylist brief (goal/impression/atmosphere/confidence_strategy)
    derived deterministically from the occasion. Feeds board curation without
    an extra Gemini reasoning round-trip."""
    try:
        from services import style_reasoning_engine as _sre

        category, _tone, _formality, _occ = _sre._occasion_category(query_text)
        return {
            "goal": _sre._fallback_goal("style_advice", category),
            "impression": _sre._fallback_impression(category),
            "atmosphere": _sre._fallback_atmosphere(category),
            "confidence_strategy": (
                "Lean into what already fits well and keep one deliberate "
                "detail — confidence reads as ease, not effort."
            ),
            "what_to_avoid": [],
            "occasion": occasion,
        }
    except Exception:  # noqa: BLE001
        return {"occasion": occasion}


def _demo_style_board_payload(
    user_id,
    query_text,
    request_wardrobe,
    user_profile=None,
    resolved_occasion: str = "",
    style_action: str = "",
    show_closest_option: bool = False,
    allow_closest_option: bool = False,
    closest: bool = False,
):
    closest_requested = (
        str(style_action or "").strip().lower() == "show_closest_option"
        or bool(show_closest_option)
        or bool(allow_closest_option)
        or bool(closest)
    )
    if closest_requested:
        style_action = "show_closest_option"
        show_closest_option = True
        allow_closest_option = True
        closest = True
    profile = _ahvi_resolve_effective_user_profile(user_id, user_profile or {})
    wardrobe = _fetch_wardrobe_for_style(user_id, request_wardrobe)
    wardrobe = [
        item
        for item in wardrobe
        if _ahvi_item_allowed_for_user_profile(item, profile, query_text)
    ]

    occasion = str(resolved_occasion or "").strip() or _ahvi_style_occasion(query_text)
    logger.info("style.intent.detected user_id=%s occasion=%s prompt=%r", user_id, occasion, query_text)
    if any(token in str(query_text or "").lower() for token in ("meeting", "client", "presentation", "interview")):
        logger.info("style.sub_intent.detected user_id=%s sub_intent=office_meeting prompt=%r", user_id, query_text)
    logger.info("style.wardrobe.loaded user_id=%s count=%s", user_id, len(wardrobe))

    try:
        response = build_style_flow_response(
            user_id=user_id,
            query=query_text,
            wardrobe=wardrobe,
            user_profile=profile,
            context={
                "occasion": occasion,
                "query": query_text,
                "user_profile": profile,
                "style_gender": _ahvi_profile_style_gender(profile),
                # AHVI Style Orchestrator agent inputs (best-effort pass-through).
                "chips": locals().get("chips") or [],
                "weather": (
                    (profile or {}).get("weather")
                    if isinstance(profile, dict)
                    else {}
                ) or {},
                "signals": {
                    "source": "routers.chat.style_flow_service_fallback",
                    "style_gender": _ahvi_profile_style_gender(profile),
                },
            },
            include_base64=False,
            # Editorial board PNG render + R2 upload is gated by an env flag so
            # it never silently adds latency. AHVI_STYLE_BOARD_RENDER=1 turns on
            # image_url generation for wardrobe boards.
            upload_to_r2=str(os.getenv("AHVI_STYLE_BOARD_RENDER", "")).strip().lower()
            in {"1", "true", "yes", "on"},
            style_action=style_action,
            show_closest_option=show_closest_option,
            allow_closest_option=allow_closest_option,
            closest=closest,
            cache_bypass=True,
        )
    except Exception as exc:
        logger.warning(
            "ahvi.chat_style_flow_fallback_failed user_id=%s error=%s",
            user_id,
            str(exc)[:180],
        )
        logger.exception("style.error user_id=%s prompt=%r", user_id, query_text)
        logger.info("style.fallback.triggered user_id=%s reason=style_flow_exception wardrobe_count=%s", user_id, len(wardrobe))
        return {
            "success": False,
            "message": (
                "I couldn't build a reliable style board from your wardrobe right now. "
                "Please try again."
            ),
            "board": "style",
            "type": "missing_outfit_cards",
            "cards": [],
            "style_boards": [],
            "board_ids": "",
            "data": {
                "outfits": [],
                "rendered_boards": [],
                "board_item_ids": [],
            },
            "meta": {
                "mode": "style_flow_service_fallback_failed",
                "fallback_used": False,
                "error": "style_flow_service_failed",
                "error_stage": "routers.chat",
                "occasion": occasion,
                "wardrobe_count": len(wardrobe),
            },
        }

    cards = response.get("cards") if isinstance(response.get("cards"), list) else []
    response_type = str(response.get("type") or "").strip()
    if (
        not cards
        and not response_type
        and not _ahvi_router_style_fallback_enabled()
    ):
        logger.info("style.fallback.triggered user_id=%s reason=no_cards wardrobe_count=%s", user_id, len(wardrobe))
        response["success"] = False
        response["message"] = (
            "I couldn't build a reliable style board from your wardrobe right now. "
            "Please try again."
        )
        response["type"] = "missing_outfit_cards"

    try:
        logger.info(
            "ahvi.chat_style_flow_service user_id=%s occasion=%s wardrobe=%s cards=%s first_card_items=%s",
            user_id,
            occasion,
            len(wardrobe),
            len(cards),
            _ahvi_style_names((cards[0].get("items") if cards else []) or [])[:8],
        )
    except Exception:
        pass
    if cards:
        logger.info("style.board.generated user_id=%s occasion=%s cards=%s", user_id, occasion, len(cards))

    # Gemini-assisted curation: rank + re-title the deterministic candidate
    # cards, enforce diversity, attach stylist metadata. Never invents items.
    if cards and len(cards) >= 1:
        try:
            _brief = _style_curation_brief(query_text, occasion)
            _enforcement: dict = {}
            curated = curate_wardrobe_boards(
                cards,
                query=query_text,
                occasion=occasion,
                reasoning=_brief,
                wardrobe_count=len(wardrobe),
                target=4,
                candidate_pool=wardrobe if isinstance(wardrobe, list) else None,
                enforcement=_enforcement,
            )
            _requested = list(_enforcement.get("requested_roles") or [])
            if curated:
                response["cards"] = curated
                if isinstance(response.get("style_boards"), list) and response.get("style_boards"):
                    response["style_boards"] = curated
                cards = curated
                _emit_style_outcome_trace(
                    user_id=user_id,
                    intent="style_pipeline_adapter",
                    occasion=occasion,
                    source_policy=_canonical_source_policy(query_text),
                    requested_roles=_requested,
                    required_roles=list(_enforcement.get("required_explicit_roles") or _requested),
                    final_cards=curated,
                    missing_roles=[],
                    repair_attempted=bool(_enforcement.get("repair_attempted")),
                    validation_result="satisfied",
                )
            elif _enforcement.get("status") == "missing_explicit_roles":
                # The user named roles we cannot satisfy from their wardrobe.
                # Return a typed gap instead of falling back to the pre-curation
                # cards, which would ship a board missing what they asked for.
                missing = list(_enforcement.get("missing_roles") or [])
                requested = _requested
                available = list(_enforcement.get("available_roles") or [])
                source_policy = _canonical_source_policy(query_text) or "wardrobe"
                _emit_style_outcome_trace(
                    user_id=user_id,
                    intent="style_pipeline_adapter",
                    occasion=occasion,
                    source_policy=source_policy,
                    requested_roles=requested,
                    required_roles=list(_enforcement.get("required_explicit_roles") or requested),
                    final_cards=[],
                    missing_roles=missing,
                    repair_attempted=bool(_enforcement.get("repair_attempted")),
                    validation_result="missing_explicit_roles",
                )
                _missing_label = ", ".join(missing or requested)
                return {
                    "success": False,
                    "reason": "missing_explicit_roles",
                    "message": (
                        "I couldn't complete this exact "
                        + (source_policy or "wardrobe")
                        + "-only look because your wardrobe doesn't currently "
                        + "contain a suitable " + _missing_label + ". I can create "
                        + "a catalog-inspired option or suggest the missing piece."
                    ),
                    "board": "style",
                    "type": "missing_explicit_roles",
                    "cards": [],
                    "style_boards": [],
                    "board_ids": "",
                    "source_policy": source_policy,
                    "can_offer_catalog_option": True,
                    "requested_roles": requested,
                    "missing_roles": missing,
                    "available_roles": available,
                    "repair_attempted": bool(_enforcement.get("repair_attempted")),
                    "data": {"outfits": [], "rendered_boards": [], "board_item_ids": []},
                    "meta": {
                        "mode": "style_flow_service_adapter_v1",
                        "intent": "style_pipeline_adapter",
                        "domain": "style",
                        "occasion": occasion,
                        "wardrobe_count": len(wardrobe),
                        "status": "missing_explicit_roles",
                        "reason": "missing_explicit_roles",
                        "missing_roles": missing,
                        "requested_roles": requested,
                        "required_explicit_roles": list(
                            _enforcement.get("required_explicit_roles") or requested
                        ),
                        "available_roles": available,
                        "repair_attempted": bool(_enforcement.get("repair_attempted")),
                        "source_policy": source_policy,
                        "can_offer_catalog_option": True,
                    },
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("ahvi.board_curation_failed user_id=%s err=%s", user_id, str(exc)[:160])

    response["meta"] = {
        **(response.get("meta") if isinstance(response.get("meta"), dict) else {}),
        "intent": "style_pipeline_adapter",
        "domain": "style",
        "mode": "style_flow_service_adapter_v1",
        "wardrobe_count": len(wardrobe),
        "style_gender": _ahvi_profile_style_gender(profile),
        "occasion": occasion,
        "style_action": style_action or None,
        "source_policy": _canonical_source_policy(query_text, action=style_action),
        "style_compliance_gated": True,
    }
    return response

# ================= AHVI CLEAN CHAT STYLE V2 END =================


def _is_explicit_style_request(text: str, module_context: str | None = None) -> bool:
    """
    True only when user is clearly asking AHVI to build/style an outfit board.

    Important:
    Generic words like "style" or "personal styling" alone must not force the
    wardrobe pipeline. Otherwise normal chat gets misrouted to daily_outfit.
    """
    q = str(text or "").lower().strip()
    module = str(module_context or "").lower().strip()

    if any(
        k in q
        for k in [
            "more looks",
            "more look",
            "next best",
            "next option",
            "next options",
            "other options",
            "other looks",
            "show more",
            "different shoes",
            "different shoe",
            "different footwear",
            "try different shoes",
        ]
    ):
        return True

    if module in {"style", "wardrobe"} and any(
        k in q
        for k in [
            "wear",
            "outfit",
            "look",
            "style me",
            "style this",
            "what should i wear",
            "date night",
            "office outfit",
            "party outfit",
            "build a board",
            "style board",
        ]
    ):
        return True

    occasion_setup_words = [
        "date",
        "dinner",
        "party",
        "club",
        "rave",
        "office",
        "meeting",
        "wedding",
        "reception",
        "event",
        "travel",
        "trip",
        "brunch",
    ]
    setup_phrases = [
        "i have",
        "ive got",
        "i've got",
        "coming up",
        "tonight",
        "tomorrow",
        "this weekend",
        "next week",
        "need to go",
        "going to",
    ]
    if any(o in q for o in occasion_setup_words) and any(
        p in q for p in setup_phrases
    ):
        return True

    if any(
        p in q
        for p in [
            "rave party",
            "club party",
            "birthday party",
            "house party",
            "cocktail party",
            "office party",
            "wedding reception",
        ]
    ):
        return True

    explicit_phrases = [
        "what should i wear",
        "what to wear",
        "what do i wear",
        "help me choose an outfit",
        "choose an outfit",
        "suggest an outfit",
        "show outfits",
        "show me outfits",
        "build an outfit",
        "create an outfit",
        "make an outfit",
        "style me",
        "style this",
        "style my",
        "style board",
        "date night outfit",
        "office outfit",
        "party outfit",
        "travel outfit",
        "airport outfit",
        "beach outfit",
        "beach wear",
        "beachwear",
        "wedding outfit",
        "brunch outfit",
        "dinner outfit",
        "gym outfit",
        "workout outfit",
    ]
    if any(p in q for p in explicit_phrases):
        return True

    # Occasion chip support: "date night" alone should create boards.
    occasion_only = {
        "date night",
        "office",
        "party",
        "travel",
        "airport",
        "beach",
        "wedding",
        "brunch",
        "dinner",
        "night out",
    }
    if q in occasion_only:
        return True

    # "I have a date tonight..." is outfit intent only when paired with wear/outfit/look.
    occasion_words = ["date", "dinner", "party", "office", "meeting", "wedding", "travel", "brunch"]
    wardrobe_words = ["wear", "outfit", "look", "clothes", "dress up", "style board"]
    if any(o in q for o in occasion_words) and any(w in q for w in wardrobe_words):
        return True

    return False


def _is_general_chat_request(text: str, module_context: str | None = None) -> bool:
    q = str(text or "").lower().strip()
    module = str(module_context or "").lower().strip()

    if module not in {"", "chat", "general", "home", "assistant", "style", "wardrobe"}:
        return False

    if _is_explicit_style_request(q, module_context):
        return False

    # Anything instructional/question-like that is not explicitly outfit-building
    # should go to the LLM.
    if q in {"hi", "hello", "hey", "chat", "talk", "talk to me"}:
        return True

    general_markers = [
        "reply with",
        "say ",
        "explain",
        "why ",
        "what is",
        "who is",
        "how are",
        "tell me",
        "can you",
        "do you",
        "help me understand",
        "summarize",
        "write",
        "draft",
        "rephrase",
        "just chat",
        "not outfit",
        "do not mention outfits",
        "ai styling should feel personal",
    ]
    if any(k in q for k in general_markers):
        return True

    # If the user mentions fashion/styling conceptually but does not ask for a board,
    # keep it as conversational LLM.
    conceptual_style_markers = [
        "why ai styling",
        "personal styling",
        "fashion advice",
        "style advice",
        "styling should",
        "why style",
    ]
    if any(k in q for k in conceptual_style_markers):
        return True

    return False


def _build_llm_messages(messages: List["Message"], english_input: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in messages[-10:]:
        role = str(getattr(msg, "role", "user") or "user").lower()
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    if not out or out[-1].get("content") != english_input:
        out.append({"role": "user", "content": english_input})
    return out


def _llm_chat_response(
    *,
    messages: List["Message"],
    english_input: str,
    user_id: str,
    user_profile: Dict[str, Any],
    user_message_style: Dict[str, str],
    module_context: str | None = None,
) -> Dict[str, Any]:
    """
    General chat path. This is the path that reaches Gemini/Vertex.
    It must not call outfit pipeline.
    """
    system_instruction = (
        "You are AHVI, a warm premium AI companion. "
        "For normal chat, answer directly and naturally. "
        "Do not create outfit boards unless the user explicitly asks what to wear or asks for an outfit. "
        "Keep replies concise, fresh, and helpful."
    )

    try:
        message = chat_completion(
            _build_llm_messages(messages, english_input),
            system_instruction=system_instruction,
            user_profile=user_profile,
            signals={
                "context_mode": module_context or "chat",
                "user_message_style": user_message_style,
            },
            timeout_seconds=45,
            options={"temperature": 0.65, "max_output_tokens": 320},
            usecase="general_chat",
        )
        mode = "llm_chat"
    except Exception as exc:
        logger.warning("chat.llm_response_failed user_id=%s error=%s", user_id, str(exc)[:180])
        message = lightweight_chat(english_input)
        mode = "llm_chat_fallback"

    try:
        message = tone_engine.apply(
            str(message or "").strip() or lightweight_chat(english_input),
            user_profile=user_profile,
            signals={
                "context_mode": module_context or "chat",
                "user_message_style": user_message_style,
            },
            context={"module_context": module_context},
        )
    except Exception:
        pass

    logger.info(
        "chat.llm_response user_id=%s mode=%s provider=%s",
        user_id,
        mode,
        os.getenv("AI_PROVIDER", ""),
    )

    return {
        "success": True,
        "message": message,
        "board": "general",
        "type": "text",
        "cards": [],
        "board_ids": "",
        "data": {},
        "meta": {
            "mode": mode,
            "intent": "general_chat",
            "provider": os.getenv("AI_PROVIDER", ""),
        },
        "audio_job_id": "offline",
    }


def _detect_mode(text: str, module_context: str | None = None) -> str:
    if _is_general_chat_request(text, module_context):
        return "casual"
    if _is_explicit_style_request(text, module_context):
        return "fashion"

    t = text.lower().strip()
    if t in ["hi", "hello", "hey"]:
        return "greeting"

    # Default must be casual/LLM, not fashion. Otherwise almost every ambiguous
    # prompt falls into wardrobe boards and returns static style fallback.
    return "casual"


def _infer_user_message_style(text: str) -> Dict[str, str]:
    raw = str(text or "")
    lowered = raw.lower()
    length = len(raw.strip())

    emoji_count = sum(1 for ch in raw if ord(ch) > 10000)
    if emoji_count >= 3:
        emoji_density = "high"
    elif emoji_count == 2:
        emoji_density = "medium"
    elif emoji_count == 1:
        emoji_density = "low"
    else:
        emoji_density = "none"

    slang_tokens = ["lowkey", "highkey", "vibe", "it's giving", "main character", "mid"]
    slang_hits = sum(1 for token in slang_tokens if token in lowered)
    if slang_hits >= 3:
        slang_presence = "high"
    elif slang_hits == 2:
        slang_presence = "medium"
    elif slang_hits == 1:
        slang_presence = "low"
    else:
        slang_presence = "none"

    if length <= 80:
        length_bucket = "short"
    elif length <= 220:
        length_bucket = "medium"
    else:
        length_bucket = "long"

    return {
        "message_length_bucket": length_bucket,
        "emoji_density": emoji_density,
        "slang_presence": slang_presence,
    }


# -------------------------
# MODELS
# -------------------------
class Message(BaseModel):
    role: str = Field(..., min_length=1, max_length=24)
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        role = str(value or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be one of user/assistant/system")
        return role


class TextChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, max_length=30)
    language: str = Field(default="en", min_length=2, max_length=8)
    current_memory: Any = Field(default_factory=dict)
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    userID: str | None = None
    module_context: str | None = None
    include_base64: bool = False
    wardrobe: Any = None
    style_action: str | None = None
    show_closest_option: bool = False
    allow_closest_option: bool = False
    closest: bool = False
    exclude_style_signatures: List[str] = Field(default_factory=list)
    requested_board_count: int | None = None
    # Style-session context handoff. Frontend should attach these on
    # chip / button / retry presses so the backend never sees a bare
    # label ("Next best options", "Try again", "Casual beach walk")
    # without the originating prompt.
    action: str | None = Field(default=None, max_length=64)
    clarification: str | None = Field(default=None, max_length=120)
    session_id: str | None = Field(default=None, max_length=80)
    previous_prompt: str | None = Field(default=None, max_length=600)
    resolved_prompt: str | None = Field(default=None, max_length=600)
    current_look_id: str | None = Field(default=None, max_length=80)
    context: Dict[str, Any] = Field(default_factory=dict)
    style_context: Dict[str, Any] = Field(default_factory=dict)
    # Compact request-carried board context for beta follow-ups. Optional and
    # additive: missing state preserves the pre-bridge behavior exactly.
    style_state: Dict[str, Any] = Field(default_factory=dict)
    weather: Any = None
    weather_context: Any = None
    weatherData: Any = None
    location: Any = None
    coordinates: Any = None
    latitude: float | None = None
    longitude: float | None = None
    lat: float | None = None
    lon: float | None = None
    lng: float | None = None


def _beta_style_response(
    response: Dict[str, Any],
    *,
    previous_state: Dict[str, Any],
    instructions: Dict[str, Any],
    query: str = "",
    wardrobe: Any = None,
    user_id: Any = "",
    action: Any = None,
    default_cta: bool = False,
) -> Dict[str, Any]:
    """Add beta fields without changing any established response field.

    This is the convergence point for every Style path returned from /api/text,
    so the explicit-role compliance gate runs here — closing the premium /
    editorial / multi-board routes that previously bypassed validation."""
    response_type = str(response.get("type") or "").strip().lower()
    has_board_response = bool(_style_response_cards(response)) and response_type not in {
        "style_explanation",
        "stylist_advice",
        "text",
        "clarification",
        "style_clarification",
    }
    if query or has_board_response:
        if has_board_response:
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            response["meta"] = {
                **meta,
                "occasion": meta.get("occasion") or instructions.get("occasion") or "",
                "source_policy": meta.get("source_policy") or instructions.get("source_mode") or "",
            }
        response = _apply_style_compliance_gate(
            response,
            query=query,
            user_id=user_id,
            wardrobe=wardrobe,
            action=action,
            default_cta=default_cta or _is_generate_style_board_request(query),
        )
    image_base64 = ""
    data = response.get("data") if isinstance(response.get("data"), dict) else {}
    rendered = data.get("rendered_boards") if isinstance(data.get("rendered_boards"), list) else []
    if rendered and isinstance(rendered[0], dict):
        image_base64 = str(
            rendered[0].get("image_base64")
            or rendered[0].get("board_base64")
            or ""
        ).strip()
    # Build the state first so the vision wrapper receives only real board IDs.
    enriched = decorate_beta_style_response(
        response,
        previous_state=previous_state,
        instructions=instructions,
    )
    visual = beta_visual_intelligence(
        state=enriched["style_state"],
        image_base64=image_base64,
        visual_items=beta_visual_items_from_response(response),
        requested=bool(instructions.get("requires_visual_analysis")),
    )
    if visual:
        enriched = decorate_beta_style_response(
            response,
            previous_state=previous_state,
            instructions=instructions,
            visual=visual,
        )
    return enriched


class OutfitFeedbackRequest(BaseModel):
    user_id: str
    feedback: str
    outfit: Dict[str, Any]


class OrganizeHubRequest(BaseModel):
    user_id: str
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)
    include_counts: bool = False


class PlanPackRequest(BaseModel):
    user_id: str
    prompt: str
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)


class DailyCardsRequest(BaseModel):
    user_id: str
    time_slot: str | None = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)


class ModuleChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=20)
    module: str | None = Field(default=None, min_length=2, max_length=32)
    domain: str | None = Field(default=None, min_length=2, max_length=32)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    weather: Any = None
    weather_context: Any = None
    weatherData: Any = None
    location: Any = None
    coordinates: Any = None
    latitude: float | None = None
    longitude: float | None = None
    lat: float | None = None
    lon: float | None = None
    lng: float | None = None


_MODULE_CHAT_PROMPTS: Dict[str, str] = {
    "medi": (
        "You are AHVI's medicine tracking assistant. Use only the user's "
        "medication inventory and reminder data. Do not give diagnosis or unsafe "
        "dosage advice. For medical uncertainty, advise consulting a doctor."
    ),
    "skincare": (
        "You are AHVI's skincare assistant. Use the user's skin profile and "
        "routine. Be practical, conservative, and avoid diagnosis."
    ),
    "meal": (
        "You are AHVI's meal planning assistant. Use the user's diet, allergies, "
        "goal, calories, and logged meals. Never suggest ingredients listed in allergies."
    ),
    "diet": (
        "You are AHVI's meal planning assistant. Use the user's diet, allergies, "
        "goal, calories, and logged meals. Never suggest ingredients listed in allergies."
    ),
    "calendar": (
        "You are AHVI's planning assistant. Use today's events, tasks, deadlines, "
        "and time context. Prioritize what reduces friction for the user."
    ),
    "bills": (
        "You are AHVI's bills and expense assistant. Use only the provided bill, "
        "transaction, budget, and due-date context. Do not provide investment advice."
    ),
    "fitness": (
        "You are AHVI's fitness preparation assistant. Use the user's workout, "
        "equipment, time, and constraint context. Keep guidance safe, practical, and concise."
    ),
    "daily_wear": (
        "You are AHVI's daily wear assistant. Use the provided style context, but "
        "do not invent wardrobe items. If the user asks for boards, route them to style."
    ),
}


def _normalize_module_name(module: str) -> str:
    value = str(module or "").strip().lower().replace("-", "_")
    allowed = {
        "style",
        "wardrobe",
        "daily_wear",
        "skincare",
        "medi",
        "bills",
        "calendar",
        "meal",
        "diet",
        "fitness",
        "planner",
    }
    if value in {"plan", "planning", "reminder", "reminders"}:
        return "planner"
    return value if value in allowed else "chat"


def _looks_incomplete_module_answer(answer: Any) -> bool:
    text = str(answer or "").strip()
    if not text:
        return True
    if len(text) < 80 and not re.search(r"[.!?)]$", text):
        return True
    tail = text.lower().rstrip(" .,:;")
    incomplete_endings = (
        "about",
        "because",
        "including",
        "such as",
        "based on",
        "depending on",
        "i need",
        "i need a bit more detail about",
    )
    return any(tail.endswith(ending) for ending in incomplete_endings)


def _module_fallback_answer(module_key: str, user_message: str) -> str:
    message = str(user_message or "").lower()
    if module_key == "skincare":
        if any(token in message for token in ("spf", "sunscreen", "sun screen", "sunblock")):
            return (
                "For SPF, choose a broad-spectrum SPF 30 or higher and apply it generously every morning. "
                "Match it to your skin type: if your skin is oily or acne-prone, try a lightweight non-comedogenic gel or fluid. "
                "If your skin is dry or sensitive, look for a fragrance-free moisturizing sunscreen. Reapply "
                "every 2-3 hours outdoors, especially with longer sun exposure."
            )
        if "dry" in message:
            return (
                "For dry skin, keep the routine simple: gentle cleanser, hydrating serum if you use one, "
                "moisturizer, and SPF in the morning. At night, cleanse and use a richer moisturizer. "
                "Avoid harsh scrubs if your skin feels tight or irritated."
            )
        if "oily" in message:
            return (
                "For oily skin, use a gentle cleanser, lightweight moisturizer, and non-comedogenic SPF. "
                "A niacinamide serum can be a reasonable option if your skin tolerates it. Avoid stripping "
                "the skin, because that can make oiliness feel worse."
            )
        if "acne" in message or "pimple" in message or "breakout" in message:
            return (
                "For acne-prone skin, keep it gentle: mild cleanser, lightweight moisturizer, and SPF. "
                "Introduce actives slowly and avoid layering too many strong products at once. For painful, "
                "persistent, or worsening acne, check with a dermatologist."
            )
        if "night" in message:
            return (
                "A simple night routine: cleanse, apply a gentle hydrating step if you use one, then moisturize. "
                "If you use treatments, add only one at a time and watch how your skin reacts."
            )
        return (
            "I can help with a simple skincare routine. A safe baseline is gentle cleanser, moisturizer, "
            "and broad-spectrum SPF in the morning, then cleanser and moisturizer at night. Tell me your "
            "skin type or concern for a more tailored routine."
        )
    if module_key in {"diet", "meal"}:
        if "protein" in message and "breakfast" in message:
            return (
                "For a high-protein breakfast, try Greek yogurt with fruit and nuts, eggs with whole-grain toast, "
                "paneer or tofu scramble, or oats with milk and seeds. Keep portions aligned with your appetite "
                "and dietary preferences."
            )
        if "pre" in message and "workout" in message:
            return (
                "A simple pre-workout meal is easy-to-digest carbs plus a little protein, such as banana with "
                "yogurt, toast with peanut butter, or oats. Give yourself enough time to digest before training."
            )
        if "post" in message and "workout" in message:
            return (
                "Post-workout, aim for protein plus carbs and fluids. Examples: rice with dal or chicken, eggs "
                "and toast, yogurt with fruit, or tofu with a grain bowl."
            )
        if "hydrat" in message or "water" in message:
            return (
                "For hydration, sip water through the day and add electrolytes when you sweat heavily or train "
                "in heat. Use thirst, urine color, and workout intensity as practical cues."
            )
        if "weight loss" in message or "lose weight" in message:
            return (
                "For weight-loss meal ideas, build plates around lean protein, vegetables or fruit, high-fiber "
                "carbs, and a small amount of healthy fat. Keep it sustainable and avoid extreme restriction."
            )
        return (
            "I can help with meal ideas. A balanced option is protein, fiber-rich carbs, vegetables or fruit, "
            "and enough fluids. Tell me your goal, dietary preference, or meal timing for a sharper suggestion."
        )
    if module_key in {"planner", "calendar"}:
        return (
            "I can help structure this plan. Reminder sync is coming next, so for now I will not mark reminders "
            "as saved. Share the tasks, deadline, and rough priority, and I will organize the day."
        )
    return lightweight_chat(user_message)


def _module_response_envelope(
    module_key: str,
    answer: str,
    chips: List[Any] | None = None,
) -> Dict[str, Any]:
    answer_text = str(answer or "").strip()
    actions = chips or []
    return {
        "success": True,
        "type": "module_chat",
        "module": module_key,
        "domain": module_key,
        "intent": module_key,
        "response": answer_text,
        "message_text": answer_text,
        "message": {"role": "assistant", "content": answer_text},
        "cards": [],
        "style_boards": [],
        "chips": actions,
        "quick_actions": actions,
        "data": {
            "module": module_key,
            "intent": module_key,
            "message": answer_text,
            "rendered_boards": [],
            "outfits": [],
        },
        "meta": {
            "mode": module_key,
            "board_count": 0,
        },
    }


def _module_style_response_envelope(
    module_key: str,
    style_payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload = style_payload if isinstance(style_payload, dict) else {}
    raw_message = payload.get("message") or payload.get("message_text") or payload.get("response")
    if isinstance(raw_message, dict):
        answer_text = str(raw_message.get("content") or "").strip()
    else:
        answer_text = str(raw_message or "").strip()
    if not answer_text:
        answer_text = (
            "I can help style that. Add a little context like weather, timing, or dress code "
            "and I will build the look."
        )

    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    cards = payload.get("cards") if isinstance(payload.get("cards"), list) else []
    style_boards = (
        payload.get("style_boards")
        if isinstance(payload.get("style_boards"), list)
        else cards
    )
    merged_data = {
        **data,
        "module": module_key,
        "message": answer_text,
        "rendered_boards": data.get("rendered_boards") or data.get("boards") or [],
        "outfits": data.get("outfits") or cards,
    }
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}

    return {
        **payload,
        # Module chat succeeded even when the style payload is a helpful
        # missing-wardrobe response. Keep that distinct from transport/provider
        # failure so the app does not render its generic thinking fallback.
        "success": True,
        "module": module_key,
        "domain": module_key,
        "response": answer_text,
        "message_text": answer_text,
        "message": {"role": "assistant", "content": answer_text},
        "cards": cards,
        "style_boards": style_boards,
        "chips": payload.get("chips") if isinstance(payload.get("chips"), list) else [],
        "board_ids": str(payload.get("board_ids") or ""),
        "data": merged_data,
        "meta": {
            **meta,
            "mode": meta.get("mode") or "style_module_chat",
            "board_count": len(cards),
            "style_payload_success": bool(payload.get("success", True)),
        },
    }


def _is_plan_pack_request(message: str) -> bool:
    text = str(message or "").lower()
    has_pack = any(token in text for token in ("pack", "packing", "carry-on", "carry on"))
    has_trip = any(token in text for token in ("trip", "travel", "beach", "vacation", "destination", "goa"))
    has_prep = any(token in text for token in ("prep", "prepare", "checklist", "plan"))
    has_event_plan = any(token in text for token in ("birthday party", "birthday", "camping", "goa trip"))
    return (has_pack and (has_trip or has_prep)) or has_event_plan or (
        has_prep and any(token in text for token in ("camping", "trip", "travel", "party"))
    )


# Visual board routing. Diet / Pack / Plan prompts return a structured
# "visual_board" envelope (rendered by AhviVisualBoard on the client)
# instead of plain text. Keyword lists mirror the AHVI routing spec.
_VB_DIET_KEYWORDS = (
    "diet", "meal plan", "meal-plan", "what should i eat", "what to eat",
    "breakfast", "lunch", "dinner", "high protein", "high-protein",
    "vegetarian meal", "vegan meal", "weight loss food", "meal idea", "meal ideas",
)
_VB_PACK_KEYWORDS = (
    "pack", "packing", "carry-on", "carry on", "what should i carry",
    "what to carry", "airport bag", "travel bag", "beach bag", "gym bag", "luggage",
    "camping checklist", "camping prep",
)
_VB_PLAN_KEYWORDS = (
    "plan my day", "plan my tomorrow", "prepare me", "prep me", "tomorrow prep",
    "trip prep", "office prep", "before leaving", "get me ready",
    "help me prep", "plan for a", "goa trip", "birthday party", "plan a birthday",
)
_VB_SKIP_MODULES = {
    "skincare", "medi", "bills", "fitness", "style", "wardrobe", "daily_wear",
}


def _detect_visual_board_type(message: str, module: str = "") -> str:
    """Return a visual_board board_type for diet/pack/plan prompts, else ''."""
    text = str(message or "").lower().strip()
    if not text:
        return ""
    if str(module or "").lower() in _VB_SKIP_MODULES:
        return ""
    if _is_plan_pack_request(text):
        return "packing_checklist"
    if any(k in text for k in _VB_DIET_KEYWORDS):
        return "diet_plan"
    if any(k in text for k in _VB_PACK_KEYWORDS):
        return "packing_checklist"
    if any(k in text for k in _VB_PLAN_KEYWORDS):
        return "trip_prep"
    return ""


# Module summary cards — diet/meds/bills/etc chips that should render the
# user's REAL Appwrite data instead of a hardcoded demo card.
# Phrases are matched as substrings of the normalized message, so chip
# labels and freeform variants ("My medicines", "my medicines today",
# "show my meds") all route to the right card.
_MODULE_SUMMARY_INTENTS: Dict[str, tuple] = {
    "medicines": (
        "my medicines",
        "my medicine",
        "my meds",
        "todays medicine",
        "today medicine",
        "today's medicine",
        "today's medicines",
        "todays medicines",
        "today medicines",
        "show medicines",
        "list medicines",
        "medication list",
    ),
    "bills": (
        "my bills",
        "pending bills",
        "unpaid bills",
        "today's bills",
        "todays bills",
        "show bills",
        "list bills",
        "bills due",
    ),
    "events": (
        "my events",
        "today's events",
        "todays events",
        "today events",
        "today event",
        "upcoming events",
        "my schedule",
        "today's schedule",
        "todays schedule",
    ),
    "meals": (
        "my meals",
        "today's meals",
        "todays meals",
        "today meals",
        "today's food",
        "meal list",
        "show meals",
        "todays food",
    ),
    "workout": (
        "my workout",
        "today's workout",
        "todays workout",
        "today workout",
        "workout today",
        "show workout",
        "show exercises",
        "todays exercises",
    ),
    "skincare": (
        "my skincare",
        "morning skincare",
        "night skincare",
        "skincare routine",
        "my routine",
        "morning routine",
        "evening routine",
        "night routine",
        "create routine",
        "open skincare",
        "skincare steps",
        "show routine",
    ),
}


def _detect_module_summary(message: str) -> str:
    """Return a module key for a summary-card intent, else ''."""
    text = str(message or "").lower().replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    for module, phrases in _MODULE_SUMMARY_INTENTS.items():
        if any(phrase in text for phrase in phrases):
            return module
    return ""


def _detect_quick_action_module(message: str) -> str:
    text = str(message or "").lower().replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text in {"home workout", "gym workout", "workout outfit"}:
        return "fitness"
    if text in {"recovery meal", "light dinner ideas", "high protein meals", "high protein meal", "create todays meal plan"}:
        return "diet"
    if text in {"create routine"}:
        return "skincare"
    if text in {"add event", "view events", "open calendar", "open events", "add reminder", "plan outfit"}:
        return "calendar"
    if text in {"packing checklist", "open checklist", "plan outfits", "weather prep", "save trip plan"}:
        return "planner"
    return ""


def _is_ask_questions_action(message: str) -> bool:
    text = str(message or "").lower().replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text in {
        "ask me 2 questions",
        "ask me two questions",
        "ask 2 questions",
        "ask two questions",
    } or text.endswith(" ask me 2 questions") or text.endswith(" ask me two questions")


def _style_two_questions_response(original_prompt: str = "") -> Dict[str, Any]:
    questions = [
        "Is this dinner casual, polished, or dressy?",
        "Do you want to use your wardrobe or build a suggested look?",
    ]
    logger.info("style.clarification.questions prompt=%r count=2", original_prompt)
    return {
        "success": True,
        "type": "clarification_questions",
        "module": "style",
        "domain": "style",
        "intent": "ask_questions",
        "message": {
            "role": "assistant",
            "content": "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(questions)),
        },
        "message_text": "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(questions)),
        "response": "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(questions)),
        "questions": questions,
        "cards": [],
        "style_boards": [],
        "chips": [
            {"label": "Casual", "value": "casual dinner"},
            {"label": "Polished", "value": "polished dinner"},
            {"label": "Use my wardrobe", "value": "use my wardrobe"},
            {"label": "Suggested look", "value": "build a suggested look"},
        ],
        "quick_actions": ["Casual", "Polished", "Use my wardrobe", "Suggested look"],
        "data": {"intent": "ask_questions", "questions": questions},
        "meta": {"mode": "style_clarification_questions", "question_count": 2},
    }


def _planner_action_intent(message: str) -> str:
    text = str(message or "").lower().replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    mapping = {
        "packing checklist": "open_checklist",
        "open checklist": "open_checklist",
        "weather prep": "weather_prep",
        "save trip plan": "save_plan",
        "plan outfits": "plan_outfits",
    }
    return mapping.get(text, "")


def _looks_like_event_create_text(message: str) -> bool:
    text = str(message or "").lower()
    if not text.strip():
        return False
    if text.strip() in {"add event", "view events", "open calendar", "open events"}:
        return False
    return bool(
        re.search(
            r"\b(birthday|appointment|doctor|meeting|call)\b.*\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december|\d{1,2}(?:st|nd|rd|th)?)\b",
            text,
        )
    )


def _event_card(event: Dict[str, Any]) -> Dict[str, Any]:
    start_raw = str(event.get("start_time") or "")
    when = ""
    try:
        start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
        when = start.strftime("%d %b · %I:%M %p")
    except Exception:
        when = start_raw
    return {
        "type": "module_card",
        "module": "calendar",
        "title": event.get("title") or "Event",
        "subtitle": when,
        "summary": event.get("description") or when,
        "items": [
            item
            for item in [
                f"Type: {event.get('type')}" if event.get("type") else "",
                f"Status: {event.get('status')}" if event.get("status") else "",
            ]
            if item
        ],
        "cta": {"label": "Open calendar", "module": "calendar", "route": "calendar"},
        "open_module": "calendar",
    }


def _calendar_event_created_response(
    event: Dict[str, Any], *, reused: bool = False
) -> Dict[str, Any]:
    title = str(event.get("title") or "Event")
    start_raw = str(event.get("start_time") or "")
    day_text = "your calendar"
    try:
        start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
        day_text = start.strftime("%d %B")
    except Exception:
        pass
    message = f"{title} added for {day_text}."
    if reused:
        message = f"{title} is already on your calendar for {day_text}."
    actions = ["View events", "Add reminder", "Plan outfit"]
    cta = {"label": "Open calendar", "route": "calendar", "module": "calendar"}
    return {
        "success": True,
        "type": "module_response",
        "module": "calendar",
        "domain": "calendar",
        "intent": "calendar_event_reused" if reused else "event_created",
        "reused": reused,
        "message": {"role": "assistant", "content": message},
        "message_text": message,
        "response": message,
        "card": _event_card(event),
        "cards": [_event_card(event)],
        "chips": actions,
        "quick_actions": actions,
        "cta": cta,
        "open_module": cta,
        "data": {
            "module": "calendar",
            "intent": "calendar_event_reused" if reused else "event_created",
            "reused": reused,
            "event": event,
        },
    }


def _calendar_capture_response() -> Dict[str, Any]:
    message = "Tell me the event name and date/time. For example: Birthday on 23 July or Doctor appointment tomorrow at 6 PM."
    actions = ["Birthday on 23 July", "Meeting tomorrow at 4 PM", "View events"]
    return {
        "success": True,
        "type": "module_response",
        "module": "calendar",
        "domain": "calendar",
        "intent": "create_event",
        "message": {"role": "assistant", "content": message},
        "message_text": message,
        "response": message,
        "cards": [],
        "chips": actions,
        "quick_actions": actions,
        "cta": {"label": "Open calendar", "route": "calendar", "module": "calendar"},
        "open_module": {"label": "Open calendar", "route": "calendar", "module": "calendar"},
        "data": {"module": "calendar", "intent": "create_event"},
    }


def _build_visual_board_envelope(
    *,
    board_type: str,
    module_key: str,
    user_message: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a visual_board envelope, reusing existing module engines."""
    from services.board_service import (
        _detect_diet_variant,
        build_diet_visual_board,
        build_pack_visual_board,
        build_plan_visual_board,
    )

    # Keep `cards` empty: the board's own `sections` carry all content.
    # Populating `cards` makes visual-board-unaware clients render the
    # plan/pack rows as style outfit boards.
    cards: List[Dict[str, Any]] = []

    if board_type == "diet_plan":
        diet_variant = _detect_diet_variant(user_message)
        board = build_diet_visual_board(
            engine_result=None,
            user_context=context,
            diet_variant=diet_variant,
        )
    elif board_type == "packing_checklist":
        # Reuse the existing plan/pack engine for trip-aware sections.
        engine_result: Dict[str, Any] = {}
        try:
            engine_result = build_plan_pack_response(user_message, context)
        except Exception:
            logger.warning("visual_board.pack_engine_failed", exc_info=True)
        board = build_pack_visual_board(engine_result=engine_result or None)
    else:  # trip_prep
        board = build_plan_visual_board(engine_result=None, user_context=context)

    title = str(board.get("title") or "")
    subtitle = str(board.get("subtitle") or "")
    message_text = (f"{title} — {subtitle}" if subtitle else title).strip(" —")
    actions = ["Packing checklist", "Plan outfits", "Weather prep", "Save trip plan"] if board_type in {"packing_checklist", "trip_prep"} else []

    return {
        "success": True,
        "type": "visual_board",
        "response_type": "visual_board",
        "module": module_key or "planner",
        "domain": module_key or "planner",
        "intent": "plan_pack" if board_type in {"packing_checklist", "trip_prep"} else board_type,
        "board_type": board.get("board_type"),
        "title": title,
        "subtitle": subtitle,
        "principles": board.get("principles") or [],
        "sections": board.get("sections") or [],
        "why_this_plan": board.get("why_this_plan") or "",
        "visual_board": board,
        # Text fallback for clients that do not render visual boards yet.
        "message": message_text,
        "message_text": message_text,
        "response": message_text,
        "cards": cards,
        "style_boards": [],
        "chips": actions,
        "quick_actions": actions,
        "data": {
            "module": module_key or "planner",
            "intent": "plan_pack" if board_type in {"packing_checklist", "trip_prep"} else board_type,
            "visual_board": board,
            "message": message_text,
            "rendered_boards": [],
            "outfits": [],
        },
        "meta": {
            "mode": "visual_board",
            "board_type": board.get("board_type"),
            "module_route": module_key or "planner",
        },
    }


def _plan_pack_emoji(scenario: str) -> str:
    return {
        "camping": "🏕️",
        "birthday": "🎉",
        "travel": "🧳",
        "business": "💼",
        "wedding": "💍",
    }.get(str(scenario or "").lower(), "🧳")


def _save_plan_pack_payload(*, user_id: str, payload: Dict[str, Any], reminder: bool = False) -> str:
    uid = str(user_id or "").strip()
    if not uid:
        return "Sign in again so I can save this plan to your account."

    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    cards = payload.get("cards") if isinstance(payload.get("cards"), list) else []
    scenario = str(data.get("scenario") or "travel")
    destination = str(data.get("destination") or "Trip")
    duration = str(data.get("duration_label") or "")
    first_title = ""
    if cards and isinstance(cards[0], dict):
        first_title = str(cards[0].get("title") or "").strip()
    occasion = first_title or (f"{duration} {destination}".strip() if destination else "Trip Plan")
    now = datetime.now(timezone.utc)
    plan_doc = {
        "userId": uid,
        "occasion": occasion,
        "emoji": _plan_pack_emoji(scenario),
        "outfitDescription": json.dumps(
            {
                "type": "plan_pack",
                "occasion": occasion,
                "items": cards,
                "checkedItems": [],
                "destination": destination,
                "duration": duration,
                "scenario": scenario,
                "weatherContext": {
                    "weather": data.get("weather"),
                    "time_of_day": data.get("time_of_day"),
                },
            },
            ensure_ascii=False,
        ),
        "dateTime": now.isoformat(),
        "reminder": bool(reminder),
        "outfitId": "",
    }
    try:
        created = AppwriteProxy().create_document("plans", plan_doc)
        plan_id = str(created.get("$id") or created.get("id") or "")
        if reminder and plan_id:
            from services.notification_store import notification_store

            notification_store.schedule_reminders(
                user_id=uid,
                event_id=plan_id,
                reminders=[
                    {
                        "status": "pending",
                        "priority": "normal",
                        "offsetMinutes": -60,
                        "message": f"Your {occasion} checklist is ready. Want to review outfits?",
                        "sendAtISO": (now + timedelta(hours=1)).isoformat(),
                    }
                ],
                source="ahvi_plan_pack",
            )
        if "goa" in occasion.lower():
            return "Saved your Goa trip plan."
        if scenario == "birthday":
            return "Saved your birthday party plan."
        return f"Saved your {occasion}."
    except Exception as exc:
        logger.warning("plan_pack.save_failed user_id=%s error=%s", uid, exc)
        return "I could not save this plan yet. Please try again."


def _module_plan_pack_response(
    *,
    module_key: str,
    user_message: str,
    context_data: Dict[str, Any],
    user_profile: Dict[str, Any],
    user_id: str = "",
) -> Dict[str, Any]:
    context = dict(context_data or {})
    if user_profile:
        context["user_profile"] = user_profile
    if not isinstance(context.get("wardrobe"), list) and user_id:
        try:
            wardrobe = _fetch_wardrobe_for_style(user_id, None) or []
            context["wardrobe"] = wardrobe
            context["wardrobe_items"] = wardrobe
            logger.info(
                "chat.plan_pack_wardrobe_fetch user_id=%s count=%s",
                user_id,
                len(wardrobe),
            )
        except Exception as exc:
            logger.warning(
                "chat.plan_pack_wardrobe_fetch_failed user_id=%s error=%s",
                user_id,
                exc,
            )
            context.setdefault("wardrobe", [])
    action_intent = _planner_action_intent(user_message)
    active_prompt = str(
        context.get("source_text")
        or context.get("original_prompt")
        or context.get("resolved_prompt")
        or context.get("active_plan_prompt")
        or user_message
    ).strip()
    if action_intent and active_prompt.lower() == user_message.lower():
        active_prompt = str(context.get("last_plan_prompt") or context.get("plan_prompt") or user_message).strip()

    if action_intent == "plan_outfits":
        style_query = str(
            context.get("resolved_prompt")
            or context.get("active_plan_prompt")
            or context.get("source_text")
            or "travel outfits for this plan"
        ).strip()
        if not re.search(r"\boutfit|wear|style\b", style_query, re.I):
            style_query = f"{style_query} outfits"
        wardrobe = context.get("wardrobe") if isinstance(context.get("wardrobe"), list) else []
        style_payload = _demo_style_board_payload(
            user_id=user_id or str(user_profile.get("user_id") or user_profile.get("$id") or ""),
            query_text=style_query,
            request_wardrobe=wardrobe,
            user_profile=user_profile,
        )
        envelope = _module_style_response_envelope("style", style_payload)
        envelope["intent"] = "plan_outfits"
        envelope.setdefault("meta", {})["source_action"] = "plan_outfits"
        return envelope

    payload = build_plan_pack_response(user_message, context)
    if action_intent:
        payload = build_plan_pack_response(active_prompt, context)
        if action_intent == "open_checklist":
            cards = [
                c for c in payload.get("cards", [])
                if isinstance(c, dict) and c.get("id") in {"packing_clothes", "packing_essentials"}
            ] or payload.get("cards", [])
            payload["cards"] = cards
            visual_sections = [
                s for s in payload.get("visual_sections", [])
                if isinstance(s, dict) and s.get("id") in {"clothes", "essentials", "tech", "documents"}
            ]
            if visual_sections:
                payload["visual_sections"] = visual_sections
            payload["message"] = "Here is your checklist. Tap items as you pack them."
            payload["intent"] = "open_checklist"
        elif action_intent == "weather_prep":
            cards = [
                c for c in payload.get("cards", [])
                if isinstance(c, dict) and c.get("id") == "weather_time_adjustments"
            ]
            payload["cards"] = cards
            visual_sections = [
                s for s in payload.get("visual_sections", [])
                if isinstance(s, dict) and s.get("id") == "weather"
            ]
            if visual_sections:
                payload["visual_sections"] = visual_sections
            payload["message"] = "Here is the weather prep for this plan."
            payload["intent"] = "weather_prep"
        elif action_intent == "save_plan":
            payload["intent"] = "save_plan"
            payload["message"] = _save_plan_pack_payload(
                user_id=user_id,
                payload=payload,
                reminder=bool(context.get("reminder")),
            )

    message = str(payload.get("message") or "I built your trip plan and packing checklist.")
    cards = payload.get("cards") if isinstance(payload.get("cards"), list) else []
    visual_sections = payload.get("visual_sections") if isinstance(payload.get("visual_sections"), list) else []
    actions = payload.get("quick_actions") if isinstance(payload.get("quick_actions"), list) else (
        payload.get("chips") if isinstance(payload.get("chips"), list) else []
    )
    logger.info(
        "chat.plan_pack_module_route module=%s cards=%s prompt=%r",
        module_key,
        len(cards),
        user_message,
    )
    return {
        "success": True,
        "type": payload.get("type") or "checklists",
        "visual_type": payload.get("visual_type") or "visual_packing_checklist",
        "module": "plan_pack" if action_intent else (module_key or "planner"),
        "domain": "plan_pack" if action_intent else (module_key or "planner"),
        "intent": payload.get("intent") or "plan_pack",
        "response": message,
        "message_text": message,
        "message": {"role": "assistant", "content": message},
        "chips": actions,
        "quick_actions": actions,
        "cards": cards,
        "visual_sections": visual_sections,
        "style_boards": payload.get("style_boards") if isinstance(payload.get("style_boards"), list) else [],
        "data": payload.get("data") if isinstance(payload.get("data"), dict) else {},
        "meta": {
            "intent": payload.get("intent") or "plan_pack",
            "board": payload.get("board") or "plan_pack",
            "module_route": module_key or "planner",
            "action_intent": action_intent,
        },
        "context_usage": context.get("context_usage") or {},
    }


def _state_user_id(http_request: Request) -> str:
    state_user = getattr(http_request.state, "user", None)
    if isinstance(state_user, dict):
        return str(
            state_user.get("user_id")
            or state_user.get("$id")
            or state_user.get("id")
            or ""
        ).strip()
    return ""


def _module_llm_response(
    *,
    module: str,
    user_message: str,
    history: List[Dict[str, str]],
    context_data: Dict[str, Any],
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    module_key = _normalize_module_name(module)
    system_instruction = _MODULE_CHAT_PROMPTS.get(
        module_key,
        "You are AHVI. Answer directly using the provided context. Do not invent missing data.",
    )

    if context_data:
        system_instruction += (
            "\n\nUse this app context. Do not invent missing data:\n"
            + json.dumps(context_data, ensure_ascii=False, default=str)[:6000]
        )

    messages: List[Dict[str, str]] = []
    for item in list(history or [])[-10:]:
        role = str(item.get("role") or "user").lower().strip()
        content = str(item.get("content") or item.get("text") or "").strip()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        if content:
            messages.append({"role": role, "content": content[:1200]})
    messages.append({"role": "user", "content": user_message})

    answer: str = ""
    started_at = time.perf_counter()
    try:
        answer = chat_completion(
            messages,
            system_instruction=system_instruction,
            user_profile=user_profile,
            signals={"context_mode": module_key},
            # Hard cap LLM call at 30s so the module-chat endpoint always
            # returns within the frontend's 75s budget. If the LLM is slow,
            # we'd rather ship lightweight_chat() than time the user out.
            timeout_seconds=30,
            options={"temperature": 0.45, "max_output_tokens": 320},
            usecase=f"{module_key}_chat",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        logger.warning(
            "chat.module_chat_failed module=%s elapsed=%.2fs error=%s",
            module_key, elapsed, str(exc)[:180],
        )
        answer = ""
    if _looks_incomplete_module_answer(answer):
        # Always return something. lightweight_chat covers greetings and
        # falls through to a generic helpful sentence for anything else.
        answer = _module_fallback_answer(module_key, user_message)
    logger.info(
        "chat.module_chat_ok module=%s elapsed=%.2fs len=%s",
        module_key, time.perf_counter() - started_at, len(answer),
    )

    answer_text = str(answer or "").strip()
    # Canonical AHVI chat response shape. Every chat endpoint
    # (/api/text, /api/chat/module-chat) MUST return this same envelope
    # so the frontend parser doesn't have to special-case the source.
    return {
        "success": True,
        "type": "module_chat",
        "module": module_key,
        # Three keys carrying the same payload — different clients pick
        # different keys. Keep them aligned to avoid empty-message bugs.
        "response": answer_text,
        "message_text": answer_text,
        "message": {"role": "assistant", "content": answer_text},
        "cards": [],
        "style_boards": [],
        "chips": [],
        "data": {
            "module": module_key,
            "message": answer_text,
            "rendered_boards": [],
            "outfits": [],
        },
        "meta": {
            "mode": module_key,
            "board_count": 0,
        },
    }


@router.post("/module-chat")
@router.post("/chat/module-chat")
async def module_chat(request: ModuleChatRequest, http_request: Request):
    module = _normalize_module_name(request.domain or request.module or "")
    profile = dict(request.user_profile or {})
    user_id = _state_user_id(http_request)
    if user_id:
        profile["user_id"] = user_id
    user_message = str(request.message or "").strip()
    merged_context = {**(request.context_data or {}), **(request.context or {})}
    resolved_context = resolve_location_weather_context(
        user_id=user_id,
        request_data=request.model_dump(exclude_none=True),
        profile=profile,
    )
    profile = resolved_context["profile"]
    merged_context["location_context"] = resolved_context["location"]
    merged_context["weather_context"] = resolved_context["weather"]
    merged_context["weather_data"] = resolved_context["weather"]
    merged_context["context_usage"] = resolved_context["context_usage"]
    if resolved_context["weather"].get("status") == "available":
        profile["weather"] = resolved_context["weather"]
    # Forward chat history so module handlers (e.g. calendar) can slot-fill
    # across turns ("tomorrow" from an earlier turn + "shopping at 5pm").
    if request.history and "history" not in merged_context:
        merged_context["history"] = request.history

    if _is_ask_questions_action(user_message) and module in {"style", "wardrobe", "daily_wear", "chat", ""}:
        return _style_two_questions_response(user_message)

    _qa_module = _detect_quick_action_module(user_message)
    if _qa_module == "calendar":
        text = user_message.lower().strip()
        if text in {"add event"}:
            return _calendar_capture_response()
        if text in {"view events", "open events", "open calendar"} and user_id:
            from services.module_summary_service import build_module_summary

            return build_module_summary("events_upcoming", user_id)
        return await handle_module_chat(
            {
                "domain": "calendar",
                "module": "calendar",
                "message": user_message,
                "context": merged_context,
                "user_profile": profile,
            },
            user_id=user_id,
        )
    if _qa_module == "planner":
        return _module_plan_pack_response(
            module_key=module or "planner",
            user_message=user_message,
            context_data=merged_context,
            user_profile=profile,
            user_id=user_id,
        )
    if _qa_module in {"fitness", "diet", "skincare"}:
        return await handle_module_chat(
            {
                "domain": _qa_module,
                "module": _qa_module,
                "message": user_message,
                "context": merged_context,
                "user_profile": profile,
            },
            user_id=user_id,
        )

    if module in {"calendar", "planner", "chat", ""} and user_id and _looks_like_event_create_text(user_message):
        from services.calendar_service import (
            create_calendar_event,
            find_existing_event,
            parse_plan_text_to_payload,
        )

        try:
            payload = parse_plan_text_to_payload(user_message, timezone_name="Asia/Kolkata")
            # Same idempotency rule as module_chat_service: reuse the existing
            # event (same user + normalized title + start minute) instead of
            # creating a duplicate. Shared helper, not a second implementation.
            # ponytail: query-then-create, still non-atomic — concurrent
            # identical requests can duplicate; needs a unique index to close.
            existing = find_existing_event(
                user_id, payload.get("title"), payload.get("start_time")
            )
            if existing:
                return _calendar_event_created_response(existing, reused=True)
            event = create_calendar_event(user_id, payload)
            return _calendar_event_created_response(event)
        except ValueError as exc:
            if str(exc) == "time_required":
                message = "What time should I save this for?"
                actions = ["Today 6 PM", "Tomorrow 9 AM", "Open calendar"]
                return {
                    "success": True,
                    "type": "module_response",
                    "module": "calendar",
                    "domain": "calendar",
                    "intent": "event_needs_time",
                    "message": {"role": "assistant", "content": message},
                    "message_text": message,
                    "response": message,
                    "cards": [],
                    "chips": actions,
                    "quick_actions": actions,
                    "cta": {"label": "Open calendar", "route": "calendar", "module": "calendar"},
                    "open_module": {"label": "Open calendar", "route": "calendar", "module": "calendar"},
                }

    _ms_module = _detect_module_summary(user_message)
    if _ms_module and user_id:
        from services.module_summary_service import build_module_summary

        if _ms_module == "events" and "upcoming" in user_message.lower():
            _ms_module = "events_upcoming"
        _ms_card = build_module_summary(_ms_module, user_id)
        if _ms_card:
            return _ms_card

    if module in {"planner", "calendar", "prep", "chat", ""} and _is_plan_pack_request(user_message):
        return _module_plan_pack_response(
            module_key=module or "planner",
            user_message=user_message,
            context_data=merged_context,
            user_profile=profile,
            user_id=user_id,
        )

    _vb_type = _detect_visual_board_type(user_message, module)
    if _vb_type == "diet_plan" and _is_style_priority_query(user_message):
        logger.info("AHVI_STYLE_PRIORITY_GUARD_APPLIED prompt=%r", str(user_message)[:80])
        logger.info("AHVI_DIET_FALSE_POSITIVE_BLOCKED prompt=%r", str(user_message)[:80])
        _vb_type = ""
    if _vb_type:
        _vb_context = dict(merged_context or {})
        if profile:
            _vb_context["user_profile"] = profile
        return _build_visual_board_envelope(
            board_type=_vb_type,
            module_key=module,
            user_message=user_message,
            context=_vb_context,
        )

    if module in {"skincare", "diet", "meal", "planner", "calendar", "medi", "bills", "fitness"}:
        return await handle_module_chat(
            {
                "domain": module,
                "module": module,
                "message": user_message,
                "context": merged_context,
                "user_profile": profile,
            },
            user_id=user_id,
        )

    if module in {"style", "wardrobe", "daily_wear"} and (
        _is_explicit_style_request(user_message, module)
        or _needs_style_clarification(user_message, _ahvi_style_occasion(user_message))
        or _ahvi_style_occasion(user_message) != "today"
    ):
        wardrobe = (
            merged_context.get("wardrobe")
            or merged_context.get("outfits")
            or merged_context.get("items")
            or request.context_data.get("wardrobe")
            or request.context.get("wardrobe")
        )
        module_intent_row = detect_intent(user_message)
        module_intent = str(module_intent_row.get("intent") or "general").strip().lower()
        visual_first = _should_default_visual_inspiration(
            user_message,
            intent=module_intent,
            module_context=module,
            multi_event=merged_context.get("multi_event")
            if isinstance(merged_context.get("multi_event"), dict)
            else None,
        )
        wardrobe_override = _is_use_wardrobe_action(prompt=user_message) or module in {
            "wardrobe",
            "closet",
        }
        selected_mode = VISUAL_INSPIRATION if visual_first else WARDROBE_STYLE
        logger.info(
            "AHVI_VISUAL_FIRST_ROUTE endpoint=module_chat toggle=%s selected_mode=%s intent=%s occasion=%s wardrobe_override=%s",
            _style_default_visual_inspiration_enabled(),
            selected_mode,
            module_intent,
            _ahvi_style_occasion(user_message),
            wardrobe_override,
        )
        if visual_first:
            reasoning = style_reasoning_engine.reason(
                query=user_message,
                intent={"intent": VISUAL_INSPIRATION, "confidence": 0.95},
                user_profile=profile,
                context={
                    **merged_context,
                    "module_context": module,
                    "user_id": user_id,
                    "occasion": _ahvi_style_occasion(user_message),
                    "wardrobe": wardrobe if isinstance(wardrobe, list) else [],
                    "style_action": VISUAL_INSPIRATION,
                },
            )
            return _style_reasoning_chat_response(
                reasoning,
                user_message,
                module,
                wardrobe=wardrobe if isinstance(wardrobe, list) else [],
            )
        style_payload = _demo_style_board_payload(
            user_id=user_id or str(profile.get("user_id") or profile.get("$id") or ""),
            query_text=user_message,
            request_wardrobe=wardrobe,
            user_profile=profile,
        )
        # /api/module-chat must stamp the durable board contract too — it does
        # not converge on _beta_style_response like /api/text, so without this
        # the wardrobe boards ship with id=outfit_card_N but no board_id /
        # revision / source_policy and the app disables locked Shuffle.
        style_payload = _apply_style_compliance_gate(
            style_payload,
            query=user_message,
            user_id=user_id or str(profile.get("user_id") or profile.get("$id") or ""),
            wardrobe=wardrobe,
        )
        return _module_style_response_envelope(module, style_payload)

    return _module_llm_response(
        module=module,
        user_message=user_message,
        history=request.history,
        context_data=merged_context,
        user_profile=profile,
    )


@router.post("/text")
def text_chat(request: TextChatRequest, http_request: Request):

    # -------------------------
    # INPUT VALIDATION
    # -------------------------
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_input = (request.messages[-1].content or "").strip()

    # Backwards-compat: older calendar.dart builds wrap user text as
    # "Occasion: <X>\n\n<user text>". Strip the prefix and promote
    # occasion into user_profile so the orchestrator sees it as a
    # structured slot instead of polluting the LLM prompt.
    import re as _re_occ
    _occ_match = _re_occ.match(r"^\s*Occasion:\s*([^\n]+)\n+", user_input, flags=_re_occ.IGNORECASE)
    if _occ_match:
        _legacy_occasion = _occ_match.group(1).strip().lower()
        user_input = user_input[_occ_match.end():].strip()
        if isinstance(request.user_profile, dict):
            if not request.user_profile.get("occasion"):
                request.user_profile["occasion"] = _legacy_occasion
        else:
            request.user_profile = {"occasion": _legacy_occasion}

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    if _is_ask_questions_action(user_input):
        return _style_two_questions_response(user_input)

    # ──────────────────────────────────────────────────────────────────
    # CHIP / BUTTON / RETRY CONTEXT RESOLUTION
    # ──────────────────────────────────────────────────────────────────
    # The FE may send action chips (Next best options / More looks /
    # Make it casual) or a retry. These need the original style
    # context, not the bare label. Either:
    #   (a) action + resolved_prompt is set → use resolved_prompt
    #   (b) clarification is set → merge with previous_prompt
    #   (c) bare action label arrives without context → ask the user
    #       which look to continue from instead of running the
    #       orchestrator on the label text alone.
    _BARE_ACTION_PROMPTS = {
        "try again", "next best options", "more looks", "try different shoes",
        "make it casual", "make it polished", "use my wardrobe",
        "ask me 2 questions",
        # Weak-match recovery chips:
        "show closest option", "show closest", "closest option",
        "try another occasion", "find missing item", "use safer outfit",
    }
    # Map action labels → orchestrator style_action so the safety gate
    # can unlock the closest-option path / more-looks path.
    _ACTION_LABEL_TO_STYLE_ACTION = {
        "show closest option": "show_closest_option",
        "show closest": "show_closest_option",
        "closest option": "show_closest_option",
        "show_closest_option": "show_closest_option",
        "next best options": "more_options",
        "next_best_options": "more_options",
        "more looks": "more_options",
        "more_looks": "more_options",
        "try different shoes": "more_options",
        "try_different_shoes": "more_options",
        "try again": "retry",
        "retry": "retry",
    }
    _action_label = ""
    _action_key = ""
    style_action = None
    _resolved_in = ""
    _previous_in = ""
    _clarification_in = ""
    _lower_input = user_input.strip().lower()
    _log_user_id = _state_user_id(http_request) or str(request.user_id or request.userID or "").strip()

    try:
        payload: Any = request
        prompt = user_input
        if isinstance(payload, dict):
            _action_label = (
                payload.get("action")
                or payload.get("message")
                or payload.get("prompt")
                or prompt
                or ""
            )
            _resolved_in = str(payload.get("resolved_prompt") or "").strip()
            _previous_in = str(payload.get("previous_prompt") or "").strip()
            _clarification_in = str(payload.get("clarification") or "").strip()
        else:
            _action_label = (
                getattr(payload, "action", None)
                or getattr(payload, "message", None)
                or getattr(payload, "prompt", None)
                or prompt
                or ""
            )
            _resolved_in = str(getattr(payload, "resolved_prompt", None) or "").strip()
            _previous_in = str(getattr(payload, "previous_prompt", None) or "").strip()
            _clarification_in = str(getattr(payload, "clarification", None) or "").strip()
            _style_context_in = (
                getattr(payload, "style_context", None)
                if isinstance(getattr(payload, "style_context", None), dict)
                else {}
            )
            _request_context_in = (
                getattr(payload, "context", None)
                if isinstance(getattr(payload, "context", None), dict)
                else {}
            )
            if _request_context_in and not _style_context_in:
                _style_context_in = _request_context_in
            if _style_context_in:
                _resolved_in = _resolved_in or str(
                    _style_context_in.get("resolved_prompt")
                    or _style_context_in.get("resolvedPrompt")
                    or ""
                ).strip()
                _previous_in = _previous_in or str(
                    _style_context_in.get("original_prompt")
                    or _style_context_in.get("originalPrompt")
                    or ""
                ).strip()

        _action_label = str(_action_label or "").strip()
        _action_key = _action_label.lower()
        style_action = (
            _ACTION_LABEL_TO_STYLE_ACTION.get(_action_key)
            if _action_key
            else None
        )
        if not style_action and _lower_input:
            style_action = _ACTION_LABEL_TO_STYLE_ACTION.get(_lower_input)
        if style_action and not request.style_action:
            request.style_action = style_action
        if style_action == "show_closest_option":
            request.style_action = "show_closest_option"
            request.show_closest_option = True
            request.allow_closest_option = True
            request.closest = True

        logger.info(
            "style_action_context_parsed user_id=%s prompt=%r action_label=%r action=%r",
            _log_user_id,
            prompt,
            _action_label,
            style_action,
        )
    except Exception:
        logger.exception(
            "style_action_parse_failed user_id=%s prompt=%r",
            _log_user_id,
            user_input,
        )
        _action_label = ""
        _action_key = ""
        style_action = None
        safe_message = (
            "What are we dressing for today? Pick an occasion, or tell me the weather, "
            "timing, mood, and any dress code."
        )
        return {
            "success": True,
            "response": safe_message,
            "message": safe_message,
            "text": safe_message,
            "type": "clarification",
            "chips": ["Office", "Casual", "Date", "Party", "Travel", "Workout"],
            "data": {
                "intent": "style",
                "requires_clarification": True,
                "safe_recovery": True,
                "original_prompt": user_input,
            },
        }

    if _clarification_in and _previous_in and " · " not in user_input:
        user_input = f"{_previous_in} · {_clarification_in}"
        logger.info(
            "style_clarification_selected user_id=%s original_prompt=%r selected_chip=%r resolved_prompt=%r",
            "", _previous_in, _clarification_in, user_input,
        )
    elif _resolved_in and _resolved_in.lower() != user_input.lower():
        # FE sent a resolved prompt explicitly — trust it.
        user_input = _resolved_in

    def _style_context_cache_key() -> str:
        return str(_log_user_id or request.session_id or "").strip()

    def _is_richer_style_prompt(candidate: str, current: str = "") -> bool:
        cand = str(candidate or "").strip()
        cur = str(current or "").strip()
        if not cand:
            return False
        cand_has_merge = " · " in cand or " Â· " in cand or " Ã‚Â· " in cand
        cur_has_merge = " · " in cur or " Â· " in cur or " Ã‚Â· " in cur
        if cand_has_merge and not cur_has_merge:
            return True
        if cand_has_merge == cur_has_merge and len(cand) > len(cur):
            return True
        return False

    # Older frontend builds may send only the chip label ("Show closest
    # option") even though the /api/text payload still contains prior chat
    # messages. Recover the last real style prompt from history so action
    # chips can continue the existing request instead of asking the user to
    # start over.
    _lower_input = user_input.strip().lower()
    _history_recovered_prompt = ""
    if (
        _lower_input in _BARE_ACTION_PROMPTS
        and request.style_action
        and not _resolved_in
        and not _previous_in
        and not _clarification_in
        and " Â· " not in user_input
    ):
        _fallback_recovered_prompt = ""
        for _hist_msg in reversed(list(request.messages or [])[:-1]):
            _hist_role = str(getattr(_hist_msg, "role", "") or "").strip().lower()
            if _hist_role and _hist_role != "user":
                continue
            _hist_text = str(getattr(_hist_msg, "content", "") or "").strip()
            _hist_key = _hist_text.lower()
            if (
                not _hist_text
                or _hist_key in _BARE_ACTION_PROMPTS
                or _hist_key in {"try again", "retry"}
            ):
                continue
            if (
                _is_explicit_style_request(_hist_text, request.module_context)
                or _ahvi_style_occasion(_hist_text) != "today"
            ):
                if (
                    " · " in _hist_text
                    or " Ã‚Â· " in _hist_text
                    or _is_explicit_style_request(_hist_text, request.module_context)
                ):
                    _history_recovered_prompt = _hist_text
                    break
                if not _fallback_recovered_prompt:
                    _fallback_recovered_prompt = _hist_text
        if not _history_recovered_prompt:
            _history_recovered_prompt = _fallback_recovered_prompt
        _cached_prompt = ""
        _cache_key_for_style = _style_context_cache_key()
        if _cache_key_for_style:
            _cached_prompt = str(_STYLE_CONTEXT_CACHE.get(_cache_key_for_style) or "").strip()
        if _is_richer_style_prompt(_cached_prompt, _history_recovered_prompt):
            _history_recovered_prompt = _cached_prompt
        if _history_recovered_prompt:
            _previous_in = _history_recovered_prompt
            user_input = _history_recovered_prompt
            _lower_input = user_input.strip().lower()
            logger.info(
                "style_action_context_recovered user_id=%s action=%s recovered_prompt=%r",
                _log_user_id,
                _action_key or request.style_action,
                user_input,
            )
            if request.style_action == "show_closest_option":
                request.show_closest_option = True
                request.allow_closest_option = True
                request.closest = True
                logger.info(
                    "style_closest_option_requested user_id=%s recovered_prompt=%r style_action=%s",
                    _log_user_id,
                    user_input,
                    request.style_action,
                )
    else:
        _cache_key_for_style = _style_context_cache_key()
        if (
            _cache_key_for_style
            and _lower_input not in _BARE_ACTION_PROMPTS
            and (
                _is_explicit_style_request(user_input, request.module_context)
                or _ahvi_style_occasion(user_input) != "today"
            )
        ):
            _existing_prompt = str(_STYLE_CONTEXT_CACHE.get(_cache_key_for_style) or "").strip()
            if _is_richer_style_prompt(user_input, _existing_prompt) or not _existing_prompt:
                _STYLE_CONTEXT_CACHE.set(_cache_key_for_style, user_input)
                logger.info(
                    "style_action_context_cached user_id=%s prompt=%r",
                    _log_user_id,
                    user_input,
                )

    _occasion_chip_values = {
        "office": "Office outfit",
        "casual": "Casual outfit",
        "date": "Date outfit tonight",
        "party": "Party outfit tonight",
        "travel": "Airport travel outfit",
        "workout": "Workout outfit",
    }
    _cache_key_for_style = _style_context_cache_key()
    if (
        _cache_key_for_style
        and _lower_input in _occasion_chip_values
        and not _resolved_in
        and not _previous_in
        and not _clarification_in
    ):
        _cached_prompt = str(_STYLE_CONTEXT_CACHE.get(_cache_key_for_style) or "").strip()
        if _cached_prompt and _cached_prompt.lower() != _lower_input:
            _chip_label = _occasion_chip_values[_lower_input]
            user_input = f"{_cached_prompt} · {_chip_label}"
            _lower_input = user_input.strip().lower()
            logger.info(
                "style_clarification_context_recovered user_id=%s chip=%s resolved_prompt=%r",
                _log_user_id,
                _chip_label,
                user_input,
            )

    if _action_key:
        logger.info(
            "style_action_context_received action=%s original_prompt=%r resolved_prompt=%r session_id=%s current_look_id=%s",
            _action_key, _previous_in, user_input,
            request.session_id or "", request.current_look_id or "",
        )

    # Detect bare action prompts (no context attached). Reject early
    # rather than letting the orchestrator chase "Next best options"
    # as if it were a brand new style request.
    _lower_input = user_input.strip().lower()
    _is_bare_action = (
        _lower_input in _BARE_ACTION_PROMPTS
        and not _resolved_in
        and not _previous_in
        and not _clarification_in
        and " · " not in user_input
    )
    if _is_bare_action:
        logger.warning(
            "frontend_context_missing action=%s prompt=%r",
            _action_key or _lower_input, user_input,
        )
        chips = [
            {"label": "Start new style request", "value": "What should I wear today?"},
            {"label": "Use my wardrobe", "value": "Use my wardrobe"},
            {"label": "Beach wear", "value": "Beach wear"},
            {"label": "Office wear", "value": "Office wear"},
        ]
        return {
            "success": True,
            "ok": True,
            "type": "context_required",
            "intent": "style",
            "message": {
                "role": "assistant",
                "content": "I need the previous style context to continue. What look should I build from?",
            },
            "message_text": "I need the previous style context to continue. What look should I build from?",
            "response": "I need the previous style context to continue. What look should I build from?",
            "chips": chips,
            "cards": [],
            "style_boards": [],
            "data": {
                "intent": "style",
                "requires_context": True,
                "missing_context_for_action": _action_key or _lower_input,
            },
            "meta": {"mode": "context_required"},
            "audio_job_id": "offline",
        }

    # SECURITY: user_id MUST come from the authenticated bearer token.
    # Falling back to request body / "user_1" sentinel allows cross-account
    # cache + wardrobe contamination on the same instance.
    auth_user_id = ""
    state_user = getattr(http_request.state, "user", None)
    if isinstance(state_user, dict):
        auth_user_id = str(
            state_user.get("user_id") or state_user.get("$id") or ""
        ).strip()
    if not auth_user_id:
        raise HTTPException(
            status_code=401, detail="Authenticated user is required"
        )
    # If the client sent a user_id, it must match the authed user.
    for supplied in (
        (request.user_id or "").strip(),
        (request.userID or "").strip(),
        (
            str(request.user_profile.get("user_id") or "").strip()
            if isinstance(request.user_profile, dict)
            else ""
        ),
    ):
        if supplied and supplied != auth_user_id:
            raise HTTPException(
                status_code=403, detail="user_id does not match authenticated user"
            )
    user_id = auth_user_id
    user_message_style = _infer_user_message_style(user_input)
    profile_started = time.perf_counter()
    effective_user_profile = _ahvi_resolve_effective_user_profile(
        user_id,
        request.user_profile if isinstance(request.user_profile, dict) else {},
    )
    profile_ms = round((time.perf_counter() - profile_started) * 1000, 2)
    resolved_context = resolve_location_weather_context(
        user_id=user_id,
        request_data=request.model_dump(exclude_none=True),
        profile=effective_user_profile,
    )
    effective_user_profile = resolved_context["profile"]
    weather_data = resolved_context["weather"]
    weather_for_consumers = weather_data if weather_data.get("status") == "available" else {}
    if weather_data.get("status") == "available":
        effective_user_profile["weather"] = weather_data
    request.context["location_context"] = resolved_context["location"]
    request.context["weather_context"] = resolved_context["weather"]
    request.context["weather_data"] = resolved_context["weather"]
    request.context["context_usage"] = resolved_context["context_usage"]

    # Beta Intelligence Bridge. This is request-carried and persistence-free;
    # it does not add another intent/model invocation.
    beta_state = normalize_style_state(request.style_state)
    beta_instructions = interpret_style_followup(user_input, beta_state)
    beta_dispatch = beta_style_engine_dispatch(beta_instructions)
    if beta_state.get("board_items"):
        logger.info(
            "beta_style_bridge action=%s engine=%s board_hash=%s",
            beta_instructions.get("action"),
            beta_dispatch.get("engine"),
            str(beta_state.get("board_content_hash") or "")[:12],
        )
        if beta_instructions.get("needs_clarification"):
            question = str(
                beta_instructions.get("clarification_question")
                or "What should I preserve, and what should I change?"
            )
            return {
                "success": True,
                "ok": True,
                "type": "style_clarification",
                "message": {"role": "assistant", "content": question},
                "message_text": question,
                "response": question,
                "cards": [],
                "style_boards": [],
                "chips": [],
                "style_state": beta_state,
                "understood": beta_instructions,
                "constraint_status": {
                    "passed_constraints": [],
                    "unresolved_constraints": ["request_ambiguity"],
                    "fallback_reason": "clarification_required",
                    "repair_attempted": False,
                    "final_validation_status": "clarification",
                },
                "visual_intelligence": None,
                "recommended_actions": [],
                "meta": {
                    "mode": "beta_style_clarification",
                    "beta_style_bridge_version": "1.0",
                },
            }
        if beta_instructions.get("action") in {
            "explain_current_board",
            "critique_current_board",
            "identify_wardrobe_gap",
        }:
            action_name = str(beta_instructions.get("action") or "")
            if action_name == "identify_wardrobe_gap":
                text = (
                    "I can assess gaps from the current board, but I need the "
                    "wardrobe-gap context from the existing Style flow before "
                    "claiming a missing piece."
                )
            elif action_name == "critique_current_board":
                text = (
                    "I kept the current board intact. Visual critique is "
                    "shown when the beta vision flag is enabled and the item "
                    "images can be composed privately."
                )
            else:
                item_names = [
                    str(item.get("name") or item.get("role") or "item")
                    for item in beta_state.get("board_items") or []
                ]
                text = (
                    "This look is built around "
                    + ", ".join(item_names[:4])
                    + ". I am using the actual board item IDs, not regenerating it."
                )
            base = {
                "success": True,
                "ok": True,
                "type": "style_explanation",
                "message": {"role": "assistant", "content": text},
                "message_text": text,
                "response": text,
                "cards": [{
                    "id": beta_state.get("board_id"),
                    "occasion": beta_state.get("occasion"),
                    "items": beta_state.get("board_items") or [],
                }],
                "style_boards": [],
                "chips": [],
            }
            return _beta_style_response(
                base,
                previous_state=beta_state,
                instructions=beta_instructions,
            )
        if beta_instructions.get("action") in {
            "refine_current_board",
            "switch_source_mode",
        }:
            refined = beta_refine_style_response(
                state=beta_state,
                instructions=beta_instructions,
                candidate_pool=request.wardrobe,
            )
            # Never fall through to an unconstrained generator: an exact
            # mutation either succeeds by construction or fails honestly.
            if refined.get("success"):
                return _beta_style_response(
                    refined,
                    previous_state=beta_state,
                    instructions=beta_instructions,
                )
            refined.setdefault("style_state", beta_state)
            refined.setdefault("understood", beta_instructions)
            refined.setdefault("visual_intelligence", None)
            refined.setdefault("recommended_actions", [])
            return refined

    # ROUTE PRIORITY (P0): explicit wardrobe > multi_event_style > missing_pieces
    # > visual_inspiration > style_advice > plan_pack > birthday_workflow.
    # A multi-event style prompt ("office meeting then birthday party") must
    # never be hijacked by the plan_pack / birthday workflow below.
    try:
        from services.style_context_service import detect_multi_event as _detect_me

        _multi_event_route = _detect_me(user_input)
    except Exception:  # noqa: BLE001
        _multi_event_route = None
    _early_wardrobe_override = _is_use_wardrobe_action(
        action=request.action or request.style_action,
        prompt=user_input,
    ) or is_wardrobe_style_request(
        user_input,
        module_context=request.module_context or "",
    )
    if _multi_event_route and not _early_wardrobe_override:
        logger.info(
            "AHVI_ROUTE_PRIORITY_APPLIED winner=multi_event_style sub_occasions=%s style_strategy=%s",
            _multi_event_route.get("sub_occasions"),
            _multi_event_route.get("style_strategy"),
        )
        logger.info(
            "AHVI_STYLE_TRANSITION_DETECTED events=%s strategy=%s",
            _multi_event_route.get("sub_occasions"),
            _multi_event_route.get("style_strategy"),
        )
    elif _is_style_priority_query(user_input):
        logger.info("AHVI_ADAPTIVE_STYLE_ROUTER winner=style prompt=%r", user_input[:80])

    early_intent_row = detect_intent(user_input)
    early_intent = str(early_intent_row.get("intent") or "general").strip().lower()
    early_slots = early_intent_row.get("slots") if isinstance(early_intent_row.get("slots"), dict) else {}
    early_module = str(early_slots.get("module") or "").strip().lower()
    if early_intent == "organize_hub" and not _multi_event_route:
        domain = _organize_domain_for_module(early_module)
        logger.info(
            "organize_hub routed module=%s query=%s",
            early_module or domain,
            user_input[:100],
        )
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            early_intent,
            early_module,
            f"module:{domain}",
            user_input[:80],
        )
        return _run_module_chat_sync(
            {
                "domain": domain,
                "module": domain,
                "message": user_input,
                "context": {"user_profile": effective_user_profile},
                "user_profile": effective_user_profile,
            },
            user_id=user_id,
        )

    # -------------------------
    # MODULE SUMMARY CARD FAST PATH
    # -------------------------
    # "My medicines" etc. return the user's real Appwrite data as a
    # module_card, replacing the old hardcoded demo cards.
    _ms_module = _detect_module_summary(user_input)
    if _ms_module:
        from services.module_summary_service import build_module_summary

        _ms_card = build_module_summary(_ms_module, user_id)
        if _ms_card:
            logger.info(
                "chat.module_summary_route user_id=%s module=%s", user_id, _ms_module
            )
            return _ms_card

    if _is_plan_pack_request(user_input) and not _multi_event_route:
        logger.info(
            "chat.plan_pack_text_route user_id=%s prompt=%r", user_id, user_input
        )
        return _module_plan_pack_response(
            module_key=str(request.module_context or "planner"),
            user_message=user_input,
            context_data={"context_usage": resolved_context["context_usage"]},
            user_profile=effective_user_profile,
            user_id=user_id,
        )

    # Multi-event style: force the stylist reasoning path even when the prompt
    # has no explicit "outfit" word ("office meeting then birthday party").
    if _multi_event_route and not _early_wardrobe_override:
        _me_visual_first = _should_default_visual_inspiration(
            user_input,
            intent=STYLE_ADVICE,
            module_context=request.module_context or "",
            style_action=request.style_action or request.action or "",
            multi_event=_multi_event_route,
        )
        _me_mode = VISUAL_INSPIRATION if _me_visual_first else STYLE_ADVICE
        logger.info(
            "AHVI_VISUAL_FIRST_ROUTE endpoint=text toggle=%s selected_mode=%s intent=multi_event_style occasion=multi_event wardrobe_override=%s",
            _style_default_visual_inspiration_enabled(),
            _me_mode,
            _early_wardrobe_override,
        )
        _me_reasoning = style_reasoning_engine.reason(
            query=user_input,
            intent={"intent": _me_mode, "confidence": 0.9},
            user_profile=effective_user_profile,
            context={
                "module_context": request.module_context or "",
                "user_id": user_id,
                "occasion": "multi_event",
                "wardrobe": request.wardrobe if isinstance(request.wardrobe, list) else [],
                "multi_event": _multi_event_route,
                "sub_occasions": _multi_event_route.get("sub_occasions"),
                "style_strategy": _multi_event_route.get("style_strategy"),
                "style_action": _me_mode if _me_visual_first else "",
            },
        )
        logger.info(
            "chat.intent.route intent=multi_event_style module= path=style_reasoning text=%r",
            user_input[:80],
        )
        return _style_reasoning_chat_response(
            _me_reasoning,
            user_input,
            request.module_context or "chat",
            wardrobe=request.wardrobe if isinstance(request.wardrobe, list) else [],
        )

    # -------------------------
    # VISUAL BOARD FAST PATH
    # -------------------------
    # Diet / Pack / Plan prompts return a structured visual_board envelope
    # instead of plain text. Runs before the style orchestrator; style and
    # wardrobe module contexts are excluded inside _detect_visual_board_type.
    _vb_type = _detect_visual_board_type(user_input, request.module_context)
    # Adaptive style router: a style question that merely mentions a meal-time
    # word ("what should I wear for dinner") must not fall to the diet board.
    if _vb_type == "diet_plan" and _is_style_priority_query(user_input):
        logger.info("AHVI_STYLE_PRIORITY_GUARD_APPLIED prompt=%r", user_input[:80])
        logger.info("AHVI_DIET_FALSE_POSITIVE_BLOCKED prompt=%r", user_input[:80])
        _vb_type = ""
    if _vb_type:
        logger.info(
            "chat.visual_board_route user_id=%s board_type=%s prompt=%r",
            user_id, _vb_type, user_input,
        )
        return _build_visual_board_envelope(
            board_type=_vb_type,
            module_key=str(request.module_context or "chat"),
            user_message=user_input,
            context={"user_profile": effective_user_profile},
        )

    # -------------------------
    # FAST PATH
    # -------------------------
    if _is_fast_wardrobe_count_query(user_input):
        fast = _fast_wardrobe_count_response(user_id, user_input)
        fast["message"] = tone_engine.apply(
            str(fast.get("message") or ""),
            user_profile=effective_user_profile,
            signals={"context_mode": "home", "user_message_style": user_message_style},
            context={},
        )
        return fast

    # -------------------------
    # CACHE
    # -------------------------
    style_query = _is_explicit_style_request(user_input, request.module_context)
    style_action = str(request.style_action or "").strip().lower()
    closest_requested = (
        style_action == "show_closest_option"
        or bool(request.show_closest_option)
        or bool(request.allow_closest_option)
        or bool(request.closest)
    )
    if closest_requested:
        style_action = "show_closest_option"
        request.style_action = "show_closest_option"
        request.show_closest_option = True
        request.allow_closest_option = True
        request.closest = True
        logger.info(
            "style_closest_option_requested user_id=%s recovered_prompt=%r style_action=%s",
            user_id,
            user_input,
            style_action,
        )
    visual_context = (
        str(request.module_context or "").lower() in {"style", "wardrobe"}
        or style_query
        or bool(style_action)
    )
    cache_key = _cache_key(
        user_input,
        user_id,
        module_context=str(request.module_context or ""),
        wardrobe=request.wardrobe,
        occasion=_ahvi_style_occasion(user_input) if visual_context else "",
    )
    include_base64_for_chat = bool(request.include_base64 and _CHAT_INCLUDE_BASE64_ALLOWED)
    if request.include_base64 and not _CHAT_INCLUDE_BASE64_ALLOWED:
        logger.info(
            "chat.base64_ignored user_id=%s module=%s style_query=%s style_action=%s",
            user_id,
            request.module_context or "",
            bool(style_query),
            style_action or "",
        )
    cache_visual_boards = bool(
        (include_base64_for_chat or style_query or style_action) and visual_context
    )
    cached = None if cache_visual_boards else _CHAT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    logger.info(
        "chat.text_request user_id=%s module=%s visual=%s style_query=%s style_action=%s include_base64=%s",
        user_id,
        request.module_context or "",
        bool(visual_context),
        bool(style_query),
        style_action or "",
        bool(include_base64_for_chat),
    )

    # -------------------------
    # LANGUAGE
    # -------------------------
    try:
        preferred_lang = (request.language or "en").lower()

        if preferred_lang in ("te", "hi"):
            english_input = GoogleTranslator(
                source=preferred_lang, target="en"
            ).translate(user_input)
            target_lang = preferred_lang
        else:
            english_input = user_input
            target_lang = "en"

    except Exception:
        english_input = user_input
        target_lang = "en"

    # Restore style-pairing session: a bare typed follow-up ("use my wardrobe",
    # "show visual inspiration", "find missing pieces") after a pairing turn
    # carries no anchor. If the echoed last_style_context says we were pairing,
    # rewrite the prompt to include the selected route + anchor so the follow-up
    # builds the right thing instead of a generic outfit. CTA chips already
    # self-describe; this covers the typed path.
    _mem_early = request.current_memory if isinstance(request.current_memory, dict) else {}
    _lsc_early = _mem_early.get("last_style_context") if isinstance(_mem_early.get("last_style_context"), dict) else {}
    if _lsc_early.get("last_style_mode") == "style_pairing":
        _anchor_name = str((_lsc_early.get("anchor_item") or {}).get("name") or "").strip()
        _route = str(_lsc_early.get("selected_route") or "").strip()
        _ql = re.sub(r"\s+", " ", english_input.lower()).strip()
        _bare = _ql in {
            "use my wardrobe", "show visual inspiration", "find missing pieces",
            "visual inspiration", "missing pieces",
        }
        if _bare and _anchor_name and _anchor_name.lower() not in _ql:
            _suffix = f"{_route} with {_anchor_name}".strip() if _route else _anchor_name
            english_input = f"{english_input.strip()} to build {_suffix}"
            logger.info(
                "AHVI_PAIRING_CONTEXT_RESTORED anchor=%r route=%s rewritten=%r",
                _anchor_name, _route, english_input[:80],
            )
            if "wardrobe" in _ql:
                logger.info("AHVI_PAIRING_TO_WARDROBE_ROUTE anchor=%r route=%s", _anchor_name, _route)
            elif "visual" in _ql:
                logger.info("AHVI_PAIRING_VISUAL_FOLLOWUP anchor=%r route=%s", _anchor_name, _route)

    if _is_greeting(english_input):
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            "general",
            "",
            "greeting",
            english_input[:80],
        )
        return _ahvi_greeting_response(request.module_context)

    if _is_help_identity_request(english_input):
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            "general",
            "",
            "help_identity",
            english_input[:80],
        )
        return _ahvi_help_identity_response(english_input, request.module_context)

    if _is_small_talk(english_input):
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            "general",
            "",
            "small_talk",
            english_input[:80],
        )
        return _ahvi_small_talk_response(request.module_context)

    # Find This CTA (from missing-piece / inspiration cards). Route to a
    # shopping_intent placeholder — never generic style advice.
    if _is_find_this_request(english_input):
        response = _shopping_intent_response(english_input, user_profile=effective_user_profile)
        if not cache_visual_boards:
            _CHAT_CACHE.set(cache_key, response)
        return response

    if _is_use_wardrobe_action(action=request.action or request.style_action, prompt=english_input):
        forced_prompt = _wardrobe_action_prompt(english_input)
        forced_occasion = _ahvi_style_occasion(forced_prompt)
        logger.info(
            "ahvi.action.route action=%s occasion=%s forced_pipeline=%s prompt=%r resolved_prompt=%r",
            "use_wardrobe",
            forced_occasion,
            "outfit_pipeline",
            english_input[:100],
            forced_prompt[:100],
        )
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            WARDROBE_STYLE,
            "",
            "outfit_pipeline",
            forced_prompt[:80],
        )
        style_payload = _demo_style_board_payload(
            user_id,
            forced_prompt,
            request.wardrobe,
            effective_user_profile,
            style_action="use_wardrobe",
            show_closest_option=False,
            allow_closest_option=False,
            closest=False,
        )
        style_payload["success"] = bool(
            style_payload.get("cards")
            or style_payload.get("style_boards")
            or style_payload.get("missing_slots")
            or style_payload.get("shopping_gaps")
        )
        style_payload["style_boards"] = (
            style_payload.get("style_boards")
            or style_payload.get("cards")
            or []
        )
        style_payload.setdefault("chips", _ahvi_style_action_chips())
        style_payload.setdefault("data", {})
        style_payload["data"] = {
            **(style_payload.get("data") or {}),
            "intent": WARDROBE_STYLE,
            "style_mode": WARDROBE_STYLE,
            "forced_pipeline": "outfit_pipeline",
        }
        style_payload["meta"] = {
            **(style_payload.get("meta") or {}),
            "mode": "wardrobe_action",
            "action": "use_wardrobe",
            "style_mode": WARDROBE_STYLE,
            "forced_pipeline": "outfit_pipeline",
            "original_prompt": english_input,
            "resolved_prompt": forced_prompt,
            "board_type": "wardrobe_style",
        }
        return _beta_style_response(
            style_payload,
            previous_state=beta_state,
            instructions=beta_instructions,
            query=english_input,
            wardrobe=request.wardrobe,
            user_id=user_id,
            action=request.action or request.style_action,
        )

    intent_row = detect_intent(english_input)
    intent = str(intent_row.get("intent") or "general").strip().lower()
    slots = intent_row.get("slots") if isinstance(intent_row.get("slots"), dict) else {}
    detected_module = str(slots.get("module") or "").strip().lower()
    style_mode = intent if intent in STYLE_MODES else classify_style_mode(
        english_input,
        module_context=request.module_context or "",
        style_action=style_action,
    )
    visual_first = _should_default_visual_inspiration(
        english_input,
        intent=style_mode or intent,
        module_context=request.module_context or "",
        style_action=style_action,
    )
    wardrobe_override = _is_use_wardrobe_action(
        action=request.action or request.style_action,
        prompt=english_input,
    )
    selected_mode = VISUAL_INSPIRATION if visual_first else style_mode
    logger.info(
        "AHVI_VISUAL_FIRST_ROUTE endpoint=text toggle=%s selected_mode=%s intent=%s occasion=%s wardrobe_override=%s",
        _style_default_visual_inspiration_enabled(),
        selected_mode,
        intent,
        _ahvi_style_occasion(english_input),
        wardrobe_override,
    )
    reasoning_intent = (
        {"intent": VISUAL_INSPIRATION, "confidence": 0.95}
        if visual_first
        else intent_row
    )
    _mem = request.current_memory if isinstance(request.current_memory, dict) else {}
    _last_style_context = _mem.get("last_style_context") if isinstance(_mem.get("last_style_context"), dict) else {}
    if _last_style_context.get("last_style_mode") == "style_pairing":
        logger.info(
            "AHVI_PAIRING_CONTEXT_RESTORED anchor=%r route=%s archetypes=%s",
            (_last_style_context.get("anchor_item") or {}).get("name"),
            _last_style_context.get("selected_route"),
            _last_style_context.get("selected_archetypes"),
        )
    reasoning = style_reasoning_engine.reason(
        query=english_input,
        intent=reasoning_intent,
        user_profile=effective_user_profile,
        context={
            "module_context": request.module_context or "",
            "style_action": VISUAL_INSPIRATION if visual_first else style_action,
            "show_closest_option": closest_requested,
            # Real situational data for the Stylist Brain V2 context builder.
            "user_id": user_id,
            "occasion": _ahvi_style_occasion(english_input),
            "wardrobe": request.wardrobe if isinstance(request.wardrobe, list) else [],
            "weather": weather_for_consumers,
            "last_style_context": _last_style_context,
        },
        wardrobe_summary={
            "provided_count": len(request.wardrobe or [])
            if isinstance(request.wardrobe, list)
            else 0
        },
        history=_mem.get("history", []),
    )
    style_mode = str(reasoning.get("mode") or style_mode or "").strip().lower()

    # Default "complete outfit" CTA is a GENERATION request, not advice. Force
    # the wardrobe board path (so it flows through the universal completeness
    # gate) instead of the advice / visual-directions route that truncated and
    # fell back to unvalidated cards.
    if _is_generate_style_board_request(english_input):
        if isinstance(reasoning, dict):
            reasoning["should_generate_board"] = True
            reasoning["should_use_wardrobe"] = True
            if str(reasoning.get("mode") or "").strip().lower() in {
                STYLE_ADVICE,
                VISUAL_INSPIRATION,
                "style_advice",
            }:
                reasoning["mode"] = WARDROBE_STYLE
        style_mode = WARDROBE_STYLE
        logger.info(
            "AHVI_DEFAULT_CTA_ROUTE intent=style_request action=complete_outfit "
            "occasion=%s text=%r",
            _ahvi_style_occasion(english_input),
            english_input[:80],
        )

    if style_mode in {
        STYLE_ADVICE,
        COLOR_BODY_ADVICE,
        STYLE_EDUCATION,
        SHOPPING_ASSIST,
        VISUAL_INSPIRATION,
        STYLE_PAIRING,
        "body_proportion_advice",
        "color_advice",
        "occasion_advice",
    } and not reasoning.get("should_generate_board"):
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            style_mode,
            detected_module,
            "style_reasoning",
            english_input[:80],
        )
        response = _style_reasoning_chat_response(
            reasoning,
            english_input,
            request.module_context or "chat",
            wardrobe=request.wardrobe if isinstance(request.wardrobe, list) else [],
        )
        if not cache_visual_boards:
            _CHAT_CACHE.set(cache_key, response)
        return response

    if style_mode == WARDROBE_STYLE or reasoning.get("should_use_wardrobe"):
        intent = "occasion_outfit"
        visual_context = True
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            style_mode,
            detected_module,
            "wardrobe_style",
            english_input[:80],
        )

    if intent == "organize_hub":
        domain = _organize_domain_for_module(detected_module)
        logger.info(
            "organize_hub routed module=%s query=%s",
            detected_module or domain,
            english_input[:100],
        )
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            intent,
            detected_module,
            f"module:{domain}",
            english_input[:80],
        )
        return _run_module_chat_sync(
            {
                "domain": domain,
                "module": domain,
                "message": english_input,
                "context": {"user_profile": effective_user_profile},
                "user_profile": effective_user_profile,
            },
            user_id=user_id,
        )

    if intent == "plan_pack":
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            intent,
            detected_module,
            "plan_pack",
            english_input[:80],
        )
        return _module_plan_pack_response(
            module_key=str(request.module_context or "planner"),
            user_message=english_input,
            context_data={},
            user_profile=effective_user_profile,
            user_id=user_id,
        )

    if intent == "general":
        logger.info(
            "chat.intent.route intent=%s module=%s path=%s text=%r",
            intent,
            detected_module,
            "general",
            english_input[:80],
        )
        response = _llm_chat_response(
            messages=request.messages,
            english_input=english_input,
            user_id=user_id,
            user_profile=effective_user_profile,
            user_message_style=user_message_style,
            module_context=request.module_context,
        )
        if not cache_visual_boards:
            _CHAT_CACHE.set(cache_key, response)
        return response

    logger.info(
        "chat.intent.route intent=%s module=%s path=%s text=%r",
        intent,
        detected_module,
        "style_candidate",
        english_input[:80],
    )

    # Style clarification guard. Run for EVERY style-shaped prompt, not
    # only when visual_context is set, so vague 1-4 word prompts like
    # "beach wear" never enter the 8-20s orchestrator round-trip.
    # Earlier we keyed off _ahvi_style_occasion, but that helper doesn't
    # recognise "beach" / "gym" / "wedding" as occasion tokens, so
    # "beach wear" fell through to the LLM path and returned an empty
    # body, which the FE rendered as "I'm having trouble thinking
    # right now." The clarification helper itself already has a
    # precise short-prompt + broad-fashion-token + specificity check,
    # so just call it directly.
    style_intent_candidate = (
        style_mode == WARDROBE_STYLE
        or bool(style_action)
        or bool(closest_requested)
        or (request.module_context or "").lower() in {"wardrobe"}
        or _needs_style_clarification(english_input)
    )

    if style_intent_candidate:
        daily_wear_payload = _daily_wear_style_tips_payload(english_input, user_id)
        if daily_wear_payload:
            return daily_wear_payload
        try:
            style_interpretation = interpret_occasion(
                english_input,
                {
                    "module_context": request.module_context or "style",
                    "user_profile": effective_user_profile,
                    "wardrobe": request.wardrobe,
                    "weather": weather_for_consumers.get("condition"),
                },
            )
        except Exception:
            style_interpretation = {"board_generation_notes": {"occasion_kind": _ahvi_style_occasion(english_input)}}
        interpreted_occasion = (
            (style_interpretation.get("board_generation_notes") or {}).get("occasion_kind")
            or _ahvi_style_occasion(english_input)
        )
        resolved_occasion = (
            beta_instructions.get("occasion")
            or beta_state.get("occasion")
            or interpreted_occasion
        )
        # The predefined one-tap CTA ("Suggest an outfit for today.") is a
        # complete-outfit GENERATION request. It must never be intercepted by the
        # "What are you dressing for?" occasion clarification — generate now with
        # occasion=today.
        needs_clarify = _needs_style_clarification(
            english_input, interpreted_occasion
        ) and not _is_complete_outfit_cta(english_input)
        intent_status = "clarify" if needs_clarify else "generate"
        logger.info(
            "style_intent user_id=%s intent_status=%s prompt=%s interpreted_occasion=%s visual_context=%s",
            user_id,
            intent_status,
            english_input,
            interpreted_occasion,
            bool(visual_context),
        )
        if needs_clarify:
            logger.info(
                "style_clarification_triggered user_id=%s prompt=%r interpreted_occasion=%s reason=short_or_vague_style_prompt",
                user_id, english_input, interpreted_occasion,
            )
            return _style_clarification_response(english_input, style_interpretation)
        if visual_context or interpreted_occasion:
            logger.info(
                "style.fast_board_route user_id=%s prompt=%r interpreted_occasion=%s",
                user_id,
                english_input,
                interpreted_occasion,
            )
            style_payload = _demo_style_board_payload(
                user_id,
                english_input,
                request.wardrobe,
                effective_user_profile,
                resolved_occasion=resolved_occasion,
                style_action=style_action,
                show_closest_option=closest_requested,
                allow_closest_option=closest_requested,
                closest=closest_requested,
            )
            # Do not fall through to the heavy orchestrator for known style
            # requests. When no card survives the guards, the style service
            # already returns a wardrobe-gap/closest-option payload that is
            # more useful than the timeout fallback loop.
            style_payload["success"] = bool(
                style_payload.get("cards")
                or style_payload.get("style_boards")
                or style_payload.get("missing_slots")
                or style_payload.get("shopping_gaps")
            )
            style_payload["style_boards"] = (
                style_payload.get("style_boards")
                or style_payload.get("cards")
                or []
            )
            style_payload.setdefault("chips", _ahvi_style_action_chips())
            style_payload.setdefault("data", {})
            style_payload["meta"] = {
                **(style_payload.get("meta") or {}),
                "fast_style_route": True,
                "interpreted_occasion": interpreted_occasion,
            }
            # Phase 6: editorial wardrobe board, built from the user's ACTUAL
            # wardrobe items (real images), when this is a wardrobe-style ask.
            try:
                _is_wardrobe_ask = (
                    (request.module_context or "").lower() in {"wardrobe", "closet", "daily_wear"}
                    or is_wardrobe_style_request(
                        english_input,
                        module_context=request.module_context or "",
                        style_action=style_action,
                    )
                )
                _items = request.wardrobe if isinstance(request.wardrobe, list) else []
                if _is_wardrobe_ask and _items:
                    from services.style_context_service import (
                        build_style_context,
                        build_editorial_wardrobe_board,
                    )
                    from services.style_reasoning_engine import (
                        sanitize_board_items_for_visual_board,
                    )

                    _ctx = build_style_context(
                        query=english_input,
                        occasion=interpreted_occasion,
                        mode="wardrobe_style",
                        wardrobe_items=_items,
                        user_profile=effective_user_profile,
                    )
                    # Never dump raw wardrobe_items into a visual board. Repair
                    # into deduped, family-capped, slot-based items first. If the
                    # sanitizer can't assemble a viable outfit, skip the visual
                    # board entirely and let the text/cards stand.
                    _slots = sanitize_board_items_for_visual_board(
                        _ctx.get("wardrobe_items", []),
                        occasion=str(interpreted_occasion or ""),
                        style_direction=str(style_payload.get("message_text") or ""),
                    )
                    if _slots:
                        _occ_label = str(interpreted_occasion or "Your Look").replace("_", " ").title()
                        _board = build_editorial_wardrobe_board(
                            title=f"{_occ_label} — From Your Wardrobe",
                            goal=str((style_payload.get("meta") or {}).get("goal") or ""),
                            impression="",
                            stylist_reasoning=str(style_payload.get("message_text") or "").strip(),
                            wardrobe_items=_slots["items"],
                            palette=[],
                            why_it_works="",
                        )
                        if _board:
                            existing = style_payload.get("blocks")
                            style_payload["blocks"] = (
                                existing if isinstance(existing, list) else []
                            ) + [_board]
                            style_payload["data"]["editorial_wardrobe_board"] = _board
                    else:
                        logger.info(
                            "ahvi.editorial_board_skipped reason=no_viable_board occ=%r items=%d",
                            interpreted_occasion,
                            len(_items),
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ahvi.editorial_board_failed err=%s", str(exc)[:140])
            return _beta_style_response(
                style_payload,
                previous_state=beta_state,
                instructions=beta_instructions,
                query=english_input,
                wardrobe=request.wardrobe,
                user_id=user_id,
                action=request.action or request.style_action,
            )

    # -------------------------
    # GENERAL CHAT / LLM ROUTE
    # -------------------------
    # Must happen before orchestrator, because orchestrator can classify broad
    # style/fashion language as daily_outfit and return static wardrobe messages.
    if _is_general_chat_request(english_input, request.module_context):
        response = _llm_chat_response(
            messages=request.messages,
            english_input=english_input,
            user_id=user_id,
            user_profile=effective_user_profile,
            user_message_style=user_message_style,
            module_context=request.module_context,
        )
        if not cache_visual_boards:
            _CHAT_CACHE.set(cache_key, response)
        return response

    # -------------------------
    # HYBRID ROUTING
    # -------------------------
    mode = _detect_mode(english_input, request.module_context)

    if mode == "greeting":
        return _llm_chat_response(
            messages=request.messages,
            english_input=english_input,
            user_id=user_id,
            user_profile=effective_user_profile,
            user_message_style=user_message_style,
            module_context=request.module_context,
        )

    if mode == "casual" and not style_query:
        return _llm_chat_response(
            messages=request.messages,
            english_input=english_input,
            user_id=user_id,
            user_profile=effective_user_profile,
            user_message_style=user_message_style,
            module_context=request.module_context,
        )

    # -------------------------
    # WEATHER
    # -------------------------
    # -------------------------
    # ORCHESTRATOR (TIMEOUT SAFE)
    # -------------------------
    history = _build_history(request.messages[:-1]) if len(request.messages) > 1 else []
    memory_history = (
        request.current_memory.get("history", [])
        if isinstance(request.current_memory, dict)
        else []
    )
    merged_history = _normalize_memory_history(memory_history) + history

    def run():
        return ahvi_orchestrator.run(
            text=english_input,
            user_id=user_id,
            context={
                "memory": request.current_memory,
                "user_profile": effective_user_profile,
                "module_context": request.module_context,
                # Final style rendering is owned by style_flow_service after
                # card sanitization; avoid stale pre-rendered boards here.
                "include_base64": False,
                "wardrobe": request.wardrobe,
                "history": merged_history[-20:],
                "weather": weather_data.get("condition"),
                "weather_data": weather_data,
                "weather_context": weather_data,
                "location_context": resolved_context["location"],
                "context_usage": resolved_context["context_usage"],
                "time_of_day": weather_data.get("time_of_day"),
                "signals": {"user_message_style": user_message_style},
                "style_action": style_action,
                "show_closest_option": closest_requested,
                "allow_closest_option": closest_requested,
                "closest": closest_requested,
                "exclude_style_signatures": [
                    str(x or "").strip().lower()
                    for x in (request.exclude_style_signatures or [])
                    if str(x or "").strip()
                ],
                "requested_board_count": request.requested_board_count,
            },
        )

    _orch_started = time.time()
    logger.info(
        "chat.orchestrator_start user_id=%s prompt=%r timeout_seconds=%s module=%s",
        user_id, english_input, _ORCH_TIMEOUT_SECONDS, request.module_context or "",
    )
    try:
        result = _ORCHESTRATOR_EXECUTOR.submit(run).result(
            timeout=_ORCH_TIMEOUT_SECONDS
        )
        logger.info(
            "chat.orchestrator_success user_id=%s prompt=%r latency_ms=%s",
            user_id, english_input, int((time.time() - _orch_started) * 1000),
        )
    except concurrent.futures.TimeoutError:
        logger.warning(
            "chat.orchestrator_timeout user_id=%s prompt=%r timeout_seconds=%s",
            user_id, english_input, _ORCH_TIMEOUT_SECONDS,
        )
        # Softer recovery. Always carries a Try-again chip whose VALUE is
        # the original prompt, so the FE retry resends the same query
        # (not the literal word "Try again"). Plus directional chips so
        # the user can narrow without retyping.
        return _structured_error_response(
            code="ORCHESTRATOR_TIMEOUT",
            message="I could not finish the style board in time. Try again and I will reuse this same request.",
            status_type="provider_timeout",
            details={
                "timeout_seconds": _ORCH_TIMEOUT_SECONDS,
                "module_context": request.module_context or "",
                "original_prompt": english_input,
            },
            chips=[
                {"label": "Try again", "value": english_input},
                {"label": "Use my wardrobe", "value": f"{english_input} · Use my wardrobe"},
                {"label": "Make it casual", "value": f"{english_input} · casual"},
                {"label": "Make it polished", "value": f"{english_input} · polished"},
                {"label": "Ask me 2 questions", "value": f"{english_input} · ask me 2 questions"},
            ],
        )
    except Exception as exc:
        logger.exception("chat.orchestrator_exception user_id=%s prompt=%s", user_id, english_input)
        return _structured_error_response(
            code="style_generation_error" if visual_context else "chat_generation_error",
            message="AHVI hit a backend error while preparing this. Please try again in a moment.",
            status_type="backend_error",
            details={
                "error": str(exc)[:240],
                "module_context": request.module_context or "",
                "original_prompt": english_input,
            },
            chips=[{"label": "Try again", "value": english_input}],
        )

    message = result.get("message") or ""
    if isinstance(message, dict):
        message = str(message.get("content") or "")
    else:
        message = str(message or "")

    # -------------------------
    # TRANSLATE BACK
    # -------------------------
    try:
        if target_lang != "en" and message:
            lower_msg = message.strip().lower()
            if lower_msg in ("hi", "hello", "hey", "hi there", "hello there"):
                pass
            else:
                message = GoogleTranslator(source="en", target=target_lang).translate(
                    message
                )
    except Exception:
        pass

    # Final guardrail: every /api/text answer should leave through the tone layer,
    # even if a newer orchestrator branch forgot to apply it internally.
    try:
        message = tone_engine.apply(
            message,
            user_profile=effective_user_profile,
            signals={
                "context_mode": request.module_context or "chat",
                "user_message_style": user_message_style,
            },
            context={
                "module_context": request.module_context,
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
            },
        )
    except Exception:
        pass

    # If this is style chat and the orchestrator came back without a visual payload,
    # build a deterministic wardrobe board so the demo never lands as plain text only.
    data_payload = result.get("data") or {}
    cards_payload = result.get("cards") or []
    style_payload = {}

    # AHVI no visual boards on error responses:
    # If the orchestrator failed, do not attach deterministic fallback boards.
    # This avoids showing "temporary issue" text and outfit boards together.
    result_message_text = str(result.get("message") or "").lower()
    is_error_style_response = any(
        marker in result_message_text
        for marker in (
            "temporary issue",
            "please try again",
            "pipeline temporarily unavailable",
            "no outfits generated",
        )
    )

    if is_error_style_response:
        cards_payload = []
        data_payload = {"outfits": [], "rendered_boards": []}
        result["board_ids"] = ""
    else:
        has_visual_board = bool(
            isinstance(cards_payload, list) and cards_payload
        ) or bool(
            isinstance(data_payload, dict)
            and (data_payload.get("rendered_boards") or data_payload.get("outfits"))
        )
        if visual_context and not has_visual_board:
            style_payload = _demo_style_board_payload(
                user_id,
                english_input,
                request.wardrobe,
                effective_user_profile,
                style_action=style_action,
                show_closest_option=closest_requested,
                allow_closest_option=closest_requested,
                closest=closest_requested,
            )
            if style_payload.get("cards"):
                cards_payload = style_payload.get("cards") or []
                data_payload = style_payload.get("data") or {}
                result["type"] = style_payload.get("type") or result.get("type")
                result["board_ids"] = (
                    style_payload.get("board_ids") or result.get("board_ids") or ""
                )
            result["meta"] = {
                **(result.get("meta") or {}),
                **(style_payload.get("meta") or {}),
            }
        lower_message = (message or "").lower()
        if (
            not message
            or "clarification" in lower_message
            or "balance isn't quite" in lower_message
            or (
                style_payload.get("cards")
                and "i will assume smart casual" in lower_message
            )
        ):
            replacement = (
                style_payload.get("message")
                or "I will assume smart casual for today: start with a clean hero piece, add a neutral base, and finish with footwear or one accessory. Once your wardrobe has saved items, I will pick the exact pieces from it."
            )
            try:
                message = tone_engine.apply(
                    replacement,
                    user_profile=effective_user_profile,
                    signals={
                        "context_mode": request.module_context or "style",
                        "user_message_style": user_message_style,
                    },
                    context={"module_context": request.module_context},
                )
            except Exception:
                message = replacement

    # -------------------------
    # AUDIO
    # -------------------------
    try:
        audio_job_id = (
            enqueue_task(
                task_func=run_heavy_audio_task,
                args=[message, target_lang],
                kwargs={
                    "request_id": str(
                        getattr(http_request.state, "request_id", "") or ""
                    )
                },
                kind="chat_audio",
                user_id=user_id,
                source="routers.chat.text",
                request_id=str(getattr(http_request.state, "request_id", "") or ""),
            )
            if run_heavy_audio_task
            else "offline"
        )
    except Exception:
        audio_job_id = "offline"

    # -------------------------
    # FINAL RESPONSE
    # -------------------------
    if not isinstance(cards_payload, list):
        cards_payload = []

    board_ids_text = str(result.get("board_ids") or "")
    result_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    already_finalized_style_payload = (
        result_meta.get("mode") == "style_flow_service_adapter_v1"
        and isinstance(data_payload, dict)
        and bool(data_payload.get("outfits") or data_payload.get("rendered_boards"))
    )

    if (
        visual_context
        and isinstance(cards_payload, list)
        and cards_payload
        and not already_finalized_style_payload
    ):
        canonical_style = finalize_style_response_payload(
            {**result, "cards": cards_payload, "data": data_payload},
            user_id=user_id,
            query=english_input,
            context={
                "query": english_input,
                "occasion": _ahvi_style_occasion(english_input),
                "user_profile": effective_user_profile,
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
            },
            include_base64=include_base64_for_chat,
            style_action=style_action,
            show_closest_option=closest_requested,
            allow_closest_option=closest_requested,
            closest=closest_requested,
            exclude_style_signatures=request.exclude_style_signatures,
            requested_board_count=request.requested_board_count,
            cache_bypass=bool(cache_visual_boards),
        )
        cards_payload = canonical_style["cards"]
        data_payload = canonical_style["data"]
        board_ids_text = str(canonical_style.get("board_ids") or board_ids_text)
        result["meta"] = {
            **(result.get("meta") or {}),
            **(canonical_style.get("meta") or {}),
        }
        result["style_boards"] = canonical_style.get("style_boards") or cards_payload
    elif already_finalized_style_payload:
        result["style_boards"] = result.get("style_boards") or cards_payload
        logger.info(
            "chat.skip_duplicate_style_finalize user_id=%s cards=%d",
            user_id,
            len(cards_payload),
        )

    logger.info(
        "chat.text_response user_id=%s cards=%d signatures=%s style_action=%s profile_ms=%s",
        user_id,
        len(cards_payload),
        [style_card_signature(c) for c in cards_payload],
        style_action or "",
        profile_ms,
    )

    final_signatures = [
        style_card_signature(c) for c in cards_payload if isinstance(c, dict)
    ]
    style_signature = hashlib.sha1(
        "|".join(final_signatures).encode("utf-8")
    ).hexdigest() if final_signatures else ""

    response = {
        "success": True,
        "message": message,
        "board": result.get("board"),
        "type": result.get("type"),
        "style_boards": cards_payload if visual_context else [],
        "cards": cards_payload,
        "chips": (
            result.get("chips")
            if isinstance(result.get("chips"), list)
            else (
                style_payload.get("chips")
                if isinstance(style_payload.get("chips"), list)
                else (_ahvi_style_action_chips() if visual_context and cards_payload else [])
            )
        ),
        "board_ids": board_ids_text,
        "data": data_payload if isinstance(data_payload, dict) else {},
        "meta": {
            **(result.get("meta") or {}),
            "weather": weather_data,
            "history_used": len(merged_history[-20:]),
            "style_action": style_action or None,
            "style_signature": style_signature or None,
            "board_count": len(cards_payload),
            "has_more_style_options": bool(visual_context and len(cards_payload) > 0),
            "style_cache_bypass": bool(cache_visual_boards),
        },
        "audio_job_id": audio_job_id,
        "context_usage": resolved_context["context_usage"],
    }

    # -------------------------
    # CACHE SAVE
    # -------------------------
    if not cache_visual_boards:
        _CHAT_CACHE.set(cache_key, response)

    if visual_context and (
        response.get("cards") or response.get("style_boards")
    ):
        response = _beta_style_response(
            response,
            previous_state=beta_state,
            instructions=beta_instructions,
            query=english_input,
            wardrobe=request.wardrobe,
            user_id=user_id,
            action=request.action or request.style_action,
        )
    return response
