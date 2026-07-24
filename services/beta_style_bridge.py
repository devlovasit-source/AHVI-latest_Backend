"""Request-carried Style intelligence for the beta chat experience.

The bridge is deliberately persistence-free.  It normalizes client-carried
state, interprets follow-ups without another model call, validates existing
engine output, and optionally invokes the existing vision wrapper behind an
off-by-default flag.
"""

from __future__ import annotations

import json
import os
import re
import threading
import base64
import io
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from services.canonical_style_board import stable_hash
from services.style_item_contract import (
    canonical_item_id,
    canonical_item_role,
    canonical_item_source,
    normalize_style_item,
)
try:
    from brain.engines.outfit_quality_guard import reject_board_for_occasion
except Exception:  # pragma: no cover - optional in partial deployments
    reject_board_for_occasion = None

BETA_STYLE_BRIDGE_VERSION = "1.0"
VISUAL_INTELLIGENCE_VERSION = "1.0"
_TRUE = {"1", "true", "yes", "on"}
_ALLOWED_ACTIONS = {
    "create_board",
    "refine_current_board",
    "explain_current_board",
    "critique_current_board",
    "identify_wardrobe_gap",
    "switch_source_mode",
    "answer_style_question",
    "ask_clarification",
}
_ALLOWED_SOURCE_MODES = {"wardrobe", "wardrobe_only", "mixed", "style_asset", "unknown"}
_ROLE_TERMS = {
    "top": ("top", "shirt", "tee", "blouse", "sweater", "jacket", "blazer"),
    "bottom": ("bottom", "trouser", "pants", "jeans", "skirt", "shorts"),
    "footwear": ("footwear", "shoe", "shoes", "heels", "sneakers", "boots", "loafers"),
    "dress": ("dress", "gown", "jumpsuit"),
    "outerwear": ("outerwear", "coat", "jacket", "blazer"),
    "accessory": ("accessory", "bag", "belt", "watch", "jewelry", "scarf"),
}
_VISUAL_CACHE: Dict[str, Dict[str, Any]] = {}
_VISUAL_CACHE_LOCK = threading.Lock()
_VISUAL_INFLIGHT: Dict[str, threading.Event] = {}
_IMAGE_KEYS = ("image_url", "imageUrl", "normalized_url", "masked_url", "url")


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _string_list(value: Any, *, limit: int = 40) -> List[str]:
    raw = value if isinstance(value, (list, tuple, set)) else []
    out: List[str] = []
    for item in raw:
        text = _text(item)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _item_blob(item: Mapping[str, Any]) -> str:
    return " ".join(
        _text(item.get(key)).lower()
        for key in ("item_id", "id", "name", "label", "role", "category", "sub_category")
    )


def _safe_image_url(item: Mapping[str, Any]) -> str:
    for key in _IMAGE_KEYS:
        value = _text(item.get(key))
        if value.startswith(("https://", "http://", "asset:")):
            return value
    return ""


def normalize_style_state(raw: Any) -> Dict[str, Any]:
    """Return a compact, deterministic, image-free state envelope."""
    source = dict(raw) if isinstance(raw, Mapping) else {}
    items: List[Dict[str, Any]] = []
    for item in source.get("board_items") or source.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        normalized = normalize_style_item(dict(item))
        item_id = canonical_item_id(normalized)
        if not item_id or item_id in {canonical_item_id(existing) for existing in items}:
            continue
        items.append(
            {
                "item_id": item_id,
                "role": canonical_item_role(normalized),
                "source": canonical_item_source(normalized),
                **(
                    {"name": _text(normalized.get("name") or normalized.get("label"))}
                    if _text(normalized.get("name") or normalized.get("label"))
                    else {}
                ),
                **(
                    {"image_url": _safe_image_url(normalized)}
                    if _safe_image_url(normalized)
                    else {}
                ),
            }
        )
    item_ids = {item["item_id"] for item in items}
    locked = [item_id for item_id in _string_list(
        source.get("locked_item_ids") or source.get("preserved_item_ids")
    ) if item_id in item_ids]
    active = source.get("active_constraints") if isinstance(source.get("active_constraints"), Mapping) else {}
    preserve = [
        item_id for item_id in _string_list(active.get("preserve"))
        if item_id in item_ids
    ]
    excluded = _string_list(
        active.get("exclude") or source.get("excluded_terms"),
        limit=30,
    )
    source_mode = _text(source.get("source_mode") or "unknown").lower()
    if source_mode not in _ALLOWED_SOURCE_MODES:
        source_mode = "unknown"
    board_hash = _text(source.get("board_content_hash") or source.get("snapshot_hash"))
    if not re.fullmatch(r"[0-9a-f]{64}", board_hash):
        board_hash = stable_hash({"items": items, "occasion": _text(source.get("occasion")).lower()})
    return {
        "board_id": _text(source.get("board_id")) or None,
        "board_items": items,
        "occasion": _text(source.get("occasion")).lower() or None,
        "source_mode": source_mode,
        "hero_item_id": (
            _text(source.get("hero_item_id"))
            if _text(source.get("hero_item_id")) in item_ids
            else None
        ),
        "locked_item_ids": list(dict.fromkeys([*locked, *preserve])),
        "active_constraints": {
            "exclude": excluded,
            "preserve": list(dict.fromkeys([*preserve, *locked])),
        },
        "requested_change_scope": _text(source.get("requested_change_scope")) or None,
        "board_content_hash": board_hash,
    }


def _mentioned_roles(query: str) -> List[str]:
    found: List[str] = []
    for role, terms in _ROLE_TERMS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", query) for term in terms):
            found.append(role)
    return found


def _resolve_item_references(query: str, state: Mapping[str, Any]) -> List[str]:
    matched: List[str] = []
    for item in state.get("board_items") or []:
        if not isinstance(item, Mapping):
            continue
        item_id = canonical_item_id(dict(item))
        name = _text(item.get("name")).lower()
        role = canonical_item_role(dict(item))
        if (
            item_id.lower() in query
            or (name and name in query)
            or (role and any(term in query for term in _ROLE_TERMS.get(role, (role,))))
        ):
            matched.append(item_id)
    hero = _text(state.get("hero_item_id"))
    if hero and any(term in query for term in ("this", "that", "hero", "same", "keep it")):
        matched.append(hero)
    return list(dict.fromkeys(item for item in matched if item))


_CHANGE_TRIGGER_RE = re.compile(r"\b(?:change|replace|swap|remove|except|instead|different)\b")


def _split_keep_change(query: str) -> tuple[str, str]:
    """Split a follow-up into (keep_clause, change_clause) at the first change
    trigger. This keeps "keep the shirt and change only the shoes" from letting
    both clauses claim every mentioned role: the shirt stays in the keep clause,
    only the shoes fall in the change clause. No trigger -> all keep, no change.
    """
    match = _CHANGE_TRIGGER_RE.search(query)
    if not match:
        return query, ""
    return query[: match.start()], query[match.start() :]


_WARDROBE_ONLY_PHRASES = (
    "use only my wardrobe", "only my wardrobe", "wardrobe only", "wardrobe-only",
    "use my wardrobe", "use what i own", "what i own", "my own clothes",
    "own wardrobe", "no inspiration", "from my closet", "only what i have",
)
_INSPIRATION_PHRASES = (
    "inspiration option", "inspiration alternative", "inspiration shoes",
    "inspiration item", "inspiration piece", "add inspiration", "one inspiration",
    "shopping option", "style asset", "visual inspiration",
)


def interpret_style_followup(message: str, style_state: Any = None) -> Dict[str, Any]:
    """Extend existing deterministic routing with additive Style instructions."""
    query = _text(message).lower()
    state = normalize_style_state(style_state)
    has_state = bool(state["board_items"])
    roles = _mentioned_roles(query)
    referenced_ids = _resolve_item_references(query, state)
    exclusions: List[str] = []
    exclusion_query = re.sub(r"\s+n\s+", " and ", query)
    for match in re.finditer(
        r"(?:no|without|exclude|hate|remove|not)\s+([a-z][a-z -]{1,32}?)(?=$|[,.!?]|\s+(?:and|but|this|that|pls|please)\b)",
        exclusion_query,
    ):
        exclusions.extend(
            token.strip() for token in re.split(r"\s+(?:and|or)\s+|,", match.group(1))
            if token.strip()
        )
    exclusions = list(dict.fromkeys(exclusions))

    explain = any(term in query for term in ("why", "explain", "how does this work"))
    critique = any(term in query for term in ("visually weak", "what looks wrong", "critique", "boring"))
    gap = any(term in query for term in ("missing", "wardrobe gap", "need to buy", "upgrade piece"))
    wardrobe_only_req = any(term in query for term in _WARDROBE_ONLY_PHRASES)
    inspiration_req = "no inspiration" not in query and (
        "inspiration" in query
        or any(term in query for term in _INSPIRATION_PHRASES)
    )
    source_switch = wardrobe_only_req or inspiration_req
    refine = has_state and any(
        term in query
        for term in (
            "keep", "same", "change", "replace", "except", "everything else",
            "less serious", "more formal", "less formal", "make it", "no ", "without ",
        )
    )
    contradictory = wardrobe_only_req and inspiration_req
    query_words = query.strip(" .!?")
    ambiguous = has_state and query_words in {
        "make it better", "change it", "make it work", "something else",
    }

    if contradictory or ambiguous:
        action = "ask_clarification"
    elif critique:
        action = "critique_current_board"
    elif explain:
        action = "explain_current_board"
    elif gap:
        action = "identify_wardrobe_gap"
    elif source_switch and has_state:
        action = "switch_source_mode"
    elif refine:
        action = "refine_current_board"
    elif has_state and len(query.split()) <= 2:
        action = "ask_clarification"
    else:
        action = "create_board" if not has_state else "answer_style_question"

    preserve_roles: List[str] = []
    replace_roles: List[str] = []
    preserve_ids: List[str] = []
    replace_ids: List[str] = []
    # Segment the message so a keep-clause and a change-clause in the same
    # sentence do not both claim every mentioned role. Keep-targets come from
    # the keep clause; change-targets from the change clause.
    keep_seg, change_seg = _split_keep_change(query)
    keep_roles = _mentioned_roles(keep_seg)
    keep_ref_ids = _resolve_item_references(keep_seg, state)
    change_roles = _mentioned_roles(change_seg)
    change_ref_ids = _resolve_item_references(change_seg, state)
    if "keep everything except" in query:
        replace_roles = change_roles
        replace_ids = change_ref_ids
        replace_set = set(replace_ids)
        replace_role_set = set(replace_roles)
        preserve_ids = [
            item["item_id"] for item in state["board_items"]
            if item["item_id"] not in replace_set
            and item.get("role") not in replace_role_set
        ]
    elif "keep" in query or "same" in query:
        preserve_roles = keep_roles
        preserve_ids = keep_ref_ids
    if any(term in query for term in ("change", "replace", "remove", "everything else")):
        replace_roles = list(dict.fromkeys([*replace_roles, *change_roles]))
        if "everything else" not in query:
            replace_ids = list(dict.fromkeys([*replace_ids, *change_ref_ids]))

    current_items = state["board_items"]
    if "everything else" in query:
        preserved = set(preserve_ids)
        replace_roles = list(dict.fromkeys(
            item.get("role") for item in current_items
            if item.get("item_id") not in preserved and item.get("role")
        ))
    if "change only" in query or "replace only" in query:
        changed = set(replace_roles)
        preserve_ids = list(dict.fromkeys([
            *preserve_ids,
            *(
                item["item_id"] for item in current_items
                if item.get("role") not in changed
            ),
        ]))
    if exclusions:
        violating_roles = {
            item.get("role") for item in current_items
            if item.get("role") and any(term in _item_blob(item) for term in exclusions)
        }
        replace_roles = list(dict.fromkeys([*replace_roles, *violating_roles]))
        preserve_ids = list(dict.fromkeys([
            *preserve_ids,
            *(
                item["item_id"] for item in current_items
                if item.get("role") not in violating_roles
            ),
        ]))

    state_ids = {item["item_id"] for item in state["board_items"]}
    preserve_ids = [item_id for item_id in preserve_ids if item_id in state_ids]
    replace_ids = [item_id for item_id in replace_ids if item_id in state_ids]
    source_mode = state["source_mode"]
    if wardrobe_only_req and not inspiration_req:
        source_mode = "wardrobe_only"
    elif inspiration_req:
        source_mode = "mixed" if (wardrobe_only_req or "wardrobe" in query) else "style_asset"
    if action == "switch_source_mode":
        # Target only items that violate the requested source policy, so
        # already-compliant pieces are preserved and an already-compliant
        # board is affirmed unchanged downstream (no silent provenance mixing).
        def _violates_source(item: Mapping[str, Any]) -> bool:
            src = canonical_item_source(dict(item))
            if source_mode == "wardrobe_only":
                return src not in ("wardrobe", "owned")
            if source_mode == "style_asset":
                return src in ("wardrobe", "owned")
            return False
        violating_ids = [
            item["item_id"] for item in current_items if _violates_source(item)
        ]
        if violating_ids:
            replace_ids = list(dict.fromkeys([*replace_ids, *violating_ids]))
    formality_delta = (
        "increase" if any(term in query for term in ("more formal", "more serious", "professional", "dressier"))
        else "decrease" if any(term in query for term in ("less serious", "less formal", "more casual", "more relaxed", "more playful", "more fun", "fun"))
        else None
    )
    # A tone/formality-only request has no concrete role/item/exclusion scope,
    # so it cannot refine by construction. Ask one concise clarification rather
    # than returning bare "unsupported" or regenerating blindly.
    tone_only = (
        action == "refine_current_board"
        and formality_delta is not None
        and not preserve_ids
        and not replace_roles
        and not replace_ids
        and not exclusions
    )
    if tone_only:
        action = "ask_clarification"
    requires_visual = action in {"critique_current_board"} or "visually" in query
    needs_clarification = action == "ask_clarification"
    if not needs_clarification:
        clarification = None
    elif tone_only:
        clarification = (
            "Should I make it more casual, more colourful, or less formal "
            "while keeping the current outfit?"
        )
    elif contradictory:
        clarification = "Do you want a wardrobe-only look or one inspiration option?"
    else:
        clarification = "What should I preserve, and what should I change?"
    confidence = 0.98 if action in {"explain_current_board", "critique_current_board"} else (
        0.45 if needs_clarification else 0.84
    )
    return {
        "action": action,
        "preserve_item_ids": preserve_ids,
        "preserve_roles": preserve_roles,
        "replace_item_ids": replace_ids,
        "replace_roles": replace_roles,
        "excluded_terms": exclusions,
        "source_mode": source_mode,
        "occasion": state.get("occasion"),
        "formality_delta": formality_delta,
        "palette_direction": None,
        "weather_constraints": [],
        "requires_visual_analysis": requires_visual,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification,
        "confidence": confidence,
    }


def engine_dispatch(instructions: Mapping[str, Any]) -> Dict[str, Any]:
    action = _text(instructions.get("action"))
    source = _text(instructions.get("source_mode"))
    if action in {"explain_current_board", "critique_current_board"}:
        engine = "metadata_explanation"
    elif action == "identify_wardrobe_gap":
        engine = "style_context_service"
    elif instructions.get("preserve_item_ids") or instructions.get("preserve_roles"):
        engine = "fixed_item_style_flow"
    elif source in {"wardrobe", "wardrobe_only"}:
        engine = "style_flow_service"
    elif action == "refine_current_board":
        engine = "more_options"
    else:
        engine = "style_reasoning_engine"
    return {"engine": engine, "action": action, "source_mode": source or "unknown"}


def _response_boards(response: Mapping[str, Any]) -> List[Dict[str, Any]]:
    for key in ("cards", "style_boards"):
        value = response.get(key)
        if isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    data = response.get("data") if isinstance(response.get("data"), Mapping) else {}
    value = data.get("outfits")
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _board_items(board: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw: List[Any] = list(board.get("items") or [])
    for key in ("dress", "top", "bottom", "footwear", "shoes", "outerwear"):
        if isinstance(board.get(key), Mapping):
            raw.append(board[key])
    return [normalize_style_item(dict(item)) for item in raw if isinstance(item, Mapping)]


def visual_items_from_response(response: Mapping[str, Any]) -> List[Dict[str, Any]]:
    boards = _response_boards(response)
    return _board_items(boards[0]) if boards else []


def _candidate_allowed(
    item: Mapping[str, Any],
    *,
    role: str,
    excluded_terms: Sequence[str],
    source_mode: str,
    blocked_ids: set[str],
) -> bool:
    item_id = canonical_item_id(dict(item))
    if not item_id or item_id in blocked_ids:
        return False
    if canonical_item_role(dict(item)) != role:
        return False
    if source_mode == "wardrobe_only" and canonical_item_source(dict(item)) not in {
        "wardrobe", "owned",
    }:
        return False
    if source_mode == "style_asset" and canonical_item_source(dict(item)) in {
        "wardrobe", "owned",
    }:
        return False
    blob = _item_blob(item)
    return not any(term.lower() in blob for term in excluded_terms)


def _candidate_rank(
    item: Mapping[str, Any],
    retained: Sequence[Mapping[str, Any]],
    occasion: str,
) -> tuple[float, float, float, str]:
    metadata = item.get("style_metadata") if isinstance(item.get("style_metadata"), Mapping) else {}
    feedback_score = float(
        metadata.get("feedback_score")
        or item.get("feedback_score")
        or item.get("preference_score")
        or 0
    )
    try:
        from services.style_flow_service import _occasion_fit_score, _quality_score
        board = {"items": [*retained, dict(item)], "occasion": occasion}
        occasion_score = float(_occasion_fit_score(board, occasion))
        quality_score = float(_quality_score(board, occasion))
    except Exception:
        occasion_score = 0
        quality_score = 0
    return (
        feedback_score,
        occasion_score,
        quality_score,
        canonical_item_id(dict(item)),
    )


def _refined_board(
    state: Mapping[str, Any],
    instructions: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    candidate_offset: int,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    current = [dict(item) for item in state.get("board_items") or [] if isinstance(item, Mapping)]
    candidate_by_id = {
        canonical_item_id(dict(item)): dict(item)
        for item in candidates
        if isinstance(item, Mapping) and canonical_item_id(dict(item))
    }
    current = [
        {**item, **candidate_by_id.get(canonical_item_id(item), {})}
        for item in current
    ]
    current_ids = {canonical_item_id(item) for item in current}
    preserve = set(_string_list(instructions.get("preserve_item_ids")))
    replace_ids = set(_string_list(instructions.get("replace_item_ids")))
    replace_roles = set(_string_list(instructions.get("replace_roles")))
    if not preserve <= current_ids:
        return None, "preserved_item_not_in_current_board"
    if preserve & replace_ids:
        return None, "conflicting_preserve_and_replace"
    for item in current:
        if canonical_item_id(item) in preserve and canonical_item_role(item) in replace_roles:
            replace_roles.discard(canonical_item_role(item))
    if not replace_roles and not replace_ids:
        return None, "no_replacement_scope"

    retained = [
        item for item in current
        if canonical_item_id(item) not in replace_ids
        and canonical_item_role(item) not in replace_roles
    ]
    retained_ids = {canonical_item_id(item) for item in retained}
    if not preserve <= retained_ids:
        return None, "preserved_item_conflicts_with_replacement_scope"

    roles_needed = list(dict.fromkeys([
        *replace_roles,
        *(
            canonical_item_role(item) for item in current
            if canonical_item_id(item) in replace_ids
        ),
    ]))
    selected: List[Dict[str, Any]] = []
    blocked_ids = current_ids | retained_ids
    exclusions = _string_list(instructions.get("excluded_terms"))
    source_mode = _text(instructions.get("source_mode"))
    for role in roles_needed:
        eligible = [
            dict(item) for item in candidates
            if isinstance(item, Mapping) and _candidate_allowed(
                item,
                role=role,
                excluded_terms=exclusions,
                source_mode=source_mode,
                blocked_ids=blocked_ids | {
                    canonical_item_id(chosen) for chosen in selected
                },
            )
        ]
        eligible.sort(
            key=lambda item: _candidate_rank(
                item, retained, _text(instructions.get("occasion"))
            ),
            reverse=True,
        )
        if len(eligible) <= candidate_offset:
            return None, f"no_valid_{role}_replacement"
        selected.append(eligible[candidate_offset])
    final_items = [*retained, *selected]
    final_ids = {canonical_item_id(item) for item in final_items}
    if not preserve <= final_ids:
        return None, "preserved_item_missing_after_refinement"
    return {
        "id": state.get("board_id"),
        "occasion": state.get("occasion"),
        "items": final_items,
    }, None


def refine_style_response(
    *,
    state: Any,
    instructions: Mapping[str, Any],
    candidate_pool: Any,
) -> Dict[str, Any]:
    """Replace only requested roles using request-carried candidates.

    This is deliberately not a generator: it never retrieves candidates or
    invokes a model. One alternate candidate pass is allowed as repair.
    """
    normalized = normalize_style_state(state)
    candidates = [
        dict(item) for item in (candidate_pool or [])
        if isinstance(item, Mapping)
    ]
    # Already-satisfied request: nothing in the current board matches the
    # requested change scope (exclude a trait the board lacks, or "except X"
    # when there is no X). Affirm the board unchanged instead of failing.
    current_items = [
        dict(it) for it in normalized.get("board_items") or [] if isinstance(it, Mapping)
    ]
    _req_roles = set(_string_list(instructions.get("replace_roles")))
    _req_ids = set(_string_list(instructions.get("replace_item_ids")))
    _excluded = [t.lower() for t in _string_list(instructions.get("excluded_terms"))]

    def _is_change_target(item: Mapping[str, Any]) -> bool:
        return (
            canonical_item_role(dict(item)) in _req_roles
            or canonical_item_id(dict(item)) in _req_ids
            or (bool(_excluded) and any(t in _item_blob(item) for t in _excluded))
        )

    _target_mode = _text(instructions.get("source_mode"))

    def _source_complies(item: Mapping[str, Any]) -> bool:
        src = canonical_item_source(dict(item))
        if _target_mode == "wardrobe_only":
            return src in ("wardrobe", "owned")
        if _target_mode == "style_asset":
            return src not in ("wardrobe", "owned")
        return True

    _switch_satisfied = (
        _text(instructions.get("action")) == "switch_source_mode"
        and current_items
        and not _req_ids
        and not _req_roles
        and all(_source_complies(item) for item in current_items)
    )

    if _switch_satisfied or (
        current_items
        and (_req_roles or _req_ids or _excluded)
        and not any(_is_change_target(item) for item in current_items)
    ):
        affirmed: Dict[str, Any] = {
            "success": True,
            "ok": True,
            "type": "style_refinement_satisfied",
            "message_text": "Your board already meets that — nothing needed to change.",
            "response": "Your board already meets that — nothing needed to change.",
            "cards": [{
                "id": normalized.get("board_id"),
                "occasion": normalized.get("occasion"),
                "items": current_items,
            }],
            "style_boards": [],
            "chips": [],
        }
        affirmed["constraint_status"] = validate_style_response(affirmed, instructions)
        return affirmed
    board, error = _refined_board(
        normalized, instructions, candidates, candidate_offset=0
    )
    if board is None:
        return {
            "success": False,
            "ok": False,
            "type": "style_refinement_unsupported",
            "message_text": "I could not make that exact change without breaking your constraints.",
            "response": "I could not make that exact change without breaking your constraints.",
            "cards": [],
            "error": {"code": "UNSUPPORTED_BETA_REFINEMENT", "reason": error},
            "constraint_status": {
                "passed_constraints": [],
                "unresolved_constraints": [error or "refinement_unavailable"],
                "fallback_reason": error or "refinement_unavailable",
                "repair_attempted": False,
                "repair_reason": None,
                "repair_succeeded": False,
                "final_validation_status": "failed",
            },
        }
    response: Dict[str, Any] = {
        "success": True,
        "ok": True,
        "type": "style_refinement",
        "message_text": "I kept the confirmed pieces and changed only the requested roles.",
        "response": "I kept the confirmed pieces and changed only the requested roles.",
        "cards": [board],
        "style_boards": [],
        "chips": [],
    }
    first_status = validate_style_response(response, instructions)
    recoverable = bool(first_status["unresolved_constraints"])
    if recoverable:
        repaired, repair_error = _refined_board(
            normalized, instructions, candidates, candidate_offset=1
        )
        if repaired is not None:
            repaired_response = {**response, "cards": [repaired]}
            second_status = validate_style_response(repaired_response, instructions)
            second_status.update({
                "repair_attempted": True,
                "repair_reason": ",".join(first_status["unresolved_constraints"]),
                "repair_succeeded": second_status["final_validation_status"] == "passed",
            })
            repaired_response["constraint_status"] = second_status
            return repaired_response
        first_status.update({
            "repair_attempted": True,
            "repair_reason": ",".join(first_status["unresolved_constraints"]),
            "repair_succeeded": False,
            "repair_error": repair_error,
        })
    response["constraint_status"] = first_status
    return response


def validate_style_response(
    response: Mapping[str, Any],
    instructions: Mapping[str, Any],
) -> Dict[str, Any]:
    boards = _response_boards(response)
    items = _board_items(boards[0]) if boards else []
    ids = [canonical_item_id(item) for item in items if canonical_item_id(item)]
    roles = [canonical_item_role(item) for item in items]
    blobs = [_item_blob(item) for item in items]
    passed: List[str] = []
    unresolved: List[str] = []
    preserved = _string_list(instructions.get("preserve_item_ids"))
    missing = [item_id for item_id in preserved if item_id not in ids]
    (passed if not missing else unresolved).append("preserved_items")
    excluded = [term.lower() for term in _string_list(instructions.get("excluded_terms"))]
    violations = [term for term in excluded if any(term in blob for blob in blobs)]
    (passed if not violations else unresolved).append("exclusions")
    if _text(instructions.get("source_mode")) == "wardrobe_only":
        non_owned = [
            item for item in items
            if canonical_item_source(item) not in {"wardrobe", "owned"}
        ]
        (passed if not non_owned else unresolved).append("wardrobe_only")
    if len(ids) == len(set(ids)):
        passed.append("no_duplicates")
    else:
        unresolved.append("no_duplicates")
    if boards and items:
        passed.append("non_empty_board")
    else:
        unresolved.append("non_empty_board")
    invalid_roles = [role for role in roles if role in {"", "unknown"}]
    (passed if not invalid_roles else unresolved).append("valid_roles")
    role_set = set(roles)
    complete = ("dress" in role_set or {"top", "bottom"} <= role_set) and (
        "footwear" in role_set
    )
    (passed if complete else unresolved).append("outfit_completeness")
    unsafe_images = []
    for item in items:
        image = _text(
            item.get("image_url") or item.get("imageUrl")
            or item.get("normalized_url") or item.get("masked_url")
        )
        if image.startswith("data:") or (image and not re.match(r"^(https?://|asset:)", image)):
            unsafe_images.append(canonical_item_id(item))
    (passed if not unsafe_images else unresolved).append("renderable_images")
    if boards and reject_board_for_occasion is not None:
        rejected, reason = reject_board_for_occasion(
            boards[0],
            _text(instructions.get("occasion") or boards[0].get("occasion")),
        )
        (unresolved if rejected else passed).append("occasion_private_wear_guard")
    else:
        reason = ""
    return {
        "passed_constraints": list(dict.fromkeys(passed)),
        "unresolved_constraints": list(dict.fromkeys(unresolved)),
        "fallback_reason": (
            "constraints_not_fully_satisfied" if unresolved else None
        ),
        "repair_attempted": False,
        "final_validation_status": "passed" if not unresolved else "partial",
        "missing_preserved_item_ids": missing,
        "excluded_term_violations": violations,
        "guard_rejection_reason": reason or None,
    }


def state_from_response(
    response: Mapping[str, Any],
    previous_state: Any = None,
    instructions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    previous = normalize_style_state(previous_state)
    boards = _response_boards(response)
    board = boards[0] if boards else {}
    items = _board_items(board)
    compact = [
        {
            "item_id": canonical_item_id(item),
            "role": canonical_item_role(item),
            "source": canonical_item_source(item),
            **({"name": _text(item.get("name") or item.get("label"))} if _text(item.get("name") or item.get("label")) else {}),
        }
        for item in items if canonical_item_id(item)
    ]
    state = {
        "board_id": _text(
            board.get("board_id") or board.get("id") or response.get("board_ids")
        ) or previous.get("board_id"),
        "board_items": compact or previous["board_items"],
        "occasion": _text(
            board.get("occasion") or response.get("occasion") or previous.get("occasion")
        ).lower() or None,
        "source_mode": _text(
            (instructions or {}).get("source_mode") or previous.get("source_mode")
        ).lower() or "unknown",
        "hero_item_id": (
            canonical_item_id(compact[0]) if compact else previous.get("hero_item_id")
        ),
        "locked_item_ids": list(dict.fromkeys(
            _string_list((instructions or {}).get("preserve_item_ids"))
            or previous.get("locked_item_ids") or []
        )),
        "active_constraints": {
            "exclude": _string_list((instructions or {}).get("excluded_terms")),
            "preserve": _string_list((instructions or {}).get("preserve_item_ids")),
        },
        "requested_change_scope": _text((instructions or {}).get("action")) or None,
    }
    state["board_content_hash"] = stable_hash(
        {"items": state["board_items"], "occasion": state["occasion"]}
    )
    return normalize_style_state(state)


def _valid_visual_payload(payload: Any, allowed_ids: Iterable[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return None
    allowed = {str(item_id) for item_id in allowed_ids if str(item_id)}
    out = {
        "summary": _text(payload.get("summary")),
        "hero_item_id": _text(payload.get("hero_item_id")) or None,
        "strongest_item_id": _text(payload.get("strongest_item_id")) or None,
        "strongest_reason": _text(payload.get("strongest_reason")),
        "weakest_item_id": _text(payload.get("weakest_item_id")) or None,
        "weakest_reason": _text(payload.get("weakest_reason")),
        "color_story": _text(payload.get("color_story")),
        "silhouette_story": _text(payload.get("silhouette_story")),
        "occasion_fit": _text(payload.get("occasion_fit")),
        "weather_fit": _text(payload.get("weather_fit")),
        "missing_or_upgrade_piece": _text(payload.get("missing_or_upgrade_piece")),
        "recommended_actions": _string_list(payload.get("recommended_actions"), limit=3),
        "confidence": payload.get("confidence"),
        "version": VISUAL_INTELLIGENCE_VERSION,
    }
    for key in ("hero_item_id", "strongest_item_id", "weakest_item_id"):
        if out[key] and out[key] not in allowed:
            return None
    return out if out["summary"] else None


def visual_intelligence(
    *,
    state: Mapping[str, Any],
    image_base64: str = "",
    visual_items: Sequence[Mapping[str, Any]] = (),
    requested: bool,
    vision_call: Optional[Callable[..., Any]] = None,
    image_loader: Optional[Callable[[str], bytes]] = None,
) -> Optional[Dict[str, Any]]:
    if not requested or str(os.getenv("AHVI_BETA_STYLE_VISION_ENABLED", "false")).lower() not in _TRUE:
        return None
    normalized = normalize_style_state(state)
    board_hash = normalized["board_content_hash"]
    with _VISUAL_CACHE_LOCK:
        cached = _VISUAL_CACHE.get(board_hash)
        if cached is not None:
            return dict(cached)
        event = _VISUAL_INFLIGHT.get(board_hash)
        if event is None:
            event = threading.Event()
            _VISUAL_INFLIGHT[board_hash] = event
            owns_call = True
        else:
            owns_call = False
    if not owns_call:
        event.wait(8.25)
        with _VISUAL_CACHE_LOCK:
            cached = _VISUAL_CACHE.get(board_hash)
        return dict(cached) if cached is not None else None
    try:
        started = time.monotonic()
        if not _text(image_base64):
            image_base64 = private_board_image_base64(
                visual_items or normalized["board_items"],
                image_loader=image_loader,
            )
        if not image_base64:
            return None
        remaining_seconds = max(0, 8 - (time.monotonic() - started))
        if remaining_seconds < 1:
            return None
        if vision_call is None:
            from services.ai_gateway import ollama_vision_json
            vision_call = ollama_vision_json
        item_ids = [item["item_id"] for item in normalized["board_items"]]
        item_metadata = [
            {"item_id": item["item_id"], "role": item.get("role")}
            for item in normalized["board_items"]
        ]
        prompt = (
            "Return JSON visual styling critique. Use only this item metadata: "
            f"{json.dumps(item_metadata)}. Do not infer identity, age, attractiveness, "
            "body shape, health, ownership, or sensitive traits. Fields: summary, "
            "hero_item_id, strongest_item_id, strongest_reason, weakest_item_id, "
            "weakest_reason, color_story, silhouette_story, occasion_fit, "
            "weather_fit, missing_or_upgrade_piece, recommended_actions, confidence."
        )
        result = vision_call(
            prompt=prompt,
            image_base64=image_base64,
            timeout_seconds=max(1, int(remaining_seconds)),
            usecase="beta_style_visual_intelligence",
        )
        payload = result[0] if isinstance(result, tuple) else result
        valid = _valid_visual_payload(payload, item_ids)
    except Exception:
        return None
    finally:
        with _VISUAL_CACHE_LOCK:
            if "valid" in locals() and valid is not None:
                _VISUAL_CACHE[board_hash] = dict(valid)
            _VISUAL_INFLIGHT.pop(board_hash, None)
            event.set()
    return valid if valid is not None else None


def private_board_image_base64(
    items: Sequence[Mapping[str, Any]],
    *,
    image_loader: Optional[Callable[[str], bytes]] = None,
) -> str:
    """Compose a private 512px JPEG in memory; never writes or persists it."""
    from PIL import Image, ImageOps

    rows = [
        (canonical_item_id(dict(item)), canonical_item_role(dict(item)), _safe_image_url(item))
        for item in items if isinstance(item, Mapping)
    ]
    rows = [row for row in rows if row[0] and row[2]][:4]
    if not rows:
        return ""
    prefetched: Dict[str, bytes] = {}
    if image_loader is None:
        def fetch(url: str) -> bytes:
            import ipaddress
            import socket
            from urllib.parse import urlparse
            import requests
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("untrusted image URL")
            addresses = {
                row[4][0] for row in socket.getaddrinfo(parsed.hostname, 443)
            }
            if not addresses or any(
                ipaddress.ip_address(address).is_private
                or ipaddress.ip_address(address).is_loopback
                or ipaddress.ip_address(address).is_link_local
                or ipaddress.ip_address(address).is_reserved
                for address in addresses
            ):
                raise ValueError("private image host rejected")
            response = requests.get(
                url,
                timeout=(1, 2),
                allow_redirects=False,
                headers={"Accept": "image/*"},
            )
            response.raise_for_status()
            if not _text(response.headers.get("Content-Type")).lower().startswith("image/"):
                raise ValueError("non-image response")
            content = response.content
            if len(content) > 2 * 1024 * 1024:
                raise ValueError("image exceeds private-analysis limit")
            return content
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(rows)) as executor:
            futures = {url: executor.submit(fetch, url) for _, _, url in rows}
            for url, future in futures.items():
                try:
                    prefetched[url] = future.result(timeout=2.25)
                except Exception:
                    continue
        image_loader = lambda url: prefetched[url]
    canvas = Image.new("RGB", (512, 512), (248, 246, 242))
    positions = ((8, 8), (260, 8), (8, 260), (260, 260))
    loaded = 0
    for (_, _, url), position in zip(rows, positions):
        try:
            raw = image_loader(url)
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.width > 4096 or opened.height > 4096:
                    raise ValueError("image dimensions exceed private-analysis limit")
                tile = ImageOps.fit(opened.convert("RGB"), (244, 244))
            canvas.paste(tile, position)
            loaded += 1
        except Exception:
            continue
    if not loaded:
        return ""
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=70, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decorate_style_response(
    response: Mapping[str, Any],
    *,
    previous_state: Any,
    instructions: Mapping[str, Any],
    visual: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out = dict(response)
    state = state_from_response(out, previous_state, instructions)
    status = (
        dict(out["constraint_status"])
        if isinstance(out.get("constraint_status"), Mapping)
        else validate_style_response(out, instructions)
    )
    out["style_state"] = state
    out["understood"] = {
        "action": instructions.get("action"),
        "occasion": instructions.get("occasion") or state.get("occasion"),
        "source_mode": instructions.get("source_mode") or state.get("source_mode"),
        "preserve_item_ids": list(instructions.get("preserve_item_ids") or []),
        "replace_roles": list(instructions.get("replace_roles") or []),
        "excluded_terms": list(instructions.get("excluded_terms") or []),
        "confidence": instructions.get("confidence"),
    }
    out["constraint_status"] = status
    out["visual_intelligence"] = dict(visual) if isinstance(visual, Mapping) else None
    out["recommended_actions"] = (
        list(visual.get("recommended_actions") or [])[:3]
        if isinstance(visual, Mapping)
        else []
    )
    out.setdefault("meta", {})
    if isinstance(out["meta"], Mapping):
        out["meta"] = {**out["meta"], "beta_style_bridge_version": BETA_STYLE_BRIDGE_VERSION}
    return out
