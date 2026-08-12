"""Bounded conversational operations on the current Style Board.

This module is intentionally deterministic.  Conversation routing may decide
that a message is a board operation, but this service is the only place that
can choose replacement items or advance a board revision.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from brain.engines.outfit_quality_guard import is_complete_board
from services.canonical_style_board import stable_hash
from services.style_item_contract import (
    canonical_item_id,
    canonical_item_role,
    canonical_item_source,
    normalize_style_item,
)
from services.style_asset_contract import adapt_style_asset
from services.style_explicit_roles import item_explicit_role
from services.style_board_state_store import (
    BoardRevisionExistsError,
    BoardStateStoreError,
)


BOARD_OPERATIONS = frozenset(
    {"modify_current_look", "generate_alternative", "explain_current_look"}
)

def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _items(value: Any) -> List[Dict[str, Any]]:
    return [deepcopy(dict(item)) for item in value or [] if isinstance(item, Mapping)]


def _board_items(board: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = board.get("board_items") or board.get("items") or []
    return _items(raw)


def _content_hash(items: Iterable[Mapping[str, Any]], occasion: Any = "") -> str:
    return stable_hash({"items": [dict(item) for item in items], "occasion": _text(occasion).lower()})


def _find_state(value: Any, *, depth: int = 0) -> Optional[Dict[str, Any]]:
    if depth > 3 or not isinstance(value, Mapping):
        return None
    for key in ("style_state", "current_look", "current_board", "board"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            found = _find_state(nested, depth=depth + 1)
            if found:
                return found
    if value.get("board_id") or value.get("boardId"):
        items = _board_items(value)
        if items:
            state = deepcopy(dict(value))
            state["board_items"] = items
            return state
    return None


def _candidate_pool(payload: Mapping[str, Any], state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    values: List[Any] = []
    for source in (payload, payload.get("context"), payload.get("context_data"), payload.get("current_memory")):
        if not isinstance(source, Mapping):
            continue
        for key in ("candidate_items", "candidates", "wardrobe", "wardrobe_items", "style_assets", "items"):
            candidate = source.get(key)
            if isinstance(candidate, list):
                values.extend(candidate)
    for key in ("candidate_items", "candidates", "wardrobe", "wardrobe_items", "style_assets"):
        candidate = state.get(key)
        if isinstance(candidate, list):
            values.extend(candidate)
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping):
            continue
        normalized = deepcopy(dict(item))
        item_id = canonical_item_id(normalized)
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        result.append(normalized)
    return result


def _state_from_payload(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    sources = [
        payload,
        payload.get("context"),
        payload.get("context_data"),
        payload.get("style_context"),
        payload.get("current_memory"),
    ]
    for source in sources:
        state = _find_state(source)
        if state:
            state.setdefault("candidate_items", _candidate_pool(payload, state))
            return state
    return None


def board_context_for_semantics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return safe board facts for semantic interpretation, never item IDs."""
    state = _state_from_payload(payload) if isinstance(payload, Mapping) else None
    if not state:
        return {"has_current_board": False, "items": []}
    items = _board_items(state)
    protected, _invalid = _protected_ids(state, items)
    return {
        "has_current_board": True,
        "interaction_mode": _text(state.get("interaction_mode") or state.get("mode")),
        "scenario": _text(state.get("scenario")),
        "items": [
            {
                "role": _request_role(item),
                "name": _text(item.get("name") or item.get("label")) or _request_role(item),
                "protected": canonical_item_id(item) in protected or bool(item.get("locked")),
            }
            for item in items
        ],
    }


def _protected_ids(state: Mapping[str, Any], items: Iterable[Mapping[str, Any]]) -> Tuple[set[str], List[str]]:
    item_ids = {canonical_item_id(dict(item)) for item in items if canonical_item_id(dict(item))}
    protected: set[str] = set()
    for key in ("locked_item_ids", "preserved_item_ids", "protected_item_ids", "anchor_item_ids"):
        raw = state.get(key)
        if isinstance(raw, (list, tuple, set)):
            protected.update(_text(item) for item in raw if _text(item))
    active = state.get("active_constraints")
    if isinstance(active, Mapping):
        raw = active.get("preserve") or active.get("locked") or active.get("protected") or []
        if isinstance(raw, (list, tuple, set)):
            protected.update(_text(item) for item in raw if _text(item))
    for item in items:
        item_id = canonical_item_id(dict(item))
        if item_id and (item.get("locked") or item.get("protected") or item.get("preserved")):
            protected.add(item_id)
    invalid = sorted(protected - item_ids)
    return protected & item_ids, invalid


def _structured_operation(decision: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize the already-validated semantic resolver decision.

    No user text is interpreted here. Concrete item IDs are intentionally
    absent; the executor resolves roles against authoritative board state.
    """
    if not isinstance(decision, Mapping):
        return None
    operation = decision.get("operation")
    if not isinstance(operation, Mapping):
        return None
    operation_type = _text(operation.get("type")).lower()
    if operation_type not in {"modify", "generate_alternative", "explain_current_look"}:
        return None
    constraints = [
        dict(item) for item in operation.get("constraints", [])
        if isinstance(item, Mapping)
    ]
    top_constraints = decision.get("constraints") if isinstance(decision.get("constraints"), Mapping) else {}
    for value in top_constraints.get("avoid", []) if isinstance(top_constraints.get("avoid"), list) else []:
        constraints.append({"dimension": "style", "operator": "avoid", "value": _text(value).lower()})
    resolved_context = decision.get("resolved_context") if isinstance(decision.get("resolved_context"), Mapping) else {}
    for key, operator in (("negative_constraints", "avoid"), ("positive_constraints", "require"), ("style_constraints", "require")):
        values = resolved_context.get(key)
        if isinstance(values, list):
            constraints.extend(
                {"dimension": "style", "operator": operator, "value": _text(value).lower()}
                for value in values if _text(value)
            )
    adjustments = dict(operation.get("style_adjustments") or {})
    return {
        "operation": {
            "modify": "modify_current_look",
            "generate_alternative": "generate_alternative",
            "explain_current_look": "explain_current_look",
        }[operation_type],
        "target_roles": list(dict.fromkeys(
            _text(role).lower() for role in (
                list(operation.get("replace_roles") or [])
                + list(operation.get("remove_roles") or [])
            ) if _text(role)
        )),
        "preserve_roles": list(dict.fromkeys(
            _text(role).lower() for role in (operation.get("preserve_roles") or []) if _text(role)
        )),
        "constraints": constraints,
        "style_adjustments": adjustments,
        "alternative_scope": operation.get("alternative_scope") or "default",
        "explanation_target": _text(operation.get("explanation_target")),
        "requires_clarification": bool(decision.get("requires_clarification")),
        "referent": decision.get("referent") if isinstance(decision.get("referent"), Mapping) else None,
        "resolved_context": decision.get("resolved_context") if isinstance(decision.get("resolved_context"), Mapping) else {},
    }


def _source_allowed(item: Mapping[str, Any], policy: str) -> bool:
    source = canonical_item_source(dict(item))
    policy = _text(policy).lower()
    if policy in {"wardrobe", "wardrobe_only"}:
        return source == "wardrobe"
    if policy in {"style_asset", "catalog"}:
        return source in {"style_asset", "catalog"}
    if policy == "mixed":
        return source in {"wardrobe", "style_asset", "catalog"}
    return source != "unknown"


def _candidate_with_safe_provenance(item: Mapping[str, Any], policy: str) -> Optional[Dict[str, Any]]:
    """Normalize a replacement through the existing Style image contract."""
    source = canonical_item_source(dict(item))
    if not _source_allowed(item, policy):
        return None
    candidate = dict(item)
    raw_markers = ("raw_url", "raw_image_url", "original_upload_url", "upload_url")
    safe_keys = (
        "board_image_url", "boardImageUrl", "cutout_url", "cutoutUrl",
        "normalized_url", "normalizedUrl", "masked_url", "maskedUrl",
        "catalog_image_url", "catalogImageUrl", "image_url", "imageUrl",
    )
    if any(_text(candidate.get(key)) for key in raw_markers) and not any(
        _text(candidate.get(key)) for key in safe_keys if key not in raw_markers
    ):
        return None
    if candidate.get("owned") is False or (
        _text(candidate.get("owner_id") or candidate.get("user_id"))
        and _text(candidate.get("owner_id") or candidate.get("user_id")) != _text(candidate.get("_request_user_id"))
    ):
        return None
    if source == "style_asset":
        candidate = adapt_style_asset(candidate)
        selected = _text(candidate.get("selected_field"))
        if selected not in {"board_image_url", "cutout_url", "catalog_image_url", "normalized_url"}:
            return None
    normalized = normalize_style_item(candidate)
    if not canonical_item_id(normalized) or canonical_item_role(normalized) == "unknown":
        return None
    normalized.update({
        key: value for key, value in candidate.items()
        if key in {
            "board_image_url", "cutout_url", "catalog_image_url", "normalized_url",
            "masked_url", "source_kind", "expected_transparent", "requires_frame",
            "position",
        } and value not in (None, "")
    })
    return normalized


def _constraint_exclusions(constraints: Iterable[Mapping[str, Any]]) -> List[str]:
    values: List[str] = []
    for constraint in constraints:
        if not isinstance(constraint, Mapping) or _text(constraint.get("operator")).lower() != "avoid":
            continue
        value = _text(constraint.get("value")).lower()
        if value and value not in values:
            values.append(value)
    return values[:8]


def _role_matches(item: Mapping[str, Any], target_role: str) -> bool:
    if target_role == "bag":
        return item_explicit_role(dict(item)) == "bag"
    return canonical_item_role(dict(item)) == target_role


def _request_role(item: Mapping[str, Any]) -> str:
    """Keep bag targeting distinct from the generic accessory bucket."""
    return "bag" if item_explicit_role(dict(item)) == "bag" else canonical_item_role(dict(item))


def _candidate_score(
    item: Mapping[str, Any],
    *,
    adjustments: Mapping[str, Any],
    constraints: List[Mapping[str, Any]],
) -> Optional[int]:
    blob = " ".join(_text(item.get(key)).lower() for key in (
        "name", "label", "category", "sub_category", "color", "colour", "footwear_type",
        "style", "tags", "style_direction", "material", "fit",
    ))
    if any(value in blob for value in _constraint_exclusions(constraints)):
        return None
    score = 0
    for constraint in constraints:
        if _text(constraint.get("operator")).lower() == "require":
            value = _text(constraint.get("value")).lower()
            if (
                _text(constraint.get("dimension")).lower()
                in {"color", "footwear_type"}
                and value
                and value not in blob
            ):
                return None
            if value and value in blob:
                score += 4
    adjustment_values = {
        {
            "lower": "casual",
            "raise": "formal",
        }.get(_text(value).lower(), _text(value).lower())
        for value in (adjustments or {}).values() if _text(value)
    }
    if adjustment_values:
        wanted = {
            "casual": ("casual", "relaxed", "denim", "sneaker", "tee", "knit"),
            "relaxed": ("casual", "relaxed", "denim", "sneaker", "tee", "knit"),
            "formal": ("formal", "tailored", "dress", "blazer", "loafer", "leather"),
            "polished": ("formal", "tailored", "dress", "blazer", "loafer", "leather"),
            "sporty": ("sport", "performance", "training", "sneaker", "athletic"),
            "playful": ("colour", "color", "print", "pattern", "bright"),
            "colorful": ("colour", "color", "print", "pattern", "bright"),
        }
        for value in adjustment_values:
            score += sum(3 for token in wanted.get(value, ()) if token in blob)
    return score


def _choose_candidate(
    candidates: Iterable[Mapping[str, Any]],
    *,
    role: str,
    current_id: str,
    used_ids: set[str],
    policy: str,
    adjustments: Mapping[str, Any],
    constraints: List[Mapping[str, Any]],
    user_id: str,
    seed: str,
) -> Optional[Dict[str, Any]]:
    ranked: List[Tuple[int, str, Dict[str, Any]]] = []
    for candidate in candidates:
        item = dict(candidate)
        item_id = canonical_item_id(item)
        if not item_id or item_id == current_id or item_id in used_ids:
            continue
        item["_request_user_id"] = user_id
        item = _candidate_with_safe_provenance(item, policy)
        if item is None or not _role_matches(item, role):
            continue
        score = _candidate_score(
            item,
            adjustments=adjustments,
            constraints=constraints,
        )
        if score is None:
            continue
        ranked.append((score, stable_hash({"seed": seed, "item_id": item_id}), item))
    if not ranked:
        return None
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return deepcopy(ranked[0][2])


def _policy(state: Mapping[str, Any], items: Iterable[Mapping[str, Any]]) -> str:
    raw = _text(state.get("source_policy") or state.get("source_mode")).lower()
    if raw in {"wardrobe_only", "wardrobe", "style_asset", "mixed", "catalog"}:
        return raw
    sources = {canonical_item_source(dict(item)) for item in items}
    if sources == {"wardrobe"}:
        return "wardrobe"
    if sources <= {"style_asset", "catalog"} and sources:
        return "style_asset"
    return "mixed"


def _board_id(state: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    return _text(
        state.get("board_id") or state.get("boardId") or payload.get("current_look_id")
        or payload.get("board_id") or payload.get("boardId")
    )


def _revision(state: Mapping[str, Any], payload: Mapping[str, Any]) -> int:
    raw = (
        state.get("revision") or state.get("board_revision") or state.get("expected_revision")
        or payload.get("revision") or payload.get("board_revision") or payload.get("expected_revision") or 1
    )
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _state_item_ids(items: Iterable[Mapping[str, Any]]) -> List[str]:
    return [canonical_item_id(item) for item in items]


def _authoritative_state(
    payload: Mapping[str, Any], *, user_id: str, allow_initialize: bool = True
) -> Tuple[Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]]]:
    """Load server state and return ``(state, store, error)``.

    A carried board is only an identifier/revision lookup hint. Once a durable
    record exists, its items, mode, source policy and locks replace all client
    claims before execution.
    """
    client_state = _state_from_payload(payload)
    if not client_state:
        return None, None, _error("CURRENT_LOOK_REQUIRED", "I need the current look before I can change it.")
    board_id = _board_id(client_state, payload)
    if not board_id:
        return None, None, _error("CURRENT_LOOK_ID_REQUIRED", "I need the current look id before I can change it.")
    has_revision = any(
        key in client_state or key in payload
        for key in ("revision", "board_revision", "expected_revision")
    )
    if not has_revision:
        return None, None, _error("BOARD_REVISION_REQUIRED", "Refresh this look before changing it.")
    requested_revision = _revision(client_state, payload)

    try:
        from services import style_board_shuffle_service as shuffle_service

        store = shuffle_service._get_store()  # noqa: SLF001 - shared state authority
        latest = store.get_latest(board_id)
    except BoardStateStoreError:
        return None, None, _error("BOARD_STATE_UNAVAILABLE", "The current look could not be verified.")

    if latest is not None:
        owner = _text(latest.get("user_id"))
        if owner and user_id and owner != _text(user_id):
            return None, store, _error("BOARD_FORBIDDEN", "You do not have access to this look.")
        latest_revision = int(latest.get("revision") or 0)
        latest_payload = latest.get("payload") if isinstance(latest.get("payload"), Mapping) else {}
        authoritative_items = _items(latest_payload.get("items"))
        if latest_revision != requested_revision or (
            _state_item_ids(_board_items(client_state)) != _state_item_ids(authoritative_items)
        ):
            return None, store, _error(
                "STALE_BOARD_REVISION",
                "This look changed since you last saw it. Refresh it and try again.",
                current_revision=latest_revision,
                requested_revision=requested_revision,
            )
        state = deepcopy(dict(client_state))
        state.update({
            "board_id": board_id,
            "revision": latest_revision,
            "board_items": authoritative_items,
            "items": deepcopy(authoritative_items),
            "scenario": latest_payload.get("scenario") or state.get("scenario"),
            "interaction_mode": latest_payload.get("interaction_mode") or state.get("interaction_mode"),
            "source_policy": latest_payload.get("source_policy") or state.get("source_policy"),
            "source_mode": latest_payload.get("source_policy") or state.get("source_mode"),
            "occasion": latest_payload.get("occasion") or state.get("occasion"),
            "resolved_context": latest_payload.get("resolved_context") or state.get("resolved_context"),
            "active_constraints": latest_payload.get("active_constraints") or state.get("active_constraints") or {},
            "locked_item_ids": latest_payload.get("locked_item_ids") or [
                canonical_item_id(item) for item in authoritative_items if item.get("locked")
            ],
            "protected_item_ids": latest_payload.get("protected_item_ids") or [],
            "anchor_item_ids": latest_payload.get("anchor_item_ids") or [],
            "board_content_hash": latest_payload.get("board_content_hash") or state.get("board_content_hash"),
        })
        return state, store, None

    # A first mutation establishes the current carried board as revision N in
    # the existing durable store. Storage failure is fail-closed; there is no
    # synthetic/offline mutation path.
    initial_mode = _text(client_state.get("interaction_mode") or client_state.get("mode")).lower()
    initial_scenario = _text(client_state.get("scenario")).lower()
    if initial_mode in {"style_this", "build_outfit"} or initial_scenario in {"style_this", "build_outfit"}:
        return None, store, _error(
            "SPECIALIZED_BOARD_MODE",
            "This board must be changed through its specialized Style flow.",
            interaction_mode=initial_mode or initial_scenario,
        )
    initial_items = _board_items(client_state)
    if not allow_initialize:
        state = deepcopy(dict(client_state))
        state.update({
            "board_id": board_id,
            "revision": requested_revision,
            "board_items": initial_items,
            "items": deepcopy(initial_items),
        })
        return state, store, None
    protected, invalid = _protected_ids(client_state, initial_items)
    if invalid:
        return None, store, _error("INVALID_PROTECTED_ITEM_ID", "A protected item is not present on this look.", item_ids=invalid)
    if not is_complete_board(initial_items):
        return None, store, _error(
            "INCOMPLETE_BOARD",
            "This look is incomplete and cannot be opened for mutation.",
        )
    initial_payload = {
        "scenario": initial_scenario or "conversational_style",
        "interaction_mode": initial_mode or "conversational",
        "source_policy": _policy(client_state, initial_items),
        "occasion": client_state.get("occasion"),
        "resolved_context": client_state.get("resolved_context") or {},
        "active_constraints": client_state.get("active_constraints") or {},
        "locked_item_ids": sorted(protected),
        "protected_item_ids": sorted(protected),
        "items": deepcopy(initial_items),
        "board_content_hash": _content_hash(initial_items, client_state.get("occasion")),
        "previous_revision": None,
    }
    try:
        store.create_revision(
            user_id=_text(user_id), board_id=board_id,
            revision=requested_revision, payload=initial_payload,
        )
    except BoardRevisionExistsError:
        # Another request registered the board first. Re-run the authoritative
        # read so the same stale/ownership checks apply to both paths.
        latest = store.get_latest(board_id)
        if latest is None or int(latest.get("revision") or 0) != requested_revision:
            return None, store, _error("STALE_BOARD_REVISION", "This look changed while it was being opened.")
        latest_payload = latest.get("payload") if isinstance(latest.get("payload"), Mapping) else {}
        authoritative_items = _items(latest_payload.get("items"))
        if _state_item_ids(initial_items) != _state_item_ids(authoritative_items):
            return None, store, _error("STALE_BOARD_REVISION", "This look changed while it was being opened.")
        state = deepcopy(dict(client_state))
        state.update({"board_id": board_id, "revision": requested_revision, "board_items": authoritative_items, **dict(latest_payload)})
        state["items"] = deepcopy(authoritative_items)
        return state, store, None
    except BoardStateStoreError:
        return None, store, _error("BOARD_STATE_UNAVAILABLE", "The current look could not be verified.")
    state = deepcopy(dict(client_state))
    state.update({
        "board_id": board_id,
        "revision": requested_revision,
        "board_items": initial_items,
        "items": deepcopy(initial_items),
        **initial_payload,
    })
    return state, store, None


def _error(code: str, message: str, **details: Any) -> Dict[str, Any]:
    return {
        "success": False,
        "type": "style_board_mutation_error",
        "response_mode": "error",
        "error": {"code": code, "message": message, **details},
    }


def _is_durable_board_id(board_id: str) -> bool:
    return bool(re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        _text(board_id).lower(),
    ))


def _clarification(decision: Mapping[str, Any]) -> Dict[str, Any]:
    message = _text(decision.get("question")) or "What should I change while keeping the rest of this look?"
    return {
        "success": True,
        "type": "style_clarification",
        "response_mode": "clarification",
        "intent": "clarification",
        "message": {"role": "assistant", "content": message},
        "message_text": message,
        "response": message,
        "cards": [],
        "style_boards": [],
        "chips": ["Keep the top", "Change the shoes", "Make it more casual"],
        "data": {"board_operation": decision.get("operation"), "requires_clarification": True, "reason": decision.get("reason")},
        "constraint_status": {
            "passed_constraints": [],
            "unresolved_constraints": ["change_scope"],
            "fallback_reason": "clarification_required",
            "repair_attempted": False,
            "final_validation_status": "clarification",
        },
        "meta": {"mode": "style_board_clarification", "board_mutated": False},
    }


def _explanation(items: List[Mapping[str, Any]], state: Mapping[str, Any], operation: str) -> str:
    names = [(_text(item.get("name")) or canonical_item_role(dict(item))) for item in items]
    names = [name for name in names if name]
    occasion = _text(state.get("occasion"))
    if names:
        joined = ", ".join(names[:4])
        suffix = f" for {occasion}" if occasion else ""
        return f"This look works{suffix} because {joined} create a clear, balanced base."
    return "I do not have enough item detail to explain this look yet."


def _card(state: Mapping[str, Any], items: List[Dict[str, Any]], revision: int) -> Dict[str, Any]:
    card = deepcopy(dict(state))
    card.pop("candidate_items", None)
    card["board_items"] = deepcopy(items)
    card["items"] = deepcopy(items)
    card["revision"] = revision
    card["board_id"] = _text(
        card.get("board_id") or card.get("boardId")
        or state.get("board_id") or state.get("boardId")
    )
    card["source_policy"] = _policy(card, items)
    card["interaction_mode"] = "conversational_mutation"
    card["shuffle_available"] = True
    card["can_shuffle"] = True
    return card


def _response(
    state: Mapping[str, Any],
    items: List[Dict[str, Any]],
    *,
    operation: str,
    revision: int,
    previous_revision: int,
    message: str,
    mutation_id: str,
    changed_item_ids: List[str],
    explanation: Optional[str] = None,
) -> Dict[str, Any]:
    board = _card(state, items, revision)
    lineage = {
        "board_id": board["board_id"],
        "revision": revision,
        "parent_revision": previous_revision,
        "previous_revision": previous_revision,
        "mutation_id": mutation_id,
        "operation": operation,
    }
    legacy_type = (
        "style_explanation" if operation == "explain_current_look"
        else "style_refinement" if operation == "modify_current_look"
        else "style_board_mutation"
    )
    understood_action = (
        "explain_current_board" if operation == "explain_current_look"
        else "refine_current_board" if operation == "modify_current_look"
        else "create_board"
    )
    return {
        "success": True,
        "type": legacy_type,
        "response_mode": "text_only" if operation == "explain_current_look" else "wardrobe_recommendation",
        "domain": "style",
        "module": "style",
        "intent": operation,
        "action": operation,
        "board_operation": operation,
        "revision": revision,
        "previous_revision": previous_revision,
        "parent_revision": previous_revision,
        "lineage": lineage,
        "message": {"role": "assistant", "content": message},
        "message_text": message,
        "response": message,
        "cards": [] if operation == "explain_current_look" else [board],
        "style_boards": [] if operation == "explain_current_look" else [board],
        "board_ids": board["board_id"],
        "chips": ["Change the shoes", "Make it more casual", "Show another look"],
        "style_state": {
            "board_id": board["board_id"],
            "revision": revision,
            "board_items": deepcopy(items),
            "occasion": state.get("occasion"),
            "date_context": state.get("date_context"),
            "daypart": state.get("daypart"),
            "activity": state.get("activity"),
            "activity_type": state.get("activity_type"),
            "venue": state.get("venue"),
            "positive_constraints": deepcopy(state.get("positive_constraints") or []),
            "negative_constraints": deepcopy(state.get("negative_constraints") or []),
            "source_mode": board.get("source_policy"),
            "locked_item_ids": list(state.get("locked_item_ids") or []),
            "active_constraints": deepcopy(state.get("active_constraints") or {}),
            "semantic_constraints": deepcopy(state.get("semantic_constraints") or []),
            "style_adjustments": deepcopy(state.get("style_adjustments") or {}),
            "resolved_context": deepcopy(state.get("resolved_context") or {}),
            "board_content_hash": _content_hash(items, state.get("occasion")),
        },
        "understood": {
            "action": understood_action,
            "preserve_item_ids": list(state.get("locked_item_ids") or []),
            "replace_roles": list(sorted({canonical_item_role(item) for item in items if canonical_item_id(item)})),
            "confidence": 1.0,
        },
        "constraint_status": {
            "passed_constraints": ["protected_items_preserved"],
            "unresolved_constraints": [],
            "fallback_reason": None,
            "repair_attempted": False,
            "final_validation_status": "passed",
            "missing_preserved_item_ids": [],
            "excluded_term_violations": [],
            "guard_rejection_reason": None,
        },
        "visual_intelligence": None,
        "data": {
            "outfits": [] if operation == "explain_current_look" else [board],
            "rendered_boards": [] if operation == "explain_current_look" else [board],
            "board_operation": operation,
            "changed_item_ids": list(changed_item_ids),
            "lineage": lineage,
            "explanation": explanation,
        },
        "meta": {
            "mode": "style_board_mutation",
            "board_mutated": operation != "explain_current_look",
            "board_revision": revision,
            "parent_revision": previous_revision,
            "mutation_id": mutation_id,
        },
    }


def handle_board_operation(
    payload: Mapping[str, Any],
    *,
    user_id: str = "",
    semantic_decision: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Handle a conversational board operation, or return ``None``.

    ``None`` means the message is not a board operation and callers must keep
    using the established Style pipeline.  All non-``None`` responses are
    complete chat envelopes and are safe to return directly.
    """
    if not isinstance(payload, Mapping) or not isinstance(semantic_decision, Mapping):
        return None
    decision = _structured_operation(semantic_decision)
    if decision is None:
        return None
    if decision.get("requires_clarification") or semantic_decision.get("intent") == "clarification":
        return _clarification(decision)
    operation = str(decision["operation"])
    state, store, state_error = _authoritative_state(
        payload,
        user_id=user_id,
        allow_initialize=operation != "explain_current_look",
    )
    if state_error is not None:
        return state_error
    if state is None:
        return _error("CURRENT_LOOK_REQUIRED", "I need the current look before I can change it.")
    decision_context = decision.get("resolved_context") or {}
    if isinstance(decision_context, Mapping):
        for key in (
            "date_context", "daypart", "occasion", "activity", "activity_type", "venue",
            "positive_constraints", "negative_constraints",
        ):
            if decision_context.get(key) is not None:
                state[key] = deepcopy(decision_context[key])
        if decision_context.get("style_constraints") is not None:
            state["positive_constraints"] = deepcopy(decision_context["style_constraints"])

    items = _board_items(state)
    board_id = _board_id(state, payload)
    current_revision = int(state.get("revision") or _revision(state, payload))
    mode = _text(state.get("interaction_mode") or state.get("mode")).lower()
    scenario = _text(state.get("scenario")).lower()
    if mode in {"style_this", "build_outfit"} or scenario in {"style_this", "build_outfit"}:
        return _error(
            "SPECIALIZED_BOARD_MODE",
            "This board must be changed through its specialized Style flow.",
            interaction_mode=mode or scenario,
        )
    protected, invalid_protected = _protected_ids(state, items)
    if invalid_protected:
        return _error("INVALID_PROTECTED_ITEM_ID", "A protected item is not present on this look.", item_ids=invalid_protected)
    if operation == "explain_current_look":
        message_text = _explanation(items, state, operation)
        return _response(
            state, items, operation=operation, revision=current_revision,
            previous_revision=max(0, current_revision - 1), message=message_text,
            mutation_id=stable_hash({"board_id": board_id, "revision": current_revision, "operation": operation}),
            changed_item_ids=[], explanation=message_text,
        )

    candidates = _candidate_pool(payload, state)
    if not candidates:
        return _error("NO_REPLACEMENT_CANDIDATES", "I need wardrobe or style candidates before I can change this look.")
    preserve_ids = protected
    preserve_roles = set(decision.get("preserve_roles") or [])
    mutable = [
        item for item in items
        if canonical_item_id(item) not in preserve_ids
        and not any(_role_matches(item, role) for role in preserve_roles)
    ]
    if not mutable:
        return _error("ALL_ITEMS_PROTECTED", "Every piece in this look is protected; unlock one before changing it.", protected_item_ids=sorted(protected))

    target_roles = set(decision.get("target_roles") or [])
    if operation == "generate_alternative":
        targets = mutable
    else:
        targets = [
            item for item in mutable
            if target_roles and any(_role_matches(item, role) for role in target_roles)
        ]
        if not targets and (
            decision.get("preserve_roles")
            or decision.get("style_adjustments")
            or decision.get("constraints")
        ):
            targets = mutable
        if not targets:
            return _clarification({
                "operation": operation,
                "reason": "change_scope_required",
                "question": "What should I change while keeping the rest of this look?",
            })

    constraints = list(decision.get("constraints") or [])
    adjustments = dict(decision.get("style_adjustments") or {})
    state["semantic_constraints"] = deepcopy(constraints)
    state["style_adjustments"] = deepcopy(adjustments)
    state["active_constraints"] = {
        **(state.get("active_constraints") if isinstance(state.get("active_constraints"), Mapping) else {}),
        "exclude": list(dict.fromkeys(
            [
                *(_constraint_exclusions(state.get("semantic_constraints") or [])),
                *_constraint_exclusions(constraints),
            ]
        )),
        "preserve": sorted(protected),
        "semantic": deepcopy(constraints),
    }
    policy = _policy(state, items)
    out_items = deepcopy(items)
    used_ids = {canonical_item_id(item) for item in out_items if canonical_item_id(item)}
    changed: List[str] = []
    for target in targets:
        old_id = canonical_item_id(target)
        role = _request_role(target)
        replacement = _choose_candidate(
            candidates,
            role=role,
            current_id=old_id,
            used_ids=used_ids,
            policy=policy,
            adjustments=adjustments,
            constraints=constraints,
            user_id=user_id,
            seed=f"{board_id}|{current_revision}|{operation}|{old_id}",
        )
        if replacement is None:
            return _error(
                "NO_VALID_REPLACEMENT",
                f"I could not find a valid replacement for the {role} without breaking the current look.",
                role=role,
                preserved_item_ids=sorted(preserve_ids),
            )
        replacement["role"] = target.get("role") or canonical_item_role(target)
        replacement["slot"] = target.get("slot") or replacement["role"]
        if isinstance(target.get("position"), Mapping) and not isinstance(replacement.get("position"), Mapping):
            replacement["position"] = deepcopy(target["position"])
        for index, item in enumerate(out_items):
            if canonical_item_id(item) == old_id:
                out_items[index] = replacement
                break
        used_ids.discard(old_id)
        used_ids.add(canonical_item_id(replacement))
        changed.append(canonical_item_id(replacement))

    if any(canonical_item_id(item) in preserve_ids for item in out_items) is False and preserve_ids:
        return _error("PROTECTED_ITEM_LOST", "A protected item could not be preserved; the look was not changed.")
    if len({canonical_item_id(item) for item in out_items}) != len(out_items):
        return _error("DUPLICATE_ITEM_IDS", "The proposed look contains duplicate items; the look was not changed.")
    if not is_complete_board(out_items):
        return _error(
            "INCOMPLETE_BOARD",
            "The proposed change would leave this look incomplete; the look was not changed.",
        )

    new_revision = current_revision + 1
    mutation_id = stable_hash({
        "board_id": board_id, "revision": new_revision, "operation": operation,
        "items": [canonical_item_id(item) for item in out_items],
    })[:24]
    try:
        # Use the shuffle service's injected store so tests and production share
        # the same durable Appwrite revision semantics.
        payload_out = {
            "scenario": scenario or "conversational_style",
            "interaction_mode": mode or "conversational",
            "source_policy": policy,
            "allow_wardrobe_fallback": policy == "mixed",
            "occasion": state.get("occasion"),
            "resolved_context": {
                key: deepcopy(state.get(key)) for key in (
                    "date_context", "daypart", "occasion", "activity", "activity_type", "venue",
                    "positive_constraints", "negative_constraints",
                ) if state.get(key) is not None
            },
            "active_constraints": {
                "exclude": _constraint_exclusions(constraints),
                "preserve": sorted(protected),
                "semantic": deepcopy(constraints),
            },
            "locked_item_ids": sorted(protected),
            "protected_item_ids": sorted(protected),
            "items": deepcopy(out_items),
            "board_content_hash": _content_hash(out_items, state.get("occasion")),
            "previous_revision": current_revision,
            "parent_revision": current_revision,
            "operation": operation,
            "alternative_scope": decision.get("alternative_scope"),
            "mutation_id": mutation_id,
            "changed_item_ids": list(changed),
        }
        store.create_revision(
            user_id=_text(user_id) or _text((latest or {}).get("user_id")),
            board_id=board_id,
            revision=new_revision,
            payload=payload_out,
        )
    except BoardRevisionExistsError:
        latest_revision = new_revision
        try:
            latest_now = store.get_latest(board_id) if store is not None else None
            latest_revision = int((latest_now or {}).get("revision") or new_revision)
        except Exception:
            pass
        return _error("STALE_BOARD_REVISION", "This look changed while I was updating it. Refresh it and try again.", current_revision=latest_revision)
    except BoardStateStoreError:
        return _error("BOARD_STATE_UNAVAILABLE", "Look state storage is unavailable; the look was not changed.")
    except Exception:
        return _error("BOARD_STATE_UNAVAILABLE", "The current look could not be safely updated.")

    action_word = "alternative" if operation == "generate_alternative" else "updated"
    message_text = f"Here is your {action_word} look, with the protected pieces kept in place."
    return _response(
        state, out_items, operation=operation, revision=new_revision,
        previous_revision=current_revision, message=message_text,
        mutation_id=mutation_id, changed_item_ids=changed,
    )


__all__ = [
    "BOARD_OPERATIONS",
    "board_context_for_semantics",
    "handle_board_operation",
]
