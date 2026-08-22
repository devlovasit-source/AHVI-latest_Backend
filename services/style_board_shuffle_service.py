"""Style-board shuffle service: lock-aware regeneration with revisions.

Board state is DURABLE: one immutable Appwrite document per (board_id,
revision) via services.style_board_state_store. Creating revision N+1 with
its deterministic document ID is the atomic claim — of two concurrent
shuffles from revision N (even across Cloud Run instances or restarts),
exactly one create succeeds; the loser gets BOARD_REVISION_CONFLICT.

Production uses AppwriteBoardStateStore by default. Tests must EXPLICITLY
inject InMemoryBoardStateStore via set_state_store(); there is no implicit
in-memory fallback — storage outages fail typed (BOARD_STATE_UNAVAILABLE),
never silently in-process.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from brain.engines.outfit_quality_guard import is_complete_board
from services.constrained_outfit_builder import ConstrainedOutfitBuilder
from services.style_board_reasoning import build_styling_note
from services.style_board_state_store import (
    AppwriteBoardStateStore,
    BoardRevisionExistsError,
    BoardStateStoreError,
    InMemoryBoardStateStore,
)
from services.style_item_contract import (
    FixedItemLostError,
    VALID_SOURCES,
    assert_fixed_items_preserved,
    canonical_item_id,
    canonical_item_source,
)
from services.wardrobe_sanitizer import is_fashion_item

logger = logging.getLogger("ahvi.style_board_shuffle")

# Canonical board-level completion-source policy.  This is a BOARD contract
# persisted at creation time - it is never inferred from the sources of the
# locked items (a wardrobe anchor inside a Style This board does not make the
# board wardrobe-only).
VALID_BOARD_POLICIES = ("wardrobe", "style_asset", "mixed")

_POLICY_COMPLETION_SOURCES: Dict[str, "tuple[str, ...]"] = {
    "wardrobe": ("wardrobe",),
    "style_asset": ("style_asset",),
    "mixed": ("style_asset", "wardrobe"),
}


def canonical_board_policy(value: Any) -> str:
    """Normalize a board policy string; returns "" when not canonical."""
    text = str(value or "").strip().lower()
    return text if text in VALID_BOARD_POLICIES else ""


def _policy_from_sources(sources: "frozenset[str] | set") -> str:
    if sources == {"wardrobe"}:
        return "wardrobe"
    if sources == {"style_asset"}:
        return "style_asset"
    if sources and sources <= {"wardrobe", "style_asset"}:
        return "mixed"
    return "mixed" if len(sources) > 1 else ""


# Durable state store. None -> lazily constructed AppwriteBoardStateStore
# (production default). Tests inject InMemoryBoardStateStore explicitly.
_STORE: Optional[Any] = None

_builder = ConstrainedOutfitBuilder()


def set_state_store(store: Optional[Any]) -> None:
    """Inject the board state store. Pass None to restore the production
    default (AppwriteBoardStateStore). Test doubles are injected explicitly —
    never selected implicitly."""
    global _STORE
    _STORE = store


def _get_store() -> Any:
    global _STORE
    if _STORE is None:
        _STORE = AppwriteBoardStateStore()
    return _STORE

# Deterministic placement fallback (used when style_board_engine helpers are
# unavailable); mirrors the engine's stable-hash approach.
try:
    from brain.engines.style_board_engine import _stable_choice, _stable_uniform
except Exception:  # pragma: no cover - engine optional
    import hashlib

    def _stable_uniform(a: float, b: float, *, seed: str) -> float:  # type: ignore[misc]
        digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
        frac = int(digest[:8], 16) / 0xFFFFFFFF
        return a + (b - a) * frac

    def _stable_choice(options: List[Any], *, seed: str):  # type: ignore[misc]
        digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
        return options[int(digest[:8], 16) % len(options)]


def _error(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    err = {"code": code, "message": message}
    err.update(extra)
    return {"success": False, "error": err}


def _locked_item_is_fashion(item: Dict[str, Any]) -> bool:
    """is_fashion_item(), but tolerant of an id/role/slot-only locked-item
    payload with no name/category to classify against (name is optional on
    LockedBoardItem - a client may echo back only identity + position).
    is_fashion_item() itself treats a missing name as an automatic reject,
    which would misfire here; a locked item with no identifying text is not
    evidence of anything, so trust the role a non-fashion item could never
    have earned in the first place (canonical_item_role/_lite_role both
    already exclude non-fashion tokens before a role is assigned).
    """
    name = str(item.get("name") or item.get("title") or "").strip()
    if not name:
        role = str(item.get("role") or item.get("slot") or "").strip().lower()
        return role not in ("", "unknown")
    return is_fashion_item(item)


def _default_position(item_id: str, slot: str) -> Dict[str, Any]:
    """Deterministic placement for a brand-new slot - no full board relayout."""
    seed = f"{slot}:{item_id}"
    return {
        "x": round(_stable_uniform(0.08, 0.62, seed=f"x:{seed}"), 3),
        "y": round(_stable_uniform(0.08, 0.62, seed=f"y:{seed}"), 3),
        "width": round(_stable_uniform(0.24, 0.34, seed=f"w:{seed}"), 3),
        "height": round(_stable_uniform(0.24, 0.34, seed=f"h:{seed}"), 3),
        "z": 1,
        "rotation": _stable_choice([-15, -5, 5, 15], seed=f"rot:{seed}"),
    }


def reset_registry() -> None:
    """Test helper: clear board state on an injected in-memory store.

    No-op for the production Appwrite store (revision documents are
    immutable); tests inject a fresh InMemoryBoardStateStore instead.
    """
    store = _STORE
    if isinstance(store, InMemoryBoardStateStore):
        store.clear()


def _payload_to_state(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    return {
        "revision": int(record.get("revision") or 0),
        "user_id": str(record.get("user_id") or ""),
        "previous": None,
        "items": [dict(i) for i in (payload.get("items") or []) if isinstance(i, dict)],
        "anchor_item_id": str(payload.get("anchor_item_id") or ""),
        "scenario": str(payload.get("scenario") or ""),
        "source_policy": str(payload.get("source_policy") or ""),
        "allow_wardrobe_fallback": bool(payload.get("allow_wardrobe_fallback")),
        "occasion": payload.get("occasion"),
        "style_direction": payload.get("style_direction"),
        "style_strategy": dict(payload.get("style_strategy")) if isinstance(payload.get("style_strategy"), dict) else None,
        "styling_note": payload.get("styling_note"),
    }


def get_board_state(board_id: str) -> Optional[Dict[str, Any]]:
    """Test/debug helper: latest durable state for a board (copy), with a
    one-level `previous` snapshot resolved from the prior revision document."""
    store = _get_store()
    try:
        latest = store.get_latest(str(board_id))
        if latest is None:
            return None
        state = _payload_to_state(latest)
        if state["revision"] > 1:
            prev = store.get_revision(str(board_id), state["revision"] - 1)
            if prev is not None:
                prev_state = _payload_to_state(prev)
                state["previous"] = {
                    "revision": prev_state["revision"],
                    "items": prev_state["items"],
                }
        return state
    except BoardStateStoreError:
        return None


def _build_payload(
    *,
    scenario: str,
    source_policy: str,
    allow_wardrobe_fallback: bool,
    occasion: Any,
    style_direction: Any,
    style_strategy: Any,
    items: Optional[List[Dict[str, Any]]],
    anchor_item_id: Optional[str] = None,
    previous_revision: Optional[int] = None,
    styling_note: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "scenario": str(scenario or "").strip().lower(),
        "source_policy": source_policy,
        "allow_wardrobe_fallback": bool(allow_wardrobe_fallback),
        "occasion": occasion,
        "style_direction": style_direction,
        "style_strategy": dict(style_strategy) if isinstance(style_strategy, dict) else None,
        "items": [dict(i) for i in (items or []) if isinstance(i, dict)],
        "previous_revision": previous_revision,
        "styling_note": str(styling_note).strip() if styling_note else None,
    }
    if str(anchor_item_id or "").strip():
        payload["anchor_item_id"] = str(anchor_item_id).strip()
    return payload


def _payload_contract_equivalent(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = ("scenario", "source_policy", "allow_wardrobe_fallback", "style_strategy")
    return all(a.get(k) == b.get(k) for k in keys)


def register_board(
    board_id: str,
    revision: int = 1,
    scenario: str = "",
    source_policy: str = "",
    allow_wardrobe_fallback: bool = False,
    occasion: Optional[str] = None,
    style_direction: Optional[str] = None,
    style_strategy: Optional[Dict[str, Any]] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    anchor_item_id: Optional[str] = None,
    styling_note: Optional[str] = None,
    user_id: str = "",
) -> Dict[str, Any]:
    """Durably persist a board's creation-time contract (revision 1).

    The stored source_policy is what "inherit" resolves to on shuffle — it is
    the ONLY legitimate origin of a board's completion-source policy.

    Idempotent for the same owner + equivalent contract; a conflicting owner
    or incompatible existing state is never overwritten. Returns
    {"ok": bool, "error": {"code", "message"} | None}.
    """
    board_id = str(board_id or "").strip()
    policy = canonical_board_policy(source_policy)
    if not board_id or not policy:
        return {
            "ok": False,
            "error": {
                "code": "BOARD_REGISTRATION_INVALID",
                "message": "board_id and a canonical source_policy are required.",
            },
        }
    payload = _build_payload(
        scenario=scenario,
        source_policy=policy,
        allow_wardrobe_fallback=bool(allow_wardrobe_fallback) or policy == "mixed",
        occasion=occasion,
        style_direction=style_direction,
        style_strategy=style_strategy,
        styling_note=styling_note,
        items=items,
        anchor_item_id=anchor_item_id if str(scenario or "").strip().lower() == "style_this" else None,
        previous_revision=None,
    )
    store = _get_store()
    try:
        store.create_revision(
            user_id=str(user_id or ""),
            board_id=board_id,
            revision=int(revision),
            payload=payload,
        )
        return {"ok": True, "error": None}
    except BoardRevisionExistsError:
        try:
            existing = store.get_revision(board_id, int(revision))
        except BoardStateStoreError as exc:
            logger.warning("register_board verify failed board=%s err=%s", board_id, exc)
            existing = None
        if existing is not None:
            same_owner = (
                not str(user_id or "")
                or not str(existing.get("user_id") or "")
                or str(existing.get("user_id")) == str(user_id)
            )
            if same_owner and _payload_contract_equivalent(
                existing.get("payload") or {}, payload
            ):
                return {"ok": True, "error": None}  # idempotent re-registration
        logger.warning(
            "register_board conflict board=%s revision=%s user=%s",
            board_id, revision, user_id,
        )
        return {
            "ok": False,
            "error": {
                "code": "BOARD_REGISTRATION_CONFLICT",
                "message": "A different board already exists under this id.",
            },
        }
    except BoardStateStoreError as exc:
        logger.error("register_board storage unavailable board=%s err=%s", board_id, exc)
        return {
            "ok": False,
            "error": {
                "code": "BOARD_STATE_UNAVAILABLE",
                "message": "Board state storage is unavailable; shuffle is disabled for this board.",
            },
        }


def shuffle_board(
    board_id: str,
    revision: int,
    locked_items: Optional[List[Dict[str, Any]]] = None,
    shuffle_slots: Optional[List[str]] = None,
    exclude_item_ids: Optional[List[str]] = None,
    occasion: Optional[str] = None,
    source_policy: Any = None,
    wardrobe: Optional[List[Dict[str, Any]]] = None,
    style_assets: Optional[List[Dict[str, Any]]] = None,
    style_asset_provider: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
    user_id: str = "",
) -> Dict[str, Any]:
    board_id = str(board_id or "").strip()
    if not board_id:
        return _error("INVALID_ITEM_ID", "board_id is required.")
    try:
        revision = int(revision)
    except (TypeError, ValueError):
        return _error("BOARD_REVISION_CONFLICT", "revision must be an integer.")

    locked_items = [i for i in (locked_items or []) if isinstance(i, dict)]
    shuffle_slots = [str(s or "").strip().lower() for s in (shuffle_slots or []) if str(s or "").strip()]
    context = dict(context or {})
    board_items = [i for i in (context.get("board_items") or []) if isinstance(i, dict)]

    # --- Load durable state (no self-registration of unknown boards) --------
    store = _get_store()
    try:
        latest = store.get_latest(board_id)
    except BoardStateStoreError as exc:
        logger.error("shuffle_board state load failed board=%s err=%s", board_id, exc)
        return _error(
            "BOARD_STATE_UNAVAILABLE",
            "Board state storage is unavailable - please try again shortly.",
        )
    if latest is None:
        return _error(
            "BOARD_STATE_NOT_FOUND",
            "This board has no stored state (it may predate durable shuffle) - regenerate the board to continue.",
            action="regenerate_board",
        )

    # --- Ownership: stored owner must match the authenticated user ----------
    stored_owner = str(latest.get("user_id") or "")
    requester = str(user_id or "")
    if stored_owner and requester and stored_owner != requester:
        logger.warning(
            "shuffle_board forbidden board=%s owner=%s requester=%s",
            board_id, stored_owner, requester,
        )
        # Typed denial WITHOUT board contents.
        return _error("BOARD_FORBIDDEN", "You do not have access to this board.")

    if int(latest.get("revision") or 0) != revision:
        return _error(
            "BOARD_REVISION_CONFLICT",
            "This board changed since you last saw it - refresh and try again.",
            current_revision=int(latest.get("revision") or 0),
            requested_revision=revision,
        )
    stored_payload = latest.get("payload") if isinstance(latest.get("payload"), dict) else {}
    stored_policy = canonical_board_policy(stored_payload.get("source_policy"))
    stored_scenario = str(stored_payload.get("scenario") or "").strip().lower()
    stored_style_direction = stored_payload.get("style_direction")
    stored_style_strategy = (
        dict(stored_payload.get("style_strategy"))
        if isinstance(stored_payload.get("style_strategy"), dict)
        else None
    )
    if stored_scenario == "style_this":
        stored_items = [
            dict(item) for item in stored_payload.get("items") or []
            if isinstance(item, dict)
        ]
        stored_anchor_id = str(stored_payload.get("anchor_item_id") or "").strip()
        if not stored_anchor_id:
            locked_stored = [
                item for item in stored_items if item.get("locked")
            ]
            if len(locked_stored) == 1:
                stored_anchor_id = canonical_item_id(locked_stored[0])
        anchor_matches = [
            item for item in stored_items
            if canonical_item_id(item) == stored_anchor_id
        ]
        if not stored_anchor_id or len(anchor_matches) != 1:
            return _error(
                "INVALID_STYLE_THIS_ANCHOR",
                "This Style This board has no single authoritative anchor.",
            )
        stored_by_id = {
            canonical_item_id(item): item
            for item in stored_items
            if canonical_item_id(item)
        }
        requested_locked_ids = {
            canonical_item_id(item) for item in locked_items
            if canonical_item_id(item)
        }
        requested_locked_ids.add(stored_anchor_id)
        locked_items = [
            dict(stored_by_id[item_id])
            for item_id in requested_locked_ids
            if item_id in stored_by_id
        ]
        for item in locked_items:
            item["locked"] = True
        stored_payload["anchor_item_id"] = stored_anchor_id
        if stored_policy := canonical_board_policy(stored_payload.get("source_policy")):
            if stored_policy != "wardrobe":
                return _error(
                    "SOURCE_POLICY_VIOLATION",
                    "Style This revisions require wardrobe-only board state.",
                    source_policy=stored_policy,
                )
        if any(canonical_item_source(item) != "wardrobe" for item in stored_items):
            return _error(
                "SOURCE_POLICY_VIOLATION",
                "Style This revisions cannot contain style assets.",
                source_policy="wardrobe",
            )

    # --- Locked-item lifecycle guard -----------------------------------------
    # For style_this, locked_items above was just rebuilt from stored_by_id
    # (durable state), including a stored anchor the caller never has to name
    # explicitly - so this must run on the FINAL resolved list, not the raw
    # request. assert_fixed_items_preserved only proves locked items survive
    # shuffle; it has no opinion on whether they should still exist. A board
    # saved before the fashion sanitizer existed (or before it covered
    # travel-document items) must not have that item preserved forever purely
    # because it is durably locked - fail typed, before any generation or
    # state mutation, rather than silently carrying it into every future
    # revision (undo/missing-slot-repair reuse the same stored_items/locked
    # rebuild path, so this one check also covers those).
    non_fashion_locked = [item for item in locked_items if not _locked_item_is_fashion(item)]
    if non_fashion_locked:
        return _error(
            "BOARD_CONTAINS_NON_FASHION_ITEM",
            "This look contains an item that can no longer be used for "
            "styling. Refresh the look to continue.",
            item_ids=[canonical_item_id(item) for item in non_fashion_locked],
        )

    # --- Board policy resolution (NEVER inferred from locked-item sources) --
    # Precedence: explicit dict (internal) > explicit canonical string >
    # persisted board policy.  "inherit" (or absence) resolves ONLY from the
    # board registry; a legacy board without a stored policy fails typed.
    policy_dict = source_policy if isinstance(source_policy, dict) else None
    requested_policy = canonical_board_policy(source_policy) if isinstance(source_policy, str) else ""
    if policy_dict:
        allowed_set = {
            str(s or "").strip().lower()
            for s in (
                policy_dict.get("allowed_completion_sources")
                or policy_dict.get("allowed_sources")
                or []
            )
        }
        allowed_set = {s for s in allowed_set if s in VALID_SOURCES and s != "unknown"}
        if not allowed_set:
            return _error(
                "UNKNOWN_ITEM_SOURCE",
                "The requested source policy contains no recognizable sources.",
            )
        resolved_policy = _policy_from_sources(allowed_set) or "mixed"
        allowed_sources = tuple(sorted(allowed_set))
    else:
        resolved_policy = requested_policy or stored_policy
        if resolved_policy not in VALID_BOARD_POLICIES:
            return _error(
                "BOARD_SOURCE_POLICY_UNKNOWN",
                "This board has no styling-source policy on record - regenerate the board to continue.",
            )
        allowed_sources = _POLICY_COMPLETION_SOURCES[resolved_policy]
    if stored_scenario == "style_this":
        if resolved_policy != "wardrobe":
            return _error(
                "SOURCE_POLICY_VIOLATION",
                "Style This revisions require wardrobe-only completion.",
                source_policy=resolved_policy,
            )
        resolved_policy = "wardrobe"
        allowed_sources = ("wardrobe",)
    allow_wardrobe_fallback = resolved_policy == "mixed"

    # --- Style-asset candidate pool (same provider as initial generation) ---
    style_asset_items = [i for i in (style_assets or []) if isinstance(i, dict)]
    if "style_asset" in allowed_sources and not style_asset_items and callable(style_asset_provider):
        try:
            style_asset_items = [
                i for i in (style_asset_provider() or []) if isinstance(i, dict)
            ]
        except Exception:
            logger.exception("shuffle_board style asset provider failed board=%s", board_id)
            style_asset_items = []
    style_asset_items = [
        i if i.get("source") else {**i, "source": "style_asset"}
        for i in style_asset_items
    ]

    # --- All locked / nothing to shuffle ------------------------------------
    if not shuffle_slots:
        return _error(
            "ALL_ITEMS_LOCKED",
            "Every piece is locked - unlock at least one item to shuffle.",
            locked_item_ids=[canonical_item_id(i) for i in locked_items],
        )

    # --- Constrained generation ---------------------------------------------
    builder_context = {
        "occasion": occasion,
        # rotate variety with the revision so consecutive shuffles differ
        "variant": int(context.get("variant", revision)),
        "accessory_budget": max(1, shuffle_slots.count("accessory")),
        "style_strategy": stored_style_strategy,
    }
    result = _builder.generate(
        scenario="shuffle_unlocked",
        fixed_items=locked_items,
        replaceable_slots=shuffle_slots,
        exclude_item_ids=exclude_item_ids or [],
        source_policy={"allowed_completion_sources": list(allowed_sources)},
        wardrobe=wardrobe or [],
        style_assets=style_asset_items,
        context=builder_context,
    )
    if not result.get("success"):
        # Typed failure passthrough; registry state is untouched on failure.
        return result

    # --- Layout: locked positions untouched, replaced slots inherit ---------
    locked_by_id = {canonical_item_id(i): i for i in locked_items}
    prev_position_by_slot: Dict[str, Dict[str, Any]] = {}
    for prev in board_items:
        slot = str(prev.get("slot") or "").strip().lower()
        if slot and isinstance(prev.get("position"), dict):
            prev_position_by_slot.setdefault(slot, prev["position"])

    out_items: List[Dict[str, Any]] = []
    for item in result["items"]:
        item = dict(item)
        iid = item.get("item_id", "")
        slot = str(item.get("slot") or item.get("role") or "").strip().lower()
        item["slot"] = slot
        if item.get("locked") and iid in locked_by_id:
            # Preserve the caller's complete locked payload; only add the
            # canonical lock marker needed by board consumers.
            item = dict(locked_by_id[iid])
            item["locked"] = True
        elif not item.get("locked"):
            inherited = prev_position_by_slot.get(slot)
            if isinstance(inherited, dict):
                item["position"] = inherited  # replaced item inherits placement
            elif not isinstance(item.get("position"), dict):
                item["position"] = _default_position(iid, slot)  # brand-new slot
        out_items.append(item)

    # --- Final lock-preservation assert (do not repair - fail typed) --------
    try:
        assert_fixed_items_preserved(locked_items, out_items, stage="shuffle_response")
    except FixedItemLostError as exc:
        logger.error("shuffle_board lost locked items board=%s: %s", board_id, exc)
        return _error(
            "FIXED_ITEM_LOST",
            "A locked item went missing while shuffling; your board was not changed.",
            missing_ids=exc.missing_ids,
            stage=exc.stage,
        )
    if stored_scenario == "style_this":
        anchor_matches = [
            item for item in out_items
            if canonical_item_id(item) == stored_payload.get("anchor_item_id")
        ]
        if len(anchor_matches) != 1:
            return _error(
                "FIXED_ITEM_LOST",
                "The Style This anchor was not preserved; your board was not changed.",
            )

    # --- Source-policy validation on the FINAL serialized items -------------
    # Fixed items keep their exact original source (a wardrobe anchor inside a
    # Style This board is valid); every completion item must match the board
    # policy.
    fixed_ids = {canonical_item_id(i) for i in locked_items}
    for item in out_items:
        if canonical_item_id(item) in fixed_ids:
            continue
        item_source = canonical_item_source(item)
        if item_source not in allowed_sources:
            logger.error(
                "shuffle_board source policy violation board=%s policy=%s item=%s source=%s",
                board_id, resolved_policy, canonical_item_id(item), item_source,
            )
            return _error(
                "SOURCE_POLICY_VIOLATION",
                "A replacement piece did not match this board's styling source; your board was not changed.",
                source_policy=resolved_policy,
                violating_item_id=canonical_item_id(item),
                violating_source=item_source,
            )

    if not is_complete_board(out_items):
        return _error(
            "INSUFFICIENT_WARDROBE",
            "The available pieces could not produce a complete outfit; your board was not changed.",
        )

    # --- Regenerate reasoning from the FINAL shuffled items, not the pre- ---
    # shuffle set. Only style_this boards have a single well-defined anchor;
    # other scenarios keep whatever styling_note (if any) was already stored.
    new_anchor_id = (
        stored_payload.get("anchor_item_id") if stored_scenario == "style_this" else None
    )
    new_styling_note = stored_payload.get("styling_note")
    if stored_scenario == "style_this" and new_anchor_id:
        anchor_for_note = next(
            (item for item in out_items if canonical_item_id(item) == new_anchor_id),
            None,
        )
        new_styling_note = build_styling_note(anchor_for_note, out_items, stored_style_strategy)

    # --- Atomic commit: create the immutable revision N+1 document ----------
    # Creating the deterministic (board_id, N+1) document IS the claim; a
    # concurrent shuffle from the same revision loses with a typed conflict.
    # Success is never reported before the durable create succeeds.
    previous_revision = revision
    new_revision = revision + 1
    new_payload = _build_payload(
        scenario=stored_scenario,
        source_policy=resolved_policy,
        allow_wardrobe_fallback=allow_wardrobe_fallback,
        occasion=occasion if occasion is not None else stored_payload.get("occasion"),
        style_direction=stored_style_direction,
        style_strategy=stored_style_strategy,
        items=out_items,
        anchor_item_id=new_anchor_id,
        previous_revision=previous_revision,
        styling_note=new_styling_note,
    )
    try:
        store.create_revision(
            user_id=stored_owner or requester,
            board_id=board_id,
            revision=new_revision,
            payload=new_payload,
        )
    except BoardRevisionExistsError:
        current = new_revision
        try:
            latest_now = store.get_latest(board_id)
            if latest_now is not None:
                current = int(latest_now.get("revision") or new_revision)
        except BoardStateStoreError:
            pass
        return _error(
            "BOARD_REVISION_CONFLICT",
            "This board changed since you last saw it - refresh and try again.",
            current_revision=current,
            requested_revision=revision,
        )
    except BoardStateStoreError as exc:
        logger.error("shuffle_board revision commit failed board=%s err=%s", board_id, exc)
        return _error(
            "BOARD_STATE_UNAVAILABLE",
            "Board state storage is unavailable - your board was not changed.",
        )
    response_scenario = stored_scenario or None

    logger.info(
        "AHVI_STYLE_THIS_SHUFFLE_RESULT board_id=%s old_revision=%s new_revision=%s "
        "anchor_preserved=%s changed_unlocked_count=%s",
        board_id,
        previous_revision,
        new_revision,
        True,
        len(result.get("changed_slots", [])),
    )

    return {
        "success": True,
        "board_id": board_id,
        "revision": new_revision,
        "previous_revision": previous_revision,
        "locked_items_preserved": True,
        "scenario": response_scenario,
        "source_policy": resolved_policy,
        "allow_wardrobe_fallback": allow_wardrobe_fallback,
        "occasion": result.get("occasion"),
        "style_strategy": stored_style_strategy,
        "anchor_item_id": new_anchor_id,
        "styling_note": new_styling_note,
        "source": result.get("source"),
        "changed_slots": result.get("changed_slots", []),
        "missing_items": result.get("missing_items", []),
        "board_items": out_items,
    }


def undo_board(
    board_id: str,
    revision: int,
    user_id: str = "",
) -> Dict[str, Any]:
    """Restore a board to the content of its immediately-preceding durable
    revision, as a NEW forward revision - never decrements or overwrites.

    Unlike shuffle_board, undo needs no client-supplied items/locks: the
    entire restored payload (items, locks, anchor, styling_note, strategy)
    comes verbatim from the durable revision history, which is authoritative
    by construction (every stored revision already passed shuffle_board's
    validation when it was created). Atomicity/conflict detection mirrors
    shuffle_board exactly - creating (board_id, new_revision) IS the claim.
    """
    board_id = str(board_id or "").strip()
    if not board_id:
        return _error("INVALID_ITEM_ID", "board_id is required.")
    try:
        revision = int(revision)
    except (TypeError, ValueError):
        return _error("BOARD_REVISION_CONFLICT", "revision must be an integer.")

    store = _get_store()
    try:
        latest = store.get_latest(board_id)
    except BoardStateStoreError as exc:
        logger.error("undo_board state load failed board=%s err=%s", board_id, exc)
        return _error(
            "BOARD_STATE_UNAVAILABLE",
            "Board state storage is unavailable - please try again shortly.",
        )
    if latest is None:
        return _error(
            "BOARD_STATE_NOT_FOUND",
            "This board has no stored state (it may predate durable shuffle) - regenerate the board to continue.",
            action="regenerate_board",
        )

    stored_owner = str(latest.get("user_id") or "")
    requester = str(user_id or "")
    if stored_owner and requester and stored_owner != requester:
        logger.warning(
            "undo_board forbidden board=%s owner=%s requester=%s",
            board_id, stored_owner, requester,
        )
        return _error("BOARD_FORBIDDEN", "You do not have access to this board.")

    latest_revision = int(latest.get("revision") or 0)
    if latest_revision != revision:
        return _error(
            "BOARD_REVISION_CONFLICT",
            "This board changed since you last saw it - refresh and try again.",
            current_revision=latest_revision,
            requested_revision=revision,
        )

    latest_payload = latest.get("payload") if isinstance(latest.get("payload"), dict) else {}
    restore_target = latest_payload.get("previous_revision")
    if not isinstance(restore_target, int) or restore_target < 1:
        return _error(
            "NO_PREVIOUS_REVISION",
            "This board has no earlier version to undo to.",
        )

    try:
        restore_from = store.get_revision(board_id, restore_target)
    except BoardStateStoreError as exc:
        logger.error("undo_board restore load failed board=%s err=%s", board_id, exc)
        return _error(
            "BOARD_STATE_UNAVAILABLE",
            "Board state storage is unavailable - your board was not changed.",
        )
    if restore_from is None:
        return _error(
            "NO_PREVIOUS_REVISION",
            "The earlier version of this board is no longer available.",
        )
    restored_payload = (
        restore_from.get("payload") if isinstance(restore_from.get("payload"), dict) else {}
    )

    # New revision's content is the restored payload verbatim, except
    # previous_revision - which, like every shuffle-created revision,
    # always points at the revision this one was created FROM (the one
    # being undone). That keeps history walkable one step at a time
    # regardless of whether a given revision came from a shuffle or an
    # undo, and is what lets a later undo restore the immediate prior
    # user-visible board rather than jumping back to revision 1.
    new_revision = latest_revision + 1
    new_payload = dict(restored_payload)
    new_payload["previous_revision"] = latest_revision

    try:
        store.create_revision(
            user_id=stored_owner or requester,
            board_id=board_id,
            revision=new_revision,
            payload=new_payload,
        )
    except BoardRevisionExistsError:
        current = new_revision
        try:
            latest_now = store.get_latest(board_id)
            if latest_now is not None:
                current = int(latest_now.get("revision") or new_revision)
        except BoardStateStoreError:
            pass
        return _error(
            "BOARD_REVISION_CONFLICT",
            "This board changed since you last saw it - refresh and try again.",
            current_revision=current,
            requested_revision=revision,
        )
    except BoardStateStoreError as exc:
        logger.error("undo_board revision commit failed board=%s err=%s", board_id, exc)
        return _error(
            "BOARD_STATE_UNAVAILABLE",
            "Board state storage is unavailable - your board was not changed.",
        )

    logger.info(
        "AHVI_STYLE_THIS_UNDO_RESULT board_id=%s old_revision=%s new_revision=%s "
        "restored_from_revision=%s",
        board_id, latest_revision, new_revision, restore_target,
    )

    return {
        "success": True,
        "board_id": board_id,
        "revision": new_revision,
        "previous_revision": latest_revision,
        "restored_from_revision": restore_target,
        "locked_items_preserved": True,
        "scenario": str(new_payload.get("scenario") or "") or None,
        "source_policy": new_payload.get("source_policy"),
        "allow_wardrobe_fallback": bool(new_payload.get("allow_wardrobe_fallback")),
        "occasion": new_payload.get("occasion"),
        "style_strategy": new_payload.get("style_strategy"),
        "anchor_item_id": new_payload.get("anchor_item_id"),
        "styling_note": new_payload.get("styling_note"),
        "changed_slots": [],
        "missing_items": [],
        "board_items": new_payload.get("items") or [],
    }
