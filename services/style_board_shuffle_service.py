"""Style-board shuffle service: lock-aware regeneration with revisions.

LIMITATION (documented, accepted for the first verified implementation):
the board revision registry below is per-process in-memory state.  It resets
on instance restart and is NOT shared across Cloud Run instances, so under
multi-instance traffic two instances can hold different revisions for the
same board.  Unknown board ids are therefore registered at the requested
revision (self-healing), and the registry is a consistency aid rather than a
durable store.  Move to Appwrite/Redis before multi-instance rollout.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from services.constrained_outfit_builder import ConstrainedOutfitBuilder
from services.style_item_contract import (
    FixedItemLostError,
    assert_fixed_items_preserved,
    canonical_item_id,
)

logger = logging.getLogger("ahvi.style_board_shuffle")

# board_id -> {"revision": int, "previous": snapshot|None, "items": [...]}
_REGISTRY: Dict[str, Dict[str, Any]] = {}
_REGISTRY_LOCK = threading.Lock()

_builder = ConstrainedOutfitBuilder()

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
    """Test helper: clear all board state."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


def get_board_state(board_id: str) -> Optional[Dict[str, Any]]:
    """Test/debug helper: current registry entry for a board (copy)."""
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(str(board_id))
        return dict(entry) if entry else None


def shuffle_board(
    board_id: str,
    revision: int,
    locked_items: Optional[List[Dict[str, Any]]] = None,
    shuffle_slots: Optional[List[str]] = None,
    exclude_item_ids: Optional[List[str]] = None,
    occasion: Optional[str] = None,
    source_policy: Any = None,
    wardrobe: Optional[List[Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None,
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

    # --- Revision validation (register unknown boards at requested rev) ----
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(board_id)
        if entry is None:
            entry = {"revision": revision, "previous": None, "items": []}
            _REGISTRY[board_id] = entry
        elif entry["revision"] != revision:
            return _error(
                "BOARD_REVISION_CONFLICT",
                "This board changed since you last saw it - refresh and try again.",
                current_revision=entry["revision"],
                requested_revision=revision,
            )

    # --- All locked / nothing to shuffle ------------------------------------
    if not shuffle_slots:
        return _error(
            "ALL_ITEMS_LOCKED",
            "Every piece is locked - unlock at least one item to shuffle.",
            locked_item_ids=[canonical_item_id(i) for i in locked_items],
        )

    # --- Constrained generation ---------------------------------------------
    builder_context = {
        "wardrobe": wardrobe or [],
        "occasion": occasion,
        # rotate variety with the revision so consecutive shuffles differ
        "variant": int(context.get("variant", revision)),
        "accessory_budget": max(1, shuffle_slots.count("accessory")),
    }
    result = _builder.generate(
        scenario="shuffle_unlocked",
        fixed_items=locked_items,
        replaceable_slots=shuffle_slots,
        exclude_item_ids=exclude_item_ids or [],
        source_policy=source_policy if isinstance(source_policy, dict) else None,
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

    # --- Commit: bump revision + one-level undo snapshot ---------------------
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(board_id) or {"revision": revision, "previous": None, "items": []}
        previous_revision = entry["revision"]
        entry["previous"] = {"revision": previous_revision, "items": list(entry.get("items") or [])}
        entry["revision"] = previous_revision + 1
        entry["items"] = [dict(i) for i in out_items]
        _REGISTRY[board_id] = entry
        new_revision = entry["revision"]

    return {
        "success": True,
        "board_id": board_id,
        "revision": new_revision,
        "previous_revision": previous_revision,
        "locked_items_preserved": True,
        "occasion": result.get("occasion"),
        "source": result.get("source"),
        "changed_slots": result.get("changed_slots", []),
        "missing_items": result.get("missing_items", []),
        "board_items": out_items,
    }
