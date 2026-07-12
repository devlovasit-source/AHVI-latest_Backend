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
    VALID_SOURCES,
    assert_fixed_items_preserved,
    canonical_item_id,
    canonical_item_source,
)

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


# board_id -> {"revision": int, "previous": snapshot|None, "items": [...],
#              "scenario": str, "source_policy": str,
#              "allow_wardrobe_fallback": bool, "occasion": str|None,
#              "style_direction": str|None}
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


def _new_entry(revision: int) -> Dict[str, Any]:
    return {
        "revision": revision,
        "previous": None,
        "items": [],
        "scenario": "",
        "source_policy": "",
        "allow_wardrobe_fallback": False,
        "occasion": None,
        "style_direction": None,
    }


def register_board(
    board_id: str,
    revision: int = 1,
    scenario: str = "",
    source_policy: str = "",
    allow_wardrobe_fallback: bool = False,
    occasion: Optional[str] = None,
    style_direction: Optional[str] = None,
    items: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Persist a board's creation-time contract (called by initial generation).

    The stored source_policy is what "inherit" resolves to on shuffle - it is
    the ONLY legitimate origin of a board's completion-source policy.
    """
    board_id = str(board_id or "").strip()
    policy = canonical_board_policy(source_policy)
    if not board_id or not policy:
        return
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(board_id) or _new_entry(int(revision))
        entry["revision"] = int(revision)
        entry["scenario"] = str(scenario or "").strip().lower()
        entry["source_policy"] = policy
        entry["allow_wardrobe_fallback"] = bool(allow_wardrobe_fallback) or policy == "mixed"
        entry["occasion"] = occasion
        entry["style_direction"] = style_direction
        entry["items"] = [dict(i) for i in (items or []) if isinstance(i, dict)]
        _REGISTRY[board_id] = entry


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
            entry = _new_entry(revision)
            _REGISTRY[board_id] = entry
        elif entry["revision"] != revision:
            return _error(
                "BOARD_REVISION_CONFLICT",
                "This board changed since you last saw it - refresh and try again.",
                current_revision=entry["revision"],
                requested_revision=revision,
            )
        stored_policy = canonical_board_policy(entry.get("source_policy"))
        stored_scenario = str(entry.get("scenario") or "").strip().lower()

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

    # --- Commit: bump revision + one-level undo snapshot ---------------------
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(board_id) or _new_entry(revision)
        previous_revision = entry["revision"]
        entry["previous"] = {"revision": previous_revision, "items": list(entry.get("items") or [])}
        entry["revision"] = previous_revision + 1
        entry["items"] = [dict(i) for i in out_items]
        # Persist the board policy across revisions - a second or third
        # shuffle of a Style This board must stay style-asset-based.
        entry["source_policy"] = resolved_policy
        entry["allow_wardrobe_fallback"] = allow_wardrobe_fallback
        if not entry.get("scenario"):
            entry["scenario"] = stored_scenario
        if occasion is not None:
            entry["occasion"] = occasion
        _REGISTRY[board_id] = entry
        new_revision = entry["revision"]
        response_scenario = entry.get("scenario") or None

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
        "source": result.get("source"),
        "changed_slots": result.get("changed_slots", []),
        "missing_items": result.get("missing_items", []),
        "board_items": out_items,
    }
