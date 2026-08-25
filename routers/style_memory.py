"""Style learning endpoints (minimal). Records wear events so AHVI can start
remembering what the user actually wears."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.auth_helpers import enforce_owner
from services.style_item_contract import canonical_item_id, canonical_item_role
from services.wear_event_service import record_wear as record_canonical_wear

logger = logging.getLogger("ahvi.routers.style_memory")

router = APIRouter(prefix="/api/style", tags=["style-memory"])


class WearTodayRequest(BaseModel):
    user_id: str
    board_id: str = ""
    item_ids: List[str] = Field(default_factory=list)
    occasion: str = ""
    worn_at: str = ""


@router.post("/wear-today")
def wear_today(req: Request, request: WearTodayRequest):
    """Compatibility endpoint. Delegates to the same canonical
    WearEventService as POST /api/wardrobe/items/{item_id}/wear, one call
    per item, so there is exactly one wear implementation. Response shape
    (recorded/item_ids/worn_at) is unchanged for existing callers."""
    user_id = enforce_owner(req, request.user_id)
    if not str(user_id or "").strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    # De-dup while preserving order, matching the previous record_wear contract.
    item_ids = list(
        dict.fromkeys(str(x).strip() for x in (request.item_ids or []) if str(x).strip())
    )
    if not item_ids:
        raise HTTPException(status_code=400, detail="item_ids must be a non-empty list")
    worn_at = str(request.worn_at or "").strip() or datetime.now(timezone.utc).isoformat()
    recorded = 0
    try:
        for item_id in item_ids:
            record_canonical_wear(
                user_id=user_id,
                item_id=item_id,
                occurred_at_iso=worn_at,
                source="style.wear_today",
                board_id=request.board_id,
                occasion=request.occasion,
            )
            recorded += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("style.wear_today_failed user_id=%s err=%s", user_id, str(exc)[:160])
        raise HTTPException(status_code=500, detail="Failed to record wear event")
    return {"success": True, "recorded": recorded, "item_ids": item_ids, "worn_at": worn_at}


class ChangeItemRequest(BaseModel):
    user_id: str
    board_id: str
    revision: int = 1
    # Full current outfit, same item shape used elsewhere (id/$id/item_id +
    # whatever descriptive fields the item already carries — name/category/
    # type/etc). Role is derived server-side via canonical_item_role, never
    # guessed or requested from the client.
    items: List[Dict[str, Any]] = Field(default_factory=list)
    old_item_id: str
    occasion: str = ""
    # No candidate_items field: the client sends board identity + the
    # selected item only. Eligible replacements are sourced server-side
    # from the authenticated user's own canonical wardrobe — the client
    # can never fabricate/bias the candidate pool.


def _internal_board_id(user_id: str, board_id: str) -> str:
    """Namespace the mutation-state key by authenticated user so two users'
    same client-visible board_id (DailyWear ids aren't guaranteed globally
    unique) can never collide in the shared durable store. Purely an
    internal storage key — never returned to the client."""
    return f"dailywear:{user_id}:{board_id}"


def _restore_client_board_id(value: Any, internal_id: str, client_id: str) -> Any:
    if isinstance(value, dict):
        return {k: _restore_client_board_id(v, internal_id, client_id) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_client_board_id(v, internal_id, client_id) for v in value]
    if value == internal_id:
        return client_id
    return value


def _canonical_wardrobe_candidates(user_id: str) -> List[Dict[str, Any]]:
    """Same canonical wardrobe source + sanitizer already used by the
    Style/DailyWear generation path (routers/stylist.py): Appwrite
    `outfits` collection -> sanitize_fashion_wardrobe_items. Reused as-is
    so "Change it" candidates go through the same fashion/role/source
    safety the rest of the app relies on — no second wardrobe query."""
    from services.appwrite_proxy import AppwriteProxy
    from services.wardrobe_sanitizer import sanitize_fashion_wardrobe_items

    try:
        raw_wardrobe = AppwriteProxy().list_documents("outfits", user_id=user_id)
    except Exception:
        raw_wardrobe = []
    return sanitize_fashion_wardrobe_items(
        raw_wardrobe if isinstance(raw_wardrobe, list) else [],
        source="style.change_item",
    )


def _rehydrate_target_from_wardrobe(
    items: List[Dict[str, Any]], old_item_id: str, wardrobe_by_id: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The item being changed (old_item_id) is the security-sensitive one —
    its metadata directly drives role resolution and therefore
    semantic_decision. Its client-supplied descriptive fields (name/
    category/role/etc) are NOT trusted: it is replaced wholesale by the
    authenticated user's own canonical wardrobe record for that id. A
    matching id with no wardrobe record keeps only its id, stripped of
    everything else, so it can't smuggle fabricated metadata either — it
    simply fails role resolution (fail-safe, not fail-open). Every other
    item in the outfit is left as the client sent it, unchanged from the
    existing contract — they aren't driving this request's role decision,
    and the mutation engine's own role/provenance checks still apply to
    them independently."""
    rehydrated = []
    for item in items:
        iid = canonical_item_id(item)
        if iid == old_item_id:
            canonical = wardrobe_by_id.get(iid)
            rehydrated.append(dict(canonical) if canonical is not None else {"id": iid})
        else:
            rehydrated.append(item)
    return rehydrated


@router.post("/change-item")
def change_item(req: Request, request: ChangeItemRequest):
    """Change it: adapts DailyWear's role-explicit "replace this piece"
    request onto the existing generic board mutation engine
    (services.style_board_mutation_service.handle_board_operation) rather
    than duplicating replacement logic. No NLU is needed here — the role
    being changed is derived directly from the exact item the user tapped,
    via the same canonical_item_role the engine itself uses for candidate
    matching, so the resolved role can never disagree with the engine.

    Item metadata is never trusted from the client for role resolution: the
    changed item's id is rehydrated against the authenticated user's own
    canonical wardrobe before its role is resolved (see
    _rehydrate_target_from_wardrobe) — a client can supply a real, owned
    item id but cannot control what name/category/role it resolves to.

    Mutation state is stored under a per-user-namespaced internal board id
    (see _internal_board_id) so two users' identical client-visible
    board_id can never collide or leak into each other's revision history;
    the response is translated back to the client's original board_id
    before returning.

    Revision authority: if the durable store is ahead of the client's
    revision (e.g. the app was closed/reopened and lost its in-memory
    revision), the mutation is retried exactly once against the store's
    current revision rather than surfacing a stale-revision error the
    client has no way to self-heal from. This does not weaken the shared
    engine's concurrency protection — the retry still goes through the
    same optimistic-concurrency check, it just gives the caller one
    server-resolved second attempt instead of zero.

    Persists a "changed_item" memory event only after the mutation is
    actually accepted — a failed/no-op mutation never creates a feedback
    signal.
    """
    user_id = enforce_owner(req, request.user_id)
    if not str(user_id or "").strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    board_id = str(request.board_id or "").strip()
    if not board_id:
        raise HTTPException(status_code=400, detail="board_id is required")
    old_item_id = str(request.old_item_id or "").strip()
    if not old_item_id:
        raise HTTPException(status_code=400, detail="old_item_id is required")
    raw_items = [it for it in (request.items or []) if isinstance(it, dict)]
    if not raw_items:
        raise HTTPException(status_code=400, detail="items must be a non-empty list")

    wardrobe_candidates = _canonical_wardrobe_candidates(user_id)
    wardrobe_by_id = {
        canonical_item_id(w): w for w in wardrobe_candidates if canonical_item_id(w)
    }
    items = _rehydrate_target_from_wardrobe(raw_items, old_item_id, wardrobe_by_id)

    target_item = next(
        (it for it in items if canonical_item_id(it) == old_item_id), None
    )
    if target_item is None:
        # Fail safe rather than guess: never resolve a role for an item that
        # isn't actually part of the outfit the user is looking at.
        raise HTTPException(
            status_code=400, detail="old_item_id is not part of the supplied items"
        )
    role = canonical_item_role(target_item)
    if not role or role == "unknown":
        raise HTTPException(
            status_code=422,
            detail="Could not determine a canonical role for this item; refusing to guess",
        )

    from services.style_board_mutation_service import handle_board_operation

    internal_board_id = _internal_board_id(user_id, board_id)

    def _attempt(revision: int) -> Dict[str, Any]:
        payload = {
            "board_id": internal_board_id,
            "revision": revision,
            "items": items,
            "board_items": items,
            "interaction_mode": "daily_wear",
            "occasion": request.occasion or "",
            "candidate_items": wardrobe_candidates,
        }
        semantic_decision = {"operation": {"type": "modify", "replace_roles": [role]}}
        return handle_board_operation(payload, user_id=user_id, semantic_decision=semantic_decision)

    try:
        result = _attempt(int(request.revision or 1))
        if (
            isinstance(result, dict)
            and result.get("error", {}).get("code") == "STALE_BOARD_REVISION"
        ):
            current_revision = result["error"].get("current_revision")
            if isinstance(current_revision, int) and current_revision > 0:
                # One server-resolved retry against the durable store's
                # actual current revision — bounded, not a loop.
                result = _attempt(current_revision)
    except Exception as exc:  # noqa: BLE001
        logger.warning("style.change_item_failed user_id=%s board_id=%s err=%s", user_id, board_id, str(exc)[:160])
        raise HTTPException(status_code=500, detail="Failed to change item")

    if result is None:
        raise HTTPException(status_code=500, detail="Mutation engine did not recognize this request")

    result = _restore_client_board_id(result, internal_board_id, board_id)

    if not result.get("success", False):
        # Mutation was rejected (e.g. NO_VALID_REPLACEMENT, STALE_BOARD_REVISION) —
        # nothing was persisted, no feedback memory is written. Pass the
        # engine's own typed error straight through.
        return result

    changed_ids = list((result.get("data") or {}).get("changed_item_ids") or [])
    to_item_id = changed_ids[0] if changed_ids else ""
    # Board mutation success is never transactional with memory persistence:
    # the mutation has already been accepted and returned above this point
    # regardless of what happens next. learning_persisted is purely
    # informational for the caller — it never causes a rollback.
    learning_persisted = False
    if to_item_id:
        try:
            learning_persisted = _record_change_item_memory(
                user_id=user_id,
                from_item_id=old_item_id,
                to_item_id=to_item_id,
                role=role,
                occasion=request.occasion or "",
                board_id=board_id,
            )
        except Exception:  # noqa: BLE001
            # Memory write is best-effort; a failed write must never undo or
            # mask an already-accepted board mutation.
            logger.warning("style.change_item_memory_write_failed user_id=%s", user_id)
    result["learning_persisted"] = learning_persisted

    return result


def _record_change_item_memory(
    *, user_id: str, from_item_id: str, to_item_id: str, role: str, occasion: str, board_id: str
) -> bool:
    from services.qdrant_service import qdrant_service, user_memory_sentinel_vector

    payload: Dict[str, Any] = {
        "source": "style.change_item",
        "memory_type": "changed_item",
        "user_id": user_id,
        "from_item_id": from_item_id,
        "to_item_id": to_item_id,
        "role": role,
        "occasion": occasion,
        "board_id": board_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Same rationale as routers/feedback.py: user_memory is read back purely
    # by payload filter, never vector similarity, so a fixed sentinel vector
    # is sufficient and avoids the (deliberately absent) sentence-transformers
    # dependency.
    return qdrant_service.upsert_user_memory(
        user_id=user_id, vector=user_memory_sentinel_vector(), payload=payload
    )
