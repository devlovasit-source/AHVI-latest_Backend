from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import logging
import uuid
from datetime import datetime, timezone

from services.qdrant_service import qdrant_service
from services.embedding_service import encode_metadata
from services.auth_helpers import enforce_owner
from services.style_memory_service import _outfit_signature

router = APIRouter(prefix="/api/feedback")
logger = logging.getLogger("ahvi.feedback")


def _nested(data: Dict[str, Any], parent: str, key: str) -> Any:
    value = data.get(parent)
    if isinstance(value, dict):
        return value.get(key)
    return None


class ItemFeedbackRequest(BaseModel):
    item_id: str
    feedback: str  # up / down


@router.post("/item")
def feedback_item(request: ItemFeedbackRequest):
    fb = request.feedback.lower()

    if fb not in ["up", "down"]:
        raise HTTPException(status_code=400, detail="feedback must be up/down")

    try:
        qdrant_service.update_feedback(request.item_id, fb)
        return {"success": True, "message": "Item feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BoardFeedbackRequest(BaseModel):
    user_id: str
    action: str
    board_payload: Dict[str, Any]
    # Optional, additive: when the caller (e.g. DailyWear "Not for me") knows
    # the exact canonical item ids of the rejected outfit, passing them lets
    # this write carry a deterministic outfit_signature the scorer can match
    # against exactly. Existing callers that omit these keep working exactly
    # as before (signature stays empty, no exact-rejection penalty applies).
    item_ids: List[str] = Field(default_factory=list)
    occasion: str = ""
    reason: Optional[str] = None


@router.post("/board")
def feedback_board(request: BoardFeedbackRequest, http_request: Request):
    # Bind to the authenticated user; never trust the body user_id.
    # enforce_owner returns the authed id and 403s on a mismatched supplied id.
    user_id = enforce_owner(http_request, request.user_id)
    action = request.action.lower()
    passive_actions = {"shown", "saved", "dismissed", "regenerated", "clicked", "shared"}

    if action not in {"like", "dislike", *passive_actions}:
        raise HTTPException(
            status_code=400,
            detail="action must be like/dislike/shown/saved/dismissed/regenerated/clicked/shared",
        )

    try:
        board = request.board_payload or {}
        logger.info(
            "style_board.behavior user=%s action=%s board_id=%s title=%s signature=%s core_signature=%s archetype=%s direction=%s",
            user_id,
            action,
            board.get("board_id") or board.get("id") or board.get("card_id") or "",
            board.get("title") or board.get("label") or "",
            board.get("style_signature") or board.get("_style_signature") or _nested(board, "style_metadata", "style_signature"),
            board.get("core_style_signature") or board.get("_style_core_signature") or _nested(board, "style_metadata", "core_style_signature"),
            board.get("style_archetype") or _nested(board, "style_metadata", "style_archetype"),
            board.get("style_direction") or _nested(board, "style_metadata", "style_direction"),
        )

        if action in passive_actions:
            return {
                "success": True,
                "message": "Board behavior logged",
                "action": action,
            }

        item_ids = [str(x).strip() for x in (request.item_ids or []) if str(x).strip()]
        # Deterministic, order/dup-independent — same helper already used for
        # recommendation-repeat memory. Empty when no valid item ids were sent.
        outfit_signature = _outfit_signature([{"id": i} for i in item_ids])

        embedding = encode_metadata(request.board_payload)
        payload = {
            "source": "feedback.board",
            "memory_type": "liked" if action == "like" else "disliked",
            "action": action,
            "user_id": user_id,
            "board": str(request.board_payload.get("board") or ""),
            "type": str(request.board_payload.get("type") or ""),
            "item_ids": item_ids,
            "outfit_signature": outfit_signature,
            "board_id": board.get("board_id") or board.get("id") or board.get("card_id") or "",
            "occasion": request.occasion or "",
            "reason": request.reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Idempotency: for dislike/like WITH a real outfit signature, key the
        # point deterministically on (user, action, signature) so repeated
        # "Not for me" taps on the exact same outfit overwrite the same point
        # (refreshing the timestamp/reason) instead of multiplying influence.
        # Without a signature (existing untyped callers), fall back to a
        # random id — unchanged legacy behavior, one point per call.
        point_id = (
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"ahvi-feedback|{user_id}|{action}|{outfit_signature}"))
            if outfit_signature
            else None
        )
        # QdrantService.upsert_user_memory accepts (user_id, vector, payload);
        # carry the like/dislike signal inside the payload (no memory_type kwarg).
        qdrant_service.upsert_user_memory(
            user_id=user_id,
            vector=embedding,
            payload=payload,
            point_id=point_id,
        )

        return {
            "success": True,
            "message": "Board feedback recorded",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
