import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from services.auth_helpers import enforce_owner
from services.embedding_service import encode_metadata
from services.qdrant_service import qdrant_service
from services.style_feedback_store import (
    AppwriteStyleFeedbackStore,
    FeedbackStoreError,
    FeedbackValidationError,
    canonical_event,
)

router = APIRouter(prefix="/api/feedback")
logger = logging.getLogger("ahvi.feedback")


class _FeedbackModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class ItemFeedbackRequest(_FeedbackModel):
    event_id: str = Field(default="", alias="eventId")
    item_id: str = Field(alias="itemId")
    feedback: str  # up/down or like/dislike
    user_id: Optional[str] = Field(default=None, alias="userId")
    occasion: str = ""
    source_policy: str = Field(default="", alias="sourcePolicy")


class BoardFeedbackRequest(_FeedbackModel):
    event_id: str = Field(default="", alias="eventId")
    action: str
    board_payload: Dict[str, Any] = Field(default_factory=dict, alias="boardPayload")
    user_id: Optional[str] = Field(default=None, alias="userId")
    board_id: str = Field(default="", alias="boardId")
    item_ids: List[str] = Field(default_factory=list, alias="itemIds")
    occasion: str = ""
    source_policy: str = Field(default="", alias="sourcePolicy")


def _persist(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return AppwriteStyleFeedbackStore().append(event)
    except FeedbackStoreError as exc:
        logger.error("style_feedback.store_unavailable err=%s", str(exc)[:180])
        raise HTTPException(
            status_code=503,
            detail={
                "code": "FEEDBACK_STORE_UNAVAILABLE",
                "message": "Style feedback could not be stored durably.",
            },
        ) from exc


def _event_or_400(**kwargs: Any) -> Dict[str, Any]:
    try:
        return canonical_event(**kwargs)
    except FeedbackValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _compat_event_id(user_id: str, action: str, payload: Any) -> str:
    """Stable retry key for pre-eventId clients; explicit client IDs take priority."""
    try:
        encoded = json.dumps(
            {"user": user_id, "action": action, "payload": payload},
            sort_keys=True, separators=(",", ":"), default=str,
        ).encode()
    except Exception:
        encoded = f"{user_id}|{action}|{payload!s}".encode()
    return f"compat-{hashlib.sha256(encoded).hexdigest()[:32]}"


def _mirror_board_best_effort(user_id: str, event: Dict[str, Any]) -> None:
    """Qdrant is a semantic mirror only; Appwrite success is never rolled back."""
    if event.get("action") not in {"like", "dislike"}:
        return
    try:
        compact = json.loads(event.get("payload") or "{}")
        embedding = encode_metadata(compact)
        qdrant_service.upsert_user_memory(
            user_id=user_id,
            vector=embedding,
            payload={
                "source": "feedback.board",
                "memory_type": "liked" if event["action"] == "like" else "disliked",
                "action": event["action"],
                "user_id": user_id,
                "board_id": event.get("boardId") or "",
                "item_ids": event.get("itemIds") or "[]",
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("style_feedback.qdrant_mirror_failed err=%s", str(exc)[:180])


@router.post("/item")
def feedback_item(request: ItemFeedbackRequest, http_request: Request):
    user_id = enforce_owner(http_request, request.user_id)
    action = {"up": "like", "down": "dislike"}.get(
        request.feedback.strip().lower(), request.feedback.strip().lower()
    )
    event_id = request.event_id or _compat_event_id(
        user_id, action, {"itemId": request.item_id}
    )
    event = _event_or_400(
        user_id=user_id,
        event_id=event_id,
        action=action,
        item_ids=[request.item_id],
        occasion=request.occasion,
        source_policy=request.source_policy,
    )
    stored = _persist(event)
    return {
        "success": True,
        "message": "Item feedback recorded",
        "action": action,
        "idempotent": bool(stored.get("idempotent")),
    }


@router.post("/board")
def feedback_board(request: BoardFeedbackRequest, http_request: Request):
    user_id = enforce_owner(http_request, request.user_id)
    action = request.action.strip().lower()
    passive_actions = {"shown", "dismissed", "regenerated", "clicked", "shared"}
    if action not in {"like", "dislike", "saved", *passive_actions}:
        raise HTTPException(
            status_code=400,
            detail="action must be like/dislike/saved/shown/dismissed/regenerated/clicked/shared",
        )
    board = request.board_payload or {}
    logger.info(
        "style_board.behavior user=%s action=%s board_id=%s",
        user_id, action, request.board_id or board.get("board_id") or board.get("id") or "",
    )
    if action in passive_actions:
        return {"success": True, "message": "Board behavior logged", "action": action}

    event_id = request.event_id or _compat_event_id(
        user_id,
        action,
        {
            "boardId": request.board_id or board.get("board_id") or board.get("id") or "",
            "itemIds": request.item_ids,
            "board": board,
        },
    )
    event = _event_or_400(
        user_id=user_id,
        event_id=event_id,
        action=action,
        board_payload=board,
        item_ids=request.item_ids,
        board_id=request.board_id,
        occasion=request.occasion,
        source_policy=request.source_policy,
    )
    stored = _persist(event)  # Durable first; only then report/mirror learning.
    _mirror_board_best_effort(user_id, event)
    return {
        "success": True,
        "message": "Board feedback recorded",
        "action": action,
        "idempotent": bool(stored.get("idempotent")),
    }
