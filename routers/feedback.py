from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from services.qdrant_service import qdrant_service
from services.embedding_service import encode_metadata
router = APIRouter(prefix="/api/feedback")


# =========================
# ITEM FEEDBACK (KEEP)
# =========================
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

        return {
            "success": True,
            "message": "Item feedback recorded"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 🔥 BOARD FEEDBACK (NEW)
# =========================
class BoardFeedbackRequest(BaseModel):
    user_id: str
    action: str  # like / dislike
    board_payload: Dict[str, Any]


@router.post("/board")
def feedback_board(request: BoardFeedbackRequest):

    action = request.action.lower()

    if action not in ["like", "dislike"]:
        raise HTTPException(status_code=400, detail="action must be like/dislike")

    try:
        # 🔥 Build embedding from board
        embedding = encode_metadata(request.board_payload)

        # 🔥 Store in Qdrant
        point_id = qdrant_service.upsert_user_memory(
            user_id=request.user_id,
            vector=embedding,
            memory_type="liked" if action == "like" else "disliked",
            payload={
                "source": "feedback.board",
                "board": str(request.board_payload.get("board") or ""),
                "type": str(request.board_payload.get("type") or ""),
            },
        )

        return {
            "success": True,
            "message": "Board feedback recorded",
            "memory_point_id": point_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
