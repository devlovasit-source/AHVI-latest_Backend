"""Style learning endpoints (minimal). Records wear events so AHVI can start
remembering what the user actually wears."""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.auth_helpers import enforce_owner
from services.style_memory_service import record_wear

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
    user_id = enforce_owner(req, request.user_id)
    if not str(user_id or "").strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    item_ids = [str(x).strip() for x in (request.item_ids or []) if str(x).strip()]
    if not item_ids:
        raise HTTPException(status_code=400, detail="item_ids must be a non-empty list")
    try:
        from services.wear_event_service import WearEventService

        wes = WearEventService()
        res = wes.record_wear(
            user_id=user_id,
            item_ids=item_ids,
            source="daily_wear" if request.board_id else "wardrobe_item",
            entity_type="outfit" if request.board_id else "item",
            entity_id=request.board_id or item_ids[0],
            board_id=request.board_id,
            occurred_at=request.worn_at,
        )
        return {"success": True, **res}
    except Exception as exc:  # noqa: BLE001
        logger.warning("style.wear_today_failed user_id=%s err=%s", user_id, str(exc)[:160])
        raise HTTPException(status_code=500, detail="Failed to record wear event")
