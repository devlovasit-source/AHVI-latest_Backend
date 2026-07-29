from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from middleware.auth_middleware import get_current_user
from services.home_summary_service import generate_home_summary
from services.data_access_service import get_user_profile

logger = logging.getLogger("ahvi.routers.home")

router = APIRouter(prefix="/home", tags=["home"])

def _user_id(user: dict) -> str:
    uid = str(user.get("user_id") or user.get("$id") or user.get("id") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="Missing authenticated user")
    return uid

@router.get("/today-summary")
async def get_today_summary(
    request: Request,
    timezone_override: str | None = Query(default=None, alias="timezone"),
    user=Depends(get_current_user)
):
    """
    GET /api/home/today-summary
    Return concise, Home-ready summaries for all 5 cards.
    """
    user_id = _user_id(user)
    request_id = str(getattr(request.state, "request_id", "") or "")

    try:
        user_profile = get_user_profile(user_id=user_id) or {}
    except Exception:
        user_profile = user or {}

    summary = await generate_home_summary(
        user_id=user_id,
        user_profile=user_profile,
        timezone_override=timezone_override,
        request_id=request_id
    )
    return summary
