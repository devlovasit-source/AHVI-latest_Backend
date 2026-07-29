from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from fastapi import APIRouter, Depends, HTTPException, Query
from middleware.auth_middleware import get_current_user
from services.diet_service import get_diet_recommendation
from services.data_access_service import get_user_profile

logger = logging.getLogger("ahvi.routers.diet")

router = APIRouter(prefix="/diet", tags=["diet"])

def _user_id(user: dict) -> str:
    uid = str(user.get("user_id") or user.get("$id") or user.get("id") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="Missing authenticated user")
    return uid

@router.get("/today")
def get_diet_today(
    meal_type: str | None = Query(default=None),
    timezone_override: str | None = Query(default=None, alias="timezone"),
    user=Depends(get_current_user)
):
    """
    GET /api/diet/today
    Return one deterministic, non-persisted recommendation for the authenticated user's current local date and relevant meal period.
    """
    user_id = _user_id(user)

    # Resolve profile
    try:
        user_profile = get_user_profile(user_id=user_id) or {}
    except Exception:
        user_profile = user or {}

    # Resolve canonical timezone and local date (with ZoneInfo)
    timezone_candidate = timezone_override or user_profile.get("timezone") or "Asia/Kolkata"
    try:
        tz = ZoneInfo(timezone_candidate)
        resolved_timezone = timezone_candidate
    except Exception:
        tz = ZoneInfo("Asia/Kolkata")
        resolved_timezone = "Asia/Kolkata"

    local_now = datetime.now(timezone.utc).astimezone(tz)
    local_date = local_now.date()
    local_hour = local_now.hour

    res = get_diet_recommendation(
        user_id=user_id,
        local_date=local_date,
        local_hour=local_hour,
        meal_type=meal_type,
        profile=user_profile
    )

    return {
        "date": local_date.isoformat(),
        "timezone": resolved_timezone,
        "recommendation": res.get("recommendation"),
        "status": res.get("status"),
        "reason": res.get("reason")
    }
