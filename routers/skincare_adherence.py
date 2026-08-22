from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain.engines import adherence_engine
from middleware.auth_middleware import get_current_user

router = APIRouter(prefix="/skincare", tags=["skincare"])
logger = logging.getLogger("ahvi.skincare")


class SkincareStatusRequest(BaseModel):
    routine: str = Field(default="", max_length=40)
    logId: str = Field(default="", max_length=80)


def _user_id(user: Any) -> str:
    uid = str((user or {}).get("user_id") or (user or {}).get("$id") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="Missing authenticated user")
    return uid


def _error(code: str, message: str, status_code: int = 400) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"success": False, "error": {"code": code, "message": message}},
    )


@router.get("/today")
def today(user=Depends(get_current_user)) -> Dict[str, Any]:
    uid = _user_id(user)
    try:
        data = adherence_engine.auto_reconcile_skincare_overdue(uid)
        return {"success": True, "data": data}
    except Exception as exc:
        logger.exception("skincare.today.error user_id=%s error=%s", uid, exc)
        raise _error("SKINCARE_TODAY_FAILED", "Could not load today's skincare logs.", 500)


@router.post("/reconcile")
def reconcile(user=Depends(get_current_user)) -> Dict[str, Any]:
    uid = _user_id(user)
    try:
        data = adherence_engine.auto_reconcile_skincare_overdue(uid)
        return {"success": True, "data": data}
    except Exception as exc:
        logger.exception("skincare.reconcile.error user_id=%s error=%s", uid, exc)
        raise _error("SKINCARE_RECONCILE_FAILED", "Could not reconcile skincare logs.", 500)


@router.post("/mark-completed")
def mark_completed(req: SkincareStatusRequest, user=Depends(get_current_user)) -> Dict[str, Any]:
    uid = _user_id(user)
    try:
        log = adherence_engine.mark_skincare_completed(uid, routine=req.routine, log_id=req.logId)
        return {"success": True, "data": {"log": log}}
    except ValueError as exc:
        raise _error("SKINCARE_MARK_COMPLETED_INVALID", str(exc), 400)
    except Exception as exc:
        logger.exception("skincare.mark_completed.error user_id=%s error=%s", uid, exc)
        raise _error("SKINCARE_MARK_COMPLETED_FAILED", "Could not complete skincare routine.", 500)


@router.post("/mark-missed")
def mark_missed(req: SkincareStatusRequest, user=Depends(get_current_user)) -> Dict[str, Any]:
    uid = _user_id(user)
    try:
        log = adherence_engine.mark_skincare_missed(uid, routine=req.routine, log_id=req.logId)
        return {"success": True, "data": {"log": log}}
    except ValueError as exc:
        raise _error("SKINCARE_MARK_MISSED_INVALID", str(exc), 400)
    except Exception as exc:
        logger.exception("skincare.mark_missed.error user_id=%s error=%s", uid, exc)
        raise _error("SKINCARE_MARK_MISSED_FAILED", "Could not mark skincare routine as missed.", 500)


@router.post("/mark-skipped")
def mark_skipped(req: SkincareStatusRequest, user=Depends(get_current_user)) -> Dict[str, Any]:
    uid = _user_id(user)
    try:
        log = adherence_engine.mark_skincare_skipped(uid, routine=req.routine, log_id=req.logId)
        return {"success": True, "data": {"log": log}}
    except ValueError as exc:
        raise _error("SKINCARE_MARK_SKIPPED_INVALID", str(exc), 400)
    except Exception as exc:
        logger.exception("skincare.mark_skipped.error user_id=%s error=%s", uid, exc)
        raise _error("SKINCARE_MARK_SKIPPED_FAILED", "Could not mark skincare routine as skipped.", 500)
