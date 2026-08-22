from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain.engines import adherence_engine
from middleware.auth_middleware import get_current_user

router = APIRouter(prefix="/medi", tags=["medi"])
logger = logging.getLogger("ahvi.medi")


class MedStatusRequest(BaseModel):
    medId: str = Field(default="")
    logId: str = Field(default="")


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
        data = adherence_engine.auto_reconcile_overdue(uid)
        return {"success": True, "data": data}
    except Exception as exc:
        logger.exception("medi.today.error user_id=%s error=%s", uid, exc)
        raise _error("MEDI_TODAY_FAILED", "Could not load today's medicine logs.", 500)


@router.post("/reconcile")
def reconcile(user=Depends(get_current_user)) -> Dict[str, Any]:
    uid = _user_id(user)
    try:
        data = adherence_engine.auto_reconcile_overdue(uid)
        return {"success": True, "data": data}
    except Exception as exc:
        logger.exception("medi.reconcile.error user_id=%s error=%s", uid, exc)
        raise _error("MEDI_RECONCILE_FAILED", "Could not reconcile medicine logs.", 500)


@router.post("/mark-taken")
def mark_taken(req: MedStatusRequest, user=Depends(get_current_user)) -> Dict[str, Any]:
    uid = _user_id(user)
    try:
        log = adherence_engine.mark_taken(uid, med_id=req.medId, log_id=req.logId)
        return {"success": True, "data": {"log": log}}
    except ValueError as exc:
        raise _error("MEDI_MARK_TAKEN_INVALID", str(exc), 400)
    except Exception as exc:
        logger.exception("medi.mark_taken.error user_id=%s error=%s", uid, exc)
        raise _error("MEDI_MARK_TAKEN_FAILED", "Could not mark medicine as taken.", 500)


@router.post("/mark-missed")
def mark_missed(req: MedStatusRequest, user=Depends(get_current_user)) -> Dict[str, Any]:
    uid = _user_id(user)
    try:
        log = adherence_engine.mark_missed(uid, med_id=req.medId, log_id=req.logId)
        return {"success": True, "data": {"log": log}}
    except ValueError as exc:
        raise _error("MEDI_MARK_MISSED_INVALID", str(exc), 400)
    except Exception as exc:
        logger.exception("medi.mark_missed.error user_id=%s error=%s", uid, exc)
        raise _error("MEDI_MARK_MISSED_FAILED", "Could not mark medicine as missed.", 500)


@router.post("/mark-skipped")
def mark_skipped(req: MedStatusRequest, user=Depends(get_current_user)) -> Dict[str, Any]:
    uid = _user_id(user)
    try:
        log = adherence_engine.mark_skipped(uid, med_id=req.medId, log_id=req.logId)
        return {"success": True, "data": {"log": log}}
    except ValueError as exc:
        raise _error("MEDI_MARK_SKIPPED_INVALID", str(exc), 400)
    except Exception as exc:
        logger.exception("medi.mark_skipped.error user_id=%s error=%s", uid, exc)
        raise _error("MEDI_MARK_SKIPPED_FAILED", "Could not mark medicine as skipped.", 500)
