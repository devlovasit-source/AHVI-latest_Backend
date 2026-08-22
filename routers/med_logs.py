from __future__ import annotations

import traceback
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from middleware.auth_middleware import get_current_user
from services.med_log_service import create_med_log, list_med_logs, update_med_log_status

router = APIRouter(prefix="/med-logs", tags=["med-logs"])


class MedLogCreateRequest(BaseModel):
    medId: str = Field(..., min_length=1, max_length=50)
    medName: str = Field(..., min_length=1, max_length=255)
    dose: str = Field(..., min_length=1, max_length=50)
    time: str | None = None
    status: str = Field(..., min_length=1, max_length=50)


class MedLogStatusUpdateRequest(BaseModel):
    status: str = Field(..., min_length=1, max_length=50)


def _user_id(user: Any) -> str:
    uid = str((user or {}).get("user_id") or (user or {}).get("$id") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="Missing authenticated user")
    return uid


@router.post("")
def create_log(req: MedLogCreateRequest, user=Depends(get_current_user)):
    try:
        log = create_med_log(_user_id(user), req.model_dump(exclude_none=True))
        return {"success": True, "log": log}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        print("med_logs create error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Med log create failed")


@router.get("")
def list_logs(
    user=Depends(get_current_user),
    limit: int = Query(default=100, ge=1, le=500),
):
    try:
        logs = list_med_logs(_user_id(user), limit=limit)
        return {"success": True, "logs": logs, "count": len(logs)}
    except Exception:
        print("med_logs list error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Med logs list failed")


@router.patch("/{log_id}/status")
def update_log_status(log_id: str, req: MedLogStatusUpdateRequest, user=Depends(get_current_user)):
    try:
        log = update_med_log_status(_user_id(user), log_id, req.status)
        return {"success": True, "log": log}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        print("med_logs update error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Med log update failed")
