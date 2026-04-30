from __future__ import annotations

import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from middleware.auth_middleware import get_current_user
from models.calendar_models import CalendarEventInput, CalendarRuntimeResult
from brain.engines.calendar_runtime import run_calendar_runtime
from services.task_queue import enqueue_task

try:
    from worker import calendar_runtime_task
except Exception:
    calendar_runtime_task = None

try:
    from worker import calendar_daily_task
except Exception:
    calendar_daily_task = None

router = APIRouter(prefix="/calendar", tags=["calendar"])


class CalendarTextRequest(BaseModel):
    text: str = Field(..., min_length=2, description="User event text/title")
    startAtISO: str | None = None
    endAtISO: str | None = None
    timezone: str | None = None
    dressCode: str | None = None


class CalendarDailyRequest(BaseModel):
    events: list[CalendarEventInput] = Field(default_factory=list)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text_to_event(req: CalendarTextRequest) -> CalendarEventInput:
    # Keep it stable: if UI only provides text, treat it as a titled event starting "now".
    start_iso = str(req.startAtISO or _iso_now())
    return CalendarEventInput(
        eventId="text_event",
        title=req.text,
        startAtISO=start_iso,
        endAtISO=req.endAtISO,
        timezone=req.timezone,
        dressCode=req.dressCode,
    )


@router.post("/runtime", response_model=CalendarRuntimeResult)
def runtime_event(req: CalendarEventInput, user=Depends(get_current_user)):
    try:
        user_id = str((user or {}).get("user_id") or "")
        return run_calendar_runtime(req, user_id=user_id)
    except Exception:
        print("❌ /calendar/runtime error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar runtime failed")


@router.post("/process", response_model=CalendarRuntimeResult)
def process_text(req: CalendarTextRequest, user=Depends(get_current_user)):
    try:
        user_id = str((user or {}).get("user_id") or "")
        event = _text_to_event(req)
        return run_calendar_runtime(event, user_id=user_id)
    except Exception:
        print("❌ /calendar/process error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar processing failed")


@router.post("/process/async", status_code=status.HTTP_202_ACCEPTED)
def process_text_async(http_request: Request, req: CalendarTextRequest, user=Depends(get_current_user)):
    if calendar_runtime_task is None:
        raise HTTPException(status_code=503, detail="Worker not configured")

    user_id = str((user or {}).get("user_id") or "")
    event = _text_to_event(req)
    task_id = enqueue_task(
        task_func=calendar_runtime_task,
        args=[event.model_dump(), user_id],
        kwargs={"request_id": str(getattr(http_request.state, "request_id", "") or "")},
        kind="calendar_runtime",
        user_id=user_id,
        source="routers.calendar.process_text_async",
        request_id=str(getattr(http_request.state, "request_id", "") or ""),
    )
    return {"success": True, "status": "queued", "task_id": task_id}


@router.post("/daily", response_model=list[CalendarRuntimeResult])
def daily_runtime(req: CalendarDailyRequest, user=Depends(get_current_user)):
    try:
        user_id = str((user or {}).get("user_id") or "")
        return [run_calendar_runtime(event, user_id=user_id) for event in (req.events or [])]
    except Exception:
        print("❌ /calendar/daily error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar daily processing failed")


@router.post("/daily/async", status_code=status.HTTP_202_ACCEPTED)
def daily_runtime_async(http_request: Request, req: CalendarDailyRequest, user=Depends(get_current_user)):
    if calendar_daily_task is None:
        raise HTTPException(status_code=503, detail="Worker not configured")

    user_id = str((user or {}).get("user_id") or "")
    events_payload = [event.model_dump() for event in (req.events or [])]
    task_id = enqueue_task(
        task_func=calendar_daily_task,
        args=[{"events": events_payload}, user_id],
        kwargs={"request_id": str(getattr(http_request.state, "request_id", "") or "")},
        kind="calendar_daily_runtime",
        user_id=user_id,
        source="routers.calendar.daily_runtime_async",
        request_id=str(getattr(http_request.state, "request_id", "") or ""),
    )
    return {"success": True, "status": "queued", "task_id": task_id}


@router.get("/health")
def calendar_health():
    return {"status": "ok", "engine": "calendar_runtime_v1", "auth": "enabled", "ready": True}
