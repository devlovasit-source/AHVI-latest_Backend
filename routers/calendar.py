from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from middleware.auth_middleware import get_current_user
from models.calendar_models import CalendarEventInput, CalendarRuntimeResult
from brain.engines.calendar_runtime import run_calendar_runtime
from services.calendar_service import (
    calendar_plan_counts,
    calendar_event_to_runtime_input,
    create_calendar_event,
    enrich_event_with_intelligence,
    delete_calendar_event,
    get_calendar_event,
    list_calendar_events,
    list_today_calendar_events,
    list_upcoming_calendar_events,
    parse_plan_text_to_payload,
    update_calendar_event,
)
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


class CalendarEventCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    start_time: str = Field(..., min_length=4, max_length=80)
    end_time: str | None = None
    timezone: str | None = None
    type: str | None = "plan"
    source: str | None = "ahvi"
    status: str | None = "scheduled"
    dress_code: str | None = None
    venue_name: str | None = None
    venue_address: str | None = None
    reminder_minutes: int | None = 30
    metadata: Dict[str, Any] | str | None = None


class CalendarPlanTextCreateRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=500)
    category: str | None = Field(default="Plan", max_length=80)
    timezone: str | None = Field(default="Asia/Kolkata", max_length=80)


class CalendarEventUpdateRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    timezone: str | None = None
    type: str | None = None
    source: str | None = None
    status: str | None = None
    dress_code: str | None = None
    venue_name: str | None = None
    venue_address: str | None = None
    reminder_minutes: int | None = None
    metadata: Dict[str, Any] | str | None = None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_id(user: Any) -> str:
    uid = str((user or {}).get("user_id") or (user or {}).get("$id") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="Missing authenticated user")
    return uid


def _text_to_event(req: CalendarTextRequest) -> CalendarEventInput:
    start_iso = str(req.startAtISO or _iso_now())
    return CalendarEventInput(
        eventId="text_event",
        title=req.text,
        startAtISO=start_iso,
        endAtISO=req.endAtISO,
        timezone=req.timezone,
        dressCode=req.dressCode,
    )


@router.post("/events")
def create_event(req: CalendarEventCreateRequest, user=Depends(get_current_user)):
    try:
        uid = _user_id(user)
        event = create_calendar_event(uid, req.model_dump(exclude_none=True))
        extras = enrich_event_with_intelligence(uid, event)
        return {"success": True, "event": event, **extras}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        print("❌ /calendar/events create error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar event create failed")


@router.post("/events/from-text")
def create_event_from_text(req: CalendarPlanTextCreateRequest, user=Depends(get_current_user)):
    try:
        payload = parse_plan_text_to_payload(
            req.text,
            category=req.category,
            timezone_name=req.timezone or "Asia/Kolkata",
        )
        uid = _user_id(user)
        event = create_calendar_event(uid, payload)
        extras = enrich_event_with_intelligence(uid, event)
        return {"success": True, "event": event, **extras}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        print("calendar /events/from-text create error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar event create failed")


@router.get("/events")
def list_events(
    user=Depends(get_current_user),
    start_time: str | None = Query(default=None),
    end_time: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
):
    try:
        events = list_calendar_events(_user_id(user), start_time=start_time, end_time=end_time, limit=limit)
        return {
            "success": True,
            "events": events,
            "count": len(events),
            **calendar_plan_counts(events),
        }
    except Exception:
        print("❌ /calendar/events list error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar events list failed")


@router.get("/today")
def today_events(user=Depends(get_current_user), date: str | None = Query(default=None)):
    try:
        events = list_today_calendar_events(_user_id(user), date=date)
        return {"success": True, "events": events, "count": len(events)}
    except ValueError as exc:
        # Bad `date` query param is a client error, not a server failure.
        raise HTTPException(
            status_code=400, detail=f"Invalid calendar date: {date}"
        ) from exc
    except Exception:
        print("❌ /calendar/today error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar today failed")


@router.get("/events/today")
def today_events_alias(user=Depends(get_current_user), date: str | None = Query(default=None)):
    return today_events(user=user, date=date)


@router.get("/events/upcoming")
def upcoming_events(
    user=Depends(get_current_user),
    limit: int = Query(default=100, ge=1, le=300),
):
    try:
        events = list_upcoming_calendar_events(_user_id(user), limit=limit)
        return {"success": True, "events": events, "count": len(events)}
    except Exception:
        print("❌ /calendar/events/upcoming error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar upcoming failed")


@router.patch("/events/{event_id}")
def update_event(event_id: str, req: CalendarEventUpdateRequest, user=Depends(get_current_user)):
    try:
        event = update_calendar_event(_user_id(user), event_id, req.model_dump(exclude_none=True))
        return {"success": True, "event": event}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        print("❌ /calendar/events update error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar event update failed")


@router.delete("/events/{event_id}")
def delete_event(event_id: str, user=Depends(get_current_user)):
    try:
        result = delete_calendar_event(_user_id(user), event_id)
        return {"success": True, **result}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception:
        print("❌ /calendar/events delete error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar event delete failed")


@router.post("/events/{event_id}/runtime", response_model=CalendarRuntimeResult)
def runtime_saved_event(event_id: str, user=Depends(get_current_user)):
    try:
        user_id = _user_id(user)
        event = get_calendar_event(user_id, event_id)
        payload = calendar_event_to_runtime_input(event)
        return run_calendar_runtime(CalendarEventInput(**payload), user_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception:
        print("❌ /calendar/events/runtime error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar saved-event runtime failed")


@router.post("/runtime", response_model=CalendarRuntimeResult)
def runtime_event(req: CalendarEventInput, user=Depends(get_current_user)):
    try:
        return run_calendar_runtime(req, user_id=_user_id(user))
    except Exception:
        print("❌ /calendar/runtime error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar runtime failed")


@router.post("/process", response_model=CalendarRuntimeResult)
def process_text(req: CalendarTextRequest, user=Depends(get_current_user)):
    try:
        return run_calendar_runtime(_text_to_event(req), user_id=_user_id(user))
    except Exception:
        print("❌ /calendar/process error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar processing failed")


@router.post("/process/async", status_code=status.HTTP_202_ACCEPTED)
def process_text_async(http_request: Request, req: CalendarTextRequest, user=Depends(get_current_user)):
    if calendar_runtime_task is None:
        raise HTTPException(status_code=503, detail="Worker not configured")
    user_id = _user_id(user)
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
        user_id = _user_id(user)
        return [run_calendar_runtime(event, user_id=user_id) for event in (req.events or [])]
    except Exception:
        print("❌ /calendar/daily error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Calendar daily processing failed")


@router.post("/daily/async", status_code=status.HTTP_202_ACCEPTED)
def daily_runtime_async(http_request: Request, req: CalendarDailyRequest, user=Depends(get_current_user)):
    if calendar_daily_task is None:
        raise HTTPException(status_code=503, detail="Worker not configured")
    user_id = _user_id(user)
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
    return {"status": "ok", "engine": "calendar_runtime_v1", "storage": "appwrite_calendar_events", "auth": "enabled", "ready": True}
