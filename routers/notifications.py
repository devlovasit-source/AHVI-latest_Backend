from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from middleware.auth_middleware import get_current_user
from services.firebase_push_service import firebase_push_service
from services.medicine_reminder_dispatch import MedicineReminderDispatcher, scheduled_utc
from services.notification_store import (
    ReminderSchemaError,
    ReminderStoreError,
    notification_store,
)
from services.task_queue import enqueue_task

try:
    from worker import dispatch_due_reminders_task
except Exception:
    dispatch_due_reminders_task = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


def _is_production() -> bool:
    env = (
        str(os.getenv("ENV") or os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "")
        .strip()
        .lower()
    )
    return env in {"prod", "production"}


class RegisterDeviceRequest(BaseModel):
    platform: str = Field(..., min_length=2)  # android/ios/web
    token: str = Field(..., min_length=20)
    user_id: str = Field(default="")


class UnregisterDeviceRequest(BaseModel):
    token: str = Field(..., min_length=20)


class ScheduleRemindersRequest(BaseModel):
    eventId: str = Field(default="event", min_length=1)
    reminders: List[Dict[str, Any]] = Field(default_factory=list)
    source: str = "calendar"


def _require_dispatch_secret(request: Request) -> None:
    secret = str(os.getenv("NOTIFICATIONS_DISPATCH_SECRET", "")).strip()
    if not secret:
        if _is_production():
            logger.error(
                "NOTIFICATIONS_DISPATCH_SECRET is unset in production; rejecting dispatch call"
            )
            raise HTTPException(status_code=503, detail="dispatch not configured")
        logger.warning(
            "NOTIFICATIONS_DISPATCH_SECRET is unset; allowing open dispatch (dev only). "
            "Set NOTIFICATIONS_DISPATCH_SECRET before deploying to production."
        )
        return
    provided = str(request.headers.get("x-dispatch-secret", "")).strip()
    if provided != secret:
        raise HTTPException(status_code=401, detail="unauthorized")


def _durable_medicine_enabled() -> bool:
    return str(os.getenv("ENABLE_DURABLE_MED_REMINDERS", "false")).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _is_medicine_reminder(reminder: Dict[str, Any]) -> bool:
    return str(reminder.get("source") or "").strip().lower() in {"medicine", "medi"} or bool(
        reminder.get("medId") or reminder.get("med_id")
    )


def _dispatch_generic_reminders(due: List[Dict[str, Any]]) -> Dict[str, int]:
    """Legacy generic dispatch. Durable medicine records never enter here."""
    sent = failed = processed = 0
    for rem in due:
        processed += 1
        doc_id = str(rem.get("$id") or rem.get("id") or "")
        user_id = str(rem.get("userId") or "")
        message = str(rem.get("body") or rem.get("messageText") or rem.get("lastError") or rem.get("message") or "")
        devices = notification_store.list_devices(user_id=user_id)
        tokens = [str(d.get("token") or "").strip() for d in devices if str(d.get("token") or "").strip()]
        resp = firebase_push_service.send_to_tokens(tokens=tokens, title="AHVI", body=message, data={"type": "reminder"})
        if resp.get("success") and int(resp.get("sent") or 0) > 0:
            sent += int(resp.get("sent") or 0)
            if doc_id:
                notification_store.mark_reminder(reminder_doc_id=doc_id, status="sent")
        else:
            failed += int(resp.get("failed") or 1)
            if doc_id:
                notification_store.mark_reminder(reminder_doc_id=doc_id, status="failed", error=str(resp.get("error") or ""))
    return {"processed": processed, "sent": sent, "failed": failed}


def _dispatch_durable_medicine(*, dry_run: bool) -> Dict[str, int]:
    if hasattr(firebase_push_service, "ready") and not firebase_push_service.ready():
        raise HTTPException(status_code=503, detail="MED_REMINDER_FIREBASE_UNAVAILABLE")
    now = datetime.now(timezone.utc)
    dispatcher = MedicineReminderDispatcher(store=notification_store, firebase=firebase_push_service)
    try:
        occurrences = notification_store.list_medicine_occurrences_to_seed(
            earliest_iso=(now - timedelta(minutes=dispatcher.recovery_minutes)).isoformat(),
            latest_iso=(now + timedelta(minutes=15)).isoformat(),
        )
        seeded = 0
        for occurrence in occurrences:
            user_id = str(occurrence.get("userId") or occurrence.get("user_id") or "").strip()
            med_id = str(occurrence.get("medId") or occurrence.get("med_id") or "").strip()
            scheduled = str(occurrence.get("time") or occurrence.get("scheduledFor") or "").strip()
            try:
                local_scheduled = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
            except ValueError:
                continue
            if not user_id or not med_id:
                continue
            dispatcher.schedule_occurrence(
                user_id=user_id,
                medicine_id=med_id,
                scheduled_utc=scheduled_utc(local_scheduled, str(occurrence.get("timezone") or "Asia/Kolkata")),
                timezone_name=str(occurrence.get("timezone") or "Asia/Kolkata"),
                dry_run=dry_run,
            )
            seeded += 1
        counters = dispatcher.run(now=now, dry_run=dry_run)
        return {"seeded": seeded, **counters}
    except ReminderSchemaError as exc:
        raise HTTPException(status_code=503, detail="MED_REMINDER_SCHEMA_UNAVAILABLE") from exc
    except ReminderStoreError as exc:
        raise HTTPException(status_code=503, detail="MED_REMINDER_STORE_UNAVAILABLE") from exc


@router.get("/health")
def notifications_health():
    return {
        "success": True,
        "firebase": firebase_push_service.status(),
        "appwrite_resources": {
            "devices": notification_store.devices_resource,
            "reminders": notification_store.reminders_resource,
        },
    }


@router.post("/devices/register")
def register_device(req: RegisterDeviceRequest, request: Request):
    state_user = getattr(request.state, "user", None)
    user_id = str((state_user or {}).get("user_id") or req.user_id or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="missing user_id")
    doc_id = notification_store.upsert_device(
        user_id=user_id, platform=req.platform, token=req.token
    )
    if not doc_id:
        raise HTTPException(status_code=500, detail="device registration failed")
    return {"success": True, "device_id": doc_id}


@router.post("/devices/unregister")
def unregister_device(req: UnregisterDeviceRequest):
    ok = notification_store.delete_device(token=req.token)
    return {"success": True, "deleted": bool(ok)}


@router.post("/reminders/schedule")
def schedule_reminders(req: ScheduleRemindersRequest, user=Depends(get_current_user)):
    user_id = str((user or {}).get("user_id") or "")
    out = notification_store.schedule_reminders(
        user_id=user_id,
        event_id=req.eventId,
        reminders=req.reminders,
        source=req.source,
    )
    return {"success": True, "scheduled": int(out.get("scheduled") or 0)}


@router.post("/dispatch-due")
def dispatch_due(request: Request, window_seconds: int = 60, dry_run: bool = False):
    _require_dispatch_secret(request)

    try:
        due = notification_store.list_due_reminders(window_seconds=int(window_seconds))
        if not _durable_medicine_enabled():
            if dry_run:
                return {"success": True, "dry_run": True, "planned": len(due)}
            return {"success": True, **_dispatch_generic_reminders(due)}
        generic_due = [rem for rem in due if not _is_medicine_reminder(rem)]
        try:
            medicine = _dispatch_durable_medicine(dry_run=dry_run)
        except HTTPException as exc:
            if exc.status_code >= 500:
                raise HTTPException(
                    status_code=exc.status_code,
                    detail={"code": exc.detail, "generic_dispatch_deferred": True},
                ) from exc
            raise
        generic = {"processed": 0, "sent": 0, "failed": 0} if dry_run else _dispatch_generic_reminders(generic_due)
        return {"success": True, "dry_run": bool(dry_run), "generic": generic, "medicine": medicine}
    except HTTPException:
        raise
    except Exception:
        logger.error("dispatch-due failed")
        raise HTTPException(status_code=503, detail="NOTIFICATION_DISPATCH_UNAVAILABLE")


@router.post("/dispatch-due/async", status_code=status.HTTP_202_ACCEPTED)
def dispatch_due_async(http_request: Request, window_seconds: int = 60):
    if dispatch_due_reminders_task is None:
        raise HTTPException(status_code=503, detail="Worker not configured")

    _require_dispatch_secret(http_request)
    if _durable_medicine_enabled():
        raise HTTPException(status_code=503, detail="MED_REMINDER_ASYNC_UNAVAILABLE")

    task_id = enqueue_task(
        task_func=dispatch_due_reminders_task,
        args=[int(window_seconds)],
        kwargs={"request_id": str(getattr(http_request.state, "request_id", "") or "")},
        kind="notifications_dispatch_due",
        user_id="system",
        source="routers.notifications.dispatch_due_async",
        request_id=str(getattr(http_request.state, "request_id", "") or ""),
    )
    return {"success": True, "status": "queued", "task_id": task_id}
