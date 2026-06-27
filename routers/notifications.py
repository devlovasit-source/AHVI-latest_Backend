from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from middleware.auth_middleware import get_current_user
from services.firebase_push_service import firebase_push_service
from services.notification_store import notification_store
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


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_medicine_reminder(reminder: Dict[str, Any]) -> bool:
    source = str(reminder.get("source") or "").strip().lower()
    return source in {"medicine", "medi"} or bool(
        reminder.get("medId") or reminder.get("med_id")
    )


def _notification_key(reminder: Dict[str, Any]) -> str:
    key = str(
        reminder.get("notificationKey") or reminder.get("notification_key") or ""
    ).strip()
    if key:
        return key
    user_id = str(reminder.get("userId") or reminder.get("user_id") or "").strip()
    med_id = str(
        reminder.get("medId") or reminder.get("med_id") or reminder.get("eventId") or ""
    ).strip()
    scheduled_for = str(reminder.get("sendAtISO") or reminder.get("scheduledFor") or "").strip()
    if user_id and med_id and scheduled_for:
        return f"med:{user_id}:{med_id}:{scheduled_for}"
    return ""


def _push_payload(reminder: Dict[str, Any], reminder_id: str) -> Dict[str, str]:
    if not _is_medicine_reminder(reminder):
        return {"type": "reminder"}
    med_id = str(
        reminder.get("medId") or reminder.get("med_id") or reminder.get("eventId") or ""
    ).strip()
    scheduled_for = str(reminder.get("sendAtISO") or reminder.get("scheduledFor") or "").strip()
    return {
        "type": "reminder",
        "action": "med_reminder",
        "medId": med_id,
        "medName": str(
            reminder.get("medName") or reminder.get("med_name") or "Medicine"
        ),
        "scheduledFor": scheduled_for,
        "screen": "medi",
        "deepLink": f"ahvi://medi/reminder/{reminder_id}",
    }


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
def dispatch_due(
    request: Request,
    window_seconds: int = 60,
    window_minutes: int | None = None,
    dry_run: bool = False,
):
    _require_dispatch_secret(request)

    try:
        if window_minutes is not None:
            window_seconds = max(1, int(window_minutes)) * 60
        dry_run = bool(dry_run) or _truthy(request.query_params.get("dry_run"))

        due = notification_store.list_due_reminders(window_seconds=int(window_seconds))
        due.extend(
            notification_store.list_due_medicine_reminders(window_seconds=int(window_seconds))
        )
        sent = 0
        failed = 0
        skipped = 0
        processed = 0
        due_items: List[Dict[str, Any]] = []

        for rem in due:
            processed += 1
            doc_id = str(rem.get("$id") or rem.get("id") or "")
            user_id = str(rem.get("userId") or "")
            message = str(
                rem.get("body")
                or rem.get("messageText")
                or rem.get("lastError")
                or rem.get("message")
                or ""
            )
            title = "AHVI"
            is_medicine = _is_medicine_reminder(rem)
            notification_key = _notification_key(rem) if is_medicine else ""

            if (
                is_medicine
                and notification_key
                and notification_store.was_notification_sent(
                    notification_key=notification_key
                )
            ):
                skipped += 1
                logger.info(
                    "AHVI_MED_REMINDER_DUPLICATE_SKIPPED user_id=%s med_id=%s scheduled_for=%s key=%s",
                    user_id,
                    rem.get("medId") or rem.get("med_id") or "",
                    rem.get("sendAtISO") or "",
                    notification_key,
                )
                continue

            payload = _push_payload(rem, doc_id)
            if dry_run:
                due_items.append(
                    {
                        "id": doc_id,
                        "userId": user_id,
                        "source": rem.get("source") or "",
                        "medId": payload.get("medId") or "",
                        "medName": payload.get("medName") or "",
                        "scheduledFor": payload.get("scheduledFor")
                        or rem.get("sendAtISO")
                        or "",
                        "notificationKey": notification_key,
                    }
                )
                if is_medicine:
                    logger.info(
                        "AHVI_MED_REMINDER_DISPATCH_DRY_RUN user_id=%s med_id=%s scheduled_for=%s key=%s",
                        user_id,
                        payload.get("medId") or "",
                        payload.get("scheduledFor") or "",
                        notification_key,
                    )
                continue

            devices = notification_store.list_devices(user_id=user_id)
            tokens = [
                str(d.get("token") or "").strip()
                for d in devices
                if str(d.get("token") or "").strip()
            ]
            if not tokens:
                skipped += 1
                if is_medicine:
                    logger.info(
                        "AHVI_MED_REMINDER_FAILED user_id=%s med_id=%s scheduled_for=%s reason=no_tokens",
                        user_id,
                        payload.get("medId") or "",
                        payload.get("scheduledFor") or "",
                    )
                continue
            resp = firebase_push_service.send_to_tokens(
                tokens=tokens, title=title, body=message, data=payload
            )
            if resp.get("success") and int(resp.get("sent") or 0) > 0:
                sent += int(resp.get("sent") or 0)
                if is_medicine and notification_key:
                    notification_store.mark_medicine_reminder(
                        reminder_doc_id=doc_id,
                        user_id=user_id,
                        notification_key=notification_key,
                        send_at_iso=str(rem.get("sendAtISO") or ""),
                        message=message,
                        status="sent",
                    )
                    logger.info(
                        "AHVI_MED_REMINDER_SENT user_id=%s med_id=%s scheduled_for=%s sent=%s key=%s",
                        user_id,
                        payload.get("medId") or "",
                        payload.get("scheduledFor") or "",
                        int(resp.get("sent") or 0),
                        notification_key,
                    )
                elif doc_id:
                    notification_store.mark_reminder(
                        reminder_doc_id=doc_id, status="sent"
                    )
            else:
                failed += int(resp.get("failed") or 1)
                if is_medicine and notification_key:
                    notification_store.mark_medicine_reminder(
                        reminder_doc_id=doc_id,
                        user_id=user_id,
                        notification_key=notification_key,
                        send_at_iso=str(rem.get("sendAtISO") or ""),
                        message=message,
                        status="failed",
                        error=str(resp.get("error") or ""),
                    )
                    logger.info(
                        "AHVI_MED_REMINDER_FAILED user_id=%s med_id=%s scheduled_for=%s error=%s",
                        user_id,
                        payload.get("medId") or "",
                        payload.get("scheduledFor") or "",
                        str(resp.get("error") or ""),
                    )
                elif doc_id:
                    notification_store.mark_reminder(
                        reminder_doc_id=doc_id,
                        status="failed",
                        error=str(resp.get("error") or ""),
                    )

        return {
            "success": True,
            "dry_run": dry_run,
            "window_seconds": int(window_seconds),
            "processed": processed,
            "sent": sent,
            "failed": failed,
            "skipped": skipped,
            "due": due_items if dry_run else [],
        }

    except Exception:
        logger.error("dispatch-due failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="dispatch failed")


@router.post("/dispatch-due/async", status_code=status.HTTP_202_ACCEPTED)
def dispatch_due_async(http_request: Request, window_seconds: int = 60):
    if dispatch_due_reminders_task is None:
        raise HTTPException(status_code=503, detail="Worker not configured")

    _require_dispatch_secret(http_request)

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
