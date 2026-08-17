from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from middleware.auth_middleware import get_current_user
from services.firebase_push_service import firebase_push_service
from services.notification_store import (
    notification_store,
    NOTIFICATION_PREFERENCE_CATEGORIES,
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


class NotificationPreferenceRequest(BaseModel):
    enabled: bool


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


@router.get("/preferences")
def get_notification_preferences(
    user=Depends(get_current_user),
):
    user_id = str((user or {}).get("user_id") or "")

    if not user_id:
        raise HTTPException(status_code=401, detail="missing user_id")

    return {
        "success": True,
        "preferences": {
            category: notification_store.get_notification_preference(
                user_id=user_id,
                category=category,
            )
            for category in (
                "medi",
                "calendar",
                "style",
            )
        },
    }


@router.put("/preferences/{category}")
def set_notification_preference(
    category: str,
    req: NotificationPreferenceRequest,
    user=Depends(get_current_user),
):
    user_id = str((user or {}).get("user_id") or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="missing user_id")

    category = category.strip().lower()
    if category not in NOTIFICATION_PREFERENCE_CATEGORIES:
        raise HTTPException(status_code=400, detail="invalid category")

    ok = notification_store.set_notification_preference(
        user_id=user_id,
        category=category,
        enabled=req.enabled,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="failed to update preference")

    return {"success": True, "category": category, "enabled": req.enabled}


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
def schedule_reminders(
    req: ScheduleRemindersRequest,
    user=Depends(get_current_user),
):
    user_id = str((user or {}).get("user_id") or "")
    out = notification_store.schedule_reminders(
        user_id=user_id,
        event_id=req.eventId,
        reminders=req.reminders,
        source=req.source,
    )
    return {"success": True, "scheduled": int(out.get("scheduled") or 0)}


@router.post("/dispatch-due")
def dispatch_due(request: Request, window_seconds: int = 60):
    _require_dispatch_secret(request)

    try:
        due = notification_store.list_due_reminders(
            window_seconds=int(window_seconds)
        )
        sent = 0
        failed = 0
        suppressed = 0
        processed = 0
        duplicate_prevented = 0

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
            category = str(rem.get("source") or "").strip().lower()

            if not doc_id:
                # Can't be claimed without a stable id - skip rather than
                # risk an unguarded (unclaimable) send.
                failed += 1
                continue

            claim = notification_store.try_claim_reminder(reminder_doc_id=doc_id)
            if not claim.get("claimed"):
                if claim.get("reason") in ("already_sent", "already_processing"):
                    duplicate_prevented += 1
                continue

            enabled, pref_status = (
                notification_store.get_notification_preference_for_dispatch(
                    user_id=user_id, category=category
                )
            )

            if pref_status == "lookup_failed":
                # Don't guess - release the claim and leave the reminder
                # scheduled so the next dispatch cycle retries it.
                notification_store.release_claim(reminder_doc_id=doc_id)
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id,
                    status="scheduled",
                    error="preference_lookup_failed",
                )
                continue

            if pref_status in ("invalid_category", "invalid_user"):
                # Fail closed: a category outside medi/calendar/style (or a
                # reminder with no userId) is a producer bug, not a user
                # choice - keep it out of the "user_preference" bucket so
                # it's visible/greppable separately.
                suppressed += 1
                notification_store.finalize_claim(
                    reminder_doc_id=doc_id, status="suppressed"
                )
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id,
                    status="suppressed",
                    error=pref_status,
                )
                continue

            if not enabled:
                suppressed += 1
                notification_store.finalize_claim(
                    reminder_doc_id=doc_id, status="suppressed"
                )
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id,
                    status="suppressed",
                    error="user_preference",
                )
                continue

            devices = notification_store.list_devices(user_id=user_id)
            tokens = [
                str(d.get("token") or "").strip()
                for d in devices
                if str(d.get("token") or "").strip()
            ]
            resp = firebase_push_service.send_to_tokens(
                tokens=tokens,
                title=title,
                body=message,
                data={"type": "reminder"},
            )
            if resp.get("success") and int(resp.get("sent") or 0) > 0:
                sent += int(resp.get("sent") or 0)
                notification_store.finalize_claim(
                    reminder_doc_id=doc_id, status="sent"
                )
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id, status="sent"
                )
            else:
                failed += int(resp.get("failed") or 1)
                notification_store.finalize_claim(
                    reminder_doc_id=doc_id, status="failed"
                )
                notification_store.mark_reminder(
                    reminder_doc_id=doc_id,
                    status="failed",
                    error=str(resp.get("error") or ""),
                )

        return {
            "success": True,
            "processed": processed,
            "sent": sent,
            "failed": failed,
            "suppressed": suppressed,
            "duplicate_prevented": duplicate_prevented,
        }

    except Exception:
        logger.error("dispatch-due failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="dispatch failed")


@router.post("/dispatch-due/async", status_code=status.HTTP_202_ACCEPTED)
def dispatch_due_async(
    http_request: Request,
    window_seconds: int = 60,
):
    if dispatch_due_reminders_task is None:
        raise HTTPException(status_code=503, detail="Worker not configured")

    _require_dispatch_secret(http_request)

    task_id = enqueue_task(
        task_func=dispatch_due_reminders_task,
        args=[int(window_seconds)],
        kwargs={
            "request_id": str(
                getattr(http_request.state, "request_id", "") or ""
            )
        },
        kind="notifications_dispatch_due",
        user_id="system",
        source="routers.notifications.dispatch_due_async",
        request_id=str(
            getattr(http_request.state, "request_id", "") or ""
        ),
    )
    return {"success": True, "status": "queued", "task_id": task_id}