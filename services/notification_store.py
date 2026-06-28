from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _hash_id(prefix: str, raw: str, *, length: int = 32) -> str:
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}_{digest[: max(8, min(length, 48))]}"


def _tz(name: str = "Asia/Kolkata"):
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(name or "Asia/Kolkata")
    except Exception:
        return ZoneInfo("Asia/Kolkata")


def _parse_time_for_today(raw: Any, *, now: datetime, tz_name: str) -> Optional[datetime]:
    text = _safe_text(raw)
    if not text:
        return None
    zone = _tz(tz_name)
    lowered = text.lower().replace(".", "").strip()
    for fmt in ("%H:%M", "%I:%M %p", "%I %p"):
        try:
            parsed = datetime.strptime(lowered.upper(), fmt).time()
            return datetime.combine(now.astimezone(zone).date(), parsed, tzinfo=zone)
        except Exception:
            pass
    try:
        parsed_iso = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed_iso.tzinfo is None:
            parsed_iso = parsed_iso.replace(tzinfo=zone)
        return parsed_iso.astimezone(zone)
    except Exception:
        return None


class NotificationStore:
    """
    Stores:
    - device tokens in Appwrite collection: notification_devices
    - reminders in Appwrite collection: notification_reminders

    This is intentionally simple (MVP). For scale, move due-reminder queries to a DB index.
    """

    def __init__(self) -> None:
        self._appwrite = AppwriteProxy()
        self.devices_resource = (
            os.getenv("APPWRITE_COLLECTION_NOTIFICATION_DEVICES", "")
            or os.getenv("APPWRITE_RESOURCE_NOTIFICATION_DEVICES", "")
            or "notification_devices"
        )
        self.reminders_resource = (
            os.getenv("APPWRITE_COLLECTION_NOTIFICATION_REMINDERS", "")
            or os.getenv("APPWRITE_RESOURCE_NOTIFICATION_REMINDERS", "")
            or "notification_reminders"
        )
        self.max_scan = max(
            50, int(os.getenv("NOTIFICATION_REMINDER_SCAN_LIMIT", "500"))
        )

    # -------------------------
    # Devices
    # -------------------------
    def upsert_device(self, *, user_id: str, platform: str, token: str) -> str | None:
        uid = _safe_text(user_id)
        tok = _safe_text(token)
        plat = _safe_text(platform).lower() or "unknown"
        if not uid or not tok:
            return None

        # De-dupe by the real registration identity. The old token-only id could
        # still leave multiple rows when debug builds generated fresh dev ids;
        # this keeps repeated launches for the same device as updates.
        doc_id = _hash_id("dev", f"{uid}|{plat}|{tok}", length=28)
        data = {
            "userId": uid,
            "platform": plat,
            "token": tok,
            "updatedAtISO": _utcnow().isoformat(),
        }
        try:
            self._appwrite.update_document(self.devices_resource, doc_id, data)
        except Exception:
            try:
                self._appwrite.create_document(
                    self.devices_resource, data, document_id=doc_id
                )
            except Exception:
                return None
        return doc_id

    def delete_device(self, *, token: str) -> bool:
        tok = _safe_text(token)
        if not tok:
            return False
        doc_id = _hash_id("dev", tok, length=28)
        try:
            self._appwrite.delete_document(self.devices_resource, doc_id)
            return True
        except Exception:
            return False

    def list_devices(self, *, user_id: str) -> List[Dict[str, Any]]:
        uid = _safe_text(user_id)
        if not uid:
            return []
        try:
            rows = self._appwrite.list_documents(
                self.devices_resource, user_id=uid, limit=200
            )
            return [r for r in rows if isinstance(r, dict)]
        except Exception:
            return []

    # -------------------------
    # Reminders
    # -------------------------
    def schedule_reminders(
        self,
        *,
        user_id: str,
        event_id: str,
        reminders: List[Dict[str, Any]],
        source: str = "calendar",
    ) -> Dict[str, Any]:
        uid = _safe_text(user_id)
        eid = _safe_text(event_id) or "event"
        if not uid:
            return {"success": False, "scheduled": 0}

        scheduled = 0
        for r in reminders or []:
            if not isinstance(r, dict):
                continue
            send_at = _safe_text(r.get("sendAtISO") or r.get("send_at") or "")
            message = _safe_text(r.get("message") or "")
            if not send_at or not message:
                continue

            doc_id = _hash_id("rem", f"{uid}|{eid}|{send_at}|{message}", length=32)
            source_value = _safe_text(source)
            status_value = _safe_text(r.get("status") or "scheduled")
            if source_value in {"medicine", "medi"} and status_value == "pending":
                status_value = "scheduled"
            data = {
                "userId": uid,
                "eventId": eid,
                # Default scheduled (not pending) so the dispatch task, which
                # only picks status=="scheduled", can actually send them.
                "status": status_value,
                "priority": _safe_text(r.get("priority") or "normal"),
                "toneProfile": _safe_text(r.get("toneProfile") or ""),
                "offsetMinutes": int(r.get("offsetMinutes") or 0),
                "message": message,
                "sendAtISO": send_at,
                "source": source_value,
                "lastError": _safe_text(r.get("lastError") or ""),
                "updatedAtISO": _utcnow().isoformat(),
            }
            base_data = dict(data)
            for key in (
                "medId",
                "medName",
                "dose",
                "scheduledFor",
                "title",
                "body",
                "notificationKey",
            ):
                value = r.get(key)
                if _safe_text(value):
                    data[key] = _safe_text(value)
            try:
                self._appwrite.update_document(self.reminders_resource, doc_id, data)
                scheduled += 1
            except Exception as exc:
                logger.warning(
                    "AHVI_REMINDER_UPDATE_FAILED user_id=%s event_id=%s doc_id=%s send_at=%s error=%s",
                    uid,
                    eid,
                    doc_id,
                    send_at,
                    exc,
                )
                try:
                    self._appwrite.create_document(
                        self.reminders_resource, data, document_id=doc_id
                    )
                    scheduled += 1
                except Exception as create_exc:
                    logger.warning(
                        "AHVI_REMINDER_CREATE_FAILED user_id=%s event_id=%s doc_id=%s send_at=%s error=%s",
                        uid,
                        eid,
                        doc_id,
                        send_at,
                        create_exc,
                    )
                    try:
                        self._appwrite.create_document(
                            self.reminders_resource, base_data, document_id=doc_id
                        )
                        scheduled += 1
                        logger.info(
                            "AHVI_REMINDER_CREATE_BASE_FALLBACK user_id=%s event_id=%s doc_id=%s send_at=%s",
                            uid,
                            eid,
                            doc_id,
                            send_at,
                        )
                    except Exception as base_exc:
                        logger.warning(
                            "AHVI_REMINDER_CREATE_BASE_FAILED user_id=%s event_id=%s doc_id=%s send_at=%s error=%s",
                            uid,
                            eid,
                            doc_id,
                            send_at,
                            base_exc,
                        )
                        continue
            if source_value in {"medicine", "medi"}:
                logger.info(
                    "AHVI_MED_REMINDER_SCHEDULED user_id=%s event_id=%s send_at=%s status=%s",
                    uid,
                    eid,
                    send_at,
                    status_value,
                )

        return {"success": True, "scheduled": scheduled}

    def list_due_reminders(
        self, *, now: Optional[datetime] = None, window_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        now_dt = now or _utcnow()
        now_ts = now_dt.timestamp()
        oldest_ts = now_ts - float(max(5, int(window_seconds)))

        # MVP: scan recent scheduled reminders per user on demand.
        # For scale, use a real indexed query (or store reminders in Redis sorted sets).
        try:
            rows = self._appwrite.list_documents(
                self.reminders_resource, limit=self.max_scan
            )
        except Exception:
            return []

        due: List[Dict[str, Any]] = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            status = str(r.get("status") or "").lower()
            source = str(r.get("source") or "").lower()
            if status != "scheduled" and not (
                status == "pending" and source in {"medicine", "medi"}
            ):
                continue
            send_at = _safe_text(r.get("sendAtISO") or "")
            try:
                send_dt = datetime.fromisoformat(send_at.replace("Z", "+00:00"))
            except Exception:
                continue
            send_ts = send_dt.timestamp()
            if oldest_ts <= send_ts <= now_ts:
                due.append(r)
        return due

    def list_due_medicine_reminders(
        self, *, now: Optional[datetime] = None, window_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        now_dt = now or _utcnow()
        now_ts = now_dt.timestamp()
        oldest_ts = now_ts - float(max(5, int(window_seconds)))
        try:
            rows = self._appwrite.list_documents("meds", limit=self.max_scan)
        except Exception:
            return []

        due: List[Dict[str, Any]] = []
        for med in rows or []:
            if not isinstance(med, dict):
                continue
            user_id = _safe_text(med.get("userId") or med.get("user_id"))
            med_id = _safe_text(med.get("$id") or med.get("id"))
            if not user_id or not med_id:
                continue
            if med.get("reminder") is False or med.get("reminder_enabled") is False:
                continue
            if med.get("active") is False or med.get("isActive") is False:
                continue
            if _safe_text(med.get("status")).lower() in {
                "inactive",
                "paused",
                "archived",
                "deleted",
            }:
                continue

            tz_name = _safe_text(med.get("timezone") or med.get("tz") or "Asia/Kolkata")
            scheduled = _parse_time_for_today(
                med.get("time") or med.get("scheduleTime") or med.get("reminderTime"),
                now=now_dt,
                tz_name=tz_name,
            )
            if scheduled is None:
                continue
            scheduled_utc = scheduled.astimezone(timezone.utc)
            scheduled_ts = scheduled_utc.timestamp()
            if scheduled_ts > now_ts:
                continue
            if scheduled_ts < oldest_ts:
                continue

            name = _safe_text(
                med.get("name") or med.get("medName") or med.get("medicine") or "Medicine"
            )
            notification_key = f"med:{user_id}:{med_id}:{scheduled_utc.isoformat()}"
            if self.was_notification_sent(notification_key=notification_key):
                continue
            logger.info(
                "AHVI_MED_REMINDER_SCHEDULED user_id=%s event_id=%s send_at=%s status=%s",
                user_id,
                notification_key,
                scheduled_utc.isoformat(),
                "scheduled",
            )
            due.append(
                {
                    "$id": _hash_id("medrem", notification_key, length=32),
                    "userId": user_id,
                    "eventId": notification_key,
                    "status": "scheduled",
                    "source": "medicine",
                    "message": f"Time to take {name}.",
                    "sendAtISO": scheduled_utc.isoformat(),
                    "medId": med_id,
                    "medName": name,
                    "notificationKey": notification_key,
                }
            )
        return due

    def was_notification_sent(self, *, notification_key: str) -> bool:
        key = _safe_text(notification_key)
        if not key:
            return False
        try:
            rows = self._appwrite.list_documents(self.reminders_resource, limit=self.max_scan)
        except Exception:
            return False
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if (
                _safe_text(row.get("eventId")) == key
                and _safe_text(row.get("status")).lower() == "sent"
            ):
                return True
        return False

    def mark_reminder(
        self, *, reminder_doc_id: str, status: str, error: str | None = None
    ) -> None:
        rid = _safe_text(reminder_doc_id)
        if not rid:
            return
        patch = {"status": _safe_text(status), "updatedAtISO": _utcnow().isoformat()}
        if error:
            patch["lastError"] = _safe_text(error)[:600]
        try:
            self._appwrite.update_document(self.reminders_resource, rid, patch)
        except Exception:
            return

    def mark_medicine_reminder(
        self,
        *,
        reminder_doc_id: str,
        user_id: str,
        notification_key: str,
        send_at_iso: str,
        message: str,
        status: str,
        error: str | None = None,
    ) -> None:
        rid = _safe_text(reminder_doc_id)
        key = _safe_text(notification_key)
        if not rid or not key:
            return
        data = {
            "userId": _safe_text(user_id),
            "eventId": key,
            "status": _safe_text(status),
            "priority": "normal",
            "toneProfile": "",
            "offsetMinutes": 0,
            "message": _safe_text(message),
            "sendAtISO": _safe_text(send_at_iso),
            "source": "medicine",
            "lastError": _safe_text(error or "")[:600],
            "updatedAtISO": _utcnow().isoformat(),
        }
        try:
            self._appwrite.update_document(self.reminders_resource, rid, data)
        except Exception:
            try:
                self._appwrite.create_document(
                    self.reminders_resource, data, document_id=rid
                )
            except Exception:
                return


notification_store = NotificationStore()
