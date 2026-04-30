from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.appwrite_proxy import AppwriteProxy


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _hash_id(prefix: str, raw: str, *, length: int = 32) -> str:
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}_{digest[: max(8, min(length, 48))]}"


class NotificationStore:
    """
    Stores:
    - device tokens in Appwrite collection: notification_devices
    - reminders in Appwrite collection: notification_reminders

    This is intentionally simple (MVP). For scale, move due-reminder queries to a DB index.
    """

    def __init__(self) -> None:
        self._appwrite = AppwriteProxy()
        self.devices_resource = os.getenv("APPWRITE_RESOURCE_NOTIFICATION_DEVICES", "notification_devices")
        self.reminders_resource = os.getenv("APPWRITE_RESOURCE_NOTIFICATION_REMINDERS", "notification_reminders")
        self.max_scan = max(50, int(os.getenv("NOTIFICATION_REMINDER_SCAN_LIMIT", "500")))

    # -------------------------
    # Devices
    # -------------------------
    def upsert_device(self, *, user_id: str, platform: str, token: str) -> str | None:
        uid = _safe_text(user_id)
        tok = _safe_text(token)
        plat = _safe_text(platform).lower() or "unknown"
        if not uid or not tok:
            return None

        doc_id = _hash_id("dev", tok, length=28)
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
                self._appwrite.create_document(self.devices_resource, data, document_id=doc_id)
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
            rows = self._appwrite.list_documents(self.devices_resource, user_id=uid, limit=200)
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

            doc_id = _hash_id("rem", f"{uid}|{eid}|{send_at}|{message}", length=36)
            data = {
                "userId": uid,
                "eventId": eid,
                "status": "scheduled",
                "priority": _safe_text(r.get("priority") or "light"),
                "toneProfile": _safe_text(r.get("toneProfile") or ""),
                "offsetMinutes": int(r.get("offsetMinutes") or 0),
                "message": message,
                "sendAtISO": send_at,
                "source": _safe_text(source),
                "updatedAtISO": _utcnow().isoformat(),
            }
            try:
                self._appwrite.update_document(self.reminders_resource, doc_id, data)
                scheduled += 1
            except Exception:
                try:
                    self._appwrite.create_document(self.reminders_resource, data, document_id=doc_id)
                    scheduled += 1
                except Exception:
                    continue

        return {"success": True, "scheduled": scheduled}

    def list_due_reminders(self, *, now: Optional[datetime] = None, window_seconds: int = 60) -> List[Dict[str, Any]]:
        now_dt = now or _utcnow()
        cutoff = now_dt.timestamp() + float(max(5, int(window_seconds)))

        # MVP: scan recent scheduled reminders per user on demand.
        # For scale, use a real indexed query (or store reminders in Redis sorted sets).
        try:
            rows = self._appwrite.list_documents(self.reminders_resource, limit=self.max_scan)
        except Exception:
            return []

        due: List[Dict[str, Any]] = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            if str(r.get("status") or "").lower() != "scheduled":
                continue
            send_at = _safe_text(r.get("sendAtISO") or "")
            try:
                send_dt = datetime.fromisoformat(send_at.replace("Z", "+00:00"))
            except Exception:
                continue
            if send_dt.timestamp() <= cutoff:
                due.append(r)
        return due

    def mark_reminder(self, *, reminder_doc_id: str, status: str, error: str | None = None) -> None:
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


notification_store = NotificationStore()

