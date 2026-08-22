"""Wardrobe Reminder Service.

Manages wear reminder CRUD integrated with NotificationStore and timezone support.
Enforces double-entity ownership checks (item.user_id == auth_user.id AND reminder.user_id == auth_user.id AND reminder.entity_id == item.id).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional
import zoneinfo

from services.notification_store import NotificationStore
from services.wardrobe_item_service import WardrobeItemService

logger = logging.getLogger("ahvi.wardrobe_reminder_service")


def _utcnow() -> datetime:
    return datetime.now(dt_timezone.utc)


class WardrobeReminderService:
    def __init__(self) -> None:
        self.store = NotificationStore()
        self.item_service = WardrobeItemService()

    def schedule_wear_reminder(
        self,
        *,
        user_id: str,
        item_id: str,
        scheduled_at: str,
        timezone: str = "Asia/Kolkata",
        message: str = "",
    ) -> Dict[str, Any]:
        # Double-entity check: 1. Validate wardrobe item ownership
        item = self.item_service.require_owned_item(user_id, item_id)
        item_name = str(item.get("name") or "Wardrobe Item")

        tz_str = (timezone or "").strip() or "Asia/Kolkata"
        try:
            tz = zoneinfo.ZoneInfo(tz_str)
        except Exception:
            tz = zoneinfo.ZoneInfo("Asia/Kolkata")
            tz_str = "Asia/Kolkata"

        try:
            dt = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
        except Exception:
            dt = _utcnow()

        send_at_iso = dt.astimezone(tz).isoformat()
        msg_text = message.strip() or f"Wear your {item_name}"

        res = self.store.schedule_reminders(
            user_id=user_id,
            event_id=item_id,
            source="wear_item",
            reminders=[
                {
                    "sendAtISO": send_at_iso,
                    "message": msg_text,
                    "status": "scheduled",
                    "priority": "normal",
                }
            ],
        )

        return {
            "success": bool(res.get("success")),
            "user_id": user_id,
            "item_id": item_id,
            "scheduled_at": send_at_iso,
            "timezone": tz_str,
            "message": msg_text,
        }

    def list_wear_reminders(self, *, user_id: str, item_id: str) -> List[Dict[str, Any]]:
        self.item_service.require_owned_item(user_id, item_id)
        rows = self.store.list_devices(user_id=user_id)  # fetch user reminders
        try:
            reminders = self.store._appwrite.list_documents(self.store.reminders_resource, user_id=user_id, limit=100)
        except Exception:
            reminders = []

        results = []
        for r in reminders if isinstance(reminders, list) else []:
            if not isinstance(r, dict):
                continue
            # Double-entity ownership verification
            r_user = str(r.get("userId") or "").strip()
            r_entity = str(r.get("eventId") or r.get("entityId") or "").strip()
            if r_user == user_id and r_entity == item_id:
                results.append(
                    {
                        "reminder_id": str(r.get("$id") or r.get("id") or ""),
                        "item_id": item_id,
                        "scheduled_at": r.get("sendAtISO"),
                        "message": r.get("message"),
                        "status": r.get("status"),
                        "source": r.get("source"),
                    }
                )
        return results

    def patch_wear_reminder(
        self,
        *,
        user_id: str,
        item_id: str,
        reminder_id: str,
        scheduled_at: Optional[str] = None,
        status: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.item_service.require_owned_item(user_id, item_id)
        doc = self._get_owned_reminder(user_id, item_id, reminder_id)

        patch: Dict[str, Any] = {"updatedAtISO": _utcnow().isoformat()}
        if scheduled_at:
            patch["sendAtISO"] = scheduled_at
        if status:
            patch["status"] = str(status).lower()
        if message:
            patch["message"] = str(message).strip()

        self.store._appwrite.update_document(self.store.reminders_resource, reminder_id, patch)
        return {"reminder_id": reminder_id, "item_id": item_id, **patch}

    def delete_wear_reminder(self, *, user_id: str, item_id: str, reminder_id: str) -> bool:
        self.item_service.require_owned_item(user_id, item_id)
        self._get_owned_reminder(user_id, item_id, reminder_id)
        self.store.mark_reminder(reminder_doc_id=reminder_id, status="cancelled")
        return True

    def _get_owned_reminder(self, user_id: str, item_id: str, reminder_id: str) -> Dict[str, Any]:
        doc = self.store._appwrite.get_document(self.store.reminders_resource, reminder_id)
        if not isinstance(doc, dict):
            raise KeyError("Reminder not found")
        r_user = str(doc.get("userId") or "").strip()
        r_entity = str(doc.get("eventId") or doc.get("entityId") or "").strip()
        if r_user != user_id or r_entity != item_id:
            raise KeyError("Reminder not found")
        return doc
