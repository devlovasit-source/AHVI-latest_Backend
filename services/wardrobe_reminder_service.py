"""Wear reminders — "remind me to wear this item again" — built entirely on
top of the existing NotificationStore/reminders stack (notification_reminders
collection, the same dispatch-due/FCM path medi and calendar reminders use).
No second scheduler, no second collection.

A wear reminder is identified as: source == "wardrobe" AND eventId ==
<wardrobe item id>. Both conditions together are the subtype boundary (there
is currently only one wardrobe reminder kind), and reconciliation after a
wear must always require both — never source-only or item-only, or a wear on
one item could complete another user/item's reminder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.notification_store import notification_store

WEAR_REMINDER_SOURCE = "wardrobe"
DEFAULT_MESSAGE = "Time to wear this item again!"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def create_wear_reminder(
    *, user_id: str, item_id: str, send_at_iso: str, message: str = ""
) -> Dict[str, Any]:
    uid = _safe_text(user_id)
    iid = _safe_text(item_id)
    send_at = _safe_text(send_at_iso)
    if not uid or not iid:
        raise ValueError("user_id and item_id are required")
    if not send_at:
        raise ValueError("send_at_iso is required")

    out = notification_store.schedule_reminders(
        user_id=uid,
        event_id=iid,
        source=WEAR_REMINDER_SOURCE,
        reminders=[
            {
                "sendAtISO": send_at,
                "message": _safe_text(message) or DEFAULT_MESSAGE,
                "offsetMinutes": 0,
            }
        ],
    )
    if not out.get("success"):
        raise RuntimeError("failed to schedule wear reminder")
    return {"success": True, "item_id": iid, "send_at_iso": send_at}


def _to_public(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reminder_id": row.get("$id"),
        "item_id": row.get("eventId"),
        "send_at_iso": row.get("sendAtISO"),
        "message": row.get("message"),
        "status": row.get("status"),
    }


def list_wear_reminders(*, user_id: str, item_id: str) -> List[Dict[str, Any]]:
    uid = _safe_text(user_id)
    iid = _safe_text(item_id)
    if not uid or not iid:
        return []
    rows = notification_store.list_reminders(
        user_id=uid, source=WEAR_REMINDER_SOURCE, event_id=iid
    )
    active = [r for r in rows if _safe_text(r.get("status")).lower() == "scheduled"]
    return [_to_public(r) for r in active]


def cancel_wear_reminder(*, user_id: str, item_id: str, reminder_id: str) -> bool:
    """Idempotent: cancelling an already-cancelled/missing reminder returns
    True rather than erroring, so a duplicate cancel call can't corrupt
    state. Raises PermissionError if the reminder exists but does not
    belong to this exact user+item (never partially matched)."""
    uid = _safe_text(user_id)
    iid = _safe_text(item_id)
    rid = _safe_text(reminder_id)
    if not uid or not iid or not rid:
        return False

    doc = notification_store.get_reminder(reminder_id=rid)
    if doc is None:
        return True  # already gone — cancel is idempotent

    if (
        _safe_text(doc.get("userId")) != uid
        or _safe_text(doc.get("source")).lower() != WEAR_REMINDER_SOURCE
        or _safe_text(doc.get("eventId")) != iid
    ):
        raise PermissionError("Reminder does not belong to this user/item.")

    if _safe_text(doc.get("status")).lower() == "cancelled":
        return True

    notification_store.mark_reminder(reminder_doc_id=rid, status="cancelled")
    return True


def complete_reminders_for_item(*, user_id: str, item_id: str) -> int:
    """Marks every active wear reminder for this exact user+item as
    completed. Call ONLY after a newly-committed WearEvent (never on a
    duplicate/idempotent-replay wear) — see wear_event_service.record_wear.
    Failures here must never be allowed to fail the wear itself; the caller
    is expected to swallow/log exceptions from this function."""
    uid = _safe_text(user_id)
    iid = _safe_text(item_id)
    if not uid or not iid:
        return 0
    rows = notification_store.list_reminders(
        user_id=uid, source=WEAR_REMINDER_SOURCE, event_id=iid
    )
    completed = 0
    for row in rows:
        if _safe_text(row.get("status")).lower() != "scheduled":
            continue
        rid = _safe_text(row.get("$id"))
        if not rid:
            continue
        notification_store.mark_reminder(reminder_doc_id=rid, status="completed")
        completed += 1
    return completed
