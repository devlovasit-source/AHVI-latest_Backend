from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.appwrite_proxy import AppwriteProxy

CALENDAR_RESOURCE = "calendar_events"
logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _parse_iso(value: Any) -> Optional[datetime]:
    text = _safe_str(value)
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _metadata_to_string(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value if value.strip() else "{}"
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return "{}"


def _metadata_from_string(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = _safe_str(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(doc or {})
    row["id"] = _safe_str(row.get("$id") or row.get("id"))
    row["user_id"] = _safe_str(row.get("user_id") or row.get("userId"))
    row["title"] = _safe_str(row.get("title"))
    row["description"] = _safe_str(row.get("description"))
    row["start_time"] = _safe_str(row.get("start_time") or row.get("startTime") or row.get("startAtISO"))
    row["end_time"] = _safe_str(row.get("end_time") or row.get("endTime") or row.get("endAtISO"))
    row["timezone"] = _safe_str(row.get("timezone"))
    row["type"] = _safe_str(row.get("type") or "plan")
    row["source"] = _safe_str(row.get("source") or "ahvi")
    row["status"] = _safe_str(row.get("status") or "scheduled")
    row["dress_code"] = _safe_str(row.get("dress_code") or row.get("dressCode"))
    row["venue_name"] = _safe_str(row.get("venue_name") or row.get("venueName"))
    row["venue_address"] = _safe_str(row.get("venue_address") or row.get("venueAddress"))
    try:
        row["reminder_minutes"] = int(row.get("reminder_minutes") or 30)
    except Exception:
        row["reminder_minutes"] = 30
    row["metadata"] = _metadata_from_string(row.get("metadata"))
    row["created_at"] = _safe_str(row.get("created_at") or row.get("$createdAt"))
    row["updated_at"] = _safe_str(row.get("updated_at") or row.get("$updatedAt"))
    return row


def _build_document(user_id: str, payload: Dict[str, Any], *, existing: Dict[str, Any] | None = None) -> Dict[str, Any]:
    existing = existing or {}
    title = _safe_str(payload.get("title") or payload.get("text") or existing.get("title"))
    start_time = _safe_str(payload.get("start_time") or payload.get("startTime") or payload.get("startAtISO") or existing.get("start_time"))
    if not title:
        raise ValueError("title is required")
    if not start_time:
        raise ValueError("start_time is required")
    metadata = payload.get("metadata", existing.get("metadata", {}))
    metadata_string = metadata if isinstance(metadata, str) and metadata.strip() else _metadata_to_string(metadata)
    return {
        "user_id": _safe_str(user_id),
        "title": title,
        "description": _safe_str(payload.get("description", existing.get("description", ""))),
        "start_time": start_time,
        "end_time": _safe_str(payload.get("end_time") or payload.get("endTime") or payload.get("endAtISO") or existing.get("end_time", "")),
        "timezone": _safe_str(payload.get("timezone", existing.get("timezone", ""))),
        "type": _safe_str(payload.get("type", existing.get("type", "plan")) or "plan"),
        "source": _safe_str(payload.get("source", existing.get("source", "ahvi")) or "ahvi"),
        "status": _safe_str(payload.get("status", existing.get("status", "scheduled")) or "scheduled"),
        "dress_code": _safe_str(payload.get("dress_code") or payload.get("dressCode") or existing.get("dress_code", "")),
        "venue_name": _safe_str(payload.get("venue_name") or payload.get("venueName") or existing.get("venue_name", "")),
        "venue_address": _safe_str(payload.get("venue_address") or payload.get("venueAddress") or existing.get("venue_address", "")),
        "reminder_minutes": int(payload.get("reminder_minutes", existing.get("reminder_minutes", 30)) or 30),
        "metadata": metadata_string,
    }


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _next_month_day(base: datetime, month: int, day: int) -> date:
    year = base.year
    try:
        candidate = date(year, month, day)
    except ValueError:
        raise ValueError("Invalid calendar date")
    if candidate < base.date():
        candidate = date(year + 1, month, day)
    return candidate


def _extract_event_day(raw: str, base: datetime) -> date:
    lower = raw.lower()
    if "day after tomorrow" in lower:
        return (base + timedelta(days=2)).date()
    if "tomorrow" in lower:
        return (base + timedelta(days=1)).date()
    if "today" in lower:
        return base.date()

    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    day_month = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\b", lower)
    if day_month:
        return _next_month_day(base, _MONTHS[day_month.group(2)], int(day_month.group(1)))
    month_day = re.search(rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", lower)
    if month_day:
        return _next_month_day(base, _MONTHS[month_day.group(1)], int(month_day.group(2)))

    for word, weekday in _WEEKDAYS.items():
        if re.search(rf"\b{word}\b", lower):
            delta = (weekday - base.weekday()) % 7
            if delta == 0:
                delta = 7
            return (base + timedelta(days=delta)).date()
    return base.date()


def _extract_event_time(raw: str) -> tuple[int, int, bool]:
    lower = raw.lower()
    if re.search(r"\bmorning\b", lower):
        return 9, 0, True
    if re.search(r"\bafternoon\b", lower):
        return 14, 0, True
    if re.search(r"\bevening\b", lower):
        return 18, 0, True
    if re.search(r"\bnight\b", lower):
        return 20, 0, True
    patterns = [
        r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        r"\bat\s+(\d{1,2})(?::(\d{2}))\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if not match:
            continue
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3) if len(match.groups()) >= 3 else None
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        return max(0, min(hour, 23)), max(0, min(minute, 59)), True
    return 9, 0, False


def _event_type_and_title(raw: str, category: str | None) -> tuple[str, str]:
    lower = raw.lower()
    if "birthday" in lower:
        return "birthday", "Birthday"
    if "doctor" in lower:
        return "appointment", "Doctor appointment"
    if "dentist" in lower:
        return "appointment", "Dentist appointment"
    if "interview" in lower:
        return "interview", "Interview"
    if "appointment" in lower:
        return "appointment", "Appointment"
    meeting_with = re.search(r"\bmeeting\s+with\s+(.+?)(?:\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?::\d{2})?\s*(?:am|pm)?|at)\b|$)", raw, re.I)
    if meeting_with:
        person = _safe_str(meeting_with.group(1)).strip(" ,.-")
        return "meeting", f"Meeting with {person.title()}" if person else "Meeting"
    call_with = re.search(r"\bcall\s+with\s+(.+?)(?:\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?::\d{2})?\s*(?:am|pm)?|at)\b|$)", raw, re.I)
    if call_with:
        person = _safe_str(call_with.group(1)).strip(" ,.-")
        return "call", f"Call with {person.title()}" if person else "Call"
    if "meeting" in lower:
        return "meeting", "Meeting"
    if "call" in lower:
        return "call", "Call"
    event_type = _safe_str(category) or "plan"
    return event_type, raw[:80]


def parse_plan_text_to_payload(
    text: str,
    *,
    category: str | None = None,
    timezone_name: str = "Asia/Kolkata",
    now: datetime | None = None,
) -> Dict[str, Any]:
    raw = _safe_str(text)
    if len(raw) < 2:
        raise ValueError("text is required")
    base = now or datetime.now(timezone.utc)
    lower = raw.lower()
    day = _extract_event_day(raw, base)
    hour, minute, has_time = _extract_event_time(raw)
    event_type, title = _event_type_and_title(raw, category)
    if not has_time and event_type in {"meeting", "appointment", "call"}:
        raise ValueError("time_required")
    start = datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    return {
        "title": title,
        "description": raw,
        "start_time": start.isoformat(),
        "timezone": timezone_name,
        "type": event_type,
        "source": "ahvi_chat",
        "status": "scheduled",
        "metadata": {"original_text": raw, "category": event_type, "has_time": has_time},
    }


def create_calendar_event(user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = _build_document(user_id, payload)
    logger.info(
        "calendar_events.create payload_keys=%s user_id_present=%s",
        list(data.keys()),
        bool(data.get("user_id")),
    )
    doc = AppwriteProxy().create_document(CALENDAR_RESOURCE, data)
    return _normalize_doc(doc)


def get_calendar_event(user_id: str, event_id: str) -> Dict[str, Any]:
    doc = AppwriteProxy().get_document(CALENDAR_RESOURCE, event_id)
    row = _normalize_doc(doc)
    if row.get("user_id") != _safe_str(user_id):
        raise PermissionError("Calendar event does not belong to this user")
    return row


def list_calendar_events(user_id: str, *, start_time: str | None = None, end_time: str | None = None, limit: int = 200) -> List[Dict[str, Any]]:
    rows = AppwriteProxy().list_documents(CALENDAR_RESOURCE, limit=max(1, min(int(limit), 500)))
    if isinstance(rows, dict):
        rows = rows.get("documents") or []
    start_dt = _parse_iso(start_time)
    end_dt = _parse_iso(end_time)
    user = _safe_str(user_id)
    out: List[Dict[str, Any]] = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        row = _normalize_doc(raw)
        if row.get("user_id") != user:
            continue
        event_dt = _parse_iso(row.get("start_time"))
        if start_dt and event_dt and event_dt < start_dt:
            continue
        if end_dt and event_dt and event_dt > end_dt:
            continue
        out.append(row)
    out.sort(key=lambda x: _safe_str(x.get("start_time")))
    return out


def list_today_calendar_events(user_id: str, *, date: str | None = None) -> List[Dict[str, Any]]:
    if date:
        base = datetime.fromisoformat(date)
    else:
        base = datetime.now()
    start = datetime(base.year, base.month, base.day)
    end = start + timedelta(days=1)
    return list_calendar_events(user_id, start_time=start.isoformat(), end_time=end.isoformat(), limit=300)


def list_upcoming_calendar_events(user_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
    start = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()
    return list_calendar_events(user_id, start_time=start, limit=limit)


def update_calendar_event(user_id: str, event_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    existing = get_calendar_event(user_id, event_id)
    data = _build_document(user_id, {**existing, **(payload or {})}, existing=existing)
    doc = AppwriteProxy().update_document(CALENDAR_RESOURCE, event_id, data)
    return _normalize_doc(doc)


def delete_calendar_event(user_id: str, event_id: str) -> Dict[str, Any]:
    existing = get_calendar_event(user_id, event_id)
    AppwriteProxy().delete_document(CALENDAR_RESOURCE, event_id)
    return {"id": event_id, "deleted": True, "event": existing}


def calendar_event_to_runtime_input(event: Dict[str, Any]) -> Dict[str, Any]:
    row = _normalize_doc(event)
    return {
        "eventId": row.get("id") or "calendar_event",
        "title": row.get("title") or "Calendar event",
        "startAtISO": row.get("start_time") or _iso_now(),
        "endAtISO": row.get("end_time") or None,
        "timezone": row.get("timezone") or None,
        "venueName": row.get("venue_name") or None,
        "venueAddress": row.get("venue_address") or None,
        "notes": row.get("description") or None,
        "dressCode": row.get("dress_code") or None,
    }
