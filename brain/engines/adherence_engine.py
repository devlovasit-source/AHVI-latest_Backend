from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from services.appwrite_proxy import AppwriteProxy
from services.notification_store import notification_store

logger = logging.getLogger("ahvi.adherence")

MED_STATUSES = {"pending", "taken", "missed", "skipped"}
SKINCARE_STATUSES = {"pending", "completed", "missed", "skipped"}
DEFAULT_TZ = "Asia/Kolkata"


def _tz(name: str = DEFAULT_TZ):
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(name or DEFAULT_TZ)
    except Exception:
        return ZoneInfo(DEFAULT_TZ)


def _now(tz_name: str = DEFAULT_TZ) -> datetime:
    return datetime.now(_tz(tz_name))


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _doc_id(prefix: str, raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}_{digest[:28]}"


def _date_key(value: Optional[datetime] = None, tz_name: str = DEFAULT_TZ) -> str:
    return (value or _now(tz_name)).date().isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    text = _safe_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _time_for_date(day: date, raw: Any, *, default: time, tz_name: str = DEFAULT_TZ) -> datetime:
    text = _safe_text(raw)
    if text:
        lowered = text.lower().replace(".", "").strip()
        for fmt in ("%H:%M", "%I:%M %p", "%I %p"):
            try:
                parsed = datetime.strptime(lowered.upper(), fmt).time()
                return datetime.combine(day, parsed, tzinfo=_tz(tz_name))
            except Exception:
                pass
        parsed_iso = _parse_iso(text)
        if parsed_iso is not None:
            return parsed_iso.astimezone(_tz(tz_name))
    return datetime.combine(day, default, tzinfo=_tz(tz_name))


def _normalize_list_result(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("documents") or value.get("items") or []
    return [dict(row) for row in value or [] if isinstance(row, dict)]


def _owned(rows: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
    uid = _safe_text(user_id)
    return [
        row
        for row in rows
        if _safe_text(row.get("userId") or row.get("user_id")) == uid
    ]


def _id(row: Dict[str, Any]) -> str:
    return _safe_text(row.get("$id") or row.get("id"))


def _upsert(proxy: AppwriteProxy, resource: str, doc_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return proxy.update_document(resource, doc_id, data)
    except Exception:
        return proxy.create_document(resource, data, document_id=doc_id)


def _med_name(med: Dict[str, Any]) -> str:
    return _safe_text(med.get("name") or med.get("title") or med.get("medicine") or med.get("medName"))


def _med_dose(med: Dict[str, Any]) -> str:
    return _safe_text(med.get("dose") or med.get("dosage") or med.get("strength") or "1 dose")


def _active_meds(proxy: AppwriteProxy, user_id: str) -> List[Dict[str, Any]]:
    rows = _owned(_normalize_list_result(proxy.list_documents("meds", user_id=user_id, limit=200)), user_id)
    active: List[Dict[str, Any]] = []
    for row in rows:
        status = _safe_text(row.get("status")).lower()
        if status in {"inactive", "archived", "deleted"}:
            continue
        if row.get("active") is False or row.get("isActive") is False:
            continue
        active.append(row)
    return active


def _today_med_logs(proxy: AppwriteProxy, user_id: str, today: str) -> List[Dict[str, Any]]:
    rows = _owned(_normalize_list_result(proxy.list_documents("med_logs", user_id=user_id, limit=500)), user_id)
    out: List[Dict[str, Any]] = []
    for row in rows:
        at = _parse_iso(row.get("time") or row.get("scheduledAtISO"))
        if at and at.astimezone(_tz()).date().isoformat() == today:
            out.append(row)
    return out


def create_notification_reminder(
    *,
    user_id: str,
    event_id: str,
    send_at: datetime,
    message: str,
    source: str,
    priority: str = "normal",
    offset_minutes: int = 0,
) -> Dict[str, Any]:
    return notification_store.schedule_reminders(
        user_id=user_id,
        event_id=event_id,
        source=source,
        reminders=[
            {
                "priority": priority,
                "offsetMinutes": offset_minutes,
                "message": message,
                "sendAtISO": send_at.isoformat(),
            }
        ],
    )


def generate_daily_med_logs(user_id: str, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    uid = _safe_text(user_id)
    current = now or _now()
    today = _date_key(current)
    proxy = AppwriteProxy()
    meds = _active_meds(proxy, uid)
    existing = _today_med_logs(proxy, uid, today)
    by_med = {_safe_text(row.get("medId") or row.get("med_id")): row for row in existing}
    logs: List[Dict[str, Any]] = []

    for med in meds:
        med_id = _id(med)
        if not med_id:
            continue
        scheduled = _time_for_date(
            current.date(),
            med.get("time") or med.get("scheduleTime") or med.get("reminderTime"),
            default=time(9, 0),
        )
        if med_id in by_med:
            logs.append(by_med[med_id])
            continue
        log_id = _doc_id("medlog", f"{uid}|{today}|{med_id}")
        data = {
            "userId": uid,
            "medId": med_id,
            "medName": _med_name(med) or "Medicine",
            "dose": _med_dose(med),
            "time": scheduled.isoformat(),
            "status": "pending",
        }
        log = _upsert(proxy, "med_logs", log_id, data)
        logs.append(log)
        create_notification_reminder(
            user_id=uid,
            event_id=log_id,
            send_at=scheduled,
            message=f"Time to take {data['medName']}.",
            source="medicine",
            priority="normal",
        )
        create_notification_reminder(
            user_id=uid,
            event_id=f"{log_id}:warning",
            send_at=scheduled + timedelta(minutes=30),
            message=f"{data['medName']} is still pending.",
            source="medicine",
            priority="high",
            offset_minutes=30,
        )

    return {"date": today, "meds": meds, "logs": logs}


def auto_reconcile_overdue(user_id: str, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    uid = _safe_text(user_id)
    current = now or _now()
    state = generate_daily_med_logs(uid, now=current)
    proxy = AppwriteProxy()
    updated = 0
    reconciled: List[Dict[str, Any]] = []
    for row in state["logs"]:
        status = _safe_text(row.get("status")).lower() or "pending"
        if status != "pending":
            reconciled.append(row)
            continue
        scheduled = _parse_iso(row.get("time") or row.get("scheduledAtISO"))
        overdue = False
        if scheduled and current.astimezone(scheduled.tzinfo or _tz()) > scheduled + timedelta(hours=14):
            overdue = True
        if current.time() >= time(23, 59):
            overdue = True
        if overdue:
            doc_id = _id(row)
            if doc_id:
                row = proxy.update_document("med_logs", doc_id, {"status": "missed"})
                updated += 1
        reconciled.append(row)
    return {"date": state["date"], "logs": reconciled, "updated": updated}


def _find_med_log(user_id: str, med_id: str = "", log_id: str = "", *, now: Optional[datetime] = None) -> Dict[str, Any]:
    uid = _safe_text(user_id)
    state = generate_daily_med_logs(uid, now=now)
    if log_id:
        for row in state["logs"]:
            if _id(row) == log_id:
                return row
    if med_id:
        for row in state["logs"]:
            if _safe_text(row.get("medId") or row.get("med_id")) == med_id:
                return row
    if len(state["logs"]) == 1:
        return state["logs"][0]
    raise ValueError("medId or logId is required")


def _update_med_status(user_id: str, status: str, *, med_id: str = "", log_id: str = "", now: Optional[datetime] = None) -> Dict[str, Any]:
    status = _safe_text(status).lower()
    if status not in MED_STATUSES - {"pending"}:
        raise ValueError("Invalid med status")
    uid = _safe_text(user_id)
    current = now or _now()
    proxy = AppwriteProxy()
    log = _find_med_log(uid, med_id=med_id, log_id=log_id, now=current)
    doc_id = _id(log)
    if not doc_id:
        raise ValueError("Med log not found")
    patch = {"status": status, "time": log.get("time") or current.isoformat()}
    updated = proxy.update_document("med_logs", doc_id, patch)
    if status == "taken":
        actual_med_id = _safe_text(updated.get("medId") or updated.get("med_id") or med_id)
        if actual_med_id:
            try:
                med = proxy.get_document("meds", actual_med_id)
                med_patch: Dict[str, Any] = {"lastTaken": current.isoformat()}
                left = med.get("left")
                if isinstance(left, int):
                    med_patch["left"] = max(0, left - 1)
                proxy.update_document("meds", actual_med_id, med_patch)
            except Exception:
                logger.warning("med.last_taken_update_failed user_id=%s med_id=%s", uid, actual_med_id)
    return updated


def mark_taken(user_id: str, *, med_id: str = "", log_id: str = "", now: Optional[datetime] = None) -> Dict[str, Any]:
    return _update_med_status(user_id, "taken", med_id=med_id, log_id=log_id, now=now)


def mark_missed(user_id: str, *, med_id: str = "", log_id: str = "", now: Optional[datetime] = None) -> Dict[str, Any]:
    return _update_med_status(user_id, "missed", med_id=med_id, log_id=log_id, now=now)


def mark_skipped(user_id: str, *, med_id: str = "", log_id: str = "", now: Optional[datetime] = None) -> Dict[str, Any]:
    return _update_med_status(user_id, "skipped", med_id=med_id, log_id=log_id, now=now)


def _profile_for_user(proxy: AppwriteProxy, user_id: str) -> Dict[str, Any]:
    rows = _owned(_normalize_list_result(proxy.list_documents("skincare_profiles", user_id=user_id, limit=20)), user_id)
    return rows[0] if rows else {}


def _today_skincare_logs(proxy: AppwriteProxy, user_id: str, today: str) -> List[Dict[str, Any]]:
    rows = _owned(_normalize_list_result(proxy.list_documents("skincare_logs", user_id=user_id, limit=100)), user_id)
    return [row for row in rows if _safe_text(row.get("date")) == today]


def _routine_steps(profile: Dict[str, Any], routine: str) -> List[int]:
    key = "daySteps" if routine == "morning" else "nightSteps"
    steps = profile.get(key)
    if isinstance(steps, list) and steps:
        out = []
        for step in steps:
            try:
                out.append(int(step))
            except Exception:
                pass
        return out or [0, 1, 2]
    return [0, 1, 2]


def _skincare_window(day: date, routine: str) -> tuple[datetime, datetime]:
    if routine == "morning":
        return (
            datetime.combine(day, time(6, 0), tzinfo=_tz()),
            datetime.combine(day, time(11, 0), tzinfo=_tz()),
        )
    return (
        datetime.combine(day, time(20, 0), tzinfo=_tz()),
        datetime.combine(day, time(23, 30), tzinfo=_tz()),
    )


def generate_daily_skincare_logs(user_id: str, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    uid = _safe_text(user_id)
    current = now or _now()
    today = _date_key(current)
    proxy = AppwriteProxy()
    profile = _profile_for_user(proxy, uid)
    existing = {str(row.get("routine") or "").lower(): row for row in _today_skincare_logs(proxy, uid, today)}
    logs: List[Dict[str, Any]] = []
    for routine in ("morning", "night"):
        if routine in existing:
            logs.append(existing[routine])
            continue
        start, _end = _skincare_window(current.date(), routine)
        log_id = _doc_id("skinlog", f"{uid}|{today}|{routine}")
        data = {
            "userId": uid,
            "date": today,
            "routine": routine,
            "status": "pending",
            "steps": _routine_steps(profile, routine),
            "completedSteps": [],
            "scheduledAtISO": start.isoformat(),
            "updatedAtISO": current.isoformat(),
        }
        log = _upsert(proxy, "skincare_logs", log_id, data)
        logs.append(log)
        create_notification_reminder(
            user_id=uid,
            event_id=log_id,
            send_at=start,
            message="Morning skincare still pending." if routine == "morning" else "Time for your night skincare routine.",
            source="skincare",
            priority="normal",
        )
    return {"date": today, "profile": profile, "logs": logs}


def auto_reconcile_skincare_overdue(user_id: str, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    uid = _safe_text(user_id)
    current = now or _now()
    state = generate_daily_skincare_logs(uid, now=current)
    proxy = AppwriteProxy()
    updated = 0
    rows: List[Dict[str, Any]] = []
    for row in state["logs"]:
        status = _safe_text(row.get("status")).lower() or "pending"
        if status != "pending":
            rows.append(row)
            continue
        routine = _safe_text(row.get("routine")).lower()
        _start, end = _skincare_window(current.date(), "night" if routine == "night" else "morning")
        if current > end:
            doc_id = _id(row)
            if doc_id:
                row = proxy.update_document(
                    "skincare_logs",
                    doc_id,
                    {"status": "missed", "updatedAtISO": current.isoformat()},
                )
                updated += 1
        rows.append(row)
    return {"date": state["date"], "profile": state["profile"], "logs": rows, "updated": updated}


def _update_skincare_status(user_id: str, status: str, *, routine: str = "", log_id: str = "", now: Optional[datetime] = None) -> Dict[str, Any]:
    status = _safe_text(status).lower()
    if status not in SKINCARE_STATUSES - {"pending"}:
        raise ValueError("Invalid skincare status")
    routine_key = _safe_text(routine).lower()
    if routine_key in {"day", "am"}:
        routine_key = "morning"
    if routine_key in {"evening", "pm"}:
        routine_key = "night"
    current = now or _now()
    state = generate_daily_skincare_logs(user_id, now=current)
    selected: Dict[str, Any] = {}
    for row in state["logs"]:
        if log_id and _id(row) == log_id:
            selected = row
            break
        if routine_key and _safe_text(row.get("routine")).lower() == routine_key:
            selected = row
            break
    if not selected and len(state["logs"]) == 1:
        selected = state["logs"][0]
    if not selected:
        raise ValueError("routine or logId is required")
    patch: Dict[str, Any] = {"status": status, "updatedAtISO": current.isoformat()}
    if status == "completed":
        patch["completedAtISO"] = current.isoformat()
        patch["completedSteps"] = selected.get("steps") or []
    return AppwriteProxy().update_document("skincare_logs", _id(selected), patch)


def mark_skincare_completed(user_id: str, *, routine: str = "", log_id: str = "", now: Optional[datetime] = None) -> Dict[str, Any]:
    return _update_skincare_status(user_id, "completed", routine=routine, log_id=log_id, now=now)


def mark_skincare_missed(user_id: str, *, routine: str = "", log_id: str = "", now: Optional[datetime] = None) -> Dict[str, Any]:
    return _update_skincare_status(user_id, "missed", routine=routine, log_id=log_id, now=now)


def mark_skincare_skipped(user_id: str, *, routine: str = "", log_id: str = "", now: Optional[datetime] = None) -> Dict[str, Any]:
    return _update_skincare_status(user_id, "skipped", routine=routine, log_id=log_id, now=now)
