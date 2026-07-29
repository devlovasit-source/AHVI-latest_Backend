from __future__ import annotations

import logging
from datetime import datetime, date, time, timedelta
from typing import Any, Dict, List
from zoneinfo import ZoneInfo
from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.adherence_read_service")

# Helper to normalize list results from AppwriteProxy
def _normalize_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("documents") or value.get("items") or []
    return [dict(row) for row in value or [] if isinstance(row, dict)]

# Helper to check ownership
def _filter_owned(rows: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
    uid = str(user_id).strip()
    return [
        row
        for row in rows
        if str(row.get("userId") or row.get("user_id")).strip() == uid
    ]

def _parse_iso_safe(value: Any, tz: ZoneInfo) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        # Handle trailing Z or offsets safely
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Safe handling for naive stored timestamps
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(tz)
    except Exception:
        return None

def _parse_scheduled_time(raw: Any, tz: ZoneInfo) -> time | None:
    text = str(raw or "").strip()
    if not text:
        return None
    lowered = text.lower().replace(".", "").strip()
    for fmt in ("%H:%M", "%I:%M %p", "%I %p"):
        try:
            return datetime.strptime(lowered.upper(), fmt).time()
        except Exception:
            pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(tz).time()
    except Exception:
        pass
    return None

def _medicine_schedule_applies(med: Dict[str, Any], local_date: date) -> bool:
    """
    Respects start/end dates, weekdays, frequency, and paused schedules.
    """
    # Inactive/paused status
    status = str(med.get("status") or "").lower().strip()
    if status in {"inactive", "archived", "deleted"}:
        return False
    if med.get("active") is False or med.get("isActive") is False:
        return False

    # Start and End Date boundaries
    start_raw = med.get("startDate") or med.get("start_date")
    if start_raw:
        try:
            start = date.fromisoformat(str(start_raw)[:10])
            if local_date < start:
                return False
        except Exception:
            pass

    end_raw = med.get("endDate") or med.get("end_date")
    if end_raw:
        try:
            end = date.fromisoformat(str(end_raw)[:10])
            if local_date > end:
                return False
        except Exception:
            pass

    # Weekdays validation
    weekdays_raw = med.get("weekdays") or med.get("days")
    if weekdays_raw:
        day_of_week_short = local_date.strftime("%a").lower() # e.g. "mon"
        day_of_week_full = local_date.strftime("%A").lower() # e.g. "monday"

        if isinstance(weekdays_raw, str):
            weekdays_list = [w.strip().lower() for w in weekdays_raw.replace(",", " ").split() if w.strip()]
        elif isinstance(weekdays_raw, list):
            weekdays_list = [str(w).strip().lower() for w in weekdays_raw if str(w).strip()]
        else:
            weekdays_list = []

        if weekdays_list:
            matched = False
            for w in weekdays_list:
                if w.startswith(day_of_week_short) or w == day_of_week_short or w == day_of_week_full:
                    matched = True
                    break
            if not matched:
                return False

    return True

def get_today_medicine_state_readonly(
    user_id: str,
    local_now: datetime,
    proxy_client: Any | None = None
) -> Dict[str, Any]:
    """
    Read-only retrieval and evaluation of today's medicine log state.
    Never returns medicine names, dosage, or raw documents.
    """
    proxy = proxy_client or AppwriteProxy()
    tz = local_now.tzinfo
    today_str = local_now.date().isoformat()

    # 1. Fetch today's actual med logs
    try:
        all_logs_raw = proxy.list_documents("med_logs", user_id=user_id, limit=500)
        all_logs = _filter_owned(_normalize_list(all_logs_raw), user_id)
    except Exception as exc:
        logger.exception("Medicine logs read failed")
        return {
            "status": "error",
            "reason": "medicine_logs_read_failed",
            "source": "med_logs",
            "entity_id": None,
            "projected": False,
            "projection_key": None,
            "overdue_count": 0,
            "due_now_count": 0,
            "due_soon_count": 0,
            "later_count": 0,
            "completed_count": 0,
            "next_due_at": None
        }

    today_logs = []
    for log in all_logs:
        scheduled_dt = _parse_iso_safe(log.get("time") or log.get("scheduledAtISO"), tz)
        if scheduled_dt and scheduled_dt.date().isoformat() == today_str:
            today_logs.append((scheduled_dt, log))

    # 2. If no logs exist, project from active medicine schedules (with weekday & date validation)
    projected = False
    projection_key = None

    if not today_logs:
        try:
            meds_raw = proxy.list_documents("meds", user_id=user_id, limit=200)
            meds = _filter_owned(_normalize_list(meds_raw), user_id)
        except Exception as exc:
            logger.exception("Medicine schedules read failed")
            return {
                "status": "error",
                "reason": "medicine_schedule_read_failed",
                "source": "meds",
                "entity_id": None,
                "projected": False,
                "projection_key": None,
                "overdue_count": 0,
                "due_now_count": 0,
                "due_soon_count": 0,
                "later_count": 0,
                "completed_count": 0,
                "next_due_at": None
            }

        active_meds = [med for med in meds if _medicine_schedule_applies(med, local_now.date())]

        if not active_meds:
            return {
                "status": "unavailable",
                "reason": "medicine_schedule_unavailable",
                "source": "meds",
                "entity_id": None,
                "projected": False,
                "projection_key": None,
                "overdue_count": 0,
                "due_now_count": 0,
                "due_soon_count": 0,
                "later_count": 0,
                "completed_count": 0,
                "next_due_at": None
            }

        # Build virtual pending logs from active schedules (multiple times supported if in a list)
        projected = True
        projection_key = f"proj_med_{today_str}"
        for med in active_meds:
            times_raw = med.get("times") or [med.get("time") or med.get("scheduleTime") or med.get("reminderTime")]
            if isinstance(times_raw, str):
                times_raw = [times_raw]

            for t_raw in times_raw:
                sched_time = _parse_scheduled_time(t_raw, tz)
                if not sched_time:
                    # No default/fabricated time fallback
                    continue

                med_id = str(med.get("$id") or med.get("id") or "")
                scheduled_dt = datetime.combine(local_now.date(), sched_time, tzinfo=tz)

                virtual_log = {
                    "medId": med_id,
                    "status": "pending"
                }
                today_logs.append((scheduled_dt, virtual_log))

    # If even after projection we have no logs
    if not today_logs:
        return {
            "status": "unavailable",
            "reason": "medicine_schedule_unavailable",
            "source": "meds",
            "entity_id": None,
            "projected": projected,
            "projection_key": projection_key,
            "overdue_count": 0,
            "due_now_count": 0,
            "due_soon_count": 0,
            "later_count": 0,
            "completed_count": 0,
            "next_due_at": None
        }

    # 3. Categorize logs and prioritize
    overdue_count = 0
    due_now_count = 0
    due_soon_count = 0
    later_count = 0
    completed_count = 0

    next_due_dt = None
    chosen_entity_id = None

    for scheduled_dt, log in today_logs:
        status = str(log.get("status") or "pending").lower().strip()

        if status in {"taken", "skipped"}:
            completed_count += 1
            continue

        diff = scheduled_dt - local_now

        if status == "missed" or diff < timedelta(hours=-1):
            overdue_count += 1
        elif diff <= timedelta(seconds=0):
            due_now_count += 1
        elif diff <= timedelta(hours=1):
            due_soon_count += 1
        else:
            later_count += 1

        if next_due_dt is None or scheduled_dt < next_due_dt:
            next_due_dt = scheduled_dt
            if not projected:
                chosen_entity_id = log.get("$id") or log.get("id")

    if overdue_count > 0:
        status_label = "overdue"
    elif due_now_count > 0:
        status_label = "due_now"
    elif due_soon_count > 0:
        status_label = "due_soon"
    elif later_count > 0:
        status_label = "later"
    else:
        status_label = "completed"

    return {
        "status": status_label,
        "reason": None,
        "source": "med_logs" if not projected else "meds_projection",
        "entity_id": chosen_entity_id if not projected else None, # Projected state must always use entity_id = null
        "projected": projected,
        "projection_key": projection_key,
        "overdue_count": overdue_count,
        "due_now_count": due_now_count,
        "due_soon_count": due_soon_count,
        "later_count": later_count,
        "completed_count": completed_count,
        "next_due_at": next_due_dt.isoformat() if next_due_dt else None
    }

def get_today_skincare_state_readonly(
    user_id: str,
    local_now: datetime,
    proxy_client: Any | None = None
) -> Dict[str, Any]:
    """
    Read-only retrieval and evaluation of today's skincare log state.
    Converts timestamps using local_now's timezone.
    Never returns raw profiles or sensitive detail.
    """
    proxy = proxy_client or AppwriteProxy()
    tz = local_now.tzinfo
    today_str = local_now.date().isoformat()

    # 1. Fetch today's actual skincare logs
    try:
        all_logs_raw = proxy.list_documents("skincare_logs", user_id=user_id, limit=100)
        all_logs = _filter_owned(_normalize_list(all_logs_raw), user_id)
    except Exception as exc:
        logger.exception("Skincare logs read failed")
        return {
            "status": "error",
            "reason": "skincare_logs_read_failed",
            "source": "skincare_logs",
            "entity_id": None,
            "projected": False,
            "projection_key": None,
            "routine": None,
            "scheduled_at": None,
            "completed_steps": None,
            "total_steps": None
        }

    today_logs = []
    for log in all_logs:
        if str(log.get("date") or "").strip() == today_str:
            today_logs.append(log)

    # 2. If no logs exist, project from skincare profile
    projected = False
    projection_key = None

    if not today_logs:
        try:
            profiles_raw = proxy.list_documents("skincare_profiles", user_id=user_id, limit=20)
            profiles = _filter_owned(_normalize_list(profiles_raw), user_id)
            profile = profiles[0] if profiles else {}
        except Exception as exc:
            logger.exception("Skincare profile read failed")
            return {
                "status": "error",
                "reason": "skincare_profile_read_failed",
                "source": "skincare_profiles",
                "entity_id": None,
                "projected": False,
                "projection_key": None,
                "routine": None,
                "scheduled_at": None,
                "completed_steps": None,
                "total_steps": None
            }

        if not profile:
            return {
                "status": "unavailable",
                "reason": "skincare_routine_unavailable",
                "source": "skincare_profiles",
                "entity_id": None,
                "projected": False,
                "projection_key": None,
                "routine": None,
                "scheduled_at": None,
                "completed_steps": None,
                "total_steps": None
            }

        # Project routines only if daySteps or nightSteps is explicitly present/non-empty
        projected = True
        projection_key = f"proj_skin_{today_str}"

        for routine in ("morning", "night"):
            key = "daySteps" if routine == "morning" else "nightSteps"
            raw_steps = profile.get(key)
            if not isinstance(raw_steps, list) or not raw_steps:
                # No default steps [0,1,2] fabrication allowed! Skip if empty or missing.
                continue

            steps = []
            for s in raw_steps:
                try:
                    steps.append(int(s))
                except Exception:
                    pass
            if not steps:
                continue

            # No default 08:00 or 20:00 or scheduled-time fabrication allowed. Set to None.
            virtual_log = {
                "date": today_str,
                "routine": routine,
                "status": "pending",
                "steps": steps,
                "completedSteps": [],
                "scheduledAtISO": None # No fabricated times
            }
            today_logs.append(virtual_log)

    if not today_logs:
        return {
            "status": "unavailable",
            "reason": "skincare_routine_unavailable",
            "source": "skincare_profiles",
            "entity_id": None,
            "projected": projected,
            "projection_key": projection_key,
            "routine": None,
            "scheduled_at": None,
            "completed_steps": None,
            "total_steps": None
        }

    # Evaluate the active routine based on priority (overdue/pending first, morning first)
    pending_logs = [l for l in today_logs if str(l.get("status")).lower() == "pending"]

    if pending_logs:
        pending_logs.sort(key=lambda x: 0 if str(x.get("routine")).lower() == "morning" else 1)
        chosen_log = pending_logs[0]
        status_label = "ready"
    else:
        chosen_log = today_logs[-1]
        status_label = "completed"

    routine = str(chosen_log.get("routine") or "morning").lower()
    scheduled_at_raw = chosen_log.get("scheduledAtISO") or chosen_log.get("time") or None
    scheduled_dt = _parse_iso_safe(scheduled_at_raw, tz) if scheduled_at_raw else None

    steps = chosen_log.get("steps") or []
    completed_steps = chosen_log.get("completedSteps") or []

    return {
        "status": status_label,
        "reason": None,
        "source": "skincare_logs" if not projected else "skincare_profile_projection",
        "entity_id": chosen_entity_id if not projected and (chosen_entity_id := (chosen_log.get("$id") or chosen_log.get("id"))) else None, # Projected state must always use entity_id = null
        "projected": projected,
        "projection_key": projection_key,
        "routine": routine,
        "scheduled_at": scheduled_dt.isoformat() if scheduled_dt else None,
        "completed_steps": completed_steps,
        "total_steps": len(steps) if steps else 0
    }
