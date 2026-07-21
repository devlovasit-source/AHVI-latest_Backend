"""Medicine reminder dispatch: Cloud Scheduler -> /api/notifications/dispatch-due.

Stateless per-invocation dispatcher (NO in-process polling loop):
  find due primaries -> atomically claim -> Firebase push -> mark dispatched
  -> ensure exactly one +15-minute follow-up -> at follow-up time, cancel if
  the exact dose is already marked taken, otherwise send once.

Atomicity on Appwrite (no compare-and-swap available):
- the reminder record uses a deterministic document id derived from the
  notification key, so `create_document` is create-once;
- each send attempt additionally requires creating a per-attempt claim
  marker document (key + attempt number) - two workers that both read
  attemptCount=N race to create marker N+1 and exactly one wins.

All stored timestamps are UTC ISO-8601. User-entered times are interpreted
in the user's timezone (default Asia/Kolkata) by the store's ZoneInfo-based
parser - never the Cloud Run machine timezone.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KIND_PRIMARY = "primary"
KIND_FOLLOW_UP = "follow_up"

STATUS_PENDING = "pending"
STATUS_DISPATCHING = "dispatching"
STATUS_DISPATCHED = "dispatched"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"

_TERMINAL_STATUSES = {STATUS_DISPATCHED, STATUS_CANCELLED}


def _txt(value: Any) -> str:
    return str(value or "").strip()


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except Exception:
        return default


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_utc(raw: Any) -> Optional[datetime]:
    text = _txt(raw)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def primary_notification_key(user_id: str, med_id: str, scheduled_utc_iso: str) -> str:
    return f"medi:{user_id}:{med_id}:{scheduled_utc_iso}:primary"


def follow_up_notification_key(
    user_id: str, med_id: str, scheduled_utc_iso: str, offset_minutes: int = 15
) -> str:
    return f"medi:{user_id}:{med_id}:{scheduled_utc_iso}:follow_up:{int(offset_minutes)}"


class MedicineReminderDispatcher:
    """One dispatch pass over due medicine reminders. Safe to run from many
    Cloud Run instances concurrently."""

    def __init__(self, *, store: Any, firebase: Any) -> None:
        self.store = store
        self.firebase = firebase
        self.recovery_minutes = _env_int("MED_REMINDER_RECOVERY_MINUTES", 15)
        self.follow_up_minutes = _env_int("MED_REMINDER_FOLLOWUP_MINUTES", 15)
        self.claim_stale_minutes = _env_int("MED_REMINDER_CLAIM_STALE_MINUTES", 10)
        self.taken_slack_minutes = _env_int("MED_REMINDER_TAKEN_SLACK_MINUTES", 30)

    # ------------------------------------------------------------------ run

    def run(
        self,
        *,
        now: Optional[datetime] = None,
        dry_run: bool = False,
        persisted_due: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        stats: Dict[str, Any] = {
            "processed": 0, "sent": 0, "failed": 0, "skipped": 0,
            "cancelled": 0, "due": [],
        }

        for candidate in self._due_primaries(now, persisted_due or []):
            stats["processed"] += 1
            self._dispatch_primary(candidate, now=now, dry_run=dry_run, stats=stats)

        try:
            follow_ups = self.store.list_due_follow_up_records(
                now=now, window_seconds=self.recovery_minutes * 60
            )
        except Exception:
            follow_ups = []
        for record in follow_ups:
            stats["processed"] += 1
            self._dispatch_follow_up(record, now=now, dry_run=dry_run, stats=stats)
        return stats

    # ------------------------------------------------------- primary phase

    def _due_primaries(
        self, now: datetime, persisted_due: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge dispatcher-relevant primaries: rows persisted by the Flutter
        scheduling endpoint plus occurrences synthesized from the meds
        collection. De-duplicated by notification key; never future items."""
        window_seconds = self.recovery_minutes * 60
        candidates: List[Dict[str, Any]] = list(persisted_due)
        try:
            candidates.extend(
                self.store.list_due_medicine_reminders(
                    now=now, window_seconds=window_seconds
                )
            )
        except Exception:
            logger.exception("med dispatch: meds scan failed")

        seen: set = set()
        out: List[Dict[str, Any]] = []
        for raw in candidates:
            if not isinstance(raw, dict):
                continue
            normalized = self._normalize_primary(raw, now)
            if normalized is None:
                continue
            if normalized["key"] in seen:
                continue
            seen.add(normalized["key"])
            out.append(normalized)
        return out

    def _normalize_primary(
        self, raw: Dict[str, Any], now: datetime
    ) -> Optional[Dict[str, Any]]:
        user_id = _txt(raw.get("userId") or raw.get("user_id"))
        med_id = _txt(raw.get("medId") or raw.get("med_id") or raw.get("eventId"))
        scheduled = _parse_utc(raw.get("sendAtISO") or raw.get("scheduledFor"))
        if not user_id or not med_id or scheduled is None:
            return None
        # Lateness tolerance only: due now-or-earlier, inside the bounded
        # recovery window; never a future reminder, never an ancient one.
        if scheduled > now:
            return None
        if scheduled < now - timedelta(minutes=self.recovery_minutes):
            return None
        scheduled_iso = _utc_iso(scheduled)
        med_name = _txt(raw.get("medName") or raw.get("med_name") or "Medicine")
        return {
            "key": primary_notification_key(user_id, med_id, scheduled_iso),
            "legacy_key": _txt(raw.get("notificationKey") or raw.get("notification_key")),
            "user_id": user_id,
            "med_id": med_id,
            "med_name": med_name,
            "dose": _txt(raw.get("dose")),
            "scheduled_utc": scheduled_iso,
            "timezone": _txt(raw.get("timezone") or raw.get("tz") or "Asia/Kolkata"),
            "message": _txt(raw.get("message") or raw.get("body"))
            or f"Time to take {med_name}.",
            "legacy_doc_id": _txt(raw.get("$id") or raw.get("id")),
        }

    def _dispatch_primary(
        self,
        candidate: Dict[str, Any],
        *,
        now: datetime,
        dry_run: bool,
        stats: Dict[str, Any],
    ) -> None:
        key = candidate["key"]
        if dry_run:
            stats["due"].append(
                {
                    "id": self.store.reminder_record_id(key),
                    "userId": candidate["user_id"],
                    "source": "medicine",
                    "kind": KIND_PRIMARY,
                    "medId": candidate["med_id"],
                    "medName": candidate["med_name"],
                    "scheduledFor": candidate["scheduled_utc"],
                    "notificationKey": key,
                }
            )
            return

        # Legacy compatibility: reminders already marked sent under the old
        # med:{...} key format must not be sent again.
        legacy_key = candidate["legacy_key"]
        try:
            if legacy_key and self.store.was_notification_sent(
                notification_key=legacy_key
            ):
                stats["skipped"] += 1
                return
        except Exception:
            pass

        record = {
            "userId": candidate["user_id"],
            "eventId": key,
            "notificationKey": key,
            "kind": KIND_PRIMARY,
            "status": STATUS_PENDING,
            "source": "medicine",
            "medId": candidate["med_id"],
            "medName": candidate["med_name"],
            "dose": candidate["dose"],
            "sendAtISO": candidate["scheduled_utc"],
            "scheduledFor": candidate["scheduled_utc"],
            "timezone": candidate["timezone"],
            "message": candidate["message"],
            "offsetMinutes": 0,
            "priority": "normal",
            "lastError": "",
            "updatedAtISO": _utc_iso(now),
        }
        claim = self._claim(key, record, now)
        if claim is None:
            stats["skipped"] += 1
            return
        record_id, _attempt = claim

        sent = self._push(
            candidate["user_id"],
            title="AHVI",
            body=candidate["message"],
            data={
                "type": "reminder",
                "action": "med_reminder",
                "medId": candidate["med_id"],
                "medName": candidate["med_name"],
                "scheduledFor": candidate["scheduled_utc"],
                "screen": "medi",
                "deepLink": f"ahvi://medi/reminder/{record_id}",
            },
            record_id=record_id,
            now=now,
            stats=stats,
            log_prefix="AHVI_MED_REMINDER",
            candidate=candidate,
        )
        if not sent:
            return

        # Mark any legacy persisted row so old dashboards stay coherent.
        if candidate["legacy_doc_id"] and candidate["legacy_doc_id"] != record_id:
            try:
                self.store.mark_reminder(
                    reminder_doc_id=candidate["legacy_doc_id"], status="sent"
                )
            except Exception:
                pass
        self._ensure_follow_up(candidate, primary_record_id=record_id, now=now)

    # ----------------------------------------------------- follow-up phase

    def _ensure_follow_up(
        self, candidate: Dict[str, Any], *, primary_record_id: str, now: datetime
    ) -> None:
        """Exactly one follow-up at primary time + offset. Create-once by
        deterministic id: a second successful primary retry cannot add more."""
        scheduled = _parse_utc(candidate["scheduled_utc"])
        if scheduled is None:
            return
        follow_at = scheduled + timedelta(minutes=self.follow_up_minutes)
        f_key = follow_up_notification_key(
            candidate["user_id"],
            candidate["med_id"],
            candidate["scheduled_utc"],
            self.follow_up_minutes,
        )
        created = self.store.create_reminder_record(
            self.store.reminder_record_id(f_key),
            {
                "userId": candidate["user_id"],
                "eventId": f_key,
                "notificationKey": f_key,
                "kind": KIND_FOLLOW_UP,
                "status": STATUS_PENDING,
                "source": "medicine",
                "medId": candidate["med_id"],
                "medName": candidate["med_name"],
                "dose": candidate["dose"],
                "sendAtISO": _utc_iso(follow_at),
                "scheduledFor": candidate["scheduled_utc"],
                "timezone": candidate["timezone"],
                "primaryReminderId": primary_record_id,
                "offsetMinutes": self.follow_up_minutes,
                "message": (
                    f"Have you taken {candidate['med_name']}? "
                    "Open AHVI to mark it taken."
                ),
                "title": "Medicine reminder",
                "priority": "normal",
                "lastError": "",
                "updatedAtISO": _utc_iso(now),
            },
        )
        if created:
            logger.info(
                "AHVI_MED_FOLLOWUP_SCHEDULED user_id=%s med_id=%s send_at=%s",
                candidate["user_id"], candidate["med_id"], _utc_iso(follow_at),
            )

    def _dispatch_follow_up(
        self,
        record: Dict[str, Any],
        *,
        now: datetime,
        dry_run: bool,
        stats: Dict[str, Any],
    ) -> None:
        key = _txt(record.get("notificationKey") or record.get("eventId"))
        record_id = _txt(record.get("$id") or record.get("id")) or (
            self.store.reminder_record_id(key)
        )
        user_id = _txt(record.get("userId"))
        med_id = _txt(record.get("medId"))
        med_name = _txt(record.get("medName") or "Medicine")
        primary_scheduled = _txt(record.get("scheduledFor") or "")
        if not key or not user_id or not med_id:
            stats["skipped"] += 1
            return

        if dry_run:
            stats["due"].append(
                {
                    "id": record_id,
                    "userId": user_id,
                    "source": "medicine",
                    "kind": KIND_FOLLOW_UP,
                    "medId": med_id,
                    "medName": med_name,
                    "scheduledFor": primary_scheduled,
                    "notificationKey": key,
                }
            )
            return

        # Suppress when the EXACT dose (same user + medicine + occurrence)
        # was already marked taken.
        try:
            taken = self.store.is_dose_taken(
                user_id=user_id,
                med_id=med_id,
                scheduled_utc_iso=primary_scheduled or _txt(record.get("sendAtISO")),
                slack_minutes=self.taken_slack_minutes,
            )
        except Exception:
            taken = False
        if taken:
            self.store.update_reminder_record(record_id, {"status": STATUS_CANCELLED})
            stats["cancelled"] += 1
            logger.info(
                "AHVI_MED_FOLLOWUP_CANCELLED user_id=%s med_id=%s key=%s reason=dose_taken",
                user_id, med_id, key,
            )
            return

        claim = self._claim_existing(record, record_id, key, now)
        if claim is None:
            stats["skipped"] += 1
            return

        self._push(
            user_id,
            title="Medicine reminder",
            body=f"Have you taken {med_name}? Open AHVI to mark it taken.",
            data={
                "type": "reminder",
                "action": "med_reminder_follow_up",
                "medId": med_id,
                "medName": med_name,
                "scheduledFor": primary_scheduled,
                "screen": "medi",
                "deepLink": f"ahvi://medi/reminder/{record_id}",
            },
            record_id=record_id,
            now=now,
            stats=stats,
            log_prefix="AHVI_MED_FOLLOWUP",
            candidate={"user_id": user_id, "med_id": med_id,
                       "scheduled_utc": primary_scheduled},
        )

    # ------------------------------------------------------------- claiming

    def _claim(
        self, key: str, record: Dict[str, Any], now: datetime
    ) -> Optional["tuple[str, int]"]:
        """Claim a reminder for dispatch. Returns (record_id, attempt) or None
        when another worker owns it / it is already terminal."""
        record_id = self.store.reminder_record_id(key)
        created = self.store.create_reminder_record(
            record_id,
            {**record, "status": STATUS_DISPATCHING,
             "claimedAtISO": _utc_iso(now), "attemptCount": 1},
        )
        if created:
            return record_id, 1
        existing = self.store.get_reminder_record(record_id)
        if existing is None:
            return None
        return self._claim_existing(existing, record_id, key, now)

    def _claim_existing(
        self, existing: Dict[str, Any], record_id: str, key: str, now: datetime
    ) -> Optional["tuple[str, int]"]:
        status = _txt(existing.get("status")).lower()
        if status in _TERMINAL_STATUSES:
            return None
        if status == STATUS_DISPATCHING:
            claimed_at = _parse_utc(existing.get("claimedAtISO"))
            if claimed_at is not None and now - claimed_at < timedelta(
                minutes=self.claim_stale_minutes
            ):
                return None  # fresh claim held by another dispatcher
        try:
            attempt = int(existing.get("attemptCount") or 0) + 1
        except Exception:
            attempt = 1
        if not self.store.create_claim_marker(notification_key=key, attempt=attempt):
            return None  # lost the per-attempt race
        self.store.update_reminder_record(
            record_id,
            {"status": STATUS_DISPATCHING, "claimedAtISO": _utc_iso(now),
             "attemptCount": attempt},
        )
        return record_id, attempt

    # --------------------------------------------------------------- sending

    def _push(
        self,
        user_id: str,
        *,
        title: str,
        body: str,
        data: Dict[str, str],
        record_id: str,
        now: datetime,
        stats: Dict[str, Any],
        log_prefix: str,
        candidate: Dict[str, Any],
    ) -> bool:
        devices = self.store.list_devices(user_id=user_id)
        tokens = [
            _txt(d.get("token")) for d in devices if _txt(d.get("token"))
        ]
        if not tokens:
            # Not a success and not a hard failure: retryable, never marked
            # dispatched. Tokens themselves are never logged.
            self.store.update_reminder_record(
                record_id, {"status": STATUS_FAILED, "lastError": "no_device_token"}
            )
            stats["skipped"] += 1
            logger.info(
                "%s_FAILED user_id=%s med_id=%s scheduled_for=%s reason=no_tokens",
                log_prefix, user_id, candidate.get("med_id"),
                candidate.get("scheduled_utc"),
            )
            return False
        resp = self.firebase.send_to_tokens(
            tokens=tokens, title=title, body=body, data=data
        )
        if resp.get("success") and int(resp.get("sent") or 0) > 0:
            self.store.update_reminder_record(
                record_id,
                {"status": STATUS_DISPATCHED, "dispatchedAtISO": _utc_iso(now),
                 "lastError": ""},
            )
            stats["sent"] += 1
            logger.info(
                "%s_SENT user_id=%s med_id=%s scheduled_for=%s",
                log_prefix, user_id, candidate.get("med_id"),
                candidate.get("scheduled_utc"),
            )
            return True
        self.store.update_reminder_record(
            record_id,
            {"status": STATUS_FAILED,
             "lastError": _txt(resp.get("error") or "push_failed")[:600]},
        )
        stats["failed"] += 1
        logger.info(
            "%s_FAILED user_id=%s med_id=%s scheduled_for=%s error=%s",
            log_prefix, user_id, candidate.get("med_id"),
            candidate.get("scheduled_utc"), _txt(resp.get("error") or "push_failed"),
        )
        return False
