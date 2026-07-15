"""Durable, at-least-once medicine reminder state machine.

This module intentionally has no route or scheduler integration.  Appwrite
records make claims durable, but an FCM acknowledgement can precede persistence
of ``dispatched``; a later retry can therefore duplicate a push.  ``eventId``
and ``collapseKey`` let clients suppress that at-least-once duplicate safely.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python 3.9+ supplies zoneinfo.
    ZoneInfo = None  # type: ignore[assignment]

from services.notification_store import ReminderConflictError, ReminderStoreError

KIND_PRE_DOSE = "pre_dose"
KIND_FOLLOW_UP = "follow_up"
STATUS_PENDING = "pending"
STATUS_CLAIMED = "claimed"
STATUS_DISPATCHED = "dispatched"
STATUS_FAILED = "failed"
STATUS_CANCELLED_TAKEN = "cancelled_taken"
STATUS_CANCELLED_SKIPPED = "cancelled_skipped"
STATUS_MISSED = "missed"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat()


def _parse_utc(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(_text(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc(parsed)


def occurrence_id(user_id: str, medicine_id: str, scheduled_utc: datetime) -> str:
    raw = "|".join((_text(user_id), _text(medicine_id), _iso(scheduled_utc)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:36]


def notification_key(occurrence: str, kind: str) -> str:
    return f"{_text(occurrence)}|{_text(kind)}"


def timezone_for(name: str = "Asia/Kolkata"):
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(_text(name) or "Asia/Kolkata")
    except Exception:
        return ZoneInfo("Asia/Kolkata")


def scheduled_utc(local_time: datetime, timezone_name: str = "Asia/Kolkata") -> datetime:
    """Interpret a user-entered wall time in its named zone, never host time."""
    if local_time.tzinfo is None:
        local_time = local_time.replace(tzinfo=timezone_for(timezone_name))
    return _utc(local_time)


class MedicineReminderDispatcher:
    def __init__(self, *, store: Any, firebase: Any) -> None:
        self.store = store
        self.firebase = firebase
        self.recovery_minutes = self._env_int("MED_REMINDER_RECOVERY_MINUTES", 15)
        self.claim_stale_minutes = self._env_int("MED_REMINDER_CLAIM_STALE_MINUTES", 10)
        self.legacy_before_minutes = self._env_int("MED_REMINDER_LEGACY_MATCH_BEFORE_MINUTES", 30)
        self.legacy_after_minutes = self._env_int("MED_REMINDER_LEGACY_MATCH_AFTER_MINUTES", 30)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(name, str(default))))
        except ValueError:
            return default

    def schedule_occurrence(
        self, *, user_id: str, medicine_id: str, scheduled_utc: datetime,
        timezone_name: str = "Asia/Kolkata", dry_run: bool = False,
    ) -> Dict[str, str]:
        scheduled = _utc(scheduled_utc)
        occurrence = occurrence_id(user_id, medicine_id, scheduled)
        records = ((KIND_PRE_DOSE, scheduled - timedelta(minutes=15)), (KIND_FOLLOW_UP, scheduled + timedelta(minutes=15)))
        for kind, send_at in records:
            key = notification_key(occurrence, kind)
            if dry_run:
                continue
            try:
                self.store.create_reminder_record(
                    self.store.reminder_record_id(key),
                    {
                        "userId": _text(user_id), "eventId": key, "notificationKey": key,
                        "occurrenceId": occurrence, "kind": kind, "status": STATUS_PENDING,
                        "source": "medicine", "medId": _text(medicine_id),
                        "sendAtISO": _iso(send_at), "scheduledFor": _iso(scheduled),
                        "timezone": _text(timezone_name) or "Asia/Kolkata", "attemptCount": 0,
                        "message": "", "lastError": "", "updatedAtISO": _iso(scheduled),
                    },
                )
            except ReminderConflictError:
                continue
        return {"occurrence_id": occurrence, "pre_dose": notification_key(occurrence, KIND_PRE_DOSE), "follow_up": notification_key(occurrence, KIND_FOLLOW_UP)}

    def run(self, *, now: Optional[datetime] = None, dry_run: bool = False) -> Dict[str, int]:
        current = _utc(now or datetime.now(timezone.utc))
        counters = {key: 0 for key in ("due", "claimed", "dispatched", "duplicate", "cancelled_taken", "cancelled_skipped", "no_token", "failed")}
        earliest = _iso(current - timedelta(minutes=self.recovery_minutes))
        candidates = []
        for kind in (KIND_PRE_DOSE, KIND_FOLLOW_UP):
            for status in (STATUS_PENDING, STATUS_FAILED, STATUS_CLAIMED):
                candidates.extend(self.store.list_due_medicine_records(kind=kind, status=status, now_iso=_iso(current), earliest_iso=earliest))
        for record in candidates:
            counters["due"] += 1
            if not dry_run:
                self._dispatch(record, current, counters)
        return counters

    def _dispatch(self, record: Dict[str, Any], now: datetime, counters: Dict[str, int]) -> None:
        claimed = self._claim(record, now)
        if not claimed:
            counters["duplicate"] += 1
            return
        counters["claimed"] += 1
        doc_id, key = claimed
        if _text(record.get("kind")) == KIND_FOLLOW_UP:
            outcome = self._dose_outcome(record)
            if outcome:
                self.store.update_reminder_record(doc_id, {"status": f"cancelled_{outcome}"})
                counters[f"cancelled_{outcome}"] += 1
                return
        tokens = [_text(row.get("token")) for row in self.store.list_reminder_devices(user_id=_text(record.get("userId"))) if _text(row.get("token"))]
        if not tokens:
            self.store.update_reminder_record(doc_id, {"status": STATUS_FAILED, "lastError": "no_device_token"})
            counters["no_token"] += 1
            return
        response = self.firebase.send_to_tokens(tokens=tokens, title="Medicine reminder", body="", data={"eventId": key, "collapseKey": key, "type": "medicine_reminder", "kind": _text(record.get("kind")), "occurrenceId": _text(record.get("occurrenceId"))})
        if response.get("success") and int(response.get("sent") or 0) > 0:
            # If this update fails, preserve the typed failure for the caller;
            # a later run may resend and the client deduplicates by eventId.
            self.store.update_reminder_record(doc_id, {"status": STATUS_DISPATCHED, "dispatchedAtISO": _iso(now), "lastError": ""})
            counters["dispatched"] += 1
            return
        self.store.update_reminder_record(doc_id, {"status": STATUS_FAILED, "lastError": "push_failed"})
        counters["failed"] += 1

    def _claim(self, record: Dict[str, Any], now: datetime) -> Optional[tuple[str, str]]:
        doc_id = _text(record.get("$id") or record.get("id"))
        key = _text(record.get("notificationKey") or record.get("eventId"))
        if not doc_id or not key:
            return None
        status = _text(record.get("status"))
        if status == STATUS_CLAIMED:
            claimed_at = _parse_utc(record.get("claimedAtISO"))
            if claimed_at and now - claimed_at < timedelta(minutes=self.claim_stale_minutes):
                return None
        attempt = int(record.get("attemptCount") or 0) + 1
        try:
            self.store.create_claim_marker(user_id=_text(record.get("userId")), notification_key=key, attempt=attempt, claimed_at_iso=_iso(now))
        except ReminderConflictError:
            return None
        self.store.update_reminder_record(doc_id, {"status": STATUS_CLAIMED, "claimedAtISO": _iso(now), "attemptCount": attempt})
        return doc_id, key

    def _dose_outcome(self, record: Dict[str, Any]) -> str:
        user_id = _text(record.get("userId"))
        med_id = _text(record.get("medId"))
        occurrence = _text(record.get("occurrenceId"))
        scheduled = _parse_utc(record.get("scheduledFor"))
        if not user_id or not med_id or not occurrence or scheduled is None:
            return ""
        rows = self.store.list_occurrence_logs(user_id=user_id, med_id=med_id, occurrence_id=occurrence)
        if not rows:
            rows = self.store.list_legacy_dose_logs(user_id=user_id, med_id=med_id, earliest_iso=_iso(scheduled - timedelta(minutes=self.legacy_before_minutes)), latest_iso=_iso(scheduled + timedelta(minutes=self.legacy_after_minutes)))
        statuses = {_text(row.get("status")).lower() for row in rows}
        if "taken" in statuses:
            return "taken"
        if "skipped" in statuses:
            return "skipped"
        return ""
