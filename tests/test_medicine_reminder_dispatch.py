from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.appwrite_proxy import AppwriteProxyError
from services.medicine_reminder_dispatch import (
    KIND_FOLLOW_UP,
    KIND_PRE_DOSE,
    STATUS_CLAIMED,
    STATUS_DISPATCHED,
    STATUS_FAILED,
    MedicineReminderDispatcher,
    occurrence_id,
    scheduled_utc,
)
from services.notification_store import (
    NotificationStore,
    ReminderConflictError,
    ReminderStoreError,
)


NOW = datetime(2026, 5, 1, 14, 30, tzinfo=timezone.utc)


class MemoryStore:
    def __init__(self):
        self.records = {}
        self.claims = set()
        self.logs = []
        self.devices = [{"token": "token-1"}]
        self.update_fail = False

    @staticmethod
    def reminder_record_id(key):
        return f"record-{key}"

    def create_reminder_record(self, doc_id, data):
        if doc_id in self.records:
            raise ReminderConflictError("exists")
        self.records[doc_id] = {"$id": doc_id, **data}

    def list_due_medicine_records(self, *, kind, status, now_iso, earliest_iso, limit=100):
        return [
            record for record in self.records.values()
            if record["kind"] == kind and record["status"] == status
            and earliest_iso <= record["sendAtISO"] <= now_iso
        ][:limit]

    def create_claim_marker(self, *, user_id, notification_key, attempt, claimed_at_iso):
        marker = (notification_key, attempt)
        if marker in self.claims:
            raise ReminderConflictError("claimed")
        self.claims.add(marker)

    def update_reminder_record(self, doc_id, patch):
        if self.update_fail and patch.get("status") == STATUS_DISPATCHED:
            raise ReminderStoreError("state write unavailable")
        self.records[doc_id].update(patch)

    def list_reminder_devices(self, *, user_id):
        return list(self.devices)

    def list_occurrence_logs(self, *, user_id, med_id, occurrence_id):
        return [row for row in self.logs if row.get("occurrenceId") == occurrence_id]

    def list_legacy_dose_logs(self, *, user_id, med_id, earliest_iso, latest_iso):
        return [
            row for row in self.logs
            if row.get("userId") == user_id and row.get("medId") == med_id
            and earliest_iso <= row.get("time", "") <= latest_iso
        ]


class Firebase:
    def __init__(self, response=None):
        self.response = response or {"success": True, "sent": 1, "failed": 0}
        self.calls = []

    def send_to_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.response)


def make_dispatcher(store=None, firebase=None):
    return MedicineReminderDispatcher(store=store or MemoryStore(), firebase=firebase or Firebase())


def create_due(dispatcher, *, kind=KIND_PRE_DOSE, status="pending", send_at=NOW, med_id="med-1"):
    occurrence = occurrence_id("user-1", med_id, NOW)
    key = f"{occurrence}|{kind}"
    doc_id = dispatcher.store.reminder_record_id(key)
    dispatcher.store.records[doc_id] = {
        "$id": doc_id, "userId": "user-1", "medId": med_id,
        "occurrenceId": occurrence, "notificationKey": key, "kind": kind,
        "status": status, "sendAtISO": send_at.isoformat(),
        "scheduledFor": NOW.isoformat(), "attemptCount": 0,
    }
    return dispatcher.store.records[doc_id]


def test_schedule_calculates_pre_and_follow_up_times():
    dispatcher = make_dispatcher()
    dispatcher.schedule_occurrence(user_id="user-1", medicine_id="med-1", scheduled_utc=NOW)
    by_kind = {record["kind"]: record for record in dispatcher.store.records.values()}
    assert by_kind[KIND_PRE_DOSE]["sendAtISO"] == (NOW - timedelta(minutes=15)).isoformat()
    assert by_kind[KIND_FOLLOW_UP]["sendAtISO"] == (NOW + timedelta(minutes=15)).isoformat()


def test_timezone_conversion_and_invalid_timezone_fallback():
    local = datetime(2026, 5, 1, 20, 0)
    assert scheduled_utc(local, "Asia/Kolkata") == datetime(2026, 5, 1, 14, 30, tzinfo=timezone.utc)
    assert scheduled_utc(local, "not/a-timezone") == datetime(2026, 5, 1, 14, 30, tzinfo=timezone.utc)


def test_dst_timezone_conversion():
    assert scheduled_utc(datetime(2026, 11, 1, 1, 30), "America/New_York") == datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)


def test_occurrence_identity_is_stable_and_occurrence_specific():
    assert occurrence_id("u", "m", NOW) == occurrence_id("u", "m", NOW)
    assert occurrence_id("u", "m", NOW) != occurrence_id("u", "m", NOW + timedelta(minutes=1))


def test_concurrent_claim_race_has_one_winner():
    dispatcher = make_dispatcher()
    record = create_due(dispatcher)
    assert dispatcher._claim(record, NOW)
    assert dispatcher._claim(dict(record, status="pending", attemptCount=0), NOW) is None


def test_stale_claim_recovers_with_a_new_attempt():
    dispatcher = make_dispatcher()
    record = create_due(dispatcher, status=STATUS_CLAIMED)
    record.update({"attemptCount": 1, "claimedAtISO": (NOW - timedelta(minutes=11)).isoformat()})
    claimed = dispatcher._claim(record, NOW)
    assert claimed
    assert dispatcher.store.records[record["$id"]]["attemptCount"] == 2


def test_fresh_claim_is_not_recovered_during_scheduler_overlap():
    dispatcher = make_dispatcher()
    record = create_due(dispatcher, status=STATUS_CLAIMED)
    record.update({"attemptCount": 1, "claimedAtISO": (NOW - timedelta(minutes=1)).isoformat()})
    counters = dispatcher.run(now=NOW)
    assert counters["duplicate"] == 1
    assert not dispatcher.firebase.calls


def test_follow_up_cancels_for_exact_taken_occurrence():
    dispatcher = make_dispatcher()
    record = create_due(dispatcher, kind=KIND_FOLLOW_UP)
    dispatcher.store.logs = [{"occurrenceId": record["occurrenceId"], "status": "taken"}]
    counters = dispatcher.run(now=NOW)
    assert counters["cancelled_taken"] == 1


def test_follow_up_cancels_for_exact_skipped_occurrence():
    dispatcher = make_dispatcher()
    record = create_due(dispatcher, kind=KIND_FOLLOW_UP)
    dispatcher.store.logs = [{"occurrenceId": record["occurrenceId"], "status": "skipped"}]
    counters = dispatcher.run(now=NOW)
    assert counters["cancelled_skipped"] == 1


def test_unrelated_medicine_and_adjacent_dose_do_not_cancel_follow_up():
    dispatcher = make_dispatcher()
    record = create_due(dispatcher, kind=KIND_FOLLOW_UP)
    dispatcher.store.logs = [
        {"userId": "user-1", "medId": "other", "time": NOW.isoformat(), "status": "taken"},
        {"userId": "user-1", "medId": "med-1", "time": (NOW + timedelta(minutes=31)).isoformat(), "status": "skipped"},
    ]
    counters = dispatcher.run(now=NOW)
    assert counters["dispatched"] == 1


def test_failed_send_is_retryable_during_recovery_window():
    store = MemoryStore()
    dispatcher = make_dispatcher(store, Firebase({"success": False, "sent": 0, "failed": 1}))
    record = create_due(dispatcher)
    assert dispatcher.run(now=NOW)["failed"] == 1
    assert record["status"] == STATUS_FAILED
    dispatcher.firebase = Firebase()
    assert dispatcher.run(now=NOW + timedelta(minutes=1))["dispatched"] == 1


def test_fcm_success_then_state_write_failure_is_not_claimed_as_duplicate():
    store = MemoryStore()
    dispatcher = make_dispatcher(store)
    create_due(dispatcher)
    store.update_fail = True
    with pytest.raises(ReminderStoreError):
        dispatcher.run(now=NOW)
    assert len(dispatcher.firebase.calls) == 1


def test_partial_device_success_dispatches_once_with_client_deduplication_key():
    firebase = Firebase({"success": True, "sent": 1, "failed": 1})
    dispatcher = make_dispatcher(firebase=firebase)
    create_due(dispatcher)
    assert dispatcher.run(now=NOW)["dispatched"] == 1
    data = firebase.calls[0]["data"]
    assert data["eventId"] == data["collapseKey"]


def test_scheduler_overlap_only_dispatches_an_occurrence_once():
    dispatcher = make_dispatcher()
    create_due(dispatcher, send_at=NOW - timedelta(seconds=90))
    assert dispatcher.run(now=NOW)["dispatched"] == 1
    assert dispatcher.run(now=NOW + timedelta(minutes=1))["dispatched"] == 0


def test_store_conflict_is_distinct_from_outage():
    class Proxy:
        def create_document(self, *args, **kwargs):
            raise AppwriteProxyError("conflict", status_code=409)

    with patch("services.notification_store.AppwriteProxy", return_value=Proxy()):
        store = NotificationStore()
        with pytest.raises(ReminderConflictError):
            store.create_reminder_record("record", {"userId": "u"})

    class OutageProxy:
        def create_document(self, *args, **kwargs):
            raise AppwriteProxyError("outage", status_code=503)

    with patch("services.notification_store.AppwriteProxy", return_value=OutageProxy()):
        store = NotificationStore()
        with pytest.raises(ReminderStoreError) as error:
            store.create_reminder_record("record", {"userId": "u"})
    assert not isinstance(error.value, ReminderConflictError)


def test_device_deletion_uses_registration_identity():
    class Proxy:
        def __init__(self):
            self.deleted = []

        def delete_document(self, resource, document_id):
            self.deleted.append((resource, document_id))

    proxy = Proxy()
    with patch("services.notification_store.AppwriteProxy", return_value=proxy):
        store = NotificationStore()
    expected = store.device_record_id(user_id="user-1", platform="android", token="token-1")
    assert store.delete_device(user_id="user-1", platform="android", token="token-1")
    assert proxy.deleted == [(store.devices_resource, expected)]
