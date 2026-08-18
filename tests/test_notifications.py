import os
import sys
import threading
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeProxy:
    # Appwrite's real limit on custom document ids - enforced here so any
    # id-generation bug (like the 41-char ticket id caught in review) fails
    # a test locally instead of only failing silently against real Appwrite.
    APPWRITE_MAX_DOCUMENT_ID_LENGTH = 36

    def __init__(self):
        self.created = []
        self.updated = []
        self.deleted = []
        self.rows = []

    def _check_document_id_length(self, document_id):
        if document_id and len(document_id) > self.APPWRITE_MAX_DOCUMENT_ID_LENGTH:
            raise Exception(
                f"Appwrite request failed (400): document id '{document_id}' "
                f"is {len(document_id)} chars, exceeds the "
                f"{self.APPWRITE_MAX_DOCUMENT_ID_LENGTH}-char limit"
            )

    def update_document(self, resource, document_id, data):
        self._check_document_id_length(document_id)
        self.updated.append((resource, document_id, data))
        return {"$id": document_id, **data}

    def create_document(self, resource, data, document_id=None):
        self._check_document_id_length(document_id)
        self.created.append((resource, document_id, data))
        return {"$id": document_id or "doc", **data}

    def delete_document(self, resource, document_id):
        self.deleted.append((resource, document_id))
        return {"success": True}

    def find_by_attribute(self, resource, attribute, value, **kwargs):
        return [
            r
            for r in self.rows
            if isinstance(r, dict) and str(r.get(attribute)) == str(value)
        ]

    def list_documents(self, resource, **kwargs):
        return list(self.rows)

    def get_document(self, resource, document_id):
        for row in self.rows:
            if isinstance(row, dict) and row.get("$id") == document_id:
                return dict(row)
        raise Exception(
            f"Appwrite request failed (404): document '{document_id}' not found"
        )


class ClaimAwareFakeProxy(FakeProxy):
    """
    FakeProxy that actually simulates Appwrite's real create-document
    behaviour (fixed id -> 409 on a duplicate create), so tests can exercise
    try_claim_reminder's real conflict-detection logic instead of mocking
    the method away entirely.

    Each operation is guarded by a real lock, matching the guarantee real
    Appwrite gives (create/update/delete are each individually atomic) -
    that's what makes a genuine multi-threaded test of our claim logic
    meaningful: any race that shows up has to come from OUR multi-step
    logic on top of these primitives, not from the fake's own bookkeeping.
    """

    def __init__(self):
        super().__init__()
        self.store = {}  # (resource, doc_id) -> stored document
        self._lock = threading.Lock()

    def create_document(self, resource, data, document_id=None):
        self._check_document_id_length(document_id)
        with self._lock:
            key = (resource, document_id)
            if document_id and key in self.store:
                raise Exception(
                    "Appwrite request failed (409): document already exists"
                )
            doc = {"$id": document_id or "doc", **data}
            if document_id:
                self.store[key] = doc
        self.created.append((resource, document_id, data))
        return doc

    def update_document(self, resource, document_id, data):
        self._check_document_id_length(document_id)
        key = (resource, document_id)
        with self._lock:
            merged = {**self.store.get(key, {}), **data}
            self.store[key] = merged
        self.updated.append((resource, document_id, data))
        return merged

    def get_document(self, resource, document_id):
        key = (resource, document_id)
        with self._lock:
            if key not in self.store:
                raise Exception("Appwrite request failed (404): document not found")
            return dict(self.store[key])

    def delete_document(self, resource, document_id):
        with self._lock:
            self.store.pop((resource, document_id), None)
        self.deleted.append((resource, document_id))
        return {"success": True}


class NotificationStoreTests(unittest.TestCase):
    def test_notification_resources_use_collection_env_names(self):
        with patch.dict(
            os.environ,
            {
                "APPWRITE_COLLECTION_NOTIFICATION_DEVICES": "devices_collection",
                "APPWRITE_COLLECTION_NOTIFICATION_REMINDERS": "reminders_collection",
            },
            clear=False,
        ):
            with patch("services.notification_store.AppwriteProxy", return_value=FakeProxy()):
                from services.notification_store import NotificationStore

                store = NotificationStore()

        self.assertEqual(store.devices_resource, "devices_collection")
        self.assertEqual(store.reminders_resource, "reminders_collection")

    def test_device_registration_upserts_user_token(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            doc_id = store.upsert_device(
                user_id="user_1",
                platform="android",
                token="token_12345678901234567890",
            )

        self.assertTrue(doc_id.startswith("dev_"))
        self.assertEqual(proxy.updated[0][0], "notification_devices")
        self.assertEqual(proxy.updated[0][2]["userId"], "user_1")
        self.assertEqual(proxy.updated[0][2]["platform"], "android")

    def test_schedule_reminders_persists_scheduled_rows(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            result = store.schedule_reminders(
                user_id="user_1",
                event_id="med_1",
                source="medi",
                reminders=[
                    {
                        "status": "pending",
                        "sendAtISO": "2026-05-01T12:00:00+00:00",
                        "message": "Take vitamin D",
                        "priority": "high",
                        "offsetMinutes": 0,
                    }
                ],
            )

        self.assertEqual(result["scheduled"], 1)
        self.assertTrue(result["success"])
        self.assertEqual(proxy.updated[0][0], "notification_reminders")
        # Caller-local states are normalized to the state scanned by dispatch.
        self.assertEqual(proxy.updated[0][2]["status"], "scheduled")
        self.assertEqual(proxy.updated[0][2]["message"], "Take vitamin D")
        self.assertEqual(proxy.updated[0][2]["lastError"], "")
        self.assertEqual(proxy.updated[0][2]["source"], "medi")


    def test_preference_defaults_true_for_new_user(self):
        proxy = FakeProxy()  # rows stays empty -> no existing preference doc
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            result = store.get_notification_preference(user_id="new_user", category="medi")

        self.assertTrue(result)

    def test_preference_lookup_failure_defaults_true(self):
        class FailingProxy(FakeProxy):
            def get_document(self, resource, document_id):
                raise RuntimeError("appwrite down")

        with patch("services.notification_store.AppwriteProxy", return_value=FailingProxy()):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            result = store.get_notification_preference(user_id="user_1", category="calendar")

        self.assertTrue(result)

    def test_preference_invalid_category_defaults_true(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            result = store.get_notification_preference(
                user_id="user_1", category="not_a_real_category"
            )

        self.assertTrue(result)

    def test_set_then_get_preference_roundtrip_disabled(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            ok = store.set_notification_preference(
                user_id="user_1", category="style", enabled=False
            )
            self.assertTrue(ok)

            # FakeProxy.update_document/create_document don't feed
            # get_document's `rows` lookup, so mirror what a real Appwrite
            # read-back would return - keyed by the SAME deterministic id
            # get_notification_preference will actually look up.
            doc_id = store._preference_doc_id("user_1", "style")
            proxy.rows = [{"$id": doc_id, "category": "style", "enabled": False}]
            result = store.get_notification_preference(user_id="user_1", category="style")

        self.assertFalse(result)

    def test_set_preference_rejects_invalid_category(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            ok = store.set_notification_preference(
                user_id="user_1", category="bogus", enabled=True
            )

        self.assertFalse(ok)


    # -------------------------
    # get_notification_preference_for_dispatch (tri-state)
    # -------------------------
    def test_dispatch_preference_no_record_defaults_enabled(self):
        proxy = FakeProxy()  # rows stays empty
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            enabled, pref_status = store.get_notification_preference_for_dispatch(
                user_id="user_1", category="medi"
            )

        self.assertTrue(enabled)
        self.assertEqual(pref_status, "no_record")

    def test_dispatch_preference_explicit_disabled(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            doc_id = store._preference_doc_id("user_1", "medi")
            proxy.rows = [{"$id": doc_id, "category": "medi", "enabled": False}]
            enabled, pref_status = store.get_notification_preference_for_dispatch(
                user_id="user_1", category="medi"
            )

        self.assertFalse(enabled)
        self.assertEqual(pref_status, "disabled")

    def test_dispatch_preference_lookup_failure_does_not_default_enabled(self):
        """
        This is the boss-requested fix: a real lookup failure must be
        distinguishable from "no record", and must NOT be treated as enabled.
        """

        class FailingProxy(FakeProxy):
            def get_document(self, resource, document_id):
                raise RuntimeError("appwrite down")

        with patch(
            "services.notification_store.AppwriteProxy",
            return_value=FailingProxy(),
        ):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            enabled, pref_status = store.get_notification_preference_for_dispatch(
                user_id="user_1", category="medi"
            )

        self.assertFalse(enabled)
        self.assertEqual(pref_status, "lookup_failed")

    def test_preference_reads_are_isolated_per_user(self):
        """
        Boss-flagged P0: notification_preferences was never registered in
        AppwriteProxy's user_field_map, so list_documents' user_id filter
        was silently ignored for this resource - one user's row for a
        category could be returned for a completely different user. Reads
        now go straight to the exact deterministic (user, category)
        document instead, which makes cross-user leakage structurally
        impossible rather than dependent on a filter being configured
        correctly somewhere else.

        Two users, opposite values, same category, same fake backend.
        """
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            alice_doc_id = store._preference_doc_id("user_alice", "medi")
            bob_doc_id = store._preference_doc_id("user_bob", "medi")
            self.assertNotEqual(
                alice_doc_id, bob_doc_id, "sanity check: ids must differ per user"
            )

            proxy.rows = [
                {"$id": alice_doc_id, "category": "medi", "enabled": False},
                {"$id": bob_doc_id, "category": "medi", "enabled": True},
            ]

            alice_result = store.get_notification_preference(
                user_id="user_alice", category="medi"
            )
            bob_result = store.get_notification_preference(
                user_id="user_bob", category="medi"
            )

            alice_dispatch, alice_status = (
                store.get_notification_preference_for_dispatch(
                    user_id="user_alice", category="medi"
                )
            )
            bob_dispatch, bob_status = store.get_notification_preference_for_dispatch(
                user_id="user_bob", category="medi"
            )

        self.assertFalse(alice_result, "Alice disabled medi - must read as disabled")
        self.assertTrue(bob_result, "Bob enabled medi - must read as enabled")
        self.assertFalse(alice_dispatch)
        self.assertEqual(alice_status, "disabled")
        self.assertTrue(bob_dispatch)
        self.assertEqual(bob_status, "enabled")


class NotificationClaimTests(unittest.TestCase):
    """
    Duplicate-prevention: covers concurrent/double claim attempts, expired
    claim recovery, and already-sent reminders - the three scenarios the
    boss explicitly asked to have tested before the automatic scheduler is
    connected.
    """

    def _store(self, proxy):
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            return NotificationStore()

    def test_new_claim_succeeds(self):
        store = self._store(ClaimAwareFakeProxy())
        result = store.try_claim_reminder(reminder_doc_id="rem_1")

        self.assertTrue(result["claimed"])
        self.assertEqual(result["reason"], "new_claim")

    def test_concurrent_double_claim_is_rejected(self):
        """
        Simulates two dispatch runs racing over the exact same reminder:
        the second claim attempt must lose.
        """
        proxy = ClaimAwareFakeProxy()
        store = self._store(proxy)

        first = store.try_claim_reminder(reminder_doc_id="rem_2")
        second = store.try_claim_reminder(reminder_doc_id="rem_2")

        self.assertTrue(first["claimed"])
        self.assertFalse(second["claimed"])
        self.assertEqual(second["reason"], "already_processing")

    def test_expired_claim_is_recovered(self):
        """
        A claim stuck in "processing" past its TTL (e.g. the process that
        claimed it crashed mid-send) must be recoverable by a later run.
        """
        proxy = ClaimAwareFakeProxy()
        store = self._store(proxy)

        first = store.try_claim_reminder(reminder_doc_id="rem_3")
        self.assertTrue(first["claimed"])

        # Simulate time passing well beyond the claim TTL by directly
        # backdating the stored claim's expiry.
        key = (store.dispatch_claims_resource, "rem_3")
        proxy.store[key]["expiresAt"] = 0

        second = store.try_claim_reminder(reminder_doc_id="rem_3")
        self.assertTrue(second["claimed"])
        self.assertEqual(second["reason"], "reclaimed_expired")

    def test_already_sent_claim_cannot_be_reclaimed(self):
        """
        Once a claim reaches the terminal "sent" state, no later dispatch
        run - even after the TTL would otherwise have expired - may claim
        it again. This is the core "at most one FCM notification" guarantee.
        """
        proxy = ClaimAwareFakeProxy()
        store = self._store(proxy)

        first = store.try_claim_reminder(reminder_doc_id="rem_4")
        self.assertTrue(first["claimed"])
        store.finalize_claim(reminder_doc_id="rem_4", status="sent")

        second = store.try_claim_reminder(reminder_doc_id="rem_4")
        self.assertFalse(second["claimed"])
        self.assertEqual(second["reason"], "already_sent")

    def test_release_claim_allows_retry(self):
        """
        release_claim (used for preference_lookup_failed) must actually free
        the occurrence up for a subsequent claim attempt.
        """
        proxy = ClaimAwareFakeProxy()
        store = self._store(proxy)

        first = store.try_claim_reminder(reminder_doc_id="rem_5")
        self.assertTrue(first["claimed"])

        store.release_claim(reminder_doc_id="rem_5")

        second = store.try_claim_reminder(reminder_doc_id="rem_5")
        self.assertTrue(second["claimed"])
        self.assertEqual(second["reason"], "new_claim")

    def test_sequential_takeover_of_expired_claim_only_one_wins(self):
        """
        Baseline (non-concurrent) sanity check: calling takeover twice in a
        row on the same expired claim must only let the first call win.
        See test_concurrent_interleaved_takeover_never_double_claims below
        for the actual race-condition regression test using real threads.
        """
        proxy = ClaimAwareFakeProxy()
        store_a = self._store(proxy)
        store_b = self._store(proxy)  # a second "worker" sharing the same backend

        first = store_a.try_claim_reminder(reminder_doc_id="rem_6")
        self.assertTrue(first["claimed"])

        # Backdate so both competitors see it as expired.
        key = (store_a.dispatch_claims_resource, "rem_6")
        proxy.store[key]["expiresAt"] = 0

        result_a = store_a.try_claim_reminder(reminder_doc_id="rem_6")
        result_b = store_b.try_claim_reminder(reminder_doc_id="rem_6")

        won = [r for r in (result_a, result_b) if r["claimed"]]
        lost = [r for r in (result_a, result_b) if not r["claimed"]]

        self.assertEqual(len(won), 1, "exactly one competitor may win the takeover")
        self.assertEqual(len(lost), 1)
        self.assertEqual(lost[0]["reason"], "already_processing")

    def test_concurrent_interleaved_takeover_never_double_claims(self):
        """
        This is the test the boss's review specifically asked for: the
        earlier sequential version (A then B, in order) can't expose a real
        race, because nothing forces the two attempts to actually overlap.
        This one uses real threads plus a barrier to force two independent
        NotificationStore instances to attempt takeover of the SAME expired
        claim at the same moment, repeated across many trials (since a race
        window may not manifest on every single run).

        Asserts the invariant that actually matters: across all trials,
        it is never possible for both contenders to win the same takeover.
        """
        trials = 50
        double_wins = []

        for trial in range(trials):
            proxy = ClaimAwareFakeProxy()
            store_a = self._store(proxy)
            store_b = self._store(proxy)

            rid = f"rem_race_{trial}"
            seeded = store_a.try_claim_reminder(reminder_doc_id=rid)
            self.assertTrue(seeded["claimed"])

            # Seed an already-expired claim, the precondition for a takeover.
            key = (store_a.dispatch_claims_resource, rid)
            proxy.store[key]["expiresAt"] = 0

            results: Dict[str, Dict[str, Any]] = {}
            barrier = threading.Barrier(2)

            def attempt(name: str, store) -> None:
                barrier.wait()  # force both threads to start together
                results[name] = store.try_claim_reminder(reminder_doc_id=rid)

            t_a = threading.Thread(target=attempt, args=("a", store_a))
            t_b = threading.Thread(target=attempt, args=("b", store_b))
            t_a.start()
            t_b.start()
            t_a.join()
            t_b.join()

            won = [r for r in results.values() if r["claimed"]]
            if len(won) > 1:
                double_wins.append(trial)

        self.assertEqual(
            double_wins,
            [],
            f"double-claimed the same takeover on trials {double_wins} - "
            f"this means duplicate FCM sends are possible",
        )


class DispatchPreferenceFailClosedTests(unittest.TestCase):
    """
    Boss-flagged fix: an unrecognized category (or a reminder with no
    userId) must fail CLOSED, not default to enabled like a genuine
    no-record case does.
    """

    def test_unknown_category_fails_closed(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            enabled, pref_status = store.get_notification_preference_for_dispatch(
                user_id="user_1", category="ahvi_plan_pack"
            )

        self.assertFalse(enabled)
        self.assertEqual(pref_status, "invalid_category")

    def test_missing_user_id_fails_closed(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            enabled, pref_status = store.get_notification_preference_for_dispatch(
                user_id="", category="medi"
            )

        self.assertFalse(enabled)
        self.assertEqual(pref_status, "invalid_user")


class ReminderDedupeIdentityTests(unittest.TestCase):
    """
    Boss-flagged fix: the reminder occurrence id must NOT include message
    text, so a wording change updates the same occurrence instead of
    creating a duplicate reminder for it.
    """

    def test_message_change_updates_same_occurrence_not_a_new_one(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            common = dict(
                user_id="user_1",
                event_id="med_1",
                source="medi",
            )
            store.schedule_reminders(
                **common,
                reminders=[
                    {
                        "sendAtISO": "2026-05-01T12:00:00+00:00",
                        "message": "Take vitamin D",
                        "offsetMinutes": 0,
                    }
                ],
            )
            store.schedule_reminders(
                **common,
                reminders=[
                    {
                        "sendAtISO": "2026-05-01T12:00:00+00:00",
                        "message": "Time for your vitamin D!",  # reworded
                        "offsetMinutes": 0,
                    }
                ],
            )

        # Same user+source+event+sendAt+offset -> same doc id both times ->
        # the second call must be an update, not a second created row.
        self.assertEqual(len(proxy.updated), 2)
        self.assertEqual(len(proxy.created), 0)
        first_doc_id = proxy.updated[0][1]
        second_doc_id = proxy.updated[1][1]
        self.assertEqual(first_doc_id, second_doc_id)

    def test_different_offset_is_a_different_occurrence(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            store.schedule_reminders(
                user_id="user_1",
                event_id="med_1",
                source="medi",
                reminders=[
                    {
                        "sendAtISO": "2026-05-01T12:00:00+00:00",
                        "message": "Take vitamin D",
                        "offsetMinutes": 0,
                    },
                    {
                        "sendAtISO": "2026-05-01T12:00:00+00:00",
                        "message": "Take vitamin D",
                        "offsetMinutes": 15,
                    },
                ],
            )

        doc_ids = {u[1] for u in proxy.updated}
        self.assertEqual(len(doc_ids), 2, "different offsets are different occurrences")


class DeviceUnregisterTests(unittest.TestCase):
    """
    Boss-flagged bug: registration hashes userId|platform|token, but
    deletion was hashing token alone - essentially never matching, leaving
    stale FCM device rows behind. delete_device now looks the row up by its
    actual token field instead of guessing an id.
    """

    def test_delete_device_finds_row_registered_under_a_different_id_scheme(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            store.upsert_device(
                user_id="user_1",
                platform="android",
                token="token_12345678901234567890",
            )
            # upsert_device only records the call in proxy.updated/created -
            # mirror it into `rows` so find_by_attribute (which searches
            # rows, matching how a real Appwrite read-back would look) can
            # see it, the same pattern used for the preference roundtrip test.
            # Real Appwrite documents always carry their own $id - include
            # one here too, since delete_device needs it to issue the delete.
            proxy.rows = [{"$id": "dev_doc_1", **proxy.updated[0][2]}]

            ok = store.delete_device(token="token_12345678901234567890")

        self.assertTrue(ok)
        self.assertEqual(len(proxy.deleted), 1)

    def test_delete_device_with_no_matching_row_returns_false(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            ok = store.delete_device(token="token_never_registered_000000")

        self.assertFalse(ok)
        self.assertEqual(len(proxy.deleted), 0)


class DocumentIdLengthConstraintTests(unittest.TestCase):
    """
    Boss-flagged bug: _hash_id("claimgen", ..., length=32) produced a
    41-character document id, over Appwrite's real 36-character limit for
    custom document ids - invisible against the test fakes (which didn't
    enforce the limit) and only surfaced against real Appwrite. These tests
    make that constraint a first-class, always-exercised part of the suite:
    one directly on the id-generation helper, one on the fake proxy used by
    the claim/takeover tests, so any future id scheme that overflows fails
    a local test run instead of only failing in production.
    """

    def test_hash_id_rejects_a_too_long_prefix(self):
        from services.notification_store import (
            APPWRITE_MAX_DOCUMENT_ID_LENGTH,
            _hash_id,
        )

        # The exact bug that was caught in review: an 8-char prefix
        # ("claimgen") + "_" + 32 hash chars = 41 chars, over the limit.
        with self.assertRaises(ValueError):
            _hash_id("claimgen", "some raw value", length=32)

        # The fixed prefix ("cg") comes in comfortably under the limit.
        from services.notification_store import _hash_id as hash_id_ok

        ok_id = hash_id_ok("cg", "some raw value", length=32)
        self.assertLessEqual(len(ok_id), APPWRITE_MAX_DOCUMENT_ID_LENGTH)

    def test_fake_proxy_rejects_a_too_long_document_id(self):
        proxy = ClaimAwareFakeProxy()
        with self.assertRaises(Exception):
            proxy.create_document(
                "notification_dispatch_claims",
                {"status": "processing"},
                document_id="x" * 41,
            )

    def test_takeover_path_produces_ids_within_the_real_appwrite_limit(self):
        """
        End-to-end confirmation: running an actual takeover (the code path
        that generates ticket ids) against a proxy that enforces Appwrite's
        real length limit must succeed without hitting that guard.
        """
        proxy = ClaimAwareFakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            first = store.try_claim_reminder(reminder_doc_id="rem_length_check")
            self.assertTrue(first["claimed"])

            key = (store.dispatch_claims_resource, "rem_length_check")
            proxy.store[key]["expiresAt"] = 0

            second = store.try_claim_reminder(reminder_doc_id="rem_length_check")

        self.assertTrue(second["claimed"])
        self.assertEqual(second["reason"], "reclaimed_expired")


if __name__ == "__main__":
    unittest.main()