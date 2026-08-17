import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeProxy:
    def __init__(self):
        self.created = []
        self.updated = []
        self.deleted = []
        self.rows = []

    def update_document(self, resource, document_id, data):
        self.updated.append((resource, document_id, data))
        return {"$id": document_id, **data}

    def create_document(self, resource, data, document_id=None):
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


class ClaimAwareFakeProxy(FakeProxy):
    """
    FakeProxy that actually simulates Appwrite's real create-document
    behaviour (fixed id -> 409 on a duplicate create), so tests can exercise
    try_claim_reminder's real conflict-detection logic instead of mocking
    the method away entirely.
    """

    def __init__(self):
        super().__init__()
        self.store = {}  # (resource, doc_id) -> stored document

    def create_document(self, resource, data, document_id=None):
        key = (resource, document_id)
        if document_id and key in self.store:
            raise Exception("Appwrite request failed (409): document already exists")
        doc = {"$id": document_id or "doc", **data}
        if document_id:
            self.store[key] = doc
        self.created.append((resource, document_id, data))
        return doc

    def update_document(self, resource, document_id, data):
        key = (resource, document_id)
        merged = {**self.store.get(key, {}), **data}
        self.store[key] = merged
        self.updated.append((resource, document_id, data))
        return merged

    def get_document(self, resource, document_id):
        key = (resource, document_id)
        if key not in self.store:
            raise Exception("Appwrite request failed (404): document not found")
        return self.store[key]

    def delete_document(self, resource, document_id):
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
            def list_documents(self, resource, **kwargs):
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

            # FakeProxy.update_document/create_document don't feed list_documents'
            # `rows`, so mirror what a real Appwrite read-back would return.
            proxy.rows = [{"category": "style", "enabled": False}]
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
        proxy.rows = [{"category": "medi", "enabled": False}]
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
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
            def list_documents(self, resource, **kwargs):
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

    def test_concurrent_takeover_of_expired_claim_only_one_wins(self):
        """
        Boss-flagged race condition: update_document() is a blind overwrite,
        so two dispatch runs both seeing the SAME expired claim could
        previously both "reclaim" it via update and both proceed to send.
        The fix routes takeover through delete-then-create instead, since
        create is the one operation Appwrite guarantees is exclusive.
        Exactly one of two competing takeover attempts must win.
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


if __name__ == "__main__":
    unittest.main()