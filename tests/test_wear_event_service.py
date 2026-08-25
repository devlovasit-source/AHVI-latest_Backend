from services.appwrite_proxy import AppwriteProxyError
from services.wear_event_service import (
    deterministic_appwrite_id,
    get_wear_history,
    record_wear,
)


class FakeWearEventsProxy:
    """In-memory stand-in for AppwriteProxy scoped to wear_events, matching
    the real create-with-explicit-id 409-on-duplicate contract."""

    def __init__(self):
        self.docs = {}
        self.create_calls = 0

    def create_document(self, resource, data, document_id="unique()"):
        assert resource == "wear_events"
        self.create_calls += 1
        if document_id in self.docs:
            raise AppwriteProxyError("Appwrite request failed (409): already exists", status_code=409)
        doc = {"$id": document_id, **data}
        self.docs[document_id] = doc
        return doc

    def get_document(self, resource, document_id):
        assert resource == "wear_events"
        return self.docs[document_id]

    def list_documents(self, resource, user_id=None, limit=500, **kwargs):
        assert resource == "wear_events"
        return [d for d in self.docs.values() if d.get("userId") == user_id]


def _patch(monkeypatch, fake):
    import services.wear_event_service as svc

    monkeypatch.setattr(svc, "_proxy", lambda: fake)


def test_deterministic_id_is_stable_and_bounded():
    a = deterministic_appwrite_id("u1", "wear", "item1", "2026-08-17")
    b = deterministic_appwrite_id("u1", "wear", "item1", "2026-08-17")
    c = deterministic_appwrite_id("u1", "wear", "item2", "2026-08-17")
    assert a == b
    assert a != c
    assert len(a) <= 36


def test_first_wear_creates_exactly_one_event(monkeypatch):
    fake = FakeWearEventsProxy()
    _patch(monkeypatch, fake)

    result = record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")

    assert result["newly_created"] is True
    assert len(fake.docs) == 1
    assert fake.create_calls == 1


def test_retry_same_logical_wear_creates_no_new_event(monkeypatch):
    fake = FakeWearEventsProxy()
    _patch(monkeypatch, fake)

    first = record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")
    second = record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T18:00:00+00:00")

    assert first["newly_created"] is True
    assert second["newly_created"] is False
    assert first["event"]["$id"] == second["event"]["$id"]
    assert len(fake.docs) == 1  # exactly one wear_events row, not two


def test_retry_does_not_double_increment_projection(monkeypatch):
    fake = FakeWearEventsProxy()
    _patch(monkeypatch, fake)

    projection_calls = []
    import services.style_memory_service as memory_svc

    original = memory_svc.record_wear

    def spy(**kwargs):
        projection_calls.append(kwargs)
        return original(**kwargs)

    # style_memory_service itself still needs a proxy; give it the same fake
    # outfit_history-shaped store by monkeypatching its own _proxy too, using
    # a minimal separate fake scoped to outfit_history.
    class FakeOutfitHistoryProxy:
        def __init__(self):
            self.docs = {}
            self._counter = 0

        def list_documents(self, resource, user_id=None, **kwargs):
            assert resource == "outfit_history"
            return [d for d in self.docs.values() if d.get("userId") == user_id]

        def create_document(self, resource, data, document_id=None):
            assert resource == "outfit_history"
            self._counter += 1
            doc_id = document_id or f"doc{self._counter}"
            doc = {"$id": doc_id, **data}
            self.docs[doc_id] = doc
            return doc

        def update_document(self, resource, document_id, patch):
            assert resource == "outfit_history"
            doc = self.docs.get(document_id, {})
            doc.update(patch)
            return doc

    fake_outfit_history = FakeOutfitHistoryProxy()
    monkeypatch.setattr(memory_svc, "_proxy", lambda: fake_outfit_history)
    monkeypatch.setattr(memory_svc, "record_wear", spy)

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T18:00:00+00:00")
    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T20:00:00+00:00")

    assert len(projection_calls) == 1  # only the newly-created wear drove the projection
    assert fake_outfit_history.docs
    row = next(iter(fake_outfit_history.docs.values()))
    assert row["wearCount"] == 1


def test_different_items_isolated(monkeypatch):
    fake = FakeWearEventsProxy()
    _patch(monkeypatch, fake)

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u1", item_id="pants1", occurred_at_iso="2026-08-17T09:00:00+00:00")

    assert len(fake.docs) == 2


def test_different_users_isolated(monkeypatch):
    fake = FakeWearEventsProxy()
    _patch(monkeypatch, fake)

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u2", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")

    assert len(fake.docs) == 2
    history_u1 = get_wear_history(user_id="u1", item_id="shirt1")
    history_u2 = get_wear_history(user_id="u2", item_id="shirt1")
    assert history_u1["total_wears"] == 1
    assert history_u2["total_wears"] == 1


def test_wear_history_totals_match_canonical_event_count(monkeypatch):
    fake = FakeWearEventsProxy()
    _patch(monkeypatch, fake)

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-18T09:00:00+00:00")
    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-19T09:00:00+00:00")

    history = get_wear_history(user_id="u1", item_id="shirt1")
    assert history["total_wears"] == 3
    assert len(history["events"]) == 3
    assert history["last_worn_at"] == "2026-08-19T09:00:00+00:00"


def test_revoked_events_excluded_from_history(monkeypatch):
    fake = FakeWearEventsProxy()
    _patch(monkeypatch, fake)

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")
    result = record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-18T09:00:00+00:00")

    # No revoke endpoint exists yet — directly flip status to prove the data
    # model/read path honors it once something (future Phase) does revoke.
    event_id = result["event"]["$id"]
    fake.docs[event_id]["status"] = "revoked"

    history = get_wear_history(user_id="u1", item_id="shirt1")
    assert history["total_wears"] == 1


def test_item_a_wear_does_not_alter_item_b(monkeypatch):
    fake = FakeWearEventsProxy()
    _patch(monkeypatch, fake)

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")

    history_b = get_wear_history(user_id="u1", item_id="pants1")
    assert history_b["total_wears"] == 0
