"""record_wear must keep the wardrobe item's own `worn` display field in
sync with the canonical wear_events total, so the wardrobe grid / AI-insight
copy (which reads the item's own document, never wear_events) doesn't go
stale after a relaunch. This regressed when the direct client-side Appwrite
write (`_updateOutfitDocument(item.id, {'worn': item.worn})`) was replaced
by the canonical WearEventService call — nothing else was patching that
field anymore. Discovered via device E2E (relaunch showed a real Wear
History total of 1 but the item card/pill reverted to "Never worn")."""

from services.appwrite_proxy import AppwriteProxyError
from services.wear_event_service import record_wear


class FakeWearEventsProxy:
    def __init__(self):
        self.docs = {}

    def create_document(self, resource, data, document_id="unique()"):
        if document_id in self.docs:
            raise AppwriteProxyError("already exists", status_code=409)
        doc = {"$id": document_id, **data}
        self.docs[document_id] = doc
        return doc

    def get_document(self, resource, document_id):
        return self.docs[document_id]

    def list_documents(self, resource, user_id=None, limit=500, **kwargs):
        return [d for d in self.docs.values() if d.get("userId") == user_id]


class FakeOutfitHistoryProxy:
    def list_documents(self, *a, **k):
        return []

    def create_document(self, *a, **k):
        return {}

    def update_document(self, *a, **k):
        return {}


def _wire(monkeypatch):
    import services.wear_event_service as wear_svc
    import services.style_memory_service as memory_svc

    fake_events = FakeWearEventsProxy()
    monkeypatch.setattr(wear_svc, "_proxy", lambda: fake_events)
    monkeypatch.setattr(memory_svc, "_proxy", lambda: FakeOutfitHistoryProxy())
    return fake_events


def _stub_persistence(monkeypatch, *, item_id, existing_doc):
    import services.wardrobe_persistence_service as persistence

    patch_calls = []

    def fake_fetch_document(document_id):
        assert document_id == item_id
        return (existing_doc, "outfits", "db")

    def fake_patch_document(document_id, data, collection_id="", database_id=""):
        patch_calls.append((document_id, dict(data), collection_id, database_id))
        return ({"$id": document_id, **data}, [])

    monkeypatch.setattr(persistence, "_fetch_document", fake_fetch_document)
    monkeypatch.setattr(persistence, "_patch_document", fake_patch_document)
    return patch_calls


def test_first_wear_patches_item_worn_to_one(monkeypatch):
    _wire(monkeypatch)
    patch_calls = _stub_persistence(monkeypatch, item_id="shirt1", existing_doc={"$id": "shirt1", "worn": 0})

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")

    assert patch_calls == [("shirt1", {"worn": 1}, "outfits", "db")]


def test_duplicate_wear_does_not_repatch_item_worn(monkeypatch):
    _wire(monkeypatch)
    patch_calls = _stub_persistence(monkeypatch, item_id="shirt1", existing_doc={"$id": "shirt1", "worn": 0})

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T18:00:00+00:00")

    assert len(patch_calls) == 1  # only the newly-created wear syncs the item


def test_third_distinct_day_wear_patches_item_worn_to_three(monkeypatch):
    _wire(monkeypatch)
    patch_calls = _stub_persistence(monkeypatch, item_id="shirt1", existing_doc={"$id": "shirt1", "worn": 0})

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-18T09:00:00+00:00")
    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-19T09:00:00+00:00")

    assert [c[1]["worn"] for c in patch_calls] == [1, 2, 3]


def test_item_worn_sync_failure_does_not_break_the_wear(monkeypatch):
    _wire(monkeypatch)
    import services.wardrobe_persistence_service as persistence

    def boom(*a, **k):
        raise RuntimeError("appwrite outfits collection unreachable")

    monkeypatch.setattr(persistence, "_fetch_document", boom)

    result = record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")

    assert result["newly_created"] is True


def test_different_items_get_independent_worn_counts(monkeypatch):
    _wire(monkeypatch)
    calls_shirt = _stub_persistence(monkeypatch, item_id="shirt1", existing_doc={"$id": "shirt1", "worn": 0})

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00")

    calls_pants = _stub_persistence(monkeypatch, item_id="pants1", existing_doc={"$id": "pants1", "worn": 0})
    record_wear(user_id="u1", item_id="pants1", occurred_at_iso="2026-08-17T09:00:00+00:00")

    assert calls_shirt == [("shirt1", {"worn": 1}, "outfits", "db")]
    assert calls_pants == [("pants1", {"worn": 1}, "outfits", "db")]
