"""Concurrent idempotency proof for wear_event_service.record_wear.

The correctness guarantee doesn't come from anything in this codebase being
thread-safe — it comes from Appwrite enforcing document-id uniqueness
atomically server-side (a real create-with-id race always yields exactly one
201 and N-1 409s, never two 201s). This fake reproduces that exact atomicity
with a lock around the check-then-insert, so a genuine race between Python
threads exercises the same "loser gets 409, treats it as duplicate success"
path record_wear relies on — proving the invariant holds under a real race,
not just under sequential retries (already covered by
test_wear_event_service.py).
"""

import threading

from services.appwrite_proxy import AppwriteProxyError
from services.wear_event_service import record_wear


class LockedFakeWearEventsProxy:
    def __init__(self):
        self.docs = {}
        self._lock = threading.Lock()
        self.create_attempts = 0

    def create_document(self, resource, data, document_id="unique()"):
        with self._lock:
            self.create_attempts += 1
            if document_id in self.docs:
                raise AppwriteProxyError("already exists", status_code=409)
            doc = {"$id": document_id, **data}
            self.docs[document_id] = doc
            return doc

    def get_document(self, resource, document_id):
        with self._lock:
            return self.docs[document_id]

    def list_documents(self, resource, user_id=None, limit=500, **kwargs):
        with self._lock:
            return [d for d in self.docs.values() if d.get("userId") == user_id]


class FakeOutfitHistoryProxy:
    def __init__(self):
        self._lock = threading.Lock()
        self.docs = {}
        self._counter = 0
        self.create_calls = 0

    def list_documents(self, resource, user_id=None, **kwargs):
        with self._lock:
            return [d for d in self.docs.values() if d.get("userId") == user_id]

    def create_document(self, resource, data, document_id=None):
        with self._lock:
            self.create_calls += 1
            self._counter += 1
            doc_id = document_id or f"doc{self._counter}"
            doc = {"$id": doc_id, **data}
            self.docs[doc_id] = doc
            return doc

    def update_document(self, resource, document_id, patch):
        with self._lock:
            doc = self.docs.get(document_id, {})
            doc.update(patch)
            return doc


def test_concurrent_same_logical_wear_yields_exactly_one_event_and_one_projection(monkeypatch):
    import services.wear_event_service as wear_svc
    import services.style_memory_service as memory_svc

    fake_events = LockedFakeWearEventsProxy()
    fake_history = FakeOutfitHistoryProxy()
    monkeypatch.setattr(wear_svc, "_proxy", lambda: fake_events)
    monkeypatch.setattr(memory_svc, "_proxy", lambda: fake_history)

    n_threads = 12
    results = [None] * n_threads
    start_gate = threading.Barrier(n_threads)

    def worker(i):
        start_gate.wait()  # maximize actual overlap, not just "close in time"
        results[i] = record_wear(
            user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-17T09:00:00+00:00"
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(fake_events.docs) == 1, "CONCURRENT_CANONICAL_EVENTS must be 1"
    assert fake_history.create_calls == 1, "CONCURRENT_PROJECTION_INCREMENTS must be 1"

    newly_created_count = sum(1 for r in results if r["newly_created"])
    assert newly_created_count == 1, "exactly one caller must observe newly_created=True"
    event_ids = {r["event"]["$id"] for r in results}
    assert len(event_ids) == 1, "every caller must resolve to the same canonical event"

    row = next(iter(fake_history.docs.values()))
    assert row["wearCount"] == 1
