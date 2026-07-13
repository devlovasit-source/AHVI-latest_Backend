"""Durable, atomic style-board shuffle state (Batch 6 contract).

Covers: durability across store/service re-instantiation, atomic revision
claims under concurrency, ownership, unknown/legacy boards, storage outage
(no in-memory production fallback), and persistence of locked item ids,
image URLs, positions and source policy.
"""

from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from services import style_board_shuffle_service as sbs
from services.appwrite_proxy import AppwriteProxyError
from services.style_board_state_store import (
    AppwriteBoardStateStore,
    BoardRevisionExistsError,
    BoardStateStoreError,
    InMemoryBoardStateStore,
    revision_document_id,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeAppwriteProxy:
    """Dict-backed stand-in for AppwriteProxy exercising the production
    AppwriteBoardStateStore code paths (409 mapping, JSON payloads, queries).
    Thread-safe so concurrency tests are meaningful."""

    def __init__(self, docs=None):
        self.docs = docs if docs is not None else {}
        self._lock = threading.Lock()

    # -- surface used by AppwriteBoardStateStore ---------------------------
    def create_document(self, resource, data, document_id="unique()"):
        with self._lock:
            if document_id in self.docs:
                raise AppwriteProxyError("document already exists", status_code=409)
            doc = {**data, "$id": document_id}
            self.docs[document_id] = doc
            return dict(doc)

    def get_document(self, resource, document_id):
        doc = self.docs.get(document_id)
        if doc is None:
            raise AppwriteProxyError("document not found", status_code=404)
        return dict(doc)

    @staticmethod
    def _serialize_query_token(token):
        return json.dumps(token, separators=(",", ":"))

    def _url(self, collection_id, document_id=None):
        return f"fake://{collection_id}"

    def _request(self, method, url, params=None, payload=None):
        # Only the strict get_latest list query uses this path.
        return {"documents": [dict(d) for d in self.docs.values()], "total": len(self.docs)}


class _OutageProxy(_FakeAppwriteProxy):
    def create_document(self, *a, **kw):
        raise AppwriteProxyError("Appwrite connection failed: boom")

    def get_document(self, *a, **kw):
        raise AppwriteProxyError("Appwrite connection failed: boom")

    def _request(self, *a, **kw):
        raise AppwriteProxyError("Appwrite connection failed: boom")


class _OutageStore:
    """Store double whose every operation reports storage unavailability."""

    def create_revision(self, **kw):
        raise BoardStateStoreError("outage")

    def get_revision(self, board_id, revision):
        raise BoardStateStoreError("outage")

    def get_latest(self, board_id):
        raise BoardStateStoreError("outage")


@pytest.fixture(autouse=True)
def _restore_default_store():
    yield
    sbs.set_state_store(None)


_POS = {"x": 0.123, "y": 0.456, "width": 0.31, "height": 0.29, "z": 3, "rotation": -5}


def _wardrobe():
    return [
        {"id": i, "name": n, "category": c, "source": "wardrobe",
         "image_url": f"https://img/{i}.png"}
        for i, n, c in [
            ("top-1", "White Oxford Shirt", "Tops"),
            ("bottom-1", "Blue Jeans", "Bottoms"),
            ("bottom-2", "Grey Trousers", "Bottoms"),
            ("shoe-1", "White Sneakers", "Footwear"),
            ("shoe-2", "Black Heels", "Footwear"),
        ]
    ]


def _locked_top():
    return {
        "item_id": "top-1", "slot": "top", "role": "top", "source": "wardrobe",
        "image_url": "https://img/top-1.png", "position": copy.deepcopy(_POS),
    }


def _shuffle(board_id, revision, user_id="owner-1", **kw):
    return sbs.shuffle_board(
        board_id=board_id,
        revision=revision,
        locked_items=kw.pop("locked", [_locked_top()]),
        shuffle_slots=kw.pop("slots", ["bottom", "footwear"]),
        source_policy=kw.pop("source_policy", "inherit"),
        wardrobe=kw.pop("wardrobe", _wardrobe()),
        user_id=user_id,
        **kw,
    )


def _register(board_id, user_id="owner-1", policy="wardrobe", items=None):
    result = sbs.register_board(
        board_id=board_id, revision=1, scenario="build_outfit",
        source_policy=policy, items=items, user_id=user_id,
    )
    assert result["ok"], result
    return result


# ---------------------------------------------------------------------------
# Store-level: production AppwriteBoardStateStore over a fake proxy
# ---------------------------------------------------------------------------

def test_state_survives_store_reinstantiation():
    shared_docs = {}
    store_a = AppwriteBoardStateStore(proxy=_FakeAppwriteProxy(shared_docs))
    store_a.create_revision(
        user_id="owner-1", board_id="board-1", revision=1,
        payload={"scenario": "style_this", "source_policy": "style_asset",
                 "allow_wardrobe_fallback": False, "items": [_locked_top()]},
    )
    # New store instance over the same durable backend (restart / 2nd
    # Cloud Run instance).
    store_b = AppwriteBoardStateStore(proxy=_FakeAppwriteProxy(shared_docs))
    latest = store_b.get_latest("board-1")
    assert latest is not None
    assert latest["revision"] == 1
    assert latest["user_id"] == "owner-1"
    assert latest["payload"]["source_policy"] == "style_asset"
    item = latest["payload"]["items"][0]
    assert item["item_id"] == "top-1"
    assert item["image_url"] == "https://img/top-1.png"
    assert item["position"] == _POS


def test_duplicate_revision_id_raises_exists():
    store = AppwriteBoardStateStore(proxy=_FakeAppwriteProxy())
    store.create_revision(user_id="u", board_id="b", revision=2, payload={})
    with pytest.raises(BoardRevisionExistsError):
        store.create_revision(user_id="u", board_id="b", revision=2, payload={})


def test_outage_maps_to_store_error():
    store = AppwriteBoardStateStore(proxy=_OutageProxy())
    with pytest.raises(BoardStateStoreError):
        store.create_revision(user_id="u", board_id="b", revision=1, payload={})
    with pytest.raises(BoardStateStoreError):
        store.get_latest("b")


def test_deterministic_document_id_is_appwrite_safe():
    doc_id = revision_document_id("8b6c8f7e-1234-4abc-9def-aabbccddeeff", 12)
    assert doc_id == revision_document_id("8b6c8f7e-1234-4abc-9def-aabbccddeeff", 12)
    assert len(doc_id) <= 36
    assert all(c.isalnum() or c in "._-" for c in doc_id)


# ---------------------------------------------------------------------------
# Service-level: durability, ownership, concurrency, outage
# ---------------------------------------------------------------------------

def test_shuffle_state_survives_service_store_reinstantiation():
    backing = {}
    sbs.set_state_store(InMemoryBoardStateStore(backing))
    _register("board-d", policy="wardrobe")
    first = _shuffle("board-d", 1)
    assert first["success"] is True and first["revision"] == 2

    # Fresh store instance over the same backing = restart survival.
    sbs.set_state_store(InMemoryBoardStateStore(backing))
    state = sbs.get_board_state("board-d")
    assert state["revision"] == 2
    assert state["source_policy"] == "wardrobe"  # policy survives too
    second = _shuffle("board-d", 2)
    assert second["success"] is True and second["revision"] == 3
    locked = next(i for i in second["board_items"] if i["item_id"] == "top-1")
    assert locked["image_url"] == "https://img/top-1.png"
    assert locked["position"] == _POS


def test_board_owned_by_other_user_is_forbidden_without_contents():
    sbs.set_state_store(InMemoryBoardStateStore())
    _register("board-own", user_id="owner-1")
    result = _shuffle("board-own", 1, user_id="attacker")
    assert result["success"] is False
    assert result["error"]["code"] == "BOARD_FORBIDDEN"
    assert "board_items" not in result
    assert "items" not in result.get("error", {})


def test_same_board_id_cannot_be_claimed_by_another_user():
    sbs.set_state_store(InMemoryBoardStateStore())
    _register("board-claim", user_id="owner-1")
    stolen = sbs.register_board(
        board_id="board-claim", revision=1, scenario="build_outfit",
        source_policy="style_asset", user_id="attacker",
    )
    assert stolen["ok"] is False
    assert stolen["error"]["code"] == "BOARD_REGISTRATION_CONFLICT"
    # Original owner state untouched.
    assert sbs.get_board_state("board-claim")["source_policy"] == "wardrobe"


def test_registration_is_idempotent_for_same_owner_and_contract():
    sbs.set_state_store(InMemoryBoardStateStore())
    _register("board-idem", user_id="owner-1")
    again = sbs.register_board(
        board_id="board-idem", revision=1, scenario="build_outfit",
        source_policy="wardrobe", user_id="owner-1",
    )
    assert again["ok"] is True


def test_concurrent_shuffles_exactly_one_wins():
    sbs.set_state_store(InMemoryBoardStateStore())
    _register("board-race")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _shuffle("board-race", 1), range(2)))
    winners = [r for r in results if r.get("success")]
    losers = [r for r in results if not r.get("success")]
    assert len(winners) == 1, results
    assert len(losers) == 1, results
    assert losers[0]["error"]["code"] == "BOARD_REVISION_CONFLICT"
    assert losers[0]["error"]["current_revision"] == 2
    assert sbs.get_board_state("board-race")["revision"] == 2


def test_storage_outage_returns_unavailable_and_never_succeeds():
    sbs.set_state_store(_OutageStore())
    result = _shuffle("board-out", 1)
    assert result["success"] is False
    assert result["error"]["code"] == "BOARD_STATE_UNAVAILABLE"


def test_commit_outage_does_not_claim_success(monkeypatch):
    store = InMemoryBoardStateStore()
    sbs.set_state_store(store)
    _register("board-halfout")

    original = store.create_revision

    def failing_create(**kw):
        if kw.get("revision", 0) > 1:
            raise BoardStateStoreError("outage at commit")
        return original(**kw)

    monkeypatch.setattr(store, "create_revision", failing_create)
    result = _shuffle("board-halfout", 1)
    assert result["success"] is False
    assert result["error"]["code"] == "BOARD_STATE_UNAVAILABLE"
    # Revision was not advanced.
    assert sbs.get_board_state("board-halfout")["revision"] == 1


def test_production_default_store_is_appwrite_not_memory():
    sbs.set_state_store(None)
    assert isinstance(sbs._get_store(), AppwriteBoardStateStore)


def test_initial_generation_reports_shuffle_unavailable_on_outage():
    from routers.stylist import _builder_outfit

    sbs.set_state_store(_OutageStore())
    wardrobe = _wardrobe()
    outfit, _meta = _builder_outfit(dict(wardrobe[0]), wardrobe, [], None, mode="build_outfit")
    assert outfit["items"], "board itself is still returned"
    assert outfit["shuffle_available"] is False
    assert outfit["shuffle_state_error"]["code"] == "BOARD_STATE_UNAVAILABLE"


def test_initial_generation_marks_shuffle_available_when_registered():
    from routers.stylist import _builder_outfit

    sbs.set_state_store(InMemoryBoardStateStore())
    wardrobe = _wardrobe()
    outfit, _meta = _builder_outfit(dict(wardrobe[0]), wardrobe, [], None, mode="build_outfit")
    assert outfit["shuffle_available"] is True
    assert "shuffle_state_error" not in outfit
    state = sbs.get_board_state(outfit["board_id"])
    assert state["revision"] == 1
    assert state["source_policy"] == "wardrobe"
