"""Tests for scripts/create_upload_batch_collections.py.

No network: every Appwrite call goes through `requests.request`, which is
monkeypatched to a canned in-memory fake. Covers the two live-audit bugs
reported against real Appwrite:

1. float attributes were POSTed to the wrong route (or, once routed
   correctly, falsely reported as mismatched because Appwrite's GET response
   reports a float attribute's "type" as "double").
2. scalar attributes whose spec omits "array" were falsely reported as
   mismatched against Appwrite's real response, which always includes
   array=false for scalars.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from scripts import create_upload_batch_collections as mod


def _response(status_code: int, json_body: Any = None, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        status_code=status_code,
        text=text or (str(json_body) if json_body is not None else ""),
        json=lambda: json_body,
    )


class _FakeAppwrite:
    """Minimal in-memory Appwrite double: collections + their attributes."""

    def __init__(self) -> None:
        self.collections: Dict[str, Dict[str, Any]] = {}
        self.attributes: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.calls: List[tuple] = []

    def request(self, method: str, url: str, headers=None, json=None, timeout=None):
        self.calls.append((method, url))
        parts = url.split("/collections", 1)
        tail = parts[1] if len(parts) > 1 else ""
        segments = [s for s in tail.split("/") if s]

        if method == "GET" and len(segments) == 0:
            return _response(200, {"total": len(self.collections)})

        if len(segments) == 1:
            collection_id = segments[0]
            if method == "GET":
                if collection_id in self.collections:
                    return _response(200, self.collections[collection_id])
                return _response(404, {"message": "Collection with the requested ID could not be found."})
            if method == "POST":
                # POST .../collections (create collection) - url has no
                # collection segment in this branch; handled below instead.
                pass

        if method == "POST" and tail == "":
            collection_id = json["collectionId"]
            if collection_id in self.collections:
                return _response(409, {"message": "Collection already exists"})
            self.collections[collection_id] = {"$id": collection_id, **json}
            self.attributes[collection_id] = {}
            return _response(201, self.collections[collection_id])

        if len(segments) >= 2 and segments[1] == "attributes":
            collection_id = segments[0]
            if collection_id not in self.collections:
                return _response(404, {"message": "Collection with the requested ID could not be found."})
            if len(segments) == 2 and method == "GET":
                return _response(200, {"attributes": list(self.attributes[collection_id].values())})
            if len(segments) == 3 and method == "POST":
                route_type = segments[2]
                if route_type not in mod._APPWRITE_RESPONSE_TYPE and route_type not in {
                    "string", "integer", "datetime", "boolean"
                }:
                    return _response(404, {"message": "general_route_not_found"})
                key = json["key"]
                if key in self.attributes[collection_id]:
                    return _response(409, {"message": "Attribute already exists"})
                response_type = mod._APPWRITE_RESPONSE_TYPE.get(route_type, route_type)
                stored = {
                    "key": key,
                    "type": response_type,
                    "required": bool(json.get("required", False)),
                    "array": bool(json.get("array", False)),
                }
                if route_type == "string":
                    stored["size"] = json.get("size")
                self.attributes[collection_id][key] = stored
                return _response(201, stored)

        raise AssertionError(f"unhandled fake Appwrite call: {method} {url} json={json}")


@pytest.fixture
def fake_appwrite(monkeypatch):
    fake = _FakeAppwrite()
    monkeypatch.setattr(mod.requests, "request", fake.request)
    monkeypatch.setenv("APPWRITE_ENDPOINT", "https://example.test/v1")
    monkeypatch.setenv("APPWRITE_PROJECT_ID", "proj")
    monkeypatch.setenv("APPWRITE_DATABASE_ID", "db")
    monkeypatch.setenv("APPWRITE_API_KEY", "key")
    return fake


# ---------------------------------------------------------------------------
# 1. float spec maps to /attributes/float (and doesn't 404 like /attributes/double did)
# ---------------------------------------------------------------------------
def test_float_attribute_uses_float_route_not_double(fake_appwrite):
    mod._ensure_collection(mod.ITEMS_COLLECTION_ID, "upload_batch_items")
    mod._ensure_attributes(mod.ITEMS_COLLECTION_ID, (("float", {"key": "duplicate_confidence", "required": False}),))

    float_calls = [url for method, url in fake_appwrite.calls if method == "POST" and url.endswith("/attributes/float")]
    double_calls = [url for method, url in fake_appwrite.calls if method == "POST" and url.endswith("/attributes/double")]
    assert float_calls, "expected a POST to .../attributes/float"
    assert not double_calls, "must never POST to the non-existent .../attributes/double route"
    assert "duplicate_confidence" in fake_appwrite.attributes[mod.ITEMS_COLLECTION_ID]


# ---------------------------------------------------------------------------
# 2. scalar Appwrite array=false matches an unspecified/default scalar expectation
# ---------------------------------------------------------------------------
def test_scalar_attribute_without_array_key_audits_clean(fake_appwrite):
    mod._ensure_collection(mod.BATCHES_COLLECTION_ID, "upload_batches")
    # created_at/updated_at in BATCH_ATTRIBUTES omit "array" entirely, same as
    # the live schema that triggered the false-positive bug report.
    mod._ensure_attributes(mod.BATCHES_COLLECTION_ID, mod.BATCH_ATTRIBUTES)

    report = mod._audit_collection(mod.BATCHES_COLLECTION_ID, mod.BATCH_ATTRIBUTES)
    assert report["missing_attributes"] == []
    assert report["mismatched_attributes"] == [], report["mismatched_attributes"]


def test_float_attribute_reported_type_double_audits_clean_against_float_spec(fake_appwrite):
    mod._ensure_collection(mod.ITEMS_COLLECTION_ID, "upload_batch_items")
    mod._ensure_attributes(mod.ITEMS_COLLECTION_ID, mod.ITEM_ATTRIBUTES)

    report = mod._audit_collection(mod.ITEMS_COLLECTION_ID, mod.ITEM_ATTRIBUTES)
    assert report["missing_attributes"] == []
    assert report["mismatched_attributes"] == [], report["mismatched_attributes"]


# ---------------------------------------------------------------------------
# 3. a genuine array=true mismatch is still detected (fix must not weaken real checks)
# ---------------------------------------------------------------------------
def test_real_array_mismatch_is_still_detected(fake_appwrite):
    mod._ensure_collection(mod.BATCHES_COLLECTION_ID, "upload_batches")
    mod._ensure_attributes(
        mod.BATCHES_COLLECTION_ID,
        (("string", {"key": "tags", "size": 32, "required": False, "array": True}),),
    )

    # Spec now (incorrectly) expects a scalar where Appwrite actually has an array.
    bad_spec = (("string", {"key": "tags", "size": 32, "required": False, "array": False}),)
    report = mod._audit_collection(mod.BATCHES_COLLECTION_ID, bad_spec)
    assert report["mismatched_attributes"], "a real array mismatch must still be reported"
    assert "array" in report["mismatched_attributes"][0]


def test_real_type_mismatch_is_still_detected(fake_appwrite):
    mod._ensure_collection(mod.BATCHES_COLLECTION_ID, "upload_batches")
    mod._ensure_attributes(
        mod.BATCHES_COLLECTION_ID,
        (("integer", {"key": "total_items", "required": True, "array": False}),),
    )
    bad_spec = (("string", {"key": "total_items", "size": 32, "required": True, "array": False}),)
    report = mod._audit_collection(mod.BATCHES_COLLECTION_ID, bad_spec)
    assert report["mismatched_attributes"]
    assert "type" in report["mismatched_attributes"][0]


# ---------------------------------------------------------------------------
# 4. an existing attribute is skipped (verified, not recreated) during apply
# ---------------------------------------------------------------------------
def test_apply_skips_existing_attribute(fake_appwrite):
    mod._ensure_collection(mod.BATCHES_COLLECTION_ID, "upload_batches")
    mod._ensure_attributes(mod.BATCHES_COLLECTION_ID, mod.BATCH_ATTRIBUTES)
    calls_after_first_pass = len(fake_appwrite.calls)

    # Second pass: every attribute already exists - must only GET to verify,
    # never re-POST any attribute route.
    mod._ensure_attributes(mod.BATCHES_COLLECTION_ID, mod.BATCH_ATTRIBUTES)
    second_pass_calls = fake_appwrite.calls[calls_after_first_pass:]
    attribute_posts = [c for c in second_pass_calls if c[0] == "POST" and "/attributes/" in c[1]]
    assert attribute_posts == [], f"rerun must not re-POST existing attributes: {attribute_posts}"


# ---------------------------------------------------------------------------
# 5. a partially-created collection (exactly today's live state) resumes safely
# ---------------------------------------------------------------------------
def test_partially_created_collection_resumes_and_completes(fake_appwrite):
    # Simulate the exact reported live state: collection + most attributes
    # exist, duplicate_confidence/created_at/updated_at and the optional
    # latency fields are missing because the first apply run aborted on the
    # float-route 404.
    mod._ensure_collection(mod.ITEMS_COLLECTION_ID, "upload_batch_items")
    already_created = mod.ITEM_ATTRIBUTES[:-6]  # everything up to (not incl.) duplicate_confidence
    mod._ensure_attributes(mod.ITEMS_COLLECTION_ID, already_created)
    assert set(fake_appwrite.attributes[mod.ITEMS_COLLECTION_ID].keys()) == {a[1]["key"] for a in already_created}

    # Rerunning full apply must not blow up on the pre-existing ones and must
    # create exactly the six that were missing.
    mod._ensure_collection(mod.ITEMS_COLLECTION_ID, "upload_batch_items")
    mod._ensure_attributes(mod.ITEMS_COLLECTION_ID, mod.ITEM_ATTRIBUTES)

    final_keys = set(fake_appwrite.attributes[mod.ITEMS_COLLECTION_ID].keys())
    assert final_keys == {a[1]["key"] for a in mod.ITEM_ATTRIBUTES}

    # And it's safe to run a third time now that everything exists.
    mod._ensure_collection(mod.ITEMS_COLLECTION_ID, "upload_batch_items")
    mod._ensure_attributes(mod.ITEMS_COLLECTION_ID, mod.ITEM_ATTRIBUTES)
    report = mod._audit_collection(mod.ITEMS_COLLECTION_ID, mod.ITEM_ATTRIBUTES)
    assert report["missing_attributes"] == []
    assert report["mismatched_attributes"] == []


def test_apply_does_not_recreate_existing_collection(fake_appwrite):
    mod._ensure_collection(mod.BATCHES_COLLECTION_ID, "upload_batches")
    calls_after_first = len(fake_appwrite.calls)
    mod._ensure_collection(mod.BATCHES_COLLECTION_ID, "upload_batches")
    second_calls = fake_appwrite.calls[calls_after_first:]
    collection_posts = [c for c in second_calls if c[0] == "POST" and c[1].endswith("/collections")]
    assert collection_posts == [], "rerun must not re-POST an existing collection"


def test_audit_reports_ready_false_when_attributes_missing(fake_appwrite):
    mod._ensure_collection(mod.BATCHES_COLLECTION_ID, "upload_batches")
    mod._ensure_collection(mod.ITEMS_COLLECTION_ID, "upload_batch_items")
    # Only batches is fully populated; items is left empty (nothing created).
    mod._ensure_attributes(mod.BATCHES_COLLECTION_ID, mod.BATCH_ATTRIBUTES)

    assert mod.audit() is False


def test_audit_reports_ready_true_when_everything_matches(fake_appwrite):
    mod._ensure_collection(mod.BATCHES_COLLECTION_ID, "upload_batches")
    mod._ensure_attributes(mod.BATCHES_COLLECTION_ID, mod.BATCH_ATTRIBUTES)
    mod._ensure_collection(mod.ITEMS_COLLECTION_ID, "upload_batch_items")
    mod._ensure_attributes(mod.ITEMS_COLLECTION_ID, mod.ITEM_ATTRIBUTES)

    assert mod.audit() is True
