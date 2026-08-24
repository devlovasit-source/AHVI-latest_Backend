"""Idempotent migration: create the upload_batches / upload_batch_items
Appwrite collections required by services.upload_batch_orchestrator
(AHVI P0 sequential upload MVP, backend SHA 017b477).

Attribute contract is derived directly from
services/upload_batch_orchestrator.py's write sites (create_or_resume_batch,
_bump_batch_counter, claim_item, process_single_batch_item) - not from a
design doc. Re-check that module if this script and the orchestrator ever
drift.

Environment (no secrets in this file):
    APPWRITE_ENDPOINT / EXPO_PUBLIC_APPWRITE_ENDPOINT
    APPWRITE_PROJECT_ID / EXPO_PUBLIC_APPWRITE_PROJECT_ID
    APPWRITE_DATABASE_ID / EXPO_PUBLIC_APPWRITE_DATABASE_ID
    APPWRITE_API_KEY / APPWRITE_KEY

Usage:
    python scripts/create_upload_batch_collections.py            # audit only
    python scripts/create_upload_batch_collections.py --apply    # create/verify

Safe to re-run: every create returns success on 409 (already exists); with
--apply, existing attributes/indexes are verified against the expected
schema (mismatch raises loudly) rather than silently skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests


# Collection IDs are the literal strings the orchestrator uses
# (UploadBatchOrchestrator.batches_collection / .items_collection) - AppwriteProxy
# falls back to the resource name itself when no APPWRITE_COLLECTION_* override
# env var is set, so these are NOT configurable without also updating the
# orchestrator.
BATCHES_COLLECTION_ID = "upload_batches"
ITEMS_COLLECTION_ID = "upload_batch_items"

BATCH_ATTRIBUTES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("string", {"key": "user_id", "size": 128, "required": True, "array": False}),
    ("string", {"key": "client_batch_request_id", "size": 128, "required": True, "array": False}),
    ("string", {"key": "status", "size": 32, "required": True, "array": False}),
    ("integer", {"key": "total_items", "required": True, "array": False}),
    ("integer", {"key": "added_count", "required": False, "array": False, "default": 0}),
    ("integer", {"key": "needs_review_count", "required": False, "array": False, "default": 0}),
    ("integer", {"key": "rejected_count", "required": False, "array": False, "default": 0}),
    ("integer", {"key": "failed_count", "required": False, "array": False, "default": 0}),
    ("datetime", {"key": "created_at", "required": False}),
    ("datetime", {"key": "updated_at", "required": False}),
)

# Doc id is deterministic_appwrite_id(user_id, client_batch_request_id) - no
# lookup-by-attribute is needed for batches, so no indexes are required for
# the device gate.
BATCH_INDEXES: Tuple[Dict[str, Any], ...] = ()

ITEM_ATTRIBUTES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("string", {"key": "user_id", "size": 128, "required": True, "array": False}),
    ("string", {"key": "batch_id", "size": 128, "required": True, "array": False}),
    ("string", {"key": "client_upload_item_id", "size": 128, "required": True, "array": False}),
    ("string", {"key": "status", "size": 32, "required": True, "array": False}),
    ("integer", {"key": "attempt_count", "required": True, "array": False}),
    ("string", {"key": "wardrobe_item_id", "size": 128, "required": False}),
    ("string", {"key": "error_code", "size": 64, "required": False}),
    ("string", {"key": "matched_item_id", "size": 128, "required": False}),
    ("string", {"key": "duplicate_reason", "size": 64, "required": False}),
    ("float", {"key": "duplicate_confidence", "required": False}),
    ("datetime", {"key": "created_at", "required": False}),
    ("datetime", {"key": "updated_at", "required": False}),
)

# Doc id is deterministic_appwrite_id(user_id, client_upload_item_id) - the
# orchestrator never queries upload_batch_items by attribute, only by doc id,
# so no indexes are required for the device gate (Phase 1 explicitly says not
# to add optional indexes beyond what's required).
ITEM_INDEXES: Tuple[Dict[str, Any], ...] = ()


def _env(name: str) -> str:
    return str(os.getenv(name) or "").strip()


def _headers() -> Dict[str, str]:
    return {
        "X-Appwrite-Project": _env("APPWRITE_PROJECT_ID") or _env("EXPO_PUBLIC_APPWRITE_PROJECT_ID"),
        "X-Appwrite-Key": _env("APPWRITE_API_KEY") or _env("APPWRITE_KEY"),
        "Content-Type": "application/json",
    }


def _database_id() -> str:
    return _env("APPWRITE_DATABASE_ID") or _env("EXPO_PUBLIC_APPWRITE_DATABASE_ID")


def _base() -> str:
    endpoint = (_env("APPWRITE_ENDPOINT") or _env("EXPO_PUBLIC_APPWRITE_ENDPOINT")).rstrip("/")
    database_id = _database_id()
    headers = _headers()
    if not endpoint or not database_id or not headers["X-Appwrite-Project"] or not headers["X-Appwrite-Key"]:
        raise RuntimeError(
            "Missing Appwrite endpoint/project/database/api key configuration "
            "(APPWRITE_ENDPOINT, APPWRITE_PROJECT_ID, APPWRITE_DATABASE_ID, APPWRITE_API_KEY)."
        )
    return f"{endpoint}/databases/{database_id}/collections"


def _request(method: str, url: str, payload: Dict[str, Any] | None = None) -> requests.Response:
    return requests.request(method, url, headers=_headers(), json=payload, timeout=20)


def _ok(response: requests.Response, label: str) -> None:
    if response.status_code in {200, 201, 202}:
        print(f"created: {label}")
    elif response.status_code == 409:
        print(f"already exists: {label}")
    else:
        raise RuntimeError(f"{label} failed ({response.status_code}): {response.text}")


def _json(response: requests.Response, label: str) -> Dict[str, Any]:
    if response.status_code != 200:
        raise RuntimeError(f"{label} failed ({response.status_code}): {response.text}")
    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"{label} returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} returned an invalid payload")
    return data


# The Appwrite REST route to CREATE an attribute is not always the same
# string as the "type" field Appwrite reports back on GET - float is the
# known case (POST .../attributes/float, but the attribute object it
# returns has "type": "double"). Keyed by the route name used in
# BATCH_ATTRIBUTES/ITEM_ATTRIBUTES; anything not listed here reports back
# under its own name.
_APPWRITE_RESPONSE_TYPE: Dict[str, str] = {"float": "double"}


def _mismatches(actual: Dict[str, Any], expected: Dict[str, Any], fields: Iterable[str]) -> list[str]:
    mismatches = []
    for field in fields:
        if field == "array":
            # Appwrite always reports a real bool for "array" on every
            # scalar attribute; an attribute spec that never mentions
            # "array" means "ordinary scalar", i.e. expected False - NOT
            # "expected unset", which would falsely flag every scalar
            # attribute Appwrite reports back as array=false.
            expected_value = bool(expected.get("array", False))
            actual_value = bool(actual.get("array"))
        else:
            expected_value = expected.get(field)
            actual_value = actual.get(field)
        if actual_value != expected_value:
            mismatches.append(f"{field}={actual_value!r} (expected {expected_value!r})")
    return mismatches


def _diff_attribute(actual: Dict[str, Any], attr_type: str, expected: Dict[str, Any]) -> list[str]:
    fields = ["type", "required", "array"]
    if attr_type == "string":
        fields.append("size")
    response_type = _APPWRITE_RESPONSE_TYPE.get(attr_type, attr_type)
    return _mismatches(actual, {**expected, "type": response_type}, fields)


def _collection_status(collection_id: str) -> str:
    """Returns 'missing', 'ok', or raises on an unexpected response."""
    response = _request("GET", f"{_base()}/{collection_id}")
    if response.status_code == 404:
        return "missing"
    if response.status_code == 200:
        return "ok"
    raise RuntimeError(f"collection {collection_id} lookup failed ({response.status_code}): {response.text}")


def _audit_collection(collection_id: str, attributes: Tuple[Tuple[str, Dict[str, Any]], ...]) -> Dict[str, Any]:
    status = _collection_status(collection_id)
    report: Dict[str, Any] = {"collection": collection_id, "collection_exists": status == "ok"}
    if status != "ok":
        report["missing_attributes"] = [a[1]["key"] for a in attributes]
        return report

    rows = _json(_request("GET", f"{_base()}/{collection_id}/attributes"), "attribute listing").get("attributes", [])
    existing = {str(row.get("key") or ""): row for row in rows if isinstance(row, dict)}
    missing, mismatched = [], []
    for attr_type, expected in attributes:
        key = expected["key"]
        actual = existing.get(key)
        if actual is None:
            missing.append(key)
            continue
        bad = _diff_attribute(actual, attr_type, expected)
        if bad:
            mismatched.append(f"{key}: {', '.join(bad)}")
    report["missing_attributes"] = missing
    report["mismatched_attributes"] = mismatched
    return report


def _ensure_collection(collection_id: str, name: str) -> None:
    status = _collection_status(collection_id)
    if status == "ok":
        print(f"verified: collection {collection_id}")
        return
    payload = {
        "collectionId": collection_id,
        "name": name,
        "permissions": [],
        "documentSecurity": False,
        "enabled": True,
    }
    _ok(_request("POST", _base(), payload), f"collection {collection_id}")


def _ensure_attributes(collection_id: str, attributes: Tuple[Tuple[str, Dict[str, Any]], ...]) -> None:
    url = f"{_base()}/{collection_id}/attributes"
    rows = _json(_request("GET", url), "attribute listing").get("attributes", [])
    existing = {str(row.get("key") or ""): row for row in rows if isinstance(row, dict)}
    for attr_type, expected in attributes:
        key = expected["key"]
        actual = existing.get(key)
        if actual is None:
            attr_url = f"{_base()}/{collection_id}/attributes/{attr_type}"
            _ok(_request("POST", attr_url, expected), f"attribute {collection_id}.{key}")
            continue
        mismatches = _diff_attribute(actual, attr_type, expected)
        if mismatches:
            raise RuntimeError(f"attribute {collection_id}.{key} schema mismatch: {', '.join(mismatches)}")
        print(f"verified: attribute {collection_id}.{key}")


def audit() -> bool:
    """Read-only Phase-1 check. Returns True iff both collections are fully ready."""
    print(f"target database={_database_id()}")
    ready = True
    for collection_id, attrs in ((BATCHES_COLLECTION_ID, BATCH_ATTRIBUTES), (ITEMS_COLLECTION_ID, ITEM_ATTRIBUTES)):
        report = _audit_collection(collection_id, attrs)
        print(report)
        if not report["collection_exists"] or report.get("missing_attributes") or report.get("mismatched_attributes"):
            ready = False
    print("APPWRITE_SCHEMA_READY=" + ("YES" if ready else "NO"))
    return ready


def apply() -> None:
    _ensure_collection(BATCHES_COLLECTION_ID, "upload_batches")
    _ensure_attributes(BATCHES_COLLECTION_ID, BATCH_ATTRIBUTES)
    _ensure_collection(ITEMS_COLLECTION_ID, "upload_batch_items")
    _ensure_attributes(ITEMS_COLLECTION_ID, ITEM_ATTRIBUTES)
    print("upload_batches / upload_batch_items schema apply complete")


if __name__ == "__main__":
    if "--apply" in sys.argv:
        apply()
    else:
        audit()
