"""Idempotent migration: create/verify the wear_events Appwrite collection.

Attribute contract is derived directly from services.wear_event_service's
write site (record_wear) - not from a design doc. Re-check that module if
this script and the service ever drift.

Document id is deterministic_appwrite_id(user_id, "wear", item_id,
local_date) (see wear_event_service.deterministic_appwrite_id), so lookup by
id already gives per-user-per-item-per-day idempotency without an index.
get_wear_history queries by userId (equality) and sorts client-side, so a
userId index is required for that to stay a real index lookup rather than a
full collection scan as data grows.

Environment (no secrets in this file):
    APPWRITE_ENDPOINT / EXPO_PUBLIC_APPWRITE_ENDPOINT
    APPWRITE_PROJECT_ID / EXPO_PUBLIC_APPWRITE_PROJECT_ID
    APPWRITE_DATABASE_ID / EXPO_PUBLIC_APPWRITE_DATABASE_ID
    APPWRITE_API_KEY / APPWRITE_KEY
    APPWRITE_COLLECTION_WEAR_EVENTS   (optional; default "wear_events")

Usage:
    python scripts/create_wear_events_collection.py            # audit only
    python scripts/create_wear_events_collection.py --apply    # create/verify
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests


COLLECTION_ID = (
    os.getenv("APPWRITE_COLLECTION_WEAR_EVENTS")
    or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_WEAR_EVENTS")
    or "wear_events"
).strip()

ATTRIBUTE_DEFINITIONS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("string", {"key": "userId", "size": 128, "required": True, "array": False}),
    ("string", {"key": "itemId", "size": 128, "required": True, "array": False}),
    ("string", {"key": "localDate", "size": 10, "required": True, "array": False}),
    ("string", {"key": "occurredAtISO", "size": 40, "required": True, "array": False}),
    ("string", {"key": "source", "size": 32, "required": True, "array": False}),
    ("string", {"key": "entityType", "size": 32, "required": True, "array": False}),
    ("string", {"key": "entityId", "size": 128, "required": True, "array": False}),
    ("string", {"key": "status", "size": 16, "required": True, "array": False}),
    ("string", {"key": "idempotencyKey", "size": 256, "required": True, "array": False}),
    ("string", {"key": "createdAtISO", "size": 40, "required": True, "array": False}),
    ("string", {"key": "revokedAtISO", "size": 40, "required": False, "array": False}),
    ("string", {"key": "boardId", "size": 64, "required": False, "array": False}),
    ("string", {"key": "occasion", "size": 64, "required": False, "array": False}),
)

INDEX_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
    {"key": "idx_user", "type": "key", "attributes": ["userId"], "orders": ["ASC"]},
    {"key": "idx_item", "type": "key", "attributes": ["itemId"], "orders": ["ASC"]},
    {"key": "idx_user_item", "type": "key", "attributes": ["userId", "itemId"], "orders": ["ASC", "ASC"]},
)


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
    if response.status_code in {200, 201, 202, 409}:
        print(f"ready: {label}")
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


def _collection_status() -> str:
    response = _request("GET", f"{_base()}/{COLLECTION_ID}")
    if response.status_code == 404:
        return "missing"
    if response.status_code == 200:
        return "ok"
    raise RuntimeError(f"collection {COLLECTION_ID} lookup failed ({response.status_code}): {response.text}")


def _mismatches(actual: Dict[str, Any], expected: Dict[str, Any], fields) -> list:
    mismatches = []
    for field in fields:
        expected_value = bool(expected.get("array", False)) if field == "array" else expected.get(field)
        actual_value = bool(actual.get("array")) if field == "array" else actual.get(field)
        if actual_value != expected_value:
            mismatches.append(f"{field}={actual_value!r} (expected {expected_value!r})")
    return mismatches


def _diff_attribute(actual: Dict[str, Any], attr_type: str, expected: Dict[str, Any]) -> list:
    fields = ["type", "required", "array"]
    if attr_type == "string":
        fields.append("size")
    return _mismatches(actual, {**expected, "type": attr_type}, fields)


def audit() -> bool:
    """Read-only check against the real environment. Returns True iff ready."""
    print(f"target database={_database_id()} collection={COLLECTION_ID}")
    status = _collection_status()
    ready = status == "ok"
    print(f"collection_exists={ready}")
    if not ready:
        print("missing_attributes=" + ",".join(a[1]["key"] for a in ATTRIBUTE_DEFINITIONS))
        print("APPWRITE_SCHEMA_READY=NO")
        return False

    rows = _json(_request("GET", f"{_base()}/{COLLECTION_ID}/attributes"), "attribute listing").get("attributes", [])
    existing = {str(row.get("key") or ""): row for row in rows if isinstance(row, dict)}
    missing, mismatched = [], []
    for attr_type, expected in ATTRIBUTE_DEFINITIONS:
        key = expected["key"]
        actual = existing.get(key)
        if actual is None:
            missing.append(key)
            continue
        bad = _diff_attribute(actual, attr_type, expected)
        if bad:
            mismatched.append(f"{key}: {', '.join(bad)}")
    if missing:
        print("missing_attributes=" + ",".join(missing))
        ready = False
    if mismatched:
        print("mismatched_attributes=" + "; ".join(mismatched))
        ready = False

    idx_rows = _json(_request("GET", f"{_base()}/{COLLECTION_ID}/indexes"), "index listing").get("indexes", [])
    existing_idx = {str(row.get("key") or "") for row in idx_rows if isinstance(row, dict)}
    missing_idx = [i["key"] for i in INDEX_DEFINITIONS if i["key"] not in existing_idx]
    if missing_idx:
        print("missing_indexes=" + ",".join(missing_idx))
        ready = False

    print("APPWRITE_SCHEMA_READY=" + ("YES" if ready else "NO"))
    return ready


def apply() -> None:
    status = _collection_status()
    if status != "ok":
        payload = {
            "collectionId": COLLECTION_ID,
            "name": "wear_events",
            "permissions": [],
            "documentSecurity": False,
            "enabled": True,
        }
        _ok(_request("POST", _base(), payload), f"collection {COLLECTION_ID}")
    else:
        print(f"verified: collection {COLLECTION_ID}")

    attr_url = f"{_base()}/{COLLECTION_ID}/attributes"
    rows = _json(_request("GET", attr_url), "attribute listing").get("attributes", [])
    existing = {str(row.get("key") or ""): row for row in rows if isinstance(row, dict)}
    for attr_type, expected in ATTRIBUTE_DEFINITIONS:
        key = expected["key"]
        actual = existing.get(key)
        if actual is None:
            _ok(_request("POST", f"{attr_url}/{attr_type}", expected), f"attribute {key}")
            continue
        mismatches = _diff_attribute(actual, attr_type, expected)
        if mismatches:
            raise RuntimeError(f"attribute {key} schema mismatch: {', '.join(mismatches)}")
        print(f"verified: attribute {key}")

    idx_url = f"{_base()}/{COLLECTION_ID}/indexes"
    idx_rows = _json(_request("GET", idx_url), "index listing").get("indexes", [])
    existing_idx = {str(row.get("key") or "") for row in idx_rows if isinstance(row, dict)}
    for expected in INDEX_DEFINITIONS:
        key = expected["key"]
        if key in existing_idx:
            print(f"verified: index {key}")
            continue
        _ok(_request("POST", idx_url, expected), f"index {key}")

    print(f"wear_events schema apply complete (database={_database_id()} collection={COLLECTION_ID})")


if __name__ == "__main__":
    if "--apply" in sys.argv:
        apply()
    else:
        audit()
