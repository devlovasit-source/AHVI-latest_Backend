"""Idempotent migration: create the style_board_states Appwrite collection.

Holds one IMMUTABLE document per (board_id, revision) for durable, atomic
style-board shuffle state (see services/style_board_state_store.py). The
deterministic document ID (sha1 of "board_id|revision", 36 chars) makes
document creation the atomic revision claim.

Environment (no secrets in this file):
    APPWRITE_ENDPOINT / EXPO_PUBLIC_APPWRITE_ENDPOINT
    APPWRITE_PROJECT_ID / EXPO_PUBLIC_APPWRITE_PROJECT_ID
    APPWRITE_DATABASE_ID / EXPO_PUBLIC_APPWRITE_DATABASE_ID
    APPWRITE_API_KEY / APPWRITE_KEY
    APPWRITE_COLLECTION_STYLE_BOARD_STATES   (optional; default
                                              "style_board_states")

Usage:
    python scripts/create_style_board_states_collection.py

Safe to re-run: every create returns success on 409 (already exists).
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


COLLECTION_ID = (
    os.getenv("APPWRITE_COLLECTION_STYLE_BOARD_STATES") or "style_board_states"
).strip()

ATTRIBUTE_DEFINITIONS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("string", {"key": "userId", "size": 128, "required": False, "array": False, "default": None}),
    ("string", {"key": "boardId", "size": 64, "required": True, "array": False}),
    ("integer", {"key": "revision", "required": True, "array": False}),
    ("string", {"key": "payload", "size": 1000000, "required": True, "array": False}),
    ("string", {"key": "createdAtISO", "size": 40, "required": False, "array": False, "default": None}),
)

INDEX_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "idx_board_revision",
        "type": "unique",
        "attributes": ["boardId", "revision"],
        "orders": ["ASC", "DESC"],
    },
    {"key": "idx_board", "type": "key", "attributes": ["boardId"], "orders": ["ASC"]},
    {"key": "idx_user", "type": "key", "attributes": ["userId"], "orders": ["ASC"]},
    {"key": "idx_revision", "type": "key", "attributes": ["revision"], "orders": ["DESC"]},
)


def _env(name: str) -> str:
    return str(os.getenv(name) or "").strip()


def _headers() -> Dict[str, str]:
    return {
        "X-Appwrite-Project": _env("APPWRITE_PROJECT_ID") or _env("EXPO_PUBLIC_APPWRITE_PROJECT_ID"),
        "X-Appwrite-Key": _env("APPWRITE_API_KEY") or _env("APPWRITE_KEY"),
        "Content-Type": "application/json",
    }


def _base() -> str:
    endpoint = (_env("APPWRITE_ENDPOINT") or _env("EXPO_PUBLIC_APPWRITE_ENDPOINT")).rstrip("/")
    database_id = _env("APPWRITE_DATABASE_ID") or _env("EXPO_PUBLIC_APPWRITE_DATABASE_ID")
    headers = _headers()
    if not endpoint or not database_id or not headers["X-Appwrite-Project"] or not headers["X-Appwrite-Key"]:
        raise RuntimeError("Missing Appwrite endpoint/project/database/api key configuration.")
    return f"{endpoint}/databases/{database_id}/collections"


def _database_id() -> str:
    return _env("APPWRITE_DATABASE_ID") or _env("EXPO_PUBLIC_APPWRITE_DATABASE_ID")


def _request(method: str, url: str, payload: Dict[str, Any] | None = None) -> requests.Response:
    return requests.request(method, url, headers=_headers(), json=payload, timeout=20)


def _ok(response: requests.Response, label: str) -> None:
    if response.status_code in {200, 201, 202}:
        print(f"created: {label}")
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


def _ensure_collection() -> None:
    url = f"{_base()}/{COLLECTION_ID}"
    existing = _request("GET", url)
    if existing.status_code == 200:
        collection = _json(existing, f"collection {COLLECTION_ID}")
        if bool(collection.get("documentSecurity")):
            raise RuntimeError(
                f"collection {COLLECTION_ID} schema mismatch: documentSecurity must be false"
            )
        if list(collection.get("permissions") or []) != []:
            raise RuntimeError(
                f"collection {COLLECTION_ID} schema mismatch: permissions must be empty"
            )
        print(f"verified: collection {COLLECTION_ID}")
        return
    if existing.status_code != 404:
        raise RuntimeError(
            f"collection {COLLECTION_ID} lookup failed ({existing.status_code}): {existing.text}"
        )
    payload = {
        "collectionId": COLLECTION_ID,
        "name": "style_board_states",
        "permissions": [],
        "documentSecurity": False,
        "enabled": True,
    }
    _ok(_request("POST", _base(), payload), f"collection {COLLECTION_ID}")


def _attr(url_suffix: str, payload: Dict[str, Any], label: str) -> None:
    url = f"{_base()}/{COLLECTION_ID}/attributes/{url_suffix}"
    _ok(_request("POST", url, payload), f"attribute {label}")


def _mismatches(actual: Dict[str, Any], expected: Dict[str, Any], fields: Iterable[str]) -> list[str]:
    mismatches = []
    for field in fields:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if field == "array":
            actual_value = bool(actual_value)
        if actual_value != expected_value:
            mismatches.append(f"{field}={actual_value!r} (expected {expected_value!r})")
    if "default" in expected and actual.get("default") != expected.get("default"):
        mismatches.append(
            f"default={actual.get('default')!r} (expected {expected.get('default')!r})"
        )
    return mismatches


def _ensure_attributes() -> None:
    url = f"{_base()}/{COLLECTION_ID}/attributes"
    rows = _json(_request("GET", url), "attribute listing").get("attributes", [])
    existing = {str(row.get("key") or ""): row for row in rows if isinstance(row, dict)}
    for attr_type, expected in ATTRIBUTE_DEFINITIONS:
        key = expected["key"]
        actual = existing.get(key)
        if actual is None:
            _attr(attr_type, expected, key)
            continue
        expected_with_type = {**expected, "type": attr_type}
        fields = ["type", "required", "array"]
        if attr_type == "string":
            fields.append("size")
        mismatches = _mismatches(actual, expected_with_type, fields)
        if mismatches:
            raise RuntimeError(f"attribute {key} schema mismatch: {', '.join(mismatches)}")
        print(f"verified: attribute {key}")


def _index(payload: Dict[str, Any], label: str) -> None:
    url = f"{_base()}/{COLLECTION_ID}/indexes"
    _ok(_request("POST", url, payload), f"index {label}")


def _ensure_indexes() -> None:
    url = f"{_base()}/{COLLECTION_ID}/indexes"
    rows = _json(_request("GET", url), "index listing").get("indexes", [])
    existing = {str(row.get("key") or ""): row for row in rows if isinstance(row, dict)}
    for expected in INDEX_DEFINITIONS:
        key = expected["key"]
        actual = existing.get(key)
        if actual is None:
            _index(expected, key)
            continue
        mismatches = _mismatches(actual, expected, ("type", "attributes", "orders"))
        if mismatches:
            raise RuntimeError(f"index {key} schema mismatch: {', '.join(mismatches)}")
        print(f"verified: index {key}")


def main() -> None:
    print(f"target database={_database_id()} collection={COLLECTION_ID}")
    _ensure_collection()
    _ensure_attributes()
    _ensure_indexes()
    print(
        "style_board_states migration complete "
        f"(database={_database_id()} collection={COLLECTION_ID})"
    )


if __name__ == "__main__":
    main()
