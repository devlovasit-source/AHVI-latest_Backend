"""Idempotent Appwrite migration for durable Style feedback.

Usage:
    python scripts/create_style_feedback_events_collection.py

Optional collection override:
    APPWRITE_COLLECTION_STYLE_FEEDBACK_EVENTS (default style_feedback_events)
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict

import requests

COLLECTION_ID = (os.getenv("APPWRITE_COLLECTION_STYLE_FEEDBACK_EVENTS")
                 or "style_feedback_events").strip()


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
    database = _env("APPWRITE_DATABASE_ID") or _env("EXPO_PUBLIC_APPWRITE_DATABASE_ID")
    if not endpoint or not database or not all(_headers().values()):
        raise RuntimeError("Missing Appwrite endpoint/project/database/api key configuration.")
    return f"{endpoint}/databases/{database}/collections"


def _request(method: str, url: str, payload: Dict[str, Any]) -> requests.Response:
    return requests.request(method, url, headers=_headers(), json=payload, timeout=20)


def _ok(response: requests.Response, label: str) -> None:
    if response.status_code in {200, 201, 202}:
        print(f"created: {label}")
    elif response.status_code == 409:
        print(f"already exists: {label}")
    else:
        raise RuntimeError(f"{label} failed ({response.status_code}): {response.text}")


def _attribute(key: str, size: int, required: bool) -> None:
    url = f"{_base()}/{COLLECTION_ID}/attributes/string"
    _ok(_request("POST", url, {"key": key, "size": size, "required": required}), f"attribute {key}")


def _index(key: str, attributes: list[str], orders: list[str]) -> None:
    url = f"{_base()}/{COLLECTION_ID}/indexes"
    payload = {"key": key, "type": "key", "attributes": attributes, "orders": orders}
    _ok(_request("POST", url, payload), f"index {key}")


def main() -> None:
    _ok(_request("POST", _base(), {
        "collectionId": COLLECTION_ID, "name": "style_feedback_events",
        "permissions": [], "documentSecurity": False, "enabled": True,
    }), f"collection {COLLECTION_ID}")
    for key, size, required in (
        ("userId", 128, True), ("eventId", 128, True), ("action", 16, True),
        ("boardId", 128, False), ("itemIds", 4096, False),
        ("sourcePolicy", 64, False), ("occasion", 80, False),
        ("payload", 12000, False), ("createdAtISO", 40, True),
    ):
        _attribute(key, size, required)
    time.sleep(2)
    _index("idx_feedback_user", ["userId"], ["ASC"])
    _index("idx_feedback_created", ["createdAtISO"], ["DESC"])
    _index("idx_feedback_user_created", ["userId", "createdAtISO"], ["ASC", "DESC"])
    print(f"style_feedback_events migration complete (collection={COLLECTION_ID})")


if __name__ == "__main__":
    main()
