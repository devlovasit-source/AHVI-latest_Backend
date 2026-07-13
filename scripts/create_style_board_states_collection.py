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
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests


COLLECTION_ID = (
    os.getenv("APPWRITE_COLLECTION_STYLE_BOARD_STATES") or "style_board_states"
).strip()


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


def _request(method: str, url: str, payload: Dict[str, Any] | None = None) -> requests.Response:
    return requests.request(method, url, headers=_headers(), json=payload, timeout=20)


def _ok(response: requests.Response, label: str) -> None:
    if response.status_code in {200, 201, 202}:
        print(f"created: {label}")
    elif response.status_code == 409:
        print(f"already exists: {label}")
    else:
        raise RuntimeError(f"{label} failed ({response.status_code}): {response.text}")


def _create_collection() -> None:
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


def _create_attributes() -> None:
    _attr("string", {"key": "userId", "size": 128, "required": False}, "userId")
    _attr("string", {"key": "boardId", "size": 64, "required": True}, "boardId")
    _attr("integer", {"key": "revision", "required": True}, "revision")
    # Full shuffle payload (scenario, source_policy, allow_wardrobe_fallback,
    # occasion, style_direction, board items with positions,
    # previous_revision) serialized as compact JSON.
    _attr("string", {"key": "payload", "size": 1000000, "required": True}, "payload")
    _attr("string", {"key": "createdAtISO", "size": 40, "required": False}, "createdAtISO")


def _index(payload: Dict[str, Any], label: str) -> None:
    url = f"{_base()}/{COLLECTION_ID}/indexes"
    _ok(_request("POST", url, payload), f"index {label}")


def _create_indexes() -> None:
    # Attributes must be "available" before indexes can be created.
    time.sleep(2)
    _index(
        {
            "key": "idx_board_revision",
            "type": "unique",
            "attributes": ["boardId", "revision"],
            "orders": ["ASC", "DESC"],
        },
        "idx_board_revision (unique boardId+revision)",
    )
    _index(
        {"key": "idx_board", "type": "key", "attributes": ["boardId"], "orders": ["ASC"]},
        "idx_board",
    )
    _index(
        {"key": "idx_user", "type": "key", "attributes": ["userId"], "orders": ["ASC"]},
        "idx_user",
    )
    _index(
        {"key": "idx_revision", "type": "key", "attributes": ["revision"], "orders": ["DESC"]},
        "idx_revision",
    )


def main() -> None:
    _create_collection()
    _create_attributes()
    _create_indexes()
    print(f"style_board_states migration complete (collection={COLLECTION_ID})")


if __name__ == "__main__":
    main()
