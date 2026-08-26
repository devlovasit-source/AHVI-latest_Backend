"""Audit/apply schema definition for the beta_ops_events Appwrite collection.

This script is intentionally non-destructive and does not run on import. Use
``--apply`` only as an explicit future operator action; Phase 1 does not run it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLLECTION_ID = "beta_ops_events"

ATTRIBUTE_DEFINITIONS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("string", {"key": "event_type", "size": 64, "required": True, "array": False}),
    ("string", {"key": "user_id", "size": 128, "required": True, "array": False}),
    ("datetime", {"key": "occurred_at", "required": True}),
    ("string", {"key": "status", "size": 32, "required": False, "array": False}),
    ("string", {"key": "request_id", "size": 128, "required": False, "array": False}),
    ("string", {"key": "operation_id", "size": 128, "required": False, "array": False}),
    ("integer", {"key": "attempt", "required": False, "array": False}),
    ("string", {"key": "provider", "size": 32, "required": False, "array": False}),
    ("string", {"key": "model", "size": 96, "required": False, "array": False}),
    ("string", {"key": "usecase", "size": 64, "required": False, "array": False}),
    ("integer", {"key": "duration_ms", "required": False, "array": False}),
    ("integer", {"key": "input_tokens", "required": False, "array": False}),
    ("integer", {"key": "output_tokens", "required": False, "array": False}),
    ("integer", {"key": "cached_tokens", "required": False, "array": False}),
    ("string", {"key": "error_code", "size": 64, "required": False, "array": False}),
    ("string", {"key": "metadata_json", "size": 2048, "required": False, "array": False}),
)

INDEX_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
    {"key": "idx_event_type_occurred_at", "type": "key", "attributes": ["event_type", "occurred_at"], "orders": ["ASC", "ASC"]},
    {"key": "idx_user_id_occurred_at", "type": "key", "attributes": ["user_id", "occurred_at"], "orders": ["ASC", "ASC"]},
)


def schema_definition() -> Dict[str, Any]:
    return {
        "collection": COLLECTION_ID,
        "attributes": ATTRIBUTE_DEFINITIONS,
        "indexes": INDEX_DEFINITIONS,
    }


def _env(name: str) -> str:
    return str(os.getenv(name) or "").strip()


def _base() -> str:
    endpoint = (_env("APPWRITE_ENDPOINT") or _env("EXPO_PUBLIC_APPWRITE_ENDPOINT")).rstrip("/")
    database = _env("APPWRITE_DATABASE_ID") or _env("EXPO_PUBLIC_APPWRITE_DATABASE_ID")
    if not endpoint or not database:
        raise RuntimeError("Appwrite endpoint and database are required")
    return f"{endpoint}/databases/{database}/collections"


def _headers() -> Dict[str, str]:
    return {
        "X-Appwrite-Project": _env("APPWRITE_PROJECT_ID") or _env("EXPO_PUBLIC_APPWRITE_PROJECT_ID"),
        "X-Appwrite-Key": _env("APPWRITE_API_KEY") or _env("APPWRITE_KEY"),
        "Content-Type": "application/json",
    }


def _request(method: str, url: str, payload: Dict[str, Any] | None = None) -> Any:
    import requests

    return requests.request(method, url, headers=_headers(), json=payload, timeout=20)


def _apply_schema() -> None:
    base = _base()
    collection_url = f"{base}/{COLLECTION_ID}"
    collection = _request("GET", collection_url)
    if collection.status_code == 404:
        created = _request(
            "POST",
            base,
            {
                "collectionId": COLLECTION_ID,
                "name": COLLECTION_ID,
                "permissions": [],
                "documentSecurity": False,
                "enabled": True,
            },
        )
        if created.status_code not in {200, 201, 202, 409}:
            raise RuntimeError(f"collection creation failed ({created.status_code})")
    elif collection.status_code != 200:
        raise RuntimeError(f"collection lookup failed ({collection.status_code})")

    for attribute_type, definition in ATTRIBUTE_DEFINITIONS:
        response = _request(
            "POST",
            f"{collection_url}/attributes/{attribute_type}",
            dict(definition),
        )
        if response.status_code not in {200, 201, 202, 409}:
            raise RuntimeError(f"attribute {definition['key']} failed ({response.status_code})")

    for definition in INDEX_DEFINITIONS:
        response = _request("POST", f"{collection_url}/indexes", definition)
        if response.status_code not in {200, 201, 202, 409}:
            raise RuntimeError(f"index {definition['key']} failed ({response.status_code})")


def apply_schema() -> None:
    """Apply is explicit only; never called automatically."""
    _apply_schema()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit beta_ops_events schema")
    parser.add_argument("--apply", action="store_true", help="disabled during Phase 1")
    args = parser.parse_args()
    if args.apply:
        apply_schema()
    print(f"collection={COLLECTION_ID}")
    print(f"attributes={len(ATTRIBUTE_DEFINITIONS)}")
    print(f"indexes={len(INDEX_DEFINITIONS)}")
    print("mode=audit-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
