from __future__ import annotations

import argparse
import json
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
    os.getenv("APPWRITE_COLLECTION_STYLE_ASSETS")
    or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_STYLE_ASSETS")
    or "style_assets"
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
    if not endpoint or not database_id or not _headers()["X-Appwrite-Project"] or not _headers()["X-Appwrite-Key"]:
        raise RuntimeError("Missing Appwrite endpoint/project/database/api key configuration.")
    return f"{endpoint}/databases/{database_id}/collections"


def _request(method: str, url: str, payload: Dict[str, Any] | None = None) -> requests.Response:
    return requests.request(method, url, headers=_headers(), json=payload, timeout=20)


def _attribute_url(kind: str) -> str:
    return f"{_base()}/{COLLECTION_ID}/attributes/{kind}"


def _create_string(key: str, size: int = 255, *, required: bool = False) -> str:
    payload = {"key": key, "size": size, "required": required, "array": False}
    response = _request("POST", _attribute_url("string"), payload)
    if response.status_code in {200, 201, 202}:
        time.sleep(0.15)
        return "created"
    if response.status_code == 409:
        return "exists"
    raise RuntimeError(f"attribute {key} failed {response.status_code}: {response.text}")


def _list_attributes() -> Dict[str, Any]:
    response = _request("GET", f"{_base()}/{COLLECTION_ID}/attributes")
    if response.status_code != 200:
        raise RuntimeError(f"attribute list failed {response.status_code}: {response.text}")
    data = response.json()
    attrs = data.get("attributes") if isinstance(data, dict) else []
    return {
        str(attr.get("key") or ""): attr
        for attr in attrs
        if isinstance(attr, dict) and str(attr.get("key") or "").strip()
    }


def migrate(*, dry_run: bool = True) -> Dict[str, str]:
    desired = {
        "board_image_url": 2048,
        "board_r2_key": 512,
        "cutout_status": 64,
        "catalog_image_url": 2048,
    }
    existing = _list_attributes()
    results: Dict[str, str] = {}
    for key, size in desired.items():
        if key in existing:
            results[key] = "exists"
        elif dry_run:
            results[key] = "missing"
        else:
            results[key] = _create_string(key, size)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Add board cutout fields to style_assets.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect only; no writes. Default.")
    parser.add_argument("--apply", action="store_true", help="Create missing attributes.")
    args = parser.parse_args()
    dry_run = not bool(args.apply)
    results = migrate(dry_run=dry_run)
    print(
        json.dumps(
            {
                "success": True,
                "dry_run": dry_run,
                "collection": COLLECTION_ID,
                "attributes": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
