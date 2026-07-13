from __future__ import annotations

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
    response = requests.request(method, url, headers=_headers(), json=payload, timeout=20)
    return response


def _create_collection() -> None:
    payload = {
        "collectionId": COLLECTION_ID,
        "name": "style_assets",
        "permissions": [],
        "documentSecurity": False,
        "enabled": True,
    }
    response = _request("POST", _base(), payload)
    if response.status_code in {200, 201, 202, 409}:
        print(f"collection ready: {COLLECTION_ID}")
        return
    raise RuntimeError(f"collection create failed {response.status_code}: {response.text}")


def _attribute_url(kind: str) -> str:
    return f"{_base()}/{COLLECTION_ID}/attributes/{kind}"


def _create_string(key: str, size: int = 255, *, required: bool = False, array: bool = False) -> None:
    payload = {
        "key": key,
        "size": size,
        "required": required,
        "array": array,
    }
    response = _request("POST", _attribute_url("string"), payload)
    if response.status_code in {200, 201, 202, 409}:
        print(f"attribute ready: {key}")
        time.sleep(0.15)
        return
    raise RuntimeError(f"attribute {key} failed {response.status_code}: {response.text}")


def _create_datetime(key: str) -> None:
    response = _request("POST", _attribute_url("datetime"), {"key": key, "required": False})
    if response.status_code in {200, 201, 202, 409}:
        print(f"attribute ready: {key}")
        time.sleep(0.15)
        return
    raise RuntimeError(f"attribute {key} failed {response.status_code}: {response.text}")


def _create_float(key: str) -> None:
    response = _request("POST", _attribute_url("float"), {"key": key, "required": False})
    if response.status_code in {200, 201, 202, 409}:
        print(f"attribute ready: {key}")
        time.sleep(0.15)
        return
    raise RuntimeError(f"attribute {key} failed {response.status_code}: {response.text}")


def bootstrap() -> None:
    _create_collection()
    _create_string("asset_id", 128, required=True)
    _create_string("name", 255, required=True)
    _create_string("category", 96, required=True)
    _create_string("subcategory", 128)
    _create_string("source", 128)
    _create_string("source_origin", 128)
    _create_string("role", 32)
    _create_string("sub_category", 128)
    _create_string("gender_fit", 32)
    _create_string("image_url", 2048, required=True)
    _create_string("board_image_url", 2048)
    _create_string("normalized_url", 2048)
    _create_string("cutout_url", 2048)
    _create_string("r2_key", 512)
    _create_string("colors", 64, array=True)
    _create_string("archetypes", 128, array=True)
    _create_string("occasions", 96, array=True)
    _create_string("tags", 96, array=True)
    _create_string("gender", 32)
    _create_string("status", 32)
    _create_string("pattern", 96)
    _create_string("material", 96)
    _create_string("finish", 96)
    _create_string("occasion_families", 96, array=True)
    _create_string("traits", 96, array=True)
    _create_string("weather_tags", 64, array=True)
    _create_string("cultural_context", 96, array=True)
    for key in ("visual_noise", "statement_level", "formality", "energy", "movement", "metadata_score"):
        _create_float(key)
    _create_string("metadata_version", 32)
    _create_string("metadata_status", 32)
    _create_string("missing_metadata_fields", 96, array=True)
    _create_datetime("metadata_updated_at")
    _create_datetime("created_at")
    _create_datetime("updated_at")
    print(json.dumps({"success": True, "collection": COLLECTION_ID}, indent=2))


if __name__ == "__main__":
    bootstrap()
