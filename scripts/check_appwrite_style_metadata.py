"""Check whether the Appwrite wardrobe_style_metadata table is ready.

This is intentionally read-only. The outfits table should not need a
style_metadata column; metadata now persists in a companion table.
"""

from __future__ import annotations

import os
import sys

import requests


def main() -> int:
    endpoint = (os.getenv("APPWRITE_ENDPOINT") or "").rstrip("/")
    project_id = os.getenv("APPWRITE_PROJECT_ID") or ""
    api_key = os.getenv("APPWRITE_API_KEY") or ""
    database_id = os.getenv("APPWRITE_DATABASE_ID") or os.getenv("EXPO_PUBLIC_APPWRITE_DATABASE_ID") or ""
    collection_id = (
        os.getenv("APPWRITE_COLLECTION_WARDROBE_STYLE_METADATA")
        or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_WARDROBE_STYLE_METADATA")
        or "wardrobe_style_metadata"
    )
    missing = [
        name
        for name, value in {
            "APPWRITE_ENDPOINT": endpoint,
            "APPWRITE_PROJECT_ID": project_id,
            "APPWRITE_API_KEY": api_key,
            "APPWRITE_DATABASE_ID": database_id,
            "APPWRITE_COLLECTION_WARDROBE_STYLE_METADATA": collection_id,
        }.items()
        if not value
    ]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        return 2

    url = f"{endpoint}/databases/{database_id}/collections/{collection_id}/attributes"
    response = requests.get(
        url,
        headers={"X-Appwrite-Project": project_id, "X-Appwrite-Key": api_key},
        timeout=20,
    )
    if response.status_code != 200:
        print(f"Could not read attributes: {response.status_code} {response.text}")
        return 1

    attrs = response.json().get("attributes") or []
    required = {"item_id", "userId", "style_metadata"}
    found = {str(attr.get("key") or "") for attr in attrs}
    missing_attrs = sorted(required - found)
    if missing_attrs:
        print(f"wardrobe_style_metadata missing attributes: {', '.join(missing_attrs)}")
        return 1

    print("wardrobe_style_metadata: ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
