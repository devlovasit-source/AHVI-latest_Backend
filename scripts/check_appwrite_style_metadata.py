"""Check whether the Appwrite outfits collection has style_metadata.

This is intentionally read-only. Add style_metadata as a string attribute in
Appwrite before expecting wardrobe intelligence metadata to persist there.
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
        os.getenv("APPWRITE_COLLECTION_OUTFITS")
        or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_OUTFITS")
        or os.getenv("APPWRITE_COLLECTION_ID")
        or ""
    )
    missing = [
        name
        for name, value in {
            "APPWRITE_ENDPOINT": endpoint,
            "APPWRITE_PROJECT_ID": project_id,
            "APPWRITE_API_KEY": api_key,
            "APPWRITE_DATABASE_ID": database_id,
            "APPWRITE_COLLECTION_OUTFITS": collection_id,
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
    found = next((attr for attr in attrs if attr.get("key") == "style_metadata"), None)
    if not found:
        print("style_metadata: missing. Create it as a string attribute before full metadata persistence.")
        return 1

    print(
        "style_metadata: present "
        f"type={found.get('type')} size={found.get('size')} status={found.get('status')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
