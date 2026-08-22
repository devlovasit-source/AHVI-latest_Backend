"""Create the qdrant_point_id index on the Appwrite outfits collection.

Background
----------
`routers/data.py:_hydrate_existing_outfit_document` paginates up to 2000
documents (20 round trips of 100) on every duplicate save when the
qdrant_point_id is needed. Indexing this attribute lets us look it up with a
single query.

Run once
--------
Required env vars (same as the rest of the backend):

    APPWRITE_ENDPOINT, APPWRITE_PROJECT_ID, APPWRITE_DATABASE_ID,
    APPWRITE_API_KEY, APPWRITE_COLLECTION_OUTFITS

Then:

    python -m tools.appwrite_create_qdrant_index

The script is idempotent — if the index already exists it logs and exits 0.
Cloud Console alternative: Database > outfits > Indexes > Add index
  Type: key | Attributes: qdrant_point_id | Order: ASC

Note: the underlying `qdrant_point_id` *attribute* must exist on the
collection. The script validates and aborts if it does not.
"""
from __future__ import annotations

import os
import sys
import time

import requests


INDEX_KEY = "qdrant_point_id_idx"
ATTRIBUTE = "qdrant_point_id"
INDEX_TYPE = "key"


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        print(f"ERROR: missing env var {name}", file=sys.stderr)
        sys.exit(2)
    return value


def main() -> int:
    endpoint = _required_env("APPWRITE_ENDPOINT").rstrip("/")
    project_id = _required_env("APPWRITE_PROJECT_ID")
    database_id = _required_env("APPWRITE_DATABASE_ID")
    api_key = _required_env("APPWRITE_API_KEY")
    collection_id = _required_env("APPWRITE_COLLECTION_OUTFITS")

    headers = {
        "X-Appwrite-Project": project_id,
        "X-Appwrite-Key": api_key,
        "Content-Type": "application/json",
    }
    base = f"{endpoint}/databases/{database_id}/collections/{collection_id}"

    # 1. Check the attribute exists.
    attr_url = f"{base}/attributes/{ATTRIBUTE}"
    resp = requests.get(attr_url, headers=headers, timeout=15)
    if resp.status_code == 404:
        print(
            f"ERROR: attribute '{ATTRIBUTE}' not found on collection "
            f"'{collection_id}'. Add it as a String attribute first.",
            file=sys.stderr,
        )
        return 3
    if resp.status_code >= 400:
        print(
            f"ERROR: attribute lookup failed status={resp.status_code} body={resp.text}",
            file=sys.stderr,
        )
        return 4

    # 2. Check whether the index already exists (idempotency).
    index_get_url = f"{base}/indexes/{INDEX_KEY}"
    resp = requests.get(index_get_url, headers=headers, timeout=15)
    if resp.status_code == 200:
        print(f"OK: index '{INDEX_KEY}' already exists on '{collection_id}'.")
        return 0
    if resp.status_code != 404:
        print(
            f"WARN: index lookup unexpected status={resp.status_code} body={resp.text}",
            file=sys.stderr,
        )

    # 3. Create the index.
    create_url = f"{base}/indexes"
    payload = {
        "key": INDEX_KEY,
        "type": INDEX_TYPE,
        "attributes": [ATTRIBUTE],
        "orders": ["ASC"],
    }
    resp = requests.post(create_url, headers=headers, json=payload, timeout=30)
    if resp.status_code >= 400:
        print(
            f"ERROR: index create failed status={resp.status_code} body={resp.text}",
            file=sys.stderr,
        )
        return 5

    # 4. Poll until 'available' (or timeout).
    deadline = time.time() + 60
    while time.time() < deadline:
        resp = requests.get(index_get_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            status = str(data.get("status") or "").lower()
            if status == "available":
                print(f"OK: index '{INDEX_KEY}' is available.")
                return 0
            print(f"... index status={status}, waiting")
        time.sleep(2)

    print(f"WARN: index create accepted but not yet available after 60s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
