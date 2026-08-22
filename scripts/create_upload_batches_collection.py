"""Appwrite Collection, Attribute & Index Provisioning Script.

Provisions `upload_batches` and `upload_batch_items` collections idempotently
with snake_case attributes and UNIQUE indexes using Appwrite REST API.

Usage:
    python scripts/create_upload_batches_collection.py
"""

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger(__name__)

BATCHES_COLLECTION = os.getenv("APPWRITE_COLLECTION_UPLOAD_BATCHES", "upload_batches")
ITEMS_COLLECTION = os.getenv("APPWRITE_COLLECTION_UPLOAD_BATCH_ITEMS", "upload_batch_items")


def create_collection_if_missing(proxy: AppwriteProxy, collection_id: str, name: str) -> None:
    url = f"{proxy.endpoint}/databases/{proxy.database_id}/collections"
    payload = {
        "collectionId": collection_id,
        "name": name,
        "permissions": [],
        "documentSecurity": False,
    }
    try:
        res = proxy._request("POST", url, payload=payload)
        print(f"  + Collection '{collection_id}' created.")
    except Exception as exc:
        if "409" in str(exc) or "already exists" in str(exc).lower():
            print(f"  ✓ Collection '{collection_id}' already exists.")
        else:
            print(f"  ⚠️ Error creating collection '{collection_id}': {exc}")


def create_string_attribute(proxy: AppwriteProxy, collection_id: str, key: str, size: int = 255, required: bool = False) -> None:
    url = f"{proxy.endpoint}/databases/{proxy.database_id}/collections/{collection_id}/attributes/string"
    payload = {"key": key, "size": size, "required": required}
    try:
        proxy._request("POST", url, payload=payload)
        print(f"    + Attribute '{key}' (string[{size}]) added to '{collection_id}'.")
    except Exception as exc:
        if "409" in str(exc) or "already exists" in str(exc).lower():
            print(f"    ✓ Attribute '{key}' already exists in '{collection_id}'.")
        else:
            print(f"    ⚠️ Error adding attribute '{key}': {exc}")


def create_integer_attribute(proxy: AppwriteProxy, collection_id: str, key: str, required: bool = False, default: int = 0) -> None:
    url = f"{proxy.endpoint}/databases/{proxy.database_id}/collections/{collection_id}/attributes/integer"
    payload = {"key": key, "required": required, "default": default, "min": -2147483648, "max": 2147483647}
    try:
        proxy._request("POST", url, payload=payload)
        print(f"    + Attribute '{key}' (integer) added to '{collection_id}'.")
    except Exception as exc:
        if "409" in str(exc) or "already exists" in str(exc).lower():
            print(f"    ✓ Attribute '{key}' already exists in '{collection_id}'.")
        else:
            print(f"    ⚠️ Error adding attribute '{key}': {exc}")


def create_index(proxy: AppwriteProxy, collection_id: str, key: str, index_type: str, attributes: list[str]) -> None:
    url = f"{proxy.endpoint}/databases/{proxy.database_id}/collections/{collection_id}/indexes"
    payload = {"key": key, "type": index_type, "attributes": attributes}
    try:
        proxy._request("POST", url, payload=payload)
        print(f"    + Index '{key}' ({index_type} on {attributes}) added to '{collection_id}'.")
    except Exception as exc:
        if "409" in str(exc) or "already exists" in str(exc).lower():
            print(f"    ✓ Index '{key}' already exists in '{collection_id}'.")
        else:
            print(f"    ⚠️ Error adding index '{key}': {exc}")


def bootstrap_upload_batches_collections():
    print(f"🚀 Provisioning Appwrite Upload Batch Collections...")
    proxy = AppwriteProxy()

    if not proxy.is_configured:
        print("⚠️ Appwrite configuration missing (APPWRITE_ENDPOINT/APPWRITE_PROJECT_ID).")
        print("   Collections will fallback to in-memory store cleanly.")
        return

    # 1. upload_batches
    print(f"\n📦 Provisioning '{BATCHES_COLLECTION}'...")
    create_collection_if_missing(proxy, BATCHES_COLLECTION, "Upload Batches")
    create_string_attribute(proxy, BATCHES_COLLECTION, "user_id", size=128, required=True)
    create_string_attribute(proxy, BATCHES_COLLECTION, "client_batch_request_id", size=128, required=True)
    create_string_attribute(proxy, BATCHES_COLLECTION, "request_fingerprint", size=128, required=True)
    create_string_attribute(proxy, BATCHES_COLLECTION, "status", size=64, required=True)
    create_string_attribute(proxy, BATCHES_COLLECTION, "active_item_id", size=128, required=False)
    create_integer_attribute(proxy, BATCHES_COLLECTION, "total_items", required=False, default=0)
    create_integer_attribute(proxy, BATCHES_COLLECTION, "added_count", required=False, default=0)
    create_integer_attribute(proxy, BATCHES_COLLECTION, "needs_review_count", required=False, default=0)
    create_integer_attribute(proxy, BATCHES_COLLECTION, "rejected_count", required=False, default=0)
    create_integer_attribute(proxy, BATCHES_COLLECTION, "failed_count", required=False, default=0)

    # Unique index on (user_id, client_batch_request_id)
    create_index(proxy, BATCHES_COLLECTION, "unique_user_batch", "unique", ["user_id", "client_batch_request_id"])
    create_index(proxy, BATCHES_COLLECTION, "idx_user_status", "key", ["user_id", "status"])

    # 2. upload_batch_items
    print(f"\n📄 Provisioning '{ITEMS_COLLECTION}'...")
    create_collection_if_missing(proxy, ITEMS_COLLECTION, "Upload Batch Items")
    create_string_attribute(proxy, ITEMS_COLLECTION, "user_id", size=128, required=True)
    create_string_attribute(proxy, ITEMS_COLLECTION, "batch_id", size=128, required=True)
    create_string_attribute(proxy, ITEMS_COLLECTION, "client_upload_item_id", size=128, required=True)
    create_string_attribute(proxy, ITEMS_COLLECTION, "content_hash", size=128, required=False)
    create_integer_attribute(proxy, ITEMS_COLLECTION, "queue_position", required=False, default=1)
    create_string_attribute(proxy, ITEMS_COLLECTION, "status", size=64, required=True)
    create_string_attribute(proxy, ITEMS_COLLECTION, "processing_route", size=64, required=False)
    create_string_attribute(proxy, ITEMS_COLLECTION, "wardrobe_item_id", size=128, required=False)
    create_string_attribute(proxy, ITEMS_COLLECTION, "error_code", size=128, required=False)
    create_integer_attribute(proxy, ITEMS_COLLECTION, "attempt_count", required=False, default=0)
    create_string_attribute(proxy, ITEMS_COLLECTION, "lease_owner", size=128, required=False)
    create_integer_attribute(proxy, ITEMS_COLLECTION, "lease_expires_at", required=False, default=0)

    # Unique index on (user_id, client_upload_item_id)
    create_index(proxy, ITEMS_COLLECTION, "unique_user_item", "unique", ["user_id", "client_upload_item_id"])
    create_index(proxy, ITEMS_COLLECTION, "idx_batch_status", "key", ["batch_id", "status"])

    print("\n🎉 Provisioning complete!")


if __name__ == "__main__":
    bootstrap_upload_batches_collections()
