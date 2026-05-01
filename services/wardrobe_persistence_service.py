import base64
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import requests

from services.embedding_service import embedding_service
from services.qdrant_service import qdrant_service


# =========================
# ENV CONFIG
# =========================
APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY")
APPWRITE_DATABASE_ID = os.getenv("APPWRITE_DATABASE_ID")
APPWRITE_COLLECTION_ID = (
    os.getenv("APPWRITE_COLLECTION_OUTFITS")
    or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_OUTFITS")
    or os.getenv("APPWRITE_COLLECTION_ID")
)

HEADERS = {
    "X-Appwrite-Project": APPWRITE_PROJECT_ID,
    "X-Appwrite-Key": APPWRITE_API_KEY,
    "Content-Type": "application/json",
}


# =========================
# HELPERS
# =========================
_HEX6_RE = re.compile(r"^[0-9a-fA-F]{6}$")


def _appwrite_ready() -> bool:
    return bool(
        APPWRITE_ENDPOINT
        and APPWRITE_PROJECT_ID
        and APPWRITE_API_KEY
        and APPWRITE_DATABASE_ID
        and APPWRITE_COLLECTION_ID
    )


def _normalize_hex_color(value: str, default: str = "#000000") -> str:
    text = (value or "").strip()
    if not text:
        return default
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join([c * 2 for c in text])
    if not _HEX6_RE.match(text):
        return default
    return f"#{text.upper()}"


def _create_document(document_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if not _appwrite_ready():
        raise RuntimeError(
            "Appwrite wardrobe persistence is not configured. "
            "Set APPWRITE_ENDPOINT, APPWRITE_PROJECT_ID, APPWRITE_API_KEY, "
            "APPWRITE_DATABASE_ID, and APPWRITE_COLLECTION_OUTFITS."
        )

    url = f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}/collections/{APPWRITE_COLLECTION_ID}/documents"

    res = requests.post(
        url,
        json={"documentId": document_id, "data": data},
        headers=HEADERS,
        timeout=20,
    )

    if res.status_code not in (200, 201):
        raise Exception(f"Appwrite error: {res.text}")

    return res.json()


# =========================
# CATEGORY NORMALIZATION
# =========================
CATEGORY_MAP = {
    "tops": "Tops",
    "bottoms": "Bottoms",
    "outerwear": "Outerwear",
    "dresses": "Dresses",
    "footwear": "Footwear",
    "bags": "Bags",
    "accessories": "Accessories",
    "jewelry": "Jewelry",
    "indian wear": "Indian Wear",
}


def normalize_category(cat: str) -> str:
    return CATEGORY_MAP.get(str(cat).lower(), "Tops")


# =========================
# MAIN FUNCTION (UPGRADED)
# =========================
def persist_selected_items(
    user_id: str,
    selected_item_ids: List[str],
    detected_items: List[Dict[str, Any]],
):
    if not _appwrite_ready():
        raise RuntimeError(
            "Appwrite wardrobe persistence is not configured; refusing to report a fake save."
        )

    saved_items = []
    skipped = 0

    selected_ids = set(map(str, selected_item_ids or []))

    for item in detected_items or []:
        if str(item.get("item_id")) not in selected_ids:
            continue

        try:
            file_id = str(item.get("item_id") or uuid.uuid4())

            # -------------------------
            # 🔥 NEW: URL-FIRST PIPELINE
            # -------------------------
            raw_url = item.get("raw_url") or item.get("image_url")
            masked_url = item.get("masked_url")

            if not masked_url:
                skipped += 1
                continue

            # -------------------------
            # METADATA
            # -------------------------
            category = normalize_category(item.get("category"))
            sub_category = str(item.get("sub_category") or "Item")
            item_type = sub_category.lower()

            color = _normalize_hex_color(item.get("color_code"))
            pattern = str(item.get("pattern") or "plain").lower()
            occasions = item.get("occasions") or []

            embedding = embedding_service.encode_text(
                f"{color} {item_type} {category} {pattern}"
            )

            # -------------------------
            # 🔥 UPDATED SCHEMA
            # -------------------------
            doc = {
                "userId": user_id,
                "status": "active",

                # ✅ NEW FIELDS
                "image_url": raw_url,
                "masked_url": masked_url,
                "raw_url": raw_url,  # optional

                "image_id": file_id,
                "masked_id": file_id,
                "qdrant_point_id": file_id,

                "name": item.get("name", "Item"),
                "category": category,
                "sub_category": sub_category,
                "color_code": color,
                "pattern": pattern,
                "occasions": occasions,

                "worn": 0,
                "liked": False,
            }

            # -------------------------
            # SAVE
            # -------------------------
            created = _create_document(file_id, doc)

            # -------------------------
            # QDRANT
            # -------------------------
            try:
                qdrant_service.upsert_wardrobe_item(
                    {
                        "id": file_id,
                        "userId": user_id,
                        "type": item_type,
                        "category": category,
                        "color": color,
                        "image_url": masked_url,
                        "embedding": embedding,
                    }
                )
            except Exception as e:
                print("[qdrant error]", e)

            saved_items.append(created)

        except Exception as e:
            print("[persist error]", e)

    return {
        "success": bool(saved_items),
        "saved_count": len(saved_items),
        "items": saved_items,
        "skipped": skipped,
    }
