import os
import re
import uuid
from typing import Any, Dict, List

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
    "X-Appwrite-Project": APPWRITE_PROJECT_ID or "",
    "X-Appwrite-Key": APPWRITE_API_KEY or "",
    "Content-Type": "application/json",
}


# =========================
# HELPERS
# =========================
_HEX6_RE = re.compile(r"^[0-9a-fA-F]{6}$")
_SAFE_DOC_ID_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _appwrite_ready() -> bool:
    return bool(
        APPWRITE_ENDPOINT
        and APPWRITE_PROJECT_ID
        and APPWRITE_API_KEY
        and APPWRITE_DATABASE_ID
        and APPWRITE_COLLECTION_ID
    )


def _tokens(value: str) -> List[str]:
    return (
        re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
        .strip()
        .split()
    )


def _has_any(tokens: List[str], words: List[str]) -> bool:
    return any(word in tokens for word in words)


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _safe_document_id(value: Any) -> str:
    raw = _safe_text(value)
    if not raw:
        raw = str(uuid.uuid4())

    safe = _SAFE_DOC_ID_RE.sub("_", raw).strip("._-")
    if not safe:
        safe = str(uuid.uuid4())

    # Appwrite custom document IDs have length restrictions.
    return safe[:36]


def _normalize_hex_color(value: Any, default: str = "#000000") -> str:
    text = str(value or "").strip()
    if not text:
        return default

    # Common named-color fallback for AI outputs.
    named = {
        "black": "#000000",
        "white": "#FFFFFF",
        "red": "#FF0000",
        "blue": "#0000FF",
        "green": "#008000",
        "yellow": "#FFFF00",
        "brown": "#8B4513",
        "tan": "#D2B48C",
        "beige": "#F5F5DC",
        "grey": "#808080",
        "gray": "#808080",
        "navy": "#000080",
        "pink": "#FFC0CB",
        "purple": "#800080",
        "orange": "#FFA500",
    }

    lowered = text.lower()
    if lowered in named:
        return named[lowered]

    if text.startswith("#"):
        text = text[1:]

    if len(text) == 3:
        text = "".join(c * 2 for c in text)

    if not _HEX6_RE.match(text):
        return default

    return f"#{text.upper()}"


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x or "").strip()]

    if isinstance(value, str):
        if not value.strip():
            return []
        return [x.strip() for x in value.split(",") if x.strip()]

    return []


def _create_document(document_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if not _appwrite_ready():
        raise RuntimeError(
            "Appwrite wardrobe persistence is not configured. "
            "Set APPWRITE_ENDPOINT, APPWRITE_PROJECT_ID, APPWRITE_API_KEY, "
            "APPWRITE_DATABASE_ID, and APPWRITE_COLLECTION_OUTFITS."
        )

    url = (
        f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}"
        f"/collections/{APPWRITE_COLLECTION_ID}/documents"
    )

    res = requests.post(
        url,
        json={
            "documentId": document_id,
            "data": data,
        },
        headers=HEADERS,
        timeout=20,
    )

    if res.status_code not in (200, 201):
        raise RuntimeError(f"Appwrite error: {res.status_code} {res.text}")

    return res.json()


# =========================
# CATEGORY NORMALIZATION
# =========================
def normalize_category(cat: Any, name: Any = "", sub_category: Any = "") -> str:
    """
    Token-aware category inference.

    Critical demo fix:
    - "Short-Sleeved Shirt" => Tops
    - "Khaki Shorts" => Bottoms
    - "Brown Boots" => Footwear
    - "Watch" => Accessories
    """

    explicit = str(cat or "").strip().lower()

    explicit_map = {
        "tops": "Tops",
        "top": "Tops",
        "bottoms": "Bottoms",
        "bottom": "Bottoms",
        "footwear": "Footwear",
        "shoe": "Footwear",
        "shoes": "Footwear",
        "accessories": "Accessories",
        "accessory": "Accessories",
        "bags": "Accessories",
        "bag": "Accessories",
        "jewelry": "Accessories",
        "jewellery": "Accessories",
        "outerwear": "Outerwear",
        "dresses": "Dresses",
        "dress": "Dresses",
        "indian wear": "Dresses",
    }

    if explicit in explicit_map:
        return explicit_map[explicit]

    text = " ".join(
        [
            str(cat or ""),
            str(name or ""),
            str(sub_category or ""),
        ]
    )
    t = _tokens(text)

    # Tops first so "Short-Sleeved Shirt" never becomes Bottoms.
    if _has_any(
        t,
        [
            "shirt",
            "shirts",
            "tee",
            "tshirt",
            "tshirts",
            "top",
            "tops",
            "blouse",
            "blouses",
            "hoodie",
            "hoodies",
            "sweater",
            "sweaters",
            "kurta",
            "kurtas",
            "polo",
            "polos",
        ],
    ):
        return "Tops"

    # Only "shorts", not "short".
    if _has_any(
        t,
        [
            "pants",
            "pant",
            "trousers",
            "trouser",
            "jeans",
            "jean",
            "shorts",
            "skirt",
            "skirts",
            "legging",
            "leggings",
            "chino",
            "chinos",
        ],
    ):
        return "Bottoms"

    if _has_any(
        t,
        [
            "shoe",
            "shoes",
            "boot",
            "boots",
            "sneaker",
            "sneakers",
            "heel",
            "heels",
            "sandal",
            "sandals",
            "loafer",
            "loafers",
            "slipper",
            "slippers",
        ],
    ):
        return "Footwear"

    if _has_any(
        t,
        [
            "watch",
            "watches",
            "bag",
            "bags",
            "belt",
            "belts",
            "scarf",
            "scarves",
            "jewelry",
            "jewellery",
            "ring",
            "rings",
            "necklace",
            "bracelet",
            "earring",
            "earrings",
            "accessory",
            "accessories",
            "hat",
            "cap",
            "sunglass",
            "sunglasses",
        ],
    ):
        return "Accessories"

    if _has_any(t, ["jacket", "coat", "blazer", "outerwear", "cardigan", "overshirt"]):
        return "Outerwear"

    if _has_any(t, ["dress", "dresses", "gown", "jumpsuit", "saree", "lehenga", "sherwani"]):
        return "Dresses"

    return "Accessories"


def _build_appwrite_doc(
    *,
    user_id: str,
    file_id: str,
    item: Dict[str, Any],
    raw_url: str,
    masked_url: str,
) -> Dict[str, Any]:
    sub_category = _safe_text(
        item.get("sub_category")
        or item.get("subcategory")
        or item.get("type")
        or item.get("label"),
        "Item",
    )

    name = _safe_text(
        item.get("name")
        or item.get("label")
        or sub_category,
        "Item",
    )

    category = normalize_category(item.get("category"), name, sub_category)
    color = _normalize_hex_color(
        item.get("color_code")
        or item.get("color")
        or item.get("hex")
    )
    pattern = _safe_text(item.get("pattern"), "plain").lower()
    occasions = _normalize_list(item.get("occasions") or item.get("occasion_tags"))

    # Must match Appwrite outfits collection schema exactly.
    # Do not add raw_url, notes, user_id, etc.
    return {
        "image_url": raw_url,
        "category": category,
        "userId": user_id,
        "status": "active",
        "masked_url": masked_url,
        "image_id": file_id,
        "masked_id": file_id,
        "name": name,
        "sub_category": sub_category,
        "color_code": color,
        "occasions": occasions,
        "pattern": pattern,
        "worn": int(item.get("worn") or 0),
        "liked": bool(item.get("liked") or False),
        "qdrant_point_id": file_id,
    }


# =========================
# MAIN FUNCTION
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

    user_id = _safe_text(user_id)
    if not user_id:
        raise ValueError("user_id is required")

    selected_ids = {str(x).strip() for x in (selected_item_ids or []) if str(x or "").strip()}

    saved_items: List[Dict[str, Any]] = []
    errors: List[str] = []
    skipped = 0

    for item in detected_items or []:
        if not isinstance(item, dict):
            skipped += 1
            continue

        item_id = (
            item.get("item_id")
            or item.get("id")
            or item.get("image_id")
            or item.get("masked_id")
        )

        if selected_ids and str(item_id) not in selected_ids:
            continue

        try:
            file_id = _safe_document_id(item_id or uuid.uuid4())

            raw_url = _safe_text(
                item.get("raw_url")
                or item.get("image_url")
                or item.get("imageUrl")
            )

            masked_url = _safe_text(
                item.get("masked_url")
                or item.get("maskedUrl")
                or raw_url
            )

            if not raw_url and not masked_url:
                skipped += 1
                errors.append(f"{file_id}: missing image_url/masked_url")
                continue

            if not raw_url:
                raw_url = masked_url

            if not masked_url:
                masked_url = raw_url

            doc = _build_appwrite_doc(
                user_id=user_id,
                file_id=file_id,
                item=item,
                raw_url=raw_url,
                masked_url=masked_url,
            )

            created = _create_document(file_id, doc)

            try:
                embedding = embedding_service.encode_text(
                    " ".join(
                        [
                            doc["name"],
                            doc["category"],
                            doc["sub_category"],
                            doc["color_code"],
                            doc["pattern"],
                            " ".join(doc["occasions"]),
                        ]
                    )
                )

                qdrant_service.upsert_wardrobe_item(
                    {
                        "id": file_id,
                        "userId": user_id,
                        "type": str(doc["sub_category"]).lower(),
                        "category": doc["category"],
                        "color": doc["color_code"],
                        "image_url": masked_url,
                        "embedding": embedding,
                    }
                )
            except Exception as exc:
                # Do not fail wardrobe save because Qdrant failed.
                print("[qdrant error]", exc)

            saved_items.append(created)

        except Exception as exc:
            skipped += 1
            error_msg = f"{item.get('item_id') or item.get('id') or 'unknown'}: {exc}"
            errors.append(error_msg)
            print("[persist error]", error_msg)

    return {
        "success": bool(saved_items),
        "saved_count": len(saved_items),
        "items": saved_items,
        "skipped": skipped,
        "errors": errors[:10],
    }


__all__ = [
    "persist_selected_items",
    "normalize_category",
]