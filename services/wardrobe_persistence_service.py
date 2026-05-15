import os
import re
import uuid
from typing import Any, Dict, List, Tuple

import requests

from services.embedding_service import embedding_service
from services.category_taxonomy import infer_style_attributes
from services.qdrant_service import qdrant_service
from services.wardrobe_taxonomy import normalize as _taxonomy_normalize

# =========================
# ENV CONFIG
# =========================
APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY")
APPWRITE_DATABASE_ID = (
    os.getenv("APPWRITE_DATABASE_ID")
    or os.getenv("EXPO_PUBLIC_APPWRITE_DATABASE_ID")
)


def _all_known_outfits_collections() -> List[str]:
    """Every env var name we've ever used for the outfits collection.

    Cloud Run + Expo + legacy deploys have set different combinations.
    update_item_labels probes them in order on 404 so users don't get
    'Not Found' just because the env var the code reads first happens
    to be empty or pointing at an older collection.
    """
    candidates: List[str] = []
    seen: set[str] = set()
    for name in (
        "APPWRITE_COLLECTION_OUTFITS",
        "EXPO_PUBLIC_APPWRITE_COLLECTION_OUTFITS",
        "APPWRITE_COLLECTION_ID",
        "APPWRITE_OUTFITS_COLLECTION_ID",
        "APPWRITE_WARDROBE_COLLECTION_ID",
    ):
        val = os.getenv(name)
        if val and val not in seen:
            candidates.append(val)
            seen.add(val)
    return candidates


_KNOWN_COLLECTIONS = _all_known_outfits_collections()
APPWRITE_COLLECTION_ID = _KNOWN_COLLECTIONS[0] if _KNOWN_COLLECTIONS else None

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
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip().split()


def _has_any(tokens: List[str], words: List[str]) -> bool:
    return any(word in tokens for word in words)


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _first_url(item: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        text = _safe_text(value)
        if text and text.lower() != "null":
            return text
    return ""


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

    def post(payload: Dict[str, Any]):
        return requests.post(
            url,
            json={
                "documentId": document_id,
                "data": payload,
            },
            headers=HEADERS,
            timeout=20,
        )

    res = post(data)

    optional_keys = {"pixel_hash", "image_embedding", "image_vector"}
    if res.status_code not in (200, 201) and optional_keys.intersection(data):
        body = str(res.text or "").lower()
        if "unknown attribute" in body or "invalid document structure" in body:
            clean_data = {k: v for k, v in data.items() if k not in optional_keys}
            res = post(clean_data)

    if res.status_code not in (200, 201):
        raise RuntimeError(f"Appwrite error: {res.status_code} {res.text}")

    return res.json()


# =========================
# CATEGORY NORMALIZATION
# =========================
def normalize_category(cat: Any, name: Any = "", sub_category: Any = "") -> str:
    """Final save-time category. Delegates to canonical taxonomy module."""
    category, _ = _taxonomy_normalize(cat, name, sub_category)
    return category


def _legacy_normalize_category(cat: Any, name: Any = "", sub_category: Any = "") -> str:
    """
    Legacy fallback (kept for reference). Strong garment/accessory signals
    override weak explicit categories. New code should use normalize_category
    which now delegates to services.wardrobe_taxonomy.
    """

    text = " ".join(
        [
            str(cat or ""),
            str(name or ""),
            str(sub_category or ""),
        ]
    ).lower()
    tokens = _tokens(text)

    def has_any(words: List[str]) -> bool:
        return _has_any(tokens, words) or any(word in text for word in words)

    # Strong overrides FIRST. These must run before trusting explicit
    # "Tops" / "Accessories" labels from vision or older clients.
    if has_any(["saree", "sari", "lehenga"]):
        return "Dresses"
    if has_any(["dupatta", "sherwani", "kurta", "kurti", "anarkali"]):
        return "Traditional"

    if has_any(
        [
            "one piece dress",
            "one-piece dress",
            "mini dress",
            "midi dress",
            "maxi dress",
            "bodycon dress",
            "shift dress",
            "wrap dress",
            "slip dress",
            "dress",
            "dresses",
            "gown",
            "jumpsuit",
        ]
    ):
        return "Dresses"

    if has_any(
        [
            "handbag",
            "shoulder bag",
            "sling bag",
            "crossbody",
            "cross body",
            "tote",
            "clutch",
            "purse",
            "backpack",
            "bag",
            "bags",
        ]
    ):
        return "Bags"

    if has_any(
        [
            "ring",
            "rings",
            "bracelet",
            "bracelets",
            "necklace",
            "necklaces",
            "earring",
            "earrings",
            "bangle",
            "bangles",
            "pendant",
            "pendants",
            "chain",
            "chains",
            "jewelry",
            "jewellery",
        ]
    ):
        return "Jewelry"

    if has_any(["watch", "watches", "belt", "belts", "sunglass", "sunglasses", "eyewear", "glasses"]):
        return "Accessories"

    explicit = str(cat or "").strip().lower()
    explicit_map = {
        "tops": "Tops",
        "top": "Tops",
        "bottoms": "Bottoms",
        "bottom": "Bottoms",
        "footwear": "Footwear",
        "shoe": "Footwear",
        "shoes": "Footwear",
        "outerwear": "Outerwear",
        "dresses": "Dresses",
        "dress": "Dresses",
        "indian wear": "Traditional",
        "traditional": "Traditional",
        "bags": "Bags",
        "bag": "Bags",
        "jewelry": "Jewelry",
        "jewellery": "Jewelry",
        "watches": "Accessories",
        "watch": "Accessories",
        "belts": "Accessories",
        "belt": "Accessories",
        "eyewear": "Accessories",
        "accessories": "Accessories",
        "accessory": "Accessories",
    }

    if explicit in explicit_map:
        return explicit_map[explicit]

    # Tops before bottoms, but only after dress/Indian/accessory overrides.
    if has_any(
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
            "polo",
            "polos",
        ],
    ):
        return "Tops"

    # Only "shorts", not "short".
    if has_any(
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

    if has_any(
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

    if has_any(["jacket", "coat", "blazer", "outerwear", "cardigan", "overshirt"]):
        return "Outerwear"

    # Unknown / low-confidence items must NOT silently land in Accessories.
    return "Needs Review"


def normalize_display_name_and_subcategory(
    name: str,
    sub_category: str,
    category: str,
) -> tuple[str, str]:
    raw_name = str(name or "").strip()
    raw_sub = str(sub_category or "").strip()
    text = f"{raw_name} {raw_sub}".lower()

    if "sari" in text or "saree" in text:
        return (
            raw_name.replace("Sari", "Saree").replace("sari", "Saree") or "Saree",
            "Saree",
        )
    if "mini dress" in text:
        return raw_name or "Mini Dress", "Mini Dress"
    if "one piece" in text or "one-piece" in text:
        return raw_name or "One-Piece Dress", "One-Piece Dress"
    if "dress" in text:
        return raw_name or "Dress", raw_sub if raw_sub else "Dress"
    if "handbag" in text:
        return raw_name or "Handbag", "Handbag"
    if "shoulder bag" in text:
        return raw_name or "Shoulder Bag", "Shoulder Bag"
    if "tote" in text:
        return raw_name or "Tote Bag", "Tote Bag"
    if "ring" in text:
        return raw_name or "Ring", "Ring"
    if "bracelet" in text:
        return raw_name or "Bracelet", "Bracelet"
    if "necklace" in text:
        return raw_name or "Necklace", "Necklace"
    if "watch" in text:
        return raw_name or "Watch", "Watch"
    if "belt" in text:
        return raw_name or "Belt", "Belt"
    if "sunglass" in text or "eyewear" in text or "glasses" in text:
        return raw_name or "Eyewear", "Eyewear"

    return raw_name or raw_sub or "Item", raw_sub or category or "Item"


def _build_appwrite_doc(
    *,
    user_id: str,
    file_id: str,
    item: Dict[str, Any],
    raw_url: str,
    masked_url: str,
    normalized_url: str,
) -> Dict[str, Any]:
    sub_category = _safe_text(
        item.get("sub_category")
        or item.get("subcategory")
        or item.get("type")
        or item.get("label"),
        "Item",
    )

    name = _safe_text(
        item.get("name") or item.get("label") or sub_category,
        "Item",
    )

    category = normalize_category(item.get("category"), name, sub_category)
    name, sub_category = normalize_display_name_and_subcategory(
        name=name,
        sub_category=sub_category,
        category=category,
    )
    color = _normalize_hex_color(
        item.get("color_code") or item.get("color") or item.get("hex")
    )
    pattern = _safe_text(item.get("pattern"), "plain").lower()
    occasions = _normalize_list(item.get("occasions") or item.get("occasion_tags"))
    style_attrs = infer_style_attributes(
        {
            **item,
            "name": name,
            "category": category,
            "sub_category": sub_category,
            "color_code": color,
            "pattern": pattern,
            "occasions": occasions,
        }
    )

    # Must match Appwrite outfits collection schema exactly.
    # image_url must point to the cleanest available asset for downstream
    # chat/style-board rendering. Add `normalized_url` as an optional String
    # attribute in Appwrite before deploying this file.
    final_image_url = normalized_url or masked_url or raw_url
    pixel_hash = _safe_text(
        item.get("pixel_hash") or item.get("pixelHash") or item.get("masked_pixel_hash")
    )

    doc = {
        "image_url": final_image_url,
        "category": category,
        "userId": user_id,
        "status": "active",
        "masked_url": masked_url or final_image_url,
        "normalized_url": normalized_url or final_image_url,
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
    if pixel_hash:
        doc["pixel_hash"] = pixel_hash
    doc["_style_attrs"] = style_attrs
    return doc


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

    selected_ids = {
        str(x).strip() for x in (selected_item_ids or []) if str(x or "").strip()
    }

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

            raw_url = _first_url(
                item,
                "raw_url",
                "rawUrl",
                "raw_image_url",
                "rawImageUrl",
            )

            masked_url = _first_url(
                item,
                "masked_url",
                "maskedUrl",
                "masked_image_url",
                "maskedImageUrl",
                "processed_url",
                "processedUrl",
                "transparent_url",
                "transparentUrl",
            )

            normalized_url = _first_url(
                item,
                "normalized_url",
                "normalizedUrl",
                "normalized_image_url",
                "normalizedImageUrl",
                "png_url",
                "pngUrl",
                "cutout_url",
                "cutoutUrl",
            )

            # Legacy clients may only send image_url/imageUrl. Treat that as the
            # best available display image, not necessarily the raw original.
            legacy_image_url = _first_url(item, "image_url", "imageUrl", "url")

            if not normalized_url:
                normalized_url = masked_url or legacy_image_url
            if not masked_url:
                masked_url = normalized_url or legacy_image_url or raw_url
            if not raw_url:
                raw_url = legacy_image_url or masked_url or normalized_url

            if not raw_url and not masked_url and not normalized_url:
                skipped += 1
                errors.append(f"{file_id}: missing image_url/masked_url/normalized_url")
                continue

            doc = _build_appwrite_doc(
                user_id=user_id,
                file_id=file_id,
                item=item,
                raw_url=raw_url,
                masked_url=masked_url,
                normalized_url=normalized_url,
            )
            style_attrs = doc.pop("_style_attrs", {})

            created = _create_document(file_id, doc)

            try:
                pixel_hash = _safe_text(
                    item.get("pixel_hash")
                    or item.get("pixelHash")
                    or item.get("masked_pixel_hash")
                    or doc.get("pixel_hash")
                )
                image_embedding = (
                    item.get("image_embedding")
                    or item.get("imageEmbedding")
                    or item.get("image_vector")
                    or item.get("imageVector")
                    or []
                )
                embedding = embedding_service.encode_text(
                    " ".join(
                        [
                            doc["name"],
                            doc["category"],
                            doc["sub_category"],
                            doc["color_code"],
                            doc["pattern"],
                            " ".join(doc["occasions"]),
                            str(style_attrs.get("aesthetic_cluster") or ""),
                            str(style_attrs.get("silhouette_family") or ""),
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
                        "image_url": doc.get("image_url") or masked_url,
                        "pixel_hash": pixel_hash,
                        "embedding": embedding,
                        "formality": style_attrs.get("formality"),
                        "aesthetic_cluster": style_attrs.get("aesthetic_cluster"),
                        "visual_weight": style_attrs.get("visual_weight"),
                        "silhouette_family": style_attrs.get("silhouette_family"),
                        "occasion_fitness": style_attrs.get("occasion_fitness"),
                    }
                )
                if image_embedding:
                    qdrant_service.upsert_image_vector(
                        file_id,
                        image_embedding,
                        {
                            "userId": user_id,
                            "type": str(doc["sub_category"]).lower(),
                            "category": doc["category"],
                            "color": doc["color_code"],
                            "image_url": doc.get("image_url") or masked_url,
                            "masked_url": masked_url,
                            "normalized_url": normalized_url,
                            "pixel_hash": pixel_hash,
                            "qdrant_point_id": file_id,
                            "formality": style_attrs.get("formality"),
                            "aesthetic_cluster": style_attrs.get("aesthetic_cluster"),
                            "visual_weight": style_attrs.get("visual_weight"),
                            "silhouette_family": style_attrs.get("silhouette_family"),
                            "occasion_fitness": style_attrs.get("occasion_fitness"),
                        },
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


def _fetch_document(document_id: str) -> Tuple[Dict[str, Any], str]:
    """GET document, returning (doc_json, collection_id_used).

    On 404 in the primary collection, retry against every known
    outfits collection env var. Logs which collection actually held
    the document so future deploys can confirm the env is right.
    """
    import logging
    log = logging.getLogger("ahvi.wardrobe_persistence")

    if not _appwrite_ready():
        raise RuntimeError("Appwrite wardrobe persistence is not configured.")

    last_body = ""
    last_status = 0
    for collection_id in _KNOWN_COLLECTIONS or [APPWRITE_COLLECTION_ID]:
        if not collection_id:
            continue
        url = (
            f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}"
            f"/collections/{collection_id}/documents/{document_id}"
        )
        log.info(
            "ahvi.fetch_doc db=%s col=%s id=%s",
            APPWRITE_DATABASE_ID, collection_id, document_id,
        )
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code in (200, 201):
            if collection_id != APPWRITE_COLLECTION_ID:
                log.warning(
                    "ahvi.fetch_doc.fallback_hit primary=%s actual=%s id=%s",
                    APPWRITE_COLLECTION_ID, collection_id, document_id,
                )
            return res.json(), collection_id
        last_status = res.status_code
        last_body = str(res.text or "")[:300]
        if res.status_code == 404:
            log.warning(
                "ahvi.fetch_doc.not_found db=%s col=%s id=%s body=%s",
                APPWRITE_DATABASE_ID, collection_id, document_id, last_body,
            )
            continue
        raise RuntimeError(
            f"Appwrite fetch error: {res.status_code} {res.text}"
        )

    raise LookupError(
        f"Wardrobe item not found in any known collection: id={document_id} "
        f"tried={_KNOWN_COLLECTIONS} last_status={last_status} body={last_body}"
    )


def _patch_document(
    document_id: str,
    data: Dict[str, Any],
    collection_id: str = "",
) -> Tuple[Dict[str, Any], List[str]]:
    """PATCH a document, returning (response_json, dropped_keys).

    On 'Unknown attribute' the schema is missing a key. Rather than
    failing the whole update, retry without the offending key and tell
    the caller which keys were dropped so the user can be warned that
    the save was partial.

    collection_id defaults to the configured primary collection but
    callers (e.g. update_item_labels after _fetch_document fallback)
    can supply the collection the document actually lives in.
    """
    target_collection = collection_id or APPWRITE_COLLECTION_ID
    url = (
        f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}"
        f"/collections/{target_collection}/documents/{document_id}"
    )

    def post(payload: Dict[str, Any]):
        return requests.patch(
            url, headers=HEADERS, json={"data": payload}, timeout=20
        )

    dropped: List[str] = []
    attempt = dict(data)
    # Allow a small number of retries — Appwrite's error message reports
    # ONE missing attribute at a time so we may need multiple passes.
    for _ in range(4):
        res = post(attempt)
        if res.status_code in (200, 201):
            return res.json(), dropped

        body_lower = str(res.text or "").lower()
        if "unknown attribute" not in body_lower and "invalid document structure" not in body_lower:
            raise RuntimeError(f"Appwrite update error: {res.status_code} {res.text}")

        import re as _re
        m = _re.search(r"unknown attribute[:\s]+\"?([\w_]+)", body_lower)
        before = dict(attempt)
        if m:
            key = m.group(1)
            attempt.pop(key, None)
            dropped.append(key)
        else:
            for k in ("occasions", "color_code", "sub_category", "material"):
                if attempt.pop(k, None) is not None:
                    dropped.append(k)
                    break
        if attempt == before or not attempt:
            raise RuntimeError(f"Appwrite update error: {res.status_code} {res.text}")

    # Exhausted retries.
    raise RuntimeError(
        f"Appwrite update exhausted retries after dropping {dropped}"
    )


def update_item_labels(
    *,
    user_id: str,
    item_id: str,
    name: Any = None,
    category: Any = None,
    subcategory: Any = None,
    color: Any = None,
    material: Any = None,
    tags: Any = None,
) -> Dict[str, Any]:
    """Owner-verified update of wardrobe item labels via Appwrite server key.

    Only writes attributes that exist in the outfits collection schema:
      name, category, sub_category, color_code, occasions

    Unknown frontend keys (tags, material) are mapped or dropped so the
    Appwrite PATCH never fails with 'Unknown attribute'.
    """
    import logging
    log = logging.getLogger("ahvi.wardrobe_persistence")

    user_id = _safe_text(user_id)
    item_id = _safe_text(item_id)
    if not user_id:
        raise PermissionError("Missing user_id.")
    if not item_id:
        raise ValueError("Missing item_id.")

    existing, source_collection = _fetch_document(item_id)
    # Schema uses userId (camelCase) — accept user_id too for legacy docs.
    owner = _safe_text(existing.get("userId") or existing.get("user_id"))
    if not owner:
        # Missing owner: protect by refusing, otherwise anyone could edit.
        raise PermissionError("Wardrobe item has no owner. Refusing update.")
    if owner != user_id:
        log.warning(
            "ahvi.update_labels.forbidden item=%s owner=%s requester=%s",
            item_id, owner, user_id,
        )
        raise PermissionError("Item does not belong to user.")

    raw_category = category if category is not None else existing.get("category")
    raw_name = name if name is not None else existing.get("name")
    raw_sub = (
        subcategory
        if subcategory is not None
        else existing.get("sub_category") or existing.get("subcategory")
    )

    normalized_category = normalize_category(raw_category, raw_name, raw_sub)
    display_name, normalized_sub = normalize_display_name_and_subcategory(
        raw_name,
        raw_sub,
        normalized_category,
    )

    # Only write attributes the outfits collection actually has.
    patch: Dict[str, Any] = {
        "name": display_name,
        "category": normalized_category,
        "sub_category": normalized_sub,
    }
    if color is not None:
        # Schema uses color_code, not color.
        patch["color_code"] = _safe_text(color)
    # tags from the frontend edit dialog are the user's chosen occasions
    # — map them to the canonical occasions[] field.
    if tags is not None:
        patch["occasions"] = _normalize_list(tags)
    # material is not in the schema; ignore to avoid 'Unknown attribute'.

    log.info(
        "ahvi.update_labels user=%s item=%s collection=%s category=%s sub=%s",
        user_id, item_id, source_collection, normalized_category, normalized_sub,
    )

    try:
        updated, dropped_keys = _patch_document(item_id, patch, source_collection)
    except RuntimeError as exc:
        log.error("ahvi.update_labels.patch_failed item=%s err=%s", item_id, exc)
        raise

    if dropped_keys:
        log.warning(
            "ahvi.update_labels.partial item=%s dropped=%s",
            item_id, dropped_keys,
        )

    return {
        "success": True,
        "item": updated,
        # Partial-save warning so the frontend can show a toast if user
        # cares (e.g. 'Saved, but tags weren't stored — collection
        # schema is missing the field').
        "dropped_keys": dropped_keys,
        "partial": bool(dropped_keys),
    }


__all__ = [
    "persist_selected_items",
    "normalize_category",
    "normalize_display_name_and_subcategory",
    "update_item_labels",
]
