import base64
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError
from services.r2_storage import R2Storage, R2StorageError
from services.image_normalizer import normalize_style_board_image_bytes


# =========================
# HELPERS
# =========================
def clean_occasion(raw: str) -> str:
    v = (raw or "").strip().lower()
    mapping = {
        "party looks": "Party",
        "party": "Party",
        "office fit": "Office",
        "office": "Office",
        "vacation": "Vacation",
        "occasion": "Occasion",
    }
    return mapping.get(v, (raw or "Occasion").strip().title())


def decode_image_base64(value: str) -> tuple[bytes, str]:
    text = (value or "").strip()
    if not text:
        return b"", "png"

    extension = "png"

    # detect data URI
    if text.startswith("data:image/"):
        match = re.match(r"^data:image/([a-zA-Z0-9]+);base64,", text)
        if match:
            extension = match.group(1).lower()
        text = text.split(",", 1)[1] if "," in text else text

    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")

    if not data:
        raise HTTPException(status_code=400, detail="image_base64 is empty")

    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image_base64 too large (max 12MB)")

    return data, extension


# =========================
# READ APIs
# =========================
def list_saved_boards(
    *, user_id: str, occasion: Optional[str] = None, limit: int = 100
):
    proxy = AppwriteProxy()
    return proxy.list_documents(
        "saved_boards",
        user_id=user_id,
        occasion=clean_occasion(occasion) if occasion else None,
        limit=limit,
    )


def list_life_boards(*, user_id: str, limit: int = 100):
    return AppwriteProxy().list_documents("life_boards", user_id=user_id, limit=limit)


# =========================
# SAVE STYLE BOARD
# =========================
def save_board(
    *,
    user_id: str,
    occasion: str,
    image_url: str = "",
    image_base64: str = "",
    board_ids: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
):
    proxy = AppwriteProxy()
    payload = payload or {}

    final_image_url = (image_url or "").strip()

    # Upload if base64 present.
    # Long-term AHVI rule: saved board covers are normalized into a
    # consistent portrait preview before R2 upload so Planner -> Boards
    # thumbnails do not appear cropped/uneven across devices.
    if str(image_base64 or "").strip():
        image_bytes, _extension = decode_image_base64(image_base64)

        try:
            normalized_bytes = normalize_style_board_image_bytes(
                image_bytes,
                size=(1080, 1350),
                output_format="JPEG",
                quality=92,
            )
            storage = R2Storage()
            uploaded = storage.upload_style_board_image(
                user_id=user_id,
                image_bytes=normalized_bytes,
                extension="jpg",
            )
            final_image_url = uploaded.get("image_url", final_image_url)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Image upload failed: {exc}")

    # =========================
    # ITEM IDS EXTRACTION
    # =========================
    item_ids: list[str] = []

    if board_ids:
        item_ids = [x.strip() for x in board_ids.split(",") if x.strip()]
    else:
        raw_item_ids = payload.get("itemIds") or payload.get("boardIds") or []
        if isinstance(raw_item_ids, list):
            item_ids = [str(x).strip() for x in raw_item_ids if str(x).strip()]

    # =========================
    # ðŸ”¥ ELITE STRUCTURED BOARD
    # =========================
    # Demo-safe schema: Appwrite saved_boards currently supports only:
    # userId, imageUrl, itemIds, occasion.
    # Do not send aesthetic/vibe/colorStory/layout/items/styleScore/createdAt/updatedAt
    # unless those attributes are added to the Appwrite collection.
    doc = {
        "userId": user_id,
        "occasion": clean_occasion(occasion),
        "imageUrl": final_image_url,
        "itemIds": item_ids,
    }
    return proxy.create_document("saved_boards", doc)


# =========================
# SAVE LIFE BOARD
# =========================
def save_life_board(
    *,
    user_id: str,
    title: str,
    board_type: str,
    description: str,
    payload: Dict[str, Any],
):
    now_iso = datetime.now(timezone.utc).isoformat()

    doc = {
        "userId": user_id,
        "title": (title or "").strip() or "Life Board",
        "boardType": (board_type or "").strip() or "daily_wear",
        "description": (description or "").strip(),
        "payload": payload or {},
        "createdAt": now_iso,
        "updatedAt": now_iso,
    }

    return AppwriteProxy().create_document("life_boards", doc)


# =========================
# DELETE
# =========================
def delete_saved_board(*, document_id: str, user_id: str = ""):
    """Delete a saved board after verifying the requester owns it.

    user_id is the authenticated user's id. When supplied, the document's
    `userId` field must match. Empty user_id skips the check and is only
    safe for trusted callers that have already verified ownership.
    """
    proxy = AppwriteProxy()
    if user_id:
        try:
            doc = proxy.get_document("saved_boards", document_id)
        except AppwriteProxyError:
            raise
        owner = str(doc.get("userId") or doc.get("user_id") or "").strip()
        if owner and owner != user_id:
            raise AppwriteProxyError("Forbidden")
    proxy.delete_document("saved_boards", document_id)


__all__ = [
    "AppwriteProxyError",
    "R2StorageError",
    "clean_occasion",
    "list_saved_boards",
    "save_board",
    "list_life_boards",
    "save_life_board",
    "delete_saved_board",
]
