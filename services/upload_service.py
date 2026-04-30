import base64
import io
from fastapi import HTTPException
from PIL import Image
from services.r2_storage import R2Storage


# =========================
# CONFIG
# =========================
MAX_AVATAR_SIZE = 8 * 1024 * 1024
MAX_WARDROBE_SIZE = 12 * 1024 * 1024

ALLOWED_TYPES = {"jpeg", "png", "webp"}


# =========================
# VALIDATION
# =========================
def _validate_image(image_bytes: bytes, max_size: int, field_name: str):
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")

    if len(image_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} too large (max {max_size // (1024 * 1024)}MB)"
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img_type = str((img.format or "")).lower()
    except Exception:
        img_type = ""

    # PIL returns "jpeg" for jpg
    if img_type == "jpg":
        img_type = "jpeg"

    if img_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be jpeg/png/webp (got {img_type})"
        )


def _base64_to_bytes(value: str, field_name: str) -> bytes:
    if not value:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")

    text = value.strip()
    if "," in text:
        text = text.split(",", 1)[1]

    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} invalid base64: {exc}"
        )


# =========================
# CORE (BYTES FIRST)
# =========================
def upload_avatar_bytes(*, user_id: str, image_bytes: bytes) -> str:
    _validate_image(image_bytes, MAX_AVATAR_SIZE, "image_bytes")

    return R2Storage().upload_avatar(
        user_id=user_id,
        image_bytes=image_bytes,
    )


def upload_wardrobe_images_bytes(
    *,
    file_id: str,
    raw_image_bytes: bytes,
    masked_image_bytes: bytes,
):
    _validate_image(raw_image_bytes, MAX_WARDROBE_SIZE, "raw_image_bytes")
    _validate_image(masked_image_bytes, MAX_WARDROBE_SIZE, "masked_image_bytes")

    return R2Storage().upload_wardrobe_images(
        file_id=file_id,
        raw_image_bytes=raw_image_bytes,
        masked_image_bytes=masked_image_bytes,
    )


# =========================
# ADAPTER (BASE64 → BYTES)
# =========================
def upload_avatar(*, user_id: str, image_base64: str) -> str:
    image_bytes = _base64_to_bytes(image_base64, "image_base64")
    return upload_avatar_bytes(user_id=user_id, image_bytes=image_bytes)


def upload_wardrobe_images(
    *,
    file_id: str,
    raw_image_base64: str,
    masked_image_base64: str,
):
    raw_bytes = _base64_to_bytes(raw_image_base64, "raw_image_base64")
    masked_bytes = _base64_to_bytes(masked_image_base64, "masked_image_base64")

    return upload_wardrobe_images_bytes(
        file_id=file_id,
        raw_image_bytes=raw_bytes,
        masked_image_bytes=masked_bytes,
    )
