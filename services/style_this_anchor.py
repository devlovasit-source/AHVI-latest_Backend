"""Canonical identity and safe-image contract for Style This anchors."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from services.style_asset_contract import adapt_style_asset
from services.style_item_contract import (
    aliases_raw_upload,
    canonical_accessory_type,
    canonical_item_id,
    canonical_item_role,
    canonical_item_source,
)


_SAFE_IMAGE_FIELDS = (
    "board_image_url",
    "boardImageUrl",
    "cutout_url",
    "cutoutUrl",
    "catalog_image_url",
    "catalogImageUrl",
    "normalized_url",
    "normalizedUrl",
    "masked_url",
    "maskedUrl",
    "image_url",
    "imageUrl",
)
_RAW_IMAGE_FIELDS = (
    "raw_url",
    "raw_image_url",
    "original_upload_url",
    "upload_url",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def canonical_style_this_anchor(
    item: Mapping[str, Any] | None,
    *,
    expected_item_id: str = "",
    allow_missing_image: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return one authoritative, locked Style This anchor or ``None``.

    The caller must provide an authoritative wardrobe/style-asset row. This
    helper never resolves identity from a name, role, category, or image URL.
    """
    if not isinstance(item, Mapping):
        return None
    source_item = dict(item)
    item_id = canonical_item_id(source_item)
    expected_id = _text(expected_item_id)
    if not item_id or (expected_id and item_id != expected_id):
        return None

    source = canonical_item_source(source_item)
    if source not in {"wardrobe", "style_asset"}:
        return None
    if source == "style_asset":
        source_item = adapt_style_asset(source_item)

    safe_field = ""
    safe_image_url = ""
    for field in _SAFE_IMAGE_FIELDS:
        value = _text(source_item.get(field))
        if not value:
            continue
        # A masked/normalized field identical to this item's own raw upload
        # is a fabricated cutout (RMBG never ran) -- never let it count as
        # the anchor's safe image; fall through to the next field / failure.
        if field.lower() in ("masked_url", "maskedurl", "normalized_url", "normalizedurl") and aliases_raw_upload(
            value, source_item
        ):
            continue
        safe_field = field
        safe_image_url = value
        break
    if not safe_image_url:
        # A row containing only an upload/raw marker is not board-safe.
        if any(_text(source_item.get(field)) for field in _RAW_IMAGE_FIELDS):
            return None
        if not allow_missing_image:
            return None

    role = canonical_item_role(source_item)
    if role == "unknown":
        return None

    if safe_field in {"board_image_url", "boardImageUrl", "cutout_url", "cutoutUrl"}:
        default_source_kind = "style_asset_cutout" if source == "style_asset" else "wardrobe_cutout"
        default_transparent = True
    elif safe_field in {"catalog_image_url", "catalogImageUrl"}:
        default_source_kind = "catalog_fallback"
        default_transparent = False
    elif safe_field in {"normalized_url", "normalizedUrl"}:
        default_source_kind = "style_asset_processed" if source == "style_asset" else "wardrobe_processed"
        default_transparent = False
    elif safe_field in {"masked_url", "maskedUrl"}:
        default_source_kind = "wardrobe_masked"
        default_transparent = True
    else:
        default_source_kind = "style_asset_original" if source == "style_asset" else "wardrobe_original"
        default_transparent = False

    anchor = dict(source_item)
    anchor.pop("raw_url", None)
    anchor.pop("raw_image_url", None)
    anchor.pop("original_upload_url", None)
    anchor.pop("upload_url", None)
    anchor.update(
        {
            "item_id": item_id,
            "id": item_id,
            "role": role,
            "slot": role,
            "source": source,
            "safe_image_url": safe_image_url,
            "image_url": safe_image_url,
            "source_kind": _text(source_item.get("source_kind")) or default_source_kind,
            "expected_transparent": (
                source_item.get("expected_transparent")
                if isinstance(source_item.get("expected_transparent"), bool)
                else default_transparent
            ),
            "anchor_item_id": item_id,
            "anchor": True,
            "locked": True,
        }
    )
    if role == "accessory":
        accessory_type = canonical_accessory_type(source_item)
        if accessory_type:
            anchor["accessory_type"] = accessory_type
            anchor["board_role"] = accessory_type
    return anchor


__all__ = ["canonical_style_this_anchor"]
