"""Canonical identity and safe-image contract for Style This anchors."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from services.style_asset_contract import adapt_style_asset
from services.style_board_image_readiness import resolve_board_image_candidate
from services.style_item_contract import (
    canonical_accessory_type,
    canonical_item_id,
    canonical_item_role,
    canonical_item_source,
)


_RAW_IMAGE_FIELDS = (
    "raw_url",
    "raw_image_url",
    "original_upload_url",
    "upload_url",
)

# Maps resolve_board_image_candidate's selected_field to this module's
# source_kind/expected_transparent semantics -- NOT a second priority list.
# The order candidates are considered in is decided entirely by
# services.style_board_image_readiness.resolve_board_image_candidate (the
# single authority for board image safety); this only classifies WHICH kind
# of field won, after that module already picked it.
_CUTOUT_FIELDS = {
    "board_image_url", "boardImageUrl", "cutout_url", "cutoutUrl",
    "transparent_url", "transparentUrl",
    "transparent_image_url", "transparentImageUrl",
    "rmbg_url", "rmbgUrl", "processed_url", "processedUrl",
}
_MASKED_FIELDS = {"masked_url", "maskedUrl"}
_CATALOG_ONLY_FIELDS = {"catalog_image_url", "catalogImageUrl"}
_NORMALIZED_FIELDS = {"normalized_url", "normalizedUrl"}


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

    # Single authority for board image safety: never re-decide priority or
    # alias-safety here. A field that aliases image_url/raw fields (e.g. a
    # legacy record where normalized_url == image_url) is never selected,
    # even if a genuinely distinct field like masked_url exists and would
    # otherwise be shadowed by a naive field-priority list.
    candidate = resolve_board_image_candidate(source_item)
    safe_field = candidate["selected_field"] if candidate["renderable"] else ""
    safe_image_url = candidate["selected_url"] if candidate["renderable"] else ""
    if not safe_image_url:
        # A row containing only an upload/raw marker is not board-safe.
        if any(_text(source_item.get(field)) for field in _RAW_IMAGE_FIELDS):
            return None
        if not allow_missing_image:
            return None

    role = canonical_item_role(source_item)
    if role == "unknown":
        return None

    if safe_field in _CUTOUT_FIELDS:
        default_source_kind = "style_asset_cutout" if source == "style_asset" else "wardrobe_cutout"
        default_transparent = True
    elif safe_field in _CATALOG_ONLY_FIELDS:
        default_source_kind = "catalog_fallback"
        default_transparent = False
    elif safe_field in _NORMALIZED_FIELDS:
        default_source_kind = "style_asset_processed" if source == "style_asset" else "wardrobe_processed"
        default_transparent = False
    elif safe_field in _MASKED_FIELDS:
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
            # Deliberately NOT "image_url": safe_image_url. image_url is a
            # provenance-bearing field (the raw/original upload, if one
            # exists) and must survive untouched -- `anchor = dict(source_item)`
            # above already carries it through unchanged. Overwriting it with
            # the winning presentation field (often normalized_url) made
            # image_url == normalized_url for any item without its own
            # distinct raw photo, which services.style_board_image_readiness's
            # raw-alias guard then correctly rejected as fabricated
            # provenance -- a false positive on a genuinely board-safe item.
            # safe_image_url is the caller-facing "best presentation" result;
            # routers/stylist.py already reads it via that dedicated field.
            "safe_image_url": safe_image_url,
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
