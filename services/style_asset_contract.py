from __future__ import annotations

"""Canonical Style Asset board-image contract.

The style_assets collection currently uses:
- board_image_url: board-ready PNG/cutout
- catalog_image_url: opaque catalogue JPG fallback
- normalized_url: processed fallback
- image_url: original source fallback

cutout_url is supported but is not required for current production assets.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional


def _text(value: Any) -> str:
    return str(value or "").strip()


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _text(row.get(key))
        if value:
            return value
    return ""


_RAW_IMAGE_FIELDS = (
    "image_url",
    "imageUrl",
    "raw_url",
    "rawUrl",
    "url",
    "asset_url",
    "assetUrl",
    "asset_path",
    "assetPath",
    "original_image_url",
    "originalImageUrl",
    "preview_url",
    "previewUrl",
    "original_upload_url",
    "originalUploadUrl",
    "upload_url",
    "uploadUrl",
)

def _raw_image_aliases(row: Mapping[str, Any]) -> set[str]:
    return {
        value
        for field in _RAW_IMAGE_FIELDS
        if (value := _text(row.get(field)))
    }


def _is_style_asset_row(row: Mapping[str, Any]) -> bool:
    return (
        _first_text(row, "source").lower() == "style_asset"
        or any(
            _text(row.get(key))
            for key in (
                "cutout_url",
                "cutoutUrl",
                "board_image_url",
                "boardImageUrl",
                "catalog_image_url",
                "catalogImageUrl",
                "cutout_status",
                "cutoutStatus",
            )
        )
    )


def _cutout_status_ready(row: Mapping[str, Any]) -> bool:
    return _first_text(row, "cutout_status", "cutoutStatus").lower() == "ready"


def resolve_style_asset_image(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Select an image according to the persisted wardrobe/style-assets fields."""
    raw_aliases = _raw_image_aliases(row)
    style_asset = _is_style_asset_row(row)
    catalog_url = _first_text(row, "catalog_image_url", "catalogImageUrl")

    candidates: List[tuple[str, str, bool]] = []
    if style_asset:
        # Style assets can pass through this resolver more than once (e.g. a
        # visual-inspiration direction is resolved once via
        # _apply_board_image_fields, then re-resolved when the itemized board
        # contract is built). By the second pass, image_url/imageUrl may
        # already hold the SELECTED url from the first pass, not the true raw
        # source -- so checking cutout/board candidates against the full
        # image_url-derived raw_aliases set produces a false-positive
        # self-alias and drops the genuine cutout in favor of the actual raw
        # catalog_image_url. catalog_image_url is never overwritten by this
        # resolver (it always reflects the literal persisted field), so it is
        # the one stable raw reference to check style-asset candidates
        # against, independent of how many times this function has already
        # run on the row.
        style_asset_raw_aliases = {catalog_url} if catalog_url else set()

        cutout_url = _first_text(row, "cutout_url", "cutoutUrl")
        if cutout_url and cutout_url not in style_asset_raw_aliases and _cutout_status_ready(row):
            candidates.append(("cutout_url", cutout_url, True))

        board_url = _first_text(row, "board_image_url", "boardImageUrl")
        if board_url and board_url not in style_asset_raw_aliases and _cutout_status_ready(row):
            candidates.append(("board_image_url", board_url, True))

        candidates.extend(
            [
                    ("normalized_url", _first_text(row, "normalized_url", "normalizedUrl"), False),
                    ("catalog_image_url", catalog_url, False),
                ("image_url", _first_text(row, "image_url", "imageUrl"), False),
            ]
        )
    else:
        masked_url = _first_text(row, "masked_url", "maskedUrl")
        if masked_url and masked_url not in raw_aliases:
            candidates.append(("masked_url", masked_url, True))
        candidates.extend(
            [
                ("normalized_url", _first_text(row, "normalized_url", "normalizedUrl"), False),
                ("image_url", _first_text(row, "image_url", "imageUrl"), False),
            ]
        )

    selected_field = ""
    selected_url = ""
    expected_transparent = False
    for field, value, transparent in candidates:
        if value:
            selected_field = field
            selected_url = value
            expected_transparent = transparent
            break

    prefix = "style_asset" if style_asset else "wardrobe"
    if expected_transparent:
        source_kind = f"{prefix}_cutout"
    elif selected_field == "normalized_url":
        source_kind = f"{prefix}_processed"
    elif selected_field == "catalog_image_url":
        source_kind = "catalog_fallback"
    elif selected_field == "image_url":
        source_kind = f"{prefix}_original"
    else:
        source_kind = ""

    return {
        "selected_field": selected_field,
        "selected_url": selected_url,
        "source_kind": source_kind,
        "expected_transparent": expected_transparent,
        "requires_frame": bool(selected_url) and not expected_transparent,
        "board_image_url": selected_url if expected_transparent else "",
        "catalog_image_url": catalog_url,
    }


def _stable_asset_id(row: Mapping[str, Any]) -> str:
    return _first_text(
        row,
        "asset_id",
        "item_id",
        "$id",
        "id",
        "assetId",
        "itemId",
    )


def _url_values(row: Mapping[str, Any]) -> List[str]:
    values: List[str] = []

    for key in (
        "board_image_url",
        "boardImageUrl",
        "cutout_url",
        "cutoutUrl",
        "masked_url",
        "maskedUrl",
        "catalog_image_url",
        "catalogImageUrl",
        "normalized_url",
        "normalizedUrl",
        "image_url",
        "imageUrl",
    ):
        value = _text(row.get(key))
        if value and value not in values:
            values.append(value)

    # Some legacy generated items use "<name>::<url>" as item_id.
    synthetic_id = _first_text(row, "item_id", "id")
    if "::" in synthetic_id:
        suffix = synthetic_id.rsplit("::", 1)[-1].strip()
        if suffix.startswith(("http://", "https://")) and suffix not in values:
            values.append(suffix)

    return values


def _non_blank_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(base)

    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, dict)) and not value:
            continue
        merged[key] = value

    return merged


def adapt_style_asset(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return one full, canonical Style Asset board item."""

    asset = dict(row)
    asset_id = _stable_asset_id(asset)

    board_url = _first_text(
        asset,
        "board_image_url",
        "boardImageUrl",
    )
    cutout_url = _first_text(
        asset,
        "cutout_url",
        "cutoutUrl",
    )
    catalog_url = _first_text(
        asset,
        "catalog_image_url",
        "catalogImageUrl",
    )
    normalized_url = _first_text(
        asset,
        "normalized_url",
        "normalizedUrl",
    )
    original_url = _first_text(
        asset,
        "image_url",
        "imageUrl",
    )

    role = _first_text(
        asset,
        "role",
        "slot",
        "category",
        "sub_category",
        "subcategory",
    )

    if asset_id:
        asset["item_id"] = asset_id
        asset["asset_id"] = asset_id

    asset["source"] = "style_asset"

    if role:
        asset.setdefault("role", role)
        asset.setdefault("slot", role)

    # Preserve every image candidate. Never replace the original image_url
    # with the board cutout.
    if board_url:
        asset["board_image_url"] = board_url
    if cutout_url:
        asset["cutout_url"] = cutout_url
    if catalog_url:
        asset["catalog_image_url"] = catalog_url
    if normalized_url:
        asset["normalized_url"] = normalized_url
    if original_url:
        asset["image_url"] = original_url

    resolved = resolve_style_asset_image(asset)
    if resolved["expected_transparent"]:
        asset["board_image_url"] = resolved["selected_url"]
    else:
        # A bare board URL is a candidate, not proof of a transparent image.
        asset.pop("board_image_url", None)
        asset.pop("boardImageUrl", None)
    if resolved["selected_field"]:
        asset["selected_field"] = resolved["selected_field"]
        asset["source_kind"] = resolved["source_kind"]
        asset["expected_transparent"] = resolved["expected_transparent"]
        asset["requires_frame"] = resolved["requires_frame"]

    return {
        key: value
        for key, value in asset.items()
        if value is not None and value != ""
    }


def adapt_style_asset_rows(
    rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        adapt_style_asset(row)
        for row in rows
        if isinstance(row, Mapping)
    ]


def enrich_style_asset_rows(
    selected_rows: Iterable[Mapping[str, Any]],
    *,
    inventory: Optional[Iterable[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Recover full Appwrite Style Asset fields for generated selections.

    Selection/reasoning code sometimes returns only name, role and image_url,
    or uses "<name>::<url>" as the item ID. This method matches those rows
    against the complete style_assets inventory and restores asset_id,
    board_image_url, catalog_image_url and all other metadata.
    """

    inventory_rows = [
        dict(row)
        for row in (inventory or [])
        if isinstance(row, Mapping)
    ]

    by_id: Dict[str, Dict[str, Any]] = {}
    by_url: Dict[str, Dict[str, Any]] = {}

    for row in inventory_rows:
        stable_id = _stable_asset_id(row)
        if stable_id:
            by_id[stable_id] = row

        for url in _url_values(row):
            by_url[url] = row

    enriched: List[Dict[str, Any]] = []

    for selected in selected_rows:
        if not isinstance(selected, Mapping):
            continue

        selected_dict = dict(selected)
        matched: Optional[Dict[str, Any]] = None

        for candidate_id in (
            _first_text(selected_dict, "asset_id"),
            _first_text(selected_dict, "item_id"),
            _first_text(selected_dict, "$id"),
            _first_text(selected_dict, "id"),
        ):
            if candidate_id and candidate_id in by_id:
                matched = by_id[candidate_id]
                break

        if matched is None:
            for url in _url_values(selected_dict):
                if url in by_url:
                    matched = by_url[url]
                    break

        merged = (
            _non_blank_merge(matched, selected_dict)
            if matched is not None
            else selected_dict
        )

        enriched.append(adapt_style_asset(merged))

    return enriched
