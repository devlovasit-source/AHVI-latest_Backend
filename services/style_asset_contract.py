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

    # Freeze explicit image provenance for clients that understand it.
    if board_url:
        asset["selected_field"] = "board_image_url"
        asset["source_kind"] = "style_asset_cutout"
        asset["expected_transparent"] = True
        asset["requires_frame"] = False
    elif cutout_url:
        asset["selected_field"] = "cutout_url"
        asset["source_kind"] = "style_asset_cutout"
        asset["expected_transparent"] = True
        asset["requires_frame"] = False
    elif normalized_url:
        asset["selected_field"] = "normalized_url"
        asset["source_kind"] = "style_asset_processed"
        asset["expected_transparent"] = False
        asset["requires_frame"] = True
    elif catalog_url:
        asset["selected_field"] = "catalog_image_url"
        asset["source_kind"] = "catalog_fallback"
        asset["expected_transparent"] = False
        asset["requires_frame"] = True
    elif original_url:
        asset["selected_field"] = "image_url"
        asset["source_kind"] = "style_asset_original"
        asset["expected_transparent"] = False
        asset["requires_frame"] = True

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
