"""Single source of truth for whether an item has a genuine, board-safe
processed image - i.e. whether Style This / Shuffle may select it, and
whether the Flutter board-surface renderer (lib/util/wardrobe_image_resolver.dart)
will actually show something other than an empty placeholder for it.

A raw photo (image_url/raw_url/url) is never board-safe on its own - it may
be a selfie or mirror photo. A masked/cutout field that merely aliases the
raw photo (the `masked_url = image_url` healing fallback some write paths
apply when RMBG produced nothing) is fabricated provenance, not a real
cutout, and must be rejected the same way.

This module is the ONE place that answers "is this item's image safe to put
on a Style Board." services.style_flow_service._adapt_board_item and
services.constrained_outfit_builder both defer to it rather than keeping
independent rules.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Raw-photo fields. A processed-image field whose value equals one of these
# is a fabricated cutout, never a real one.
_RAW_ALIAS_KEYS = (
    "image_url",
    "imageUrl",
    "raw_url",
    "rawUrl",
    "url",
    "original_image_url",
    "originalImageUrl",
    "preview_url",
    "previewUrl",
)

# Candidate processed-image fields in priority order. `status` is an
# optional (field, expected_value) pair - when present, the field is only
# admitted if the item's status field matches (mirrors the Flutter resolver's
# cutout_status/image_status gating for these specific fields). Fields with
# status=None are admitted unconditionally once non-empty and non-aliased,
# matching both _adapt_board_item and the Flutter resolver's unconditional
# masked_url/transparent_url path.
_CANDIDATE_FIELDS: tuple[tuple[str, Optional[tuple[str, str]]], ...] = (
    ("masked_url", None),
    ("maskedUrl", None),
    ("transparent_url", None),
    ("transparent_image_url", None),
    ("cutout_url", ("cutout_status", "ready")),
    ("cutoutUrl", ("cutout_status", "ready")),
    ("board_image_url", ("board_status", "cutout_ready")),
    ("boardImageUrl", ("board_status", "cutout_ready")),
    ("rmbg_url", ("image_status", "rmbg_complete")),
    ("processed_url", ("image_status", "rmbg_complete")),
)

# normalized_url is a lower-priority, unconditional catalog-tier candidate -
# a framed product/catalog shot, not a transparent cutout. Kept and
# renderable, but never earns "cutout_ready" status.
_CATALOG_FIELDS = ("normalized_url", "normalizedUrl")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _raw_aliases(item: Dict[str, Any]) -> set:
    return {_text(item.get(k)) for k in _RAW_ALIAS_KEYS if _text(item.get(k))}


def resolve_board_image_candidate(item: Any) -> Dict[str, Any]:
    """Return the board-safe image candidate (if any) for `item`.

    {
      "renderable": bool,
      "selected_field": str | None,
      "selected_url": str | None,
      "reason": str,
    }

    Never logs or returns anything beyond what the caller already passed in;
    callers must not log the URL in production diagnostics.
    """
    if not isinstance(item, dict):
        return {
            "renderable": False,
            "selected_field": None,
            "selected_url": None,
            "reason": "invalid_item",
        }

    aliases = _raw_aliases(item)

    for field, status_req in _CANDIDATE_FIELDS:
        value = _text(item.get(field))
        if not value or value in aliases:
            continue
        if status_req is not None:
            status_field, expected = status_req
            actual = _text(item.get(status_field)).lower()
            if actual != expected:
                continue
        return {
            "renderable": True,
            "selected_field": field,
            "selected_url": value,
            "reason": "processed_cutout",
        }

    for field in _CATALOG_FIELDS:
        value = _text(item.get(field))
        if value and value not in aliases:
            return {
                "renderable": True,
                "selected_field": field,
                "selected_url": value,
                "reason": "catalog_normalized",
            }

    return {
        "renderable": False,
        "selected_field": None,
        "selected_url": None,
        "reason": "no_board_safe_image",
    }


def is_board_renderable(item: Any) -> bool:
    """True when `item` has a genuine, board-safe processed image."""
    return resolve_board_image_candidate(item)["renderable"]


__all__ = ["is_board_renderable", "resolve_board_image_candidate"]
