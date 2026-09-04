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
from urllib.parse import urlsplit

from services.style_item_contract import normalize_style_item

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
#
# transparent_image_url is unconditional here to match
# lib/util/wardrobe_image_resolver.dart's style-asset branch - the only
# place anything in this codebase currently writes it (see
# style_reasoning_engine._normalize_style_asset), and this module also
# gates the ConstrainedOutfitBuilder's mixed wardrobe/style-asset pool.
# ponytail: no wardrobe writer sets this field today, so the resolver's
# wardrobe branch's extra board_status/cutout_status OR-gate for
# transparent_image_url is unenforced here - add it if a wardrobe write
# path ever starts populating this field.
_CANDIDATE_FIELDS: tuple[tuple[str, Optional[tuple[str, str]]], ...] = (
    ("masked_url", None),
    ("maskedUrl", None),
    ("transparent_url", None),
    ("transparentUrl", None),
    ("transparent_image_url", None),
    ("transparentImageUrl", None),
    ("cutout_url", ("cutout_status", "ready")),
    ("cutoutUrl", ("cutout_status", "ready")),
    ("board_image_url", ("board_status", "cutout_ready")),
    ("boardImageUrl", ("board_status", "cutout_ready")),
    ("rmbg_url", ("image_status", "rmbg_complete")),
    ("rmbgUrl", ("image_status", "rmbg_complete")),
    ("processed_url", ("image_status", "rmbg_complete")),
    ("processedUrl", ("image_status", "rmbg_complete")),
)

# normalized_url is a lower-priority, unconditional catalog-tier candidate -
# a framed product/catalog shot, not a transparent cutout. Kept and
# renderable, but never earns "cutout_ready" status.
_CATALOG_FIELDS = ("normalized_url", "normalizedUrl")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _raw_aliases(item: Dict[str, Any]) -> set:
    return {_text(item.get(k)) for k in _RAW_ALIAS_KEYS if _text(item.get(k))}


def _board_url_identity(url: Any) -> str:
    """Mirror lib/util/wardrobe_image_resolver.dart's _urlIdentity(): two URLs
    naming the "same" underlying resource for alias-detection purposes,
    regardless of query string / fragment / scheme-host casing.

    A masked_url that is a signed/cache-busted copy of image_url (same host
    + path, different query token) is exact-string-different but IS the same
    fabricated-cutout alias Flutter itself will reject - admitting it here
    only for Flutter to hang an empty placeholder on it. Mirrors the Dart
    function line for line: parse as a URI; if it has no scheme or no host,
    fall back to the raw string (Flutter does the same); otherwise identity
    is scheme(lower) + "://" + host(lower) + explicit-port + path, with the
    query and fragment always dropped.
    """
    text = _text(url)
    if not text:
        return text
    parts = urlsplit(text)
    if not parts.scheme or not parts.hostname:
        return text
    port = f":{parts.port}" if parts.port is not None else ""
    return f"{parts.scheme.lower()}://{parts.hostname}{port}{parts.path}"


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
    alias_identities = {_board_url_identity(a) for a in aliases}

    for field, status_req in _CANDIDATE_FIELDS:
        value = _text(item.get(field))
        if not value or value in aliases or _board_url_identity(value) in alias_identities:
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
        # normalized_url is a lower-priority, unconditional catalog-tier
        # candidate - a framed product/catalog shot, not a transparent
        # cutout - but it MUST still be alias-checked against known raw
        # fields. lib/util/wardrobe_image_resolver.dart's frontend fix
        # (commit e8a53d6) closed exactly this gap: a wardrobe item whose
        # backend record never got a genuinely distinct processed asset can
        # have normalized_url aliased to the same raw upload as image_url,
        # which must never be classified as board-safe.
        if not value or value in aliases or _board_url_identity(value) in alias_identities:
            continue
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


# Canonical (snake_case) field name for each candidate, and the (status_key,
# expected_value) pair to carry alongside it when that candidate is
# status-gated. Keeps project_board_image_fields()'s output aligned with
# _CANDIDATE_FIELDS/_CATALOG_FIELDS above without duplicating the matching
# logic - see resolve_board_image_candidate for the actual selection.
#
# transparent_url/transparentUrl canonicalize to masked_url, not to
# themselves: wardrobe_persistence_service.persist_selected_items already
# treats transparent_url as a masked_url alias on the write side, and
# lib/util/wardrobe_image_resolver.dart's wardrobe (non-style-asset) branch
# never reads a bare transparent_url field - only masked_url. Projecting it
# under its own name would make a genuinely renderable item silently stop
# rendering once it reaches Flutter.
_CANONICAL_FIELD_NAMES = {
    "maskedUrl": "masked_url",
    "cutoutUrl": "cutout_url",
    "boardImageUrl": "board_image_url",
    "normalizedUrl": "normalized_url",
    "rmbgUrl": "rmbg_url",
    "processedUrl": "processed_url",
    "transparentImageUrl": "transparent_image_url",
    "transparent_url": "masked_url",
    "transparentUrl": "masked_url",
}

_STATUS_BY_CANONICAL_FIELD = {
    "cutout_url": ("cutout_status", "ready"),
    "board_image_url": ("board_status", "cutout_ready"),
    "rmbg_url": ("image_status", "rmbg_complete"),
    "processed_url": ("image_status", "rmbg_complete"),
}


def project_board_image_fields(item: Any) -> Dict[str, Any]:
    """Return the winning board-safe field(s) for `item`, in canonical
    (snake_case) form plus any required gating status - e.g.
    {"cutout_url": "...", "cutout_status": "ready"} - or {} if `item` has no
    board-safe candidate.

    Unlike resolve_board_image_candidate (which returns a flattened
    "selected_url" for callers that just want *a* URL), this is for board
    serializers that need to carry the winning candidate through as its own
    field(s) so a later is_board_renderable() check on the projected output
    still finds it - collapsing everything into a single image_url loses the
    field name is_board_renderable actually looks for, and silently produces
    an empty hanger even though the source item was genuinely renderable.
    """
    candidate = resolve_board_image_candidate(item)
    if not candidate["renderable"]:
        return {}
    field = _CANONICAL_FIELD_NAMES.get(candidate["selected_field"], candidate["selected_field"])
    result: Dict[str, Any] = {field: candidate["selected_url"]}
    status_pair = _STATUS_BY_CANONICAL_FIELD.get(field)
    if status_pair:
        status_key, status_value = status_pair
        result[status_key] = status_value
    return result


def prepare_board_item(raw: Any) -> Dict[str, Any]:
    """Produce the exact item representation a board is allowed to serialize
    for `raw` - the one object that must be both is_board_renderable() and
    Flutter-resolvable, with no further transformation between this call and
    the wire.

    normalize_style_item() falls back to masked_url/normalized_url/etc (via
    canonical_image_url()'s display-priority chain) to fill `image_url` when
    a raw item has no explicit one - a reasonable "give me *something* to
    show" default for a wardrobe grid thumbnail, but fatal for a board: this
    function's caller then projects the winning candidate field back onto
    the item under its own name (project_board_image_fields), so an item
    with only masked_url ends up served as image_url == masked_url - a
    fabricated raw-image alias resolve_board_image_candidate() never sees,
    because it only ever validated the untouched raw dict, before this
    synthesis happened. Live device bug (device gate on ce4ade1): a
    masked-only wardrobe item passed is_board_renderable(raw) - correctly,
    it has no real alias - then normalize_style_item() manufactured
    image_url=masked_url, and Flutter's own _urlIdentity() alias check
    rejected the now-self-aliased item, rendering an empty hanger.

    Style-asset items are returned untouched (590cc2e): Flutter's asset
    branch reads transparent_url/transparentImageUrl under their own names
    (never masked_url), and normalize_style_item() doesn't carry every
    camelCase candidate alias through - callers that need to gate a
    style_asset item's renderability must still check the raw dict, not this
    function's output, to avoid losing camelCase-only candidates.
    """
    norm = normalize_style_item(raw)
    if norm.get("source") == "style_asset":
        return norm
    raw_dict = raw if isinstance(raw, dict) else {}
    has_explicit_original = bool(_text(raw_dict.get("image_url")) or _text(raw_dict.get("imageUrl")))
    if not has_explicit_original:
        norm.pop("image_url", None)
    norm.update(project_board_image_fields(raw))
    return norm


__all__ = [
    "is_board_renderable",
    "resolve_board_image_candidate",
    "project_board_image_fields",
    "prepare_board_item",
]
