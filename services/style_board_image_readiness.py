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

import hashlib
import threading
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

import requests

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

# The only candidates subject to the byte-identity guard below - the class
# of bug that motivated it (RMBG no-op silently writing the raw photo back
# out under a masked_url-shaped path) is specific to this field; the other
# unconditional fields (transparent_url/transparentUrl/transparent_image_url)
# have no wardrobe writer today (see _CANDIDATE_FIELDS comment above) and
# widening the guard to them is unreviewed scope creep, not part of this fix.
_MASKED_FIELDS = frozenset({"masked_url", "maskedUrl"})

# normalized_url is a lower-priority, unconditional catalog-tier candidate -
# a framed product/catalog shot, not a transparent cutout. Kept and
# renderable, but never earns "cutout_ready" status.
_CATALOG_FIELDS = ("normalized_url", "normalizedUrl")


def _text(value: Any) -> str:
    return str(value or "").strip()


# Generic display fields that are raw provenance for an ordinary wardrobe row
# but NOT for a frozen board snapshot, where they hold the already-selected
# processed asset. Mirrors the isFrozenSnapshot carve-out in
# lib/util/wardrobe_image_resolver.dart.
_GENERIC_ALIAS_KEYS = ("image_url", "imageUrl")


def _is_frozen_snapshot(item: Dict[str, Any]) -> bool:
    """True when `item` is a board payload this contract already serialized.

    Exactly the client's rule (wardrobe_image_resolver.dart isFrozenSnapshot):
    selected_field AND source_kind AND some explicit raw provenance. On such an
    item image_url is the SELECTED asset, not the upload, so treating it as a
    raw alias would make the very field it was copied from look fabricated -
    and a second serialization pass would demote a real cutout to catalog.

    Safety is unchanged: the explicit raw fields stay in the alias set, so a
    snapshot whose "processed" asset genuinely is the upload is still rejected.
    """
    return bool(
        _text(item.get("selected_field"))
        and _text(item.get("source_kind"))
        and any(_text(item.get(k)) for k in _EXPLICIT_RAW_KEYS)
    )


def _raw_aliases(item: Dict[str, Any]) -> set:
    keys = _RAW_ALIAS_KEYS
    if _is_frozen_snapshot(item):
        keys = tuple(k for k in _RAW_ALIAS_KEYS if k not in _GENERIC_ALIAS_KEYS)
    return {_text(item.get(k)) for k in keys if _text(item.get(k))}


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


# --- masked_url byte-identity guard -----------------------------------
#
# Narrow hardening for a class of asset the URL-identity alias check above
# cannot see: a masked_url that lives at a genuinely distinct object path
# (different bucket/filename) but whose BYTES are the raw upload re-hosted
# verbatim - RMBG/masking silently no-opped and the pipeline still wrote it
# to masked_url as if it succeeded. Live example (P0 device evidence): a
# "Black Loafers" row whose masked_url and raw upload are SHA-256 identical
# despite living under different R2 prefixes.
#
# This does NOT become a population-wide quality gate: image_status/
# cutout_status - the only existing status signals other candidate fields
# already require - are unpopulated on effectively the entire wardrobe
# population that carries a masked_url (measured: 325/325 on the live P0
# audit), so requiring either would reclassify hundreds of genuinely fine
# items and blank dozens with no normalized_url fallback. Byte-identity is
# the only signal available that is both provably correct (not a guess)
# and scoped to the actual failure mode, so it is the whole of this fix -
# it will not catch a masked asset that is merely LOW QUALITY (e.g. a
# botched/noisy alpha matte) rather than a byte-for-byte raw copy. See the
# P0 forensic: the second known-bad item ("Black T-Shirt") is exactly that
# case and is NOT caught here - it needs actual reprocessing, not a smarter
# admission rule.
_BYTE_COMPARE_TIMEOUT_SECONDS = 3.0
_BYTE_COMPARE_MAX_BYTES = 8 * 1024 * 1024  # 8 MiB cap; oversize => inconclusive

# Per (masked_url, raw_url) pair. Only a DEFINITIVE comparison (both fetches
# succeeded) is ever stored - an inconclusive one (network error, timeout,
# oversize object) is never cached, so a transient failure can never
# permanently "bless" an unverified asset as safe; the next call retries.
_byte_identity_cache: Dict[Tuple[str, str], bool] = {}
_byte_identity_cache_lock = threading.Lock()


# Item-level denylist for masked assets proven unsafe by manual forensic
# inspection - content that genuinely differs from the raw upload (so the
# byte-identity guard above cannot catch it) but is itself unsafe, e.g. a
# botched mask that still shows the wearer's face/body rather than an
# isolated garment. This is NOT a quality heuristic and must never be
# inferred from name/category/state - each entry is a specific item ID with
# a documented, evidenced reason, added only after direct visual proof.
# Remove an ID once its masked_url has been reprocessed into a real cutout.
_KNOWN_UNSAFE_MASKED_ITEM_IDS = {
    # P0 device evidence ("Black T-Shirt"): masked_url is a cropped photo
    # showing the wearer's chin/beard/arm, not an isolated garment - bytes
    # differ from the raw upload, so RMBG ran but produced an unsafe result.
    # A safe normalized_url catalog cutout exists as the fallback.
    "6a96d6cf-9916-4594-a141-a3eba776700c",
}


def _item_id(item: Dict[str, Any]) -> str:
    for key in ("id", "$id", "item_id", "itemId"):
        value = _text(item.get(key))
        if value:
            return value
    return ""


def _fetch_bounded_hash(url: str) -> Optional[str]:
    """SHA-256 of `url`'s bytes, streamed and capped at
    _BYTE_COMPARE_MAX_BYTES. Returns None (never raises) on any network
    error, timeout, non-2xx response, or size overrun - None means
    "could not verify", NOT "differs"; callers must treat it as
    inconclusive, never as proof of safety.
    """
    try:
        with requests.get(
            url, stream=True, timeout=_BYTE_COMPARE_TIMEOUT_SECONDS
        ) as resp:
            resp.raise_for_status()
            hasher = hashlib.sha256()
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > _BYTE_COMPARE_MAX_BYTES:
                    return None
                hasher.update(chunk)
            return hasher.hexdigest()
    except Exception:
        return None


def _masked_asset_is_byte_identical_to_raw(masked_url: str, raw_candidates) -> bool:
    """True only when `masked_url` is PROVABLY the same bytes as one of
    `raw_candidates`. False covers both "provably different" and
    "could not verify" - this function only ever gives callers a reason to
    REJECT (True), never a reason to trust (a False here is not evidence of
    safety, it just means this specific check found no violation).

    Fetches `masked_url` at most once per call regardless of how many raw
    candidates it is compared against. Bounded by timeout/size in
    _fetch_bounded_hash; a network failure here degrades to "not proven
    identical" (fail open on availability) rather than rejecting or halting
    serialization - see module docstring on cache semantics for why that is
    still safe.
    """
    candidates = [_text(u) for u in raw_candidates if _text(u) and _text(u) != masked_url]
    if not masked_url or not candidates:
        return False

    masked_hash: Optional[str] = None
    masked_hash_attempted = False

    for raw_url in candidates:
        key = (masked_url, raw_url)
        with _byte_identity_cache_lock:
            cached = _byte_identity_cache.get(key)
        if cached is True:
            return True
        if cached is False:
            continue

        if not masked_hash_attempted:
            masked_hash_attempted = True
            masked_hash = _fetch_bounded_hash(masked_url)
        if masked_hash is None:
            continue  # inconclusive; not cached, may resolve on a later call

        raw_hash = _fetch_bounded_hash(raw_url)
        if raw_hash is None:
            continue  # inconclusive; not cached

        identical = masked_hash == raw_hash
        with _byte_identity_cache_lock:
            _byte_identity_cache[key] = identical
        if identical:
            return True

    return False


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
        if field in _MASKED_FIELDS:
            if _item_id(item) in _KNOWN_UNSAFE_MASKED_ITEM_IDS:
                # Proven-unsafe content for this exact item (see
                # _KNOWN_UNSAFE_MASKED_ITEM_IDS docstring) - fall through to
                # the next candidate (normalized/catalog, or no_board_safe_image).
                continue
            if _masked_asset_is_byte_identical_to_raw(value, aliases):
                # Distinct object path, but the same bytes as the item's own raw
                # provenance (see module docstring) - a fabricated cutout the
                # URL-identity check above cannot see. Never admitted, and never
                # re-badged as a different candidate; fall through to the next
                # field (normalized/catalog, or no_board_safe_image).
                continue
        return {
            "renderable": True,
            "selected_field": field,
            "selected_url": value,
            "reason": "processed_cutout",
        }

    for field in _CATALOG_FIELDS:
        value = _text(item.get(field))
        # Unconditional, unlike the cutout fields above: the Flutter
        # wardrobe_image_resolver's normalized_url candidates never
        # alias-check against image_url either (normalized_url is always a
        # regenerated catalog/product shot, never a raw upload copy, by the
        # contract those fields represent) - matching that here, not
        # inventing a stricter rule the frontend doesn't itself enforce.
        if value:
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


# Explicit raw-provenance fields. Unlike _RAW_ALIAS_KEYS this deliberately
# EXCLUDES the generic image_url: a privacy/catalog-only wardrobe row stores
# its catalog asset in image_url and has no raw upload at all, so presence of
# image_url alone is not evidence that a raw photo exists.
_EXPLICIT_RAW_KEYS = (
    "raw_url", "rawUrl",
    "original_image_url", "originalImageUrl",
    "preview_url", "previewUrl",
)

# source_kind values the Flutter resolver (lib/util/wardrobe_image_resolver.dart
# frozenValidated) accepts as genuinely board-safe cutout provenance.
_SOURCE_KIND_CUTOUT = "processed_cutout"
_SOURCE_KIND_CATALOG = "catalog_fallback"


def canonicalize_wardrobe_image_contract(record: Any) -> Dict[str, Any]:
    """The ONE read-side wardrobe image contract.

    Mirrors canonicalize_wardrobe_image_write() (services.wardrobe_persistence_
    service) on the read path, for records that were written before that gate
    existed. Returns:

      image_url            persisted legacy value, untouched (may be raw)
      masked_url           validated cutout field, or "" when absent/aliased
      normalized_url       validated catalog field, or "" when absent
      safe_image_url       board-safe presentation asset, or "" (never raw)
      safe_image_source    winning field name, "catalog", or "none"
      board_ready          safe_image_url != ""
      expected_transparent True only for real cutout provenance

    Selection is delegated to resolve_board_image_candidate() so this can
    never diverge from the admission gate Style This/Shuffle already use.
    """
    item = record if isinstance(record, dict) else {}
    candidate = resolve_board_image_candidate(item)
    renderable = bool(candidate["renderable"])
    selected_url = _text(candidate["selected_url"]) if renderable else ""
    is_cutout = renderable and candidate["reason"] == "processed_cutout"

    winning_field = _CANONICAL_FIELD_NAMES.get(
        candidate["selected_field"], candidate["selected_field"]
    ) if renderable else None

    has_explicit_raw = any(_text(item.get(k)) for k in _EXPLICIT_RAW_KEYS)
    has_masked = bool(_text(item.get("masked_url")) or _text(item.get("maskedUrl")))
    generic = _text(item.get("image_url") or item.get("imageUrl"))
    # Privacy/catalog-only: nothing raw and nothing masked was ever persisted,
    # so the catalog asset is the item's only image by design (CASE 7 of the
    # write contract). Determined from provenance, never from the filename -
    # a raw upload can legitimately be named catalog_*.png.
    #
    # The generic image_url must be absent or BE the winning catalog asset. An
    # image_url naming some other object is unexplained provenance - most
    # likely the raw upload, as in the legacy "Black Loafers" row - and such a
    # record is a mixed legacy record, not a catalog-only one.
    catalog_only = (
        renderable
        and not is_cutout
        and not has_explicit_raw
        and not has_masked
        and (not generic or _board_url_identity(generic) == _board_url_identity(selected_url))
    )

    if not renderable:
        source = "none"
    elif catalog_only:
        source = "catalog"
    else:
        source = winning_field or "none"

    # Only report masked_url when masked_url itself won. A different cutout
    # field winning (cutout_url/rmbg_url/...) must not be re-badged as a mask.
    masked_url = selected_url if (is_cutout and winning_field == "masked_url") else ""

    return {
        "image_url": _text(item.get("image_url") or item.get("imageUrl")),
        "masked_url": masked_url,
        "normalized_url": _text(item.get("normalized_url") or item.get("normalizedUrl")),
        "safe_image_url": selected_url,
        "safe_image_source": source,
        "board_ready": bool(selected_url),
        "expected_transparent": is_cutout,
    }


def serialize_wardrobe_board_item(record: Any) -> Optional[Dict[str, Any]]:
    """Wire representation of a wardrobe item for a VISUAL board surface.

    Returns None when the item is not board-ready - callers must skip it and
    pick the next compatible item rather than degrading to the raw upload.

    The response-level `image_url` is set to the board-safe asset, NOT the
    persisted one: legacy wardrobe rows store a raw-bucket object there (live
    example: "Black Loafers", whose masked_url/normalized_url are genuine
    cutouts) and every board surface used to emit it verbatim.

    Raw provenance is preserved separately in original_image_url, which is
    also load-bearing for the client: lib/util/wardrobe_image_resolver.dart
    only exempts image_url from its raw-alias veto when the payload is a
    "frozen snapshot" - selected_field AND source_kind AND one of
    original_image_url/raw_url/preview_url. Emitting the rewritten image_url
    WITHOUT that triple is precisely the ce4ade1 device regression (the mask
    is rejected as a self-alias and the item renders an empty hanger), so the
    three fields must always travel together.
    """
    item = record if isinstance(record, dict) else {}

    # Style assets are a DIFFERENT object contract (see services.style_asset_
    # contract): image_url is the selected board presentation and
    # catalog_image_url is the stable catalog reference, neither of which is a
    # user upload. Rewriting image_url here would regress that contract, so
    # they pass through untouched - only their admission is checked.
    if _text(item.get("source")) == "style_asset" or item.get("is_style_asset") is True:
        return dict(item) if is_board_renderable(item) else None

    contract = canonicalize_wardrobe_image_contract(item)
    if not contract["board_ready"]:
        return None

    safe = contract["safe_image_url"]
    entry = dict(item)
    entry.update(project_board_image_fields(item))
    entry["safe_image_url"] = safe
    entry["safe_image_source"] = contract["safe_image_source"]
    entry["board_ready"] = True
    entry["expected_transparent"] = contract["expected_transparent"]
    entry["selected_field"] = contract["safe_image_source"]
    entry["source_kind"] = _SOURCE_KIND_CUTOUT if contract["expected_transparent"] else _SOURCE_KIND_CATALOG

    # Only record original_image_url when the upload genuinely differs from
    # the selected asset. Its absence tells the client that image_url is the
    # sole provenance and must keep its veto - the same rule _toStyleBoardData
    # applies on the Flutter side.
    # Explicit raw fields outrank the generic image_url. That ordering is what
    # makes this function idempotent: re-serializing an already-serialized item
    # (image_url == safe, original_image_url == the upload) still finds the real
    # upload and keeps the rewrite, instead of concluding image_url == safe
    # means "no distinct provenance" and dropping both fields.
    original = ""
    for key in _EXPLICIT_RAW_KEYS:
        original = original or _text(item.get(key))
    original = original or _text(item.get("image_url") or item.get("imageUrl"))

    if original and _board_url_identity(original) != _board_url_identity(safe):
        # Distinct upload provenance exists: publish the frozen-snapshot
        # triple, which is what licenses the rewritten image_url client-side.
        entry["original_image_url"] = original
        entry["image_url"] = safe
    else:
        # No provenance distinct from the selected asset (masked-only item,
        # or privacy/catalog-only row). Publishing image_url == safe here
        # WITHOUT an original_image_url is not a frozen snapshot, so the
        # client would see the winning field aliasing image_url and reject it
        # as a fabricated cutout - the empty-hanger regression. Drop the
        # generic field instead and let the winning field stand alone. No raw
        # can leak: there is no raw on this record to leak.
        entry.pop("original_image_url", None)
        entry.pop("originalImageUrl", None)
        entry.pop("image_url", None)
        entry.pop("imageUrl", None)
    return entry


__all__ = [
    "is_board_renderable",
    "resolve_board_image_candidate",
    "project_board_image_fields",
    "prepare_board_item",
    "canonicalize_wardrobe_image_contract",
    "serialize_wardrobe_board_item",
]
