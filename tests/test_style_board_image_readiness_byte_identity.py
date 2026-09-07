"""services.style_board_image_readiness - masked_url byte-identity guard.

Narrow P0 hardening: a masked_url at a genuinely distinct object path can
still be the raw upload re-hosted verbatim (RMBG no-op). The URL-identity
alias check cannot see this; only comparing actual bytes can. This module
is deliberately NOT a general image-quality gate - see the guard's
docstring in services/style_board_image_readiness.py for why a
status-field gate (image_status/cutout_status) was rejected: those fields
are unpopulated on effectively the whole wardrobe population, so gating on
them would reclassify hundreds of fine items and blank dozens with no
catalog fallback. Byte-identity is scoped to the one provable failure mode.

Network calls are mocked throughout - these are unit tests, not live
fetch tests against real object storage.
"""

import services.style_board_image_readiness as readiness
from services.style_board_image_readiness import (
    is_board_renderable,
    resolve_board_image_candidate,
)

_RAW = "https://cdn.test/raw/photo.jpg"
_MASK_DISTINCT_PATH = "https://other-bucket.test/masked/rehosted.png"
_MASK_GENUINE = "https://cdn.test/masked/real-cutout.png"


def setup_function(_):
    # The cache is module-global and keyed by (masked_url, raw_url) - clear
    # it between tests so one test's fetch mock doesn't leak a cached
    # verdict into the next.
    readiness._byte_identity_cache.clear()


def _stub_hashes(mapping):
    """monkeypatch-friendly stand-in for _fetch_bounded_hash: returns
    mapping[url] if present, else None (simulating an unrelated/unmocked
    URL failing to fetch, which must stay inconclusive, not identical)."""

    def _fake(url):
        return mapping.get(url)

    return _fake


# TEST A -- masked_url and raw URL differ textually but bytes are identical
# => masked rejected.
def test_byte_identical_masked_rejected_despite_distinct_url(monkeypatch):
    monkeypatch.setattr(
        readiness,
        "_fetch_bounded_hash",
        _stub_hashes({_RAW: "same-hash", _MASK_DISTINCT_PATH: "same-hash"}),
    )
    item = {"image_url": _RAW, "masked_url": _MASK_DISTINCT_PATH}
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is False
    assert result["reason"] == "no_board_safe_image"


def test_byte_identical_masked_falls_back_to_normalized(monkeypatch):
    monkeypatch.setattr(
        readiness,
        "_fetch_bounded_hash",
        _stub_hashes({_RAW: "same-hash", _MASK_DISTINCT_PATH: "same-hash"}),
    )
    item = {
        "image_url": _RAW,
        "masked_url": _MASK_DISTINCT_PATH,
        "normalized_url": "https://cdn.test/catalog/shirt.png",
    }
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is True
    assert result["selected_field"] == "normalized_url"


# TEST B -- masked_url is a genuine distinct cutout => remains admitted.
def test_genuinely_distinct_masked_bytes_remain_admitted(monkeypatch):
    monkeypatch.setattr(
        readiness,
        "_fetch_bounded_hash",
        _stub_hashes({_RAW: "raw-hash", _MASK_GENUINE: "cutout-hash"}),
    )
    item = {"image_url": _RAW, "masked_url": _MASK_GENUINE}
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is True
    assert result["selected_field"] == "masked_url"
    assert result["selected_url"] == _MASK_GENUINE


# TEST C -- same URL pair encountered repeatedly => comparison reused/cached,
# not re-fetched.
def test_repeated_pair_uses_cache_not_refetched(monkeypatch):
    calls = []

    def _counting_fetch(url):
        calls.append(url)
        return {_RAW: "same-hash", _MASK_DISTINCT_PATH: "same-hash"}.get(url)

    monkeypatch.setattr(readiness, "_fetch_bounded_hash", _counting_fetch)
    item = {"image_url": _RAW, "masked_url": _MASK_DISTINCT_PATH}

    first = resolve_board_image_candidate(item)
    calls_after_first = len(calls)
    assert first["renderable"] is False
    assert calls_after_first == 2  # one fetch each for masked + raw

    second = resolve_board_image_candidate(item)
    assert second["renderable"] is False
    assert len(calls) == calls_after_first, "second call must reuse the cached verdict"


# TEST D -- comparison timeout/network failure must not bless the ambiguous
# masked asset as proven safe, i.e. no False-positive "identical" verdict
# gets cached from an inconclusive fetch, but availability is preserved
# (the item still renders using the ambiguous masked asset rather than the
# whole board serialization failing).
def test_fetch_failure_is_inconclusive_not_cached_and_stays_admitted(monkeypatch):
    monkeypatch.setattr(readiness, "_fetch_bounded_hash", lambda url: None)
    item = {"image_url": _RAW, "masked_url": _MASK_DISTINCT_PATH}

    result = resolve_board_image_candidate(item)

    assert result["renderable"] is True
    assert result["selected_field"] == "masked_url"
    # Inconclusive comparisons are never cached -- a later call with a
    # working fetch must still be able to catch a real violation.
    assert readiness._byte_identity_cache == {}


def test_fetch_failure_then_recovery_still_catches_real_violation(monkeypatch):
    item = {"image_url": _RAW, "masked_url": _MASK_DISTINCT_PATH}

    monkeypatch.setattr(readiness, "_fetch_bounded_hash", lambda url: None)
    inconclusive = resolve_board_image_candidate(item)
    assert inconclusive["renderable"] is True  # fails open while unverifiable

    monkeypatch.setattr(
        readiness,
        "_fetch_bounded_hash",
        _stub_hashes({_RAW: "same-hash", _MASK_DISTINCT_PATH: "same-hash"}),
    )
    now_caught = resolve_board_image_candidate(item)
    assert now_caught["renderable"] is False


# TEST E -- frozen snapshot / idempotency: a re-serialized board item (image_url
# rewritten to the selected asset, source/selected_field/original_image_url
# present) must not have its own selected field rejected as self-identical,
# and re-running resolution must not change the outcome or demote a good
# processed asset.
def test_frozen_snapshot_not_rejected_by_byte_guard(monkeypatch):
    # image_url == masked_url here is the EXPECTED frozen-snapshot shape (the
    # selected asset was written back into image_url at serialization time),
    # not evidence of a raw copy -- the frozen carve-out in _raw_aliases()
    # excludes generic image_url from the raw-provenance set. The guard
    # still checks against original_image_url (the genuine upload, which
    # frozen snapshots keep in the alias set on purpose), so a masked asset
    # that really is a distinct cutout must compare as non-identical to it.
    monkeypatch.setattr(
        readiness,
        "_fetch_bounded_hash",
        _stub_hashes({_MASK_GENUINE: "cutout-hash", _RAW: "raw-hash"}),
    )
    item = {
        "selected_field": "masked_url",
        "source_kind": "processed_cutout",
        "image_url": _MASK_GENUINE,  # frozen snapshot: image_url == selected asset
        "masked_url": _MASK_GENUINE,
        "original_image_url": _RAW,
    }
    first = resolve_board_image_candidate(item)
    second = resolve_board_image_candidate(item)
    assert first == second
    assert first["renderable"] is True
    assert first["selected_field"] == "masked_url"
    assert first["selected_url"] == _MASK_GENUINE


def test_frozen_snapshot_whose_mask_genuinely_is_the_upload_still_rejected(monkeypatch):
    # The opposite case: a frozen snapshot where the "selected" masked asset
    # really is byte-identical to the genuine upload recorded in
    # original_image_url. Safety must not weaken for frozen items -- this
    # is exactly the case _is_frozen_snapshot's docstring says must still
    # be rejected.
    monkeypatch.setattr(
        readiness,
        "_fetch_bounded_hash",
        _stub_hashes({_MASK_GENUINE: "same-hash", _RAW: "same-hash"}),
    )
    item = {
        "selected_field": "masked_url",
        "source_kind": "processed_cutout",
        "image_url": _MASK_GENUINE,
        "masked_url": _MASK_GENUINE,
        "original_image_url": _RAW,
    }
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is False


# TEST F -- Black-Loafers-style fixture (live P0 device evidence, byte-for-byte
# reproduction with real hash values) => RAW LEAK = NO.
def test_black_loafers_fixture_raw_leak_no(monkeypatch):
    masked = "https://pub-d4d02883ddda4a1bba452bfe6d1be814.r2.dev/wardrobe_45aa2af8-2f6a-47ec-9055-bcb99d9f5d18.png"
    raw = "https://pub-9ca6234baa424e56882e953c97ffbe14.r2.dev/raw_45aa2af8-2f6a-47ec-9055-bcb99d9f5d18.png"
    # Measured live: both objects hash to
    # 6bcd5fbc56eb8bb2772277ee22f0795785a38c38c7e90a6fc070fadc53116c5a
    shared_hash = "6bcd5fbc56eb8bb2772277ee22f0795785a38c38c7e90a6fc070fadc53116c5a"
    monkeypatch.setattr(
        readiness, "_fetch_bounded_hash", _stub_hashes({masked: shared_hash, raw: shared_hash})
    )
    item = {
        "id": "45aa2af8-2f6a-47ec-9055-bcb99d9f5d18",
        "name": "Black Loafers",
        "image_url": raw,
        "masked_url": masked,
        "normalized_url": "https://pub-d4d02883ddda4a1bba452bfe6d1be814.r2.dev/catalog_45aa2af8-2f6a-47ec-9055-bcb99d9f5d18.png",
    }
    assert is_board_renderable(item) is True  # normalized_url fallback exists
    result = resolve_board_image_candidate(item)
    assert result["selected_field"] == "normalized_url"
    assert result["selected_url"] != masked
    assert result["selected_url"] != raw
