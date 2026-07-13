"""Regression tests: professional guardrails unified across wardrobe board +
stylist CTA paths.

Ten cases covering:
  1-3: _is_office_bad_item / _is_professional_safe (style_flow_service)
  4-5: _lite_build_outfit / style_wardrobe_item (stylist CTA path)
  6:   Safe outfit passes through unchanged
  7:   Gold accessory NOT blocked (only gold tops)
  8-10: Additional professional tokens (camo jacket, cargo pants, distressed jeans)
"""

from __future__ import annotations

import pytest

from services import style_flow_service as sfs
from routers import stylist


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _item(name: str, category: str = "", sub_category: str = "", **extra) -> dict:
    # source matches the inline-wardrobe convention used by every other CTA
    # test; the constrained builder rejects source-less fixed items
    # (UNKNOWN_ITEM_SOURCE), which is covered by dedicated tests elsewhere.
    base = {
        "name": name,
        "category": category,
        "sub_category": sub_category,
        "source": "wardrobe",
    }
    base.update(extra)
    return base


def _req(
    wardrobe: list,
    mode: str = "build_outfit",
    anchor: dict | None = None,
    occasion: str | None = None,
) -> stylist.ItemStyleRequest:
    return stylist.ItemStyleRequest(
        user_id="u-test",
        mode=mode,
        anchor_item=anchor or {},
        wardrobe=wardrobe,
        occasion=occasion,
    )


def _board(*item_names: str) -> dict:
    return {"title": "Test Board", "items": [{"name": n} for n in item_names]}


# --------------------------------------------------------------------------
# Test 1 — Main wardrobe path: camo bottom blocked for client_meeting
# via _is_office_bad_item (the _OFFICE_BAD_TOKENS check)
# --------------------------------------------------------------------------


def test_camo_bottom_blocked_for_client_meeting():
    item = _item("Camo Cargo Pants", category="bottom", sub_category="trousers")
    assert sfs._is_office_bad_item(item), "camo should be in _OFFICE_BAD_TOKENS"


# --------------------------------------------------------------------------
# Test 2 — Main wardrobe path: camouflage bottom blocked for corporate_office
# --------------------------------------------------------------------------


def test_camouflage_blocked_for_corporate_office():
    item = _item("Camouflage Joggers", category="bottom")
    # _is_professional_safe uses _is_office_bad_item as the name-fallback
    safe, reason = sfs._is_professional_safe(item, "corporate_office")
    assert not safe, "camouflage should be blocked for corporate_office"
    assert "camo" in reason or "camouflage" in reason or "office_bad_item" in reason


# --------------------------------------------------------------------------
# Test 3 — Main wardrobe path: shiny gold shirt top blocked for client_meeting
# via _is_professional_safe (name-fallback shiny-top layer)
# --------------------------------------------------------------------------


def test_shiny_gold_formal_shirt_blocked_for_client_meeting():
    item = _item(
        "Shiny Gold Formal Shirt",
        category="top",
        sub_category="shirt",
        color_name="gold",
    )
    safe, reason = sfs._is_professional_safe(item, "client_meeting")
    assert not safe, "shiny gold shirt should be blocked for client_meeting"
    # reason should mention gold or shiny
    assert any(t in reason for t in ("gold", "shiny", "satin", "metallic")), reason


# --------------------------------------------------------------------------
# Test 4 — Stylist CTA path: shiny gold shirt anchor blocked for client_meeting
# -> outfit built from safe wardrobe items, not anchored on the unsafe item
# --------------------------------------------------------------------------


def test_cta_path_blocks_shiny_gold_shirt_anchor_for_client_meeting():
    gold_shirt = _item("Shiny Gold Formal Shirt", category="top", sub_category="shirt", id="gold-1")
    safe_trouser = _item("Grey Trouser", category="bottom", sub_category="trouser", id="trouser-1")
    safe_loafer = _item("Black Loafer", category="footwear", sub_category="loafer", id="loafer-1")

    result = stylist.style_wardrobe_item(
        "gold-1",
        _req(
            wardrobe=[gold_shirt, safe_trouser, safe_loafer],
            mode="build_outfit",
            anchor=gold_shirt,
            occasion="client_meeting",
        ),
    )

    assert result["success"] is True
    assert result.get("anchor_blocked") is True, (
        "anchor_blocked should be True when unsafe anchor is rejected"
    )
    # The gold shirt should NOT appear in the outfit items
    item_ids = {i.get("item_id") or i.get("id") for i in result["outfit"]["items"]}
    assert "gold-1" not in item_ids, "unsafe anchor must not be forced into professional outfit"


# --------------------------------------------------------------------------
# Test 5 — Stylist CTA path: unsafe anchor for client_meeting does not force
# the item; safe fallback outfit returned instead
# --------------------------------------------------------------------------


def test_cta_path_does_not_force_unsafe_anchor():
    camo_pants = _item("Camouflage Cargo Pants", category="bottom", id="camo-1")
    safe_shirt = _item("White Cotton Shirt", category="top", sub_category="shirt", id="shirt-1")
    safe_loafer = _item("Tan Loafer", category="footwear", sub_category="loafer", id="loafer-1")

    result = stylist.style_wardrobe_item(
        "camo-1",
        _req(
            wardrobe=[camo_pants, safe_shirt, safe_loafer],
            mode="build_outfit",
            anchor=camo_pants,
            occasion="client_meeting",
        ),
    )

    assert result["success"] is True
    assert result.get("anchor_blocked") is True
    # camo pants should not appear in outfit
    item_ids = {i.get("item_id") for i in result.get("outfit", {}).get("items", [])}
    assert "camo-1" not in item_ids, "camo pants must not be forced into professional outfit"


# --------------------------------------------------------------------------
# Test 6 — Safe outfit passes: white shirt + grey trousers + black loafers
# for client_meeting -> anchor_blocked=False, all 3 items in outfit
# --------------------------------------------------------------------------


def test_safe_outfit_passes_client_meeting():
    shirt = _item("White Oxford Shirt", category="top", sub_category="shirt", id="shirt-1")
    trouser = _item("Grey Trousers", category="bottom", sub_category="trouser", id="trouser-1")
    loafer = _item("Black Loafers", category="footwear", sub_category="loafer", id="loafer-1")

    result = stylist.style_wardrobe_item(
        "shirt-1",
        _req(
            wardrobe=[shirt, trouser, loafer],
            mode="build_outfit",
            anchor=shirt,
            occasion="client_meeting",
        ),
    )

    assert result["success"] is True
    assert result.get("anchor_blocked") is False, "safe outfit should not be blocked"
    assert result.get("validation_passed") is True
    item_ids = {i.get("item_id") for i in result["outfit"]["items"]}
    assert "shirt-1" in item_ids, "safe anchor should be in outfit"


# --------------------------------------------------------------------------
# Test 7 — Gold accessory (ring/watch) NOT blocked — only gold tops/bottoms
# --------------------------------------------------------------------------


def test_gold_accessory_not_blocked_for_professional():
    gold_watch = _item(
        "Gold Watch",
        category="accessory",
        sub_category="watch",
        color_name="gold",
        id="watch-1",
    )
    # Accessories are not in the _TOP_ROLE_TOKENS set, so gold should not block
    safe, reason = sfs._is_professional_safe(gold_watch, "client_meeting")
    assert safe, f"gold watch (accessory) should NOT be blocked; got reason={reason}"

    gold_ring = _item(
        "Gold Ring",
        category="accessory",
        sub_category="ring",
        color_name="gold",
        id="ring-1",
    )
    safe2, reason2 = sfs._is_professional_safe(gold_ring, "office")
    assert safe2, f"gold ring (accessory) should NOT be blocked; got reason={reason2}"


# --------------------------------------------------------------------------
# Test 8 — Camo jacket blocked for interview
# --------------------------------------------------------------------------


def test_camo_jacket_blocked_for_interview():
    item = _item("Camo Utility Jacket", category="top", sub_category="jacket")
    safe, reason = sfs._is_professional_safe(item, "interview")
    assert not safe, "camo jacket should be blocked for interview"
    assert "camo" in reason or "office_bad_item" in reason, reason


# --------------------------------------------------------------------------
# Test 9 — Cargo pants blocked for business_casual
# --------------------------------------------------------------------------


def test_cargo_pants_blocked_for_business_casual():
    item = _item("Cargo Pants", category="bottom", sub_category="trouser")
    # _is_office_bad_item will catch "cargo" in the name
    assert sfs._is_office_bad_item(item), "cargo pants should be caught by _OFFICE_BAD_TOKENS"
    safe, reason = sfs._is_professional_safe(item, "business_casual")
    assert not safe, "cargo pants should be blocked for business_casual"


# --------------------------------------------------------------------------
# Test 10 — Distressed jeans blocked for corporate_office
# --------------------------------------------------------------------------


def test_distressed_jeans_blocked_for_corporate_office():
    item = _item("Distressed Blue Jeans", category="bottom", sub_category="jeans")
    assert sfs._is_office_bad_item(item), "distressed jeans should be caught by _OFFICE_BAD_TOKENS"
    safe, reason = sfs._is_professional_safe(item, "corporate_office")
    assert not safe, "distressed jeans should be blocked for corporate_office"
