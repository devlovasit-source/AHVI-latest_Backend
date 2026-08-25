"""Item-detail CTAs (lite wardrobe-pairing path).

Style This -> 3 directions, Build Outfit -> 1 outfit, both built fast from
owned wardrobe items. Covers anchor injection, the dress pairing guard (no
men's/formal leather shoes), and a friendly fallback that never crashes.
"""

from routers import stylist


def _it(item_id, name, category):
    return {
        "id": item_id,
        "name": name,
        "category": category,
        # A real (non-aliased) processed image, so every fixture item is
        # board-renderable by default. Readiness gating itself is covered by
        # tests/test_style_board_image_readiness.py, not by every test here.
        "normalized_url": f"https://images.test/{item_id}-normalized.png",
    }


_DRESS = _it("dress-1", "Red Polka Dot Dress", "Dresses")


def _req(wardrobe, mode="build_outfit", anchor=None, occasion=None):
    return stylist.ItemStyleRequest(
        user_id="u1",
        mode=mode,
        anchor_item=anchor or _DRESS,
        wardrobe=wardrobe,
        occasion=occasion,
    )


def test_resolve_wardrobe_sanitizes_request_rows():
    request = _req(
        [
            _DRESS,
            _it("shirt-1", "Blue Linen Shirt", "misc"),
            _it("charger-1", "USB Phone Charger", "Accessories"),
        ]
    )

    resolved = stylist._resolve_wardrobe(request)

    assert [item["id"] for item in resolved] == ["dress-1", "shirt-1"]


def test_resolve_wardrobe_sanitizes_appwrite_rows(monkeypatch):
    class FakeProxy:
        def list_documents(self, resource, *, user_id):
            assert resource == "outfits"
            assert user_id == "u1"
            return [
                _it("sneaker-1", "White Leather Sneakers", "unknown"),
                _it("charger-1", "USB Phone Charger", "misc"),
            ]

    monkeypatch.setattr(stylist, "AppwriteProxy", FakeProxy)
    request = _req(None)

    resolved = stylist._resolve_wardrobe(request)

    assert [item["id"] for item in resolved] == ["sneaker-1"]


def test_outfit_pipeline_sanitizes_request_wardrobe_before_style_processing(
    monkeypatch,
):
    seen = {}

    monkeypatch.setattr(stylist, "AppwriteProxy", lambda: object())
    monkeypatch.setattr(
        stylist,
        "resolve_location_weather_context",
        lambda **_kwargs: {
            "location": {},
            "weather": {},
            "profile": {},
            "context_usage": {},
        },
    )
    monkeypatch.setattr(
        stylist.style_dna_engine,
        "build",
        lambda payload: seen.setdefault("style_dna", payload["wardrobe"]) or {},
    )

    def fake_style_flow(**kwargs):
        seen["style_flow"] = kwargs["wardrobe"]
        return {"meta": {}}

    monkeypatch.setattr(stylist, "build_style_flow_response", fake_style_flow)
    request = stylist.OutfitPipelineRequest(
        user_id="u1",
        wardrobe=[
            _it("shirt-1", "Blue Linen Shirt", "misc"),
            _it("charger-1", "USB Phone Charger", "accessory"),
        ],
    )

    stylist.run_outfit_pipeline(request)

    assert [item["id"] for item in seen["style_dna"]] == ["shirt-1"]
    assert seen["style_flow"] is seen["style_dna"]


def test_outfit_pipeline_sanitizes_appwrite_wardrobe_before_style_processing(
    monkeypatch,
):
    seen = {}

    class FakeProxy:
        def list_documents(self, resource, *, user_id):
            assert resource == "outfits"
            assert user_id == "u1"
            return [
                _it("sneaker-1", "White Leather Sneakers", "unknown"),
                _it("charger-1", "USB Phone Charger", "misc"),
            ]

    monkeypatch.setattr(stylist, "AppwriteProxy", FakeProxy)
    monkeypatch.setattr(
        stylist,
        "resolve_location_weather_context",
        lambda **_kwargs: {
            "location": {},
            "weather": {},
            "profile": {},
            "context_usage": {},
        },
    )
    monkeypatch.setattr(
        stylist.style_dna_engine,
        "build",
        lambda payload: seen.setdefault("style_dna", payload["wardrobe"]) or {},
    )

    def fake_style_flow(**kwargs):
        seen["style_flow"] = kwargs["wardrobe"]
        return {"meta": {}}

    monkeypatch.setattr(stylist, "build_style_flow_response", fake_style_flow)

    stylist.run_outfit_pipeline(stylist.OutfitPipelineRequest(user_id="u1"))

    assert [item["id"] for item in seen["style_dna"]] == ["sneaker-1"]
    assert seen["style_flow"] is seen["style_dna"]


def test_build_outfit_uses_owned_items_with_anchor():
    wardrobe = [_DRESS, _it("sneak-1", "White Sneakers", "Footwear"), _it("watch-1", "Gold Watch", "Accessories")]
    result = stylist.style_wardrobe_item("dress-1", _req(wardrobe, mode="build_outfit"))

    assert result["success"] is True
    outfit = result["outfit"]
    ids = {i["item_id"] for i in outfit["items"]}
    assert "dress-1" in ids and outfit["items"][0]["role"] == "hero"
    assert "sneak-1" in ids  # owned footwear paired
    assert "reason" in outfit


def test_style_this_returns_three_directions():
    wardrobe = [_DRESS, _it("sneak-1", "White Sneakers", "Footwear")]
    result = stylist.style_wardrobe_item("dress-1", _req(wardrobe, mode="style_this"))

    assert result["success"] is True
    dirs = result["style_directions"]
    titles = [d["title"] for d in dirs]
    assert len(set(titles)) == 3
    assert not {"Casual Brunch", "Date Night", "Vacation Day"} & set(titles)
    for d in dirs:
        assert "styling_note" in d
        assert d["archetype_id"]
        assert d["style_strategy"]["direction_title"] == d["title"]
        assert any(i["item_id"] == "dress-1" for i in d["items"])


def test_incomplete_style_this_is_not_registered_or_shuffleable(monkeypatch):
    shirt = _it("shirt-1", "Blue Shirt", "Tops")
    register_calls = []
    monkeypatch.setattr(
        stylist,
        "register_board",
        lambda **kwargs: register_calls.append(kwargs) or {"ok": True},
    )

    result = stylist.style_wardrobe_item(
        "shirt-1",
        _req([shirt], mode="style_this", anchor=shirt),
    )

    assert result["success"] is False
    assert register_calls == []
    assert all(
        not direction.get("shuffle_available")
        for direction in result["style_directions"]
    )


def test_dress_drops_mens_leather_shoes_and_suggests_missing():
    wardrobe = [_DRESS, _it("leather-1", "Brown Leather Shoes", "Footwear")]
    result = stylist.style_wardrobe_item("dress-1", _req(wardrobe, mode="build_outfit"))

    ids = {i["item_id"] for i in result["outfit"]["items"]}
    assert "leather-1" not in ids, "men's leather shoes must not pair with a dress"
    labels = " ".join(m["label"].lower() for m in result["outfit"]["missing_items"])
    assert any(g in labels for g in ("sneaker", "sandal", "flat"))


def test_dress_prefers_good_footwear_over_loafers():
    wardrobe = [_DRESS, _it("loafer-1", "Brown Loafers", "Footwear"), _it("sneak-1", "White Sneakers", "Footwear")]
    result = stylist.style_wardrobe_item("dress-1", _req(wardrobe, mode="build_outfit"))
    ids = {i["item_id"] for i in result["outfit"]["items"]}
    assert "sneak-1" in ids
    assert "loafer-1" not in ids


def test_non_fashion_items_are_not_paired():
    # A charger mis-saved into the wardrobe must never be paired into a look.
    wardrobe = [
        _DRESS,
        _it("charger-1", "Phone Charger", "Accessories"),
        _it("sneak-1", "White Sneakers", "Footwear"),
    ]
    result = stylist.style_wardrobe_item("dress-1", _req(wardrobe, mode="build_outfit"))
    names = " ".join(i["name"].lower() for i in result["outfit"]["items"])
    assert "charger" not in names
    ids = {i["item_id"] for i in result["outfit"]["items"]}
    assert "charger-1" not in ids


def test_swim_and_sport_gear_is_not_paired():
    trousers = _it("trouser-1", "Olive Green Trousers", "Bottoms")
    wardrobe = [
        trousers,
        _it("shirt-1", "Light Green Shirt", "Tops"),
        _it("swim-1", "Swim Cap", "Accessories"),
        _it("goggle-1", "Swimming Goggles", "Accessories"),
        _it("sneak-1", "Black Sneakers", "Footwear"),
    ]
    result = stylist.style_wardrobe_item("trouser-1", _req(wardrobe, mode="style_this", anchor=trousers))
    for d in result["style_directions"]:
        names = " ".join(i["name"].lower() for i in d["items"])
        assert "swim" not in names and "goggle" not in names


def test_directions_vary_accessories_when_multiple_owned():
    trousers = _it("trouser-1", "Olive Green Trousers", "Bottoms")
    wardrobe = [
        trousers,
        _it("shirt-1", "Light Green Shirt", "Tops"),
        _it("watch-1", "Silver Watch", "Accessories"),
        _it("belt-1", "Tan Belt", "Accessories"),
        _it("bag-1", "Canvas Tote", "Accessories"),
        _it("sneak-1", "Black Sneakers", "Footwear"),
    ]
    result = stylist.style_wardrobe_item("trouser-1", _req(wardrobe, mode="style_this", anchor=trousers))
    acc_per_dir = []
    for d in result["style_directions"]:
        accs = [i["item_id"] for i in d["items"] if i["item_id"] in {"watch-1", "belt-1", "bag-1"}]
        acc_per_dir.append(tuple(accs))
    # 3 directions should not all show the identical accessory.
    assert len(set(acc_per_dir)) > 1, "directions should vary accessories"


def test_non_dress_anchor_allows_loafers():
    shirt = _it("shirt-1", "Blue Shirt", "Tops")
    wardrobe = [shirt, _it("loafer-1", "Brown Loafers", "Footwear"), _it("jeans-1", "Blue Jeans", "Bottoms")]
    result = stylist.style_wardrobe_item("shirt-1", _req(wardrobe, mode="build_outfit", anchor=shirt))
    ids = {i["item_id"] for i in result["outfit"]["items"]}
    assert "loafer-1" in ids  # loafers fine with a shirt


def test_build_outfit_failure_returns_friendly_fallback(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("pairing down")

    monkeypatch.setattr(stylist, "_lite_build_outfit", _boom)
    result = stylist.style_wardrobe_item("dress-1", _req([_DRESS], mode="build_outfit"))

    assert result["success"] is False
    assert result["message"] == stylist._FRIENDLY_FAIL
    assert result["outfit"]["missing_items"]


def test_style_this_failure_returns_three_fallback_directions(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("pairing down")

    monkeypatch.setattr(stylist, "_lite_directions", _boom)
    result = stylist.style_wardrobe_item("dress-1", _req([_DRESS], mode="style_this"))

    assert result["success"] is False
    assert len(result["style_directions"]) == 3
    assert result["message"] == stylist._FRIENDLY_FAIL


# ---------------------------------------------------------------------------
# Accessory-anchor Style This (fix/style-this-accessory-anchor)
#
# Accessory anchors (jewelry, watches, bags, belts...) prefer a one-piece
# "hero" dress + footwear when the wardrobe has a usable dress, since the
# accessory is meant to sit on/complement it. Most wardrobes are separates
# only though, so when no dress survives filtering, the board must fall back
# to top+bottom+footwear instead of collapsing to accessory+footwear (which
# can never satisfy is_complete_board and used to zero out every direction).
# ---------------------------------------------------------------------------

_BRACELET = _it("bracelet-1", "Gold Bracelet", "Jewelry")


def test_accessory_anchor_falls_back_to_separates_when_no_dress():
    wardrobe = [
        _BRACELET,
        _it("shirt-1", "White T-Shirt", "Tops"),
        _it("pants-1", "Black Trousers", "Bottoms"),
        _it("sneak-1", "White Sneakers", "Footwear"),
    ]
    result = stylist.style_wardrobe_item(
        "bracelet-1", _req(wardrobe, mode="style_this", anchor=_BRACELET)
    )

    assert result["success"] is True
    dirs = result["style_directions"]
    assert len(dirs) >= 1
    for d in dirs:
        ids = {i["item_id"] for i in d["items"]}
        assert "bracelet-1" in ids, "accessory anchor must remain in the board"
        assert "shirt-1" in ids
        assert "pants-1" in ids
        assert "sneak-1" in ids


def test_accessory_anchor_uses_dress_when_available():
    dress = _it("dress-2", "Emerald Green Gown", "Dresses")
    wardrobe = [_BRACELET, dress, _it("sneak-1", "White Sneakers", "Footwear")]
    result = stylist.style_wardrobe_item(
        "bracelet-1", _req(wardrobe, mode="style_this", anchor=_BRACELET)
    )

    assert result["success"] is True
    dirs = result["style_directions"]
    assert len(dirs) >= 1
    for d in dirs:
        ids = {i["item_id"] for i in d["items"]}
        assert "bracelet-1" in ids
        assert "dress-2" in ids, "hero-garment pairing must still work when a dress exists"


def test_accessory_anchor_falls_back_when_only_dress_is_strategy_rejected():
    # The dress candidate exists in the wardrobe (bool(groups["dress"]) is
    # True) but every direction's strategy avoids it by name/token - the
    # fallback decision must be made at build time against the real pick,
    # not just presence in the wardrobe.
    dress = _it("dress-3", "Sequin Gown", "Dresses")
    wardrobe = [
        _BRACELET,
        dress,
        _it("shirt-1", "White T-Shirt", "Tops"),
        _it("pants-1", "Black Trousers", "Bottoms"),
        _it("sneak-1", "White Sneakers", "Footwear"),
    ]
    rejecting_strategy = {
        "direction_title": "Rejected Dress Direction",
        "archetype_id": "test-archetype",
        "formality": "casual",
        "palette": [],
        "avoid": ["gown", "sequin"],
        "reasoning_intent": "test",
    }
    monkeypatch_strategies = [dict(rejecting_strategy) for _ in range(3)]

    directions = stylist._lite_directions(
        _BRACELET, wardrobe, {}, strategies=monkeypatch_strategies
    )

    assert len(directions) == 3
    for d in directions:
        ids = {i["item_id"] for i in d["items"]}
        assert "bracelet-1" in ids
        assert "dress-3" not in ids, "the avoided dress must not be selected"
        assert "shirt-1" in ids
        assert "pants-1" in ids
        assert "sneak-1" in ids
        missing_roles = {m["label"] for m in d["missing_items"]}
        assert "A standout dress" not in missing_roles, (
            "must not surface a 'missing dress' when falling back to separates"
        )


def test_accessory_needed_slots_helper_matches_build_time_pick():
    groups_with_usable_dress = {
        "dress": [_it("dress-4", "Floral Dress", "Dresses")],
        "top": [], "bottom": [], "footwear": [], "accessory": [],
    }
    assert stylist._lite_accessory_needed_slots(
        groups_with_usable_dress, 0, None
    ) == ["dress", "footwear"]

    groups_without_dress = {
        "dress": [], "top": [], "bottom": [], "footwear": [], "accessory": [],
    }
    assert stylist._lite_accessory_needed_slots(
        groups_without_dress, 0, None
    ) == ["top", "bottom", "footwear"]

    groups_dress_rejected_by_strategy = {
        "dress": [_it("dress-5", "Sequin Gown", "Dresses")],
        "top": [], "bottom": [], "footwear": [], "accessory": [],
    }
    assert stylist._lite_accessory_needed_slots(
        groups_dress_rejected_by_strategy, 0, {"avoid": ["gown"]}
    ) == ["top", "bottom", "footwear"], (
        "presence in groups alone must not be trusted - the trial pick must "
        "respect the same avoid filtering _lite_pick would apply at build time"
    )


def test_non_accessory_anchors_are_unaffected_by_the_accessory_fallback():
    # Control: top/bottom/footwear/dress anchors must keep using the
    # unmodified _lite_needed_slots() path.
    assert stylist._lite_needed_slots("dress") == ["footwear", "accessory"]
    assert stylist._lite_needed_slots("top") == ["bottom", "footwear", "accessory"]
    assert stylist._lite_needed_slots("bottom") == ["top", "footwear", "accessory"]
    assert stylist._lite_needed_slots("footwear") == ["top", "bottom", "accessory"]


# ---------------------------------------------------------------------------
# Outerwear-anchor Style This (P0 fix: routers/stylist.py _lite_needed_slots
# had no "outerwear" branch, so a jacket/blazer/coat anchor fell into the
# accessory-anchor catch-all and requested ["dress", "footwear"] - a
# menswear-typical wardrobe with no dress returned INSUFFICIENT_WARDROBE
# even with a complete top+bottom+footwear on hand.)
# ---------------------------------------------------------------------------


def test_lite_needed_slots_outerwear_anchor():
    # The requested slots must be separates (top/bottom/footwear/accessory),
    # matching the "top" contract shape plus bottom - never a dress, and
    # never a second "outerwear" slot (so the fix can't force a second
    # outerwear item into the board by construction).
    assert stylist._lite_needed_slots("outerwear") == [
        "top", "bottom", "footwear", "accessory",
    ]
    assert "outerwear" not in stylist._lite_needed_slots("outerwear")
    assert "dress" not in stylist._lite_needed_slots("outerwear")


def test_outerwear_anchor_style_this_succeeds_without_dress():
    jacket = _it("jacket-1", "Navy Wool Blazer", "Outerwear")
    wardrobe = [
        jacket,
        # Not "Oxford Shirt" - "oxford" is also a _LITE_FOOTWEAR token (Oxford
        # shoes), a pre-existing, unrelated name-token collision in
        # _lite_role() that would misclassify this as footwear.
        _it("shirt-1", "White Cotton Shirt", "Tops"),
        _it("trouser-1", "Grey Wool Trousers", "Bottoms"),
        _it("loafer-1", "Brown Leather Loafers", "Footwear"),
    ]
    result = stylist.style_wardrobe_item(
        "jacket-1", _req(wardrobe, mode="style_this", anchor=jacket)
    )

    assert result["success"] is True
    assert result.get("error") is None
    dirs = result["style_directions"]
    assert len(dirs) == 3
    for d in dirs:
        ids = {i["item_id"] for i in d["items"]}
        roles = {i["item_id"]: i["role"] for i in d["items"]}
        assert "jacket-1" in ids, "outerwear anchor must remain on the board"
        assert "shirt-1" in ids, "a real top must be paired, not a dress"
        assert "trouser-1" in ids, "a real bottom must be paired"
        assert "loafer-1" in ids, "footwear must be paired"
        assert roles["jacket-1"] == "outerwear"
        # No second outerwear item was in this wardrobe, so this also
        # proves the board doesn't fabricate/require a second layer.
        assert len(ids) == 4


def test_outerwear_anchor_does_not_require_dress():
    jacket = _it("jacket-2", "Black Denim Jacket", "Outerwear")
    wardrobe = [
        jacket,
        _it("shirt-1", "White T-Shirt", "Tops"),
        _it("jeans-1", "Blue Jeans", "Bottoms"),
        _it("sneak-1", "White Sneakers", "Footwear"),
    ]
    result = stylist.style_wardrobe_item(
        "jacket-2", _req(wardrobe, mode="style_this", anchor=jacket)
    )

    assert result["success"] is True
    for d in result["style_directions"]:
        missing_labels = {m["label"] for m in d["missing_items"]}
        assert "A standout dress" not in missing_labels, (
            "an outerwear anchor must never surface a missing-dress prompt"
        )
        ids = {i["item_id"] for i in d["items"]}
        assert "dress" not in " ".join(i["category"].lower() for i in d["items"])
        assert ids <= {"jacket-2", "shirt-1", "jeans-1", "sneak-1"}


def test_outerwear_anchor_does_not_force_a_second_outerwear_item():
    jacket = _it("jacket-3", "Camel Overcoat", "Outerwear")
    second_layer = _it("cardigan-1", "Grey Cardigan", "Outerwear")
    wardrobe = [
        jacket,
        second_layer,
        _it("shirt-1", "White Shirt", "Tops"),
        _it("trouser-1", "Navy Trousers", "Bottoms"),
        _it("boot-1", "Black Boots", "Footwear"),
    ]
    result = stylist.style_wardrobe_item(
        "jacket-3", _req(wardrobe, mode="style_this", anchor=jacket)
    )

    assert result["success"] is True
    for d in result["style_directions"]:
        ids = {i["item_id"] for i in d["items"]}
        assert "jacket-3" in ids
        # needed_slots for "outerwear" never asks for a second "outerwear"
        # slot, so the second layer piece must not appear as a forced
        # duplicate role - it may only appear if it ever wins the "top"
        # bucket, never as a dedicated second outerwear slot.
        assert not (
            "jacket-3" in ids and "cardigan-1" in ids and len(ids) == 5
        ), "must not force both outerwear pieces plus a full separates set"


def test_outerwear_anchor_regression_no_insufficient_wardrobe():
    # The exact old-failure repro: outerwear anchor + valid top/bottom/
    # footwear + NO dress in the wardrobe used to return
    # INSUFFICIENT_WARDROBE. It must now succeed.
    jacket = _it("jacket-4", "Tan Suede Jacket", "Outerwear")
    wardrobe = [
        jacket,
        _it("shirt-1", "Light Blue Shirt", "Tops"),
        _it("chino-1", "Beige Chinos", "Bottoms"),
        _it("shoe-1", "White Leather Sneakers", "Footwear"),
    ]
    assert not any(
        stylist._lite_role(item) == "dress" for item in wardrobe
    ), "test wardrobe must genuinely contain no dress"

    result = stylist.style_wardrobe_item(
        "jacket-4", _req(wardrobe, mode="style_this", anchor=jacket)
    )

    assert result["success"] is True
    assert result.get("error") != {
        "code": "INSUFFICIENT_WARDROBE",
        "message": "The available wardrobe cannot form a complete Style This look.",
    }


def test_build_outfit_legacy_mode_unaffected_by_outerwear_fix():
    # routers/stylist.py:1201's legacy build_outfit call site never passes
    # identity_role, so anchor_role there is resolved by _lite_role() (name/
    # category token matching), which already classifies jacket/blazer/coat/
    # cardigan/overshirt names as "top" - it can never reach the new
    # "outerwear" branch in _lite_needed_slots. This proves that directly:
    # legacy build_outfit behavior for a jacket anchor is unchanged.
    jacket = _it("jacket-5", "Navy Blazer", "Outerwear")
    assert stylist._lite_role(jacket) == "top", (
        "legacy path classification of a blazer must remain 'top', not "
        "'outerwear' - otherwise this fix would change build_outfit mode too"
    )
    wardrobe = [
        jacket,
        _it("trouser-1", "Charcoal Trousers", "Bottoms"),
        _it("loafer-1", "Brown Loafers", "Footwear"),
    ]
    result = stylist.style_wardrobe_item(
        "jacket-5", _req(wardrobe, mode="build_outfit", anchor=jacket)
    )
    ids = {i["item_id"] for i in result["outfit"]["items"]}
    assert "jacket-5" in ids and "trouser-1" in ids and "loafer-1" in ids


def test_dress_top_bottom_footwear_accessory_anchors_still_unchanged():
    # Existing control coverage (unchanged branches) stays green - explicit
    # re-assertion here as part of the P0 outerwear fix's required
    # regression proof, independent of the pre-existing test above.
    assert stylist._lite_needed_slots("dress") == ["footwear", "accessory"]
    assert stylist._lite_needed_slots("top") == ["bottom", "footwear", "accessory"]
    assert stylist._lite_needed_slots("bottom") == ["top", "footwear", "accessory"]
    assert stylist._lite_needed_slots("footwear") == ["top", "bottom", "accessory"]
    assert stylist._lite_needed_slots("accessory-or-anything-unmapped") == [
        "dress", "footwear",
    ]


def test_footwear_anchor_control_case_unchanged():
    sneaker = _it("sneak-2", "White Sneakers", "Footwear")
    wardrobe = [
        sneaker,
        _it("shirt-1", "White T-Shirt", "Tops"),
        _it("pants-1", "Black Trousers", "Bottoms"),
    ]
    result = stylist.style_wardrobe_item(
        "sneak-2", _req(wardrobe, mode="build_outfit", anchor=sneaker)
    )
    ids = {i["item_id"] for i in result["outfit"]["items"]}
    assert "sneak-2" in ids and "shirt-1" in ids and "pants-1" in ids
