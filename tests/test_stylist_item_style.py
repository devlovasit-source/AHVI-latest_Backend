"""Item-detail CTAs (lite wardrobe-pairing path).

Style This -> 3 directions, Build Outfit -> 1 outfit, both built fast from
owned wardrobe items. Covers anchor injection, the dress pairing guard (no
men's/formal leather shoes), and a friendly fallback that never crashes.
"""

import pytest

from routers import stylist


def _it(item_id, name, category):
    return {"id": item_id, "name": name, "category": category, "source": "wardrobe"}


_DRESS = _it("dress-1", "Red Polka Dot Dress", "Dresses")


def _req(wardrobe, mode="build_outfit", anchor=None, occasion=None, style_assets=None,
         allow_wardrobe_fallback=False):
    assets = style_assets
    if assets is None and mode == "style_this":
        assets = [
            {**item, "id": f"asset-{item['id']}", "source": "style_asset"}
            for item in wardrobe
            if item.get("id") != (anchor or _DRESS).get("id")
        ]
    return stylist.ItemStyleRequest(
        user_id="u1",
        mode=mode,
        anchor_item=anchor or _DRESS,
        wardrobe=wardrobe,
        style_assets=assets,
        occasion=occasion,
        allow_wardrobe_fallback=allow_wardrobe_fallback,
    )


def test_build_outfit_uses_owned_items_with_anchor():
    wardrobe = [_DRESS, _it("sneak-1", "White Sneakers", "Footwear"), _it("watch-1", "Gold Watch", "Accessories")]
    result = stylist.style_wardrobe_item("dress-1", _req(wardrobe, mode="build_outfit"))

    assert result["success"] is True
    outfit = result["outfit"]
    ids = {i["item_id"] for i in outfit["items"]}
    assert "dress-1" in ids and outfit["items"][0]["role"] == "dress"
    assert outfit["items"][0]["board_role"] == "hero"
    assert "sneak-1" in ids  # owned footwear paired
    assert "reason" in outfit


def test_style_this_returns_three_directions():
    wardrobe = [_DRESS, _it("sneak-1", "White Sneakers", "Footwear")]
    result = stylist.style_wardrobe_item("dress-1", _req(wardrobe, mode="style_this"))

    assert result["success"] is True
    dirs = result["style_directions"]
    assert [d["title"] for d in dirs] == ["Casual Brunch", "Date Night", "Vacation Day"]
    for d in dirs:
        assert "styling_note" in d
        assert any(i["item_id"] == "dress-1" for i in d["items"])


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
        accs = [i["item_id"] for i in d["items"] if i["item_id"] in {"asset-watch-1", "asset-belt-1", "asset-bag-1"}]
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

    monkeypatch.setattr(stylist, "_builder_outfit", _boom)
    result = stylist.style_wardrobe_item("dress-1", _req([_DRESS], mode="build_outfit"))

    assert result["success"] is False
    assert result["error"]["code"] == "CONSTRAINED_BUILDER_FAILED"
    assert result["outfit"]["items"] == []


def test_style_this_failure_returns_three_fallback_directions(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("pairing down")

    monkeypatch.setattr(stylist, "_builder_directions", _boom)
    result = stylist.style_wardrobe_item("dress-1", _req([_DRESS], mode="style_this"))

    assert result["success"] is False
    assert result["style_directions"] == []
    assert result["error"]["code"] == "CONSTRAINED_BUILDER_FAILED"


def test_build_outfit_returns_anchor_exact_id_and_image():
    dress = {"id": "dress-1", "name": "Red Polka Dot Dress", "category": "Dresses",
             "image_url": "https://img/dress-1.png", "source": "wardrobe"}
    wardrobe = [dress, _it("sneak-1", "White Sneakers", "Footwear")]
    result = stylist.style_wardrobe_item("dress-1", _req(wardrobe, mode="build_outfit", anchor=dress))
    assert result["success"] is True
    hero = result["outfit"]["items"][0]
    assert hero["item_id"] == "dress-1"
    assert hero["image_url"] == "https://img/dress-1.png"
    assert hero["locked"] is True
    assert hero["slot"] == "dress"
    assert result["outfit"]["board_id"]
    assert result["outfit"]["revision"] == 1
    assert set(hero["position"]) == {"x", "y", "width", "height", "z", "rotation"}


def test_style_this_returns_anchor_exact_id_and_image():
    dress = {"id": "dress-1", "name": "Red Polka Dot Dress", "category": "Dresses",
             "image_url": "https://img/dress-1.png", "source": "wardrobe"}
    wardrobe = [dress, _it("sneak-1", "White Sneakers", "Footwear")]
    result = stylist.style_wardrobe_item("dress-1", _req(wardrobe, mode="style_this", anchor=dress))
    assert result["success"] is True
    for d in result["style_directions"]:
        assert d["board_id"]
        assert d["revision"] == 1
        hero = next(i for i in d["items"] if i["item_id"] == "dress-1")
        assert hero["image_url"] == "https://img/dress-1.png"
        assert set(hero["position"]) == {"x", "y", "width", "height", "z", "rotation"}


def test_name_only_anchor_is_rejected():
    anchor = {"name": "Blue Shirt", "category": "Tops", "source": "wardrobe"}
    result = stylist.style_wardrobe_item("path-id", _req([], anchor=anchor))
    assert result["success"] is False
    assert result["error"]["code"] == "INVALID_ITEM_ID"


def test_style_this_without_assets_does_not_widen_to_wardrobe():
    result = stylist.style_wardrobe_item(
        "dress-1", _req([_DRESS, _it("shoe-1", "Sneakers", "Footwear")],
                         mode="style_this", style_assets=[]),
    )
    assert result["success"] is False
    assert result["error"]["code"] == "STYLE_ASSET_POOL_EMPTY"


def test_inline_missing_source_remains_unknown():
    anchor = {"id": "top-1", "name": "Shirt", "category": "Tops"}
    result = stylist.style_wardrobe_item("top-1", _req([anchor], anchor=anchor))
    assert result["success"] is False
    assert result["error"]["code"] == "UNKNOWN_ITEM_SOURCE"


def test_professional_incompatible_anchor_is_not_replaced():
    anchor = _it("camo-1", "Camouflage Cargo Pants", "Bottoms")
    safe = _it("shirt-1", "White Shirt", "Tops")
    result = stylist.style_wardrobe_item(
        "camo-1", _req([anchor, safe], anchor=anchor, occasion="client_meeting"),
    )
    assert result["success"] is False
    assert result["error"]["code"] == "ANCHOR_INCOMPATIBLE_WITH_OCCASION"
    assert result["outfit"]["items"] == []


def test_response_preserves_canonical_role_and_board_fields():
    watch = {
        "id": "watch-1", "name": "Silver Watch", "category": "Accessories",
        "source": "wardrobe", "masked_url": "masked", "board_image_url": "board",
        "position": {"x": 0.2, "y": 0.3},
    }
    result = stylist.style_wardrobe_item("watch-1", _req([watch], anchor=watch))
    item = result["outfit"]["items"][0]
    assert item["role"] == item["slot"] == "accessory"
    assert item["board_role"] == "hero"
    assert item["accessory_type"] == "watch"
    assert item["source"] == "wardrobe"
    assert item["masked_url"] == "masked"
    assert item["board_image_url"] == "board"
    assert item["position"] == watch["position"]


def test_build_completion_items_are_wardrobe():
    result = stylist.style_wardrobe_item(
        "dress-1", _req([_DRESS, _it("shoe-1", "Sneakers", "Footwear")]),
    )
    assert all(i["source"] == "wardrobe" for i in result["outfit"]["items"])


def test_style_this_completion_items_are_style_assets():
    asset = {"id": "asset-shoe", "name": "Sneakers", "category": "Footwear", "source": "style_asset"}
    result = stylist.style_wardrobe_item(
        "dress-1", _req([_DRESS], mode="style_this", style_assets=[asset]),
    )
    assert result["success"] is True
    for direction in result["style_directions"]:
        assert all(i["source"] == "style_asset" for i in direction["items"] if not i["locked"])


@pytest.mark.parametrize("id_key", ["id", "$id", "item_id", "itemId", "image_id", "asset_id"])
def test_all_stable_anchor_id_aliases_work(id_key):
    anchor = {id_key: "dress-1", "name": "Red Dress", "category": "Dresses", "source": "wardrobe"}
    result = stylist.style_wardrobe_item("dress-1", _req([anchor], anchor=anchor))
    assert result["success"] is True
    assert result["outfit"]["items"][0]["item_id"] == "dress-1"


def test_builder_failure_never_calls_lite_fallback(monkeypatch):
    def _builder_boom(*args, **kwargs):
        raise stylist.ConstrainedOutfitError("NO_VALID_COMBINATION", "No safe combination.")

    def _lite_must_not_run(*args, **kwargs):
        raise AssertionError("legacy fallback was called")

    monkeypatch.setattr(stylist, "_builder_outfit", _builder_boom)
    monkeypatch.setattr(stylist, "_lite_build_outfit", _lite_must_not_run)
    result = stylist.style_wardrobe_item("dress-1", _req([_DRESS]))
    assert result["error"]["code"] == "NO_VALID_COMBINATION"


def test_final_serialization_rejects_changed_anchor_image(monkeypatch):
    original_generate = stylist._builder.generate

    def _changed(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        result["items"][0]["image_url"] = "https://img/changed.png"
        return result

    anchor = {**_DRESS, "image_url": "https://img/original.png"}
    monkeypatch.setattr(stylist._builder, "generate", _changed)
    result = stylist.style_wardrobe_item("dress-1", _req([anchor], anchor=anchor))
    assert result["success"] is False
    assert result["error"]["code"] == "FIXED_ITEM_LOST"


def test_trusted_appwrite_rows_are_stamped_wardrobe(monkeypatch):
    class Proxy:
        def list_documents(self, resource, **kwargs):
            if resource == "outfits":
                return [{"id": "top-1", "name": "Shirt", "category": "Tops"}]
            return []

    monkeypatch.setattr(stylist, "AppwriteProxy", Proxy)
    request = stylist.ItemStyleRequest(user_id="u1", anchor_item={})
    rows, trusted = stylist._resolve_wardrobe(request)
    assert trusted is True
    assert rows[0]["source"] == "wardrobe"


def test_caller_rows_are_not_stamped_wardrobe():
    request = stylist.ItemStyleRequest(user_id="u1", wardrobe=[{"id": "top-1"}])
    rows, trusted = stylist._resolve_wardrobe(request)
    assert trusted is False
    assert "source" not in rows[0]
