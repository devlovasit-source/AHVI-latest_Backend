"""Item-detail CTAs (lite wardrobe-pairing path).

Style This -> 3 directions, Build Outfit -> 1 outfit, both built fast from
owned wardrobe items. Covers anchor injection, the dress pairing guard (no
men's/formal leather shoes), and a friendly fallback that never crashes.
"""

from routers import stylist


def _it(item_id, name, category):
    return {"id": item_id, "name": name, "category": category}


_DRESS = _it("dress-1", "Red Polka Dot Dress", "Dresses")


def _req(wardrobe, mode="build_outfit", anchor=None, occasion=None):
    return stylist.ItemStyleRequest(
        user_id="u1",
        mode=mode,
        anchor_item=anchor or _DRESS,
        wardrobe=wardrobe,
        occasion=occasion,
    )


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
    assert len(dirs) == 3
    valid_titles = {t[0] for t in stylist._LITE_STYLE_DIRECTIONS}
    for d in dirs:
        assert d["title"] in valid_titles
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
