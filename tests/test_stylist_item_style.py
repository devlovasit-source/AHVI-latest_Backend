"""Item-detail CTAs: Style This (3 directions) + Build Outfit (1 outfit).

Covers: anchor injection, dress pairing guard (no men's/formal leather shoes),
3 directions for style_this, 1 outfit for build_outfit, and a friendly fallback
that never crashes when the pipeline fails.
"""

from routers import stylist


_DRESS = {
    "item_id": "dress-1",
    "name": "Red Polka Dot Dress",
    "category": "Dresses",
    "sub_category": "Mini Dress",
    "color_name": "Red",
}


def _card(items):
    return {
        "success": True,
        "cards": [{"title": "Look", "items": items, "styling_note": "Effortless."}],
        "data": {"outfits": [{"items": items}], "missing_items": []},
        "message": "",
    }


def _patch_flow(monkeypatch, items):
    monkeypatch.setattr(stylist, "build_style_flow_response", lambda **kw: _card(items))


def _req(**over):
    base = dict(user_id="u1", mode="build_outfit", anchor_item=_DRESS, wardrobe=[_DRESS])
    base.update(over)
    return stylist.ItemStyleRequest(**base)


def test_build_outfit_returns_one_outfit_with_anchor(monkeypatch):
    _patch_flow(monkeypatch, [{"item_id": "sneak-1", "name": "White Sneakers", "category": "Footwear"}])
    result = stylist.style_wardrobe_item("dress-1", _req(mode="build_outfit"))

    assert result["success"] is True
    assert result["mode"] == "build_outfit"
    outfit = result["outfit"]
    assert outfit["title"]
    ids = {stylist._item_id_of(i) for i in outfit["items"]}
    assert "dress-1" in ids, "anchor item must be in the outfit"
    assert "reason" in outfit


def test_style_this_returns_three_directions(monkeypatch):
    _patch_flow(monkeypatch, [{"item_id": "sneak-1", "name": "White Sneakers", "category": "Footwear"}])
    result = stylist.style_wardrobe_item("dress-1", _req(mode="style_this"))

    assert result["success"] is True
    dirs = result["style_directions"]
    assert len(dirs) == 3
    titles = [d["title"] for d in dirs]
    assert titles == ["Casual Brunch", "Date Night", "Vacation Day"]
    for d in dirs:
        assert "styling_note" in d
        assert any(stylist._item_id_of(i) == "dress-1" for i in d["items"])


def test_dress_drops_mens_leather_shoes_and_suggests_missing(monkeypatch):
    # Pipeline returns the weak pairing the client complained about.
    _patch_flow(
        monkeypatch,
        [
            {"item_id": "loafer-1", "name": "Brown Loafers", "category": "Footwear"},
            {"item_id": "leather-1", "name": "Brown Leather Shoes", "category": "Footwear"},
        ],
    )
    result = stylist.style_wardrobe_item("dress-1", _req(mode="build_outfit"))

    outfit = result["outfit"]
    names = " ".join(stylist._txt(i.get("name")).lower() for i in outfit["items"])
    assert "loafer" not in names
    assert "leather shoes" not in names
    # No good owned footwear left -> a missing suggestion is offered instead.
    labels = " ".join(stylist._txt(m.get("label")).lower() for m in outfit["missing_items"])
    assert any(g in labels for g in ("sneaker", "sandal", "flat"))


def test_dress_keeps_good_footwear(monkeypatch):
    _patch_flow(
        monkeypatch,
        [
            {"item_id": "loafer-1", "name": "Brown Loafers", "category": "Footwear"},
            {"item_id": "sneak-1", "name": "White Sneakers", "category": "Footwear"},
        ],
    )
    result = stylist.style_wardrobe_item("dress-1", _req(mode="build_outfit"))
    names = {stylist._txt(i.get("name")).lower() for i in result["outfit"]["items"]}
    assert "white sneakers" in names
    assert "brown loafers" not in names


def test_non_dress_anchor_keeps_loafers(monkeypatch):
    shirt = {"item_id": "shirt-1", "name": "Blue Shirt", "category": "Tops"}
    _patch_flow(monkeypatch, [{"item_id": "loafer-1", "name": "Brown Loafers", "category": "Footwear"}])
    req = stylist.ItemStyleRequest(user_id="u1", mode="build_outfit", anchor_item=shirt, wardrobe=[shirt])
    result = stylist.style_wardrobe_item("shirt-1", req)
    names = {stylist._txt(i.get("name")).lower() for i in result["outfit"]["items"]}
    assert "brown loafers" in names  # loafers fine with a shirt


def test_pipeline_failure_returns_friendly_fallback(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("pipeline down")

    monkeypatch.setattr(stylist, "build_style_flow_response", _boom)
    result = stylist.style_wardrobe_item("dress-1", _req(mode="build_outfit"))

    assert result["success"] is False
    assert result["message"] == stylist._FRIENDLY_FAIL
    assert result["outfit"]["missing_items"]  # offers something, never empty dead-end


def test_style_this_failure_returns_three_fallback_directions(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("pipeline down")

    monkeypatch.setattr(stylist, "build_style_flow_response", _boom)
    result = stylist.style_wardrobe_item("dress-1", _req(mode="style_this"))

    assert result["success"] is False
    assert len(result["style_directions"]) == 3
    assert result["message"] == stylist._FRIENDLY_FAIL
