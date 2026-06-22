from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain.plan_pack_flow import build_plan_pack_response
from routers import chat


def _walk_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def _visual_item(response, section_id, label):
    section = next(section for section in response["visual_sections"] if section["id"] == section_id)
    return next(item for item in section["items"] if item["label"] == label)


def test_plan_pack_returns_visual_sections_and_preserves_cards():
    response = build_plan_pack_response("Pack for a carry-on trip")

    assert response["type"] == "checklists"
    assert response["visual_type"] == "visual_packing_checklist"
    assert response["cards"]
    assert response["visual_sections"]
    assert {section["id"] for section in response["visual_sections"]}.issuperset(
        {"clothes", "essentials", "tech", "documents", "weather"}
    )


def test_plan_pack_wardrobe_items_map_to_visual_section_images():
    wardrobe = [
        {
            "$id": "shirt-1",
            "name": "White Cotton Shirt",
            "category": "Tops",
            "display_image_url": "https://example.com/shirt.png",
        }
    ]
    response = build_plan_pack_response("Pack for a carry-on trip", {"wardrobe": wardrobe})

    clothes = next(section for section in response["visual_sections"] if section["id"] == "clothes")
    tops = next(item for item in clothes["items"] if item["label"] == "Tops")

    assert tops["source"] == "wardrobe"
    assert tops["image_urls"] == ["https://example.com/shirt.png"]
    assert tops["wardrobe_item_ids"] == ["shirt-1"]


def test_plan_pack_empty_wardrobe_uses_icon_keys_without_broken_asset_paths():
    response = build_plan_pack_response("Pack for a carry-on trip", {"wardrobe": []})

    documents = next(section for section in response["visual_sections"] if section["id"] == "documents")
    tech = next(section for section in response["visual_sections"] if section["id"] == "tech")

    assert documents["items"]
    assert tech["items"]
    assert all(item["source"] == "icon" for item in documents["items"])
    assert all(item.get("assetIcon") is None for item in documents["items"])
    assert all(item.get("asset_key") is None for item in documents["items"])
    assert all("assets/icons/" not in value for value in _walk_strings(response))


def test_known_packing_items_return_semantic_icon_keys():
    travel_response = build_plan_pack_response("Pack for a carry-on trip", {"wardrobe": []})
    default_response = build_plan_pack_response("prepare my bag", {"wardrobe": []})

    sunscreen = _visual_item(travel_response, "essentials", "Sunscreen")
    charger = _visual_item(default_response, "tech", "Phone + charger")
    passport = _visual_item(default_response, "documents", "Passport/ID")

    assert sunscreen["source"] == "icon"
    assert sunscreen["iconKey"] == "sunscreen"
    assert sunscreen["assetIcon"] is None
    assert charger["iconKey"] == "charger"
    assert charger["assetIcon"] is None
    assert passport["iconKey"] == "passport"
    assert passport["assetIcon"] is None


def test_plan_pack_module_fetches_wardrobe_and_fails_open(monkeypatch):
    app = FastAPI()

    @app.middleware("http")
    async def user_middleware(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    calls = []

    def fake_fetch(user_id, request_wardrobe):
        calls.append((user_id, request_wardrobe))
        raise RuntimeError("appwrite unavailable")

    monkeypatch.setattr(chat, "_fetch_wardrobe_for_style", fake_fetch)

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "planner",
            "message": "pack for a carry-on trip",
            "history": [],
            "context_data": {},
            "user_profile": {"user_id": "user-1"},
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert calls
    assert body["type"] == "checklists"
    assert body["visual_type"] == "visual_packing_checklist"
    assert body["visual_sections"]


# ---------------------------------------------------------------------------
# Wardrobe enrichment refinements
# ---------------------------------------------------------------------------

from brain.plan_pack_flow import _image_url_from_item, _matches_visual_section  # noqa: E402


def test_generic_labels_match_specific_wardrobe_categories():
    assert _matches_visual_section("Tops", "clothes", {"category": "Top"})
    assert _matches_visual_section("Tops", "clothes", {"category": "T-Shirt"})
    assert _matches_visual_section("Bottoms", "clothes", {"category": "Jeans"})
    assert _matches_visual_section("Footwear", "clothes", {"category": "Sneakers"})
    assert _matches_visual_section("Outer layer", "clothes", {"category": "Jacket"})


def test_image_url_field_fallbacks():
    assert _image_url_from_item({"display_image_url": "a"}) == "a"
    assert _image_url_from_item({"normalized_url": "b"}) == "b"
    assert _image_url_from_item({"masked_url": "c"}) == "c"
    assert _image_url_from_item({"imageUrl": "d"}) == "d"
    assert _image_url_from_item({}) == ""


def test_private_items_never_use_wardrobe_images():
    wardrobe = [
        {"$id": "inner1", "name": "Cotton Innerwear", "category": "Innerwear", "display_image_url": "https://cdn/inner.png"},
        {"$id": "sock1", "name": "Ankle Socks", "category": "Socks", "display_image_url": "https://cdn/sock.png"},
    ]
    resp = build_plan_pack_response("Pack for a carry-on trip", {"wardrobe": wardrobe})
    clothes = next(s for s in resp["visual_sections"] if s["id"] == "clothes")
    inner = next(i for i in clothes["items"] if "Innerwear" in i["label"])
    assert inner["source"] != "wardrobe" and inner["image_urls"] == []
    from brain.plan_pack_flow import _is_private_label
    assert all(
        _is_private_label(x)
        for x in ("Innerwear", "Socks x3", "Underwear", "Sleepwear", "Boxers", "Bra")
    )


def test_clothes_map_to_wardrobe_thumbnails():
    wardrobe = [
        {"$id": "top1", "name": "Blue T-Shirt", "category": "T-Shirt", "display_image_url": "https://cdn/top.png"},
        {"$id": "jeans1", "name": "Blue Jeans", "category": "Jeans", "display_image_url": "https://cdn/jeans.png"},
        {"$id": "shoe1", "name": "White Sneakers", "category": "Sneakers", "display_image_url": "https://cdn/shoe.png"},
    ]
    resp = build_plan_pack_response("Pack for a carry-on trip", {"wardrobe": wardrobe})
    clothes = next(s for s in resp["visual_sections"] if s["id"] == "clothes")
    tops = next(i for i in clothes["items"] if i["label"] == "Tops")
    bottoms = next(i for i in clothes["items"] if i["label"] == "Bottoms")
    footwear = next(i for i in clothes["items"] if i["label"] == "Footwear")

    assert tops["source"] == "wardrobe" and "https://cdn/top.png" in tops["image_urls"]
    assert bottoms["source"] == "wardrobe" and "https://cdn/jeans.png" in bottoms["image_urls"]
    assert footwear["source"] == "wardrobe" and "https://cdn/shoe.png" in footwear["image_urls"]
    # No broken asset paths anywhere.
    assert all("assets/icons/" not in v for v in _walk_strings(resp))


def test_plan_pack_module_injects_fetched_wardrobe_images(monkeypatch):
    app = FastAPI()

    @app.middleware("http")
    async def user_middleware(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    monkeypatch.setattr(
        chat,
        "_fetch_wardrobe_for_style",
        lambda user_id, rw: [
            {"$id": "top1", "name": "Blue T-Shirt", "category": "T-Shirt", "display_image_url": "https://cdn/top.png"},
        ],
    )

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "planner",
            "message": "pack for a carry-on trip",
            "history": [],
            "context_data": {},  # no wardrobe -> route must fetch it
            "user_profile": {"user_id": "user-1"},
        },
    )
    body = response.json()
    assert response.status_code == 200
    clothes = next(s for s in body["visual_sections"] if s["id"] == "clothes")
    tops = next(i for i in clothes["items"] if i["label"] == "Tops")
    assert tops["source"] == "wardrobe"
    assert "https://cdn/top.png" in tops["image_urls"]


# ---------------------------------------------------------------------------
# Strict source gating: non-clothing items never use wardrobe images
# ---------------------------------------------------------------------------

# A wardrobe full of clothing AND a non-clothing bottle-like item — none of the
# essentials/tech/documents may pull a wardrobe image.
_GATE_WARDROBE = [
    {"$id": "top1", "name": "Blue T-Shirt", "category": "T-Shirt", "display_image_url": "https://cdn/top.png"},
    {"$id": "top2", "name": "Linen Shirt", "category": "Tops", "display_image_url": "https://cdn/top2.png"},
    {"$id": "bot1", "name": "Blue Jeans", "category": "Jeans", "display_image_url": "https://cdn/jeans.png"},
    {"$id": "bot2", "name": "Khaki Chinos", "category": "Bottoms", "display_image_url": "https://cdn/chino.png"},
    {"$id": "shoe1", "name": "White Sneakers", "category": "Sneakers", "display_image_url": "https://cdn/shoe.png"},
    {"$id": "jkt1", "name": "Denim Jacket", "category": "Outerwear", "display_image_url": "https://cdn/jacket.png"},
    {"$id": "cap1", "name": "Black Cap", "category": "Accessories", "display_image_url": "https://cdn/cap.png"},
    {"$id": "bottle1", "name": "Steel Water Bottle", "category": "Accessories", "display_image_url": "https://cdn/bottle.png"},
]


def _all_visual_items(resp):
    for section in resp["visual_sections"]:
        for item in section["items"]:
            yield item


def _find_item(resp, label_contains):
    for item in _all_visual_items(resp):
        if label_contains.lower() in item["label"].lower():
            return item
    raise AssertionError(f"no item matching {label_contains!r}")


def test_non_clothing_items_are_icon_only_even_with_wardrobe():
    resp = build_plan_pack_response("Pack for a carry-on trip", {"wardrobe": _GATE_WARDROBE})
    for label in ("Sunscreen", "Sunglasses", "Toiletries", "Power bank", "Passport", "Tickets"):
        try:
            item = _find_item(resp, label)
        except AssertionError:
            continue  # label not present for this scenario — fine
        assert item["source"] == "icon", f"{label} must be icon-only"
        assert item["image_urls"] == []
        assert item["wardrobe_item_ids"] == []
        assert item.get("asset_key") is None


def test_passport_and_tickets_have_distinct_icons():
    resp = build_plan_pack_response("prepare my bag", {"wardrobe": []})
    passport = _find_item(resp, "Passport")
    tickets = _find_item(resp, "Tickets")
    assert passport["iconKey"] == "passport"
    assert tickets["iconKey"] == "tickets"
    assert passport["iconKey"] != tickets["iconKey"]


def test_clothing_groups_still_use_wardrobe():
    resp = build_plan_pack_response("Pack for a 5 day Goa trip", {"wardrobe": _GATE_WARDROBE})
    clothes = next(s for s in resp["visual_sections"] if s["id"] == "clothes")
    tops = next(i for i in clothes["items"] if i["label"] == "Tops")
    bottoms = next(i for i in clothes["items"] if i["label"] == "Bottoms")
    footwear = next(i for i in clothes["items"] if i["label"] == "Footwear")
    assert tops["source"] == "wardrobe" and tops["image_urls"]
    assert bottoms["source"] == "wardrobe" and bottoms["image_urls"]
    assert footwear["source"] == "wardrobe" and footwear["image_urls"]


def test_outer_layer_does_not_match_cap_or_accessory():
    from brain.plan_pack_flow import _matches_visual_section
    assert _matches_visual_section("Outer layer", "clothes", {"category": "Outerwear", "name": "Denim Jacket"})
    assert not _matches_visual_section("Outer layer", "clothes", {"category": "Accessories", "name": "Black Cap"})
    assert not _matches_visual_section("Outer layer", "clothes", {"category": "Accessories", "name": "Tote Bag"})


def test_section_item_count_is_display_groups_not_quantity_sum():
    resp = build_plan_pack_response("Pack for a 5 day Goa trip", {"wardrobe": _GATE_WARDROBE})
    clothes = next(s for s in resp["visual_sections"] if s["id"] == "clothes")
    # Count == number of display tiles, not the summed quantities (which is bigger).
    assert clothes["item_count"] == len(clothes["items"])
    assert clothes["piece_count"] >= clothes["item_count"]


# ---------------------------------------------------------------------------
# Outer-layer slot must never take headwear (swim cap / hat)
# ---------------------------------------------------------------------------

from brain.plan_pack_flow import _visual_section_item  # noqa: E402


def test_outer_layer_picks_jacket_over_swim_cap():
    w = [
        {"$id": "cap1", "name": "Black Swim Cap", "category": "Accessories", "display_image_url": "https://cdn/cap.png"},
        {"$id": "jkt1", "name": "Denim Jacket", "category": "Outerwear", "display_image_url": "https://cdn/jkt.png"},
    ]
    item = _visual_section_item("Outer layer x1", section="weather", wardrobe=w)
    assert item["source"] == "wardrobe"
    assert item["image_urls"] == ["https://cdn/jkt.png"]
    assert "cap1" not in item["wardrobe_item_ids"]


def test_outer_layer_only_swim_cap_is_icon():
    w = [{"$id": "cap1", "name": "Black Swim Cap", "category": "Accessories", "display_image_url": "https://cdn/cap.png"}]
    item = _visual_section_item("Outer layer x1", section="weather", wardrobe=w)
    assert item["source"] == "icon"
    assert item["image_urls"] == []
    assert item["wardrobe_item_ids"] == []
    assert item["iconKey"] in ("jacket", "outerwear")


def test_swimwear_slot_still_matches_swimwear():
    assert _matches_visual_section("Beachwear/swimwear", "clothes", {"category": "Swimwear", "name": "Swim Shorts"})


def test_outer_layer_rejects_headwear_accepts_garments():
    for bad in ("Black Cap", "Wool Beanie", "Sun Hat", "Silk Scarf", "Bike Helmet"):
        assert not _matches_visual_section("Outer layer", "weather", {"name": bad, "category": "Accessories"})
    for good in ("Denim Jacket", "Wool Coat", "Knit Cardigan", "Rain Windbreaker", "Grey Hoodie"):
        assert _matches_visual_section("Outer layer", "weather", {"name": good, "category": "Outerwear"})


def test_light_layer_matches_overshirt_not_cap():
    assert _matches_visual_section("Light layer for evenings", "weather", {"name": "Linen Overshirt", "category": "Outerwear"})
    assert not _matches_visual_section("Light layer for evenings", "weather", {"name": "Swim Cap", "category": "Accessories"})
