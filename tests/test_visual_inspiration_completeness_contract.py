from pathlib import Path

from services import style_reasoning_engine as engine


TOP = {
    "asset_id": "top-1",
    "name": "Navy Oxford Shirt",
    "category": "top",
    "role": "top",
    "board_image_url": "https://cdn.example/top.png",
    "image_url": "https://cdn.example/top.jpg",
}

BOTTOM = {
    "asset_id": "bottom-1",
    "name": "Tailored Navy Trouser",
    "category": "bottom",
    "role": "bottom",
    "board_image_url": "https://cdn.example/bottom.png",
    "image_url": "https://cdn.example/bottom.jpg",
}

FOOTWEAR = {
    "asset_id": "shoe-1",
    "name": "Minimal Navy Sneaker",
    "category": "footwear",
    "role": "footwear",
    "board_image_url": "https://cdn.example/shoe.png",
    "image_url": "https://cdn.example/shoe.jpg",
}

BAG = {
    "asset_id": "bag-1",
    "name": "Black Backpack",
    "category": "bag",
    "role": "accessory",
    "board_image_url": "https://cdn.example/bag.png",
    "image_url": "https://cdn.example/bag.jpg",
}

CAP = {
    "asset_id": "cap-1",
    "name": "Navy Cap",
    "category": "accessory",
    "role": "accessory",
    "board_image_url": "https://cdn.example/cap.png",
    "image_url": "https://cdn.example/cap.jpg",
}


def test_repairs_missing_top_and_footwear(monkeypatch):
    assets = [TOP, BOTTOM, FOOTWEAR, BAG]

    def fake_best_assets(_assets, *, direction, **_kwargs):
        probe = str(direction.get("hero_piece") or "").lower()

        if "shirt" in probe:
            return [dict(TOP)]
        if "shoe" in probe:
            return [dict(FOOTWEAR)]
        if "trouser" in probe:
            return [dict(BOTTOM)]

        return []

    monkeypatch.setattr(
        engine,
        "_best_style_assets",
        fake_best_assets,
    )

    repaired = engine._repair_visual_board_core(
        [dict(BOTTOM), dict(BAG)],
        assets=assets,
        direction={
            "title": "Refined Weekend",
            "hero_piece": "Fine-Gauge Sweater",
        },
        occasion="weekend",
        target_gender="male",
        allow_feminine_accessory=False,
        brief={},
    )

    roles = {
        engine._visual_board_item_role(item)
        for item in repaired
    }

    assert {"top", "bottom", "footwear"}.issubset(roles)
    assert engine._board_items_viable(repaired)


def test_accessories_cannot_displace_core_slots():
    assets = [TOP, BOTTOM, FOOTWEAR, BAG, CAP]

    repaired = engine._repair_visual_board_core(
        [TOP, BOTTOM, FOOTWEAR, BAG, CAP],
        assets=assets,
        direction={"title": "Contemporary Classic"},
        occasion="today",
        target_gender="male",
        allow_feminine_accessory=False,
        brief={},
    )

    roles = [
        engine._visual_board_item_role(item)
        for item in repaired
    ]

    assert roles.count("top") == 1
    assert roles.count("bottom") == 1
    assert roles.count("footwear") == 1
    assert roles.count("accessory") <= 1
    assert engine._board_items_viable(repaired)


def test_visible_copy_uses_actual_selected_hero():
    direction = {
        "title": "Contemporary Classic",
        "hero_piece": "Light Blue Oxford Shirt",
        "asset_id": "missing-generated-hero",
        "why_it_works": "Old inconsistent copy.",
    }

    board_items = [TOP, BOTTOM, FOOTWEAR]

    engine._sync_visual_direction_copy(
        direction,
        board_items,
    )

    assert direction["hero_piece"] == "Navy Oxford Shirt"
    assert "Navy Oxford Shirt" in direction["why_it_works"]
    assert "Light Blue Oxford Shirt" not in direction["why_it_works"]
    assert direction["asset_id"] == "top-1"


def test_chat_router_does_not_slice_styling_tip_at_80_characters():
    source = Path("routers/chat.py").read_text(encoding="utf-8")

    assert (
        '"styling_tip": str(item.get("styling_tip") '
        'or item.get("style_note") or "")[:80]'
        not in source
    )
