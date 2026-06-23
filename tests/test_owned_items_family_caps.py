"""Owned-item family-cap tests (P0 Bad path B).

``_build_owned_items`` matched LLM-suggested text pieces against wardrobe
rows but only de-duped item ids, so two pants / two belts could both land in
the owned list. It must now enforce family caps and de-dupe by id, image_url,
and normalized name.
"""

from __future__ import annotations

from services.style_reasoning_engine import _build_owned_items


def _row(name: str, image_url: str = "", _id: str = ""):
    return {
        "name": name,
        "image_url": image_url or f"https://img/{name.replace(' ', '_').lower()}.png",
        "id": _id or name.lower(),
        "category": "",
    }


def _families(owned):
    return [o["family"] for o in owned]


def test_owned_items_one_bottom_max():
    direction = {"items": ["shirt", "trousers", "chinos", "loafers"]}
    wardrobe = [
        _row("White Shirt"),
        _row("Grey Trousers"),
        _row("Navy Chinos"),
        _row("Black Loafers"),
    ]
    owned, _, _ = _build_owned_items(direction, wardrobe)
    assert _families(owned).count("bottom") == 1


def test_owned_items_one_belt_max():
    direction = {"items": ["shirt", "trousers", "belt", "belt", "loafers"]}
    wardrobe = [
        _row("White Shirt"),
        _row("Grey Trousers"),
        _row("Brown Leather Belt"),
        _row("Black Leather Belt"),
        _row("Black Loafers"),
    ]
    owned, _, _ = _build_owned_items(direction, wardrobe)
    belts = [o for o in owned if o["name"].lower().endswith("belt")]
    assert len(belts) == 1


def test_owned_items_dedupe_duplicate_image_url():
    dup = "https://img/same.png"
    direction = {"items": ["shirt", "trousers", "loafers"]}
    wardrobe = [
        _row("White Shirt", image_url=dup, _id="a"),
        _row("Grey Trousers", image_url=dup, _id="b"),
        _row("Black Loafers"),
    ]
    owned, _, _ = _build_owned_items(direction, wardrobe)
    urls = [o["image_url"] for o in owned if o["image_url"]]
    assert len(urls) == len(set(urls))
