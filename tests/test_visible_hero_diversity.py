"""Phase 1/2: visible-hero diversity + placeholder-hero demotion.

The carousel must not show the same visible hero (dress > outerwear > top)
on multiple boards when alternatives exist, and must never lead with an
image-less placeholder hero when an image-backed board exists.
"""

from __future__ import annotations

from brain.engines.outfit_quality_guard import (
    _enforce_outfit_diversity,
    _visible_hero,
    _item_has_image,
)


def _outfit(score, hero_name, hero_img=True, shirt="White Shirt", pants="Black Pants"):
    blazer = {"name": hero_name, "category": "blazer"}
    if hero_img:
        blazer["image_url"] = f"https://r2/{hero_name}.png"
    return {
        "score": score,
        "items": [
            blazer,
            {"name": shirt, "category": "shirt", "image_url": "https://r2/s.png"},
            {"name": pants, "category": "pants", "image_url": "https://r2/p.png"},
        ],
    }


# ---------- visible hero = outerwear (blazer), matches UI ----------

def test_visible_hero_is_outerwear():
    hero = _visible_hero(_outfit(1.0, "Gray Blazer"))
    assert hero.get("name") == "Gray Blazer"


# ---------- same blazer hero collapses to one board ----------

def test_same_visible_hero_deduped():
    outfits = [
        _outfit(0.9, "Gray Blazer", shirt="White Shirt"),
        _outfit(0.8, "Gray Blazer", shirt="Blue Shirt"),
        _outfit(0.7, "Gray Blazer", shirt="Polo"),
    ]
    kept = _enforce_outfit_diversity(outfits, min_keep=1)
    heroes = [_visible_hero(o).get("name") for o in kept]
    assert heroes.count("Gray Blazer") == 1, heroes


def test_distinct_heroes_all_survive():
    outfits = [
        _outfit(0.9, "Gray Blazer", pants="Black Pants"),
        _outfit(0.8, "Navy Blazer", pants="Beige Chinos"),
        _outfit(0.7, "Brown Jacket", pants="Grey Trousers"),
    ]
    kept = _enforce_outfit_diversity(outfits, min_keep=1)
    heroes = {_visible_hero(o).get("name") for o in kept}
    assert heroes == {"Gray Blazer", "Navy Blazer", "Brown Jacket"}


# ---------- placeholder hero demoted ----------

def test_image_backed_hero_beats_placeholder():
    outfits = [
        _outfit(0.95, "Ghost Blazer", hero_img=False),  # higher score, no image
        _outfit(0.80, "Navy Blazer", hero_img=True),
    ]
    kept = _enforce_outfit_diversity(outfits, min_keep=1)
    # The image-less higher-scoring hero must not lead; image-backed wins.
    assert _visible_hero(kept[0]).get("name") == "Navy Blazer"


def test_image_helper():
    assert _item_has_image({"image_url": "x"}) is True
    assert _item_has_image({"name": "no image"}) is False
