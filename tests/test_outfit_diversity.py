"""Diversity contract for set-level board selection."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from brain.engines.style_brief import build_brief, select_board_set


def _card(idx: int, top_id: str, bottom_id: str, footwear_id: str = "loafer_1") -> Dict[str, Any]:
    return {
        "id": f"card_{idx}",
        "title": f"Look {idx + 1}",
        "items": [
            {"id": top_id, "name": "Shirt", "category": "button down", "role": "top"},
            {"id": bottom_id, "name": "Trouser", "category": "trousers", "role": "bottom"},
            {"id": footwear_id, "name": "Loafer", "category": "loafers", "role": "footwear"},
        ],
        "score_meta": {"score": 7.0, "reasons": ["palette aligned"]},
    }


def test_repeated_top_avoided_when_other_tops_exist():
    brief = build_brief(router_occasion="office", query="office")
    cards = [
        _card(0, "shirt_a", "trs_a"),
        _card(1, "shirt_a", "trs_b"),  # same top
        _card(2, "shirt_b", "trs_b"),  # different top
    ]
    chosen = select_board_set(cards, brief, max_n=3)
    heroes = [c["items"][0]["id"] for c in chosen]
    assert len(set(heroes)) >= 2


def test_single_top_keeps_card_count():
    brief = build_brief(router_occasion="office", query="office")
    cards = [
        _card(0, "shirt_only", "trs_a"),
        _card(1, "shirt_only", "trs_b"),
    ]
    chosen = select_board_set(cards, brief, max_n=3)
    # Only one top exists — keep both boards, don't drop them.
    assert len(chosen) == 2


def test_set_roles_assigned_to_first_three():
    brief = build_brief(router_occasion="office", query="office")
    cards = [
        _card(0, "shirt_a", "trs_a"),
        _card(1, "shirt_b", "trs_b"),
        _card(2, "shirt_c", "trs_c"),
    ]
    chosen = select_board_set(cards, brief, max_n=3)
    roles = [c.get("set_role") for c in chosen]
    assert roles[0] == "primary"
    assert roles[1] == "alternate"
    if len(chosen) == 3:
        assert roles[2] == "expressive"


def test_broader_pool_picks_distinct_heroes_for_first_two():
    """4 office candidates: first 2 share hero, 3rd has different hero.
    Final top 2 must use different heroes — set selection wins over rank."""
    brief = build_brief(router_occasion="office", query="office today")
    shirt_a = "shirt_a"
    shirt_b = "shirt_b"
    shirt_c = "shirt_c"
    cards = [
        _card(0, shirt_a, "trs_a"),  # rank 1 — shirt_a
        _card(1, shirt_a, "trs_b"),  # rank 2 — same shirt_a
        _card(2, shirt_b, "trs_b"),  # rank 3 — shirt_b
        _card(3, shirt_c, "trs_c"),  # rank 4 — shirt_c
    ]
    # Stack scores so the duplicate-hero card would naturally win rank 2.
    cards[0]["score_meta"]["score"] = 9.0
    cards[1]["score_meta"]["score"] = 8.5
    cards[2]["score_meta"]["score"] = 6.0
    cards[3]["score_meta"]["score"] = 5.5

    chosen = select_board_set(cards, brief, max_n=2)
    assert len(chosen) == 2
    heroes = [c["items"][0]["id"] for c in chosen]
    assert heroes[0] != heroes[1], (
        f"first two heroes must differ; got {heroes}"
    )


def test_combo_pool_grants_quota_to_every_master():
    """6 master heroes; the first can produce 40 combos. Old code would let
    that one fill the 40-slot cap. New pool must give every master its
    per_master_min quota, and end up with at least 4 unique heroes."""
    from brain.outfit_pipeline import _build_diverse_combo_pool

    def make_combo(hero: str, idx: int) -> Dict[str, Any]:
        return {"combo_id": f"{hero}_{idx}", "top": {"name": hero}, "score": 7}

    hero_names = [
        "White Bird Shirt",
        "Blue Button-Down",
        "Short Sleeve Navy",
        "Mint Green Shirt",
        "Green T-Shirt",
        "Shiny Gold Formal",
    ]
    master_to_combos: List[Any] = []
    for i, name in enumerate(hero_names):
        # First hero produces 40 combos; the rest produce 12 each.
        n = 40 if i == 0 else 12
        master_to_combos.append((name, [make_combo(name, k) for k in range(n)]))

    pool, counts = _build_diverse_combo_pool(
        master_to_combos, per_master_min=6, global_max=60
    )

    # Total cap respected.
    assert len(pool) <= 60

    # Every hero got at least its quota (6) — that's the whole point.
    for name in hero_names:
        assert counts.get(name, 0) == 6, f"{name} got {counts.get(name)}"

    unique_heroes = {c["top"]["name"] for c in pool}
    assert len(unique_heroes) >= 4, f"only {len(unique_heroes)} unique heroes"
    assert len(unique_heroes) == 6  # all six should be represented


def test_combo_pool_dedupes_by_combo_id():
    from brain.outfit_pipeline import _build_diverse_combo_pool

    dup = {"combo_id": "dup", "top": {"name": "Shirt A"}}
    master_to_combos = [
        ("Shirt A", [dup, dup, dup]),
        ("Shirt B", [{"combo_id": f"b_{i}", "top": {"name": "Shirt B"}} for i in range(3)]),
    ]
    pool, counts = _build_diverse_combo_pool(
        master_to_combos, per_master_min=6, global_max=60
    )
    assert counts["Shirt A"] == 1
    assert counts["Shirt B"] == 3
    assert len(pool) == 4


def test_combo_pool_empty_inputs_safe():
    from brain.outfit_pipeline import _build_diverse_combo_pool

    pool, counts = _build_diverse_combo_pool([], per_master_min=6, global_max=60)
    assert pool == []
    assert counts == {}


def test_forbidden_signal_card_is_filtered_out():
    brief = build_brief(query="gym workout")
    good = {
        "id": "good",
        "title": "Training",
        "items": [
            {"id": "t1", "name": "Tee", "category": "tee", "role": "top"},
            {"id": "b1", "name": "Track Pants", "category": "track", "role": "bottom"},
            {"id": "f1", "name": "Sneakers", "category": "sneakers", "role": "footwear"},
        ],
    }
    bad = {
        "id": "bad",
        "title": "Boardroom",
        "items": [
            {"id": "t2", "name": "Shirt", "category": "button down", "role": "top"},
            {"id": "b2", "name": "Trouser", "category": "trousers", "role": "bottom"},
            {"id": "f2", "name": "Loafers", "category": "loafers", "role": "footwear"},
        ],
    }
    chosen = select_board_set([good, bad], brief, max_n=3)
    titles = [c["title"] for c in chosen]
    assert "Training" in titles
    assert "Boardroom" not in titles
