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
