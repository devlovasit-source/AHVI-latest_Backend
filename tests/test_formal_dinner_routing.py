from __future__ import annotations

from brain import outfit_pipeline
from brain.engines import outfit_quality_guard as oqg
from services.style_flow_service import (
    _ahvi_missing_occasion_response,
    finalize_style_response_payload,
    interpret_occasion,
)


def _item(name: str, role: str, category: str) -> dict:
    return {"id": name.lower().replace(" ", "_"), "name": name, "role": role, "category": category}


def _casual_board() -> dict:
    items = [
        _item("White T-shirt", "top", "Tops"),
        _item("Blue Jeans", "bottom", "Bottoms"),
        _item("Black Loafers", "footwear", "Footwear"),
    ]
    return {"items": items, "top": items[0], "bottom": items[1], "footwear": items[2]}


def _formal_board() -> dict:
    items = [
        _item("White Oxford Formal Shirt", "top", "Tops"),
        _item("Navy Tailored Trousers", "bottom", "Bottoms"),
        _item("Black Oxford Formal Shoes", "footwear", "Footwear"),
    ]
    return {"items": items, "top": items[0], "bottom": items[1], "footwear": items[2]}


def test_formal_dinner_uses_existing_high_polish_archetype():
    interpretation = interpret_occasion(
        "Create a refined formal dinner outfit using only my wardrobe."
    )

    assert interpretation["occasion"] == "client_dinner"
    assert interpretation["formality_target"] >= 3.4
    assert "formal dinner" in interpretation["resolved_brief"]


def test_ordinary_date_night_remains_date_night():
    interpretation = interpret_occasion("date night")

    assert interpretation["occasion"] == "date_night"
    assert interpretation["formality_target"] == 3.1


def test_client_dinner_survives_pipeline_occasion_normalization():
    query = "Create a refined formal dinner outfit using only my wardrobe."
    interpretation = interpret_occasion(query)

    normalized = outfit_pipeline._normalize_pipeline_occasion(
        "client_dinner",
        {
            "occasion": "client_dinner",
            "occasion_interpretation": interpretation,
            "query": query,
        },
    )

    assert normalized == "client_dinner"


def test_client_dinner_reaches_pipeline_scoring_and_quality_guard(monkeypatch):
    calls = {"rank": 0, "guard": 0}
    original_guard = outfit_pipeline.filter_and_guard_outfits

    def rank_outfits(*, user_id, outfits, top_n):
        calls["rank"] += 1
        return list(outfits)[:top_n]

    def guard_outfits(outfits, **kwargs):
        calls["guard"] += 1
        return original_guard(outfits, **kwargs)

    monkeypatch.setattr(outfit_pipeline, "_semantic_retrieval", lambda **_kwargs: ([], {}))
    monkeypatch.setattr(outfit_pipeline, "_load_user_memory", lambda _user_id: outfit_pipeline._default_user_memory())
    monkeypatch.setattr(outfit_pipeline, "_save_user_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outfit_pipeline, "_index_outfit_vector", lambda **_kwargs: None)
    monkeypatch.setattr(outfit_pipeline.outfit_ranker, "rank", rank_outfits)
    monkeypatch.setattr(outfit_pipeline, "filter_and_guard_outfits", guard_outfits)
    monkeypatch.setenv("ENABLE_CLOSED_LOOP_FIX", "false")

    result = outfit_pipeline.get_daily_outfits(
        {
            "user_id": "formal-pipeline-test",
            "wardrobe": _formal_board()["items"],
            "context": {
                "occasion": "client_dinner",
                "query": "Create a refined formal dinner outfit using only my wardrobe.",
                "occasion_interpretation": interpret_occasion(
                    "Create a refined formal dinner outfit using only my wardrobe."
                ),
                "requested_board_count": 1,
                "raw_candidate_target": 24,
            },
        }
    )

    assert calls == {"rank": 1, "guard": 1}
    assert result["cards"]


def test_t_shirt_is_not_a_polished_shirt_token():
    top = _item("White T-shirt", "top", "Tops")
    bottom = _item("Dark Jeans", "bottom", "Bottoms")
    footwear = _item("Black Loafers", "footwear", "Footwear")

    _, reasons = oqg._contextual_occasion_weather_adjustment(
        top=top,
        bottom=bottom,
        footwear=footwear,
        accessories=[],
        occasion_text="date_night",
        query="date night",
        outfit={},
    )

    assert "Shirt and polished footwear improve smart-casual balance" not in reasons


def test_formal_dinner_does_not_activate_office_context():
    assert oqg._is_office_context("formal dinner") is False


def test_t_shirt_jeans_loafers_rejected_for_formal_dinner():
    allowed, _penalty, reasons, _fixed = oqg.guard_outfit(
        _casual_board(),
        intent="client_dinner",
        query="Create a refined formal dinner outfit using only my wardrobe.",
    )

    assert allowed is False
    assert "formal_dinner_forbidden_t_shirt" in reasons


def test_same_board_remains_eligible_for_casual_dinner():
    allowed, _penalty, _reasons, _fixed = oqg.guard_outfit(
        _casual_board(),
        intent="casual_dinner",
        query="Create a casual dinner outfit using my wardrobe.",
    )

    assert allowed is True


def test_formal_combination_remains_eligible_for_client_dinner():
    allowed, _penalty, reasons, _fixed = oqg.guard_outfit(
        _formal_board(),
        intent="client_dinner",
        query="Create a refined formal dinner outfit using only my wardrobe.",
    )

    assert allowed is True, reasons


def test_formal_dinner_gap_does_not_attach_closest_board():
    response = _ahvi_missing_occasion_response(
        "client_dinner",
        {"top": 1, "bottom": 1, "footwear": 1, "accessory": 0, "outerwear": 0, "total": 3},
        closest_board=_casual_board(),
    )

    assert response["success"] is False
    assert response["type"] == "missing_occasion_wardrobe"
    assert response["cards"] == []
    assert response["data"]["weak_occasion_match"] is False


def test_formal_dinner_finalizer_returns_clean_gap_instead_of_closest_board():
    board = _casual_board()
    response = finalize_style_response_payload(
        {"cards": [board], "outfits": [board]},
        user_id="u1",
        query="Create a refined formal dinner outfit using only my wardrobe.",
        wardrobe=board["items"],
        context={
            "occasion": "client_dinner",
            "occasion_interpretation": interpret_occasion(
                "Create a refined formal dinner outfit using only my wardrobe."
            ),
        },
        style_action="show_closest_option",
    )

    assert response["success"] is False
    assert response["type"] == "missing_occasion_wardrobe"
    assert response["cards"] == []
    assert response["style_boards"] == []
    assert response["data"]["occasion"] == "client_dinner"
    assert response["data"]["missing_items"]
    assert not response["data"].get("missing_slots")
    assert response["data"].get("closest_board") is None


def test_complete_casual_wardrobe_returns_formal_dinner_occasion_gap():
    board = _casual_board()

    response = finalize_style_response_payload(
        {"cards": [board], "outfits": [board]},
        user_id="u1",
        query="Create a refined formal dinner outfit using only my wardrobe.",
        wardrobe=board["items"],
        context={
            "occasion": "client_dinner",
            "occasion_interpretation": interpret_occasion(
                "Create a refined formal dinner outfit using only my wardrobe."
            ),
        },
    )

    assert response["type"] == "missing_occasion_wardrobe"
    assert response["data"]["occasion"] == "client_dinner"
    assert response["data"]["slot_counts"]["top"] == 1
    assert response["data"]["slot_counts"]["bottom"] == 1
    assert response["data"]["slot_counts"]["footwear"] == 1
    assert response["data"]["missing_items"]
    assert not response["data"].get("missing_slots")
    assert response["cards"] == []
    assert response["style_boards"] == []
    assert response["data"].get("closest_board") is None


def test_wardrobe_missing_only_footwear_returns_literal_core_gap():
    wardrobe = [
        _item("White Oxford Formal Shirt", "top", "Tops"),
        _item("Navy Tailored Trousers", "bottom", "Bottoms"),
    ]

    response = finalize_style_response_payload(
        {"cards": [], "outfits": []},
        user_id="u1",
        query="Create a refined formal dinner outfit using only my wardrobe.",
        wardrobe=wardrobe,
        context={
            "occasion": "client_dinner",
            "occasion_interpretation": interpret_occasion(
                "Create a refined formal dinner outfit using only my wardrobe."
            ),
        },
    )

    assert response["type"] == "missing_core_wardrobe_slots"
    assert response["data"]["missing_slots"] == ["footwear"]
    assert "still need footwear" in response["message"]


def test_valid_formal_combination_can_produce_a_board():
    board = _formal_board()

    response = finalize_style_response_payload(
        {"cards": [board], "outfits": [board]},
        user_id="u1",
        query="Create a refined formal dinner outfit using only my wardrobe.",
        wardrobe=board["items"],
        context={
            "occasion": "client_dinner",
            "occasion_interpretation": interpret_occasion(
                "Create a refined formal dinner outfit using only my wardrobe."
            ),
        },
    )

    assert response.get("type") != "missing_occasion_wardrobe"
    assert len(response["cards"]) == 1
