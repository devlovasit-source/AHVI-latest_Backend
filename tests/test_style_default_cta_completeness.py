"""Default Style CTA -> complete validated outfits.

"Suggest a complete outfit for me today." is a generation request, not advice.
Every returned board must be a real outfit (top+bottom+footwear OR dress+footwear);
accessories never fill a core slot. The universal gate repairs or returns a typed
no_complete_outfit failure — never ships an incomplete board as a success.
Nothing here touches the network.
"""

import logging

from routers.chat import (
    _apply_style_compliance_gate,
    _enforce_generic_completeness,
    _is_complete_outfit_cta,
)
from services.style_reasoning_engine import _extract_anchor_piece


def _it(name, category, source="wardrobe", **extra):
    row = {"id": name.lower().replace(" ", "-"), "name": name,
           "category": category, "source": source}
    row.update(extra)
    return row


TOP = _it("White Cotton Shirt", "shirt")
BOTTOM = _it("Blue Denim Jeans", "jeans")
SHOE = _it("Brown Leather Loafers", "loafers")
DRESS = _it("Midnight Silk Dress", "dress")
HEELS = _it("Black Leather Heels", "heels")
CAP = _it("Baseball Cap", "cap")
WOMENS_TOP = _it("Women's Floral Blouse", "blouse", gender="female")
CTA = "Suggest a complete outfit for me today."


def _board_resp(cards, occasion="today", **meta):
    return {"success": True, "cards": list(cards), "style_boards": list(cards),
            "board": "style", "meta": {"occasion": occasion, **meta}}


# 1. routing recognises the CTA as generation, not advice

def test_default_cta_is_recognised():
    assert _is_complete_outfit_cta(CTA) is True
    assert _is_complete_outfit_cta("what should I wear today") is True
    assert _is_complete_outfit_cta("style me for today") is True


def test_specific_and_advice_prompts_are_not_the_default_cta():
    assert _is_complete_outfit_cta("Create a dinner outfit with a dress") is False
    assert _is_complete_outfit_cta("what colours suit me") is False
    assert _is_complete_outfit_cta("show me shoes") is False


# 2. generic CTA repaired to top + bottom + footwear

def test_generic_cta_repairs_to_complete_core():
    bad = [{"title": "Look", "items": [BOTTOM, SHOE, CAP]}]  # missing top
    out = _apply_style_compliance_gate(
        _board_resp(bad), query=CTA, user_id="u", wardrobe=[TOP]
    )
    assert out.get("type") != "no_complete_outfit"
    from brain.engines.outfit_quality_guard import is_complete_board
    assert is_complete_board(out["cards"][0]["items"])


# 3. dress + footwear accepted

def test_dress_plus_footwear_is_accepted():
    cards = [{"title": "Look", "items": [DRESS, HEELS]}]
    out = _apply_style_compliance_gate(
        _board_resp(cards), query=CTA, user_id="u", wardrobe=[]
    )
    assert out.get("type") != "no_complete_outfit"
    assert len(out["cards"]) == 1


# 4. bottom + footwear + accessory rejected when unrepairable

def test_bottom_footwear_accessory_is_rejected_when_unrepairable():
    bad = [{"title": "Look", "items": [BOTTOM, SHOE, CAP]}]
    out = _apply_style_compliance_gate(
        _board_resp(bad), query=CTA, user_id="u", wardrobe=[]
    )
    assert out["success"] is False
    assert out["type"] == "no_complete_outfit"
    assert out["cards"] == []


# 5. every card validated independently

def test_all_cards_validated_independently():
    # Good is a one-piece (dress+heels) so it donates no top; Bad is missing a
    # top with none available anywhere -> Bad drops, Good stays.
    good = {"title": "Good", "items": [DRESS, HEELS]}
    bad = {"title": "Bad", "items": [BOTTOM, SHOE, CAP]}
    out = _apply_style_compliance_gate(
        _board_resp([good, bad]), query=CTA, user_id="u", wardrobe=[]
    )
    titles = [c["title"] for c in out["cards"]]
    assert "Good" in titles
    assert "Bad" not in titles  # incomplete sibling dropped, complete one kept


# 6-8. truncation-fallback behaviour (the gate is where fallback converges)

def test_truncation_fallback_style_incomplete_boards_never_ship():
    # Simulates the validator fallback handing incomplete cards to the serializer.
    bad = [{"title": f"L{i}", "items": [BOTTOM, SHOE, CAP]} for i in range(3)]
    out = _apply_style_compliance_gate(
        _board_resp(bad), query=CTA, user_id="u", wardrobe=[]
    )
    assert out["type"] == "no_complete_outfit"
    assert out["cards"] == []  # original unvalidated cards NOT restored


def test_fallback_emits_outcome_trace(caplog):
    bad = [{"title": "L", "items": [BOTTOM, SHOE, CAP]}]
    with caplog.at_level(logging.INFO):
        _apply_style_compliance_gate(
            _board_resp(bad), query=CTA, user_id="user_secret", wardrobe=[]
        )
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "AHVI_STYLE_OUTCOME_TRACE" in text
    assert "no_complete_outfit" in text
    assert "user_secret" not in text  # raw id never logged


# 9. default CTA capped at 2 boards

def test_default_cta_caps_at_two_boards():
    many = [{"title": f"L{i}", "items": [TOP, BOTTOM, SHOE]} for i in range(4)]
    out = _apply_style_compliance_gate(
        _board_resp(many), query=CTA, user_id="u", wardrobe=[], default_cta=True
    )
    assert len(out["cards"]) == 2


# 10. copy/items describe only final validated boards

def test_only_complete_boards_survive_for_copy():
    good = {"title": "Good", "items": [TOP, BOTTOM, SHOE]}
    bad = {"title": "Bad", "items": [BOTTOM, CAP]}
    out = _apply_style_compliance_gate(
        _board_resp([good, bad]), query=CTA, user_id="u", wardrobe=[]
    )
    for card in out["cards"]:
        from brain.engines.outfit_quality_guard import is_complete_board
        assert is_complete_board(card["items"])


# 11. generic CTA creates no false anchor

def test_generic_cta_creates_no_false_anchor():
    assert _extract_anchor_piece("suggest a complete outfit for me today") == ""
    assert _extract_anchor_piece("style me for today") == ""


def test_real_garment_anchor_is_still_extracted():
    assert "blazer" in _extract_anchor_piece("how do i style my navy blazer")


# 12. male final board never gains female assets via repair

def test_male_board_repair_excludes_female_assets():
    bad = [{"title": "Look", "items": [BOTTOM, SHOE]}]  # missing top
    # Only a women's top is available -> must NOT be pulled into a male board.
    out = _apply_style_compliance_gate(
        _board_resp(bad, style_gender="male"),
        query=CTA, user_id="u", wardrobe=[WOMENS_TOP],
    )
    assert out["type"] == "no_complete_outfit"


def test_completeness_gender_filter_prefers_matching_gender():
    complete, missing = _enforce_generic_completeness(
        [{"title": "L", "items": [BOTTOM, SHOE]}],
        wardrobe=[WOMENS_TOP, TOP],
        gender="male",
    )
    assert len(complete) == 1
    names = {i["name"] for i in complete[0]["items"]}
    assert TOP["name"] in names
    assert WOMENS_TOP["name"] not in names


# 14. non-CTA style paths untouched by the CTA routing helper

def test_generic_complete_boards_pass_through_unchanged():
    cards = [{"title": "Look", "items": [TOP, BOTTOM, SHOE]}]
    out = _apply_style_compliance_gate(
        _board_resp(cards), query="show me a smart casual look", user_id="u", wardrobe=[]
    )
    assert out["cards"][0]["title"] == "Look"
    assert out.get("type") != "no_complete_outfit"
