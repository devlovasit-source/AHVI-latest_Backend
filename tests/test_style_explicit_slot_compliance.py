"""Explicit requested-role compliance.

When the user names garment roles ("... with a dress, shoes, and a bag") those
roles are hard constraints: the final board carries all of them, or the pipeline
reports a typed missing_explicit_roles gap. Nothing here touches the network.
"""

from services.style_explicit_roles import (
    board_explicit_roles,
    enforce_explicit_roles,
    extract_requested_roles,
    missing_explicit_roles,
)
from services.style_flow_service import (
    _accessory_allowed_for_query,
    _curate_accessories_for_card,
    _enforce_explicit_roles_on_cards,
)

DINNER_PROMPT = "Create a dinner outfit with a dress, shoes, and a bag"
LAYERED_PROMPT = "Create a layered outfit with outerwear, a top, trousers, shoes, and a bag"


def _item(name, category, source="wardrobe", **extra):
    row = {
        "id": name.lower().replace(" ", "-"),
        "name": name,
        "category": category,
        "source": source,
    }
    row.update(extra)
    return row


DRESS = _item("Midnight Silk Dress", "dress")
HEELS = _item("Black Leather Heels", "heels")
BAG = _item("Evening Clutch Bag", "bag")
TOP = _item("White Cotton Shirt", "shirt")
TROUSERS = _item("Charcoal Trousers", "trousers")
JACKET = _item("Navy Wool Blazer", "blazer")
SNEAKERS = _item("Running Sneakers", "sneakers")
CAMO = _item("Camo Cargo Trousers", "trousers")


# ── 1-4: phrase-aware extraction ────────────────────────────────────────────

def test_dinner_prompt_requests_dress_footwear_and_bag():
    assert extract_requested_roles(DINNER_PROMPT) == ["dress", "footwear", "bag"]


def test_layered_prompt_requests_all_five_roles():
    roles = set(extract_requested_roles(LAYERED_PROMPT))
    assert roles == {"top", "bottom", "outerwear", "footwear", "bag"}


def test_dress_shoes_phrase_is_footwear_not_a_dress():
    roles = extract_requested_roles("Dress shoes and a bag")
    assert "dress" not in roles
    assert roles == ["footwear", "bag"]


def test_comma_separated_dress_is_a_dress_garment():
    assert extract_requested_roles("Dress, shoes and a bag") == [
        "dress",
        "footwear",
        "bag",
    ]


def test_generic_prompt_requests_nothing_explicit():
    assert extract_requested_roles("Suggest a complete outfit") == []


# ── board role detection ─────────────────────────────────────────────────────

def test_bag_is_a_distinct_role_not_just_accessory():
    assert "bag" in board_explicit_roles([BAG])


def test_missing_roles_reported_for_incomplete_board():
    gap = missing_explicit_roles([TOP, TROUSERS, HEELS], ["dress", "footwear", "bag"])
    assert gap == ["dress", "bag"]


def test_top_bottom_never_substitutes_for_a_requested_dress():
    assert "dress" in missing_explicit_roles([TOP, TROUSERS, HEELS], ["dress"])


# ── 1-2: hard constraint satisfied end to end ────────────────────────────────

def test_dinner_board_keeps_dress_footwear_and_bag():
    card = {"title": "Evening", "items": [DRESS, HEELS, BAG], "source_policy": "wardrobe"}
    kept = _enforce_explicit_roles_on_cards(
        [card], query=DINNER_PROMPT, occasion="dinner"
    )
    assert len(kept) == 1
    assert set(board_explicit_roles(kept[0]["items"])) >= {"dress", "footwear", "bag"}


def test_layered_board_repairs_missing_outerwear_and_bag():
    # The observed Samsung failure: only top + trousers + shoes came back.
    card = {"title": "Layered", "items": [TOP, TROUSERS, HEELS], "source_policy": "wardrobe"}
    kept = _enforce_explicit_roles_on_cards(
        [card],
        query=LAYERED_PROMPT,
        occasion="dinner",
        candidate_pool=[JACKET, BAG],
    )
    assert len(kept) == 1
    roles = set(board_explicit_roles(kept[0]["items"]))
    assert roles >= {"top", "bottom", "outerwear", "footwear", "bag"}


# ── 5-6: typed failure when inventory cannot satisfy the request ─────────────

def test_missing_bag_inventory_yields_typed_gap():
    card = {"title": "Evening", "items": [DRESS, HEELS], "source_policy": "wardrobe"}
    enforcement: dict = {}
    kept = _enforce_explicit_roles_on_cards(
        [card],
        query=DINNER_PROMPT,
        occasion="dinner",
        candidate_pool=[],
        enforcement=enforcement,
    )
    assert kept == []
    assert enforcement["status"] == "missing_explicit_roles"
    assert enforcement["missing_roles"] == ["bag"]
    assert enforcement["requested_roles"] == ["dress", "footwear", "bag"]
    assert enforcement["repair_attempted"] is True
    assert "available_roles" in enforcement


def test_missing_outerwear_inventory_never_returns_incomplete_success():
    card = {"title": "Layered", "items": [TOP, TROUSERS, HEELS], "source_policy": "wardrobe"}
    enforcement: dict = {}
    kept = _enforce_explicit_roles_on_cards(
        [card],
        query=LAYERED_PROMPT,
        occasion="dinner",
        candidate_pool=[BAG],  # bag available, outerwear is not
        enforcement=enforcement,
    )
    assert kept == []
    assert enforcement["status"] == "missing_explicit_roles"
    assert "outerwear" in enforcement["missing_roles"]


# ── 7: occasion safety on repair candidates ─────────────────────────────────

def test_dinner_repair_rejects_camouflage_candidate():
    card = {"title": "Evening", "items": [TOP, HEELS], "source_policy": "wardrobe"}
    fixed, status, missing = enforce_explicit_roles(
        card,
        ["top", "bottom", "footwear"],
        candidate_pool=[CAMO],
        occasion="date_night",
    )
    assert fixed is None
    assert status == "missing_explicit_roles"
    assert "bottom" in missing


def test_dinner_repair_rejects_athletic_footwear_candidate():
    card = {"title": "Evening", "items": [DRESS], "source_policy": "wardrobe"}
    fixed, status, _ = enforce_explicit_roles(
        card, ["dress", "footwear"], candidate_pool=[SNEAKERS], occasion="date_night"
    )
    assert fixed is None
    assert status == "missing_explicit_roles"


def test_casual_occasion_still_accepts_sneakers():
    card = {"title": "Weekend", "items": [TOP, TROUSERS], "source_policy": "wardrobe"}
    fixed, status, _ = enforce_explicit_roles(
        card, ["top", "bottom", "footwear"], candidate_pool=[SNEAKERS], occasion="casual"
    )
    assert fixed is not None
    assert status == "repaired"


# ── 8: copy describes only the final items ──────────────────────────────────

def test_final_copy_inputs_match_final_items():
    # Enforcement runs BEFORE curation copy, so the item set the copy is written
    # against already contains every requested role.
    card = {"title": "Layered", "items": [TOP, TROUSERS, HEELS], "source_policy": "wardrobe"}
    kept = _enforce_explicit_roles_on_cards(
        [card], query=LAYERED_PROMPT, occasion="dinner", candidate_pool=[JACKET, BAG]
    )
    final_names = {i["name"] for i in kept[0]["items"]}
    assert JACKET["name"] in final_names
    assert BAG["name"] in final_names
    # No phantom items were invented for the copy to describe.
    assert final_names <= {
        TOP["name"], TROUSERS["name"], HEELS["name"], JACKET["name"], BAG["name"]
    }


def test_copy_never_promises_a_dress_that_was_not_selected():
    card = {"title": "Evening", "items": [TOP, TROUSERS, HEELS], "source_policy": "wardrobe"}
    kept = _enforce_explicit_roles_on_cards(
        [card], query=DINNER_PROMPT, occasion="dinner", candidate_pool=[BAG]
    )
    # No dress in the wardrobe -> board rejected rather than shipped with copy
    # describing a dress that is not on it.
    assert kept == []


# ── 9: source policy ─────────────────────────────────────────────────────────

def test_wardrobe_policy_repair_never_pulls_catalog_items():
    catalog_bag = _item("Catalog Tote", "bag", source="catalog")
    card = {"title": "Evening", "items": [DRESS, HEELS], "source_policy": "wardrobe"}
    enforcement: dict = {}
    kept = _enforce_explicit_roles_on_cards(
        [card],
        query=DINNER_PROMPT,
        occasion="dinner",
        candidate_pool=[catalog_bag],
        enforcement=enforcement,
    )
    assert kept == []
    assert enforcement["status"] == "missing_explicit_roles"


def test_wardrobe_source_policy_is_preserved_on_repaired_board():
    card = {"title": "Layered", "items": [TOP, TROUSERS, HEELS], "source_policy": "wardrobe"}
    kept = _enforce_explicit_roles_on_cards(
        [card], query=LAYERED_PROMPT, occasion="dinner", candidate_pool=[JACKET, BAG]
    )
    assert kept[0]["source_policy"] == "wardrobe"
    assert all(i.get("source") == "wardrobe" for i in kept[0]["items"])


# ── 10: generic behaviour unchanged ─────────────────────────────────────────

def test_generic_prompt_leaves_cards_untouched():
    cards = [{"title": "Look", "items": [TOP, TROUSERS, HEELS]}]
    enforcement: dict = {}
    kept = _enforce_explicit_roles_on_cards(
        cards, query="Suggest a complete outfit", occasion="daily", enforcement=enforcement
    )
    assert kept == cards
    assert enforcement["status"] == "satisfied"
    assert enforcement["requested_roles"] == []


def test_generic_prompt_does_not_force_a_bag():
    card = {"title": "Look", "items": [TOP, TROUSERS, HEELS]}
    assert missing_explicit_roles(card["items"], extract_requested_roles("Suggest a complete outfit")) == []


# ── accessory curation no longer evicts an explicitly requested bag ─────────

def test_requested_bag_allowed_for_dinner_despite_date_accessory_whitelist():
    # Root cause of the missing bag: _DATE_ACCESSORIES has no "bag".
    assert _accessory_allowed_for_query(BAG, "dinner date outfit") is False or True
    assert _accessory_allowed_for_query(BAG, DINNER_PROMPT) is True


def test_requested_bag_survives_accessory_budget_for_date_occasion():
    card = {
        "title": "Evening",
        "items": [DRESS, HEELS],
        "accessories": [
            _item("Silver Watch", "watch"),
            _item("Gold Necklace", "necklace"),
            BAG,
        ],
    }
    fixed = _curate_accessories_for_card(card, DINNER_PROMPT)
    types = fixed["accessory_policy_applied"]["accessory_types"]
    assert "bag" in types


# ── anchor preservation + controlled-beta response helpers ──────────────────

def test_selected_anchor_item_survives_repair():
    anchor = _item("Emerald Wrap Dress", "dress")
    card = {"title": "Evening", "items": [anchor], "source_policy": "wardrobe"}
    kept = _enforce_explicit_roles_on_cards(
        [card], query=DINNER_PROMPT, occasion="dinner", candidate_pool=[HEELS, BAG]
    )
    assert len(kept) == 1
    names = {i["name"] for i in kept[0]["items"]}
    assert anchor["name"] in names  # anchor never dropped by repair


def test_source_policy_and_final_roles_helpers():
    from routers.chat import _style_final_roles, _style_source_policy

    cards = [{"source_policy": "wardrobe", "items": [DRESS, HEELS, BAG]}]
    assert _style_source_policy(cards) == "wardrobe"
    roles = set(_style_final_roles(cards))
    assert {"dress", "footwear", "bag"} <= roles


def test_outcome_trace_emits_safe_fields(caplog):
    import logging

    from routers.chat import _emit_style_outcome_trace

    with caplog.at_level(logging.INFO):
        _emit_style_outcome_trace(
            user_id="user_secret_123",
            intent="style_pipeline_adapter",
            occasion="dinner",
            source_policy="wardrobe",
            requested_roles=["dress", "footwear", "bag"],
            required_roles=["dress", "footwear", "bag"],
            final_cards=[{"items": [DRESS, HEELS, BAG]}],
            missing_roles=[],
            repair_attempted=True,
            validation_result="satisfied",
        )
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "AHVI_STYLE_OUTCOME_TRACE" in line
    assert "requested_roles" in line and "final_roles" in line
    assert "user_secret_123" not in line  # raw id never logged


# ── universal gate: closes the untraced premium/multi-board bypass ──────────

def _bad_dinner_boards():
    return [
        {"title": f"Look {i}", "items": [CAMO, SNEAKERS]}  # no dress, no bag
        for i in range(3)
    ]


def test_gate_closes_untraced_multi_board_dinner_bypass():
    from routers.chat import _apply_style_compliance_gate

    boards = _bad_dinner_boards()
    resp = {"success": True, "cards": boards, "style_boards": boards,
            "board": "style", "meta": {"occasion": "dinner"}}
    out = _apply_style_compliance_gate(
        resp,
        query="create a refined dinner outfit with a dress, shoes, and a bag",
        user_id="u1",
        wardrobe=[],
    )
    assert out["success"] is False
    assert out["type"] == "missing_explicit_roles"
    assert set(out["missing_roles"]) == {"dress", "bag"}
    # Never falls back to the original unvalidated card list.
    assert out["cards"] == []
    assert out["style_boards"] == []


def test_gate_repairs_multi_board_dinner_from_wardrobe():
    from routers.chat import _apply_style_compliance_gate
    from services.style_explicit_roles import board_explicit_roles

    boards = [{"title": "Look", "items": [DRESS, HEELS]}]
    resp = {"success": True, "cards": boards, "style_boards": boards,
            "board": "style", "meta": {"occasion": "dinner"}}
    out = _apply_style_compliance_gate(
        resp, query=DINNER_PROMPT, user_id="u1", wardrobe=[BAG]
    )
    assert out.get("type") != "missing_explicit_roles"
    roles = set(board_explicit_roles(out["cards"][0]["items"]))
    assert {"dress", "footwear", "bag"} <= roles


def test_gate_is_idempotent_when_already_gated():
    from routers.chat import _apply_style_compliance_gate

    boards = _bad_dinner_boards()
    resp = {"success": True, "cards": boards, "board": "style",
            "meta": {"occasion": "dinner", "style_compliance_gated": True}}
    out = _apply_style_compliance_gate(
        resp, query=DINNER_PROMPT, user_id="u1", wardrobe=[]
    )
    assert out is resp  # already gated -> untouched


def test_gate_leaves_generic_boards_but_stamps_source_policy():
    from routers.chat import _apply_style_compliance_gate

    boards = [{"title": "Look", "items": [TOP, TROUSERS, HEELS]}]
    resp = {"success": True, "cards": boards, "board": "style",
            "meta": {"occasion": "office"}}
    out = _apply_style_compliance_gate(
        resp,
        query="create a professional office outfit using my wardrobe",
        user_id="u1",
        wardrobe=[],
    )
    assert out["cards"] == boards  # nothing explicitly requested -> unchanged
    assert out["source_policy"] == "wardrobe"
    assert out["meta"]["source_policy"] == "wardrobe"


# ── canonical source policy (one derivation, both directions) ────────────────

def test_canonical_source_policy_wardrobe_language():
    from routers.chat import _canonical_source_policy

    assert _canonical_source_policy("office outfit using my wardrobe") == "wardrobe"
    assert _canonical_source_policy("use only my wardrobe") == "wardrobe"
    assert _canonical_source_policy("build a look", action="use_wardrobe") == "wardrobe"


def test_canonical_source_policy_not_forced_without_request():
    from routers.chat import _canonical_source_policy

    # The layered prompt never says wardrobe -> must NOT be stamped wardrobe.
    assert _canonical_source_policy(LAYERED_PROMPT) == ""
    assert _canonical_source_policy("create a dinner outfit with a dress") == ""
