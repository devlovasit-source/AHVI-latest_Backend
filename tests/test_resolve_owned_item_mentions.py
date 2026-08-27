"""Unit tests for services.style_item_contract.resolve_owned_item_mentions --
the canonical first-turn owned-item resolver (see GO IMPLEMENT audit, item 3)."""
from __future__ import annotations

from services.style_item_contract import resolve_owned_item_mentions


def _it(name, role, item_id=None, source="wardrobe"):
    iid = item_id or name.lower().replace(" ", "-")
    return {"id": iid, "item_id": iid, "name": name, "role": role, "category": role, "source": source}


RED_TOP = _it("Red Top", "top")
WHITE_SHIRT = _it("White Shirt", "top")
LEOPARD_SKIRT = _it("Leopard Print Skirt", "bottom")
BLUE_JEANS = _it("Blue Jeans", "bottom")
LOAFERS = _it("Brown Loafers", "footwear")
WARDROBE = [RED_TOP, WHITE_SHIRT, LEOPARD_SKIRT, BLUE_JEANS, LOAFERS]

BLACK_SHIRT_1 = _it("Black Shirt", "top", item_id="black-shirt-1")
BLACK_SHIRT_2 = _it("Black Shirt", "top", item_id="black-shirt-2")


def test_exact_single_item_match():
    result = resolve_owned_item_mentions("Create an outfit using my Red Top", WARDROBE)
    assert result["ambiguous"] == []
    assert result["unresolved"] == []
    assert len(result["resolved"]) == 1
    hit = result["resolved"][0]
    assert hit["item_id"] == "red-top"
    assert hit["role"] == "top"
    assert hit["source"] == "wardrobe"
    assert hit["match_type"] == "exact"


def test_normalized_match_ignores_case_and_punctuation():
    result = resolve_owned_item_mentions("style me with my RED-TOP please", WARDROBE)
    ids = [r["item_id"] for r in result["resolved"]]
    assert "red-top" in ids


def test_two_named_items_both_resolve():
    result = resolve_owned_item_mentions(
        "Style me using my Red Top and Leopard Print Skirt", WARDROBE
    )
    ids = {r["item_id"] for r in result["resolved"]}
    assert ids == {"red-top", "leopard-print-skirt"}
    assert result["ambiguous"] == []
    assert result["unresolved"] == []


def test_longer_name_wins_over_shorter_contained_name():
    wardrobe = WARDROBE + [_it("Top", "top", item_id="generic-top")]
    result = resolve_owned_item_mentions("Create an outfit using my Red Top", wardrobe)
    ids = [r["item_id"] for r in result["resolved"]]
    assert ids == ["red-top"]
    assert "generic-top" not in ids


def test_bare_role_word_alone_never_arbitrarily_resolves():
    result = resolve_owned_item_mentions("what goes with a top", WARDROBE)
    assert result["resolved"] == []
    assert result["ambiguous"] == []


def test_ambiguous_same_name_items_are_flagged_not_silently_picked():
    wardrobe = [BLACK_SHIRT_1, BLACK_SHIRT_2, BLUE_JEANS, LOAFERS]
    result = resolve_owned_item_mentions("Create an outfit using my Black Shirt", wardrobe)
    assert result["resolved"] == []
    assert len(result["ambiguous"]) == 1
    entry = result["ambiguous"][0]
    assert entry["mention"] == "Black Shirt"
    assert set(entry["candidate_item_ids"]) == {"black-shirt-1", "black-shirt-2"}
    assert entry["match_type"] == "ambiguous"


def test_no_match_owned_item_is_explicitly_unresolved_not_substituted():
    result = resolve_owned_item_mentions("Style me using my Purple Spacesuit", WARDROBE)
    assert result["resolved"] == []
    assert result["ambiguous"] == []
    assert len(result["unresolved"]) == 1
    assert result["unresolved"][0]["mention"].lower() == "purple spacesuit"
    assert result["unresolved"][0]["match_type"] == "none"


def test_no_mentions_at_all_returns_empty_everything():
    result = resolve_owned_item_mentions("What will suit my body type?", WARDROBE)
    assert result == {"resolved": [], "ambiguous": [], "unresolved": []}


def test_items_missing_canonical_id_are_skipped_not_crashed_on():
    broken = {"name": "Ghost Item", "role": "top"}  # no id/item_id field
    result = resolve_owned_item_mentions("style me using my Ghost Item", [broken])
    assert result["resolved"] == []


# ── Hardening: request-changes review round 2 ──────────────────────────────

def test_mixed_owned_and_unowned_second_item_no_silent_bottom_substitution():
    """"my Red Top and Purple Skirt" -- Red Top is owned, Purple Skirt is not.
    Red Top must resolve; Purple Skirt must be explicitly unresolved, not
    silently dropped or backfilled with some other owned bottom."""
    result = resolve_owned_item_mentions(
        "Style me using my Red Top and Purple Skirt", WARDROBE
    )
    resolved_ids = {r["item_id"] for r in result["resolved"]}
    assert resolved_ids == {"red-top"}
    assert result["ambiguous"] == []
    assert len(result["unresolved"]) == 1
    assert result["unresolved"][0]["mention"].lower() == "purple skirt"
    # Nothing from the wardrobe (e.g. Leopard Print Skirt / Blue Jeans) was
    # silently pulled in to stand in for the unresolved "Purple Skirt".
    assert "leopard-print-skirt" not in resolved_ids
    assert "blue-jeans" not in resolved_ids


def test_ampersand_conjunction_supported():
    result = resolve_owned_item_mentions(
        "Style me using my Red Top & Leopard Print Skirt", WARDROBE
    )
    ids = {r["item_id"] for r in result["resolved"]}
    assert ids == {"red-top", "leopard-print-skirt"}
    assert result["unresolved"] == []


def test_ampersand_conjunction_with_one_unowned_item():
    result = resolve_owned_item_mentions(
        "Style me using my Red Top & Purple Skirt", WARDROBE
    )
    resolved_ids = {r["item_id"] for r in result["resolved"]}
    assert resolved_ids == {"red-top"}
    assert len(result["unresolved"]) == 1
    assert result["unresolved"][0]["mention"].lower() == "purple skirt"


def test_ambiguous_candidate_ids_are_sorted_deterministic():
    # item_ids intentionally out of sort order in wardrobe list.
    wardrobe = [
        _it("Black Shirt", "top", item_id="zzz-shirt"),
        _it("Black Shirt", "top", item_id="aaa-shirt"),
    ]
    result = resolve_owned_item_mentions("Create an outfit using my Black Shirt", wardrobe)
    assert result["ambiguous"][0]["candidate_item_ids"] == ["aaa-shirt", "zzz-shirt"]


def test_match_type_exact_reflects_literal_query_text_not_stored_name_tautology():
    result = resolve_owned_item_mentions("Create an outfit using my Red Top", WARDROBE)
    assert result["resolved"][0]["match_type"] == "exact"


def test_match_type_normalized_when_query_text_differs_from_stored_name():
    # Stored name is "Red Top"; query uses different punctuation/casing/
    # spacing, so it can only match after normalization, not literally.
    result = resolve_owned_item_mentions("style me with my RED--TOP please", WARDROBE)
    hit = next(r for r in result["resolved"] if r["item_id"] == "red-top")
    assert hit["match_type"] == "normalized"
