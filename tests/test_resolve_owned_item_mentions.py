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
