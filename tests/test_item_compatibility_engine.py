from unittest.mock import patch
from services.item_compatibility_engine import ItemCompatibilityEngine, canonical_role, MAX_PER_ROLE
from routers.stylist import style_wardrobe_item, get_item_compatibility, ItemStyleRequest


def item(i, name, category, color, **kw):
    return {"id": i, "name": name, "category": category, "color": color, **kw}


def ids(result):
    return [x["item_id"] for x in result]


def test_top_anchor_prefers_complementary_roles_and_colors():
    anchor = item("s", "Grey Button Down Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual"),
        item("f1", "Brown Loafers", "footwear", "brown", formality="smart_casual"),
        item("b2", "Navy Trousers", "bottoms", "navy", formality="smart_casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert set(ids(result)) == {"b1", "b2", "f1"}
    assert len({x["role"] for x in result[:3]}) >= 2


def test_footwear_anchor_does_not_saturate_shirts():
    anchor = item("f", "Brown Loafers", "footwear", "brown", formality="smart_casual")
    wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual"),
        item("b2", "Navy Chinos", "bottoms", "navy", formality="smart_casual"),
        item("t1", "White Oxford Shirt", "tops", "white", formality="smart_casual"),
        item("t2", "Blue Oxford Shirt", "tops", "blue", formality="smart_casual"),
        item("t3", "Cream Shirt", "tops", "cream", formality="smart_casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert len(result) <= 5
    assert sum(x["role"] == "top" for x in result) <= 2
    assert {x["role"] for x in result[:3]} >= {"top", "bottom"}


def test_ethnicwear_rejects_running_shoes_and_keeps_accessory_diversity():
    anchor = item("k", "Festive Kurta", "ethnicwear", "cream", formality="festive")
    wardrobe = [
        item("j", "Gold Bracelet", "jewellery", "gold", formality="festive"),
        item("f", "Festive Jutti", "footwear", "gold", formality="festive"),
        item("b", "Cream Bottom", "bottoms", "cream", formality="festive"),
        item("r", "Running Shoes", "footwear", "orange", formality="athletic"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert "r" not in ids(result)
    assert {"j", "f"}.issubset(set(ids(result)))


def test_hard_filters_same_item_deleted_and_formality_conflict():
    anchor = item("a", "Formal Blazer", "outerwear", "navy", formality="formal")
    wardrobe = [
        anchor,
        item("gym", "Gym Shorts", "bottoms", "black", formality="athletic"),
        item("deleted", "Beige Trousers", "bottoms", "beige", formality="smart_casual", deleted=True),
        item("good", "Grey Trousers", "bottoms", "grey", formality="formal"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert ids(result) == ["good"]


def test_role_cap_is_two():
    anchor = item("a", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item(str(i), f"Trouser {i}", "bottoms", "beige", formality="smart_casual") for i in range(5)
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert sum(x["role"] == "bottom" for x in result) <= 2


def test_occasion_changes_ranking_and_conflict_is_filtered():
    anchor = item("d", "Evening Dress", "dress", "black", formality="formal")
    wardrobe = [
        item("w", "Wedding Heels", "footwear", "gold", formality="festive", occasions=["wedding"]),
        item("d1", "Casual Sneakers", "footwear", "white", formality="casual", avoid_for=["wedding"]),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe, occasion="wedding")
    assert ids(result) == ["w"]


def test_reason_codes_and_score_shape():
    anchor = item("a", "Grey Shirt", "tops", "grey", formality="smart_casual")
    candidate = item("b", "Beige Trousers", "bottoms", "beige", formality="smart_casual")
    result = ItemCompatibilityEngine.rank(anchor, [candidate])
    assert result
    row = result[0]
    assert 55 <= row["match_score"] <= 100
    assert row["match_level"] in {"Excellent match", "Strong match", "Possible match"}
    assert "ROLE_COMPLEMENT" in row["reason_codes"]


def test_unknown_or_unrelated_items_are_not_exposed():
    anchor = item("a", "Grey Shirt", "tops", "grey")
    wardrobe = [
        item("x", "Laptop Charger", "other", "black"),
        item("y", "Mystery Object", "other", "neon pink"),
    ]
    assert ItemCompatibilityEngine.rank(anchor, wardrobe) == []


def test_disliked_penalty_demotes_or_filters_disliked_item():
    anchor = item("a", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual", disliked=True),
        item("b2", "Navy Trousers", "bottoms", "navy", formality="smart_casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert ids(result)[0] == "b2"


def test_duplicate_image_url_deduplication():
    anchor = item("a", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual", image_url="https://img.com/pant.jpg"),
        item("b2", "Duplicate Beige Trousers", "bottoms", "beige", formality="smart_casual", image_url="https://img.com/pant.jpg"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert len(result) == 1
    assert ids(result) == ["b1"]


def test_tag_and_season_matching_boosts_score():
    anchor = item("a", "Linen Shirt", "tops", "white", formality="casual", tags=["minimal", "breezy"], season=["summer"])
    candidate = item("b", "Linen Shorts", "bottoms", "beige", formality="casual", tags=["minimal"], season=["summer"])
    result = ItemCompatibilityEngine.rank(anchor, [candidate])
    assert result
    codes = result[0]["reason_codes"]
    assert "STYLE_MATCH" in codes
    assert "SEASON_MATCH" in codes


def test_outerwear_anchor_prefers_top_bottom_and_footwear():
    anchor = item("o", "Navy Blazer", "outerwear", "navy", formality="formal")
    wardrobe = [
        item("t", "White Oxford Shirt", "tops", "white", formality="formal"),
        item("b", "Grey Trousers", "bottoms", "grey", formality="formal"),
        item("f", "Brown Derbies", "footwear", "brown", formality="formal"),
        item("g", "Gym Shorts", "bottoms", "black", formality="athletic"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    result_ids = ids(result)
    assert {"t", "b", "f"}.issubset(set(result_ids))
    assert "g" not in result_ids


def test_dress_anchor_prefers_footwear_bag_and_accessory():
    anchor = item("d", "Black Evening Dress", "dress", "black", formality="formal")
    wardrobe = [
        item("f", "Gold Heels", "footwear", "gold", formality="formal"),
        item("b", "Black Clutch", "bag", "black", formality="formal"),
        item("a", "Gold Bracelet", "jewellery", "gold", formality="formal"),
        item("x", "Running Shorts", "bottoms", "neon", formality="athletic"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    result_ids = ids(result)
    assert {"f", "b", "a"}.issubset(set(result_ids))
    assert "x" not in result_ids


def test_accessory_anchor_prefers_garment_complements():
    anchor = item("a", "Brown Leather Belt", "accessory", "brown", formality="smart_casual")
    wardrobe = [
        item("t", "White Shirt", "tops", "white", formality="smart_casual"),
        item("b", "Beige Chinos", "bottoms", "beige", formality="smart_casual"),
        item("f", "Brown Loafers", "footwear", "brown", formality="smart_casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    result_ids = ids(result)
    assert {"t", "b", "f"}.issubset(set(result_ids))


def test_bag_anchor_prefers_dress_top_or_bottom_and_footwear():
    anchor = item("bag", "Tan Crossbody Bag", "bag", "tan", formality="casual")
    wardrobe = [
        item("d", "Cream Dress", "dress", "cream", formality="casual"),
        item("t", "White Tee", "tops", "white", formality="casual"),
        item("b", "Blue Jeans", "bottoms", "blue", formality="casual"),
        item("f", "White Sneakers", "footwear", "white", formality="casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert {"d", "t", "b", "f"}.intersection(ids(result))


def test_jewellery_anchor_prefers_dress_or_ethnicwear():
    anchor = item("j", "Gold Necklace", "jewellery", "gold", formality="festive")
    wardrobe = [
        item("d", "Emerald Dress", "dress", "green", formality="festive"),
        item("e", "Cream Kurta", "ethnicwear", "cream", formality="festive"),
        item("t", "White Shirt", "tops", "white", formality="casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    result_ids = ids(result)
    assert {"d", "e"}.issubset(set(result_ids))


def test_max_results_three_respects_role_cap():
    anchor = item("s", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual"),
        item("b2", "Navy Trousers", "bottoms", "navy", formality="smart_casual"),
        item("b3", "Black Trousers", "bottoms", "black", formality="smart_casual"),
        item("f", "Brown Loafers", "footwear", "brown", formality="smart_casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe, max_results=3)
    assert len(result) <= 3
    assert sum(x["role"] == "bottom" for x in result) <= MAX_PER_ROLE


def test_sparse_wardrobe_returns_only_valid_candidates():
    anchor = item("s", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("x", "Laptop Charger", "other", "black"),
        item("b", "Beige Trousers", "bottoms", "beige", formality="smart_casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert ids(result) == ["b"]


def test_no_valid_candidates_returns_empty():
    anchor = item("s", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("x", "Gym Shorts", "bottoms", "black", formality="athletic"),
        item("y", "Laptop Charger", "other", "black"),
        item("z", "Grey Shirt Duplicate", "tops", "grey", formality="smart_casual"),
    ]
    assert ItemCompatibilityEngine.rank(anchor, wardrobe) == []


def test_inactive_candidate_is_excluded():
    anchor = item("s", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("x", "Beige Trousers", "bottoms", "beige", formality="smart_casual", status="inactive"),
        item("y", "Navy Trousers", "bottoms", "navy", formality="smart_casual"),
    ]
    assert ids(ItemCompatibilityEngine.rank(anchor, wardrobe)) == ["y"]


def test_not_for_me_flag_is_penalized():
    anchor = item("s", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("bad", "Beige Trousers", "bottoms", "beige", formality="smart_casual", not_for_me=True),
        item("good", "Navy Trousers", "bottoms", "navy", formality="smart_casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert ids(result)[0] == "good"


def test_pixel_hash_duplicate_is_deduplicated():
    anchor = item("s", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual", pixel_hash="abc123"),
        item("b2", "Same Image Different ID", "bottoms", "beige", formality="smart_casual", pixel_hash="abc123"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert len(result) == 1
    assert ids(result) == ["b1"]


def test_avoid_for_accepts_comma_separated_string():
    anchor = item("d", "Evening Dress", "dress", "black", formality="formal")
    wardrobe = [
        item("bad", "Casual Sneakers", "footwear", "white", formality="casual", avoid_for="wedding, formal"),
        item("good", "Gold Heels", "footwear", "gold", formality="formal"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe, occasion="wedding")
    assert "bad" not in ids(result)
    assert "good" in ids(result)


def test_formality_boundary_gap_two_is_penalized_not_hard_rejected():
    anchor = item("a", "Smart Blazer", "outerwear", "navy", formality="formal")
    candidate = item("b", "Casual Chinos", "bottoms", "beige", formality="casual")
    result = ItemCompatibilityEngine.rank(anchor, [candidate])
    assert result == [] or result[0]["match_score"] < 70


def test_formality_boundary_gap_three_is_hard_rejected():
    anchor = item("a", "Formal Blazer", "outerwear", "navy", formality="formal")
    candidate = item("b", "Gym Shorts", "bottoms", "black", formality="athletic")
    assert ItemCompatibilityEngine.rank(anchor, [candidate]) == []


# ─── K16 ADDITIONAL BACKEND TESTS ───

def test_missing_metadata_does_not_automatically_score_eighty():
    """K16.1: Candidate with missing color, occasion, style, season does not inflate score."""
    anchor = item("a", "Generic Shirt", "tops", "")
    candidate = item("b", "Generic Pant", "bottoms", "")
    result = ItemCompatibilityEngine.rank(anchor, [candidate])
    # Role (30) + Formality baseline (5) = 35 < 55 -> filtered out or low score!
    assert result == [] or result[0]["match_score"] < 70


def test_boxy_shirt_is_not_rejected_by_non_fashion_filter():
    """K16.2: 'Boxy Shirt' and 'Box Pleat Skirt' must remain valid fashion items."""
    anchor = item("a", "Boxy Shirt", "tops", "white", formality="casual")
    candidate = item("b", "Box Pleat Skirt", "bottoms", "black", formality="casual")
    result = ItemCompatibilityEngine.rank(anchor, [candidate])
    assert len(result) == 1
    assert result[0]["item_id"] == "b"


def test_same_pixel_hash_with_different_urls_dedupes():
    """K16.3: Duplicate items with different URLs but same content_hash / pixel_hash dedupe."""
    anchor = item("a", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual", image_url="https://url1.com/img.jpg", content_hash="hash_xyz"),
        item("b2", "Beige Trousers Dup", "bottoms", "beige", formality="smart_casual", image_url="https://url2.com/img.jpg", content_hash="hash_xyz"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert len(result) == 1


def test_duplicate_order_keeps_best_scored_candidate():
    """K16.4: Deduplication keeps the duplicate instance with the HIGHER score."""
    anchor = item("a", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual", image_url="https://url.com/pant.jpg"),
        item("b2", "Beige Trousers Plus", "bottoms", "beige", formality="smart_casual", image_url="https://url.com/pant.jpg", liked=True),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert len(result) == 1
    assert result[0]["item_id"] == "b2"


def test_occasions_as_comma_separated_string():
    """K16.5: Occasion metadata stored as comma-separated string resolves correctly."""
    anchor = item("a", "Shirt", "tops", "blue", formality="casual")
    candidate = item("b", "Pants", "bottoms", "beige", formality="casual", occasions="office, casual, date")
    result = ItemCompatibilityEngine.rank(anchor, [candidate], occasion="office")
    assert result
    assert "OCCASION_MATCH" in result[0]["reason_codes"]


def test_mixed_case_occasions():
    """K16.6: Occasion matching handles mixed-case strings ('Office', 'Casual')."""
    anchor = item("a", "Shirt", "tops", "blue", formality="casual")
    candidate = item("b", "Pants", "bottoms", "beige", formality="casual", occasions=["Office", "Casual"])
    result = ItemCompatibilityEngine.rank(anchor, [candidate], occasion="office")
    assert result
    assert "OCCASION_MATCH" in result[0]["reason_codes"]


def test_output_contains_no_raw_nested_wardrobe_document():
    """K16.7: Response minimization (K8) omits raw 'item' payload dictionary."""
    anchor = item("a", "Grey Shirt", "tops", "grey", formality="smart_casual")
    candidate = item("b", "Beige Trousers", "bottoms", "beige", formality="smart_casual")
    result = ItemCompatibilityEngine.rank(anchor, [candidate])
    assert result
    row = result[0]
    assert "item" not in row
    assert "raw_document" not in row


def test_output_includes_canonical_image_url():
    """K16.8: Response includes canonical image_url."""
    anchor = item("a", "Grey Shirt", "tops", "grey", formality="smart_casual")
    candidate = item("b", "Beige Trousers", "bottoms", "beige", formality="smart_casual", image_url="https://cdn.com/pant.png")
    result = ItemCompatibilityEngine.rank(anchor, [candidate])
    assert result
    assert result[0]["image_url"] == "https://cdn.com/pant.png"


def test_compatibility_engine_exception_does_not_break_style_this():
    """K16.9: Engine exception during style_wardrobe_item mode=style_this is isolated."""
    anchor = item("a", "Grey Shirt", "tops", "grey")
    req = ItemStyleRequest(user_id="test_usr", mode="style_this", anchor_item=anchor, wardrobe=[anchor])
    with patch.object(ItemCompatibilityEngine, "rank", side_effect=Exception("Engine crashed")):
        res = style_wardrobe_item("a", req)
        assert res["success"] is True
        assert "style_directions" in res
        assert res["works_well_with"] == []


def test_compatibility_engine_exception_does_not_break_build_outfit():
    """K16.10: Engine exception during style_wardrobe_item mode=build_outfit is isolated."""
    anchor = item("a", "Grey Shirt", "tops", "grey")
    req = ItemStyleRequest(user_id="test_usr", mode="build_outfit", anchor_item=anchor, wardrobe=[anchor])
    with patch.object(ItemCompatibilityEngine, "rank", side_effect=Exception("Engine crashed")):
        res = style_wardrobe_item("a", req)
        assert res["success"] is True
        assert "outfit" in res
        assert res["works_well_with"] == []


def test_authoritative_anchor_metadata_beats_minimal_client_anchor():
    """K16.11: Canonical stored wardrobe record fills missing client anchor fields."""
    minimal_client_anchor = {"item_id": "b1"}
    stored_wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual"),
        item("f1", "Brown Loafers", "footwear", "brown", formality="smart_casual"),
    ]
    result = ItemCompatibilityEngine.rank(minimal_client_anchor, stored_wardrobe)
    assert result
    assert result[0]["item_id"] == "f1"


def test_dedicated_compatibility_endpoint_response_contract():
    """K16.12: Dedicated endpoint GET/POST /items/{item_id}/compatibility contract."""
    anchor = item("s1", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [anchor, item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual")]
    req = ItemStyleRequest(user_id="u1", anchor_item=anchor, wardrobe=wardrobe)
    with patch("routers.stylist.AppwriteProxy") as mock_proxy_cls:
        mock_proxy = mock_proxy_cls.return_value
        mock_proxy.list_documents.return_value = wardrobe
        res = get_item_compatibility("s1", req, x_user_id="u1")
        assert res["success"] is True
        assert res["anchor_item_id"] == "s1"
        assert "works_well_with" in res
        assert res["meta"]["engine"] == "item_compatibility_v1"


def test_first_three_recommendations_have_multiple_roles_when_inventory_allows():
    """K16.14: Top 3 recommendations prefer >= 2 distinct roles when available."""
    anchor = item("s", "Grey Shirt", "tops", "grey", formality="smart_casual")
    wardrobe = [
        item("b1", "Beige Trousers", "bottoms", "beige", formality="smart_casual"),
        item("b2", "Navy Trousers", "bottoms", "navy", formality="smart_casual"),
        item("f1", "Brown Loafers", "footwear", "brown", formality="smart_casual"),
    ]
    result = ItemCompatibilityEngine.rank(anchor, wardrobe)
    assert len(result[:3]) >= 2
    roles_in_top3 = {x["role"] for x in result[:3]}
    assert len(roles_in_top3) >= 2


def test_route_security_missing_auth_identity_returns_401():
    """Route Security 0: Missing auth header (x-user-id / x-authenticated-user) returns 401 Unauthorized."""
    import pytest
    from fastapi import HTTPException
    req = ItemStyleRequest(user_id="attacker")
    with pytest.raises(HTTPException) as exc_info:
        get_item_compatibility("item_A", req)
    assert exc_info.value.status_code == 401


def test_route_security_user_a_accesses_own_item_returns_200():
    """Route Security 1: User A requesting User A's owned item returns 200 OK."""
    with patch("routers.stylist.AppwriteProxy") as mock_proxy_cls:
        mock_proxy = mock_proxy_cls.return_value
        mock_proxy.list_documents.return_value = [
            item("item_A", "User A Shirt", "tops", "grey", formality="smart_casual"),
            item("item_B", "User A Pants", "bottoms", "beige", formality="smart_casual"),
        ]

        res = get_item_compatibility("item_A", x_user_id="user_A")
        assert res["success"] is True
        assert res["anchor_item_id"] == "item_A"
        assert len(res["works_well_with"]) == 1
        assert res["works_well_with"][0]["item_id"] == "item_B"


def test_route_security_user_b_accesses_user_a_item_raises_403():
    """Route Security 2: User B requesting User A's item raises 403 Forbidden."""
    import pytest
    from fastapi import HTTPException
    with patch("routers.stylist.AppwriteProxy") as mock_proxy_cls:
        mock_proxy = mock_proxy_cls.return_value
        mock_proxy.list_documents.side_effect = lambda collection, user_id: (
            [item("item_A", "User A Shirt", "tops", "blue")] if user_id == "user_A" else []
        )

        # User B authenticated via header attempts to access User A's item -> HTTP 403
        with pytest.raises(HTTPException) as exc_info:
            get_item_compatibility("item_A", x_user_id="user_B")
        assert exc_info.value.status_code == 403


def test_route_security_body_user_id_is_ignored_for_authorization():
    """Route Security 3: Body user_id = User B is ignored; header user_id = User A wins."""
    import pytest
    from fastapi import HTTPException
    with patch("routers.stylist.AppwriteProxy") as mock_proxy_cls:
        mock_proxy = mock_proxy_cls.return_value
        mock_proxy.list_documents.side_effect = lambda collection, user_id: (
            [item("item_A", "User A Shirt", "tops", "blue")] if user_id == "user_A" else []
        )

        req = ItemStyleRequest(user_id="user_B")  # Spoofed user_B in request body
        # Authenticated as user_A via header -> tries accessing item_B (doesn't exist in user_A wardrobe) -> HTTP 403
        with pytest.raises(HTTPException) as exc_info:
            get_item_compatibility("item_B", req, x_user_id="user_A")
        assert exc_info.value.status_code == 403


def test_route_security_fake_body_wardrobe_is_ignored():
    """Route Security 4: Fake body wardrobe is ignored; server DB wardrobe wins."""
    with patch("routers.stylist.AppwriteProxy") as mock_proxy_cls:
        mock_proxy = mock_proxy_cls.return_value
        mock_proxy.list_documents.return_value = [
            item("s1", "Stored Shirt", "tops", "grey", formality="smart_casual"),
            item("b1", "Stored Trousers", "bottoms", "beige", formality="smart_casual"),
        ]

        fake_client_wardrobe = [item("fake_b2", "Fake Trousers", "bottoms", "red", formality="smart_casual")]
        req = ItemStyleRequest(user_id="u1", wardrobe=fake_client_wardrobe)

        res = get_item_compatibility("s1", req, x_user_id="u1")
        assert res["success"] is True
        rec_ids = [x["item_id"] for x in res["works_well_with"]]
        assert "b1" in rec_ids
        assert "fake_b2" not in rec_ids


def test_route_security_fake_body_anchor_is_ignored():
    """Route Security 5: Fake body anchor_item is ignored; server DB anchor lookup wins."""
    import pytest
    from fastapi import HTTPException
    with patch("routers.stylist.AppwriteProxy") as mock_proxy_cls:
        mock_proxy = mock_proxy_cls.return_value
        mock_proxy.list_documents.return_value = [
            item("s1", "Stored Shirt", "tops", "grey", formality="smart_casual")
        ]

        fake_anchor = item("unowned_item", "Fake Anchor Shirt", "tops", "blue")
        req = ItemStyleRequest(user_id="u1", anchor_item=fake_anchor)

        # Unowned item_id requested -> HTTP 403 Forbidden even if fake anchor passed in body!
        with pytest.raises(HTTPException) as exc_info:
            get_item_compatibility("unowned_item", req, x_user_id="u1")
        assert exc_info.value.status_code == 403
