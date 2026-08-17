from brain.engines.style_scorer import EXACT_RECOMMENDATION_REPEAT_PENALTY, style_scorer

OUTFIT_A = [
    {"id": "top-a", "name": "Blue Shirt", "category": "Tops"},
    {"id": "bottom-a", "name": "Navy Chinos", "category": "Bottoms"},
]
OUTFIT_B = [
    {"id": "top-b", "name": "White Shirt", "category": "Tops"},
    {"id": "bottom-b", "name": "Grey Trousers", "category": "Bottoms"},
]


def _signature(items):
    return "|".join(sorted({it["id"] for it in items}))


def test_exact_recommendation_repeat_is_penalized():
    context = {
        "occasion": "casual",
        "recent_recommended_signatures": [_signature(OUTFIT_A)],
    }

    result = style_scorer.score_outfit(OUTFIT_A, context, {})

    assert result["breakdown"]["exact_recommendation_repeat_penalty"] == (
        EXACT_RECOMMENDATION_REPEAT_PENALTY
    )
    assert any("exact combination" in r for r in result["reasons"])


def test_exact_recommendation_repeat_ignores_item_order():
    context = {
        "occasion": "casual",
        "recent_recommended_signatures": [_signature(OUTFIT_A)],
    }
    reordered = list(reversed(OUTFIT_A))

    result = style_scorer.score_outfit(reordered, context, {})

    assert result["breakdown"]["exact_recommendation_repeat_penalty"] == (
        EXACT_RECOMMENDATION_REPEAT_PENALTY
    )


def test_different_outfit_receives_no_exact_repeat_penalty():
    context = {
        "occasion": "casual",
        "recent_recommended_signatures": [_signature(OUTFIT_A)],
    }

    result = style_scorer.score_outfit(OUTFIT_B, context, {})

    assert result["breakdown"]["exact_recommendation_repeat_penalty"] == 0.0


def test_recently_worn_penalty_still_works_unchanged():
    context = {"occasion": "casual", "recently_worn_ids": ["top-a", "bottom-a"]}

    result = style_scorer.score_outfit(OUTFIT_A, context, {})

    assert result["breakdown"]["recent_repeat_penalty"] == -3.0  # -1.5 * 2 matched items
    assert result["breakdown"]["exact_recommendation_repeat_penalty"] == 0.0


def test_underworn_boost_still_works_unchanged():
    context = {"occasion": "casual", "underworn_ids": ["top-a", "bottom-a"]}

    result = style_scorer.score_outfit(OUTFIT_A, context, {})

    assert result["breakdown"]["underworn_boost"] == 2.4  # 1.2 * 2 matched items
    assert result["breakdown"]["exact_recommendation_repeat_penalty"] == 0.0


def test_saved_board_affinity_still_works_unchanged():
    context = {"occasion": "casual", "saved_item_ids": ["top-a", "bottom-a"]}

    result = style_scorer.score_outfit(OUTFIT_A, context, {})

    assert result["breakdown"]["saved_board_affinity"] == 2.0  # 1.0 * 2 matched items
    assert result["breakdown"]["exact_recommendation_repeat_penalty"] == 0.0


def test_no_memory_context_is_neutral():
    result = style_scorer.score_outfit(OUTFIT_A, {"occasion": "casual"}, {})

    assert result["breakdown"]["exact_recommendation_repeat_penalty"] == 0.0
    assert result["breakdown"]["recent_repeat_penalty"] == 0.0
    assert result["breakdown"]["underworn_boost"] == 0.0
    assert result["breakdown"]["saved_board_affinity"] == 0.0


def test_scoring_is_deterministic():
    context = {
        "occasion": "casual",
        "recent_recommended_signatures": [_signature(OUTFIT_A)],
        "recently_worn_ids": ["top-a"],
    }

    first = style_scorer.score_outfit(OUTFIT_A, context, {})
    second = style_scorer.score_outfit(OUTFIT_A, context, {})

    assert first["score"] == second["score"]
    assert first["breakdown"] == second["breakdown"]


def _raw_score(items, context):
    # Sum the unclamped breakdown instead of result["score"] (which floors at
    # 0.0) so the memory delta is visible regardless of other components.
    breakdown = style_scorer.score_outfit(items, context, {})["breakdown"]
    return sum(breakdown.values())


def test_exact_repeat_penalty_can_flip_ranking():
    # A starts ahead of B (saved-board affinity boost on A only).
    context = {"occasion": "casual", "saved_item_ids": ["top-a", "bottom-a"]}
    score_a_before = _raw_score(OUTFIT_A, context)
    score_b_before = _raw_score(OUTFIT_B, context)
    assert score_a_before > score_b_before

    # A was also just recommended verbatim -> exact-repeat penalty should
    # outweigh its saved-board affinity boost and flip the ranking.
    context_with_repeat = dict(context, recent_recommended_signatures=[_signature(OUTFIT_A)])
    score_a_after = _raw_score(OUTFIT_A, context_with_repeat)
    score_b_after = _raw_score(OUTFIT_B, context_with_repeat)

    assert score_b_after > score_a_after
