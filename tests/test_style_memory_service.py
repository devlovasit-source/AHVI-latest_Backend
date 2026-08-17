from services.style_memory_service import (
    _outfit_signature,
    build_style_memory_context,
    load_recommendation_memory,
)


def test_outfit_signature_is_order_independent():
    a = _outfit_signature([{"id": "shirt1"}, {"id": "pant1"}, {"id": "shoe1"}])
    b = _outfit_signature([{"id": "shoe1"}, {"id": "shirt1"}, {"id": "pant1"}])
    assert a == b
    assert a != ""


def test_outfit_signature_ignores_duplicate_ids():
    a = _outfit_signature([{"id": "shirt1"}, {"id": "pant1"}])
    b = _outfit_signature([{"id": "shirt1"}, {"id": "shirt1"}, {"id": "pant1"}])
    assert a == b


def test_outfit_signature_handles_empty_or_malformed_input():
    assert _outfit_signature([]) == ""
    assert _outfit_signature(None) == ""
    assert _outfit_signature([{"name": ""}, {}]) == ""


def test_load_recommendation_memory_builds_deduped_signatures(monkeypatch):
    def fake_load_user_memory(user_id):
        return {
            "recent_outfits": [
                {"items": [{"id": "shirt1"}, {"id": "pant1"}]},
                {"items": [{"id": "pant1"}, {"id": "shirt1"}]},  # same set, different order
                {"items": []},  # no ids -> ignored
                {"items": None},  # malformed -> ignored
                "not-a-dict",  # malformed -> ignored
            ]
        }

    monkeypatch.setattr("brain.outfit_pipeline._load_user_memory", fake_load_user_memory)

    result = load_recommendation_memory("user-1")

    assert result["recent_recommended_signatures"] == ["pant1|shirt1"]


def test_load_recommendation_memory_empty_when_no_history(monkeypatch):
    monkeypatch.setattr(
        "brain.outfit_pipeline._load_user_memory", lambda user_id: {"recent_outfits": []}
    )

    assert load_recommendation_memory("user-1") == {"recent_recommended_signatures": []}


def test_load_recommendation_memory_empty_for_blank_user_id():
    assert load_recommendation_memory("") == {"recent_recommended_signatures": []}


def test_load_recommendation_memory_degrades_neutral_on_error(monkeypatch):
    def boom(user_id):
        raise RuntimeError("appwrite unavailable")

    monkeypatch.setattr("brain.outfit_pipeline._load_user_memory", boom)

    assert load_recommendation_memory("user-1") == {"recent_recommended_signatures": []}


def test_build_style_memory_context_neutral_without_user_id():
    ctx = build_style_memory_context("", wardrobe=[])
    assert ctx["recent_recommended_signatures"] == []


def test_build_style_memory_context_preserves_existing_shape_when_no_recommendations(monkeypatch):
    import services.style_memory_service as svc

    monkeypatch.setattr(
        svc,
        "load_wear_memory",
        lambda user_id, wardrobe=None: {
            "recently_worn_ids": [],
            "underworn_ids": [],
            "wear_counts": {},
            "last_worn_at": {},
        },
    )
    monkeypatch.setattr(
        svc,
        "load_saved_board_memory",
        lambda user_id: {
            "saved_item_ids": [],
            "saved_board_patterns": [],
            "favorite_colors": [],
            "favorite_categories": [],
        },
    )
    monkeypatch.setattr(
        svc, "load_recommendation_memory", lambda user_id: {"recent_recommended_signatures": []}
    )

    ctx = svc.build_style_memory_context("user-1", wardrobe=[])

    assert ctx["recent_recommended_signatures"] == []
    assert ctx["recently_worn_ids"] == []
    assert ctx["saved_item_ids"] == []
