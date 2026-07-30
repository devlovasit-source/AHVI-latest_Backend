from services.professional_safety import evaluate_professional_safety


def test_professional_safe_item_is_accepted():
    result = evaluate_professional_safety(
        {"professional_safe": True, "professionalism_score": 0.8},
        "client_meeting",
    )

    assert result["allowed"] is True


def test_unsafe_professional_item_is_rejected():
    result = evaluate_professional_safety(
        {"professional_safe": False, "professionalism_score": 1.0},
        "office",
    )

    assert result["allowed"] is False
    assert result["reason_code"] == "professional_safe_false"


def test_casual_context_is_unaffected():
    result = evaluate_professional_safety(
        {"professional_safe": False, "safety_tags": ["not_professional"]},
        "date_night",
    )

    assert result["allowed"] is True
    assert result["reason_code"] == "not_professional_context"
