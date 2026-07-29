from __future__ import annotations

from datetime import date
from fastapi.testclient import TestClient
from main import app
from services.diet_service import get_diet_recommendation

client = TestClient(app)

def test_diet_today_unauthenticated():
    """Unauthenticated requests are rejected."""
    response = client.get("/api/diet/today")
    assert response.status_code in {401, 403}

def test_get_diet_recommendation_deterministic():
    """Diet recommendation is deterministic for the same user/date/meal/version."""
    user_id = "user_abc"
    local_date = date(2026, 7, 29)
    meal_type = "lunch"
    profile = {"diet_type": "veg"}

    res1 = get_diet_recommendation(user_id, local_date, local_hour=12, meal_type=meal_type, profile=profile)
    res2 = get_diet_recommendation(user_id, local_date, local_hour=12, meal_type=meal_type, profile=profile)

    assert res1["status"] == "ready"
    assert res2["status"] == "ready"
    assert res1["recommendation"]["id"] == res2["recommendation"]["id"]
    assert res1["recommendation"]["title"] == res2["recommendation"]["title"]

def test_get_diet_recommendation_changes_by_meal_period():
    """Diet recommendation can change by meal period."""
    user_id = "user_abc"
    local_date = date(2026, 7, 29)
    profile = {"diet_type": "veg"}

    res_breakfast = get_diet_recommendation(user_id, local_date, local_hour=8, meal_type="breakfast", profile=profile)
    res_lunch = get_diet_recommendation(user_id, local_date, local_hour=12, meal_type="lunch", profile=profile)

    assert res_breakfast["recommendation"]["meal_type"] == "breakfast"
    assert res_lunch["recommendation"]["meal_type"] == "lunch"
    assert res_breakfast["recommendation"]["title"] != res_lunch["recommendation"]["title"]

def test_get_diet_recommendation_honours_veg_preference():
    """Diet honours supported vegetarian/non-vegetarian preference."""
    user_id = "user_abc"
    local_date = date(2026, 7, 29)

    # Vegetarian profile
    profile_veg = {"diet_type": "vegetarian"}
    res_veg = get_diet_recommendation(user_id, local_date, local_hour=12, meal_type="lunch", profile=profile_veg)
    assert res_veg["status"] == "ready"
    assert "paneer" in res_veg["recommendation"]["title"].lower() or "vegetable" in res_veg["recommendation"]["title"].lower()

    # Non-vegetarian profile (can get meat/fish options)
    profile_nonveg = {"diet_type": "non-vegetarian"}
    res_nonveg = get_diet_recommendation(user_id, local_date, local_hour=12, meal_type="lunch", profile=profile_nonveg)
    assert res_nonveg["status"] == "ready"

def test_get_diet_recommendation_excludes_allergies():
    """Diet excludes known restricted ingredients where supported."""
    user_id = "user_abc"
    local_date = date(2026, 7, 29)

    # Profile with peanut allergy should not receive Traditional Poha because it contains peanuts
    profile_allergic = {"diet_type": "vegetarian", "allergies": ["peanut"]}
    res = get_diet_recommendation(user_id, local_date, local_hour=8, meal_type="breakfast", profile=profile_allergic)

    assert res["status"] == "ready"
    assert "poha" not in res["recommendation"]["title"].lower()  # Excluded due to peanuts!

def test_get_diet_recommendation_unresolved_constraints():
    """Diet returns typed unavailable when constraints cannot safely be resolved."""
    user_id = "user_abc"
    local_date = date(2026, 7, 29)

    # Overly restrictive allergy list which blocks everything
    profile_restrictive = {
        "diet_type": "vegetarian",
        "allergies": ["oats", "rice", "chickpeas", "paneer"]
    }
    res = get_diet_recommendation(user_id, local_date, local_hour=12, meal_type="lunch", profile=profile_restrictive)

    assert res["status"] == "unavailable"
    assert res["reason"] == "diet_constraints_unresolved"
    assert res["recommendation"] is None

def test_get_diet_recommendation_non_persisted():
    """Diet does not claim persistence (persisted=False)."""
    user_id = "user_abc"
    local_date = date(2026, 7, 29)
    profile = {"diet_type": "veg"}

    res = get_diet_recommendation(user_id, local_date, local_hour=20, meal_type="dinner", profile=profile)
    assert res["recommendation"]["persisted"] is False

def test_unknown_diet_preference_unresolved():
    """Unknown/ambiguous diet preference does not default to non-vegetarian."""
    user_id = "user_abc"
    local_date = date(2026, 7, 29)

    # Empty or ambiguous diet preference should return unavailable
    res_empty = get_diet_recommendation(user_id, local_date, local_hour=12, meal_type="lunch", profile={})
    assert res_empty["status"] == "unavailable"
    assert res_empty["reason"] == "diet_constraints_unresolved"

    res_ambiguous = get_diet_recommendation(user_id, local_date, local_hour=12, meal_type="lunch", profile={"diet_type": "something_else"})
    assert res_ambiguous["status"] == "unavailable"
