import pytest
from services.workout_card_service import (
    _display_title,
    build_workout_card,
    get_workout_recommendations,
    get_today_workout_card,
)


def test_workout_router_and_home_summary_imports():
    import routers.workouts as workouts_router
    import services.home_summary_service as home_summary_svc

    assert hasattr(workouts_router, "router")
    assert hasattr(home_summary_svc, "generate_home_summary")
    assert callable(get_today_workout_card)
    assert callable(get_workout_recommendations)


def test_title_normalization_cases():
    assert _display_title({"title": "Women — 10 Min Mobility"}) == "10-Min Mobility"
    assert _display_title({"title": "Men - 20 Min Strength"}) == "20-Min Strength"
    assert _display_title({"title": "Universal — 12 Min Mobility"}) == "12-Min Mobility"
    assert _display_title({"title": "30 Min Cardio"}) == "30-Min Cardio"
    assert _display_title({"name": "15 Min HIIT"}) == "15-Min HIIT"
    assert _display_title({}) == "Today's Workout"
    assert _display_title({"title": None, "name": None}) == "Today's Workout"


def test_home_summary_generate_move_card():
    from services.home_summary_service import generate_home_summary

    summary = generate_home_summary("test_user_49")
    assert isinstance(summary, dict)
    assert "move_card" in summary
    assert "cards" in summary
    assert "move" in summary["cards"]
