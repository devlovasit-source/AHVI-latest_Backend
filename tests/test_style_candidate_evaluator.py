from __future__ import annotations

import pytest

from routers import stylist
from services import style_candidate_evaluator as evaluator
from services.style_execution_policy import (
    activate_style_execution,
    create_style_execution_session,
)


def _enable(monkeypatch):
    monkeypatch.setenv("STYLE_EVALUATION_RUNNER_ENABLED", "true")
    monkeypatch.setenv("STYLE_EVALUATION_TEST_USER_ID", "style-evaluation-test")
    monkeypatch.setenv("STYLE_EVALUATION_AUTHORIZED_USER_ID", "style-evaluation-test")


def test_runner_is_not_an_http_route():
    assert all(route.endpoint is not evaluator.run_internal_candidate_evaluation for route in stylist.router.routes)


def test_identity_mismatch_stops_before_repository_read(monkeypatch):
    monkeypatch.setenv("STYLE_EVALUATION_RUNNER_ENABLED", "true")
    monkeypatch.setenv("STYLE_EVALUATION_TEST_USER_ID", "test-a")
    monkeypatch.setenv("STYLE_EVALUATION_AUTHORIZED_USER_ID", "test-b")
    monkeypatch.setattr(evaluator, "_load_style_assets", lambda: pytest.fail("must not read"))
    with pytest.raises(evaluator.CandidateEvaluationConfigurationError):
        evaluator.run_internal_candidate_evaluation()


def test_fixed_runner_uses_server_policy_and_returns_sanitized_ids(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(evaluator, "_load_style_assets", lambda: [{"asset_id": "a", "source": "style_asset"}])
    calls = []

    def fake_style(item_id, request, http_request):
        calls.append((item_id, request.user_id, request.mode, request.context))
        return {
            "success": True,
            "source": "style_asset",
            "anchor_item": {"asset_id": item_id},
            "style_directions": [{"items": [{"item_id": item_id}]}],
        }

    monkeypatch.setattr(stylist, "style_wardrobe_item", fake_style)
    result = evaluator.run_internal_candidate_evaluation()
    assert result["scenario_count"] == 5
    assert result["model_calls"] == 0
    assert result["blocked_write_attempts"] == 0
    assert all(user == "style-evaluation-test" and mode == "style_this" for _, user, mode, _ in calls)
    assert all("image_url" not in row for row in result["results"])


def test_denied_write_fails_the_evaluation(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(evaluator, "_load_style_assets", lambda: [])

    def fake_style(*_args):
        from services.style_execution_policy import run_board_registration
        run_board_registration(lambda **_kwargs: {"ok": True})
        return {"success": True, "style_directions": []}

    monkeypatch.setattr(stylist, "style_wardrobe_item", fake_style)
    with pytest.raises(evaluator.CandidateEvaluationWriteError):
        evaluator.run_internal_candidate_evaluation()


def test_connected_builder_skips_board_store_under_evaluation(monkeypatch):
    called = []
    monkeypatch.setattr(stylist, "register_board", lambda **kwargs: called.append(kwargs) or {"ok": True})
    anchor = {"item_id": "top-1", "id": "top-1", "name": "White Shirt", "category": "Tops", "source": "style_asset"}
    shoes = {"item_id": "shoe-1", "id": "shoe-1", "name": "Loafers", "category": "Footwear", "source": "style_asset"}
    with activate_style_execution(create_style_execution_session("read_only_evaluation")):
        outfit, _ = stylist._builder_outfit(anchor, [], [anchor, shoes], "daily", mode="style_this")
    assert called == []
    assert outfit["shuffle_available"] is False
    assert outfit["shuffle_state_error"]["code"] == "BOARD_REGISTRATION_NOT_ALLOWED"
