import asyncio
from types import SimpleNamespace

import pytest

from routers import feedback, home, upload_batches
from services import style_flow_service


def _request(request_id="req-1"):
    return SimpleNamespace(state=SimpleNamespace(request_id=request_id))


def test_home_event_emitted_after_success(monkeypatch):
    events = []
    monkeypatch.setattr(home, "get_user_profile", lambda user_id: {})
    monkeypatch.setattr(home, "generate_home_summary", lambda **kwargs: _async_value({"ok": True}))
    monkeypatch.setattr(home, "record_event", lambda **kwargs: events.append(kwargs))

    result = asyncio.run(
        home.get_today_summary(_request("home-1"), user={"user_id": "auth-user"})
    )

    assert result == {"ok": True}
    assert events == [{
        "event_type": "home.summary_requested",
        "user_id": "auth-user",
        "request_id": "home-1",
    }]


async def _async_value(value):
    return value


def test_upload_start_event_uses_authenticated_user_and_batch_id(monkeypatch):
    events = []
    monkeypatch.setattr(upload_batches, "_effective_user_id", lambda request, supplied: "auth-user")
    monkeypatch.setattr(
        upload_batches.upload_batch_orchestrator,
        "create_or_resume_batch",
        lambda **kwargs: {"success": True, "batch_id": "batch-1", "resumed": False},
    )
    monkeypatch.setattr(upload_batches, "record_event", lambda **kwargs: events.append(kwargs))

    result = upload_batches.create_batch(_request("upload-1"), upload_batches.CreateBatchRequest(
        user_id="attacker-supplied", client_batch_request_id="client-1", total_items=1
    ))

    assert result == {"success": True, "batch_id": "batch-1", "resumed": False}
    assert events[0]["user_id"] == "auth-user"
    assert events[0]["request_id"] == "upload-1"
    assert events[0]["metadata"] == {"batch_id": "batch-1"}
    assert events[0]["idempotency_key"] == "auth-user|wardrobe.upload_started|batch-1"


@pytest.mark.parametrize("action", ["shown", "saved", "dismissed", "regenerated", "clicked", "shared"])
def test_passive_board_actions_persist_without_board_payload(monkeypatch, action):
    events = []
    monkeypatch.setattr(feedback, "enforce_owner", lambda request, supplied: "auth-user")
    monkeypatch.setattr(feedback, "record_event", lambda **kwargs: events.append(kwargs))
    payload = {"board_id": "board-1", "items": [{"secret": "must not be persisted"}]}

    result = feedback.feedback_board(
        feedback.BoardFeedbackRequest(user_id="body-user", action=action, board_payload=payload),
        _request("feedback-1"),
    )

    assert result["success"] is True
    assert len(events) == 1
    assert events[0]["event_type"] == "product.event"
    assert events[0]["status"] == action
    assert events[0]["metadata"] == {"board_id": "board-1"}
    assert "board_payload" not in events[0]
    assert "items" not in str(events[0])


def test_like_dislike_qdrant_behavior_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(feedback, "enforce_owner", lambda request, supplied: "auth-user")
    monkeypatch.setattr(feedback, "record_event", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(feedback.qdrant_service, "upsert_user_memory", lambda **kwargs: True)

    result = feedback.feedback_board(
        feedback.BoardFeedbackRequest(user_id="auth-user", action="like", board_payload={"board_id": "b"}),
        _request("feedback-like"),
    )

    assert result["success"] is True
    assert calls == []


def test_repeated_passive_actions_have_no_implicit_idempotency_key(monkeypatch):
    events = []
    monkeypatch.setattr(feedback, "enforce_owner", lambda request, supplied: "auth-user")
    monkeypatch.setattr(feedback, "record_event", lambda **kwargs: events.append(kwargs))
    request = feedback.BoardFeedbackRequest(user_id="auth-user", action="shown", board_payload={"board_id": "b"})

    feedback.feedback_board(request, _request("shown-1"))
    feedback.feedback_board(request, _request("shown-2"))

    assert len(events) == 2
    assert all("idempotency_key" not in event for event in events)


def test_feedback_telemetry_failure_does_not_change_response(monkeypatch):
    monkeypatch.setattr(feedback, "enforce_owner", lambda request, supplied: "auth-user")
    monkeypatch.setattr(
        feedback,
        "record_event",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
    )
    result = feedback.feedback_board(
        feedback.BoardFeedbackRequest(user_id="auth-user", action="shown", board_payload={}),
        _request("feedback-failure"),
    )
    assert result == {
        "success": True,
        "message": "Board behavior logged",
        "action": "shown",
    }


def test_style_request_emits_one_summary_with_existing_rejection_reasons(monkeypatch):
    events = []

    def fake_flow(**kwargs):
        style_flow_service._note_style_rejections(
            [({}, "missing_required_slots_after_sanitize"), ({}, "occasion_mismatch")]
        )
        return {"success": True, "cards": [{"id": "board-1"}, {"id": "board-2"}], "meta": {}}

    monkeypatch.setattr(style_flow_service, "_build_style_flow_response", fake_flow)
    monkeypatch.setattr(style_flow_service, "record_beta_ops_event", lambda **kwargs: events.append(kwargs))

    response = style_flow_service.build_style_flow_response(
        user_id="auth-user",
        query="dinner",
        wardrobe=[],
        context={"request_id": "style-1", "source": "style", "flow": "dailywear"},
        requested_board_count=4,
    )

    assert len(events) == 1
    assert events[0]["event_type"] == "style.request_outcome"
    assert events[0]["user_id"] == "auth-user"
    assert events[0]["request_id"] == "style-1"
    assert events[0]["metadata"]["requested_count"] == 4
    assert events[0]["metadata"]["generated_count"] == 2
    assert events[0]["metadata"]["rejected_count"] == 2
    assert events[0]["metadata"]["reason_counts"] == {
        "missing_required_slots_after_sanitize": 1,
        "occasion_mismatch": 1,
    }
    assert response["cards"] == [{"id": "board-1"}, {"id": "board-2"}]


def test_style_telemetry_failure_does_not_change_response_or_exception(monkeypatch):
    response = {"success": True, "cards": []}
    monkeypatch.setattr(style_flow_service, "_build_style_flow_response", lambda **kwargs: response)
    monkeypatch.setattr(
        style_flow_service,
        "record_beta_ops_event",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
    )
    assert style_flow_service.build_style_flow_response(user_id="u1", query="x", wardrobe=[]) is response

    error = RuntimeError("original")
    def failing_flow(**kwargs):
        raise error
    monkeypatch.setattr(style_flow_service, "_build_style_flow_response", failing_flow)
    with pytest.raises(RuntimeError) as caught:
        style_flow_service.build_style_flow_response(user_id="u1", query="x", wardrobe=[])
    assert caught.value is error
