import logging

from routers import chat


def test_style_trace_is_bounded_and_does_not_mutate_or_log_sensitive_payload(caplog):
    request = chat.ModuleChatRequest(
        domain="style",
        message="What colours suit me?",
        request_id="style-trace-1",
        history=[{"role": "user", "content": "Give me style tips"}],
        context_data={
            "conversation_id": "conversation-1",
            "auth_token": "must-not-log",
        },
        style_state={
            "board_id": "board-1",
            "revision": 4,
            "board_items": [{"item_id": "shoe-1", "image_url": "secret"}],
        },
    )
    envelope = {
        "intent": "information",
        "action": "explain_style_concept",
        "response_mode": "text_only",
        "requires_clarification": False,
        "resolved_context": {
            "date_context": "today",
            "activity": None,
            "activity_type": None,
            "occasion": None,
            "referent": None,
        },
        "context_used": ["history.previous_turn"],
    }
    original_context = dict(request.context_data)

    with caplog.at_level(logging.INFO, logger="ahvi.routers.chat"):
        chat._log_style_trace("/api/module-chat", request, envelope)

    trace = next(record.message for record in caplog.records if "AHVI_STYLE_TRACE" in record.message)
    assert "request_id=style-trace-1" in trace
    assert "endpoint=/api/module-chat" in trace
    assert "history_count=1" in trace
    assert "board_id=board-1" in trace
    assert "board_revision=4" in trace
    assert "response_mode=text_only" in trace
    assert "resolved_date=today" in trace
    assert "must-not-log" not in trace
    assert "What_colours_suit_me" not in trace
    assert request.context_data == original_context


def test_non_style_trace_is_not_emitted(caplog):
    request = chat.ModuleChatRequest(
        domain="fitness",
        message="What is my workout?",
        request_id="fitness-trace-1",
    )

    with caplog.at_level(logging.INFO, logger="ahvi.routers.chat"):
        chat._log_style_trace(
            "/api/module-chat",
            request,
            {"response_mode": "text_only", "message_text": "Workout"},
        )

    assert not any("AHVI_STYLE_TRACE" in record.message for record in caplog.records)
