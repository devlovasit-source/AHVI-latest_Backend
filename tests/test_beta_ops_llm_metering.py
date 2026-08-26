from types import SimpleNamespace

import pytest

from services import ai_gateway, gemini_multi_garment_detector as multi, llm_service
from services.request_context import (
    get_authenticated_user_id,
    reset_authenticated_user_id,
    set_authenticated_user_id,
)


class FakeUsage:
    prompt_token_count = 11
    candidates_token_count = 7
    cached_content_token_count = 3


class FakeGeminiResponse:
    text = "generated"
    usage_metadata = FakeUsage()


def test_llm_gemini_provider_boundary_emits_one_attempt(monkeypatch):
    events = []

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeGeminiResponse()

    monkeypatch.setattr(llm_service, "_get_gemini_client", lambda timeout=None: SimpleNamespace(models=FakeModels()))
    monkeypatch.setattr(llm_service, "_thinking_config_disabled", lambda: None)
    monkeypatch.setattr(llm_service.types, "GenerateContentConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(llm_service.tone_engine, "build_prompt_tone", lambda *_: {})
    monkeypatch.setattr(llm_service.tone_engine, "apply", lambda text, **_: text)
    monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

    result = llm_service._call_gemini_text(
        "safe prompt", user_id="u1", request_id="r1", operation_id="op1", usecase="chat"
    )

    assert result == "generated"
    assert len(events) == 1
    assert events[0]["attempt"] == 1
    assert events[0]["operation_id"] == "op1"
    assert events[0]["input_tokens"] == 11
    assert events[0]["output_tokens"] == 7
    assert events[0]["cached_tokens"] == 3


def test_generate_text_gemini_truncation_retry_uses_same_operation(monkeypatch):
    events = []
    calls = []
    token = set_authenticated_user_id("u1")
    try:
        def fake_gemini(prompt, **kwargs):
            calls.append(kwargs)
            llm_service._meter_attempt(
                user_id="u1", request_id=kwargs["request_id"],
                operation_id=kwargs["operation_id"], attempt=kwargs["meter_state"]["attempt"] + 1,
                provider="gemini", model="gemini-test", usecase=kwargs["usecase"], started=0,
                status="success", response=FakeGeminiResponse(),
            )
            kwargs["meter_state"]["attempt"] += 1
            return "first" if len(calls) == 1 else "second"

        monkeypatch.setattr(llm_service, "_gemini_enabled", lambda: True)
        monkeypatch.setattr(llm_service, "_call_gemini_text", fake_gemini)
        monkeypatch.setattr(llm_service, "_guard_truncation", lambda text, **_: "trimmed" if text == "first" else text)
        monkeypatch.setattr(llm_service, "looks_truncated_safe", lambda text: True)
        monkeypatch.setattr(llm_service, "RETRY_ON_TRUNCATION", True)
        monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

        assert llm_service.generate_text("safe", usecase="chat", request_id="r1") == "second"
    finally:
        reset_authenticated_user_id(token)

    assert len(events) == 2
    assert [event["attempt"] for event in events] == [1, 2]
    assert len({event["operation_id"] for event in events}) == 1
    assert all("prompt" not in event for event in events)


def test_ollama_fallback_models_emit_one_attempt_each(monkeypatch):
    events = []

    class Response:
        status_code = 200

        def json(self):
            return {"response": "ok", "prompt_eval_count": 5, "eval_count": 4}

    class Session:
        def __init__(self):
            self.models = []

        def post(self, url, json, timeout):
            self.models.append(json["model"])
            return Response()

    session = Session()
    monkeypatch.setattr(llm_service, "session", session)
    monkeypatch.setattr(llm_service, "MODEL_FALLBACKS", ["model-b"])
    monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

    result = llm_service._call_ollama(
        {"model": "model-a"}, user_id="u1", request_id="r1", operation_id="op1", usecase="chat"
    )

    assert result["response"] == "ok"
    assert len(events) == 1
    assert events[0]["model"] == "model-a"
    assert events[0]["input_tokens"] == 5


def test_ollama_failed_model_then_fallback_are_separate(monkeypatch):
    events = []

    class Response:
        def __init__(self, status): self.status_code = status
        def json(self): return {"response": "ok"}

    class Session:
        def post(self, url, json, timeout):
            return Response(500 if json["model"] == "model-a" else 200)

    monkeypatch.setattr(llm_service, "session", Session())
    monkeypatch.setattr(llm_service, "MODEL_FALLBACKS", ["model-b"])
    monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

    llm_service._call_ollama(
        {"model": "model-a"}, user_id="u1", request_id="r1", operation_id="op1", usecase="chat"
    )

    assert [(event["model"], event["attempt"], event["status"]) for event in events] == [
        ("model-a", 1, "failed"), ("model-b", 2, "success")
    ]


def test_ollama_http_200_body_parse_failure_is_one_success_attempt(monkeypatch):
    events = []

    class Response:
        status_code = 200
        def json(self): raise ValueError("malformed body")

    class Session:
        def post(self, *args, **kwargs): return Response()

    monkeypatch.setattr(llm_service, "session", Session())
    monkeypatch.setattr(llm_service, "MODEL_FALLBACKS", [])
    monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

    assert llm_service._call_ollama(
        {"model": "model-a"}, user_id="u1", operation_id="op1"
    ) is None
    assert len(events) == 1
    assert events[0]["status"] == "success"


def test_ollama_vision_invalid_generated_json_is_one_success_attempt(monkeypatch):
    events = []

    class Response:
        status_code = 200
        text = ""
        def json(self): return {"response": "not-json"}

    monkeypatch.setattr(ai_gateway, "_vision_model_candidates", lambda: ["vision-a"])
    monkeypatch.setattr(ai_gateway.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(ai_gateway, "parse_json_object", lambda value: (_ for _ in ()).throw(ValueError("bad json")))
    monkeypatch.setattr(ai_gateway, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

    with pytest.raises(RuntimeError):
        ai_gateway.ollama_vision_json(
            prompt="safe", image_base64="opaque", user_id="u1", request_id="r1"
        )
    assert len(events) == 1
    assert events[0]["status"] == "success"


def test_generate_text_gemini_failure_then_ollama_fallback_shares_operation(monkeypatch):
    events = []
    token = set_authenticated_user_id("u1")
    try:
        def failed_gemini(prompt, **kwargs):
            llm_service._meter_attempt(
                user_id="u1", request_id=kwargs["request_id"], operation_id=kwargs["operation_id"],
                attempt=kwargs["meter_state"]["attempt"] + 1, provider="gemini", model="gemini-test",
                usecase=kwargs["usecase"], started=0, status="failed", error=RuntimeError("provider"),
            )
            kwargs["meter_state"]["attempt"] += 1
            return None

        def successful_ollama(payload, **kwargs):
            llm_service._meter_attempt(
                user_id="u1", request_id=kwargs["request_id"], operation_id=kwargs["operation_id"],
                attempt=kwargs["meter_state"]["attempt"] + 1, provider="ollama", model=payload["model"],
                usecase=kwargs["usecase"], started=0, status="success", response={"response": "ok"},
            )
            kwargs["meter_state"]["attempt"] += 1
            return {"response": "ok"}

        monkeypatch.setattr(llm_service, "_gemini_enabled", lambda: True)
        monkeypatch.setattr(llm_service, "_call_gemini_text", failed_gemini)
        monkeypatch.setattr(llm_service, "_call_ollama", successful_ollama)
        monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: events.append(kwargs))
        monkeypatch.setattr(llm_service.tone_engine, "build_prompt_tone", lambda *_: {})
        monkeypatch.setattr(llm_service.tone_engine, "apply", lambda text, **_: text)

        assert llm_service.generate_text("safe", usecase="chat", request_id="r1") == "ok"
    finally:
        reset_authenticated_user_id(token)

    assert len(events) == 2
    assert [(event["provider"], event["status"], event["attempt"]) for event in events] == [
        ("gemini", "failed", 1), ("ollama", "success", 2)
    ]
    assert len({event["operation_id"] for event in events}) == 1


def test_missing_authenticated_user_is_not_fabricated(monkeypatch):
    events = []
    token = set_authenticated_user_id("")
    try:
        class Response:
            status_code = 503
            text = ""
        class Session:
            def post(self, *args, **kwargs): return Response()
        monkeypatch.setattr(llm_service, "session", Session())
        monkeypatch.setattr(llm_service, "MODEL_FALLBACKS", [])
        monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: events.append(kwargs))
        assert llm_service._call_ollama(
            {"model": "model-a"}, operation_id="op1", request_id="r1", user_id=None
        ) is None
    finally:
        reset_authenticated_user_id(token)
    assert get_authenticated_user_id() == ""
    assert len(events) == 1
    assert events[0]["user_id"] is None


def test_untrusted_signal_user_id_cannot_override_authenticated_context(monkeypatch):
    events = []
    token = set_authenticated_user_id("auth-user")
    try:
        def fake_gemini(prompt, **kwargs):
            llm_service._meter_attempt(
                user_id=kwargs.get("user_id"), request_id=kwargs["request_id"],
                operation_id=kwargs["operation_id"], attempt=1, provider="gemini",
                model="gemini-test", usecase=kwargs["usecase"], started=0,
                status="success", response=FakeGeminiResponse(),
            )
            return "ok"
        monkeypatch.setattr(llm_service, "_gemini_enabled", lambda: True)
        monkeypatch.setattr(llm_service, "_call_gemini_text", fake_gemini)
        monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: events.append(kwargs))
        llm_service.generate_text("safe", signals={"user_id": "attacker"}, usecase="chat")
    finally:
        reset_authenticated_user_id(token)
    assert events[0]["user_id"] == "auth-user"


def test_gateway_does_not_forward_signals_user_id_as_authority(monkeypatch):
    captured = {}
    token = set_authenticated_user_id("auth-user")
    try:
        def fake_generate_text(**kwargs):
            captured.update(kwargs)
            return "ok"
        monkeypatch.setattr(ai_gateway.llm_service, "generate_text", fake_generate_text)
        assert ai_gateway.generate_text(
            "safe", signals={"user_id": "attacker"}, request_id="r1"
        ) == "ok"
    finally:
        reset_authenticated_user_id(token)
    assert "user_id" not in captured


@pytest.mark.parametrize("status,expected", [(400, "http_4xx"), (500, "http_5xx")])
def test_ollama_http_failure_classification(monkeypatch, status, expected):
    events = []
    class Response:
        status_code = status
        text = "private response body"
    class Session:
        def post(self, *args, **kwargs): return Response()
    monkeypatch.setattr(llm_service, "session", Session())
    monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: events.append(kwargs))
    llm_service._call_ollama({"model": "model-a"}, user_id="u1", operation_id="op1")
    assert events[0]["error_code"] == expected


def test_request_context_user_id_resets_after_scope():
    assert get_authenticated_user_id() == ""
    token = set_authenticated_user_id("scoped-user")
    try:
        assert get_authenticated_user_id() == "scoped-user"
    finally:
        reset_authenticated_user_id(token)
    assert get_authenticated_user_id() == ""


def test_gateway_circuit_breaker_skip_emits_zero_attempts(monkeypatch):
    events = []
    monkeypatch.setattr(ai_gateway, "_breaker_allows", lambda key: False)
    monkeypatch.setattr(ai_gateway, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

    with pytest.raises(RuntimeError):
        ai_gateway.ollama_vision_json(
            prompt="safe", image_base64="not persisted", user_id="u1", request_id="r1"
        )

    assert events == []


def test_gateway_pre_request_url_failure_emits_zero_attempts(monkeypatch):
    events = []
    monkeypatch.setattr(ai_gateway, "_ollama_generate_url", lambda: (_ for _ in ()).throw(ValueError("bad url")))
    monkeypatch.setattr(ai_gateway, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

    with pytest.raises(RuntimeError):
        ai_gateway.ollama_vision_json(
            prompt="safe", image_base64="opaque", user_id="u1", request_id="r1"
        )
    assert events == []


def test_gateway_text_wrapper_does_not_meter_itself(monkeypatch):
    events = []
    monkeypatch.setattr(ai_gateway, "record_llm_attempt", lambda **kwargs: events.append(kwargs))
    monkeypatch.setattr(ai_gateway.llm_service, "generate_text", lambda **kwargs: "ok")

    assert ai_gateway.generate_text("safe", request_id="r1") == "ok"
    assert events == []


def test_gateway_vision_candidates_share_operation_and_count_attempts(monkeypatch):
    events = []
    monkeypatch.setattr(ai_gateway, "_vision_model_candidates", lambda: ["vision-a", "vision-b"])
    monkeypatch.setattr(ai_gateway, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

    class Response:
        status_code = 200
        text = ""
        def json(self): return {"response": '{"ok": true}', "prompt_eval_count": 2, "eval_count": 3}

    monkeypatch.setattr(ai_gateway.requests, "post", lambda *args, **kwargs: Response())
    result = ai_gateway.ollama_vision_json(
        prompt="safe", image_base64="ignored", user_id="u1", request_id="r1"
    )

    assert result[1] == "vision-a"
    assert len(events) == 1
    assert events[0]["attempt"] == 1


def test_multi_garment_actual_calls_share_operation_and_increment_attempt(monkeypatch):
    events = []
    monkeypatch.setattr(multi, "record_llm_attempt", lambda **kwargs: events.append(kwargs))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: FakeGeminiResponse()))
    state = {"attempt": 0}

    multi._metered_gemini_request(
        client, model="gemini-test", contents=["opaque"], config={}, user_id="u1",
        request_id="r1", operation_id="op1", meter_state=state
    )
    multi._metered_gemini_request(
        client, model="gemini-test", contents=["opaque"], config={}, user_id="u1",
        request_id="r1", operation_id="op1", meter_state=state
    )

    assert [(event["attempt"], event["operation_id"]) for event in events] == [(1, "op1"), (2, "op1")]


def test_multi_garment_config_failure_emits_no_attempt(monkeypatch):
    events = []
    monkeypatch.setattr(multi, "record_llm_attempt", lambda **kwargs: events.append(kwargs))
    monkeypatch.setattr(multi, "_get_gemini_client", lambda: SimpleNamespace())
    monkeypatch.setattr(multi, "types", SimpleNamespace(
        Part=SimpleNamespace(from_bytes=lambda **kwargs: object()),
        ThinkingConfig=lambda **kwargs: None,
        GenerateContentConfig=lambda **kwargs: (_ for _ in ()).throw(ValueError("bad config")),
    ))

    assert multi._call_gemini_vision(b"bytes", user_id="u1", request_id="r1") is None
    assert events == []


def test_telemetry_failure_does_not_change_ollama_result(monkeypatch):
    class Response:
        status_code = 200
        def json(self): return {"response": "ok"}

    class Session:
        def post(self, *args, **kwargs): return Response()

    monkeypatch.setattr(llm_service, "session", Session())
    monkeypatch.setattr(llm_service, "record_llm_attempt", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down")))
    result = llm_service._call_ollama(
        {"model": "model-a"}, user_id="u1", request_id="r1", operation_id="op1"
    )
    assert result == {"response": "ok"}


def test_metering_never_receives_prompt_or_image_payload(monkeypatch):
    events = []
    monkeypatch.setattr(ai_gateway, "_vision_model_candidates", lambda: ["vision-a"])
    monkeypatch.setattr(ai_gateway, "record_llm_attempt", lambda **kwargs: events.append(kwargs))

    class Response:
        status_code = 200
        text = ""
        def json(self): return {"response": '{"ok": true}'}

    monkeypatch.setattr(ai_gateway.requests, "post", lambda *args, **kwargs: Response())
    ai_gateway.ollama_vision_json(
        prompt="PRIVATE PROMPT", image_base64="PRIVATE IMAGE", user_id="u1"
    )
    assert all("PRIVATE" not in str(event) for event in events)
