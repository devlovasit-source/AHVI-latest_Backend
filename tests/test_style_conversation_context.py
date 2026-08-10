from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat
from services import semantic_intent_resolver as resolver
from services.style_conversation_context import (
    resolve_style_conversation_context,
)
from services.style_reasoning_engine import validate_activity_compatibility


def _semantic_clarification():
    return {
        "domain": "style",
        "intent": "clarification",
        "action": "request_clarification",
        "response_mode": "clarification",
        "confidence": 0.95,
        "requires_clarification": True,
        "resolved_context": {},
        "constraints": {"required": [], "avoid": []},
        "referent": None,
        "reason_codes": ["context_required"],
        "missing_information": ["occasion_or_activity"],
    }


def test_three_turn_context_preserves_tomorrow_and_badminton():
    history = [
        {"role": "user", "content": "I need something for tomorrow"},
        {"role": "user", "content": "I have a badminton game"},
    ]

    context, diagnostics = resolve_style_conversation_context(
        current_message="show visual inspiration for this",
        recent_history=history,
    )

    assert context.date_context == "tomorrow"
    assert context.activity == "badminton"
    assert context.activity_type == "court_sport"
    assert context.referent["text"] == "this"
    assert context.referent["type"] == "activity"
    assert context.referent["label"] == "badminton"
    assert context.referent["temporal"] == {"relative_date": "tomorrow"}
    assert context.referent["resolved_to"] == "badminton tomorrow"
    assert diagnostics["requires_clarification"] is False
    assert context.occasion is None


def test_current_turn_overrides_carried_date_without_losing_activity():
    context, _ = resolve_style_conversation_context(
        current_message="Actually tonight",
        recent_history=[{"role": "user", "content": "I have badminton tomorrow"}],
        carried_context={"date_context": "tomorrow", "activity": "badminton"},
    )

    assert context.date_context == "tonight"
    assert context.activity == "badminton"
    assert context.activity_type == "court_sport"


def test_required_context_matrix_retains_occasion_activity_and_constraints():
    client_context, _ = resolve_style_conversation_context(
        current_message="show me inspiration for this",
        recent_history=[{"role": "user", "content": "I have a client meeting tomorrow"}],
    )
    assert client_context.date_context == "tomorrow"
    assert client_context.occasion == "client_meeting"

    gym_context, _ = resolve_style_conversation_context(
        current_message="show inspiration",
        recent_history=[{"role": "user", "content": "Gym this evening"}],
    )
    assert gym_context.activity == "gym"
    assert gym_context.activity_type == "training"
    assert gym_context.date_context == "this_evening"

    wedding_context, _ = resolve_style_conversation_context(
        current_message="something more casual",
        recent_history=[{"role": "user", "content": "Wedding Saturday"}],
    )
    assert wedding_context.occasion == "wedding"
    assert wedding_context.date_context == "saturday"
    assert wedding_context.style_constraints == ["more casual"]

    override_context, _ = resolve_style_conversation_context(
        current_message="Actually tonight",
        recent_history=[{"role": "user", "content": "I have badminton tomorrow"}],
    )
    assert override_context.date_context == "tonight"
    assert override_context.activity == "badminton"


def test_client_meeting_followup_preserves_tomorrow():
    context, _ = resolve_style_conversation_context(
        current_message="show visual inspiration for this",
        recent_history=[{"role": "user", "content": "I have a client meeting tomorrow"}],
    )

    assert context.date_context == "tomorrow"
    assert context.occasion == "client_meeting"
    assert context.activity is None


def test_dinner_tonight_followup_preserves_time_and_casual_constraint():
    context, _ = resolve_style_conversation_context(
        current_message="show something more casual",
        recent_history=[{"role": "user", "content": "I'm going for dinner tonight"}],
    )

    assert context.date_context == "tonight"
    assert context.occasion == "dinner"
    assert context.style_constraints == ["more casual"]


def test_badminton_no_black_followup_preserves_activity_and_constraint():
    context, _ = resolve_style_conversation_context(
        current_message="show visual inspiration",
        recent_history=[
            {"role": "user", "content": "I have a badminton game tomorrow"},
            {"role": "user", "content": "no black"},
        ],
    )

    assert context.date_context == "tomorrow"
    assert context.activity == "badminton"
    assert context.activity_type == "court_sport"
    assert context.negative_constraints == ["black"]


def test_explicit_occasion_correction_replaces_previous_activity():
    context, _ = resolve_style_conversation_context(
        current_message="actually make that a client meeting",
        recent_history=[{"role": "user", "content": "I have a badminton game tomorrow"}],
    )

    assert context.date_context == "tomorrow"
    assert context.occasion == "client_meeting"
    assert context.activity is None
    assert context.activity_type is None


def test_missing_activity_or_occasion_requires_clarification():
    context, diagnostics = resolve_style_conversation_context(
        current_message="show visual inspiration for this",
        recent_history=[{"role": "user", "content": "I need something tomorrow"}],
    )

    assert context.date_context == "tomorrow"
    assert context.activity is None
    assert context.occasion is None
    assert diagnostics["requires_clarification"] is True
    assert diagnostics["missing_information"] == ["occasion_or_activity"]


def test_activity_normalization_covers_common_families():
    cases = {
        "Gym this evening": ("gym", "training", "this_evening"),
        "Wedding Saturday": (None, None, "saturday"),
        "I have a client meeting tomorrow": (None, None, "tomorrow"),
    }
    for message, expected in cases.items():
        context, _ = resolve_style_conversation_context(current_message=message)
        assert (context.activity, context.activity_type, context.date_context) == expected

    context, _ = resolve_style_conversation_context(
        current_message="I have a client meeting tomorrow"
    )
    assert context.occasion == "client_meeting"


def test_court_sport_validator_rejects_formal_directions_and_fills_safe_fallbacks():
    directions = [
        {
            "title": "Contemporary Classic",
            "archetype": "Refined Weekend",
            "items": ["Oxford shirt", "Chinos", "Loafers"],
        }
    ]

    result = validate_activity_compatibility(directions, "court_sport")

    assert len(result) == 3
    text = str(result).lower()
    assert "oxford" not in text
    assert "loafers" not in text
    assert "refined weekend" not in text
    assert "court shoes" in text


def test_semantic_resolver_rejects_protected_and_unknown_context_fields():
    base = _semantic_clarification()
    assert resolver.validate_semantic_decision({**base, "anchor_item_id": "abc"}) is None
    assert resolver.validate_semantic_decision(
        {**base, "resolved_context": {"unknown_context": "x"}}
    ) is None
    assert resolver.validate_semantic_decision(
        {**base, "resolved_context": {"activity": "padel", "activity_type": "unknown"}}
    ) is None
    assert resolver.validate_semantic_decision(
        {**base, "execution": "delete board"}
    ) is None


def test_deterministic_information_path_does_not_call_semantic_resolver(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_generate_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM called")),
    )
    result = resolver.resolve_semantic_intent(
        current_message="What is color analysis?",
        recent_history=[{"role": "user", "content": "I have badminton tomorrow"}],
        deterministic={
            "domain": "style",
            "intent": "information",
            "action": "explain_style_concept",
            "response_mode": "text_only",
        },
    )
    assert result["decision_source"] == "deterministic_fast_path"
    assert result["response_mode"] == "text_only"


def test_module_chat_three_turns_returns_canonical_context(monkeypatch):
    def fake_semantic(**kwargs):
        if kwargs.get("deterministic") is not None:
            result = dict(kwargs["deterministic"])
            result["decision_source"] = "deterministic_fast_path"
            return result
        return _semantic_clarification()

    monkeypatch.setattr(chat, "resolve_semantic_intent", fake_semantic)
    monkeypatch.setattr(
        chat.style_reasoning_engine,
        "reason",
        lambda **kwargs: {
            "mode": "visual_inspiration",
            "occasion": "badminton",
            "advice": "A movement-ready court direction.",
            "visual_directions": [
                {
                    "title": "Court Performance",
                    "archetype": "Court Performance",
                    "items": ["Performance Tee", "Performance Shorts", "Court Shoes"],
                }
            ],
            "cta": [],
            "meta": {"source": "test"},
        },
    )
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    first = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "I need something for tomorrow", "request_id": "r1"},
    )
    assert first.status_code == 200
    assert first.json()["resolved_context"]["date_context"] == "tomorrow"
    assert first.json()["requires_clarification"] is True

    second = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "I have a badminton game",
            "history": [{"role": "user", "content": "I need something for tomorrow"}],
            "request_id": "r2",
        },
    )
    assert second.status_code == 200
    assert second.json()["resolved_context"]["date_context"] == "tomorrow"
    assert second.json()["resolved_context"]["activity"] == "badminton"

    third = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "show visual inspiration for this",
            "history": [
                {"role": "user", "content": "I need something for tomorrow"},
                {"role": "user", "content": "I have a badminton game"},
            ],
            "request_id": "r3",
        },
    )
    body = third.json()
    assert third.status_code == 200
    assert body["domain"] == "style"
    assert body["intent"] == "visual_inspiration"
    assert body["response_mode"] == "visual_inspiration"
    assert body["resolved_context"]["date_context"] == "tomorrow"
    assert body["resolved_context"]["activity"] == "badminton"
    assert body["resolved_context"]["activity_type"] == "court_sport"
    assert body["diagnostics"]["request_id"] == "r3"
