from services import semantic_intent_resolver as resolver
from services.style_conversation_context import resolve_style_conversation_context


def test_temporal_override_replaces_date_and_keeps_occasion():
    context, _ = resolve_style_conversation_context(
        current_message="Actually Friday",
        recent_history=[{"role": "user", "content": "Wedding tomorrow"}],
    )

    assert context.occasion == "wedding"
    assert context.date_context == "friday"


def test_daypart_survives_referential_follow_up():
    context, _ = resolve_style_conversation_context(
        current_message="show inspiration for this",
        recent_history=[{"role": "user", "content": "Wedding Saturday night"}],
    )

    assert context.occasion == "wedding"
    assert context.date_context == "saturday"
    assert context.daypart == "night"
    assert context.referent["resolved_to"] == "wedding saturday night"


def test_temporal_correction_replaces_date_and_daypart_field_wisely():
    context, _ = resolve_style_conversation_context(
        current_message="Actually Sunday morning",
        recent_history=[{"role": "user", "content": "Wedding Saturday night"}],
    )

    assert context.occasion == "wedding"
    assert context.date_context == "sunday"
    assert context.daypart == "morning"


def test_unseen_bouldering_referent_is_not_a_game():
    context, _ = resolve_style_conversation_context(
        current_message="show inspiration for this",
        recent_history=[{"role": "user", "content": "I'm going bouldering tomorrow"}],
        semantic_context={"activity": "bouldering", "activity_type": "outdoor_active"},
    )

    assert context.activity == "bouldering"
    assert context.date_context == "tomorrow"
    assert "bouldering game" not in context.referent["resolved_to"]
    assert context.referent["resolved_to"] == "bouldering tomorrow"


def test_theme_park_referent_is_generic():
    context, _ = resolve_style_conversation_context(
        current_message="what should I wear for that?",
        recent_history=[
            {"role": "user", "content": "I'm taking the kids to a theme park this weekend"}
        ],
        semantic_context={"activity": "theme park", "activity_type": "outdoor_active"},
    )

    assert context.activity == "theme park"
    assert context.date_context == "this_weekend"
    assert "theme park game" not in context.referent["resolved_to"]


def test_court_sport_fallback_copy_is_activity_neutral():
    from services.style_reasoning_engine import _COURT_SPORT_FALLBACKS

    fallback_text = str(_COURT_SPORT_FALLBACKS).lower()
    assert "badminton look" not in fallback_text
    assert "court look" in fallback_text


def test_professional_temporal_correction_preserves_context():
    context, _ = resolve_style_conversation_context(
        current_message="Actually Friday",
        recent_history=[{"role": "user", "content": "Investor presentation tomorrow"}],
        semantic_context={"occasion": "investor_presentation"},
    )

    assert context.occasion == "investor_presentation"
    assert context.date_context == "friday"


def test_ambiguous_context_still_clarifies():
    context, diagnostics = resolve_style_conversation_context(
        current_message="show inspiration for this",
        recent_history=[{"role": "user", "content": "I need something tomorrow"}],
    )

    assert context.date_context == "tomorrow"
    assert context.activity is None
    assert context.occasion is None
    assert diagnostics["requires_clarification"] is True


def test_deterministic_fast_path_remains_llm_free(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_generate_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected LLM call")),
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


def test_semantic_resolver_accepts_bounded_structured_referent():
    result = resolver.validate_semantic_decision(
        {
            "domain": "style",
            "intent": "inspiration",
            "action": "provide_visual_inspiration",
            "response_mode": "visual_inspiration",
            "confidence": 0.95,
            "requires_clarification": False,
            "resolved_context": {
                "date_context": "tomorrow",
                "daypart": "night",
                "activity": "bouldering",
                "activity_type": "outdoor_active",
            },
            "constraints": {"required": [], "avoid": []},
            "referent": {
                "text": "this",
                "type": "activity",
                "label": "bouldering",
                "temporal": {"relative_date": "tomorrow", "daypart": "night"},
                "resolved_to": "bouldering tomorrow night",
                "confidence": 0.95,
            },
            "reason_codes": [],
            "missing_information": [],
        }
    )

    assert result["referent"]["type"] == "activity"
    assert result["resolved_context"]["daypart"] == "night"


def test_referent_variants_can_be_structurally_resolved():
    for phrase in ("for this", "for that", "for it", "same thing", "what about this?", "the event I mentioned"):
        context, _ = resolve_style_conversation_context(
            current_message=phrase,
            recent_history=[{"role": "user", "content": "Client dinner tomorrow"}],
            semantic_referent={
                "text": phrase,
                "type": "occasion",
                "label": "client dinner",
                "temporal": {"relative_date": "tomorrow"},
                "resolved_to": "client dinner tomorrow",
                "confidence": 0.95,
            },
        )
        assert context.referent["resolved_to"] == "client dinner tomorrow"
