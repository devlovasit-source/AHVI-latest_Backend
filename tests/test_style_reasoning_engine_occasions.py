
import pytest
from services.style_reasoning_engine import (_coerce_mode, _occasion_category, _build_reasoning_prompt)
from brain.engines.style_brief import build_brief, detect_occasion_from_tokens, resolve_occasion_archetype, tokenize
from services import stylist_knowledge_service

# Mock classify_style_mode since it's an external dependency
def mock_classify_style_mode(query, module_context, style_action, occasion=None):
    if occasion:
        if "coffee_date" in occasion:
            return "visual_inspiration" # For visual tests
        if "date_night" in occasion:
            return "style_advice"
        if "brunch" in occasion:
            return "visual_inspiration"
        if "conference" in occasion:
            return "style_advice"
        if "funeral" in occasion:
            return "style_advice"
    return "general"

# Patch the classify_style_mode in style_reasoning_engine for testing
stylist_knowledge_service.classify_style_mode = mock_classify_style_mode


def test_style_reasoning_uses_style_brief_contract():
    query = "show visual inspiration for a casual coffee date"
    context = {}
    mode = _coerce_mode(query, None, context)

    assert "canonical_occasion" in context
    assert context["canonical_occasion"] == "coffee_date"

    category, _, _, _ = _occasion_category(query, context)
    assert category == "social_occasion"

    # Verify that the canonical occasion is present in the prompt
    prompt = _build_reasoning_prompt(
        query=query,
        mode=mode,
        category=category,
        user_profile={},
        context=context
    )
    assert "Detected canonical occasion: coffee_date" in prompt
    assert "Coffee date: approachable confidence, intentional not corporate." in prompt # Check a rule from prompt

def test_long_first_coffee_date_routes_to_coffee_date():
    query = "first coffee date at a cafe on Saturday afternoon"
    context = {}
    mode = _coerce_mode(query, None, context)
    assert mode == "visual_inspiration" # Because mock_classify_style_mode returns visual_inspiration for coffee_date

    assert "canonical_occasion" in context
    assert context["canonical_occasion"] == "coffee_date"

    category, _, _, _ = _occasion_category(query, context)
    assert category == "social_occasion"

def test_date_night_dinner_remains_date_night():
    query = "date night dinner"
    context = {}
    mode = _coerce_mode(query, None, context)
    assert mode == "style_advice" # Because mock_classify_style_mode returns style_advice for date_night

    assert "canonical_occasion" in context
    assert context["canonical_occasion"] == "date_night"

    category, _, _, _ = _occasion_category(query, context)
    assert category == "social_occasion"

def test_brunch_date_not_generic_office_or_casual():
    query = "brunch date"
    context = {}
    mode = _coerce_mode(query, None, context)
    assert mode == "visual_inspiration" # Because mock_classify_style_mode returns visual_inspiration for brunch

    assert "canonical_occasion" in context
    assert context["canonical_occasion"] == "brunch"

    category, _, _, _ = _occasion_category(query, context)
    assert category == "social_occasion"

def test_conference_routes_to_work_occasion():
    query = "conference presentation"
    context = {}
    mode = _coerce_mode(query, None, context)
    assert mode == "style_advice"

    assert "canonical_occasion" in context
    assert context["canonical_occasion"] == "client_presentation" # From style_brief.py

    category, _, _, _ = _occasion_category(query, context)
    assert category == "work_occasion"

def test_funeral_routes_to_sensitive_occasion():
    query = "funeral service"
    context = {}
    mode = _coerce_mode(query, None, context)
    assert mode == "style_advice"

    assert "canonical_occasion" in context
    assert context["canonical_occasion"] == "funeral"

    category, _, _, _ = _occasion_category(query, context)
    assert category == "sensitive_occasion"
