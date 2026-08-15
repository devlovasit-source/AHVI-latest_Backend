"""Release-blocker regression: a semantically dependent follow-up
("similar looks with available items", "show another") must preserve the
occasion carried from the prior turn instead of silently downgrading to
generic daily wear. An explicit new occasion in the follow-up still
overrides, and a fresh conversation with no prior occasion must never invent
one."""

from __future__ import annotations

from services.style_flow_service import interpret_occasion, resolve_occasion_with_continuity


def _resolve(ctx: dict, query: str) -> str:
    ctx = dict(ctx)
    occasion_interpretation = interpret_occasion(query, ctx)
    return resolve_occasion_with_continuity(query, ctx, occasion_interpretation)


def test_dependent_followup_preserves_prior_occasion():
    assert _resolve({"occasion": "date"}, "similar looks with available items") == "date_night"


def test_show_another_preserves_prior_occasion():
    assert _resolve({"occasion": "date"}, "show another") == "date_night"


def test_explicit_new_occasion_overrides_prior_occasion():
    assert _resolve({"occasion": "date"}, "actually make it for office") == "office"


def test_fresh_followup_with_no_prior_context_does_not_invent_occasion():
    result = _resolve({}, "similar looks with available items")
    assert result != "date_night"
    assert result == "daily"


def test_fresh_first_request_still_detects_its_own_occasion():
    """Unaffected control: a first message with no prior context and an
    explicit occasion of its own must resolve normally."""
    assert _resolve({}, "Outfit for a first date") == "first_date"


def test_prior_occasion_switches_when_new_message_is_explicit():
    """Unaffected control: an explicit occasion change (not a bare
    dependent follow-up) must switch, not just append to, the prior one."""
    assert _resolve({"occasion": "office"}, "give me a workout outfit") == "workout"
