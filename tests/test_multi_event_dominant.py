"""Multi-event (chat path) dominant-occasion ordering.

The style_reasoning prompt reads sub_occasions in order, so the
higher-formality event must lead — otherwise "gym then brunch" steers the
look toward athleisure. Chronology stays in time_sequence.
"""

from __future__ import annotations

import pytest

from services.style_context_service import detect_multi_event


@pytest.mark.parametrize(
    "query,dominant_in,not_first",
    [
        ("gym then brunch", {"brunch"}, "workout"),
        ("workout then office", {"office"}, "workout"),
        ("wedding reception after work", {"wedding", "reception"}, "work"),
        ("airport then office meeting", {"office_meeting"}, "travel"),
        ("client meeting then drinks", {"client_meeting"}, "drinks"),
    ],
)
def test_dominant_leads(query, dominant_in, not_first):
    ctx = detect_multi_event(query)
    assert ctx is not None, f"{query!r} not detected as multi-event"
    assert ctx["dominant_occasion"] in dominant_in, ctx["sub_occasions"]
    assert ctx["sub_occasions"][0] in dominant_in
    assert ctx["sub_occasions"][0] != not_first


def test_chronology_preserved_in_time_sequence():
    ctx = detect_multi_event("gym then brunch")
    seq = [s["event"] for s in ctx["time_sequence"]]
    # time_sequence keeps the order as written (gym first), even though
    # sub_occasions now leads with brunch.
    assert seq[0] == "workout"


def test_single_event_still_none():
    assert detect_multi_event("gym workout today") is None
    assert detect_multi_event("client meeting tomorrow") is None
