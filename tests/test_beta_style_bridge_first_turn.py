"""_resolve_item_references must delegate to the canonical owned-item resolver
on a first turn (no style_state.board_items yet), while existing board-state
follow-up resolution stays untouched (see GO IMPLEMENT audit, item 4).

Request-changes review round 2, item 4: prove this isn't dead code -- the
private-function tests below are necessary but not sufficient; also exercise
the PUBLIC interpret_style_followup entry point (the actual production
caller's boundary, routers/chat.py:_text_beta_instructions) end to end.
"""
from __future__ import annotations

from services.beta_style_bridge import _resolve_item_references, interpret_style_followup


def _it(name, role, item_id=None):
    iid = item_id or name.lower().replace(" ", "-")
    return {"id": iid, "item_id": iid, "name": name, "role": role, "category": role, "source": "wardrobe"}


RED_TOP = _it("Red Top", "top")
WHITE_SHIRT = _it("White Shirt", "top")
WARDROBE = [RED_TOP, WHITE_SHIRT]


def test_first_turn_with_no_board_items_and_no_wardrobe_returns_empty():
    result = _resolve_item_references("create an outfit using my red top", {"board_items": []})
    assert result == []


def test_first_turn_delegates_to_canonical_resolver_when_wardrobe_given():
    result = _resolve_item_references(
        "create an outfit using my red top", {"board_items": []}, wardrobe=WARDROBE
    )
    assert result == ["red-top"]


def test_first_turn_ambiguous_or_unresolved_never_silently_picked():
    wardrobe = [_it("Black Shirt", "top", "shirt-1"), _it("Black Shirt", "top", "shirt-2")]
    result = _resolve_item_references(
        "create an outfit using my black shirt", {"board_items": []}, wardrobe=wardrobe
    )
    assert result == []  # ambiguous -- never arbitrarily resolved


def test_existing_board_state_resolution_is_unaffected_by_wardrobe_param():
    """Passing a wardrobe when board_items already exist must not change the
    existing follow-up matching behavior."""
    state = {"board_items": [RED_TOP], "hero_item_id": ""}
    without = _resolve_item_references("keep the red top", state)
    with_wardrobe = _resolve_item_references("keep the red top", state, wardrobe=WARDROBE)
    assert without == with_wardrobe == ["red-top"]


# ── Public entry point (routers/chat.py's actual call boundary) ────────────

def test_public_interpret_style_followup_resolves_first_turn_wardrobe_item():
    """This is the exact call shape routers/chat.py uses:
    interpret_style_followup(user_input, _text_beta_state, wardrobe=request.wardrobe)
    on a first turn (no prior style_state). Without wardrobe wired through to
    _resolve_item_references AND without the state_ids filter being scoped to
    has_state, this silently produced empty preserve_item_ids even when the
    wardrobe was passed -- both were fixed together."""
    result = interpret_style_followup(
        "keep my Red Top", style_state=None, wardrobe=WARDROBE
    )
    assert result["preserve_item_ids"] == ["red-top"]


def test_public_interpret_style_followup_without_wardrobe_stays_empty():
    """Backward compatibility: omitting wardrobe (existing call shape) must
    not raise and must behave exactly as before (empty on first turn)."""
    result = interpret_style_followup("keep my Red Top", style_state=None)
    assert result["preserve_item_ids"] == []


def test_public_interpret_style_followup_existing_board_mutation_unaffected():
    """has_state=True path (existing board) must still restrict
    preserve/replace ids to items actually on the board, wardrobe or not --
    this is the invariant the has_state guard on the state_ids filter
    protects."""
    state = {
        "board_id": "b1",
        "board_items": [RED_TOP],
        "hero_item_id": "",
        "occasion": "daily",
        "source_mode": "wardrobe",
    }
    result = interpret_style_followup("keep the red top", style_state=state, wardrobe=WARDROBE)
    assert result["preserve_item_ids"] == ["red-top"]
    # Blue Jeans is in the wardrobe but not on the current board (and its
    # role, "bottom", doesn't collide with the board's "top") -- must never
    # leak into preserve/replace ids for a board mutation.
    wardrobe_with_jeans = WARDROBE + [_it("Blue Jeans", "bottom")]
    result2 = interpret_style_followup(
        "keep the blue jeans", style_state=state, wardrobe=wardrobe_with_jeans
    )
    assert result2["preserve_item_ids"] == []
