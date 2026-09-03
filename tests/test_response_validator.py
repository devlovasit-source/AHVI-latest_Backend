"""Focused tests for brain.response_validator.looks_truncated's phrase-level
hanging-construction detector (the "...might feel out." PR#54 follow-up)."""

from __future__ import annotations

from brain.response_validator import looks_truncated


# ---------- hanging phrase: "feel(s/ing) out" missing its complement ----------

def test_might_feel_out_is_truncated():
    assert looks_truncated("That might feel out.") is True


def test_may_feel_out_is_truncated():
    assert looks_truncated("It may feel out.") is True


def test_could_feel_out_is_truncated():
    assert looks_truncated("It could feel out.") is True


def test_live_regression_paragraph_flagged():
    text = (
        "For the office, the priority is always to look polished and capable. "
        "We'll focus on creating distinct looks that feel intentional and appropriate, "
        "avoiding anything too casual or overly formal that might feel out."
    )
    assert looks_truncated(text) is True


# ---------- existing hanging-connector-word cases remain covered ----------

def test_hanging_while_still_truncated():
    assert looks_truncated("This works well while.") is True


def test_hanging_and_still_truncated():
    assert looks_truncated("Balanced proportions and.") is True


# ---------- negative cases: valid sentences containing/ending in "out" ----------

def test_stands_out_is_not_truncated():
    assert looks_truncated("The blazer really stands out.") is False


def test_head_out_is_not_truncated():
    assert looks_truncated("Let's head out.") is False


def test_worked_out_is_not_truncated():
    assert looks_truncated("That worked out.") is False


def test_lights_are_out_is_not_truncated():
    assert looks_truncated("The lights are out.") is False


def test_feel_off_is_not_truncated():
    assert looks_truncated("I feel off today.") is False


def test_plain_complete_sentence_is_not_truncated():
    assert looks_truncated("This looks polished and intentional.") is False


def test_feel_balanced_is_not_truncated():
    assert looks_truncated("The proportions feel balanced.") is False
