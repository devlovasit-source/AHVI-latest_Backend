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


# ---------- hanging phrase: unfinished copular gerund after a subordinator ----------

def test_without_being_is_truncated():
    assert looks_truncated("That works without being.") is True


def test_without_being_with_lead_in_is_truncated():
    assert looks_truncated("This can feel polished without being.") is True


def test_live_without_being_regression_paragraph_flagged():
    text = (
        "For the office, the priority is always to look polished and capable. "
        "We will focus on creating distinct looks that balance professionalism "
        "with comfort, ensuring you feel confident and ready for any task, "
        "without being."
    )
    assert looks_truncated(text) is True


# ---------- negative cases: valid sentences with/without "being" ----------

def test_being_prepared_matters_is_not_truncated():
    assert looks_truncated("Being prepared matters.") is False


def test_left_without_looking_is_not_truncated():
    assert looks_truncated("She left without looking.") is False


def test_crossed_without_stopping_is_not_truncated():
    assert looks_truncated("He crossed without stopping.") is False


def test_without_being_with_complement_is_not_truncated():
    assert looks_truncated(
        "This avoids anything too casual without being overly formal."
    ) is False


# ---------- hanging class: bare contracted subject+auxiliary/copula ----------

def test_meeting_or_is_truncated():
    assert looks_truncated(
        "These looks balance professional polish with modern comfort, "
        "ensuring you feel confident and appropriate for any meeting or."
    ) is True


def test_ensuring_youre_is_truncated():
    assert looks_truncated(
        "Focus on clean silhouettes and smart details that project "
        "confidence and efficiency, ensuring you're."
    ) is True


def test_so_youre_is_truncated():
    assert looks_truncated("Keep the silhouette clean so you're.") is True


def test_and_youre_is_truncated():
    assert looks_truncated("Choose the blazer and you're.") is True


def test_comma_were_is_truncated():
    assert looks_truncated("Once the layers are balanced, we're.") is True


def test_ive_alone_is_truncated():
    assert looks_truncated("I've.") is True


def test_youll_alone_is_truncated():
    assert looks_truncated("You'll.") is True


# ---------- negative cases: contraction present but sentence is complete ----------

def test_youre_ready_for_meeting_is_not_truncated():
    assert looks_truncated("You're ready for the meeting.") is False


def test_were_keeping_palette_neutral_is_not_truncated():
    assert looks_truncated("We're keeping the palette neutral.") is False


def test_youll_feel_polished_is_not_truncated():
    assert looks_truncated("You'll feel polished.") is False


def test_ive_kept_styling_minimal_is_not_truncated():
    assert looks_truncated("I've kept the styling minimal.") is False


def test_blazer_youre_wearing_is_not_truncated():
    assert looks_truncated("The blazer you're wearing works well.") is False


def test_direction_youre_after_is_not_truncated():
    assert looks_truncated("That's the direction you're after.") is False
