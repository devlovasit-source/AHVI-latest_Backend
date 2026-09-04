"""Focused tests for brain.response_validator.looks_truncated's phrase-level
hanging-construction detector (the "...might feel out." PR#54 follow-up)."""

from __future__ import annotations

from brain.response_validator import looks_truncated, salvage_before_trailing_adjunct


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


# ---------- hanging class: bare terminal article ("a"/"an"/"the") ----------

def test_finish_the_outfit_with_a_is_truncated():
    assert looks_truncated("Finish the outfit with a.") is True


def test_choose_an_is_truncated():
    assert looks_truncated("Choose an.") is True


def test_layer_with_the_is_truncated():
    assert looks_truncated("Layer with the.") is True


def test_article_followed_by_noun_is_not_truncated():
    assert looks_truncated("Finish the outfit with a blazer.") is False


def test_choose_the_blazer_is_not_truncated():
    assert looks_truncated("Choose the blazer.") is False


# ---------- bare-article false positive: capitalized single-letter label/noun ----------

def test_plan_a_is_not_truncated():
    assert looks_truncated("Plan A.") is False


def test_option_a_is_not_truncated():
    assert looks_truncated("Option A.") is False


def test_vitamin_a_is_not_truncated():
    assert looks_truncated("Vitamin A.") is False


def test_use_plan_a_is_not_truncated():
    assert looks_truncated("Use Plan A.") is False


def test_take_vitamin_a_is_not_truncated():
    assert looks_truncated("Take vitamin A.") is False


def test_lowercase_a_still_truncated_even_capitalized_label_valid():
    """The case-sensitivity fix must not weaken the original bare-article
    detection -- only a capitalized single-letter label is exempt."""
    assert looks_truncated("Finish with a.") is True
    assert looks_truncated("Choose an.") is True
    assert looks_truncated("Layer with the.") is True


# ---------- hanging class: unfinished trailing adjunct (subordinator + bare gerund) ----------

def test_while_maintaining_is_truncated():
    assert looks_truncated(
        "This works well while maintaining."
    ) is True


def test_while_maintaining_a_is_truncated():
    assert looks_truncated(
        "Move through the day with ease while maintaining a."
    ) is True


def test_live_meeting_or_paragraph_still_truncated():
    assert looks_truncated(
        "These looks balance professional polish with modern comfort, "
        "ensuring you feel confident and appropriate for any meeting or."
    ) is True


def test_while_keeping_the_is_truncated():
    assert looks_truncated("Keep it simple while keeping the.") is True


def test_while_maintaining_a_polished_silhouette_is_not_truncated():
    assert looks_truncated(
        "Choose tailored pieces while maintaining a polished silhouette."
    ) is False


def test_while_keeping_the_palette_neutral_is_not_truncated():
    assert looks_truncated(
        "Layer with confidence while keeping the palette neutral."
    ) is False


def test_without_looking_overly_formal_is_not_truncated():
    assert looks_truncated(
        "Soften the blazer without looking overly formal."
    ) is False


def test_after_adding_a_lightweight_layer_is_not_truncated():
    assert looks_truncated(
        "The look feels complete after adding a lightweight layer."
    ) is False


# ---------- salvage: safe prefix extraction before a malformed trailing adjunct ----------

def test_salvage_strips_bare_while_maintaining():
    assert salvage_before_trailing_adjunct(
        "Keep the look polished while maintaining."
    ) == "Keep the look polished."


def test_salvage_strips_while_maintaining_a():
    assert salvage_before_trailing_adjunct(
        "Move through the day with ease while maintaining a."
    ) == "Move through the day with ease."


def test_salvage_returns_none_when_nothing_precedes_adjunct():
    assert salvage_before_trailing_adjunct("While maintaining a.") is None


def test_salvage_returns_none_for_clean_text():
    assert salvage_before_trailing_adjunct("This is already complete.") is None
