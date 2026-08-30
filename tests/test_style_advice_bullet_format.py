"""P1: Style Advice bullet-format presentation contract.

Two independent LLM boundaries answer open-ended style advice questions
(skin tone, body proportions, general styling):
  A. /api/text -> services.style_reasoning_engine (JSON-schema advice modes)
  B. /api/module-chat -> routers.chat._module_llm_response (free-text)

Both must receive the same STYLE_ADVICE_FORMAT_CONTRACT instruction, and
neither boundary's post-processing may destroy a bulleted answer once the
model returns one. These tests check the structural contract (prompt
contains the instruction; a bulleted response survives post-processing
unmodified) -- never exact LLM prose, per task instructions.
"""
from __future__ import annotations

import json

import services.style_reasoning_engine as sre
from routers import chat


BULLETED_ADVICE = (
    "Higher-rise trousers and continuous tones read taller.\n"
    "- Choose higher-rise trousers to visually lengthen your legs.\n"
    "- Keep top and bottom tones relatively continuous.\n"
    "- Avoid excessive trouser pooling at the ankle.\n"
    "- Minimize strong horizontal breaks through the outfit."
)


def test_advice_mode_prompt_includes_format_contract():
    prompt = sre._build_reasoning_prompt(
        query="How can I dress to look taller?",
        mode=sre.BODY_PROPORTION_ADVICE,
        category="daily",
        user_profile={},
        context={},
    )
    assert sre.STYLE_ADVICE_FORMAT_CONTRACT in prompt


def test_non_advice_mode_prompt_omits_format_contract():
    prompt = sre._build_reasoning_prompt(
        query="Style me for a Pooja",
        mode=sre.WARDROBE_STYLE,
        category="daily",
        user_profile={},
        context={},
    )
    assert sre.STYLE_ADVICE_FORMAT_CONTRACT not in prompt


def test_advice_mode_bulleted_response_survives_to_final_advice_field(monkeypatch):
    """A bulleted stylist_reasoning from the model must reach response["advice"]
    unmodified -- the generic 2-sentence/320-char board-caption compaction
    must be bypassed for advice modes (see STYLE_ADVICE_FORMAT_CONTRACT)."""
    monkeypatch.setattr(
        sre,
        "generate_text",
        lambda *a, **k: json.dumps({
            "mode": "body_proportion_advice",
            "stylist_reasoning": BULLETED_ADVICE,
            "principles": ["Vertical lines elongate"],
            "do": ["Choose higher-rise trousers"],
            "avoid": ["Excessive trouser pooling"],
            "outfit_examples": ["Tailored trouser with a tucked-in top"],
            "confidence": 0.9,
        }),
    )
    result = sre.reason(
        "How can I dress to look taller?",
        intent={"intent": "body_proportion_advice", "confidence": 0.9},
        user_profile={},
        context={},
    )
    assert result["advice"].count("\n- ") == 4, f"bullets were not preserved: {result['advice']!r}"
    assert "Choose higher-rise trousers" in result["advice"]


def test_non_advice_mode_still_compacted_to_two_sentences(monkeypatch):
    """Regression guard: the bypass introduced for advice modes must not
    leak into board/execution modes, which still need a short caption."""
    long_prose = " ".join([f"Sentence number {i}." for i in range(1, 20)])
    monkeypatch.setattr(
        sre,
        "generate_text",
        lambda *a, **k: json.dumps({
            "mode": "style_advice",
            "stylist_reasoning": long_prose,
            "confidence": 0.9,
        }),
    )
    result = sre.reason(
        "Style me for a Pooja",
        intent={"intent": "style_advice", "confidence": 0.9},
        user_profile={},
        context={"occasion": "pooja"},
    )
    assert len(result["advice"]) <= 320, f"non-advice mode was not compacted: {result['advice']!r}"


def test_module_chat_advice_reply_includes_format_contract(monkeypatch):
    captured = {}

    def _fake_chat_completion(messages, *, system_instruction="", **kwargs):
        captured["system_instruction"] = system_instruction
        return "placeholder answer"

    monkeypatch.setattr(chat, "chat_completion", _fake_chat_completion)

    chat._module_llm_response(
        module="style",
        user_message="What colour suits my skin tone?",
        history=[],
        context_data={},
        user_profile={},
        is_advice=True,
    )
    assert chat.STYLE_ADVICE_FORMAT_CONTRACT in captured["system_instruction"]


def test_module_chat_non_advice_reply_omits_format_contract(monkeypatch):
    captured = {}

    def _fake_chat_completion(messages, *, system_instruction="", **kwargs):
        captured["system_instruction"] = system_instruction
        return "placeholder answer"

    monkeypatch.setattr(chat, "chat_completion", _fake_chat_completion)

    chat._module_llm_response(
        module="style",
        user_message="How many jackets do I own?",
        history=[],
        context_data={},
        user_profile={},
        is_advice=False,
    )
    assert chat.STYLE_ADVICE_FORMAT_CONTRACT not in captured["system_instruction"]


# ─────────────────────────────────────────────────────────────────────────
# PHASE 1 TRACE FIXTURE -- same shape as the task's own worked example:
# an opening sentence, a blank-line paragraph break, then bullet lines using
# a marker (deliberately "•", NOT the "- " our own prompt asks for) so
# these tests prove the preservation logic is structural (any real newline
# survives) rather than sniffing for one specific bullet character.
# ─────────────────────────────────────────────────────────────────────────

BULLET_MARK = "•"
BULLETED_HEIGHT_ADVICE = (
    "For a taller-looking silhouette:\n\n"
    f"{BULLET_MARK} Choose higher-rise trousers.\n"
    f"{BULLET_MARK} Keep stronger vertical continuity.\n"
    f"{BULLET_MARK} Avoid excessive pooling at the ankle.\n"
    f"{BULLET_MARK} Minimize strong horizontal breaks."
)
BULLETED_SKIN_TONE_ADVICE = (
    "Your saved shade pairs well with a few color families:\n\n"
    f"{BULLET_MARK} Lean into warm neutrals like camel and olive.\n"
    f"{BULLET_MARK} Try jewel tones such as emerald or burgundy.\n"
    f"{BULLET_MARK} Keep bright whites for high-contrast moments.\n"
    f"{BULLET_MARK} Avoid washed-out pastels near your face."
)


def _bullet_line_count(text: str) -> int:
    return sum(1 for line in text.split("\n") if line.strip().startswith(BULLET_MARK))


def test_skin_tone_advice_bullets_survive_and_no_undertone_invented(monkeypatch):
    monkeypatch.setattr(
        sre,
        "generate_text",
        lambda *a, **k: json.dumps({
            "mode": "color_advice",
            "stylist_reasoning": BULLETED_SKIN_TONE_ADVICE,
            "recommended_colors": ["camel", "olive", "emerald", "burgundy"],
            "avoid_colors": ["pastel"],
            "why": ["Warm neutrals echo the saved shade family"],
            "outfit_palettes": ["Camel + white + emerald accent"],
            "confidence": 0.9,
        }),
    )
    result = sre.reason(
        "What colour suits my skin tone?",
        intent={"intent": "color_advice", "confidence": 0.9},
        user_profile={"personal_style_profile": {"skin_tone": {"swatch_hex": "#a86f4f"}}},
        context={},
    )
    assert result["mode"] == "color_advice"
    assert not result.get("should_generate_board")
    assert _bullet_line_count(result["advice"]) >= 3, f"bullets lost: {result['advice']!r}"
    for word in ("warm undertone", "cool undertone", "neutral undertone"):
        assert word not in result["advice"].lower()

    prompt = sre._build_reasoning_prompt(
        query="What colour suits my skin tone?",
        mode=sre.COLOR_ADVICE,
        category="daily",
        user_profile={},
        context={},
        style_ctx={"personal_style_profile": {"skin_tone": {"swatch_hex": "#a86f4f"}}},
    )
    assert "SHADE EVIDENCE ONLY" in prompt
    assert "Never state or" in prompt
    assert sre.STYLE_ADVICE_FORMAT_CONTRACT in prompt


def test_height_exact_query_bullets_survive_and_no_board(monkeypatch):
    monkeypatch.setattr(
        sre,
        "generate_text",
        lambda *a, **k: json.dumps({
            "mode": "body_proportion_advice",
            "stylist_reasoning": BULLETED_HEIGHT_ADVICE,
            "principles": ["Vertical lines elongate"],
            "do": ["Choose higher-rise trousers"],
            "avoid": ["Excessive pooling"],
            "outfit_examples": ["Tailored trouser with a tucked-in top"],
            "confidence": 0.9,
        }),
    )
    result = sre.reason(
        "I am 5feet 8inches how will I look taller",
        intent={"intent": "body_proportion_advice", "confidence": 0.9},
        user_profile={},
        context={},
    )
    assert not result.get("should_generate_board")
    assert _bullet_line_count(result["advice"]) >= 3, f"bullets lost: {result['advice']!r}"


def test_height_semantic_variant_bullets_survive(monkeypatch):
    monkeypatch.setattr(
        sre,
        "generate_text",
        lambda *a, **k: json.dumps({
            "mode": "body_proportion_advice",
            "stylist_reasoning": BULLETED_HEIGHT_ADVICE,
            "principles": ["Vertical lines elongate"],
            "confidence": 0.9,
        }),
    )
    result = sre.reason(
        "How can I dress to look taller?",
        intent={"intent": "body_proportion_advice", "confidence": 0.9},
        user_profile={},
        context={},
    )
    assert not result.get("should_generate_board")
    assert _bullet_line_count(result["advice"]) >= 3, f"bullets lost: {result['advice']!r}"


def test_body_type_query_bullets_survive(monkeypatch):
    monkeypatch.setattr(
        sre,
        "generate_text",
        lambda *a, **k: json.dumps({
            "mode": "body_proportion_advice",
            "stylist_reasoning": BULLETED_HEIGHT_ADVICE,
            "principles": ["Balance proportions"],
            "confidence": 0.9,
        }),
    )
    result = sre.reason(
        "What will suit my body type?",
        intent={"intent": "body_proportion_advice", "confidence": 0.9},
        user_profile={},
        context={},
    )
    assert _bullet_line_count(result["advice"]) >= 3, f"bullets lost: {result['advice']!r}"


def test_style_surface_and_wardrobe_surface_share_the_same_format_contract(monkeypatch):
    """/api/text (style_reasoning_engine) and /api/module-chat
    (_module_llm_response) are two different LLM boundaries; both must be
    instructed with the identical STYLE_ADVICE_FORMAT_CONTRACT text for an
    advice-classified question, regardless of which surface asked."""
    prompt = sre._build_reasoning_prompt(
        query="What will suit my body type?",
        mode=sre.BODY_PROPORTION_ADVICE,
        category="daily",
        user_profile={},
        context={},
    )
    assert sre.STYLE_ADVICE_FORMAT_CONTRACT in prompt

    captured = {}

    def _fake_chat_completion(messages, *, system_instruction="", **kwargs):
        captured["system_instruction"] = system_instruction
        return "placeholder"

    monkeypatch.setattr(chat, "chat_completion", _fake_chat_completion)
    chat._module_llm_response(
        module="style",
        user_message="What will suit my body type?",
        history=[],
        context_data={},
        user_profile={},
        is_advice=True,
    )
    assert chat.STYLE_ADVICE_FORMAT_CONTRACT in captured["system_instruction"]
    assert chat.STYLE_ADVICE_FORMAT_CONTRACT == sre.STYLE_ADVICE_FORMAT_CONTRACT


def test_execution_mode_unaffected_by_advice_bypass(monkeypatch):
    """Board/execution modes (e.g. WARDROBE_STYLE, reached by 'Style me to
    look taller' / 'Use my wardrobe to make me look taller') never enter the
    advice branch and must keep generating boards exactly as before."""
    result = sre.reason(
        "Use my wardrobe to make me look taller",
        intent={"intent": "wardrobe_style", "confidence": 0.9},
        user_profile={},
        context={},
    )
    assert result["mode"] == sre.WARDROBE_STYLE
    assert result.get("should_generate_board") is True
    assert sre.STYLE_ADVICE_FORMAT_CONTRACT not in json.dumps(result.get("meta") or {})


def test_non_advice_tone_engine_output_exactly_unchanged():
    """Direct regression on the shared brain.tone.tone_engine module itself
    (reverted to base/P0 state, never modified by this P1 work): a
    multi-line, non-advice string is still collapsed to a single line,
    proving the newline-preservation trick lives entirely in
    style_reasoning_engine.py and never alters tone_engine.py's own
    behavior for any of its other ~17 call sites."""
    from brain.tone.tone_engine import tone_engine

    raw = "Board caption line one.\nBoard caption line two.\n\nExtra line."
    expected = tone_engine.apply(
        raw.replace("\n", " "),
        user_profile={}, signals={"context_mode": "styling"}, context={},
    )
    actual = tone_engine.apply(
        raw, user_profile={}, signals={"context_mode": "styling"}, context={},
    )
    assert "\n" not in actual, f"tone_engine.py must still flatten newlines by default: {actual!r}"
    assert actual == expected
