"""P1 (RC3): the real newline-destruction chokepoint.

The pre-existing tests/test_style_advice_bullet_format.py suite is green by
mocking services.style_reasoning_engine.generate_text and
routers.chat.chat_completion directly -- both mocks bypass
services.llm_service._call_gemini_text entirely, which is the one function
every real Gemini call in the app actually goes through
(chat_completion -> generate_text -> _call_gemini_text). That's exactly why
the physical device still lost bullet newlines despite the P1 commit: the
existing suite never exercised the real chokepoint. These tests do.
"""
from __future__ import annotations

from types import SimpleNamespace

import services.llm_service as llm_service


BULLETED = (
    "For a taller-looking silhouette:\n"
    "- Choose higher-rise trousers.\n"
    "- Keep stronger vertical continuity.\n"
    "- Avoid excessive pooling at the ankle."
)


class _FakeModels:
    def __init__(self, text: str):
        self._text = text

    def generate_content(self, *, model, contents, config):
        return SimpleNamespace(text=self._text)


class _FakeGeminiClient:
    def __init__(self, text: str):
        self.models = _FakeModels(text)


def test_call_gemini_text_preserves_newlines_through_tone_engine(monkeypatch):
    monkeypatch.setattr(llm_service, "_gemini_enabled", lambda: True)
    monkeypatch.setattr(
        llm_service, "_get_gemini_client", lambda timeout_seconds=None: _FakeGeminiClient(BULLETED)
    )

    result = llm_service._call_gemini_text("How can I dress to look taller?")

    assert result is not None
    assert "\n" in result, f"tone_engine.apply destroyed newlines at the real chokepoint: {result!r}"
    assert result.count("\n- ") >= 2 or result.count("\n-") >= 2, (
        f"bullet lines were not preserved: {result!r}"
    )


def test_call_gemini_text_still_scrubs_tone_as_before(monkeypatch):
    """Regression guard: the newline guard must not skip tone_engine.apply
    entirely -- forbidden-phrase scrubbing must still run."""
    monkeypatch.setattr(llm_service, "_gemini_enabled", lambda: True)
    monkeypatch.setattr(
        llm_service,
        "_get_gemini_client",
        lambda timeout_seconds=None: _FakeGeminiClient("Sure! This looks great."),
    )

    result = llm_service._call_gemini_text("What should I wear?")

    assert result is not None
    from brain.tone.tone_engine import tone_engine

    expected = tone_engine.apply("Sure! This looks great.", user_profile=None, signals=None)
    assert result == expected, f"tone_engine.apply must still run on single-line text: {result!r}"


def test_generate_text_end_to_end_preserves_newlines(monkeypatch):
    """One layer up: generate_text() (chat_completion's actual delegate for
    every /api/module-chat and /api/text advice reply) must not re-introduce
    flattening after _call_gemini_text returns."""
    monkeypatch.setattr(llm_service, "_gemini_enabled", lambda: True)
    monkeypatch.setattr(
        llm_service, "_get_gemini_client", lambda timeout_seconds=None: _FakeGeminiClient(BULLETED)
    )

    result = llm_service.generate_text("How can I dress to look taller?", usecase="style_chat")

    assert "\n" in result, f"newlines lost between _call_gemini_text and generate_text: {result!r}"
