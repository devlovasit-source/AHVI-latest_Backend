"""Regression tests for the system-instruction propagation bug.

Root cause: chat_completion(system_instruction=X) folded X into the prompt
text as a "System:\\n{X}" line and never forwarded it to generate_text() /
_call_gemini_text(), which meant every Gemini call's real system_instruction
config field silently fell back to the global, styling-centric
AHVI_SYSTEM_PROMPT -- regardless of what module-specific instruction the
caller built.

These tests intercept at the Gemini client boundary (fake genai client) so
no live model call is made and no other business logic is mocked away.
"""
import services.llm_service as llm_service
from brain.tone.tone_engine import normalize_context_mode, tone_engine
from prompts.core_prompts import AHVI_SYSTEM_PROMPT


class _FakeResponse:
    text = "ok"


class _FakeModels:
    def __init__(self, captured):
        self._captured = captured

    def generate_content(self, *, model, contents, config):
        self._captured["model"] = model
        self._captured["contents"] = contents
        self._captured["system_instruction"] = config.system_instruction
        return _FakeResponse()


class _FakeClient:
    def __init__(self, captured):
        self.models = _FakeModels(captured)


def _patch_gemini(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service, "AI_PROVIDER", "gemini")
    monkeypatch.setattr(llm_service, "_get_gemini_client", lambda timeout_seconds=None: _FakeClient(captured))
    return captured


# ---------------------------------------------------------------------------
# A. system_instruction propagation
# ---------------------------------------------------------------------------
def test_chat_completion_propagates_caller_system_instruction_to_gemini(monkeypatch):
    captured = _patch_gemini(monkeypatch)

    llm_service.chat_completion(
        [{"role": "user", "content": "hello"}],
        system_instruction="CONVERSATION_SENTINEL",
        signals={"context_mode": "chat"},
    )

    assert captured["system_instruction"] == "CONVERSATION_SENTINEL"
    # The instruction must be a first-class config field, not just folded
    # into the free-text contents.
    assert "CONVERSATION_SENTINEL" not in captured["contents"]


# ---------------------------------------------------------------------------
# B. fallback when caller supplies nothing
# ---------------------------------------------------------------------------
def test_chat_completion_falls_back_to_global_prompt_when_no_override(monkeypatch):
    captured = _patch_gemini(monkeypatch)

    llm_service.chat_completion(
        [{"role": "user", "content": "hello"}],
        signals={"context_mode": "styling"},
    )

    assert captured["system_instruction"] == AHVI_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# C. context-mode alias normalization
# ---------------------------------------------------------------------------
def test_context_mode_aliases_normalize_to_canonical_names():
    expected = {
        "general_chat": "conversation",
        "chat": "conversation",
        "general": "conversation",
        "style": "styling",
        "wardrobe": "styling",
        "daily_wear": "styling",
        "calendar": "planning",
        "planner": "planning",
    }
    for raw, canonical in expected.items():
        assert normalize_context_mode(raw) == canonical


# ---------------------------------------------------------------------------
# D. conversation config resolves via build_prompt_tone
# ---------------------------------------------------------------------------
def test_build_prompt_tone_resolves_conversation_rules_for_chat_context():
    tone = tone_engine.build_prompt_tone(signals={"context_mode": "chat"})
    assert tone["context_mode"] == "conversation"
    assert "Do not proactively introduce" in tone["tone_instruction"]
    # Must not be the generic styling-flavored default fallback string.
    assert tone["tone_instruction"] != "Warm, concise, practical AHVI styling tone."


# ---------------------------------------------------------------------------
# E. no style leakage in the conversation instruction
# ---------------------------------------------------------------------------
def test_conversation_instruction_does_not_tell_ahvi_to_pitch_style():
    tone = tone_engine.build_prompt_tone(signals={"context_mode": "general_chat"})
    instruction = tone["tone_instruction"].lower()
    forbidden_affirmative_phrases = [
        "suggest an outfit",
        "recommend an outfit",
        "suggest wardrobe",
        "recommend wardrobe",
        "suggest what to wear",
        "a fresh outfit can help",
    ]
    for phrase in forbidden_affirmative_phrases:
        assert phrase not in instruction
    assert "do not proactively introduce" in instruction


# ---------------------------------------------------------------------------
# F. styling context is unaffected (no regression)
# ---------------------------------------------------------------------------
def test_styling_context_mode_still_resolves_and_is_unaffected(monkeypatch):
    captured = _patch_gemini(monkeypatch)

    llm_service.chat_completion(
        [{"role": "user", "content": "what should I wear tonight"}],
        system_instruction="STYLE_SENTINEL",
        signals={"context_mode": "style"},
    )

    assert captured["system_instruction"] == "STYLE_SENTINEL"
    tone = tone_engine.build_prompt_tone(signals={"context_mode": "style"})
    assert tone["context_mode"] == "styling"


def test_planning_context_mode_normalizes_and_is_unaffected():
    tone = tone_engine.build_prompt_tone(signals={"context_mode": "calendar"})
    assert tone["context_mode"] == "planning"
