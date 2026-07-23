import os
import io
import random

import pytest
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from services.beta_style_bridge import (
    decorate_style_response,
    engine_dispatch,
    interpret_style_followup,
    normalize_style_state,
    refine_style_response,
    validate_style_response,
    visual_intelligence,
)


STATE = {
    "board_id": "board-1",
    "board_items": [
        {"item_id": "shirt-123", "role": "top", "source": "wardrobe", "name": "Navy shirt"},
        {"item_id": "pants-456", "role": "bottom", "source": "wardrobe", "name": "Beige trousers"},
        {"item_id": "shoe-789", "role": "footwear", "source": "wardrobe", "name": "Black heels"},
    ],
    "occasion": "office",
    "source_mode": "wardrobe_only",
    "hero_item_id": "shirt-123",
    "locked_item_ids": ["shirt-123"],
    "active_constraints": {"exclude": [], "preserve": ["shirt-123"]},
}


BOARD = {
    "success": True,
    "cards": [{
        "id": "board-1",
        "occasion": "office",
        "items": [
            {"item_id": "shirt-123", "role": "top", "source": "wardrobe", "name": "Navy shirt"},
            {"item_id": "pants-456", "role": "bottom", "source": "wardrobe", "name": "Beige trousers"},
            {"item_id": "shoe-999", "role": "footwear", "source": "wardrobe", "name": "Brown loafers"},
        ],
    }],
}


def test_state_is_compact_deterministic_and_rejects_image_bytes():
    state = normalize_style_state({**STATE, "image_base64": "abc", "board_items": [
        {**STATE["board_items"][0], "image_base64": "abc"},
        *STATE["board_items"][1:],
    ]})
    assert "image_base64" not in state
    assert all("image_base64" not in item for item in state["board_items"])
    assert len(state["board_content_hash"]) == 64
    assert state == normalize_style_state(state)


@pytest.mark.parametrize(
    ("prompt", "action"),
    [
        ("Keep the top. Hate everything else.", "refine_current_board"),
        ("no heels n no black pls", "refine_current_board"),
        ("Change only the shoes.", "refine_current_board"),
        ("Why does this work visually?", "explain_current_board"),
        ("What is visually weak here?", "critique_current_board"),
        ("What is missing from my wardrobe?", "identify_wardrobe_gap"),
        ("Make it better.", "ask_clarification"),
        ("Use my wardrobe but show one inspiration option.", "ask_clarification"),
    ],
)
def test_followup_actions(prompt, action):
    assert interpret_style_followup(prompt, STATE)["action"] == action


def test_router_never_invents_item_ids():
    routed = interpret_style_followup("keep this shirt and change only shoes", STATE)
    allowed = {item["item_id"] for item in STATE["board_items"]}
    assert set(routed["preserve_item_ids"] + routed["replace_item_ids"]) <= allowed


def test_informal_multiple_exclusions_are_preserved():
    routed = interpret_style_followup("no heels n no black pls", STATE)
    assert routed["excluded_terms"] == ["heels", "black"]


def test_dispatch_reuses_existing_engine_names():
    routed = interpret_style_followup("keep this shirt", STATE)
    assert engine_dispatch(routed)["engine"] == "fixed_item_style_flow"


def test_validator_reports_honest_partial_result():
    routed = interpret_style_followup("keep top no loafers", STATE)
    status = validate_style_response(BOARD, routed)
    assert status["final_validation_status"] == "partial"
    assert status["fallback_reason"] == "constraints_not_fully_satisfied"


def test_decorator_is_additive_and_preserves_legacy_fields():
    routed = interpret_style_followup("change only shoes", STATE)
    result = decorate_style_response(BOARD, previous_state=STATE, instructions=routed)
    assert result["success"] is True
    assert result["cards"] == BOARD["cards"]
    assert result["style_state"]["board_items"]
    assert "understood" in result and "constraint_status" in result


def test_visual_flag_defaults_off_and_does_not_call_provider(monkeypatch):
    monkeypatch.delenv("AHVI_BETA_STYLE_VISION_ENABLED", raising=False)
    calls = []
    assert visual_intelligence(
        state=STATE,
        image_base64="abc",
        requested=True,
        vision_call=lambda **kwargs: calls.append(kwargs),
    ) is None
    assert calls == []


def test_visual_timeout_is_non_blocking(monkeypatch):
    monkeypatch.setenv("AHVI_BETA_STYLE_VISION_ENABLED", "true")
    def fail(**_kwargs):
        raise TimeoutError
    assert visual_intelligence(
        state=STATE, image_base64="abc", requested=True, vision_call=fail
    ) is None


def test_visual_rejects_invented_item_ids(monkeypatch):
    monkeypatch.setenv("AHVI_BETA_STYLE_VISION_ENABLED", "true")
    def fake(**_kwargs):
        return {"summary": "Balanced", "hero_item_id": "invented"}, "mock"
    assert visual_intelligence(
        state=STATE, image_base64="abc2", requested=True, vision_call=fake
    ) is None


def test_visual_is_cached_once_per_hash(monkeypatch):
    monkeypatch.setenv("AHVI_BETA_STYLE_VISION_ENABLED", "true")
    calls = []
    def fake(**kwargs):
        calls.append(kwargs)
        return {
            "summary": "Balanced smart-professional outfit",
            "hero_item_id": "shirt-123",
            "strongest_item_id": "shirt-123",
            "weakest_item_id": "shoe-789",
            "recommended_actions": ["Try brown loafers"],
            "confidence": 0.8,
        }, "mock"
    first = visual_intelligence(state=STATE, image_base64="unique", requested=True, vision_call=fake)
    second = visual_intelligence(state=STATE, image_base64="ignored", requested=True, vision_call=fake)
    assert first == second
    assert len(calls) == 1


def _image_bytes(color="navy"):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _visual_state(suffix):
    return normalize_style_state({
        **STATE,
        "board_id": f"board-{suffix}",
        "board_items": [
            {**item, "item_id": f"{item['item_id']}-{suffix}", "image_url": f"https://img/{suffix}/{index}.png"}
            for index, item in enumerate(STATE["board_items"])
        ],
    })


def test_normal_response_is_visually_analyzable_without_response_base64(monkeypatch):
    monkeypatch.setenv("AHVI_BETA_STYLE_VISION_ENABLED", "true")
    state = _visual_state("private")
    calls = []
    result = visual_intelligence(
        state=state,
        visual_items=state["board_items"],
        requested=True,
        image_loader=lambda _url: _image_bytes(),
        vision_call=lambda **kwargs: calls.append(kwargs) or ({
            "summary": "The selected garments form a balanced outfit.",
            "hero_item_id": state["board_items"][0]["item_id"],
        }, "mock"),
    )
    assert result is not None
    assert len(calls) == 1
    assert calls[0]["image_base64"]
    assert not calls[0]["image_base64"].startswith("data:")


def test_changed_board_hash_causes_one_new_visual_call(monkeypatch):
    monkeypatch.setenv("AHVI_BETA_STYLE_VISION_ENABLED", "true")
    calls = []
    def fake(**kwargs):
        calls.append(kwargs)
        return {"summary": "Balanced"}, "mock"
    for suffix in ("hash-a", "hash-b"):
        state = _visual_state(suffix)
        assert visual_intelligence(
            state=state,
            visual_items=state["board_items"],
            requested=True,
            image_loader=lambda _url: _image_bytes(),
            vision_call=fake,
        )
    assert len(calls) == 2


def test_malformed_visual_output_keeps_analysis_optional(monkeypatch):
    monkeypatch.setenv("AHVI_BETA_STYLE_VISION_ENABLED", "true")
    state = _visual_state("malformed")
    assert visual_intelligence(
        state=state,
        visual_items=state["board_items"],
        requested=True,
        image_loader=lambda _url: _image_bytes(),
        vision_call=lambda **_kwargs: ({"hero_item_id": "invented"}, "mock"),
    ) is None


REFINE_STATE = {
    **STATE,
    "board_items": [
        *STATE["board_items"],
        {"item_id": "bag-1", "role": "accessory", "source": "wardrobe", "name": "Tan bag"},
    ],
}
REFINE_CANDIDATES = [
    {"item_id": "shirt-2", "role": "top", "source": "wardrobe", "name": "White shirt"},
    {"item_id": "pants-2", "role": "bottom", "source": "wardrobe", "name": "Navy trousers"},
    {"item_id": "shoe-2", "role": "footwear", "source": "wardrobe", "name": "Brown loafers"},
    {"item_id": "shoe-3", "role": "footwear", "source": "wardrobe", "name": "White sneakers"},
    {"item_id": "bag-2", "role": "accessory", "source": "wardrobe", "name": "Black bag"},
    {"item_id": "asset-top", "role": "top", "source": "style_asset", "name": "Editorial top"},
    {"item_id": "asset-bottom", "role": "bottom", "source": "style_asset", "name": "Editorial trouser"},
    {"item_id": "asset-shoe", "role": "footwear", "source": "style_asset", "name": "Editorial shoe"},
    {"item_id": "asset-bag", "role": "accessory", "source": "style_asset", "name": "Editorial bag"},
]


@pytest.mark.parametrize(
    ("prompt", "kept", "removed"),
    [
        ("Keep this shirt and change everything else.", {"shirt-123"}, {"pants-456", "shoe-789", "bag-1"}),
        ("Change only the shoes.", {"shirt-123", "pants-456", "bag-1"}, {"shoe-789"}),
        ("Keep everything except the bag.", {"shirt-123", "pants-456", "shoe-789"}, {"bag-1"}),
        ("No heels.", {"shirt-123", "pants-456", "bag-1"}, {"shoe-789"}),
        ("No black.", {"shirt-123", "pants-456", "bag-1"}, {"shoe-789"}),
    ],
)
def test_exact_refinement_preserves_and_replaces_by_construction(prompt, kept, removed):
    instructions = interpret_style_followup(prompt, REFINE_STATE)
    result = refine_style_response(
        state=REFINE_STATE,
        instructions=instructions,
        candidate_pool=REFINE_CANDIDATES,
    )
    assert result["success"] is True
    ids = {item["item_id"] for item in result["cards"][0]["items"]}
    assert kept <= ids
    assert not (removed & ids)
    assert set(instructions["preserve_item_ids"]) <= ids


def test_keep_one_item_and_change_only_another_segments_clauses():
    # Regression: a keep-clause and a change-clause in the same sentence must
    # not both claim every mentioned role. Previously "Keep the shirt and
    # change only the shoes" put the shirt in both preserve and replace and
    # failed with conflicting_preserve_and_replace.
    instructions = interpret_style_followup(
        "Keep the shirt and change only the shoes.", REFINE_STATE
    )
    result = refine_style_response(
        state=REFINE_STATE, instructions=instructions, candidate_pool=REFINE_CANDIDATES
    )
    assert result["success"] is True
    ids = {item["item_id"] for item in result["cards"][0]["items"]}
    assert {"shirt-123", "pants-456", "bag-1"} <= ids  # kept, incl. the shirt
    assert "shoe-789" not in ids  # only the footwear changed
    assert any(item["role"] == "footwear" for item in result["cards"][0]["items"])


def test_already_satisfied_exclusion_affirms_board_unchanged():
    # The board has no neon / polka-dot item, so the exclusion is already met:
    # affirm the current board instead of failing with no_replacement_scope.
    instructions = interpret_style_followup("No neon and no polka dots.", REFINE_STATE)
    result = refine_style_response(
        state=REFINE_STATE, instructions=instructions, candidate_pool=REFINE_CANDIDATES
    )
    assert result["success"] is True
    assert result["type"] == "style_refinement_satisfied"
    result_ids = {item["item_id"] for item in result["cards"][0]["items"]}
    assert result_ids == {it["item_id"] for it in REFINE_STATE["board_items"]}


def test_except_absent_item_affirms_board_unchanged():
    # STATE has no accessory; "except the bag" has nothing to remove -> affirm.
    instructions = interpret_style_followup("Keep everything except the bag.", STATE)
    result = refine_style_response(
        state=STATE, instructions=instructions, candidate_pool=REFINE_CANDIDATES
    )
    assert result["success"] is True
    assert result["type"] == "style_refinement_satisfied"
    result_ids = {item["item_id"] for item in result["cards"][0]["items"]}
    assert result_ids == {it["item_id"] for it in STATE["board_items"]}


def test_wardrobe_only_and_inspiration_refinement_use_proven_sources():
    wardrobe = interpret_style_followup("Change only the shoes using my wardrobe.", REFINE_STATE)
    wardrobe_result = refine_style_response(
        state=REFINE_STATE, instructions=wardrobe, candidate_pool=REFINE_CANDIDATES
    )
    changed = next(item for item in wardrobe_result["cards"][0]["items"] if item["role"] == "footwear")
    assert changed["source"] == "wardrobe"

    inspiration = interpret_style_followup("Show a visual inspiration option.", REFINE_STATE)
    inspiration_result = refine_style_response(
        state=REFINE_STATE, instructions=inspiration, candidate_pool=REFINE_CANDIDATES
    )
    assert inspiration_result["success"] is True
    assert all(item["source"] == "style_asset" for item in inspiration_result["cards"][0]["items"])


def test_refinement_rejects_nonexistent_id_conflict_and_no_candidate():
    base = interpret_style_followup("Change only the shoes.", REFINE_STATE)
    missing = {**base, "preserve_item_ids": ["does-not-exist"]}
    conflict = {**base, "preserve_item_ids": ["shoe-789"], "replace_item_ids": ["shoe-789"]}
    for instructions in (missing, conflict):
        result = refine_style_response(
            state=REFINE_STATE, instructions=instructions, candidate_pool=REFINE_CANDIDATES
        )
        assert result["success"] is False
        assert result["constraint_status"]["final_validation_status"] == "failed"
    none = refine_style_response(
        state=REFINE_STATE, instructions=base, candidate_pool=[]
    )
    assert none["success"] is False


def test_one_repair_attempt_uses_second_candidate(monkeypatch):
    import services.beta_style_bridge as bridge
    instructions = interpret_style_followup("Change only the shoes.", REFINE_STATE)
    real_validate = bridge.validate_style_response
    calls = []
    def flaky(response, routed):
        calls.append(response["cards"][0]["items"])
        if len(calls) == 1:
            return {
                "passed_constraints": [],
                "unresolved_constraints": ["occasion_private_wear_guard"],
                "fallback_reason": "constraints_not_fully_satisfied",
                "repair_attempted": False,
                "final_validation_status": "partial",
            }
        return real_validate(response, routed)
    monkeypatch.setattr(bridge, "validate_style_response", flaky)
    result = bridge.refine_style_response(
        state=REFINE_STATE, instructions=instructions, candidate_pool=REFINE_CANDIDATES
    )
    assert len(calls) == 2
    assert result["constraint_status"]["repair_attempted"] is True
    assert result["constraint_status"]["repair_succeeded"] is True
    shoes = [item for item in result["cards"][0]["items"] if item["role"] == "footwear"]
    assert shoes[0]["item_id"] == "shoe-3"


def _prompt_harness():
    bases = [
        "Keep the top. Hate everything else.",
        "no heels n no black pls",
        "This is too serious. Make it fun.",
        "Change only the shoes.",
        "Why does this work visually?",
        "I said no blazer.",
        "Need smth decent for tmro.",
        "Keep everything except the bag.",
        "What is visually weak here?",
        "Make it better.",
    ]
    suffixes = ["", " pls", "!!!", " for office", " but use my clothes", " tomorrow", " quickly", " same vibe", " no black", " thanks"]
    return [f"{base}{suffix}" for base in bases for suffix in suffixes]


def test_offline_unpredictable_prompt_harness_has_100_safe_results():
    prompts = _prompt_harness()
    assert len(prompts) == 100
    allowed = {item["item_id"] for item in STATE["board_items"]}
    for prompt in prompts:
        result = interpret_style_followup(prompt, STATE)
        assert result["action"] in {
            "create_board", "refine_current_board", "explain_current_board",
            "critique_current_board", "identify_wardrobe_gap", "switch_source_mode",
            "answer_style_question", "ask_clarification",
        }
        assert set(result["preserve_item_ids"] + result["replace_item_ids"]) <= allowed
        assert result["needs_clarification"] == (result["action"] == "ask_clarification")


def test_randomized_paraphrase_harness_has_120_safe_results():
    randomizer = random.Random(20260723)
    verbs = ["keep", "retain", "change", "replace", "remove", "explain"]
    targets = ["shirt", "shoes", "bag", "top", "everything else"]
    modifiers = ["no black", "wardrobe only", "for office", "same vibe", "please", "visually"]
    prompts = [
        f"{randomizer.choice(verbs)} {randomizer.choice(targets)} {randomizer.choice(modifiers)}"
        for _ in range(120)
    ]
    allowed = {item["item_id"] for item in STATE["board_items"]}
    for prompt in prompts:
        result = interpret_style_followup(prompt, STATE)
        assert result["action"] in {
            "create_board", "refine_current_board", "explain_current_board",
            "critique_current_board", "identify_wardrobe_gap", "switch_source_mode",
            "answer_style_question", "ask_clarification",
        }
        assert set(result["preserve_item_ids"] + result["replace_item_ids"]) <= allowed


def _chat_client():
    from routers.chat import router
    app = FastAPI()
    @app.middleware("http")
    async def auth(request: Request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_chat_explanation_uses_carried_state_without_generation(monkeypatch):
    import routers.chat as chat
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *_args, **_kwargs: pytest.fail("must not regenerate explanation"),
    )
    monkeypatch.setattr(
        chat.style_reasoning_engine,
        "reason",
        lambda *_args, **_kwargs: pytest.fail("must not call reasoning provider"),
    )
    response = _chat_client().post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "Why does this work visually?"}],
            "style_state": STATE,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "style_explanation"
    assert body["style_state"]["board_id"] == "board-1"
    assert body["understood"]["action"] == "explain_current_board"
    assert body["visual_intelligence"] is None


def test_chat_ambiguous_followup_clarifies_without_generation(monkeypatch):
    import routers.chat as chat
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *_args, **_kwargs: pytest.fail("must not generate ambiguous request"),
    )
    response = _chat_client().post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "Make it better."}],
            "style_state": STATE,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "style_clarification"
    assert body["constraint_status"]["final_validation_status"] == "clarification"


def test_chat_exact_refinement_never_falls_through_to_generator(monkeypatch):
    import routers.chat as chat
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *_args, **_kwargs: pytest.fail("exact refinement must not regenerate"),
    )
    monkeypatch.setattr(
        chat.style_reasoning_engine,
        "reason",
        lambda *_args, **_kwargs: pytest.fail("exact refinement must not use reasoning"),
    )
    response = _chat_client().post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "Keep this shirt and change everything else."}],
            "style_state": REFINE_STATE,
            "wardrobe": REFINE_CANDIDATES,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "style_refinement"
    ids = {item["item_id"] for item in body["cards"][0]["items"]}
    assert "shirt-123" in ids
    assert not ({"pants-456", "shoe-789", "bag-1"} & ids)
    assert body["constraint_status"]["final_validation_status"] == "passed"
