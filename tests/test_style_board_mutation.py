from __future__ import annotations

import copy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services import style_board_shuffle_service as shuffle_service
from services.semantic_intent_resolver import validate_semantic_decision
from services.style_board_mutation_service import handle_board_operation
from services.style_board_state_store import InMemoryBoardStateStore


@pytest.fixture(autouse=True)
def _state_store():
    shuffle_service.set_state_store(InMemoryBoardStateStore())
    yield
    shuffle_service.set_state_store(None)


def _item(item_id, name, role, *, locked=False, source="wardrobe", **extra):
    return {
        "item_id": item_id,
        "id": item_id,
        "name": name,
        "category": role,
        "role": role,
        "slot": role,
        "source": source,
        "image_url": f"https://img/{item_id}.png",
        **({"locked": True} if locked else {}),
        **extra,
    }


def _decision(
    operation="modify",
    *,
    replace_roles=(),
    preserve_roles=(),
    remove_roles=(),
    constraints=(),
    adjustments=None,
    alternative_scope="default",
    requires_clarification=False,
    resolved_context=None,
    referent=None,
):
    intent = {
        "modify": "modify_current_look",
        "generate_alternative": "generate_alternative",
        "explain_current_look": "explain_current_look",
    }[operation]
    return {
        "domain": "style",
        "intent": intent,
        "action": intent,
        "response_mode": "text_only" if operation == "explain_current_look" else "wardrobe_recommendation",
        "confidence": 0.96,
        "requires_clarification": requires_clarification,
        "resolved_context": resolved_context or {},
        "constraints": {"required": [], "avoid": []},
        "referent": referent,
        "reason_codes": [],
        "missing_information": [],
        "operation": {
            "type": operation,
            "replace_roles": list(replace_roles),
            "preserve_roles": list(preserve_roles),
            "remove_roles": list(remove_roles),
            "constraints": [dict(item) for item in constraints],
            "style_adjustments": dict(adjustments or {}),
            "alternative_scope": alternative_scope,
            "explanation_target": None,
        },
    }


def _payload(message="change", *, revision=1, items=None, candidates=None, state_extra=None):
    board_items = items or [
        _item("top-1", "White shirt", "top"),
        _item("outer-1", "Navy jacket", "outerwear", locked=True),
        _item("bottom-1", "Blue jeans", "bottom"),
        _item("shoe-1", "White sneakers", "footwear"),
    ]
    state = {
        "board_id": "board-1",
        "revision": revision,
        "source_mode": "wardrobe",
        "occasion": "client meeting",
        "date_context": "tomorrow",
        "activity": "badminton",
        "activity_type": "court_sport",
        "board_items": copy.deepcopy(board_items),
        "locked_item_ids": ["outer-1"],
        **(state_extra or {}),
    }
    return {
        "message": message,
        "style_state": state,
        "wardrobe": candidates or [
            _item("top-2", "Black tee", "top"),
            _item("outer-2", "Grey coat", "outerwear"),
            _item("bottom-2", "Grey trousers", "bottom"),
            _item("shoe-2", "Brown loafers", "footwear"),
        ],
    }


def _run(payload, decision, user_id="user-1"):
    return handle_board_operation(payload, user_id=user_id, semantic_decision=decision)


def test_semantic_operation_schema_rejects_concrete_id_control():
    decision = _decision(replace_roles=["footwear"])
    assert validate_semantic_decision(decision)["operation"]["replace_roles"] == ["footwear"]
    injected = copy.deepcopy(decision)
    injected["operation"]["target_item_ids"] = ["malicious-new-id"]
    assert validate_semantic_decision(injected) is None


def test_change_shoes_is_minimal_delta():
    before = _payload()
    result = _run(before, _decision(replace_roles=["footwear"]))

    assert result["success"] is True
    assert result["data"]["changed_item_ids"] == ["shoe-2"]
    output = {item["item_id"]: item for item in result["cards"][0]["items"]}
    assert output["top-1"]
    assert output["outer-1"]
    assert output["bottom-1"]
    assert "shoe-1" not in output


def test_multi_role_replacement_changes_only_top_and_footwear():
    result = _run(
        _payload(),
        _decision(replace_roles=["top", "footwear"]),
    )

    assert result["success"] is True
    assert set(result["data"]["changed_item_ids"]) == {"top-2", "shoe-2"}
    ids = {item["item_id"] for item in result["cards"][0]["items"]}
    assert {"outer-1", "bottom-1"} <= ids


def test_preserve_role_keeps_exact_item_identity():
    result = _run(
        _payload(),
        _decision(preserve_roles=["outerwear"]),
    )

    assert result["success"] is True
    assert "outer-1" in {item["item_id"] for item in result["cards"][0]["items"]}
    assert "outer-2" not in {item["item_id"] for item in result["cards"][0]["items"]}


def test_alternative_preserves_hard_context_constraints_and_changes_board():
    result = _run(
        _payload(
            state_extra={"active_constraints": {"exclude": ["black"]}},
            candidates=[
                _item("top-2", "White tee", "top"),
                _item("outer-2", "Grey coat", "outerwear"),
                _item("bottom-2", "Grey trousers", "bottom"),
                _item("shoe-2", "Brown loafers", "footwear"),
            ],
        ),
        _decision(
            "generate_alternative",
            constraints=[{"dimension": "color", "operator": "avoid", "value": "black"}],
            resolved_context={
                "date_context": "tomorrow",
                "activity": "badminton",
                "activity_type": "court_sport",
                "negative_constraints": ["black"],
            },
        ),
    )

    assert result["success"] is True
    state = result["style_state"]
    assert state["date_context"] == "tomorrow"
    assert state["activity"] == "badminton"
    assert state["activity_type"] == "court_sport"
    assert "black" in state["active_constraints"]["exclude"]
    assert result["data"]["changed_item_ids"]
    assert result["revision"] == 2


def test_broad_alternative_is_structured_and_lineage_is_deterministic():
    decision = _decision("generate_alternative", alternative_scope="broad")
    first = _run(_payload(), decision)
    shuffle_service.set_state_store(InMemoryBoardStateStore())
    second = _run(_payload(), decision)

    assert first["success"] is True and second["success"] is True
    assert first["lineage"] == second["lineage"]
    assert first["lineage"]["parent_revision"] == 1


def test_explanation_is_text_only_and_does_not_create_revision():
    result = _run(_payload("why these shoes"), _decision("explain_current_look"))

    assert result["success"] is True
    assert result["response_mode"] == "text_only"
    assert result["revision"] == 1
    assert result["board_ids"] == "board-1"
    assert result["data"]["explanation"]
    assert shuffle_service.get_board_state("board-1") is None


def test_ambiguous_structured_operation_clarifies_without_mutation():
    result = _run(
        _payload("change it"),
        _decision(requires_clarification=True),
    )

    assert result["success"] is True
    assert result["response_mode"] == "clarification"
    assert result["meta"]["board_mutated"] is False
    assert shuffle_service.get_board_state("board-1") is None


def test_authoritative_durable_lock_survives_client_lock_omission():
    authoritative_items = [
        _item("top-1", "White shirt", "top"),
        _item("jacket-123", "Navy jacket", "outerwear", locked=True),
        _item("bottom-1", "Blue jeans", "bottom"),
        _item("shoe-1", "Sneakers", "footwear"),
    ]
    store = shuffle_service._get_store()
    store.create_revision(
        user_id="user-1",
        board_id="board-1",
        revision=1,
        payload={
            "scenario": "conversational_style",
            "interaction_mode": "conversational",
            "source_policy": "wardrobe",
            "items": authoritative_items,
            "locked_item_ids": ["jacket-123"],
            "protected_item_ids": ["jacket-123"],
        },
    )
    client_state = {
        "board_id": "board-1",
        "revision": 1,
        "source_mode": "wardrobe",
        "board_items": [
            _item("top-1", "White shirt", "top"),
            _item("jacket-123", "Navy jacket", "outerwear"),
            _item("bottom-1", "Blue jeans", "bottom"),
            _item("shoe-1", "Sneakers", "footwear"),
        ],
    }
    result = _run(
        {"style_state": client_state, "wardrobe": [_item("jacket-2", "Grey coat", "outerwear"), _item("shoe-2", "Loafers", "footwear")]},
        _decision(replace_roles=["outerwear", "footwear"]),
    )

    assert result["success"] is True
    assert "jacket-123" in {item["item_id"] for item in result["cards"][0]["items"]}


def test_incomplete_carried_board_is_not_initialized():
    items = [
        _item("top-1", "White shirt", "top"),
        _item("shoe-1", "White sneakers", "footwear"),
    ]

    result = _run(
        _payload(
            items=items,
            candidates=[_item("top-2", "Black tee", "top")],
            state_extra={"locked_item_ids": []},
        ),
        _decision(replace_roles=["top"]),
    )

    assert result["success"] is False
    assert result["error"]["code"] == "INCOMPLETE_BOARD"
    assert shuffle_service.get_board_state("board-1") is None


def test_incomplete_mutation_result_does_not_create_next_revision():
    incomplete = [
        _item("top-1", "White shirt", "top"),
        _item("shoe-1", "White sneakers", "footwear"),
    ]
    store = shuffle_service._get_store()
    store.create_revision(
        user_id="user-1",
        board_id="board-1",
        revision=1,
        payload={
            "scenario": "conversational_style",
            "interaction_mode": "conversational",
            "source_policy": "wardrobe",
            "items": incomplete,
        },
    )

    result = _run(
        {
            "style_state": {
                "board_id": "board-1",
                "revision": 1,
                "board_items": incomplete,
            },
            "wardrobe": [_item("top-2", "Black tee", "top")],
        },
        _decision(replace_roles=["top"]),
    )

    assert result["success"] is False
    assert result["error"]["code"] == "INCOMPLETE_BOARD"
    assert store.get_latest("board-1")["revision"] == 1
    assert store.get_revision("board-1", 2) is None


def test_dress_and_footwear_carried_board_can_mutate_with_revision_semantics():
    items = [
        _item("dress-1", "Midnight Dress", "dress"),
        _item("shoe-1", "Black Heels", "footwear"),
    ]

    result = _run(
        _payload(
            items=items,
            candidates=[_item("shoe-2", "Silver Heels", "footwear")],
            state_extra={"locked_item_ids": []},
        ),
        _decision(replace_roles=["footwear"]),
    )

    assert result["success"] is True
    assert result["board_ids"] == "board-1"
    assert result["revision"] == 2
    assert result["lineage"]["parent_revision"] == 1


@pytest.mark.parametrize("mode", ["style_this", "build_outfit"])
def test_specialized_modes_are_rejected(mode):
    store = shuffle_service._get_store()
    item = _item("anchor-1", "Anchor", "top", locked=True)
    store.create_revision(
        user_id="user-1", board_id="board-1", revision=1,
        payload={
            "scenario": mode,
            "interaction_mode": mode,
            "source_policy": "wardrobe",
            "items": [item],
            "locked_item_ids": ["anchor-1"],
        },
    )
    result = _run(
        {"style_state": {"board_id": "board-1", "revision": 1, "board_items": [item]}},
        _decision(replace_roles=["top"]),
    )
    assert result["success"] is False
    assert result["error"]["code"] == "SPECIALIZED_BOARD_MODE"


def test_stale_durable_revision_is_rejected_without_offline_fallback():
    store = shuffle_service._get_store()
    item = _item("shoe-1", "Sneakers", "footwear")
    store.create_revision(
        user_id="user-1", board_id="board-1", revision=4,
        payload={"scenario": "conversational_style", "interaction_mode": "conversational", "source_policy": "wardrobe", "items": [item]},
    )
    result = _run(
        {"style_state": {"board_id": "board-1", "revision": 3, "board_items": [item]}, "wardrobe": [_item("shoe-2", "Loafers", "footwear")]},
        _decision(replace_roles=["footwear"]),
    )
    assert result["success"] is False
    assert result["error"]["code"] == "STALE_BOARD_REVISION"
    assert store.get_latest("board-1")["revision"] == 4


def test_unsafe_raw_candidate_is_rejected():
    result = _run(
        _payload(candidates=[_item("shoe-raw", "Raw shoe", "footwear", raw_image_url="https://raw/x.png", image_url="")]),
        _decision(replace_roles=["footwear"]),
    )
    assert result["success"] is False
    assert result["error"]["code"] == "NO_VALID_REPLACEMENT"


def test_module_chat_and_text_use_the_same_structured_semantic_executor(monkeypatch):
    semantic = _decision(replace_roles=["footwear"])
    monkeypatch.setattr(
        "services.semantic_intent_resolver._generate_text",
        lambda *args, **kwargs: json.dumps(semantic),
    )
    app = FastAPI()

    @app.middleware("http")
    async def _user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    from routers import chat

    app.include_router(chat.router, prefix="/api")
    module_response = TestClient(app).post(
        "/api/module-chat",
        json={"domain": "style", "message": "change the shoes", "style_state": _payload()["style_state"], "context": {"wardrobe": _payload()["wardrobe"]}},
    )
    assert module_response.status_code == 200
    assert module_response.json()["board_operation"] == "modify_current_look"

    shuffle_service.set_state_store(InMemoryBoardStateStore())
    text_response = TestClient(app).post(
        "/api/text",
        json={"messages": [{"role": "user", "content": "change the shoes"}], "style_state": _payload()["style_state"], "wardrobe": _payload()["wardrobe"]},
    )
    assert text_response.status_code == 200
    assert text_response.json()["board_operation"] == "modify_current_look"


@pytest.mark.parametrize(
    "phrase",
    [
        "replace the shoes with white sneakers",
        "change the shoes to white sneakers",
        "swap the shoes for white sneakers",
        "give me white sneakers instead",
        "keep everything but change the shoes to white sneakers",
    ],
)
def test_natural_language_white_sneaker_phrases_mutate_only_footwear(monkeypatch, phrase):
    before = _payload(
        items=[
            _item("top-1", "White shirt", "top"),
            _item("outer-1", "Navy jacket", "outerwear", locked=True),
            _item("bottom-1", "Blue jeans", "bottom"),
            _item("shoe-1", "Brown loafers", "footwear"),
        ],
        candidates=[
            _item(
                "shoe-white-2",
                "White sneakers",
                "footwear",
                color="white",
                footwear_type="sneaker",
            )
        ]
    )
    decision = _decision(
        replace_roles=["footwear"],
        constraints=[
            {"dimension": "color", "operator": "require", "value": "white"},
            {"dimension": "footwear_type", "operator": "require", "value": "sneaker"},
        ],
    )
    monkeypatch.setattr(
        "services.semantic_intent_resolver._generate_text",
        lambda *args, **kwargs: json.dumps(decision),
    )
    app = FastAPI()

    @app.middleware("http")
    async def _user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    from routers import chat

    app.include_router(chat.router, prefix="/api")
    response = TestClient(app).post(
        "/api/text",
        json={
            "messages": [{"role": "user", "content": phrase}],
            "style_state": before["style_state"],
            "wardrobe": before["wardrobe"],
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["board_operation"] == "modify_current_look"
    assert body["revision"] == 2
    assert body["data"]["changed_item_ids"] == ["shoe-white-2"]
    output = {item["item_id"]: item for item in body["cards"][0]["items"]}
    assert {"top-1", "outer-1", "bottom-1"} <= set(output)
    assert "shoe-1" not in output
    replacement = output["shoe-white-2"]
    assert replacement["role"] == "footwear"
    assert "white" in replacement["name"].lower()
    assert "sneaker" in replacement["name"].lower()


def test_colour_and_type_constraints_select_the_requested_footwear():
    black = _run(
        _payload(candidates=[_item("shoe-black-2", "Black sneakers", "footwear", color="black")]),
        _decision(
            replace_roles=["footwear"],
            constraints=[
                {"dimension": "color", "operator": "require", "value": "black"},
                {"dimension": "footwear_type", "operator": "require", "value": "sneaker"},
            ],
        ),
    )
    assert black["cards"][0]["items"][-1]["name"] == "Black sneakers"

    shuffle_service.set_state_store(InMemoryBoardStateStore())
    sneaker = _run(
        _payload(
            items=[
                _item("top-1", "White shirt", "top"),
                _item("outer-1", "Navy jacket", "outerwear"),
                _item("bottom-1", "Blue jeans", "bottom"),
                _item("shoe-1", "Brown loafers", "footwear"),
            ],
            state_extra={"locked_item_ids": []},
            candidates=[_item("shoe-sneaker-2", "White sneakers", "footwear")],
        ),
        _decision(
            replace_roles=["footwear"],
            constraints=[{"dimension": "footwear_type", "operator": "require", "value": "sneaker"}],
        ),
    )
    ids = {item["item_id"] for item in sneaker["cards"][0]["items"]}
    assert "shoe-sneaker-2" in ids and "shoe-1" not in ids


def test_no_black_constraint_and_other_role_mutation_preserve_unrelated_items():
    footwear = _run(
        _payload(
            candidates=[
                _item("shoe-black-2", "Black sneakers", "footwear", color="black"),
                _item("shoe-white-2", "White sneakers", "footwear", color="white"),
            ]
        ),
        _decision(
            replace_roles=["footwear"],
            constraints=[
                {"dimension": "color", "operator": "require", "value": "white"},
                {"dimension": "style", "operator": "avoid", "value": "black"},
            ],
        ),
    )
    assert footwear["data"]["changed_item_ids"] == ["shoe-white-2"]

    shuffle_service.set_state_store(InMemoryBoardStateStore())
    blazer = _run(
        _payload(
            items=[
                _item("top-1", "White shirt", "top"),
                _item("outer-1", "Grey jacket", "outerwear"),
                _item("bottom-1", "Blue jeans", "bottom"),
                _item("shoe-1", "Brown loafers", "footwear"),
            ],
            state_extra={"locked_item_ids": []},
            candidates=[_item("outer-2", "Navy blazer", "outerwear", color="navy")],
        ),
        _decision(
            replace_roles=["outerwear"],
            constraints=[{"dimension": "color", "operator": "require", "value": "navy"}],
        ),
    )
    blazer_ids = {item["item_id"] for item in blazer["cards"][0]["items"]}
    assert {"top-1", "bottom-1", "shoe-1", "outer-2"} <= blazer_ids
    assert "outer-1" not in blazer_ids


def test_colour_and_type_requirements_do_not_silently_fallback():
    result = _run(
        _payload(
            candidates=[
                _item("shoe-black-2", "Black sneakers", "footwear", color="black"),
                _item("shoe-white-2", "White loafers", "footwear", color="white"),
            ]
        ),
        _decision(
            replace_roles=["footwear"],
            constraints=[
                {"dimension": "color", "operator": "require", "value": "white"},
                {"dimension": "footwear_type", "operator": "require", "value": "sneaker"},
            ],
        ),
    )
    assert result["success"] is False
    assert result["error"]["code"] == "NO_VALID_REPLACEMENT"


def test_follow_up_mutates_the_new_revision_not_the_historical_board():
    first = _run(
        _payload(candidates=[_item("shoe-white-2", "White sneakers", "footwear")]),
        _decision(replace_roles=["footwear"]),
    )
    follow_up_decision = _decision(
        replace_roles=["footwear"],
        constraints=[{"dimension": "style", "operator": "require", "value": "minimal"}],
        referent={
            "type": "context",
            "text": "them",
            "resolved_to": "footwear",
            "confidence": 0.9,
        },
    )
    assert validate_semantic_decision(follow_up_decision)["referent"]["resolved_to"] == "footwear"
    second = _run(
        {
            "style_state": first["style_state"],
            "wardrobe": [_item("shoe-minimal-3", "Minimal white sneakers", "footwear")],
        },
        follow_up_decision,
    )
    assert first["revision"] == 2
    assert second["revision"] == 3
    assert second["lineage"]["parent_revision"] == 2
    assert second["data"]["changed_item_ids"] == ["shoe-minimal-3"]
