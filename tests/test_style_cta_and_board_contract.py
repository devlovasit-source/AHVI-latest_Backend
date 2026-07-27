"""Default one-tap CTA must generate (not clarify) and every Style card must
carry a durable board contract (board_id / revision / source_policy / occasion)
so the frontend can run the locked Shuffle flow instead of a natural-language
"another look" that drops the occasion. Nothing here touches the network.
"""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat


def _client():
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def _it(name, category, source="wardrobe"):
    return {"id": name.lower().replace(" ", "-"), "item_id": name.lower().replace(" ", "-"),
            "name": name, "category": category, "source": source}


TOP = _it("White Shirt", "shirt")
BOTTOM = _it("Blue Jeans", "jeans")
SHOE = _it("Brown Loafers", "loafers")
CAP = _it("Baseball Cap", "cap")


def _complete_cards():
    return [{"title": "Composed Authority", "items": [TOP, BOTTOM, SHOE]}]


# ── 1. default CTA generates, never clarifies ────────────────────────────────

def test_default_cta_returns_cards_not_clarification(monkeypatch):
    def fake_reason(*args, **kwargs):
        return {"mode": "style_advice"}

    def fake_board(user_id, query_text, request_wardrobe, user_profile=None, **kwargs):
        return {
            "success": True,
            "type": "cards",
            "message": "Here are two looks for today.",
            "message_text": "Here are two looks for today.",
            "response": "Here are two looks for today.",
            "cards": _complete_cards(),
            "style_boards": _complete_cards(),
            "chips": [],
            "data": {"outfits": [{"id": "look-1"}]},
            "meta": {"mode": "style_flow_service_adapter_v1", "occasion": "today"},
        }

    monkeypatch.setattr(chat.style_reasoning_engine, "reason", fake_reason)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_board)
    client = _client()

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "Suggest an outfit for today."}],
            "wardrobe": [TOP, BOTTOM, SHOE],
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body.get("type") != "clarification"
    assert "dressing for" not in json.dumps(body).lower()
    assert body.get("cards")
    # Board contract stamped on the returned card.
    card = body["cards"][0]
    assert card.get("board_id")
    assert int(card.get("revision")) >= 1
    assert card.get("source_policy")
    assert card.get("occasion")


def test_board_payload_uses_resolved_occasion_before_text_fallback(monkeypatch):
    contexts = []

    def fake_flow(**kwargs):
        contexts.append(kwargs["context"])
        return {"cards": [], "style_boards": [], "data": {}, "meta": {}}

    monkeypatch.setattr(chat, "build_style_flow_response", fake_flow)
    monkeypatch.setattr(chat, "_fetch_wardrobe_for_style", lambda *_args: [])
    monkeypatch.setattr(chat, "_ahvi_resolve_effective_user_profile", lambda *_args: {})

    chat._demo_style_board_payload("u", "Another look", [], resolved_occasion="dinner")
    chat._demo_style_board_payload("u", "Another look", [])

    assert contexts[0]["occasion"] == "dinner"
    assert contexts[1]["occasion"] == "today"


# ── 2. board contract on every card (gate-level) ─────────────────────────────

def _gate(cards, query="Suggest an outfit for today.", **meta):
    resp = {"success": True, "cards": list(cards), "style_boards": list(cards),
            "board": "style", "meta": {"occasion": "office", **meta}}
    return chat._apply_style_compliance_gate(resp, query=query, user_id="u", wardrobe=[])


def test_every_card_has_full_board_contract():
    out = _gate(_complete_cards())
    for card in out["cards"]:
        assert card["board_id"]
        assert card["revision"] >= 1
        assert card["source_policy"] == "wardrobe"  # all items are wardrobe-sourced
        assert card["occasion"] == "office"
    assert out["board_ids"]


def test_board_id_is_stable_across_serialization():
    a = _gate(_complete_cards())["cards"][0]["board_id"]
    b = _gate(_complete_cards())["cards"][0]["board_id"]
    assert a == b


def test_mixed_source_board_policy_is_mixed():
    catalog_shoe = _it("Catalog Loafers", "loafers", source="catalog")
    out = _gate([{"title": "L", "items": [TOP, BOTTOM, catalog_shoe]}])
    assert out["cards"][0]["source_policy"] == "mixed"


# ── 3. alternative-board occasion + source preservation ─────────────────────

def test_alternative_board_preserves_office_occasion_and_source():
    # An "another look" alternative arrives with the ORIGINAL office context in
    # meta; the gate must stamp office + wardrobe, never downgrade to daily.
    out = _gate(
        _complete_cards(),
        query="Show me another look for Composed Authority",
        occasion="office",
    )
    card = out["cards"][0]
    assert card["occasion"] == "office"
    assert card["source_policy"] == "wardrobe"
    assert out["meta"]["source_policy"] in {"wardrobe", ""}  # canonical stays honest


def test_incomplete_alternative_is_rejected_for_cta():
    # A bottom+footwear+accessory alternative for the default CTA is not a
    # complete outfit -> typed failure, not a stamped incomplete board.
    out = _gate([{"title": "Bad", "items": [BOTTOM, SHOE, CAP]}],
                query="Suggest an outfit for today.")
    assert out.get("type") == "no_complete_outfit"
    assert out["cards"] == []


def test_wardrobe_source_policy_preserved_on_contract():
    out = _gate(_complete_cards())
    assert all(c["source_policy"] == "wardrobe" for c in out["cards"])
    assert all(i.get("source") == "wardrobe" for c in out["cards"] for i in c["items"])


# ── contract reaches EVERY alias Flutter consumes ───────────────────────────

def _gate_aliases(cards, query="Suggest an outfit for today.", **meta):
    resp = {
        "success": True,
        "cards": list(cards),
        "style_boards": [],
        "rendered_boards": [],
        "data": {"outfits": [], "rendered_boards": []},
        "board": "style",
        "meta": {"occasion": "office", **meta},
    }
    return chat._apply_style_compliance_gate(resp, query=query, user_id="u", wardrobe=[])


def _board_ids(coll):
    return [b.get("board_id") for b in (coll or [])]


def test_contract_present_in_all_board_aliases_with_same_id():
    out = _gate_aliases(_complete_cards())
    bid = out["cards"][0]["board_id"]
    assert bid
    assert _board_ids(out["style_boards"]) == [bid]
    assert _board_ids(out["rendered_boards"]) == [bid]
    assert _board_ids(out["data"]["outfits"]) == [bid]
    assert _board_ids(out["data"]["rendered_boards"]) == [bid]


def test_every_item_has_a_usable_snake_case_position():
    out = _gate_aliases(_complete_cards())
    for coll in (out["cards"], out["style_boards"], out["rendered_boards"]):
        for board in coll:
            for item in board["items"]:
                pos = item["position"]
                for key in ("x", "y", "width", "height"):
                    assert pos.get(key) is not None


def test_flutter_snake_case_contract_keys_present():
    card = _gate_aliases(_complete_cards())["cards"][0]
    for key in ("board_id", "revision", "source_policy", "occasion"):
        assert key in card
    assert isinstance(card["revision"], int) and card["revision"] >= 1


# ── alternative-look regeneration hits the completeness gate ─────────────────

ALT = "Show me another look for Understated Ease"


def test_alternative_look_is_a_generate_board_request():
    assert chat._is_alternative_look_request(ALT) is True
    assert chat._is_alternative_look_request("shuffle this look") is True
    assert chat._is_generate_style_board_request(ALT) is True
    # A specific styled request is not an alternative-look regeneration.
    assert chat._is_alternative_look_request("Create a dinner outfit with a dress") is False


def test_alternative_look_cannot_return_bottom_footwear_accessory():
    out = _gate_aliases([{"title": "Alt", "items": [BOTTOM, SHOE, CAP]}], query=ALT)
    assert out.get("type") == "no_complete_outfit"
    assert out["cards"] == []
    assert out["style_boards"] == []


def test_alternative_look_preserves_occasion_and_source_policy():
    out = _gate_aliases(_complete_cards(), query=ALT, occasion="office")
    card = out["cards"][0]
    assert card["occasion"] == "office"
    assert card["source_policy"] == "wardrobe"


def test_beta_refinement_without_query_stamps_every_board_alias():
    response = {
        "success": True,
        "type": "style_refinement",
        "cards": _complete_cards(),
        "style_boards": [],
        "rendered_boards": [],
        "data": {"outfits": [], "rendered_boards": []},
    }
    instructions = {
        "action": "refine_current_board",
        "occasion": "office",
        "source_mode": "wardrobe_only",
        "preserve_item_ids": [],
        "replace_roles": [],
        "excluded_terms": [],
        "confidence": 1,
    }
    out = chat._beta_style_response(
        response, previous_state={}, instructions=instructions
    )
    fields = ("board_id", "revision", "source_policy", "occasion")
    aliases = (
        out["cards"], out["style_boards"], out["rendered_boards"],
        out["data"]["outfits"], out["data"]["rendered_boards"],
    )
    expected = {field: out["cards"][0][field] for field in fields}
    assert expected["revision"] >= 1
    assert expected["occasion"] == "office"
    assert all(
        {field: boards[0][field] for field in fields} == expected
        for boards in aliases
    )


def test_beta_explanation_without_query_is_not_converted_to_board_aliases():
    response = {
        "success": True,
        "type": "style_explanation",
        "cards": _complete_cards(),
        "style_boards": [],
        "data": {"outfits": [], "rendered_boards": []},
    }
    out = chat._beta_style_response(
        response,
        previous_state={},
        instructions={"action": "explain_current_board", "occasion": "office"},
    )
    assert out["style_boards"] == []
    assert out["data"]["outfits"] == []


def test_alternative_boards_are_independently_complete():
    dress = _it("Silk Wrap Dress", "dress")
    heels = _it("Black Heels", "heels")
    # Good is a one-piece (donates no top); Bad is missing a top with none
    # available -> Bad drops, Good stays. Proves per-card validation.
    good = {"title": "Good", "items": [dress, heels]}
    bad = {"title": "Bad", "items": [BOTTOM, SHOE, CAP]}
    out = _gate_aliases([good, bad], query=ALT)
    from brain.engines.outfit_quality_guard import is_complete_board
    from services.style_explicit_roles import board_explicit_roles
    for board in out["cards"]:
        roles = set(board_explicit_roles(board["items"]))
        assert "footwear" in roles and ("dress" in roles or {"top", "bottom"} <= roles)
    assert "Bad" not in [b["title"] for b in out["cards"]]
