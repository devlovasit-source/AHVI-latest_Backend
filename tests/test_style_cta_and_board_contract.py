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
