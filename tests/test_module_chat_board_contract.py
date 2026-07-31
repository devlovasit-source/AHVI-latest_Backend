"""/api/module-chat wardrobe boards must carry the durable board contract on
every alias (board_id / revision / source_policy), same as /api/text. This route
does not converge on _beta_style_response, so it applies the gate directly.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat


def _it(name, role):
    return {"id": name, "item_id": name, "name": name, "category": role,
            "role": role, "source": "wardrobe", "image_url": f"https://x/{name}.png"}


def _fake_payload(*a, **k):
    card = {"id": "outfit_card_2", "title": "Polished Office", "occasion": "office",
            "items": [_it("shirt", "top"), _it("trousers", "bottom"), _it("loafers", "footwear")]}
    return {"success": True, "type": "cards", "cards": [dict(card)],
            "style_boards": [dict(card)],
            "data": {"outfits": [dict(card)], "rendered_boards": [dict(card)]},
            "meta": {"occasion": "office"}}


def _client(monkeypatch):
    monkeypatch.setattr(chat, "_demo_style_board_payload", _fake_payload)
    monkeypatch.setattr(chat.style_reasoning_engine, "reason", lambda *a, **k: {"mode": "wardrobe_style"})
    app = FastAPI()

    @app.middleware("http")
    async def _u(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def test_module_chat_office_board_has_contract_on_all_aliases(monkeypatch):
    client = _client(monkeypatch)
    r = client.post("/api/module-chat", json={
        "domain": "style", "module": "style",
        "message": "Create a polished office outfit using only my wardrobe.",
        "context": {"wardrobe": [_it("shirt", "top"), _it("trousers", "bottom"), _it("loafers", "footwear")]},
    })
    b = r.json()
    assert r.status_code == 200
    card = b["cards"][0]
    assert card["board_id"]  # not the transient id=outfit_card_2
    assert card["id"] == "outfit_card_2"  # presentation id preserved, not reused
    assert card["board_id"] != card["id"]
    assert int(card["revision"]) >= 1
    assert card["source_policy"] == "wardrobe"
    assert card["interaction_mode"] == "recommendation"
    assert card["shuffle_available"] is False
    assert card["can_shuffle"] is False
    bid = card["board_id"]
    assert b["style_boards"][0]["board_id"] == bid
    assert b["data"]["outfits"][0]["board_id"] == bid
    assert b["data"]["rendered_boards"][0]["board_id"] == bid
    assert b["style_boards"][0]["shuffle_available"] is False
    assert b["data"]["outfits"][0]["can_shuffle"] is False
