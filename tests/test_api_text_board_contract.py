"""Deployed-shape /api/text contract test.

The wardrobe adapter enforces roles and sets meta.style_compliance_gated=True but
never stamps the durable board contract; the gate then early-returned on that
flag and shipped cards with id=outfit_card_N and no board_id (the Samsung bug).
This exercises that exact path (adapter-flagged response) end to end and proves
every alias now carries the contract. Only the LLM boundary is mocked.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from brain import outfit_pipeline
from routers import chat


def _it(name, role):
    return {"id": name, "item_id": name, "name": name, "category": role,
            "role": role, "source": "wardrobe", "image_url": f"https://x/{name}.png"}


_WARDROBE = [_it("shirt", "top"), _it("trousers", "bottom"), _it("loafers", "footwear")]


@pytest.fixture(autouse=True)
def _isolated_outfit_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        outfit_pipeline,
        "_MEMORY_FILE",
        str(tmp_path / "outfit_memory.json"),
    )


def _adapter_shaped(*a, **k):
    # Shape the wardrobe adapter produces: cards with a transient id, no board
    # contract, and meta.style_compliance_gated set (roles already enforced).
    card = {"id": "outfit_card_1", "title": "Polished Office", "occasion": "office",
            "items": [_it("shirt", "top"), _it("trousers", "bottom"), _it("loafers", "footwear")]}
    return {"success": True, "type": "cards", "message": "Office look.",
            "cards": [dict(card)], "style_boards": [dict(card)],
            "data": {"outfits": [dict(card)], "rendered_boards": [dict(card)]},
            "meta": {"mode": "style_flow_service_adapter_v1", "occasion": "office",
                     "style_compliance_gated": True},
            "board_ids": ""}


def _client(monkeypatch):
    monkeypatch.setattr(chat.ahvi_orchestrator, "run", _adapter_shaped)
    app = FastAPI()

    @app.middleware("http")
    async def _u(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def _assert_contract(body):
    card = body["cards"][0]
    assert card["board_id"]
    assert card["board_id"] not in ("outfit_card_1", "outfit_card_2")
    assert card["id"] == "outfit_card_1"  # transient presentation id preserved
    assert int(card["revision"]) >= 1
    assert card["source_policy"] == "wardrobe"
    assert card["interaction_mode"] == "recommendation"
    assert card["shuffle_available"] is False
    assert card["can_shuffle"] is False
    bid = card["board_id"]
    assert body["style_boards"][0]["board_id"] == bid
    assert body["data"]["outfits"][0]["board_id"] == bid
    assert body["data"]["rendered_boards"][0]["board_id"] == bid
    assert body["style_boards"][0]["shuffle_available"] is False
    assert body["data"]["outfits"][0]["can_shuffle"] is False


def test_api_text_office_wardrobe_board_has_contract_on_all_aliases(monkeypatch):
    client = _client(monkeypatch)
    r = client.post("/api/text", json={
        "module_context": "style",
        "include_base64": False,
        "messages": [{"role": "user", "content": "Create a polished office outfit using only my wardrobe."}],
        "wardrobe": _WARDROBE,
    })
    assert r.status_code == 200
    _assert_contract(r.json())


def test_api_text_wardrobe_retry_shape_also_stamped(monkeypatch):
    # The requires_wardrobe retry resends with the fetched wardrobe + use_wardrobe.
    client = _client(monkeypatch)
    r = client.post("/api/text", json={
        "module_context": "style",
        "include_base64": False,
        "style_action": "use_wardrobe",
        "messages": [{"role": "user", "content": "Create a polished office outfit using only my wardrobe."}],
        "wardrobe": _WARDROBE,
    })
    assert r.status_code == 200
    _assert_contract(r.json())
