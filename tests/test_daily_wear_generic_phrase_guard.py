"""Daily Wear (RC3): generic wardrobe phrases must not become named-garment
mentions.

Physical evidence: the Daily Wear screen sends the synthetic prompt "Build
my wardrobe-first looks for today" to /api/module-chat. Cloud Run logs
showed style.owned_item.not_found mentions=['wardrobe-first looks']
wardrobe_count=100 -- the owned-item-mention resolver (services.
style_item_contract._extract_my_phrases) captured "wardrobe-first looks" as
if it were a specific garment name the user was asking for, failed to
resolve it (no such item), and diverted the whole request into the
fixed-item-failure path instead of building a wardrobe board -- despite the
user genuinely owning 100 items.

Root cause: _MY_PHRASE_NON_ITEM_PHRASES only denylisted the bare word
"wardrobe", not "wardrobe" compounded with a following generic noun
("wardrobe-first looks", "wardrobe items", "wardrobe picks"). Fixed in
_extract_my_phrases to also exempt any "my <phrase>" capture whose
normalized form starts with "wardrobe ".
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat
from services.style_item_contract import _extract_my_phrases, resolve_owned_item_mentions
import services.llm_service as llm_service
import services.ai_gateway as ai_gateway
import services.semantic_intent_resolver as semantic_intent_resolver
import brain.intent_engine as intent_engine


@pytest.fixture(autouse=True)
def _fast_fail_llm(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("llm disabled in test")

    monkeypatch.setattr(llm_service, "generate_text", _raise)
    monkeypatch.setattr(ai_gateway, "generate_text", _raise)
    monkeypatch.setattr(semantic_intent_resolver, "_generate_text", _raise)
    monkeypatch.setattr(intent_engine, "generate_text", _raise)


GENERIC_WARDROBE_PHRASES = [
    "Build my wardrobe-first looks for today",
    "Use my wardrobe to make me look taller",
    "Show me looks from my wardrobe",
    "Show me today's outfit from my wardrobe",
]

EXPLICIT_ITEM_QUERIES = [
    "Create an outfit using my Black Trousers",
    "Create an outfit using my Light Green Polo Shirt",
    "Create an outfit using my White Shirt",
]


@pytest.mark.parametrize("phrase", GENERIC_WARDROBE_PHRASES)
def test_generic_wardrobe_phrase_produces_no_mention(phrase):
    assert _extract_my_phrases(phrase) == [], (
        f"generic wardrobe phrasing was captured as a named-item mention: {phrase!r}"
    )


@pytest.mark.parametrize("query", EXPLICIT_ITEM_QUERIES)
def test_explicit_item_name_still_produces_a_mention(query):
    assert _extract_my_phrases(query) != [], (
        f"an explicit garment name stopped being captured as a mention: {query!r}"
    )


def _wardrobe(*names_and_roles):
    return [
        {
            "id": name.lower().replace(" ", "-"), "item_id": name.lower().replace(" ", "-"),
            "name": name, "category": role, "role": role, "source": "wardrobe",
            "image_url": f"https://x/{name}.png", "masked_url": f"https://x/{name}-masked.png",
        }
        for name, role in names_and_roles
    ]


def test_generic_daily_wear_prompt_resolves_no_owned_items_but_stays_unresolved_free():
    wardrobe = _wardrobe(
        ("Black Trousers", "bottom"), ("Light Green Polo Shirt", "top"),
        ("White Sneakers", "footwear"),
    )
    result = resolve_owned_item_mentions("Build my wardrobe-first looks for today", wardrobe)
    assert result == {"resolved": [], "ambiguous": [], "unresolved": []}, (
        f"a generic wardrobe request must not produce any fixed-item mentions "
        f"(resolved, ambiguous, or unresolved). result={result}"
    )


class _FakeAppwriteProxy:
    def list_documents(self, collection, user_id=None, limit=100, offset=0):
        docs = [
            {"id": "black-trousers", "item_id": "black-trousers", "name": "Black Trousers",
             "category": "bottom", "image_url": "https://x/bt.png", "masked_url": "https://x/bt-m.png"},
            {"id": "polo", "item_id": "polo", "name": "Light Green Polo Shirt",
             "category": "top", "image_url": "https://x/p.png", "masked_url": "https://x/p-m.png"},
            {"id": "sneakers", "item_id": "sneakers", "name": "White Sneakers",
             "category": "footwear", "image_url": "https://x/s.png", "masked_url": "https://x/s-m.png"},
            {"id": "jeans", "item_id": "jeans", "name": "Blue Jeans",
             "category": "bottom", "image_url": "https://x/j.png", "masked_url": "https://x/j-m.png"},
        ]
        return [] if offset else docs


def _client_with_user():
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def test_daily_wear_synthetic_prompt_does_not_hit_unknown_item_source(monkeypatch):
    """End-to-end: with a populated wardrobe (>=4 items, matching the
    physical evidence's wardrobe_count=100 case), the Daily Wear synthetic
    prompt must not be diverted into the fixed-item-failure path at all --
    response_type must never be unknown_item_source for this message."""
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client_with_user()

    r = client.post(
        "/api/text",
        json={
            "module_context": "wardrobe",
            "include_base64": False,
            "messages": [{"role": "user", "content": "Build my wardrobe-first looks for today"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("reason") != "unknown_item_source", (
        f"Daily Wear's generic prompt was misparsed as a named-item mention again. body={body}"
    )
