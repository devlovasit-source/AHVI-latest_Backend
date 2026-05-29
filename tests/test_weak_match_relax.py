"""Test the relaxed weak-match safety gate.

Behavior contract:
- When the style flow generates cards but the best occasion score is below
  the threshold, cards are NOT suppressed. They are returned with
  weak_match=True, badge="CLOSEST OPTION", and confidence_note set.
- When the style flow generates zero cards, the existing missing/weak
  wardrobe responses are preserved.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import services.style_flow_service as sfs


def _card(idx: int, score: float, title: str) -> Dict[str, Any]:
    return {
        "id": f"card_{idx}",
        "title": title,
        "items": [
            {
                "id": f"top_{idx}",
                "name": "Test Top",
                "category": "Tops",
                "image_url": "https://example/top.png",
                "role": "top",
            },
            {
                "id": f"bot_{idx}",
                "name": "Test Trouser",
                "category": "Bottoms",
                "image_url": "https://example/bot.png",
                "role": "bottom",
            },
            {
                "id": f"foot_{idx}",
                "name": "Test Loafers",
                "category": "Footwear",
                "image_url": "https://example/foot.png",
                "role": "footwear",
            },
        ],
        "score_meta": {
            "score": 7.5,
            "occasion_compatibility_score": score,
            "reasons": ["palette aligned"],
        },
    }


def test_weak_match_relaxation_via_finalize(monkeypatch):
    """Unit-test the relax block directly through finalize_style_response_payload."""
    cards = [_card(0, 0.65, "Boardroom Casual"), _card(1, 0.60, "Creative Professional")]

    result = {
        "intent": "daily_outfit",
        "outfits": cards,
        "cards": cards,
        "boards": cards,
        "context": "ok",
        "meta": {},
    }

    # Pin threshold high so weak_match fires.
    import brain.engines.style_scorer as scorer

    monkeypatch.setattr(scorer, "occasion_confidence_threshold", lambda _: 0.72)

    wardrobe = (
        [{"id": f"t_{i}", "name": "shirt", "category": "Tops"} for i in range(5)]
        + [{"id": f"b_{i}", "name": "trousers", "category": "Bottoms"} for i in range(3)]
        + [{"id": f"f_{i}", "name": "loafers", "category": "Footwear"} for i in range(3)]
    )
    ctx = {"occasion": "office", "query": "outfit for client meeting"}

    finalized = sfs.finalize_style_response_payload(
        result,
        user_id="u1",
        query="outfit for client meeting",
        wardrobe=wardrobe,
        context=ctx,
        cache_bypass=True,
    )

    # Critical: cards still returned, not suppressed.
    assert finalized.get("type") != "weak_match"
    assert finalized.get("cards"), "weak-match safety gate must NOT suppress cards"
    for card in finalized["cards"]:
        assert card.get("weak_match") is True
        # Badge: set to CLOSEST OPTION only when missing; pre-existing
        # occasion badges (e.g. OFFICE) are preserved.
        assert card.get("badge"), "card must carry some badge"
        assert "closest wardrobe-safe" in (card.get("confidence_note") or "").lower()


def test_weak_match_relax_skips_when_cards_empty(monkeypatch):
    def fake_finalize(result, **kwargs):
        return {
            "cards": [],
            "style_boards": [],
            "board_ids": "",
            "data": {"outfits": [], "rendered_boards": []},
            "meta": {},
        }

    def fake_get_daily_outfits(payload):
        return {"intent": "daily_outfit", "outfits": [], "cards": [], "boards": [], "context": "", "meta": {}}

    monkeypatch.setattr(sfs, "finalize_style_response_payload", fake_finalize)
    import brain.outfit_pipeline as op

    monkeypatch.setattr(op, "get_daily_outfits", fake_get_daily_outfits)

    wardrobe = [{"id": f"w_{i}", "name": "x", "category": "Tops"} for i in range(10)]
    resp = sfs.build_style_flow_response(
        user_id="u1",
        query="client meeting",
        wardrobe=wardrobe,
        user_profile={"gender": "male"},
        context={"occasion": "office"},
    )
    # Empty cards still produces the legacy gap response, not the relaxed
    # path (we have nothing to relax).
    assert resp.get("cards") == []


def test_orch_timeout_default_is_45_in_source():
    """Verify the literal default in routers/chat.py is now 45 (was 20)."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "routers" / "chat.py"
    text = src.read_text(encoding="utf-8")
    assert 'or "45"' in text, "expected default 45 fallback literal in routers/chat.py"
    assert 'or "20"' not in text, "legacy default 20 still present"
