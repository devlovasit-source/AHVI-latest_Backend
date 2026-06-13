"""works-with + best-for endpoints and ethnic taxonomy mapping.

Covers:
- works-with returns ONLY real user wardrobe items (no hardcoded examples)
- works-with excludes the current item itself
- best-for derives occasions from enrich_wardrobe_item metadata
- sherwani/bandhgala/nehru guardrail maps to Outerwear
"""

from __future__ import annotations

from routers import wardrobe_capture as wc


class _FakeRequestState:
    def __init__(self, user_id="u1"):
        self.user = {"user_id": user_id}
        self.request_id = "req-1"


class _FakeHttpRequest:
    def __init__(self, user_id="u1"):
        self.state = _FakeRequestState(user_id)


class _FakeProxy:
    """Stands in for AppwriteProxy().list_documents."""

    rows: list = []

    def list_documents(self, resource, *, user_id=None, limit=100, **_kw):
        return list(_FakeProxy.rows)


# ---------- works-with: real wardrobe only, excludes self ----------

def test_works_with_returns_only_real_user_items(monkeypatch):
    monkeypatch.setattr(
        wc, "_ahvi_fetch_outfit_doc", lambda _id: {"category": "Tops", "userId": "u1"}
    )
    _FakeProxy.rows = [
        {"$id": "self-1", "category": "Tops", "name": "White Tee"},
        {"$id": "b-1", "category": "Bottoms", "name": "Blue Jeans", "masked_url": "u1"},
        {"$id": "f-1", "category": "Footwear", "name": "Sneakers", "image_url": "u2"},
        {"$id": "x-1", "category": "Bags", "name": "Tote"},  # not complementary to Tops
    ]
    monkeypatch.setattr(wc, "AppwriteProxy", _FakeProxy)

    result = wc.get_works_with("self-1", _FakeHttpRequest())

    names = {m["name"] for m in result["matches"]}
    assert names == {"Blue Jeans", "Sneakers"}  # only real complementary items
    assert result["count"] == 2  # honest count == len(matches)
    # Every returned item came from the wardrobe rows, nothing invented.
    real_ids = {"b-1", "f-1"}
    assert {m["item_id"] for m in result["matches"]} == real_ids


def test_works_with_excludes_current_item(monkeypatch):
    monkeypatch.setattr(
        wc, "_ahvi_fetch_outfit_doc", lambda _id: {"category": "Bottoms", "userId": "u1"}
    )
    _FakeProxy.rows = [
        {"$id": "me", "category": "Bottoms", "name": "My Jeans"},
        {"$id": "t-1", "category": "Tops", "name": "Linen Shirt"},
    ]
    monkeypatch.setattr(wc, "AppwriteProxy", _FakeProxy)

    result = wc.get_works_with("me", _FakeHttpRequest())

    ids = {m["item_id"] for m in result["matches"]}
    assert "me" not in ids
    assert ids == {"t-1"}


# ---------- best-for: from enrich_wardrobe_item metadata ----------

def test_best_for_uses_enrich_metadata(monkeypatch):
    # "loafer" -> occasion_affinity ["office","date","cocktail"] in
    # enrich_wardrobe_item; verify mapped display values come through.
    monkeypatch.setattr(
        wc,
        "_ahvi_fetch_outfit_doc",
        lambda _id: {"category": "Footwear", "name": "Brown Loafers", "userId": "u1"},
    )

    result = wc.get_best_for("shoe-1", _FakeHttpRequest())

    assert result["best_for"] == ["Office", "Dinner", "Cocktail"]
    assert "Gym" in result["avoid"]


def test_best_for_fallback_only_when_metadata_empty(monkeypatch):
    monkeypatch.setattr(
        wc, "_ahvi_fetch_outfit_doc", lambda _id: {"category": "Tops", "userId": "u1"}
    )
    # enrich_wardrobe_item defaults occasion_affinity to ["daily"] when nothing
    # matches, so best_for is never empty -> generic "Everyday Wear".
    result = wc.get_best_for("plain-1", _FakeHttpRequest())
    assert result["best_for"] == ["Everyday Wear"]


# ---------- guardrail: ethnic menswear -> Outerwear ----------

def test_sherwani_bandhgala_nehru_map_to_outerwear():
    for term, sub in [
        ("sherwani", "Sherwani"),
        ("bandhgala", "Bandhgala"),
        ("nehru jacket", "Nehru Jacket"),
    ]:
        cat, sub_cat, corrected = wc._guardrail_category(
            raw_label=term,
            vision_name=term,
            vision_category="Tops",
            vision_sub_category="",
            fallback_category="Tops",
            fallback_sub_category="",
        )
        assert cat == "Outerwear", term
        assert sub_cat == sub
        assert corrected is True
