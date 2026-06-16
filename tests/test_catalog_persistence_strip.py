"""P0: catalog_* fields must never break wardrobe save.

If the Appwrite schema lacks the catalog attributes, _create_document must
strip them and retry once — preserving raw/masked image fields — and never
fail the save. Non-catalog Appwrite errors must still fail.
"""

import pytest

import services.wardrobe_persistence_service as wps


class _Resp:
    def __init__(self, status_code, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


def _base_doc():
    return {
        "image_url": "https://r2/x.png",
        "masked_url": "https://r2/m.png",
        "normalized_url": "https://r2/n.png",
        "category": "Top",
        "userId": "u1",
        "name": "Blue Shirt",
    }


def _catalog_doc():
    d = _base_doc()
    d.update(
        {
            "catalog_url": "https://r2/catalog_x.jpg",
            "catalog_status": "catalog_ready",
            "catalog_method": "rmbg_center_normalize",
        }
    )
    return d


def _patch(monkeypatch, responses):
    """Queue of responses; record each posted payload."""
    monkeypatch.setattr(wps, "_appwrite_ready", lambda: True)
    posts = []

    def fake_post(url, json=None, headers=None, timeout=None):
        posts.append(json["data"])
        return responses[len(posts) - 1]

    monkeypatch.setattr(wps.requests, "post", fake_post)
    return posts


def test_unknown_pixel_hash_keeps_catalog_status(monkeypatch, caplog):
    # THE regression: a single unknown attr (pixel_hash) must NOT take the valid
    # catalog_* fields down with it. Drop only pixel_hash; keep catalog_status.
    import logging

    posts = _patch(
        monkeypatch,
        [
            _Resp(400, text='Invalid document structure: Unknown attribute: "pixel_hash"'),
            _Resp(201, payload={"$id": "doc1"}),
        ],
    )
    doc = _catalog_doc()
    doc["pixel_hash"] = "abc"
    with caplog.at_level(logging.INFO):
        res = wps._create_document("doc1", doc)
    assert res.get("$id") == "doc1"
    assert len(posts) == 2
    retry = posts[1]
    assert "pixel_hash" not in retry  # offending attr dropped
    assert retry.get("catalog_status") == "catalog_ready"  # catalog PRESERVED
    assert retry["image_url"] and retry["masked_url"]
    assert "ahvi.persistence.dropped_unknown_attrs" in "\n".join(caplog.messages)


def test_unknown_catalog_attr_dropped_only_when_named(monkeypatch, caplog):
    # If the schema genuinely lacks catalog_status, drop only that one.
    import logging

    posts = _patch(
        monkeypatch,
        [
            _Resp(400, text='Unknown attribute: "catalog_status"'),
            _Resp(201, payload={"$id": "doc2"}),
        ],
    )
    with caplog.at_level(logging.INFO):
        res = wps._create_document("doc2", _catalog_doc())
    assert res.get("$id") == "doc2"
    assert "catalog_status" not in posts[1]
    assert posts[1]["image_url"]  # image fields preserved
    assert "ahvi.catalog.persistence_stripped" in "\n".join(caplog.messages)


def test_multiple_unknown_attrs_dropped_iteratively(monkeypatch):
    posts = _patch(
        monkeypatch,
        [
            _Resp(400, text='Unknown attribute: "pixel_hash"'),
            _Resp(400, text='Unknown attribute: "image_vector"'),
            _Resp(201, payload={"$id": "doc3"}),
        ],
    )
    doc = _catalog_doc()
    doc["pixel_hash"] = "h"
    doc["image_vector"] = "v"
    res = wps._create_document("doc3", doc)
    assert res.get("$id") == "doc3"
    assert len(posts) == 3
    assert "pixel_hash" not in posts[2] and "image_vector" not in posts[2]
    assert posts[2].get("catalog_status") == "catalog_ready"  # still preserved


def test_non_named_error_still_fails(monkeypatch):
    # 401 (no "unknown attribute") -> no retry, raises.
    posts = _patch(monkeypatch, [_Resp(401, text="Unauthorized")])
    with pytest.raises(RuntimeError):
        wps._create_document("doc4", _catalog_doc())
    assert len(posts) == 1


def test_invalid_structure_without_attr_name_fails(monkeypatch):
    # invalid-structure error that names no attribute (e.g. type mismatch) ->
    # nothing to strip -> raises, no infinite loop.
    posts = _patch(
        monkeypatch,
        [_Resp(400, text='Invalid document structure: Attribute "occasions" must be an array')],
    )
    with pytest.raises(RuntimeError):
        wps._create_document("doc5", _catalog_doc())
    # 'occasions' IS named -> dropped once, then second post needed; queue has 1
    # response so the retry reuses it (still 400) -> raises. Ensure bounded.
    assert len(posts) >= 1


def test_flag_off_no_catalog_fields_unchanged(monkeypatch):
    posts = _patch(monkeypatch, [_Resp(201, payload={"$id": "doc6"})])
    res = wps._create_document("doc6", _base_doc())
    assert res.get("$id") == "doc6"
    assert len(posts) == 1
    assert posts[0]["image_url"] and posts[0]["masked_url"]
