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


def test_unknown_catalog_attr_retries_without_catalog_fields(monkeypatch, caplog):
    posts = _patch(
        monkeypatch,
        [
            _Resp(400, text='Unknown attribute: "catalog_status"'),
            _Resp(201, payload={"$id": "doc1"}),
        ],
    )
    import logging

    with caplog.at_level(logging.INFO, logger="ahvi"):
        res = wps._create_document("doc1", _catalog_doc())

    assert res.get("$id") == "doc1"  # save succeeded (returns parsed json)
    assert len(posts) == 2  # retried once
    # retry stripped every catalog field...
    retry = posts[1]
    for k in ("catalog_url", "catalog_status", "catalog_method"):
        assert k not in retry
    # ...but kept raw/masked image fields
    assert retry["image_url"] and retry["masked_url"] and retry["normalized_url"]
    assert "ahvi.catalog.persistence_stripped" in "\n".join(caplog.messages)


def test_non_catalog_error_still_fails(monkeypatch):
    # 401 has no "unknown attribute" -> no retry, save fails (raises).
    posts = _patch(monkeypatch, [_Resp(401, text="Unauthorized")])
    with pytest.raises(RuntimeError):
        wps._create_document("doc2", _catalog_doc())
    assert len(posts) == 1  # no retry


def test_unknown_attr_for_other_field_not_masked(monkeypatch):
    # Unknown attribute about a NON-catalog field: stripped retry still includes
    # it, so the save still fails (we don't silently swallow real schema bugs).
    posts = _patch(
        monkeypatch,
        [
            _Resp(400, text='Unknown attribute: "bogus_field"'),
            _Resp(400, text='Unknown attribute: "bogus_field"'),
        ],
    )
    doc = _catalog_doc()
    doc["bogus_field"] = "x"
    with pytest.raises(RuntimeError):
        wps._create_document("doc3", doc)
    assert posts[1].get("bogus_field") == "x"  # bogus field NOT stripped
    # catalog fields WERE stripped on the retry even though it still failed
    assert "catalog_status" not in posts[1]


def test_flag_off_no_catalog_fields_unchanged(monkeypatch):
    # No catalog fields present -> single post, no strip path.
    posts = _patch(monkeypatch, [_Resp(201, payload={"$id": "doc4"})])
    res = wps._create_document("doc4", _base_doc())
    assert res.get("$id") == "doc4"
    assert len(posts) == 1
    assert posts[0]["image_url"] and posts[0]["masked_url"]
