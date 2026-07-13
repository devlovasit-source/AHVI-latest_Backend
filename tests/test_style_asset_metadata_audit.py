import pytest

from scripts.audit_style_asset_metadata import audit_style_assets, build_audit


class _Proxy:
    def __init__(self, rows):
        self.rows = rows
        self.writes = []

    def list_documents(self, resource, **kwargs):
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 100)
        page = self.rows[offset:offset + limit]
        return {"documents": page, "meta": {"has_more": offset + len(page) < len(self.rows)}}

    def get_document(self, resource, document_id):
        return next(row for row in self.rows if row.get("$id") == document_id)

    def update_document(self, resource, document_id, data):
        self.writes.append((resource, document_id, data))


def _row():
    return {
        "$id": "doc-1",
        "asset_id": "asset-1",
        "name": "White Oxford Shirt",
        "category": "top",
        "image_url": "https://cdn.test/shirt.png",
        "gender": "male",
        "colors": ["white"],
        "occasions": ["office"],
        "formality": 7,
        "traits": ["crisp"],
        "professional_safe": True,
        "professionalism_score": 0.9,
        "client_meeting_score": 0.85,
        "boardroom_score": 0.8,
        "safety_tags": ["office", "client meeting"],
        "source": "legacy_import",
    }


def test_audit_is_read_only_by_default_and_emits_proposals():
    proxy = _Proxy([_row()])
    report = audit_style_assets(proxy=proxy)

    assert proxy.writes == []
    assert report["summary"]["status_counts"] == {"ready": 1}
    assert report["proposed_update_count"] == 1
    assert report["applied_count"] == 0
    assert report["assets"][0]["professional_safe"] is True
    assert report["assets"][0]["client_meeting_score"] == 0.85
    assert report["assets"][0]["safety_tags"] == ["office", "client meeting"]


def test_apply_requires_explicit_confirmation():
    proxy = _Proxy([_row()])
    with pytest.raises(ValueError, match="confirm-apply"):
        audit_style_assets(proxy=proxy, apply=True)
    assert proxy.writes == []


def test_confirmed_apply_writes_only_proposed_canonical_update():
    proxy = _Proxy([_row()])
    report = audit_style_assets(proxy=proxy, apply=True, confirm_apply=True)

    assert report["applied_count"] == 1
    assert proxy.writes[0][0:2] == ("style_assets", "doc-1")
    assert proxy.writes[0][2]["metadata_status"] == "ready"
    assert "metadata_updated_at" in proxy.writes[0][2]


def test_applied_canonical_fields_produce_no_second_proposal():
    first = build_audit([_row()])
    persisted = {**_row(), **first["proposed_updates"][0]["changes"]}
    second = build_audit([persisted])

    assert second["proposed_updates"] == []


def test_asset_id_filter_falls_back_when_document_id_differs():
    proxy = _Proxy([_row()])
    report = audit_style_assets(proxy=proxy, asset_id="asset-1")

    assert report["summary"]["total"] == 1
    assert proxy.writes == []
