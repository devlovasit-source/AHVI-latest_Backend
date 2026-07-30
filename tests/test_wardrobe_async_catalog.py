"""WARDROBE_ASYNC_CATALOG: defer catalog PNG gen off the save-selected path.

Guards the enable flag (incl. the privacy hard-disable) and the save-gating
bypass for pending items — the item persists with its rmbg/raw display image
and the catalog is patched in later.
"""
import routers.wardrobe_capture as wc


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("WARDROBE_ASYNC_CATALOG", raising=False)
    monkeypatch.delenv("WARDROBE_PRIVACY_CATALOG_ONLY", raising=False)
    assert wc._async_catalog_enabled() is False


def test_flag_on(monkeypatch):
    monkeypatch.delenv("WARDROBE_PRIVACY_CATALOG_ONLY", raising=False)
    monkeypatch.setenv("WARDROBE_ASYNC_CATALOG", "true")
    assert wc._async_catalog_enabled() is True


def test_privacy_catalog_only_hard_disables_async(monkeypatch):
    # Faces may only be stored via the catalog, so it can't be deferred.
    monkeypatch.setenv("WARDROBE_ASYNC_CATALOG", "true")
    monkeypatch.setenv("WARDROBE_PRIVACY_CATALOG_ONLY", "true")
    assert wc._async_catalog_enabled() is False


def test_pending_item_is_not_blocked_from_save():
    reason = wc._save_selected_block_reason(
        {"item_id": "x", "catalogStatus": "catalog_pending", "validation_status": "ok"}
    )
    assert reason == ""


def test_pending_still_honours_prior_gates():
    # A pending item that failed validation must still be blocked.
    reason = wc._save_selected_block_reason(
        {"item_id": "x", "catalogStatus": "catalog_pending", "validation_status": "rejected"}
    )
    assert reason == "validation_not_ok"
