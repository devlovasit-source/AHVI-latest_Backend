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


def test_background_catalog_emits_completed_diagnostic(monkeypatch, caplog):
    item = {"item_id": "item-1"}

    def generate(value):
        value["catalogStatus"] = "catalog_generated"
        value["normalized_url"] = "https://images.example/item-1.png"

    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", generate)
    monkeypatch.setattr(wc, "_apply_display_image_fields", lambda value: value)
    monkeypatch.setattr(wc, "update_wardrobe_item_images", lambda **kwargs: True)

    with caplog.at_level("INFO"):
        wc._run_bg_finalize_catalog("user-1", [item])

    assert "AHVI_ASYNC_CATALOG_COMPLETED item_id=item-1" in caplog.text
    assert "images.example" not in caplog.text


def test_background_catalog_accepts_catalog_ready(monkeypatch, caplog):
    item = {"item_id": "item-1"}
    patches = []

    def generate(value):
        value["catalogStatus"] = "catalog_ready"
        value["normalized_url"] = "https://images.example/item-1.png"

    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", generate)
    monkeypatch.setattr(wc, "_apply_display_image_fields", lambda value: value)
    monkeypatch.setattr(
        wc,
        "update_wardrobe_item_images",
        lambda **kwargs: patches.append(kwargs) or True,
    )

    with caplog.at_level("INFO"):
        wc._run_bg_finalize_catalog("user-1", [item])

    assert patches[0]["catalog_status"] == "catalog_ready"
    assert "AHVI_ASYNC_CATALOG_COMPLETED item_id=item-1" in caplog.text


def test_background_catalog_failure_keeps_fallback_and_is_diagnostic(monkeypatch, caplog):
    item = {"item_id": "item-1", "catalogStatus": "fallback_cutout"}
    patches = []
    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", lambda value: None)
    monkeypatch.setattr(wc, "_apply_display_image_fields", lambda value: value)
    monkeypatch.setattr(
        wc,
        "update_wardrobe_item_images",
        lambda **kwargs: patches.append(kwargs) or True,
    )

    with caplog.at_level("WARNING"):
        wc._run_bg_finalize_catalog("user-1", [item])

    assert patches == [{
        "user_id": "user-1",
        "item_id": "item-1",
        "catalog_status": "catalog_failed",
    }]
    assert "AHVI_ASYNC_CATALOG_FAILED item_id=item-1" in caplog.text
