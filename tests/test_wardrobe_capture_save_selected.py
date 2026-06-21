import base64
import io

from PIL import Image

from routers import wardrobe_capture as wc


class _State:
    user = {"user_id": "user-1"}
    request_id = "request-1"


class _Request:
    state = _State()


class _R2:
    calls = []

    def upload_wardrobe_images(self, *, file_id, raw_image_bytes, masked_image_bytes):
        self.calls.append((file_id, raw_image_bytes, masked_image_bytes))
        return {
            "raw_image_url": f"https://raw.test/{file_id}.png",
            "masked_image_url": f"https://masked.test/{file_id}.png",
            "normalized_image_url": f"https://normalized.test/{file_id}.png",
        }


def _png_base64(color):
    image = Image.new("RGB", (12, 12), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _item(item_id, color=(200, 20, 20)):
    crop = _png_base64(color)
    return {
        "item_id": item_id,
        "name": f"Item {item_id}",
        "category": "Tops",
        "label_source": "vision:gemini_multi",
        "preview_cutout_pending": True,
        "raw_image_base64": crop,
        "masked_image_base64": crop,
        "imageUrl": f"https://preview.test/{item_id}.png",
        "cropUrl": f"https://crop.test/{item_id}.png",
    }


def _wire(monkeypatch, remove_bg):
    _R2.calls = []
    persisted = {}
    monkeypatch.setattr(wc, "R2Storage", _R2)
    monkeypatch.setattr(wc, "remove_bg_bytes", remove_bg)

    def _persist(*, user_id, selected_item_ids, detected_items):
        persisted["items"] = [dict(item) for item in detected_items]
        return {"success": True, "saved_count": len(selected_item_ids), "errors": []}

    monkeypatch.setattr(wc, "persist_selected_items", _persist)
    return persisted


def test_save_selected_runs_separate_rmbg_and_sets_masked_urls(monkeypatch):
    calls = []

    async def _remove_bg(raw):
        calls.append(raw)
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)
    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["one", "two"],
        detected_items=[_item("one"), _item("two", (20, 20, 200))],
    )

    result = wc.save_selected(_Request(), request)

    assert result["success"] is True
    assert len(calls) == 2
    assert len(_R2.calls) == 2
    for item in persisted["items"]:
        assert item["imageStatus"] == "rmbg_complete"
        assert item["maskedUrl"] == item["masked_url"]
        assert item["imageUrl"] == f"https://preview.test/{item['item_id']}.png"
        assert item["cropUrl"] == f"https://crop.test/{item['item_id']}.png"


def test_save_selected_rmbg_failure_falls_back_without_failing_save(monkeypatch):
    async def _remove_bg(_raw):
        raise RuntimeError("service unavailable")

    persisted = _wire(monkeypatch, _remove_bg)
    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["one"],
        detected_items=[_item("one")],
    )

    result = wc.save_selected(_Request(), request)

    assert result["success"] is True
    saved = persisted["items"][0]
    assert saved["imageStatus"] == "rmbg_failed"
    assert saved["maskedUrl"]
    assert _R2.calls[0][1] == _R2.calls[0][2]


def test_save_selected_treats_fail_open_original_bytes_as_rmbg_failure(monkeypatch):
    async def _remove_bg(raw):
        return raw

    persisted = _wire(monkeypatch, _remove_bg)
    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["one"],
        detected_items=[_item("one")],
    )

    result = wc.save_selected(_Request(), request)

    assert result["success"] is True
    assert persisted["items"][0]["imageStatus"] == "rmbg_failed"


def test_save_selected_skips_unselected_items_for_rmbg_catalog_and_persist(monkeypatch):
    calls = {"rmbg": 0, "catalog": 0}

    async def _remove_bg(raw):
        calls["rmbg"] += 1
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)

    def _catalog(item):
        calls["catalog"] += 1

    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", _catalog)
    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["one"],
        detected_items=[_item("one"), _item("two", (20, 20, 200))],
    )

    result = wc.save_selected(_Request(), request)

    assert result["success"] is True
    assert calls == {"rmbg": 1, "catalog": 1}
    assert [i["item_id"] for i in persisted["items"]] == ["one"]
    assert result["selected_count"] == 1
    assert result["regen_skipped_count"] == 1


def test_save_selected_skips_needs_review_and_rejected_items(monkeypatch):
    calls = {"rmbg": 0, "catalog": 0}

    async def _remove_bg(raw):
        calls["rmbg"] += 1
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)
    monkeypatch.setattr(
        wc,
        "_maybe_generate_catalog_image",
        lambda item: calls.__setitem__("catalog", calls["catalog"] + 1),
    )
    ok = _item("ok")
    ok["validation_status"] = "ok"
    review = _item("review")
    review["validation_status"] = "needs_review"
    review["needs_review"] = True
    rejected = _item("rejected")
    rejected["validation_status"] = "rejected"

    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["ok", "review", "rejected"],
        detected_items=[ok, review, rejected],
    )

    result = wc.save_selected(_Request(), request)

    assert result["success"] is True
    assert calls == {"rmbg": 1, "catalog": 1}
    assert [i["item_id"] for i in persisted["items"]] == ["ok"]
    assert result["selected_count"] == 1
    assert result["rejected_selected_count"] == 2
    assert result["regen_skipped_count"] == 2


def test_save_selected_response_reports_drop_accounting(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    _wire(monkeypatch, _remove_bg)
    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", lambda item: None)

    ok = _item("ok")
    ok["validation_status"] = "ok"
    rejected = _item("rejected")
    rejected["validation_status"] = "rejected"
    rejected["rejection_reason"] = "accessory_low_confidence"

    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["ok", "rejected"],
        detected_items=[ok, rejected],
    )

    result = wc.save_selected(_Request(), request)

    assert result["requested_count"] == 2
    assert result["saved_count"] == 1
    assert result["dropped_count"] == 1
    reasons = result["dropped_reasons"]
    assert len(reasons) == 1
    assert reasons[0]["item_id"] == "rejected"
    assert reasons[0]["validation_status"] == "rejected"
    assert reasons[0]["reason"] == "accessory_low_confidence"


def test_save_selected_skips_unsafe_catalog_generation_failure(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)

    def _catalog(item):
        item["catalogStatus"] = "blocked_unsafe_fallback"
        item["catalogProvider"] = "nanobanana"
        item["catalog_reason"] = "unsafe_source_nanobanana_failed"

    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", _catalog)
    unsafe = _item("unsafe")
    unsafe["validation_status"] = "ok"
    unsafe["source"] = "person_body_crop"

    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["unsafe"],
        detected_items=[unsafe],
    )

    result = wc.save_selected(_Request(), request)

    assert result["success"] is True
    assert result["saved_count"] == 0
    assert persisted["items"] == []
    assert result["selected_count"] == 0
    assert result["dropped_count"] == 1
    assert result["dropped_reasons"][0]["item_id"] == "unsafe"
    assert result["dropped_reasons"][0]["validation_status"] == "rejected"
    assert result["dropped_reasons"][0]["reason"] == "unsafe_catalog_generation_failed"


def test_save_selected_skips_blank_catalog_generation_failure(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)

    def _catalog(item):
        item["catalogStatus"] = "blocked_blank_catalog"
        item["catalogProvider"] = "nanobanana"
        item["catalog_reason"] = "blank_transparent_catalog"

    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", _catalog)
    blank = _item("blank")
    blank["validation_status"] = "ok"
    blank["category"] = "Jewelry"
    blank["sub_category"] = "Necklace"

    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["blank"],
        detected_items=[blank],
    )

    result = wc.save_selected(_Request(), request)

    assert result["success"] is True
    assert result["saved_count"] == 0
    assert persisted["items"] == []
    assert result["dropped_count"] == 1
    assert result["dropped_reasons"][0]["reason"] == "blank_catalog_image"


def test_save_selected_keeps_clean_catalog_fallback(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)

    def _catalog(item):
        item["catalogStatus"] = "fallback_cutout"
        item["catalogProvider"] = "cutout"
        item["normalized_url"] = item.get("masked_url")

    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", _catalog)
    clean = _item("clean")
    clean["validation_status"] = "ok"

    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["clean"],
        detected_items=[clean],
    )

    result = wc.save_selected(_Request(), request)

    assert result["success"] is True
    assert result["saved_count"] == 1
    assert [i["item_id"] for i in persisted["items"]] == ["clean"]
    assert persisted["items"][0]["catalogStatus"] == "fallback_cutout"
    assert persisted["items"][0]["display_image_url"] == persisted["items"][0]["masked_url"]
    assert persisted["items"][0]["display_image_source"] == "masked_fallback"


def test_save_selected_catalog_generated_uses_normalized_display_url(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)

    def _catalog(item):
        item["catalogStatus"] = "catalog_generated"
        item["catalogProvider"] = "nanobanana"
        item["normalized_url"] = "https://normalized.test/catalog-clean.png"
        item["normalizedUrl"] = item["normalized_url"]

    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", _catalog)
    clean = _item("clean")
    clean["validation_status"] = "ok"

    request = wc.SaveSelectedRequest(
        user_id="user-1",
        selected_item_ids=["clean"],
        detected_items=[clean],
    )

    result = wc.save_selected(_Request(), request)

    assert result["success"] is True
    assert result["saved_count"] == 1
    assert persisted["items"][0]["display_image_url"] == "https://normalized.test/catalog-clean.png"
    assert persisted["items"][0]["display_image_source"] == "catalog"


def test_unsafe_source_catalog_generated_is_saved(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)
    monkeypatch.setattr(wc, "_fetch_wardrobe_profile_gender", lambda user_id: "unknown")

    def _catalog(item):
        item["catalogStatus"] = "catalog_generated"
        item["catalogProvider"] = "nanobanana"
        item["normalized_url"] = "https://normalized.test/shirt.png"
        item["normalizedUrl"] = item["normalized_url"]

    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", _catalog)
    shirt = _item("shirt")
    shirt["validation_status"] = "ok"
    shirt["source_contains_person"] = True
    shirt["unsafe_source"] = True

    result = wc.save_selected(
        _Request(),
        wc.SaveSelectedRequest(user_id="user-1", selected_item_ids=["shirt"], detected_items=[shirt]),
    )

    assert result["saved_count"] == 1
    assert persisted["items"][0]["display_image_source"] == "catalog"


def test_unsafe_source_fallback_cutout_is_skipped(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)
    monkeypatch.setattr(wc, "_fetch_wardrobe_profile_gender", lambda user_id: "unknown")

    def _catalog(item):
        item["catalogStatus"] = "fallback_cutout"
        item["catalogProvider"] = "cutout"
        item["masked_url"] = "https://masked.test/jeans.png"

    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", _catalog)
    jeans = _item("jeans")
    jeans["validation_status"] = "ok"
    jeans["name"] = "Distressed Dark Blue Jeans"
    jeans["category"] = "Bottoms"
    jeans["source_contains_person"] = True
    jeans["unsafe_source"] = True

    result = wc.save_selected(
        _Request(),
        wc.SaveSelectedRequest(user_id="user-1", selected_item_ids=["jeans"], detected_items=[jeans]),
    )

    assert result["saved_count"] == 0
    assert persisted["items"] == []
    assert result["dropped_reasons"][0]["reason"] == "unsafe_non_catalog"


def test_screenshot_collage_item_is_not_saved(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)
    monkeypatch.setattr(wc, "_fetch_wardrobe_profile_gender", lambda user_id: "unknown")
    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", lambda item: None)
    collage = _item("collage")
    collage["validation_status"] = "ok"
    collage["input_type"] = "screenshot_style_collage"
    collage["detected_text"] = "Save Remix Like Share"

    result = wc.save_selected(
        _Request(),
        wc.SaveSelectedRequest(user_id="user-1", selected_item_ids=["collage"], detected_items=[collage]),
    )

    assert result["saved_count"] == 0
    assert persisted["items"] == []
    assert result["dropped_reasons"][0]["reason"] == "screenshot_or_style_collage"


def test_male_profile_strong_womenswear_is_blocked(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)
    monkeypatch.setattr(wc, "_fetch_wardrobe_profile_gender", lambda user_id: "male")
    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", lambda item: None)
    saree = _item("saree")
    saree["validation_status"] = "ok"
    saree["name"] = "Teal Saree"
    saree["category"] = "Traditional"
    saree["sub_category"] = "Saree"

    result = wc.save_selected(
        _Request(),
        wc.SaveSelectedRequest(user_id="user-1", selected_item_ids=["saree"], detected_items=[saree]),
    )

    assert result["saved_count"] == 0
    assert persisted["items"] == []
    assert result["dropped_reasons"][0]["reason"] == "outside_current_wardrobe_profile"


def test_male_profile_unisex_shirt_and_jeans_are_allowed(monkeypatch):
    async def _remove_bg(raw):
        return b"masked-" + raw

    persisted = _wire(monkeypatch, _remove_bg)
    monkeypatch.setattr(wc, "_fetch_wardrobe_profile_gender", lambda user_id: "male")
    monkeypatch.setattr(wc, "_maybe_generate_catalog_image", lambda item: None)
    shirt = _item("shirt")
    shirt["validation_status"] = "ok"
    shirt["name"] = "Patterned Long-Sleeve Shirt"
    jeans = _item("jeans")
    jeans["validation_status"] = "ok"
    jeans["name"] = "Distressed Dark Blue Jeans"
    jeans["category"] = "Bottoms"

    result = wc.save_selected(
        _Request(),
        wc.SaveSelectedRequest(
            user_id="user-1",
            selected_item_ids=["shirt", "jeans"],
            detected_items=[shirt, jeans],
        ),
    )

    assert result["saved_count"] == 2
    assert {item["item_id"] for item in persisted["items"]} == {"shirt", "jeans"}
