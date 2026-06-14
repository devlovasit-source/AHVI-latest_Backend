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
