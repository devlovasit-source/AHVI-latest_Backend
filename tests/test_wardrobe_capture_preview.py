import asyncio
import base64
import io

from PIL import Image

from routers import wardrobe_capture as wc


class _State:
    user = {"user_id": "user-1"}
    request_id = "request-1"


class _Request:
    state = _State()


def _source_image():
    image = Image.new("RGB", (40, 40), (120, 80, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image, buffer.getvalue()


def test_gemini_multi_preview_keeps_raw_crops_and_does_not_call_rmbg(monkeypatch):
    image, raw = _source_image()
    calls = {"rmbg": 0}

    async def _detect(*_args, **_kwargs):
        return [
            {
                "name": "Blue Shirt",
                "category": "Tops",
                "sub_category": "Shirt",
                "color": "Blue",
                "confidence": 0.9,
                "bbox_px": [0, 0, 20, 40],
                "crop_bytes": raw,
            },
            {
                "name": "Black Trousers",
                "category": "Bottoms",
                "sub_category": "Trousers",
                "color": "Black",
                "confidence": 0.9,
                "bbox_px": [20, 0, 40, 40],
                "crop_bytes": raw,
            },
        ]

    async def _remove_bg(data):
        calls["rmbg"] += 1
        return data

    monkeypatch.setattr(wc._gemini_multi, "is_enabled", lambda: True)
    monkeypatch.setattr(wc._gemini_multi, "detect_and_crop", _detect)
    monkeypatch.setattr(wc, "remove_bg_bytes", _remove_bg)
    monkeypatch.setattr(
        wc,
        "_find_upload_duplicate",
        lambda **_kwargs: wc._duplicate_result(checked=False, is_duplicate=False),
    )

    request = wc.CaptureAnalyzeRequest(
        user_id="user-1",
        image_base64=base64.b64encode(raw).decode("ascii"),
        auto_save=False,
        save_duplicates=False,
    )
    result = asyncio.run(wc.analyze_capture(_Request(), request))

    assert result["stage_trace"]["detection"] == "gemini_multi_garment"
    assert calls["rmbg"] == 0
    assert len(result["items"]) == 2
    for item in result["items"]:
        assert item["preview_cutout_pending"] is True
        assert item["raw_image_base64"]
        assert item["masked_image_base64"] == item["raw_image_base64"]
