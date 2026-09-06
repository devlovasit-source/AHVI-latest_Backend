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


def test_gemini_multi_preview_runs_physical_analysis_on_each_crop(monkeypatch):
    image, raw = _source_image()
    calls = []

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

    def _analyze(crop_bytes, detector_metadata, request_id):
        calls.append(
            {
                "crop_bytes": crop_bytes,
                "metadata": detector_metadata,
                "request_id": request_id,
            }
        )
        return {
            "status": "success",
            "provider": "ollama",
            "model": "test-vision-model",
            "latency_ms": 12,
            "confidence_summary": {
                "fabric_weight": 0.9,
                "fit": 0.9,
            },
            "failure_reason": "",
            "observations": {
                "fabric_weight": {
                    "value": "medium",
                    "confidence": 0.9,
                },
                "fabric_structure": {
                    "value": "woven",
                    "confidence": 0.9,
                },
                "fit": {
                    "value": "regular",
                    "confidence": 0.9,
                },
                "drape": {
                    "value": "structured",
                    "confidence": 0.8,
                },
                "coverage_level": {
                    "value": "full_length",
                    "confidence": 0.9,
                },
                "lining": {
                    "value": "unknown",
                    "confidence": 0.0,
                },
                "surface_texture": {
                    "value": "smooth",
                    "confidence": 0.9,
                },
                "material_family_candidates": [],
            },
        }

    monkeypatch.setattr(wc._gemini_multi, "is_enabled", lambda: True)
    monkeypatch.setattr(wc._gemini_multi, "detect_and_crop", _detect)
    monkeypatch.setattr(wc, "analyze_garment", _analyze)
    monkeypatch.setattr(
        wc,
        "_find_upload_duplicate",
        lambda **_kwargs: wc._duplicate_result(
            checked=False,
            is_duplicate=False,
        ),
    )

    request = wc.CaptureAnalyzeRequest(
        user_id="user-1",
        image_base64=base64.b64encode(raw).decode("ascii"),
        auto_save=False,
        save_duplicates=False,
    )

    result = asyncio.run(wc.analyze_capture(_Request(), request))

    assert len(calls) == 2
    item_names = {c["metadata"]["name"] for c in calls}
    assert "Blue Shirt" in item_names
    assert "Black Trousers" in item_names

    assert result["items"][0]["physical_garment_observations"]["fabric_weight"]["value"] == "medium"
    assert result["items"][0]["physical_garment_observations"]["fit"]["value"] == "regular"
    assert result["items"][1]["physical_garment_observations"]["fabric_structure"]["value"] == "woven"


def test_physical_analysis_failure_does_not_break_capture(monkeypatch):
    image, raw = _source_image()

    async def _detect(*_args, **_kwargs):
        return [
            {
                "name": "Black Trousers",
                "category": "Bottoms",
                "sub_category": "Trousers",
                "color": "Black",
                "confidence": 0.9,
                "bbox_px": [0, 0, 40, 40],
                "crop_bytes": raw,
            }
        ]

    def _analyze(*_args, **_kwargs):
        raise TimeoutError("physical analysis timed out")

    monkeypatch.setattr(wc._gemini_multi, "is_enabled", lambda: True)
    monkeypatch.setattr(wc._gemini_multi, "detect_and_crop", _detect)
    monkeypatch.setattr(wc, "analyze_garment", _analyze)
    monkeypatch.setattr(
        wc,
        "_find_upload_duplicate",
        lambda **_kwargs: wc._duplicate_result(
            checked=False,
            is_duplicate=False,
        ),
    )

    request = wc.CaptureAnalyzeRequest(
        user_id="user-1",
        image_base64=base64.b64encode(raw).decode("ascii"),
        auto_save=False,
        save_duplicates=False,
    )

    result = asyncio.run(wc.analyze_capture(_Request(), request))

    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["name"] == "Black Trousers"
    assert item["category"] == "Bottoms"
    assert item["physical_garment_observations"] is None


def test_multi_item_physical_analysis_is_concurrent_and_bounded(monkeypatch):
    import time
    image, raw = _source_image()
    active_concurrent = 0
    max_concurrent = 0

    async def _detect(*_args, **_kwargs):
        return [
            {
                "name": f"Garment {i}",
                "category": "Tops",
                "sub_category": "Shirt",
                "color": "Blue",
                "confidence": 0.9,
                "bbox_px": [0, 0, 10, 10],
                "crop_bytes": raw,
            }
            for i in range(5)
        ]

    def _analyze(crop_bytes, detector_metadata, request_id):
        nonlocal active_concurrent, max_concurrent
        active_concurrent += 1
        max_concurrent = max(max_concurrent, active_concurrent)
        time.sleep(0.05)
        active_concurrent -= 1
        return {
            "status": "success",
            "provider": "ollama",
            "model": "test-model",
            "latency_ms": 50,
            "confidence_summary": {},
            "failure_reason": "",
            "observations": {
                "fabric_weight": {"value": "medium", "confidence": 0.9},
                "fabric_structure": {"value": "woven", "confidence": 0.9},
                "fit": {"value": "regular", "confidence": 0.9},
                "drape": {"value": "structured", "confidence": 0.8},
                "coverage_level": {"value": "full_length", "confidence": 0.9},
                "lining": {"value": "unknown", "confidence": 0.0},
                "surface_texture": {"value": "smooth", "confidence": 0.9},
                "material_family_candidates": [],
            },
        }

    monkeypatch.setattr(wc._gemini_multi, "is_enabled", lambda: True)
    monkeypatch.setattr(wc._gemini_multi, "detect_and_crop", _detect)
    monkeypatch.setattr(wc, "analyze_garment", _analyze)
    monkeypatch.setattr(
        wc,
        "_find_upload_duplicate",
        lambda **_kwargs: wc._duplicate_result(
            checked=False,
            is_duplicate=False,
        ),
    )

    request = wc.CaptureAnalyzeRequest(
        user_id="user-1",
        image_base64=base64.b64encode(raw).decode("ascii"),
        auto_save=False,
        save_duplicates=False,
    )

    result = asyncio.run(wc.analyze_capture(_Request(), request))
    assert len(result["items"]) == 5
    assert 1 < max_concurrent <= 4
