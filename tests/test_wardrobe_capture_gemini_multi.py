"""AHVI Gemini bbox + RMBG multi-garment preview MVP.

Covers:
- Gemini multi-garment detection + crop (valid JSON -> bbox/crop fields)
- invalid JSON / disabled detector -> existing flow runs unchanged
- deterministic taxonomy (saree) and suitability (boxers/private) guards are
  still applied to Gemini-detected items
- the preview item schema is unchanged regardless of detection path
"""

from __future__ import annotations

import asyncio
import base64
import io

from PIL import Image

from routers import wardrobe_capture as wc
from services import gemini_multi_garment_detector as gmg


def _run(coro):
    return asyncio.run(coro)


def _test_image(size=(200, 200), color=(120, 60, 200)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return img, buf.getvalue()


def _image_b64(size=(200, 200)) -> str:
    _, raw = _test_image(size)
    return base64.b64encode(raw).decode("utf-8")


class _FakeRequestState:
    def __init__(self, user_id="u1", request_id="req-1"):
        self.user = {"user_id": user_id}
        self.request_id = request_id


class _FakeHttpRequest:
    def __init__(self, user_id="u1", request_id="req-1"):
        self.state = _FakeRequestState(user_id, request_id)


def _capture_request(**overrides):
    base = dict(
        user_id="u1",
        image_base64=_image_b64(),
        auto_save=False,
        save_duplicates=False,
    )
    base.update(overrides)
    return wc.CaptureAnalyzeRequest(**base)


async def _passthrough_bg(data: bytes) -> bytes:
    return data


def _no_duplicate(**_kwargs):
    return wc._duplicate_result(checked=False, is_duplicate=False)


def _wire_router_for_gemini(monkeypatch, raw_json: str):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": raw_json
    )
    monkeypatch.setattr(wc, "remove_bg_bytes", _passthrough_bg)
    monkeypatch.setattr(wc, "_find_upload_duplicate", _no_duplicate)


GOOD_RESULT = """
[
  {"name": "Red Eyelet Dress", "category": "Dresses", "sub_category": "Mini Dress", "color": "Red", "confidence": 0.92, "bbox": [0.05, 0.05, 0.45, 0.95]},
  {"name": "Black Handbag", "category": "Bags", "sub_category": "Handbag", "color": "Black", "confidence": 0.85, "bbox": [0.5, 0.5, 0.8, 0.85]},
  {"name": "Sunglasses", "category": "Accessories", "sub_category": "Eyewear", "color": "Black", "confidence": 0.8, "bbox": [0.1, 0.02, 0.3, 0.12]},
  {"name": "White Sneakers", "category": "Footwear", "sub_category": "Sneakers", "color": "White", "confidence": 0.88, "bbox": [0.55, 0.6, 0.95, 0.98]}
]
"""

SAREE_AND_BOXERS_RESULT = """
[
  {"name": "Red Saree", "category": "Dresses", "sub_category": "Saree", "color": "Red", "confidence": 0.9, "bbox": [0.05, 0.05, 0.5, 0.95]},
  {"name": "Cotton Boxers", "category": "Bottoms", "sub_category": "Shorts", "color": "Blue", "confidence": 0.7, "bbox": [0.55, 0.5, 0.95, 0.95]}
]
"""


# ---------- detector: multi-item detection ----------

def test_gemini_multi_detects_dress_bag_sunglasses_footwear(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": GOOD_RESULT
    )

    image, raw = _test_image()
    items = _run(gmg.detect_and_crop(image, raw, request_id="t1"))

    assert len(items) == 4
    names = {i["name"] for i in items}
    assert names == {"Red Eyelet Dress", "Black Handbag", "Sunglasses", "White Sneakers"}
    categories = {i["category"] for i in items}
    assert {"Dresses", "Bags", "Accessories", "Footwear"} <= categories


# ---------- detector: bbox/crop fields present ----------

def test_gemini_multi_bbox_crop_fields_present(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": GOOD_RESULT
    )

    image, raw = _test_image()
    items = _run(gmg.detect_and_crop(image, raw, request_id="t2"))

    assert len(items) >= 2
    for item in items:
        for key in (
            "name", "category", "sub_category", "color", "confidence",
            "bbox", "bbox_px", "crop_bytes", "needs_review",
        ):
            assert key in item, key
        assert len(item["bbox"]) == 4
        x1, y1, x2, y2 = item["bbox_px"]
        assert 0 <= x1 < x2 <= image.size[0]
        assert 0 <= y1 < y2 <= image.size[1]
        assert isinstance(item["crop_bytes"], bytes) and len(item["crop_bytes"]) > 0


# ---------- invalid JSON / disabled -> existing flow unchanged ----------

def test_gemini_multi_invalid_json_falls_back(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": "not json at all"
    )

    image, raw = _test_image()
    items = _run(gmg.detect_and_crop(image, raw, request_id="t3"))
    assert items == []


def test_gemini_multi_invalid_json_router_falls_back_to_existing_flow(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setenv("WARDROBE_CAPTURE_SINGLE_GARMENT_MODE", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": "not json at all"
    )
    monkeypatch.setattr(wc, "remove_bg_bytes", _passthrough_bg)
    monkeypatch.setattr(wc, "_find_upload_duplicate", _no_duplicate)

    http_request = _FakeHttpRequest()
    request = _capture_request()
    result = _run(wc.analyze_capture(http_request, request))

    assert result["stage_trace"]["detection"] != "gemini_multi_garment"
    assert result["count"] >= 1


# ---------- end-to-end: saree taxonomy enforced for Gemini items ----------

def test_gemini_multi_saree_taxonomy_enforced(monkeypatch):
    _wire_router_for_gemini(monkeypatch, SAREE_AND_BOXERS_RESULT)

    http_request = _FakeHttpRequest()
    request = _capture_request()
    result = _run(wc.analyze_capture(http_request, request))

    assert result["stage_trace"]["detection"] == "gemini_multi_garment"
    saree_items = [i for i in result["items"] if "saree" in str(i.get("name", "")).lower()]
    assert saree_items
    item = saree_items[0]
    assert item["category"] == "Dresses"
    assert "saree" in str(item.get("sub_category", "")).lower()
    assert item.get("publicWear") is True
    assert item.get("styleEligible") is True


# ---------- end-to-end: private wear guard for Gemini items ----------

def test_gemini_multi_boxers_private(monkeypatch):
    _wire_router_for_gemini(monkeypatch, SAREE_AND_BOXERS_RESULT)

    http_request = _FakeHttpRequest()
    request = _capture_request()
    result = _run(wc.analyze_capture(http_request, request))

    boxer_items = [i for i in result["items"] if "boxer" in str(i.get("name", "")).lower()]
    assert boxer_items
    item = boxer_items[0]
    assert item["category"] == "Innerwear"
    assert item.get("privateWear") is True
    assert item.get("publicWear") is False
    assert item.get("styleEligible") is False


# ---------- save schema unchanged regardless of detection path ----------

def test_gemini_multi_does_not_change_save_schema(monkeypatch):
    # Gemini multi-garment path.
    _wire_router_for_gemini(monkeypatch, SAREE_AND_BOXERS_RESULT)
    http_request = _FakeHttpRequest()
    gemini_result = _run(wc.analyze_capture(http_request, _capture_request()))
    assert gemini_result["stage_trace"]["detection"] == "gemini_multi_garment"
    gemini_keys = set(gemini_result["items"][0].keys())

    # Existing single-garment fallback path (Gemini disabled).
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "false")
    monkeypatch.setenv("WARDROBE_CAPTURE_SINGLE_GARMENT_MODE", "true")
    fallback_result = _run(wc.analyze_capture(http_request, _capture_request()))
    assert fallback_result["stage_trace"]["detection"] != "gemini_multi_garment"
    fallback_keys = set(fallback_result["items"][0].keys())

    # The two paths may legitimately differ on keys that the *existing*
    # taxonomy/suitability guards already add conditionally (e.g. a saree
    # gets publicWear/styleEligible/subCategory, a generic placeholder item
    # gets needs_review instead) - that behavior predates this feature.
    # What must NOT happen is the Gemini path introducing any field the
    # existing pipeline doesn't already know how to produce.
    taxonomy_optional_keys = {
        "privateWear", "publicWear", "styleEligible", "subCategory",
        "needs_review", "requires_manual_entry", "review_reason",
    }
    assert (gemini_keys - fallback_keys) <= taxonomy_optional_keys
    assert (fallback_keys - gemini_keys) <= taxonomy_optional_keys

    # Required preview/save fields survive on the Gemini path.
    for key in (
        "item_id", "name", "category", "sub_category", "color_code",
        "color_name", "pattern", "occasions", "confidence", "label_source",
        "requires_manual_entry", "reasoning", "bbox", "raw_url", "rawUrl",
        "masked_url", "maskedUrl", "normalized_url", "normalizedUrl",
        "image_url", "imageUrl", "raw_image_base64", "masked_image_base64",
        "upload_error", "pixel_hash", "duplicate", "image_embedding",
    ):
        assert key in gemini_keys, key
