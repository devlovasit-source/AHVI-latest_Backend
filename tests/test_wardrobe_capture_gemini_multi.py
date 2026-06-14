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


# ---------- end-to-end: Gemini metadata survives into the preview ----------

def test_gemini_multi_preview_keeps_gemini_metadata_not_review_item(monkeypatch):
    """4 detected items must reach the preview with Gemini names/categories,
    not degrade to the manual 'Review item' card."""
    _wire_router_for_gemini(monkeypatch, GOOD_RESULT)

    http_request = _FakeHttpRequest()
    request = _capture_request()
    result = _run(wc.analyze_capture(http_request, request))

    assert result["stage_trace"]["detection"] == "gemini_multi_garment"
    assert result["count"] == 4

    names = [str(i.get("name") or "") for i in result["items"]]
    categories = [str(i.get("category") or "") for i in result["items"]]

    for name in names:
        assert name.lower() not in {"", "item", "review item", "needs review"}, names
    for category in categories:
        assert category not in {"", "Needs Review"}, categories

    assert {"Dresses", "Bags", "Accessories", "Footwear"} <= set(categories)
    assert any("dress" in n.lower() for n in names)
    assert any("handbag" in n.lower() or "bag" in n.lower() for n in names)
    assert any("sunglasses" in n.lower() for n in names)
    assert any("sneaker" in n.lower() for n in names)

    for item in result["items"]:
        assert item.get("requires_manual_entry") is False, item.get("name")
        # Internal pipeline fields are stripped from the response.
        assert "label_source" not in item


# ---------- crop framing: category-aware bbox padding ----------

def test_dress_bbox_expands_and_clamps(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    dress_json = """
    [
      {"name": "Red Maxi Dress", "category": "Dresses", "sub_category": "Maxi Dress", "color": "Red", "confidence": 0.9, "bbox": [0.3, 0.3, 0.5, 0.7]},
      {"name": "Edge Gown", "category": "Dresses", "sub_category": "Gown", "color": "Blue", "confidence": 0.9, "bbox": [0.0, 0.0, 0.5, 0.98]}
    ]
    """
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": dress_json
    )
    image, raw = _test_image(size=(1000, 1000))
    items = _run(gmg.detect_and_crop(image, raw, request_id="pad1"))
    assert len(items) == 2

    # Unpadded dress bbox: [300, 300, 500, 700] (200w x 400h, tall ratio 2.0).
    x1, y1, x2, y2 = items[0]["bbox_px"]
    assert x1 < 300 and x2 > 500          # horizontal pad applied
    assert y1 < 300 and y2 > 700          # vertical pad applied
    assert (300 - y1) > (y2 - 700) - 1    # top pad >= bottom pad
    assert (300 - y1) >= 0.14 * 400       # tall garment: >= 14% of box height on top

    # Edge gown: padding must clamp to image boundaries.
    ex1, ey1, ex2, ey2 = items[1]["bbox_px"]
    assert ex1 >= 0 and ey1 >= 0
    assert ex2 <= 1000 and ey2 <= 1000


def test_accessory_bbox_gets_extra_padding(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    acc_json = """
    [
      {"name": "Sunglasses", "category": "Accessories", "sub_category": "Eyewear", "color": "Black", "confidence": 0.85, "bbox": [0.4, 0.4, 0.6, 0.5]},
      {"name": "Black Handbag", "category": "Bags", "sub_category": "Handbag", "color": "Black", "confidence": 0.85, "bbox": [0.1, 0.6, 0.3, 0.8]}
    ]
    """
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": acc_json
    )
    image, raw = _test_image(size=(1000, 1000))
    items = _run(gmg.detect_and_crop(image, raw, request_id="pad2"))
    assert len(items) == 2

    # Sunglasses unpadded: [400, 400, 600, 500] (200w x 100h).
    # Accessory pad 0.13 > default 0.04: expect ~26px horizontal, ~13px vertical.
    x1, y1, x2, y2 = items[0]["bbox_px"]
    assert (400 - x1) >= 0.10 * 200
    assert (x2 - 600) >= 0.10 * 200
    assert (400 - y1) >= 0.10 * 100
    assert (y2 - 500) >= 0.10 * 100


def test_invalid_bbox_still_rejected(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    bad_json = """
    [
      {"name": "Ghost Dress", "category": "Dresses", "sub_category": "Dress", "color": "Red", "confidence": 0.9, "bbox": [0.9, 0.9, 0.1, 0.1]},
      {"name": "No Box Top", "category": "Tops", "sub_category": "Top", "color": "Blue", "confidence": 0.9}
    ]
    """
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": bad_json
    )
    image, raw = _test_image(size=(1000, 1000))
    items = _run(gmg.detect_and_crop(image, raw, request_id="pad3"))
    assert items == []


# ---------- preview response: no internal debug chips ----------

def test_gemini_multi_preview_has_no_internal_debug_fields(monkeypatch):
    _wire_router_for_gemini(monkeypatch, GOOD_RESULT)

    http_request = _FakeHttpRequest()
    result = _run(wc.analyze_capture(http_request, _capture_request()))

    assert result["stage_trace"]["detection"] == "gemini_multi_garment"
    for item in result["items"]:
        # Internal fields stripped from the response entirely.
        assert "label_source" not in item
        assert "metadata_validator" not in item
        # No internal markers leaking through any visible string value.
        for key, value in item.items():
            if isinstance(value, str):
                assert "vision:gemini_multi" not in value, key
                assert "heuristic" not in value, key
        # Useful chips survive.
        assert str(item.get("category") or "") not in {"", "Needs Review"}
        assert str(item.get("color_name") or "")
        assert str(item.get("pattern") or "")


# ---------- fast path: preview skips RMBG entirely ----------

def test_gemini_multi_preview_skips_rmbg(monkeypatch):
    calls = {"count": 0}

    async def _counting_bg(data: bytes) -> bytes:
        calls["count"] += 1
        return data

    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": GOOD_RESULT
    )
    monkeypatch.setattr(wc, "remove_bg_bytes", _counting_bg)
    monkeypatch.setattr(wc, "_find_upload_duplicate", _no_duplicate)

    http_request = _FakeHttpRequest()
    result = _run(wc.analyze_capture(http_request, _capture_request()))

    assert result["stage_trace"]["detection"] == "gemini_multi_garment"
    assert result["count"] == 4
    assert calls["count"] == 0
    for item in result["items"]:
        assert item.get("preview_cutout_pending") is True


def test_gemini_multi_preview_returns_raw_crops_fast(monkeypatch):
    _wire_router_for_gemini(monkeypatch, GOOD_RESULT)

    http_request = _FakeHttpRequest()
    result = _run(wc.analyze_capture(http_request, _capture_request()))

    assert result["stage_trace"]["detection"] == "gemini_multi_garment"
    assert result["count"] == 4
    for item in result["items"]:
        raw_b64 = str(item.get("raw_image_base64") or "")
        assert raw_b64
        # Fast path: raw crop doubles as the preview cutout.
        assert item.get("masked_image_base64") == raw_b64
        assert str(item.get("name") or "").lower() not in {"", "item", "review item"}
        assert str(item.get("category") or "") not in {"", "Needs Review"}


# ---------- save-selected: deferred RMBG cleanup ----------

class _FakeR2Upload:
    last_calls: list = []

    def upload_wardrobe_images(self, *, file_id, raw_image_bytes, masked_image_bytes):
        _FakeR2Upload.last_calls.append(
            {"file_id": file_id, "raw": raw_image_bytes, "masked": masked_image_bytes}
        )
        return {
            "raw_image_url": f"https://r2.test/{file_id}_raw.png",
            "masked_image_url": f"https://r2.test/{file_id}_masked.png",
            "normalized_image_url": f"https://r2.test/{file_id}_norm.png",
            "raw_file_name": f"{file_id}_raw.png",
            "masked_file_name": f"{file_id}_masked.png",
            "normalized_file_name": f"{file_id}_norm.png",
        }


def _gemini_preview_save_item(item_id: str, name: str = "Red Dress"):
    _, raw = _test_image()
    raw_b64 = base64.b64encode(raw).decode("utf-8")
    return {
        "item_id": item_id,
        "name": name,
        "category": "Dresses",
        "sub_category": "Dress",
        "label_source": "vision:gemini_multi",
        "preview_cutout_pending": True,
        "raw_image_base64": raw_b64,
        "masked_image_base64": raw_b64,
        "upload_error": "",
    }


def _wire_save_selected(monkeypatch, bg_fn):
    monkeypatch.setattr(wc, "remove_bg_bytes", bg_fn)
    monkeypatch.setattr(wc, "R2Storage", _FakeR2Upload)
    _FakeR2Upload.last_calls = []
    persisted = {}

    def _fake_persist(*, user_id, selected_item_ids, detected_items):
        persisted["user_id"] = user_id
        persisted["selected_item_ids"] = list(selected_item_ids)
        persisted["detected_items"] = [dict(i) for i in detected_items]
        return {"success": True, "saved_count": len(selected_item_ids), "errors": []}

    monkeypatch.setattr(wc, "persist_selected_items", _fake_persist)
    return persisted


def test_save_selected_runs_rmbg_for_gemini_multi_items(monkeypatch):
    calls = {"count": 0}
    cleaned = b"CLEANED_PNG_BYTES"

    async def _bg(data: bytes) -> bytes:
        calls["count"] += 1
        return cleaned

    persisted = _wire_save_selected(monkeypatch, _bg)

    items = [_gemini_preview_save_item("id-1"), _gemini_preview_save_item("id-2", "Blue Bag")]
    request = wc.SaveSelectedRequest(
        user_id="u1",
        selected_item_ids=["id-1", "id-2"],
        detected_items=items,
    )
    result = wc.save_selected(_FakeHttpRequest(), request)

    assert result["success"] is True
    assert calls["count"] == 2
    assert len(_FakeR2Upload.last_calls) == 2
    for call in _FakeR2Upload.last_calls:
        assert call["masked"] == cleaned

    for saved in persisted["detected_items"]:
        assert saved.get("masked_url"), saved.get("item_id")
        assert saved.get("maskedUrl") == saved.get("masked_url")
        assert saved.get("imageStatus") == "rmbg_complete"
        assert saved.get("image_status") == "rmbg_complete"
        assert saved.get("preview_cutout_pending") is False


def test_save_selected_survives_rmbg_failure(monkeypatch):
    async def _failing_bg(data: bytes) -> bytes:
        raise RuntimeError("rmbg service down")

    persisted = _wire_save_selected(monkeypatch, _failing_bg)

    items = [_gemini_preview_save_item("id-1")]
    request = wc.SaveSelectedRequest(
        user_id="u1",
        selected_item_ids=["id-1"],
        detected_items=items,
    )
    result = wc.save_selected(_FakeHttpRequest(), request)

    # Save must not fail; raw crop is uploaded as the cutout.
    assert result["success"] is True
    assert len(_FakeR2Upload.last_calls) == 1
    call = _FakeR2Upload.last_calls[0]
    assert call["masked"] == call["raw"]
    saved = persisted["detected_items"][0]
    assert saved.get("image_url")
    assert saved.get("imageStatus") == "rmbg_failed"
    assert saved.get("image_status") == "rmbg_failed"
    assert saved.get("preview_cutout_pending") is False


def test_save_selected_prefers_each_item_crop_over_shared_collage_url(monkeypatch):
    crop_by_id = {
        "dress-1": b"DRESS_CROP",
        "bag-1": b"BAG_CROP",
        "sandals-1": b"SANDALS_CROP",
        "sunglasses-1": b"SUNGLASSES_CROP",
    }
    shared_collage_url = "https://r2.test/original-collage.png"
    items = []
    for item_id, crop in crop_by_id.items():
        encoded = base64.b64encode(crop).decode("utf-8")
        items.append(
            {
                "item_id": item_id,
                "name": item_id.split("-", 1)[0].title(),
                "category": "Dresses" if item_id.startswith("dress") else "Accessories",
                "raw_image_base64": encoded,
                "masked_image_base64": encoded,
                "image_url": shared_collage_url,
                "masked_url": shared_collage_url,
            }
        )

    persisted = _wire_save_selected(monkeypatch, lambda data: data)
    request = wc.SaveSelectedRequest(
        user_id="u1",
        selected_item_ids=list(crop_by_id),
        detected_items=items,
    )

    result = wc.save_selected(_FakeHttpRequest(), request)

    assert result["success"] is True
    calls_by_id = {call["file_id"]: call for call in _FakeR2Upload.last_calls}
    assert set(calls_by_id) == set(crop_by_id)
    for item_id, crop in crop_by_id.items():
        assert calls_by_id[item_id]["raw"] == crop
        assert calls_by_id[item_id]["masked"] == crop

    saved_by_id = {
        item["item_id"]: item for item in persisted["detected_items"]
    }
    dress = saved_by_id["dress-1"]
    assert dress["masked_url"] == "https://r2.test/dress-1_masked.png"
    assert dress["image_url"] == "https://r2.test/dress-1_norm.png"
    assert dress["image_url"] != shared_collage_url
    assert dress["_save_image_source"] == "inline_crop_upload"


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
        "color_name", "pattern", "occasions", "confidence",
        "requires_manual_entry", "reasoning", "bbox", "raw_url", "rawUrl",
        "masked_url", "maskedUrl", "normalized_url", "normalizedUrl",
        "image_url", "imageUrl", "raw_image_base64", "masked_image_base64",
        "upload_error", "pixel_hash", "duplicate", "image_embedding",
    ):
        assert key in gemini_keys, key
