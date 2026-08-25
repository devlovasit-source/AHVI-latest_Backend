"""AHVI Gemini bbox + RMBG multi-garment preview MVP.

Covers:
- Gemini multi-garment detection + crop (valid JSON -> bbox/crop fields)
- invalid JSON / disabled detector -> existing flow runs unchanged
- deterministic taxonomy (saree) and suitability (boxers/private) guards are
  still applied to Gemini-detected items
- the preview item schema is unchanged regardless of detection path
"""

from __future__ import annotations

from fastapi import BackgroundTasks

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


def _jpeg_b64_with_orientation(w=120, h=40, orientation=6) -> str:
    img = Image.new("RGB", (w, h), (210, 210, 210))
    exif = img.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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

ONE_SHORTS_RESULT = """
[
  {"name": "Black Shorts", "category": "Bottoms", "sub_category": "Shorts", "color": "Black", "confidence": 0.88, "bbox": [0.25, 0.45, 0.75, 0.72]}
]
"""


# ---------- detector parser: Gemini JSON-ish response shapes ----------

def test_gemini_multi_parser_accepts_object_with_items():
    raw = '{"items":[{"name":"Kurta","category":"Tops","bbox":[0.1,0.2,0.3,0.4],"confidence":0.9}]}'
    items, mode, error = gmg._parse_gemini_items(raw)

    assert mode == "object"
    assert error == ""
    assert len(items) == 1
    assert items[0]["name"] == "Kurta"


def test_gemini_multi_parser_accepts_bare_array():
    raw = '[{"name":"Kurta","category":"Tops","bbox":[0.1,0.2,0.3,0.4],"confidence":0.9}]'
    items, mode, error = gmg._parse_gemini_items(raw)

    assert mode == "array"
    assert error == ""
    assert len(items) == 1
    assert items[0]["name"] == "Kurta"


def test_gemini_multi_parser_accepts_markdown_fenced_json():
    raw = '```json\n{"items":[{"name":"Kurta","category":"Tops","bbox":[0.1,0.2,0.3,0.4],"confidence":0.9}]}\n```'
    items, mode, error = gmg._parse_gemini_items(raw)

    assert mode == "fenced"
    assert error == ""
    assert len(items) == 1
    assert items[0]["name"] == "Kurta"


def test_gemini_multi_parser_accepts_extra_text_around_json():
    raw = 'Here is the JSON:\n{"items":[{"name":"Kurta","category":"Tops","bbox":[0.1,0.2,0.3,0.4],"confidence":0.9}]}\nDone.'
    items, mode, error = gmg._parse_gemini_items(raw)

    assert mode == "repaired"
    assert error == ""
    assert len(items) == 1
    assert items[0]["name"] == "Kurta"


def test_gemini_multi_parser_accepts_single_item_object():
    raw = '{"name":"Kurta","category":"Tops","bbox":[0.1,0.2,0.3,0.4],"confidence":0.9}'
    items, mode, error = gmg._parse_gemini_items(raw)

    assert mode == "object"
    assert error == ""
    assert len(items) == 1
    assert items[0]["name"] == "Kurta"


def test_gemini_multi_parser_truncated_array_has_clear_reason():
    items, mode, error = gmg._parse_gemini_items("[")

    assert items == []
    assert mode == "failed"
    assert error == "invalid_json:truncated_or_empty"


def test_gemini_multi_parser_invalid_prose_has_invalid_json_reason():
    items, mode, error = gmg._parse_gemini_items("not json at all")

    assert items == []
    assert mode == "failed"
    assert error.startswith("invalid_json:")


def test_gemini_multi_parser_always_returns_items_list():
    items, _, error = gmg._parse_gemini_items('{"items":[]}')

    assert isinstance(items, list)
    assert error == ""


def test_gemini_multi_seed_is_valid_signed_int32():
    seed = gmg._stable_int32_seed(b"\xff" * 128)

    assert 1 <= seed <= 2_147_483_647


def test_gemini_multi_seed_is_never_zero_or_negative():
    seed = gmg._stable_int32_seed(b"")

    assert seed > 0


def test_gemini_multi_seed_invalid_argument_retries_without_seed(monkeypatch):
    calls = []

    class _FakePart:
        @staticmethod
        def from_bytes(data, mime_type):
            return {"data": data, "mime_type": mime_type}

    class _FakeConfig(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)

    class _FakeTypes:
        Part = _FakePart
        GenerateContentConfig = _FakeConfig

    class _FakeResponse:
        text = '{"items":[]}'

    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append(dict(config))
            if len(calls) == 1:
                raise Exception(
                    "ClientError 400 INVALID_ARGUMENT Invalid value at "
                    "'generation_config.seed' (TYPE_INT32), 3898845386"
                )
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setattr(gmg, "types", _FakeTypes)
    monkeypatch.setattr(gmg, "_get_gemini_client", lambda: _FakeClient())

    result = gmg._call_gemini_vision(b"image-bytes", request_id="seed-test")

    assert result == '{"items":[]}'
    assert len(calls) == 2
    assert "seed" in calls[0]
    assert 1 <= calls[0]["seed"] <= 2_147_483_647
    assert "seed" not in calls[1]


class _FakePart:
    @staticmethod
    def from_bytes(data, mime_type):
        return {"data": data, "mime_type": mime_type}


class _FakeConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)


class _FakeTypes:
    Part = _FakePart
    GenerateContentConfig = _FakeConfig


def test_gemini_multi_live_config_has_no_response_schema(monkeypatch):
    """Primary live call must NOT pass response_schema (SDK crashes on it)."""
    calls = []

    class _FakeResponse:
        text = '{"items":[]}'

    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append(dict(config))
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.delenv("GEMINI_MULTI_GARMENT_USE_SCHEMA", raising=False)
    monkeypatch.setattr(gmg, "types", _FakeTypes)
    monkeypatch.setattr(gmg, "_get_gemini_client", lambda: _FakeClient())

    result = gmg._call_gemini_vision(b"image-bytes", request_id="no-schema")

    assert result == '{"items":[]}'
    assert len(calls) == 1
    assert "response_schema" not in calls[0]
    assert calls[0]["response_mime_type"] == "application/json"
    assert calls[0]["temperature"] == 0
    assert calls[0].get("candidate_count") == 1


def test_gemini_multi_disables_thinking_and_raises_token_budget(monkeypatch):
    """gemini-2.5-flash is a thinking model; reasoning tokens were truncating
    the JSON (MAX_TOKENS). Live config must raise max_output_tokens and disable
    thinking so the whole budget goes to the response."""
    calls = []

    class _FakeThinking:
        def __init__(self, thinking_budget):
            self.thinking_budget = thinking_budget

    class _FakeTypesThinking:
        Part = _FakePart
        GenerateContentConfig = _FakeConfig
        ThinkingConfig = _FakeThinking

    class _FakeResponse:
        text = '{"items":[]}'

    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append(dict(config))
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.delenv("GEMINI_MULTI_GARMENT_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setattr(gmg, "types", _FakeTypesThinking)
    monkeypatch.setattr(gmg, "_get_gemini_client", lambda: _FakeClient())

    result = gmg._call_gemini_vision(b"image-bytes", request_id="thinking")

    assert result == '{"items":[]}'
    assert len(calls) == 1
    assert calls[0]["max_output_tokens"] >= 4096
    assert "thinking_config" in calls[0]
    assert calls[0]["thinking_config"].thinking_budget == 0


def test_gemini_multi_schema_error_retries_without_schema(monkeypatch):
    """If response_schema is ever present and the SDK fails converting it
    (AttributeError: 'list' object has no attribute 'upper'), retry once with
    the schema stripped."""
    calls = []

    class _FakeResponse:
        text = '{"items":[]}'

    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append(dict(config))
            if len(calls) == 1:
                raise AttributeError("'list' object has no attribute 'upper'")
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setenv("GEMINI_MULTI_GARMENT_USE_SCHEMA", "true")
    monkeypatch.setattr(gmg, "types", _FakeTypes)
    monkeypatch.setattr(gmg, "_get_gemini_client", lambda: _FakeClient())

    result = gmg._call_gemini_vision(b"image-bytes", request_id="schema-retry")

    assert result == '{"items":[]}'
    assert len(calls) == 2
    assert "response_schema" in calls[0]
    assert "response_schema" not in calls[1]


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


def test_gemini_single_valid_item_is_accepted_for_analyze(monkeypatch):
    _wire_router_for_gemini(monkeypatch, ONE_SHORTS_RESULT)

    http_request = _FakeHttpRequest()
    result = _run(wc.analyze_capture(http_request, _capture_request()))

    assert result["stage_trace"]["detection"] == "gemini_single_garment"
    assert result["count"] == 1
    item = result["items"][0]
    assert item["name"] == "Black Shorts"
    assert item["category"] == "Bottoms"
    assert item["sub_category"] == "Shorts"
    assert item["crop_source"] == "gemini"
    assert item["crop_quality"] == "tight"


def test_full_image_fallback_shorts_marked_person_risk():
    item = wc._normalize_capture_preview_item(
        {
            "item_id": "shorts-risk",
            "name": "Black Shorts",
            "category": "Bottoms",
            "sub_category": "Shorts",
            "confidence": 0.91,
            "label_source": "vision",
            "crop_source": "full_image_fallback",
            "crop_quality": "full_image",
        }
    )

    assert item["category"] == "Bottoms"
    assert item["sub_category"] == "Shorts"
    assert item["crop_quality"] == "full_image_person_risk"
    assert item["needs_review"] is True
    assert item["requires_manual_entry"] is True
    assert item["validation_status"] == "needs_review"
    assert item["selected_by_default"] is False
    assert item["rejection_reason"] in {"detector_fallback_full_image", "partial_bottomwear_visible"}


# ---------- P0: safe single-garment fallback is saveable ----------

def _fallback_item(**overrides):
    base = {
        "item_id": "fallback-1",
        "crop_source": "full_image_fallback",
        "crop_quality": "full_image",
        "confidence": 0.9,
        "label_source": "vision",
    }
    base.update(overrides)
    return wc._normalize_capture_preview_item(base)


def test_full_image_fallback_saree_is_saveable():
    item = _fallback_item(
        name="Saree",
        category="Dresses",
        sub_category="Saree",
    )
    assert item["validation_status"] == "ok"
    assert item["selected_by_default"] is True
    assert not item["rejection_reason"]
    assert item["needs_review"] is False
    assert item["requires_manual_entry"] is False


def test_full_image_fallback_teal_dress_is_saveable():
    item = _fallback_item(
        name="Teal Floral Dress",
        category="Dresses",
        sub_category="One-piece Dress",
    )
    assert item["validation_status"] == "ok"
    assert item["selected_by_default"] is True
    assert item["needs_review"] is False


def test_full_image_fallback_accessory_remains_blocked():
    for name, sub in (
        ("Gold Necklace", "Necklace"),
        ("Silver Bracelet", "Bracelet"),
        ("Steel Watch", "Watch"),
        ("Diamond Ring", "Ring"),
    ):
        item = _fallback_item(
            item_id=f"acc-{name}",
            name=name,
            category="Accessories",
            sub_category=sub,
            confidence=0.6,
        )
        assert item["validation_status"] != "ok", name
        assert item["selected_by_default"] is False, name


def test_full_image_fallback_privatewear_remains_blocked():
    for name in ("Cotton Underwear", "White Briefs", "Blue Boxers", "Lace Lingerie"):
        item = _fallback_item(
            item_id=f"pw-{name}",
            name=name,
            category="Innerwear",
            sub_category="Underwear",
        )
        assert item["validation_status"] != "ok", name
        assert item["selected_by_default"] is False, name


def test_full_image_fallback_person_skin_risk_remains_needs_review():
    item = _fallback_item(
        name="Black Shorts",
        category="Bottoms",
        sub_category="Shorts",
        confidence=0.91,
    )
    assert item["validation_status"] in {"needs_review", "rejected"}
    assert item["selected_by_default"] is False


def test_weak_necklace_crop_is_rejected():
    item = wc._normalize_capture_preview_item(
        {
            "item_id": "neck-risk",
            "name": "Necklace",
            "category": "Accessories",
            "sub_category": "Necklace",
            "confidence": 0.41,
            "bbox": [0.46, 0.18, 0.51, 0.23],
            "label_source": "vision:gemini_multi",
            "crop_source": "gemini",
            "crop_quality": "tight",
        }
    )

    assert item["validation_status"] == "rejected"
    assert item["selected_by_default"] is False
    assert item["rejection_reason"] in {
        "accessory_low_confidence",
        "accessory_bbox_too_small",
    }


def test_partial_trousers_crop_needs_review():
    item = wc._normalize_capture_preview_item(
        {
            "item_id": "trouser-risk",
            "name": "Olive Trousers",
            "category": "Bottoms",
            "sub_category": "Trousers",
            "confidence": 0.86,
            "bbox": [0.25, 0.54, 0.76, 0.62],
            "label_source": "vision:gemini_multi",
            "crop_source": "gemini",
            "crop_quality": "tight",
            "reasoning": "partial waistband crop",
        }
    )

    assert item["validation_status"] == "needs_review"
    assert item["selected_by_default"] is False
    assert item["rejection_reason"] == "partial_bottomwear_visible"


def test_approved_shirt_is_selected_by_default():
    item = wc._normalize_capture_preview_item(
        {
            "item_id": "shirt-ok",
            "name": "Sage Green Casual Shirt",
            "category": "Tops",
            "sub_category": "Shirt",
            "confidence": 0.91,
            "bbox": [0.20, 0.12, 0.80, 0.50],
            "label_source": "vision:gemini_multi",
            "crop_source": "gemini",
            "crop_quality": "tight",
        }
    )

    assert item["validation_status"] == "ok"
    assert item["selected_by_default"] is True


def test_gemini_needs_review_reason_survives_preview(monkeypatch):
    needs_review_json = """
    [
      {"name": "Olive Trousers", "category": "Bottoms", "sub_category": "Trousers", "color": "Olive", "confidence": 0.86, "bbox": [0.20, 0.52, 0.80, 0.82], "needs_review": true, "reason": "partial_bottomwear_visible"}
    ]
    """
    _wire_router_for_gemini(monkeypatch, needs_review_json)

    result = _run(wc.analyze_capture(_FakeHttpRequest(), _capture_request()))

    item = result["items"][0]
    assert item["validation_status"] == "needs_review"
    assert item["selected_by_default"] is False
    assert item["rejection_reason"] == "partial_bottomwear_visible"
    assert result["stage_trace"]["needs_review_count"] == 1


def test_analyze_sends_corrected_orientation_bytes_to_gemini(monkeypatch):
    captured = {}

    def _capture_payload(image_bytes, request_id=""):
        img = Image.open(io.BytesIO(image_bytes))
        captured["payload_size"] = img.size
        return ONE_SHORTS_RESULT

    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(gmg, "_call_gemini_vision", _capture_payload)
    monkeypatch.setattr(wc, "remove_bg_bytes", _passthrough_bg)
    monkeypatch.setattr(wc, "_find_upload_duplicate", _no_duplicate)

    http_request = _FakeHttpRequest()
    result = _run(
        wc.analyze_capture(
            http_request,
            _capture_request(image_base64=_jpeg_b64_with_orientation()),
        )
    )

    assert captured["payload_size"] == (40, 120)
    assert result["stage_trace"]["detection"] == "gemini_single_garment"


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
    result = wc.save_selected(_FakeHttpRequest(), request, BackgroundTasks())

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
    result = wc.save_selected(_FakeHttpRequest(), request, BackgroundTasks())

    # Save must not fail, but the raw crop must not be published as a cutout.
    assert result["success"] is True
    assert _FakeR2Upload.last_calls == []
    saved = persisted["detected_items"][0]
    assert not saved.get("masked_url")
    assert not saved.get("masked_image_base64")
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

    result = wc.save_selected(_FakeHttpRequest(), request, BackgroundTasks())

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
        "validation_status", "rejection_reason", "selected_by_default",
        "crop_quality_score", "detection_mode", "regen_provider", "input_type",
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
        "validation_status", "rejection_reason", "selected_by_default",
        "crop_quality_score", "detection_mode", "regen_provider", "input_type",
    ):
        assert key in gemini_keys, key


# ── save_selected must not report success when nothing was saved ───────────

async def _passthrough_bg(data: bytes) -> bytes:
    return data


def test_save_selected_reports_failure_when_nothing_is_saved(monkeypatch):
    """requested>0 and saved==0 previously still returned success=True, so the
    app showed 'added' for items that were never persisted."""
    _wire_save_selected(monkeypatch, _passthrough_bg)

    def _persist_nothing(*, user_id, selected_item_ids, detected_items):
        return {"success": True, "saved_count": 0, "errors": []}

    monkeypatch.setattr(wc, "persist_selected_items", _persist_nothing)

    items = [_gemini_preview_save_item("id-1")]
    request = wc.SaveSelectedRequest(
        user_id="u1", selected_item_ids=["id-1"], detected_items=items
    )
    result = wc.save_selected(_FakeHttpRequest(), request, BackgroundTasks())

    assert result["requested_count"] == 1
    assert result["saved_count"] == 0
    assert result["success"] is False
    assert result.get("retryable") is True
    assert "could not be saved" in str(result.get("message", "")).lower()


# ---------- P0: name quality + pattern contract (wardrobe vision name/pattern fix) ----------

PATTERNED_MULTI_RESULT = """
[
  {"name": "Navy Blue Striped Shirt", "category": "Tops", "sub_category": "Shirt", "color": "Navy Blue", "pattern": "striped", "confidence": 0.9, "bbox": [0.05, 0.05, 0.45, 0.5]},
  {"name": "Beige Pleated Trousers", "category": "Bottoms", "sub_category": "Trousers", "color": "Beige", "pattern": "plain", "confidence": 0.88, "bbox": [0.1, 0.5, 0.5, 0.95]},
  {"name": "Black Leather Loafers", "category": "Footwear", "sub_category": "Loafers", "color": "Black", "pattern": "plain", "confidence": 0.85, "bbox": [0.55, 0.8, 0.9, 0.98]},
  {"name": "Red Floral Midi Dress", "category": "Dresses", "sub_category": "Midi Dress", "color": "Red", "pattern": "floral", "confidence": 0.92, "bbox": [0.5, 0.05, 0.9, 0.5]}
]
"""

NO_PATTERN_FIELD_RESULT = """
[
  {"name": "Grey Wool Coat", "category": "Outerwear", "sub_category": "Coat", "color": "Grey", "confidence": 0.9, "bbox": [0.1, 0.1, 0.5, 0.7]}
]
"""

NULL_PATTERN_RESULT = """
[
  {"name": "White Cotton Shirt", "category": "Tops", "sub_category": "Shirt", "color": "White", "pattern": null, "confidence": 0.9, "bbox": [0.1, 0.1, 0.5, 0.7]}
]
"""

UNKNOWN_PATTERN_VALUE_RESULT = """
[
  {"name": "Green Paisley Scarf", "category": "Accessories", "sub_category": "Scarf", "color": "Green", "pattern": "Paisley Print", "confidence": 0.8, "bbox": [0.1, 0.1, 0.3, 0.3]}
]
"""

SOLID_PATTERN_RESULT = """
[
  {"name": "Black T-Shirt", "category": "Tops", "sub_category": "T-Shirt", "color": "Black", "pattern": "solid", "confidence": 0.9, "bbox": [0.1, 0.1, 0.5, 0.7]}
]
"""


def test_gemini_multi_prompt_forbids_bare_category_names():
    prompt = gmg._build_prompt()
    lowered = prompt.lower()
    assert "bare category" in lowered or "bare taxonomy" in lowered
    assert "color + pattern" in lowered or "color+pattern" in lowered.replace(" ", "")
    # The naming rule must not demand invented detail.
    assert "do not invent" in lowered


def test_gemini_multi_prompt_requests_pattern_from_canonical_list():
    prompt = gmg._build_prompt()
    assert '"pattern"' in prompt
    for value in gmg.PATTERN_VALUES:
        assert value in prompt


def test_gemini_multi_schema_requests_pattern_field():
    item_schema = gmg._ITEM_SCHEMA["properties"]["items"]["items"]
    assert item_schema["properties"]["pattern"] == {"type": "string"}
    # Pattern stays optional/best-effort - never required, so a Gemini
    # response omitting it is still valid and must not be rejected wholesale.
    assert "pattern" not in item_schema["required"]


def test_gemini_multi_pattern_survives_detector_parsing(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": PATTERNED_MULTI_RESULT
    )
    image, raw = _test_image()
    items = _run(gmg.detect_and_crop(image, raw, request_id="pattern1"))

    assert len(items) == 4
    by_name = {i["name"]: i["pattern"] for i in items}
    assert by_name["Navy Blue Striped Shirt"] == "striped"
    assert by_name["Beige Pleated Trousers"] == "plain"
    assert by_name["Black Leather Loafers"] == "plain"
    assert by_name["Red Floral Midi Dress"] == "floral"


def test_gemini_multi_missing_pattern_field_defaults_to_plain(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": NO_PATTERN_FIELD_RESULT
    )
    image, raw = _test_image()
    items = _run(gmg.detect_and_crop(image, raw, request_id="pattern2"))

    assert len(items) == 1
    assert items[0]["pattern"] == "plain"


def test_gemini_multi_null_pattern_defaults_to_plain(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": NULL_PATTERN_RESULT
    )
    image, raw = _test_image()
    items = _run(gmg.detect_and_crop(image, raw, request_id="pattern3"))

    assert len(items) == 1
    assert items[0]["pattern"] == "plain"


def test_gemini_multi_solid_pattern_normalizes_to_plain(monkeypatch):
    # services/category_taxonomy.py already treats "plain" and "solid" as
    # equivalent no-pattern signals - normalize at the source so only one
    # canonical value is ever persisted.
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": SOLID_PATTERN_RESULT
    )
    image, raw = _test_image()
    items = _run(gmg.detect_and_crop(image, raw, request_id="pattern4"))

    assert items[0]["pattern"] == "plain"


def test_gemini_multi_unrecognized_pattern_value_does_not_crash(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": UNKNOWN_PATTERN_VALUE_RESULT
    )
    image, raw = _test_image()
    items = _run(gmg.detect_and_crop(image, raw, request_id="pattern5"))

    assert len(items) == 1
    # Not one of the canonical values, but safely normalized (lowercased,
    # underscored, non-empty) rather than dropped or raising.
    assert items[0]["pattern"] == "paisley_print"


def test_normalize_pattern_helper_handles_all_edge_cases():
    assert gmg._normalize_pattern(None) == "plain"
    assert gmg._normalize_pattern("") == "plain"
    assert gmg._normalize_pattern("null") == "plain"
    assert gmg._normalize_pattern("unknown") == "plain"
    assert gmg._normalize_pattern("SOLID") == "plain"
    assert gmg._normalize_pattern("Striped") == "striped"
    assert gmg._normalize_pattern("animal_print") == "animal_print"
    assert gmg._normalize_pattern("Something Odd") == "something_odd"


def test_gemini_multi_pattern_reaches_preview_response(monkeypatch):
    """End-to-end: trusted-Gemini pattern must reach the preview item, not
    the previously hardcoded 'plain' regardless of what Gemini returned."""
    _wire_router_for_gemini(monkeypatch, PATTERNED_MULTI_RESULT)

    http_request = _FakeHttpRequest()
    result = _run(wc.analyze_capture(http_request, _capture_request()))

    assert result["stage_trace"]["detection"] == "gemini_multi_garment"
    by_name = {i["name"]: i["pattern"] for i in result["items"]}
    assert by_name["Navy Blue Striped Shirt"] == "striped"
    assert by_name["Red Floral Midi Dress"] == "floral"
    assert by_name["Beige Pleated Trousers"] == "plain"


def test_gemini_multi_descriptive_names_survive_end_to_end(monkeypatch):
    _wire_router_for_gemini(monkeypatch, PATTERNED_MULTI_RESULT)

    http_request = _FakeHttpRequest()
    result = _run(wc.analyze_capture(http_request, _capture_request()))

    names = {str(i.get("name") or "") for i in result["items"]}
    assert names == {
        "Navy Blue Striped Shirt", "Beige Pleated Trousers",
        "Black Leather Loafers", "Red Floral Midi Dress",
    }
    # None of the names degraded to a bare taxonomy word.
    bare_words = {"shirt", "trousers", "footwear", "dress", "item"}
    for name in names:
        assert name.lower() not in bare_words


def test_gemini_multi_existing_category_subcategory_parsing_unchanged(monkeypatch):
    # Control: the pattern/name work must not touch category/sub_category
    # parsing at all - reuses the pre-existing GOOD_RESULT fixture/assertions.
    _wire_router_for_gemini(monkeypatch, GOOD_RESULT)

    http_request = _FakeHttpRequest()
    result = _run(wc.analyze_capture(http_request, _capture_request()))

    categories = {str(i.get("category") or "") for i in result["items"]}
    assert {"Dresses", "Bags", "Accessories", "Footwear"} <= categories


def test_gemini_single_garment_path_unaffected_by_pattern_field(monkeypatch):
    # ONE_SHORTS_RESULT has no "pattern" key at all - single-garment Gemini
    # path must keep working exactly as before.
    _wire_router_for_gemini(monkeypatch, ONE_SHORTS_RESULT)

    http_request = _FakeHttpRequest()
    result = _run(wc.analyze_capture(http_request, _capture_request()))

    assert result["stage_trace"]["detection"] == "gemini_single_garment"
    assert result["count"] == 1
    item = result["items"][0]
    assert item["name"] == "Black Shorts"
    assert item["category"] == "Bottoms"
    assert item["pattern"] == "plain"


def test_trusted_gemini_pattern_does_not_force_ollama_enrichment(monkeypatch):
    """Regression guard for the strict-scope rule: obtaining pattern for a
    trusted Gemini item must never trigger the Ollama single-garment
    enrichment call."""
    calls = {"count": 0}

    def _tracking_vision_extract(*args, **kwargs):
        calls["count"] += 1
        return {
            "name": "Item", "category": "", "sub_category": "", "pattern": "plain",
            "color_name": "", "occasions": [], "label_source": "heuristic",
            "requires_manual_entry": True,
        }

    monkeypatch.setattr(wc, "_vision_extract_attributes", _tracking_vision_extract)
    _wire_router_for_gemini(monkeypatch, PATTERNED_MULTI_RESULT)

    http_request = _FakeHttpRequest()
    result = _run(wc.analyze_capture(http_request, _capture_request()))

    assert result["stage_trace"]["detection"] == "gemini_multi_garment"
    assert calls["count"] == 0, "trusted Gemini items must not call Ollama enrichment for pattern"
    by_name = {i["name"]: i["pattern"] for i in result["items"]}
    assert by_name["Navy Blue Striped Shirt"] == "striped"


def test_gemini_multi_json_parsing_robust_with_mixed_pattern_presence(monkeypatch):
    """Some items have pattern, some don't, one is null - parser must not
    crash and must handle each independently."""
    mixed_json = """
    [
      {"name": "Blue Striped Shirt", "category": "Tops", "sub_category": "Shirt", "color": "Blue", "pattern": "striped", "confidence": 0.9, "bbox": [0.05, 0.05, 0.4, 0.5]},
      {"name": "Grey Trousers", "category": "Bottoms", "sub_category": "Trousers", "color": "Grey", "confidence": 0.85, "bbox": [0.1, 0.5, 0.5, 0.9]},
      {"name": "White Sneakers", "category": "Footwear", "sub_category": "Sneakers", "color": "White", "pattern": null, "confidence": 0.8, "bbox": [0.5, 0.7, 0.9, 0.98]}
    ]
    """
    monkeypatch.setenv("ENABLE_GEMINI_MULTI_GARMENT_PREVIEW", "true")
    monkeypatch.setattr(
        gmg, "_call_gemini_vision", lambda image_bytes, request_id="": mixed_json
    )
    image, raw = _test_image()
    items = _run(gmg.detect_and_crop(image, raw, request_id="pattern-mixed"))

    assert len(items) == 3
    by_name = {i["name"]: i["pattern"] for i in items}
    assert by_name["Blue Striped Shirt"] == "striped"
    assert by_name["Grey Trousers"] == "plain"
    assert by_name["White Sneakers"] == "plain"


def test_save_selected_reports_partial_success_when_some_dropped(monkeypatch):
    _wire_save_selected(monkeypatch, _passthrough_bg)

    def _persist_one(*, user_id, selected_item_ids, detected_items):
        return {"success": True, "saved_count": 1, "errors": []}

    monkeypatch.setattr(wc, "persist_selected_items", _persist_one)

    items = [_gemini_preview_save_item("id-1"), _gemini_preview_save_item("id-2", "Blue Bag")]
    request = wc.SaveSelectedRequest(
        user_id="u1", selected_item_ids=["id-1", "id-2"], detected_items=items
    )
    result = wc.save_selected(_FakeHttpRequest(), request, BackgroundTasks())

    assert result["requested_count"] == 2
    assert result["saved_count"] == 1
    assert result["dropped_count"] == 1
    assert result.get("partial_success") is True
    assert "1 of 2" in str(result.get("message", ""))
