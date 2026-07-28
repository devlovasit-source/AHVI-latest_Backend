"""Catalogue generation must never publish an opaque (white-background) image.

The provider (Vertex Imagen / nanobanana) returns an OPAQUE studio image, and
convert("RGBA") does not remove a white background — that is what produced white
boxes in Wardrobe and Style Boards. RMBG is the enforcement step because the
model may ignore the prompt's transparency instruction. No network here.
"""
from io import BytesIO

from PIL import Image

import services.catalog_png_generation_service as cg


def _png(mode="RGBA", size=(256, 256), alpha=255, color=(255, 255, 255)):
    img = Image.new(mode, size, color + (alpha,) if mode == "RGBA" else color)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _garment_on_white():
    """Opaque white studio image with a dark garment — the provider's output."""
    img = Image.new("RGBA", (256, 256), (255, 255, 255, 255))
    for x in range(70, 190):
        for y in range(50, 210):
            img.putpixel((x, y), (40, 60, 90, 255))
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _cutout():
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    for x in range(70, 190):
        for y in range(50, 210):
            img.putpixel((x, y), (40, 60, 90, 255))
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ── forced provider reason: explicit risk signals ──────────────────────────

def test_unsafe_source_flag_forces_generation():
    assert cg._forced_provider_reason({"unsafe_source": True}, "top") == "unsafe_person_source"


def test_source_contains_person_forces_generation():
    assert cg._forced_provider_reason({"source_contains_person": True}, "top") == "unsafe_person_source"


def test_unsafe_reason_string_forces_generation():
    assert cg._forced_provider_reason({"unsafe_reason": "mirror selfie"}, "top") == "unsafe_person_source"


def test_clean_flat_lay_is_not_forced_to_the_provider():
    meta = {"crop_source": "flat_lay", "crop_quality": "good", "unsafe_source": False}
    assert cg._forced_provider_reason(meta, "top") == ""


# ── transparency enforcement ───────────────────────────────────────────────

def test_opaque_image_is_detected_as_not_transparent():
    assert cg.is_effectively_transparent(_png(alpha=255)) is False
    assert cg.is_effectively_transparent(_garment_on_white()) is False


def test_real_cutout_is_detected_as_transparent():
    assert cg.is_effectively_transparent(_cutout()) is True


def test_opaque_provider_output_is_sent_through_rmbg(monkeypatch):
    calls = []

    def _fake_rmbg(data):
        calls.append(len(data))
        return _cutout()

    import services.bg_service as bg
    monkeypatch.setattr(bg, "remove_bg_external_sync", _fake_rmbg)
    out, reason = cg._provider_output_to_transparent(_garment_on_white(), "top", "item-1")
    assert calls, "RMBG must be invoked for an opaque provider output"
    assert reason == "ok"
    assert cg.is_effectively_transparent(out) is True


def test_already_transparent_provider_output_skips_rmbg(monkeypatch):
    calls = []

    def _fake_rmbg(data):
        calls.append(1)
        return data

    import services.bg_service as bg
    monkeypatch.setattr(bg, "remove_bg_external_sync", _fake_rmbg)
    out, reason = cg._provider_output_to_transparent(_cutout(), "top", "item-2")
    assert calls == []
    assert reason == "ok"
    assert cg.is_effectively_transparent(out) is True


def test_rmbg_failure_returns_empty_so_caller_keeps_original(monkeypatch):
    def _boom(data):
        raise RuntimeError("rmbg down")

    import services.bg_service as bg
    monkeypatch.setattr(bg, "remove_bg_external_sync", _boom)
    out, reason = cg._provider_output_to_transparent(_garment_on_white(), "top", "item-3")
    assert out == b""
    assert reason == "rmbg_failed"


def test_rmbg_returning_opaque_is_rejected(monkeypatch):
    # bg_service fails OPEN — a still-opaque result must never be published.
    import services.bg_service as bg
    monkeypatch.setattr(bg, "remove_bg_external_sync", lambda d: _garment_on_white())
    out, reason = cg._provider_output_to_transparent(_garment_on_white(), "top", "item-4")
    assert out == b""
    assert reason in {"still_opaque", "canvas_failed"}


# ── prompt no longer asks for a white background ───────────────────────────

def test_catalog_prompt_requests_transparency_not_white():
    prompt = cg._build_catalog_prompt("top", {})
    lowered = prompt.lower()
    assert "transparent background" in lowered
    assert "pure white studio background only" not in lowered
