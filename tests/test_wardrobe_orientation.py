"""EXIF orientation normalization at the capture decode entry.

_decode_image_base64 must apply ImageOps.exif_transpose BEFORE downstream
(Gemini/crop/RMBG/catalog), so camera photos with an orientation tag are upright.
"""

import base64
import io
import logging

from PIL import Image

from routers import wardrobe_capture as wc


def _jpeg_with_orientation(w, h, orientation=None, color=(200, 60, 60)):
    im = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    if orientation is not None:
        exif = im.getexif()
        exif[0x0112] = orientation  # 0x0112 = Orientation tag
        im.save(buf, "JPEG", exif=exif)
    else:
        im.save(buf, "JPEG")
    return buf.getvalue()


def _b64(data):
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def test_orientation_6_is_transposed():
    # Orientation=6 -> 90deg; a 120x40 stored image should become 40x120 upright.
    img = wc._decode_image_base64(_b64(_jpeg_with_orientation(120, 40, orientation=6)))
    assert img.size == (40, 120)
    assert img.mode == "RGB"


def test_orientation_8_is_transposed():
    img = wc._decode_image_base64(_b64(_jpeg_with_orientation(120, 40, orientation=8)))
    assert img.size == (40, 120)


def test_no_exif_unchanged(caplog):
    with caplog.at_level(logging.INFO):
        img = wc._decode_image_base64(_b64(_jpeg_with_orientation(120, 40, orientation=None)))
    assert img.size == (120, 40)  # untouched
    assert "ahvi.image.orientation.skipped" in "\n".join(caplog.messages)


def test_orientation_applied_logged(caplog):
    with caplog.at_level(logging.INFO):
        wc._decode_image_base64(_b64(_jpeg_with_orientation(120, 40, orientation=6)))
    assert "ahvi.image.orientation.applied" in "\n".join(caplog.messages)


def test_idempotent_no_double_rotation():
    # Decode once (transposed to 40x120), re-encode, decode again -> stable.
    first = wc._decode_image_base64(_b64(_jpeg_with_orientation(120, 40, orientation=6)))
    assert first.size == (40, 120)
    buf = io.BytesIO()
    first.save(buf, "JPEG")  # re-encoded, no orientation tag now
    second = wc._decode_image_base64(_b64(buf.getvalue()))
    assert second.size == (40, 120)  # unchanged, no further rotation


def test_full_image_fallback_uses_corrected_bytes(monkeypatch):
    class _State:
        user = {"user_id": "u1"}
        request_id = "orientation-fallback"

    class _Request:
        state = _State()

    monkeypatch.setattr(wc._gemini_multi, "is_enabled", lambda: False)
    monkeypatch.setenv("WARDROBE_CAPTURE_SINGLE_GARMENT_MODE", "true")
    monkeypatch.setattr(
        wc,
        "_find_upload_duplicate",
        lambda **_kwargs: wc._duplicate_result(checked=False, is_duplicate=False),
    )

    request = wc.CaptureAnalyzeRequest(
        user_id="u1",
        image_base64=_b64(_jpeg_with_orientation(120, 40, orientation=6)),
        auto_save=False,
        save_duplicates=False,
    )
    result = __import__("asyncio").run(wc.analyze_capture(_Request(), request))

    raw = result["items"][0]["raw_image_base64"].split(",", 1)[1]
    decoded = Image.open(io.BytesIO(base64.b64decode(raw)))
    assert decoded.size == (40, 120)
    assert result["items"][0]["orientation_corrected"] is True
