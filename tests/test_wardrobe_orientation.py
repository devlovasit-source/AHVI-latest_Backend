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
