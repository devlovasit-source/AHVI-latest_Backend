from __future__ import annotations

import io
from typing import Tuple

from PIL import Image, ImageOps


TRANSPARENT: Tuple[int, int, int, int] = (255, 255, 255, 0)
WHITE: Tuple[int, int, int, int] = (255, 255, 255, 255)


def _open_rgba(image_bytes: bytes) -> Image.Image:
    """
    Decode image bytes into an RGBA PIL image and respect EXIF orientation.
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGBA")
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes: {exc}") from exc


def _trim_transparent_bounds(img: Image.Image) -> Image.Image:
    """
    Trim empty transparent pixels around an RGBA image.

    This is useful after background removal because the cutout may contain
    large transparent margins. If there is no visible alpha content, the
    original image is returned.
    """
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def _trim_near_white_bounds(
    img: Image.Image,
    *,
    threshold: int = 248,
) -> Image.Image:
    """
    Best-effort trim for non-transparent images with plain white backgrounds.

    This is intentionally conservative. It helps if the background remover was
    not used and the object sits on a near-white photo background. It does not
    remove the background; it only trims obvious outer whitespace.
    """
    img = img.convert("RGBA")
    pixels = img.load()
    width, height = img.size

    min_x, min_y = width, height
    max_x, max_y = -1, -1

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            is_near_white = r >= threshold and g >= threshold and b >= threshold
            if not is_near_white:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if max_x < min_x or max_y < min_y:
        return img

    # Avoid over-cropping tiny noise. Only crop if the detected region is sane.
    detected_w = max_x - min_x + 1
    detected_h = max_y - min_y + 1
    if detected_w < width * 0.08 or detected_h < height * 0.08:
        return img

    return img.crop((min_x, min_y, max_x + 1, max_y + 1))


def _resize_inside_canvas(
    img: Image.Image,
    *,
    canvas_size: Tuple[int, int],
    object_fill: float,
    background: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Center an image inside a target canvas without cropping.
    """
    if not 0.1 <= object_fill <= 1.0:
        raise ValueError("object_fill must be between 0.1 and 1.0")

    target_w, target_h = canvas_size
    max_object_w = int(target_w * object_fill)
    max_object_h = int(target_h * object_fill)

    img = img.convert("RGBA")
    img.thumbnail((max_object_w, max_object_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (target_w, target_h), background)
    x = (target_w - img.width) // 2
    y = (target_h - img.height) // 2
    canvas.alpha_composite(img, (x, y))
    return canvas


def normalize_wardrobe_image_bytes(
    image_bytes: bytes,
    *,
    canvas_size: int = 1024,
    object_fill: float = 0.82,
    output_format: str = "PNG",
    trim_near_white: bool = False,
) -> bytes:
    """
    Normalize a wardrobe item image for AHVI style boards.

    Output standard:
    - square transparent canvas
    - centered product/item
    - consistent padding
    - no crop
    - PNG by default

    Recommended input:
    - background-removed/masked PNG bytes when available.

    Args:
        image_bytes: Raw or masked image bytes.
        canvas_size: Final square canvas size. Default: 1024.
        object_fill: Max object size relative to canvas. Default: 0.82.
        output_format: PNG recommended for transparency.
        trim_near_white: Optional conservative whitespace trim for white-background
            images. Keep False for masked PNGs.

    Returns:
        Encoded image bytes.
    """
    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")

    img = _open_rgba(image_bytes)
    img = _trim_transparent_bounds(img)

    if trim_near_white:
        img = _trim_near_white_bounds(img)

    canvas = _resize_inside_canvas(
        img,
        canvas_size=(canvas_size, canvas_size),
        object_fill=object_fill,
        background=TRANSPARENT,
    )

    out = io.BytesIO()
    fmt = output_format.upper()

    if fmt in {"JPG", "JPEG"}:
        # JPEG does not support alpha; use white background.
        canvas = _flatten_rgba(canvas, WHITE)
        canvas.save(out, format="JPEG", quality=94, optimize=True)
    else:
        canvas.save(out, format=fmt, optimize=True)

    return out.getvalue()


def normalize_style_board_image_bytes(
    image_bytes: bytes,
    *,
    size: Tuple[int, int] = (1080, 1350),
    background: Tuple[int, int, int, int] = WHITE,
    output_format: str = "JPEG",
    quality: int = 92,
) -> bytes:
    """
    Normalize a saved style-board cover/thumbnail.

    Output standard:
    - portrait canvas, default 1080x1350
    - no crop
    - centered preview
    - JPEG by default for saved board covers

    Use this before uploading saved board previews to R2/style-boards.
    """
    target_w, target_h = size
    if target_w <= 0 or target_h <= 0:
        raise ValueError("size must contain positive width and height")

    img = _open_rgba(image_bytes)
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (target_w, target_h), background)
    x = (target_w - img.width) // 2
    y = (target_h - img.height) // 2
    canvas.alpha_composite(img, (x, y))

    out = io.BytesIO()
    fmt = output_format.upper()

    if fmt in {"JPG", "JPEG"}:
        canvas = _flatten_rgba(canvas, background)
        canvas.save(out, format="JPEG", quality=quality, optimize=True)
    else:
        canvas.save(out, format=fmt, optimize=True)

    return out.getvalue()


def _flatten_rgba(
    img: Image.Image,
    background: Tuple[int, int, int, int] = WHITE,
) -> Image.Image:
    """
    Flatten RGBA image onto a solid background for JPEG output.
    """
    img = img.convert("RGBA")
    base = Image.new("RGBA", img.size, background)
    base.alpha_composite(img)
    return base.convert("RGB")


def get_image_size(image_bytes: bytes) -> Tuple[int, int]:
    """
    Small helper for tests/logging.
    """
    img = Image.open(io.BytesIO(image_bytes))
    return img.size
