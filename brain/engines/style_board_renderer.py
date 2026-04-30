import io
import base64
import os
from typing import Dict, Any

import requests
from PIL import Image, ImageFilter, ImageDraw, ImageFont

# NEW ✅
from services.bg_service import remove_bg_external as remove_background_sync

class StyleBoardRenderer:
    """
    🔥 FINAL ELITE RENDERER

    Features:
    - Engine-driven layout
    - Fallback safety
    - Collision avoidance
    - Editorial layering
    - Premium shadows
    """

    CANVAS_SIZE = (1024, 1280)

    # =========================
    # MAIN ENTRY
    # =========================
    def render(self, board: Dict[str, Any]) -> bytes:

        canvas = self._create_background(board)

        items = board.get("items", [])
        layout = board.get("layout", {})

        # 🔥 VALIDATE OR FALLBACK
        if not self._is_valid_layout(layout, items):
            layout = self._build_fallback_layout(items)

        placements = layout.get("placements", {})
        layers = layout.get("layers", {})

        # 🔥 COLLISION FIX
        placements = self._resolve_collisions(placements)

        # 🔥 RENDER BY DEPTH
        for layer_name in ["background", "midground", "foreground"]:
            for item in layers.get(layer_name, []):
                self._place_item(canvas, item, placements.get(item.get("id"), {}))

        # 🔥 TYPOGRAPHY
        canvas = self._add_text(canvas, board)

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        buffer.seek(0)

        return buffer.read()

    # =========================
    # BACKGROUND
    # =========================
    def _create_background(self, board):

        aesthetic = str(board.get("aesthetic", "")).lower()

        if "luxury" in aesthetic:
            base = (245, 240, 232)
        elif "street" in aesthetic:
            base = (20, 20, 20)
        else:
            base = (250, 250, 250)

        img = Image.new("RGB", self.CANVAS_SIZE, base)

        overlay = Image.new("RGBA", self.CANVAS_SIZE, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for y in range(0, self.CANVAS_SIZE[1], 4):
            alpha = int(30 * (y / self.CANVAS_SIZE[1]))
            draw.rectangle([(0, y), (self.CANVAS_SIZE[0], y + 4)], fill=(255, 255, 255, alpha))

        img.paste(overlay, (0, 0), overlay)
        return img

    # =========================
    # LAYOUT VALIDATION
    # =========================
    def _is_valid_layout(self, layout, items):

        if not layout:
            return False

        layers = layout.get("layers")
        placements = layout.get("placements")

        if not isinstance(layers, dict) or not isinstance(placements, dict):
            return False

        item_ids = {str(i.get("id")) for i in items if i.get("id")}
        placed_ids = set(placements.keys())

        return bool(item_ids.intersection(placed_ids))

    # =========================
    # FALLBACK LAYOUT
    # =========================
    def _build_fallback_layout(self, items):

        if not items:
            return {}

        hero = items[0]
        supporting = items[1:]

        layers = {
            "foreground": [hero],
            "midground": supporting[:2],
            "background": supporting[2:]
        }

        placements = {}

        placements[hero.get("id")] = {
            "x": 0.5,
            "y": 0.45,
            "scale": 1.1,
            "rotation": 0,
            "z": 3
        }

        positions = [(0.25, 0.75), (0.75, 0.75), (0.25, 0.2), (0.75, 0.2)]

        for i, item in enumerate(supporting):
            placements[item.get("id")] = {
                "x": positions[i % len(positions)][0],
                "y": positions[i % len(positions)][1],
                "scale": 0.7,
                "rotation": 0,
                "z": 2
            }

        return {"layers": layers, "placements": placements}

    # =========================
    # PLACEMENT
    # =========================
    def _place_item(self, canvas, item, placement):

        img = self._prepare_image(item, placement)
        if img is None:
            return

        img = self._add_shadow(img)

        x = int(placement.get("x", 0.5) * self.CANVAS_SIZE[0] - img.size[0] / 2)
        y = int(placement.get("y", 0.5) * self.CANVAS_SIZE[1] - img.size[1] / 2)

        canvas.paste(img, (x, y), img)

    # =========================
    # IMAGE PREP
    # =========================
    def _prepare_image(self, item, placement):

        img = self._load_image(item)
        if img is None:
            return None

        scale = placement.get("scale", 1.0)
        rotation = placement.get("rotation", 0)

        base_width = int(400 * scale)

        w, h = img.size
        ratio = base_width / max(w, 1)

        img = img.resize((int(w * ratio), int(h * ratio)))
        img = img.rotate(rotation, expand=True)

        return img

    # =========================
    # IMAGE LOADING
    # =========================
    def _load_image(self, item):
        # Prefer pre-masked/sticker assets from R2 whenever available.
        masked_url = (
            item.get("masked_url")
            or item.get("masked_image_url")
            or item.get("sticker_url")
            or item.get("cutout_url")
        )
        url = masked_url or item.get("image_url")
        if not url and item.get("image_base64"):
            try:
                text = str(item.get("image_base64") or "").strip()
                if "," in text:
                    text = text.split(",", 1)[1]
                base = Image.open(io.BytesIO(base64.b64decode(text))).convert("RGBA")
                return base
            except Exception:
                return None
        if not url:
            return None

        try:
            res = requests.get(url, timeout=5)
            if not res.ok:
                return None
            base = Image.open(io.BytesIO(res.content)).convert("RGBA")

            # If this is already a masked/sticker image (or has alpha transparency), don't re-run BG removal.
            if masked_url or self._has_transparency(base):
                return base

            # Optional fallback only for raw images.
            # Keep this disabled by default to avoid re-segmenting already clean assets.
            if str(os.getenv("STYLE_BOARD_APPLY_BG_REMOVAL", "false")).lower() not in {"1", "true", "yes"}:
                return base

            b64 = base64.b64encode(res.content).decode()
            result = remove_background_sync(b64)
            clean = (
                str(result.get("image_base64") or "").split(",")[-1]
                if result.get("success") and result.get("bg_removed")
                else b64
            )
            return Image.open(io.BytesIO(base64.b64decode(clean))).convert("RGBA")

        except Exception:
            return None

    def _has_transparency(self, image: Image.Image) -> bool:
        try:
            if image.mode != "RGBA":
                return False
            alpha = image.getchannel("A")
            lo, hi = alpha.getextrema()
            return int(lo) < 255
        except Exception:
            return False

    # =========================
    # SHADOW
    # =========================
    def _add_shadow(self, img):

        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)

        draw.rectangle([(10, 10), img.size], fill=(0, 0, 0, 80))
        shadow = shadow.filter(ImageFilter.GaussianBlur(12))

        base = Image.new("RGBA", img.size, (0, 0, 0, 0))
        base.paste(shadow, (0, 0), shadow)
        base.paste(img, (0, 0), img)

        return base

    # =========================
    # COLLISION SYSTEM
    # =========================
    def _get_bbox(self, placement, size=(400, 400)):

        scale = placement.get("scale", 1.0)
        w = int(size[0] * scale)
        h = int(size[1] * scale)

        x = placement.get("x", 0.5) * self.CANVAS_SIZE[0]
        y = placement.get("y", 0.5) * self.CANVAS_SIZE[1]

        return {"x1": x - w / 2, "y1": y - h / 2, "x2": x + w / 2, "y2": y + h / 2}

    def _is_overlapping(self, b1, b2):

        padding = 20

        return not (
            b1["x2"] + padding < b2["x1"] or
            b1["x1"] - padding > b2["x2"] or
            b1["y2"] + padding < b2["y1"] or
            b1["y1"] - padding > b2["y2"]
        )

    def _resolve_collisions(self, placements):

        keys = list(placements.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):

                p1 = placements[keys[i]]
                p2 = placements[keys[j]]

                b1 = self._get_bbox(p1)
                b2 = self._get_bbox(p2)

                if self._is_overlapping(b1, b2):

                    dx = (p2["x"] - p1["x"]) or 0.1
                    dy = (p2["y"] - p1["y"]) or 0.1

                    mag = (dx**2 + dy**2) ** 0.5
                    dx /= mag
                    dy /= mag

                    push = 0.08

                    p2["x"] += dx * push
                    p2["y"] += dy * push

                    p2["x"] = max(0.1, min(0.9, p2["x"]))
                    p2["y"] = max(0.1, min(0.9, p2["y"]))

        return placements

    # =========================
    # TYPOGRAPHY
    # =========================
    def _add_text(self, img, board):

        draw = ImageDraw.Draw(img)

        title = str(board.get("vibe", "STYLE")).upper()
        subtitle = str(board.get("aesthetic", "")).title()

        try:
            title_font = ImageFont.truetype("PlayfairDisplay-Bold.ttf", 64)
            sub_font = ImageFont.truetype("Montserrat-Regular.ttf", 28)
        except:
            title_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()

        draw.text((60, 80), title, fill=(20, 20, 20), font=title_font)
        draw.text((60, 150), subtitle, fill=(90, 90, 90), font=sub_font)

        return img


# Singleton
style_board_renderer = StyleBoardRenderer()
