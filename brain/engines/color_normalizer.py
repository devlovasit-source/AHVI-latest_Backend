from typing import Tuple, Dict
import re
import math


class ColorNormalizer:
    """
    🔥 PREMIUM COLOR INTELLIGENCE

    - hex → nearest color name (distance-based)
    - tone detection (warm / cool / neutral)
    - consistent mapping across system

    Used by:
    - style_scorer
    - palette engine
    - tone engine (future)
    """

    def __init__(self):
        # 🎯 anchor colors (expandable)
        self.color_map: Dict[str, Tuple[int, int, int]] = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "grey": (128, 128, 128),
            "red": (220, 20, 60),
            "blue": (30, 144, 255),
            "green": (34, 139, 34),
            "yellow": (255, 215, 0),
            "orange": (255, 140, 0),
            "purple": (138, 43, 226),
            "pink": (255, 105, 180),
            "beige": (245, 222, 179),
            "brown": (139, 69, 19),
            "navy": (0, 0, 128),
            "cream": (255, 253, 208),
        }

    # =========================
    # MAIN ENTRY
    # =========================
    def normalize(self, color: str) -> str:
        if not color:
            return ""

        color = color.strip().lower()

        # already a name
        if not color.startswith("#"):
            return color

        rgb = self._hex_to_rgb(color)
        if not rgb:
            return color

        return self._closest_color(rgb)

    # =========================
    # HEX → RGB
    # =========================
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int] | None:
        hex_color = hex_color.replace("#", "")

        if len(hex_color) == 3:
            hex_color = "".join([c * 2 for c in hex_color])

        if len(hex_color) != 6 or not re.match(r"[0-9a-f]{6}", hex_color):
            return None

        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )

    # =========================
    # DISTANCE MATCH
    # =========================
    def _closest_color(self, rgb: Tuple[int, int, int]) -> str:
        min_dist = float("inf")
        closest = "unknown"

        for name, ref_rgb in self.color_map.items():
            dist = self._distance(rgb, ref_rgb)
            if dist < min_dist:
                min_dist = dist
                closest = name

        return closest

    def _distance(self, c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        return math.sqrt(
            (c1[0] - c2[0]) ** 2 +
            (c1[1] - c2[1]) ** 2 +
            (c1[2] - c2[2]) ** 2
        )

    # =========================
    # TONE DETECTION
    # =========================
    def detect_tone(self, color: str) -> str:
        """
        Returns:
        - warm
        - cool
        - neutral
        """

        color = self.normalize(color)

        warm = {"red", "orange", "yellow", "brown"}
        cool = {"blue", "green", "purple", "navy"}
        neutral = {"black", "white", "grey", "beige", "cream"}

        if color in warm:
            return "warm"
        if color in cool:
            return "cool"
        if color in neutral:
            return "neutral"

        return "neutral"


# Singleton
color_normalizer = ColorNormalizer()
