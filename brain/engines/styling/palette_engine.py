# backend/brain/engines/palette_engine.py

import os
import json


class PaletteEngine:
    def __init__(self):
        # Keep the engine data-driven and avoid hard paths that drift from the repo layout.
        here = os.path.dirname(os.path.abspath(__file__))  # brain/engines/styling
        brain_dir = os.path.abspath(os.path.join(here, "..", ".."))  # brain/
        file_path = os.path.join(brain_dir, "banks", "events", "color_palette_bank_v1.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except:
            self.data = {}

        self.palettes = self.data.get("palettes", [])
        self.microthemes = self.data.get("microtheme_overrides", [])

    # =========================
    # GET BY EVENT
    # =========================
    def get_palette_for_event(self, event_type: str):
        for p in self.palettes:
            if event_type in p.get("best_for", []):
                return p
        return None

    # =========================
    # GET BY MICROTHEME
    # =========================
    def get_palette_by_microtheme(self, microtheme: str):
        override = next(
            (m for m in self.microthemes if m["microtheme"] == microtheme),
            None
        )

        if not override:
            return None

        keys = override.get("palette_keys", [])
        return [p for p in self.palettes if p["key"] in keys]

    # =========================
    # SMART SELECTOR
    # =========================
    def select_palette(self, context: dict):
        """
        context = {
            "event": "mehendi",
            "time": "day",
            "microtheme": "minimal_modern"
        }
        """

        event = context.get("event")
        microtheme = context.get("microtheme")

        # 1. microtheme override (highest priority)
        if microtheme:
            palettes = self.get_palette_by_microtheme(microtheme)
            if palettes:
                return palettes[0]

        # 2. event match
        palette = self.get_palette_for_event(event)
        if palette:
            return palette

        # 3. fallback
        return self.palettes[0] if self.palettes else {}

    # =========================
    # FORMAT OUTPUT
    # =========================
    def build_palette_response(self, context: dict):
        palette = self.select_palette(context)

        return {
            "palette_key": palette.get("key"),
            "name": palette.get("name"),
            "colors": palette.get("hex", []),
            "tags": palette.get("tags", [])
        }


# =========================
# SINGLETON INSTANCE
# =========================
palette_engine = PaletteEngine()
