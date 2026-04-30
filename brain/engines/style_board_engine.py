import hashlib
from typing import Any, Dict, List

from brain.engines.styling.palette_engine import palette_engine
from brain.engines.color_normalizer import color_normalizer


def _stable_unit(seed: str) -> float:
    digest = hashlib.sha256(str(seed or "").encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _stable_uniform(a: float, b: float, *, seed: str) -> float:
    u = _stable_unit(seed)
    return float(a) + (float(b) - float(a)) * u


def _stable_choice(options: List[Any], *, seed: str):
    if not options:
        return None
    digest = hashlib.sha256(str(seed or "").encode("utf-8", errors="ignore")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(options)
    return options[idx]


class StyleBoardEngine:
    """
    🔥 ELITE V2 — EDITORIAL + PERSONALIZED

    - Multi-aesthetic aware
    - Visual importance scoring
    - Smart hero selection
    - Contrast-aware layouts
    - Personalized boards
    """

    def build_board(self, outfit: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:

        items = outfit.get("items", [])
        if not items:
            return {}

        style_dna = context.get("style_dna", {}) or {}

        # 🔥 MULTI-AESTHETIC SUPPORT
        primary = style_dna.get("primary_aesthetic")
        secondary = style_dna.get("secondary_aesthetics", [])

        dominant = primary or (secondary[0] if secondary else None)

        palette = palette_engine.select_palette({
            "event": context.get("occasion"),
            "microtheme": dominant
        })

        aesthetic = self._derive_aesthetic(items, style_dna, palette)
        vibe = self._derive_vibe(context, aesthetic)
        color_story = self._build_color_story(items, palette)

        # 🔥 NEW: importance scoring
        importance = self._compute_visual_importance(items)

        layout = self._build_editorial_layout(items, importance, aesthetic)

        return {
            "aesthetic": aesthetic,
            "vibe": vibe,
            "color_story": color_story,
            "layout": layout,
            "items": items,
            "score": outfit.get("score", 0)
        }

    # =========================
    # 🎨 AESTHETIC
    # =========================
    def _derive_aesthetic(self, items, dna, palette):

        colors = [color_normalizer.normalize(i.get("color")) for i in items if i.get("color")]
        fits = [str(i.get("fit", "")).lower() for i in items]

        if len(set(colors)) == 1:
            return "monochrome luxury"

        if len(set(colors)) == 2:
            return "balanced minimal"

        if len(set(colors)) >= 3:
            return "editorial contrast"

        if any("oversized" in f for f in fits):
            return "relaxed luxury"

        if any("tailored" in f for f in fits):
            return "structured refinement"

        return dna.get("primary_aesthetic", "modern refined")

    # =========================
    # 💫 VIBE
    # =========================
    def _derive_vibe(self, context, aesthetic):

        occasion = (context.get("occasion") or "").lower()

        vibe_map = {
            "office": "quiet authority",
            "date": "soft allure",
            "party": "elevated presence"
        }

        if occasion in vibe_map:
            return vibe_map[occasion]

        if "luxury" in aesthetic:
            return "quiet luxury mood"

        return "refined everyday"

    # =========================
    # 🎨 COLOR STORY
    # =========================
    def _build_color_story(self, items, palette):

        item_colors = [color_normalizer.normalize(i.get("color")) for i in items]
        palette_colors = [color_normalizer.normalize(c) for c in palette.get("hex", [])]

        return list(dict.fromkeys(item_colors + palette_colors))[:5]

    # =========================
    # 🔥 IMPORTANCE SCORING
    # =========================
    def _compute_visual_importance(self, items):

        scores = {}

        for item in items:
            score = 0
            t = str(item.get("type", "")).lower()
            color = color_normalizer.normalize(item.get("color"))

            # type priority
            if any(k in t for k in ["outerwear", "dress", "blazer"]):
                score += 2.0
            elif "top" in t:
                score += 1.5

            # contrast colors
            if color in ["red", "black", "white"]:
                score += 1.0

            # unique style boost
            if item.get("style"):
                score += 0.5

            scores[item.get("id")] = score

        return scores

    # =========================
    # 🧠 LAYOUT
    # =========================
    def _build_editorial_layout(self, items, importance, aesthetic):

        sorted_items = sorted(items, key=lambda x: importance.get(x.get("id"), 0), reverse=True)

        hero = sorted_items[0]
        supporting = sorted_items[1:]

        return {
            "composition": self._composition_type(len(items), aesthetic),
            "layers": self._build_layers(hero, supporting),
            "placements": self._build_placements(hero, supporting)
        }

    # =========================
    # 🧱 LAYERS
    # =========================
    def _build_layers(self, hero, supporting):

        return {
            "foreground": [hero],
            "midground": supporting[:2],
            "background": supporting[2:]
        }

    # =========================
    # 📐 PLACEMENT
    # =========================
    def _build_placements(self, hero, supporting):

        placements = {}

        placements[hero.get("id")] = {
            "x": 0.5,
            "y": 0.4,
            "scale": 1.25,
            "rotation": 0,
            "z": 3
        }

        positions = [(0.2, 0.75), (0.8, 0.75), (0.3, 0.2), (0.7, 0.2)]

        for i, item in enumerate(supporting):
            item_id = str(item.get("id") or "")
            placements[item.get("id")] = {
                "x": positions[i % len(positions)][0],
                "y": positions[i % len(positions)][1],
                # Deterministic placements: stable per item id (Pinterest-style but not random each request).
                "scale": round(_stable_uniform(0.6, 0.85, seed=f"scale:{item_id}:{i}"), 2),
                "rotation": _stable_choice([-15, -5, 5, 15], seed=f"rot:{item_id}:{i}"),
                "z": 2 if i < 2 else 1
            }

        return placements

    # =========================
    # 🧾 COMPOSITION
    # =========================
    def _composition_type(self, count, aesthetic):

        if "luxury" in aesthetic:
            return "editorial_focus"

        if count <= 3:
            return "minimal"

        if count <= 5:
            return "balanced"

        return "spread"


# Singleton
style_board_engine = StyleBoardEngine()
