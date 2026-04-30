from typing import List, Dict
import json
import os

from brain.utils.style_language_engine import style_language_engine


class StyleExplainer:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._global_path = os.path.join(base_dir, "data", "global_style_memory.json")

    def explain_outfit(self, items: List[Dict], context: Dict) -> str:

        if not items:
            return ""

        style_dna = context.get("style_dna", {}) or {}

        # 🔥 base narrative
        sentence = style_language_engine.outfit_to_sentence(items, context)

        # 🔥 emotional tone
        emotion = self._derive_emotion(style_dna, context)

        # 🔥 global + personal reasoning
        reasoning = self._build_reasoning(items, context, style_dna, emotion)

        return f"{sentence} {reasoning}".strip()

    # =========================
    # 🔥 GLOBAL + PERSONAL
    # =========================
    def _build_reasoning(self, items, context, style_dna, emotion):

        colors = [str(i.get("color", "")).lower() for i in items if i.get("color")]
        fits = [str(i.get("fit", "")).lower() for i in items if i.get("fit")]

        confidence = float(style_dna.get("confidence", 0.5))

        global_memory = self._load_global()

        reasoning_parts = []

        # =========================
        # 🎨 COLOR (GLOBAL + PERSONAL)
        # =========================
        unique_colors = len(set(colors))

        global_colors = set(global_memory.get("colors", []))

        if any(c in global_colors for c in colors):
            reasoning_parts.append(
                self._tone(emotion,
                    "It aligns with current style trends.",
                    "It taps into what’s trending right now.",
                    "It reflects a strong, current aesthetic direction."
                )
            )

        if unique_colors == 1:
            reasoning_parts.append(
                self._tone(emotion,
                    "The palette stays clean and controlled.",
                    "Keeps everything tight and intentional.",
                    "The palette feels refined and commanding."
                )
            )

        elif unique_colors >= 3:
            reasoning_parts.append(
                self._tone(emotion,
                    "The contrast adds dimension.",
                    "The mix brings energy and edge.",
                    "The contrast builds presence."
                )
            )

        # =========================
        # 🧍 FIT BALANCE
        # =========================
        if self._has_balance(fits):
            reasoning_parts.append(
                self._tone(emotion,
                    "The silhouette stays balanced.",
                    "The fit contrast gives it movement.",
                    "The proportions feel structured and deliberate."
                )
            )

        # =========================
        # 🔥 CONFIDENCE MIX
        # =========================
        if confidence < 0.4:
            reasoning_parts.append(
                self._tone(emotion,
                    "It’s a safe, widely appealing choice.",
                    "Easy to wear and broadly styled.",
                    "A reliable and widely accepted direction."
                )
            )

        elif confidence > 0.7:
            reasoning_parts.append(
                self._tone(emotion,
                    "It reflects your personal style strongly.",
                    "This feels very aligned with your style.",
                    "It’s clearly tailored to your aesthetic."
                )
            )

        return " ".join(reasoning_parts[:2])

    # =========================
    # GLOBAL MEMORY
    # =========================
    def _load_global(self):

        if not os.path.exists(self._global_path):
            return {}

        try:
            with open(self._global_path, "r") as f:
                return json.load(f)
        except:
            return {}

    # =========================
    # EMOTION ENGINE
    # =========================
    def _derive_emotion(self, style_dna: Dict, context: Dict) -> str:

        aesthetic = str(style_dna.get("primary_aesthetic", "")).lower()
        confidence = float(style_dna.get("confidence", 0.5))
        occasion = str(context.get("occasion", "")).lower()

        if "party" in occasion or "street" in aesthetic:
            return "bold"

        if "luxury" in aesthetic:
            return "dominant"

        if confidence > 0.7:
            return "confident"

        if confidence < 0.4:
            return "soft"

        return "confident"

    # =========================
    # TONE SWITCH
    # =========================
    def _tone(self, emotion, soft, bold, dominant):

        if emotion == "bold":
            return bold

        if emotion == "dominant":
            return dominant

        if emotion == "soft":
            return soft

        return soft

    # =========================
    # HELPERS
    # =========================
    def _has_balance(self, fits: List[str]) -> bool:

        combos = [
            ("slim", "relaxed"),
            ("oversized", "slim"),
        ]

        for f1 in fits:
            for f2 in fits:
                if f1 == f2:
                    continue
                for a, b in combos:
                    if (a in f1 and b in f2) or (a in f2 and b in f1):
                        return True

        return False


# Singleton
style_explainer = StyleExplainer()
