import os
import json
from typing import List, Dict


class ArchetypeEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "tone", "archetypes.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.config = json.load(f).get("archetypes", {})
        except Exception:
            self.config = {}

    # =========================
    # SELECT ARCHETYPE (BLENDED)
    # =========================
    def select(self, context: dict, user_memory: dict = None) -> List[Dict]:

        context = context or {}
        user_memory = user_memory or {}

        domain = context.get("domain")
        emotion = context.get("signals", {}).get("emotion_state")

        scores = user_memory.get("archetype_scores", {})

        # -------------------------
        # 🔥 HARD OVERRIDE (CRITICAL)
        # -------------------------
        if emotion == "vulnerable":
            return [{"type": "advisor", "weight": 1.0}]

        # -------------------------
        # 🔥 USER-LEARNED BLEND
        # -------------------------
        if scores:
            # pick top 2 archetypes
            sorted_arcs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]

            total = sum(v for _, v in sorted_arcs) or 1.0

            return [
                {
                    "type": arc,
                    "weight": round(score / total, 3)
                }
                for arc, score in sorted_arcs
                if arc in self.config
            ]

        # -------------------------
        # 🔥 CONTEXT FALLBACK (SMART)
        # -------------------------
        if domain == "styling":
            return [{"type": "stylist", "weight": 1.0}]

        if domain in ["lifestyle", "chat"]:
            return [{"type": "best_friend", "weight": 1.0}]

        return [{"type": "advisor", "weight": 1.0}]

    # =========================
    # GET CONFIG
    # =========================
    def get_config(self, arc_type: str) -> dict:
        return self.config.get(arc_type, {})


# Singleton
archetype_engine = ArchetypeEngine()
