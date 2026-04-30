
# backend/brain/nlu/intent_router.py

import re
from typing import Dict, Any


class IntentRouter:
    """
    🔥 ELITE NLU ROUTER

    Responsibilities:
    ✔ fast intent detection (no LLM)
    ✔ slot extraction
    ✔ confidence scoring
    ✔ hybrid-ready output
    """

    def __init__(self):

        # -------------------------
        # STYLING KEYWORDS
        # -------------------------
        self.styling_patterns = self._compile_patterns([
            "wear", "outfit", "dress", "clothes", "style",
            "look", "matching", "fit", "what should i wear"
        ])

        # -------------------------
        # OCCASIONS
        # -------------------------
        self.occasions = self._compile_dict_patterns({
            "party": ["party", "club", "birthday", "pub"],
            "office": ["office", "work", "interview", "meeting", "corporate"],
            "vacation": ["vacation", "trip", "holiday", "beach", "travel", "goa"],
            "wedding": ["wedding", "reception", "festival", "event", "pooja"],
            "casual": ["casual", "daily", "everyday", "grocery"]
        })

        # -------------------------
        # WEATHER
        # -------------------------
        self.weather_conditions = self._compile_dict_patterns({
            "rainy": ["rain", "rainy", "monsoon"],
            "summer": ["hot", "summer", "sunny"],
            "winter": ["cold", "winter", "freezing"]
        })

        # -------------------------
        # LIFE MODULES
        # -------------------------
        self.life_keywords = self._compile_dict_patterns({
            "meal_planner": ["meal", "diet", "food", "protein", "recipe"],
            "life_goals": ["goal", "habit", "progress"],
            "health_wellness": ["workout", "gym", "skincare", "fitness"],
            "finance_home": ["bill", "budget", "expense", "savings"]
        })

    # =========================
    # HELPERS
    # =========================
    def _compile_patterns(self, keywords):
        return [re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in keywords]

    def _compile_dict_patterns(self, data):
        return {
            key: [re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in values]
            for key, values in data.items()
        }

    def normalize_text(self, text: str) -> str:
        return (text or "").lower().strip()

    # =========================
    # SLOT EXTRACTION
    # =========================
    def extract_slots(self, text: str) -> Dict[str, Any]:

        text = self.normalize_text(text)

        slots = {
            "occasion": None,
            "weather": None,
            "life_category": None
        }

        # Occasion
        for occasion, patterns in self.occasions.items():
            hits = sum(1 for p in patterns if p.search(text))
            if hits:
                slots["occasion"] = occasion
                break

        # Weather
        for weather, patterns in self.weather_conditions.items():
            if any(p.search(text) for p in patterns):
                slots["weather"] = weather
                break

        # Life category
        for category, patterns in self.life_keywords.items():
            if any(p.search(text) for p in patterns):
                slots["life_category"] = category
                break

        return slots

    # =========================
    # SCORING ENGINE
    # =========================
    def _score_styling(self, text: str, slots: Dict) -> float:

        score = 0.0

        # keyword hits
        hits = sum(1 for p in self.styling_patterns if p.search(text))
        score += hits * 0.3

        # context boosts
        if slots.get("occasion"):
            score += 0.4

        if slots.get("weather"):
            score += 0.2

        return score

    # =========================
    # MAIN CLASSIFIER
    # =========================
    def classify_intent(self, text: str) -> Dict[str, Any]:

        text = self.normalize_text(text)
        slots = self.extract_slots(text)

        # -------------------------
        # 1. LIFE INTENTS (PRIORITY)
        # -------------------------
        if slots["life_category"]:
            return {
                "status": "success",
                "intent": slots["life_category"],
                "slots": slots,
                "confidence": 0.95,
                "source": "router"
            }

        # -------------------------
        # 2. STYLING INTENT
        # -------------------------
        styling_score = self._score_styling(text, slots)

        if styling_score > 0:
            return {
                "status": "success",
                "intent": "styling",
                "slots": slots,
                "confidence": min(0.5 + styling_score, 0.9),
                "source": "router"
            }

        # -------------------------
        # 3. LIGHT MATCH (LOW CONF)
        # -------------------------
        if any(p.search(text) for p in self.styling_patterns):
            return {
                "status": "partial",
                "intent": "styling",
                "slots": slots,
                "confidence": 0.4,
                "source": "router_partial"
            }

        # -------------------------
        # 4. UNKNOWN
        # -------------------------
        return {
            "status": "unrecognized",
            "intent": "general",
            "slots": slots,
            "confidence": 0.2,
            "source": "router_none"
        }


# singleton
nlu_router = IntentRouter()
