
from typing import Dict, Any
from datetime import datetime


class ProactiveEngine:
    """
    🔥 ELITE PROACTIVE ENGINE

    Injects intelligent signals BEFORE orchestration:

    - time-based intent
    - weather adaptation
    - style DNA bias
    - memory signals
    - exploration vs confidence
    - auto refinement suggestions
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def inject(self, context: Dict[str, Any]) -> Dict[str, Any]:

        signals = {}

        now = datetime.now()
        hour = now.hour

        style_dna = context.get("style_dna", {}) or {}
        memory = context.get("user_memory", {}) or {}
        weather = context.get("weather", {}) or {}

        # -------------------------
        # 1. TIME INTENT
        # -------------------------
        signals.update(self._time_signals(hour))

        # -------------------------
        # 2. WEATHER INTENT
        # -------------------------
        signals.update(self._weather_signals(weather))

        # -------------------------
        # 3. STYLE DNA BIAS
        # -------------------------
        signals.update(self._dna_signals(style_dna))

        # -------------------------
        # 4. MEMORY SIGNALS
        # -------------------------
        signals.update(self._memory_signals(memory))

        # -------------------------
        # 5. EXPLORATION MODE
        # -------------------------
        signals.update(self._exploration_mode(style_dna))

        # -------------------------
        # 6. AUTO REFINEMENT
        # -------------------------
        signals.update(self._auto_refinement(style_dna, memory))

        # attach
        context["proactive_signals"] = signals

        return context

    # =========================
    # TIME SIGNALS
    # =========================
    def _time_signals(self, hour):

        if 6 <= hour <= 11:
            return {
                "suggestion_type": "morning_outfit",
                "energy": "fresh"
            }

        if 12 <= hour <= 17:
            return {
                "suggestion_type": "day_outfit",
                "energy": "balanced"
            }

        if 18 <= hour <= 22:
            return {
                "suggestion_type": "evening_outfit",
                "energy": "elevated"
            }

        return {
            "suggestion_type": "casual_outfit",
            "energy": "relaxed"
        }

    # =========================
    # WEATHER SIGNALS
    # =========================
    def _weather_signals(self, weather):

        if not weather:
            return {}

        temp = weather.get("temperature")

        if temp is None:
            return {}

        if temp >= 32:
            return {
                "weather_mode": "hot",
                "fabric_preference": "lightweight"
            }

        if temp <= 15:
            return {
                "weather_mode": "cold",
                "fabric_preference": "layered"
            }

        return {
            "weather_mode": "mild"
        }

    # =========================
    # STYLE DNA SIGNALS
    # =========================
    def _dna_signals(self, dna):

        if not dna:
            return {}

        primary = dna.get("primary_aesthetic")

        signals = {}

        if primary:
            signals["style_bias"] = primary

        # dominant preference
        formality = dna.get("formality", {})
        if formality:
            if formality.get("sharp", 0) > 0.6:
                signals["default_refinement"] = "sharp"

            elif formality.get("casual", 0) > 0.6:
                signals["default_refinement"] = "relaxed"

        return signals

    # =========================
    # MEMORY SIGNALS
    # =========================
    def _memory_signals(self, memory):

        signals = memory.get("memory_signals", {})

        result = {}

        if signals.get("preferred_styles"):
            result["memory_style_bias"] = signals["preferred_styles"][-2:]

        if signals.get("liked_colors"):
            result["memory_color_bias"] = signals["liked_colors"][-3:]

        return result

    # =========================
    # EXPLORATION MODE
    # =========================
    def _exploration_mode(self, dna):

        confidence = float(dna.get("confidence", 0.5))

        if confidence < 0.3:
            return {"exploration_mode": "high"}

        if confidence < 0.6:
            return {"exploration_mode": "medium"}

        return {"exploration_mode": "low"}

    # =========================
    # AUTO REFINEMENT
    # =========================
    def _auto_refinement(self, dna, memory):

        signals = {}

        preferred_styles = dna.get("preferred_styles", [])
        memory_styles = memory.get("memory_signals", {}).get("preferred_styles", [])

        combined = list(set(preferred_styles + memory_styles))

        if "minimal" in combined:
            signals["auto_refinement"] = "minimal"

        elif "formal" in combined:
            signals["auto_refinement"] = "sharp"

        elif "casual" in combined:
            signals["auto_refinement"] = "relaxed"

        return signals


# singleton
proactive_engine = ProactiveEngine()
