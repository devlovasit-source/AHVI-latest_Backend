
from typing import Any, Dict, List

from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.styling.palette_engine import palette_engine
from brain.engines.color_normalizer import color_normalizer
from brain.engines.memory_scorer import memory_scorer

from services.embedding_service import encode_metadata


class UnifiedStyleScorer:
    """
    🔥 FINAL ELITE STYLE SCORER

    Responsibilities:
    ✔ single scoring authority
    ✔ integrates DNA, rules, palette
    ✔ integrates memory via memory_scorer
    ✔ session-aware
    ✔ refinement-aware
    ✔ stable + explainable
    """

    def score_outfit(
        self,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not items:
            return {"score": 0.0, "label": "Weak", "reasons": []}

        style_dna = context.get("style_dna", {}) or {}
        refinement = context.get("refinement")
        session = context.get("session", {}).get("derived", {})

        confidence = float(style_dna.get("confidence", 0.5))
        exploration_factor = max(0.0, 1.0 - confidence)

        reasons = []
        score = 0.0

        # =========================
        # RULES + PALETTE
        # =========================
        rules = style_engine.get_scoring_rules(style_dna, context)

        palette = palette_engine.select_palette({
            "event": context.get("occasion"),
            "microtheme": style_dna.get("primary_aesthetic")
        })

        palette_colors = [
            color_normalizer.normalize(c)
            for c in palette.get("hex", [])
        ]

        # =========================
        # 1. GRAPH COMPATIBILITY
        # =========================
        graph_score = 0.0

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_id = items[i].get("id")
                b_id = items[j].get("id")

                if a_id and b_id:
                    graph_score += style_graph_engine.pair_weight(graph, a_id, b_id)

        score += graph_score

        if graph_score > 1:
            reasons.append("items pair well together")

        # =========================
        # 2. COLOR + RULES
        # =========================
        for item in items:
            color = color_normalizer.normalize(item.get("color"))
            item_type = str(item.get("type", "")).lower()

            if color in palette_colors:
                score += 1.0
                reasons.append("palette aligned")

            elif self._is_neutral(color):
                score += 0.4

            if color in rules.get("preferred_colors", []):
                score += 0.6

            if item_type in rules.get("avoided_items", []):
                score -= 2.0
                reasons.append("conflicts with style")

        # =========================
        # 3. AESTHETIC BALANCE
        # =========================
        aesthetic_score = self._aesthetic_score(items)
        score += aesthetic_score

        if aesthetic_score > 0.7:
            reasons.append("clean aesthetic balance")

        # =========================
        # 4. STYLE DNA
        # =========================
        dna_score = self._dna_score(items, style_dna)
        score += dna_score * (0.5 + confidence)

        if dna_score > 0:
            reasons.append("matches your style")

        # =========================
        # 5. MEMORY (🔥 CENTRALIZED)
        # =========================
        vector = self._build_outfit_embedding(items)

        if vector:
            memory_score = memory_scorer.score(vector, context)
            score += memory_score

            if memory_score > 0:
                reasons.append("aligned with your past choices")

        # =========================
        # 6. SESSION (🔥 PRIORITY)
        # =========================
        dominant = session.get("dominant_refinement")

        if dominant:
            session_score = 0.6
            score += session_score
            reasons.append(f"fits your current {dominant} preference")

        # =========================
        # 7. REFINEMENT BOOST
        # =========================
        if refinement:
            refine_score = self._refinement_score(items, refinement)
            score += refine_score

            if refine_score > 0:
                reasons.append(f"refined for {refinement}")

        # =========================
        # 8. EXPLORATION
        # =========================
        score += self._exploration_boost(items, exploration_factor)

        # =========================
        # FINAL NORMALIZATION
        # =========================
        score = max(0, min(score, 10))

        label = self._label(score)

        return {
            "score": round(score, 3),
            "label": label,
            "reasons": list(set(reasons))[:3]
        }

    # =========================
    # DNA
    # =========================
    def _dna_score(self, items, dna):

        if not dna:
            return 0

        score = 0

        preferred_styles = dna.get("preferred_styles", [])
        preferred_colors = dna.get("preferred_colors", [])

        for i in items:
            style = str(i.get("style", "")).lower()
            color = str(i.get("color", "")).lower()

            if style in preferred_styles:
                score += 0.6

            if color in preferred_colors:
                score += 0.5

        return score

    # =========================
    # REFINEMENT
    # =========================
    def _refinement_score(self, items, refinement):

        score = 0

        for item in items:
            style = str(item.get("style", "")).lower()

            if refinement == "sharp" and style in ["formal", "structured"]:
                score += 0.5

            if refinement == "relaxed" and style in ["casual", "loose"]:
                score += 0.5

            if refinement == "minimal" and item.get("pattern") == "solid":
                score += 0.4

        return score

    # =========================
    # EMBEDDING
    # =========================
    def _build_outfit_embedding(self, items):

        text = " ".join([
            f"{i.get('type','')} {i.get('color','')} {i.get('style','')}"
            for i in items
        ])

        try:
            return encode_metadata({"text": text}) or []
        except Exception:
            return []

    # =========================
    # EXPLORATION
    # =========================
    def _exploration_boost(self, items, factor):

        if factor <= 0:
            return 0

        colors = [i.get("color") for i in items if i.get("color")]
        styles = [i.get("style") for i in items if i.get("style")]

        score = 0

        if len(set(colors)) >= 3:
            score += 0.5 * factor

        if len(set(styles)) >= 2:
            score += 0.4 * factor

        # Keep deterministic scoring in production (avoid random drift between calls).
        # If we want "exploration" later, make it an explicit, client-controlled parameter.
        score += 0.0

        return score

    # =========================
    # AESTHETIC
    # =========================
    def _aesthetic_score(self, items):

        colors = [
            color_normalizer.normalize(i.get("color"))
            for i in items if i.get("color")
        ]

        unique = len(set(colors))

        if unique == 1:
            return 1.0
        elif unique == 2:
            return 0.7
        elif unique >= 3:
            return 0.5

        return 0

    # =========================
    # LABEL
    # =========================
    def _label(self, score):

        if score >= 6:
            return "Excellent"
        if score >= 4:
            return "Strong"
        if score >= 2:
            return "Good"
        return "Basic"

    def _is_neutral(self, color):
        return color in ["black", "white", "grey", "gray", "beige", "navy", "cream"]


# singleton
style_scorer = UnifiedStyleScorer()
