from typing import Any, Dict, List, Tuple

from brain.engines.color_normalizer import color_normalizer
from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.styling.palette_engine import palette_engine
from services.embedding_service import encode_metadata

try:
    from brain.engines.memory_scorer import memory_scorer
except Exception:  # pragma: no cover - optional Qdrant/dependency path
    memory_scorer = None


class UnifiedStyleScorer:
    """
    Single scoring authority for outfit candidates.

    This scorer is intentionally read-only: it scores the supplied items and
    returns reasons/breakdown/warnings, but never swaps or mutates garments.
    """

    def score_outfit(
        self,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not items:
            return {
                "score": 0.0,
                "label": "Weak",
                "reasons": [],
                "breakdown": {},
                "rejection_warnings": ["missing outfit items"],
            }

        context = context or {}
        style_dna = context.get("style_dna", {}) or {}
        refinement = context.get("refinement")
        session = context.get("session", {}).get("derived", {})

        confidence = float(style_dna.get("confidence", 0.5) or 0.5)
        exploration_factor = max(0.0, 1.0 - confidence)

        reasons: List[str] = []
        warnings: List[str] = []
        breakdown = {
            "style_graph": 0.0,
            "palette": 0.0,
            "rules": 0.0,
            "aesthetic_balance": 0.0,
            "style_dna": 0.0,
            "memory": 0.0,
            "session": 0.0,
            "refinement": 0.0,
            "exploration": 0.0,
            "footwear_occasion": 0.0,
        }

        rules = style_engine.get_scoring_rules(style_dna, context)
        palette = palette_engine.select_palette(
            {
                "event": context.get("occasion"),
                "microtheme": style_dna.get("primary_aesthetic"),
            }
        )
        palette_colors = [
            color_normalizer.normalize(c) for c in palette.get("hex", []) if c
        ]

        graph_score = self._graph_score(items, graph or {})
        breakdown["style_graph"] = graph_score
        if graph_score > 1:
            reasons.append("items pair well together")

        for item in items:
            color = color_normalizer.normalize(item.get("color") or item.get("color_code"))
            item_type = str(item.get("type") or item.get("sub_category") or "").lower()

            try:
                rule_delta, rule_reasons = style_engine.score_item_rule_adjustment(
                    item, rules
                )
                breakdown["rules"] += rule_delta
                reasons.extend(rule_reasons)
                if rule_delta <= -10:
                    warnings.extend(rule_reasons)
            except Exception:
                pass

            if color in palette_colors:
                breakdown["palette"] += 1.0
                reasons.append("palette aligned")
            elif self._is_neutral(color):
                breakdown["palette"] += 0.4

            if color in rules.get("preferred_colors", []):
                breakdown["palette"] += 0.6

            if item_type in rules.get("avoided_items", []):
                breakdown["rules"] -= 2.0
                reasons.append("conflicts with style")
                warnings.append(f"conflicts with style: {item_type}")

        footwear_delta, footwear_reasons, footwear_warnings = (
            self._footwear_occasion_score(items, context)
        )
        breakdown["footwear_occasion"] = footwear_delta
        reasons.extend(footwear_reasons)
        warnings.extend(footwear_warnings)

        aesthetic_score = self._aesthetic_score(items)
        breakdown["aesthetic_balance"] = aesthetic_score
        if aesthetic_score > 0.7:
            reasons.append("clean aesthetic balance")

        dna_score = self._dna_score(items, style_dna)
        breakdown["style_dna"] = dna_score * (0.5 + confidence)
        if dna_score > 0:
            reasons.append("matches your style")

        vector = self._build_outfit_embedding(items)
        if vector and memory_scorer is not None:
            try:
                memory_score = float(memory_scorer.score(vector, context) or 0.0)
            except Exception:
                memory_score = 0.0
            breakdown["memory"] = memory_score
            if memory_score > 0:
                reasons.append("aligned with your past choices")

        dominant = session.get("dominant_refinement")
        if dominant:
            breakdown["session"] = 0.6
            reasons.append(f"fits your current {dominant} preference")

        if refinement:
            refine_score = self._refinement_score(items, refinement)
            breakdown["refinement"] = refine_score
            if refine_score > 0:
                reasons.append(f"refined for {refinement}")

        breakdown["exploration"] = self._exploration_boost(items, exploration_factor)

        raw_score = sum(float(v or 0.0) for v in breakdown.values())
        score = max(0.0, min(raw_score, 10.0))

        return {
            "score": round(score, 3),
            "label": self._label(score),
            "reasons": list(dict.fromkeys(reasons))[:3],
            "breakdown": {
                key: round(float(value or 0.0), 3)
                for key, value in breakdown.items()
            },
            "rejection_warnings": list(dict.fromkeys(warnings))[:5],
        }

    def _graph_score(self, items: List[Dict[str, Any]], graph: Dict[str, Any]) -> float:
        score = 0.0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_id = items[i].get("id")
                b_id = items[j].get("id")
                if a_id and b_id:
                    score += style_graph_engine.pair_weight(graph, a_id, b_id)
        return score

    def _item_text(self, item: Dict[str, Any]) -> str:
        if not isinstance(item, dict):
            return ""
        fields = (
            "name",
            "label",
            "title",
            "type",
            "category",
            "sub_category",
            "subcategory",
            "role",
            "slot",
            "style",
            "material",
            "fabric",
            "color",
        )
        return " ".join(str(item.get(k) or "").lower() for k in fields)

    def _item_role(self, item: Dict[str, Any]) -> str:
        text = self._item_text(item)
        if any(
            x in text
            for x in [
                "shoe",
                "sneaker",
                "loafer",
                "boot",
                "sandal",
                "slipper",
                "slider",
            ]
        ):
            return "footwear"
        if any(x in text for x in ["shirt", "tee", "tshirt", "top", "polo", "hoodie", "kurta"]):
            return "top"
        if any(x in text for x in ["pant", "trouser", "jean", "chino", "shorts", "skirt"]):
            return "bottom"
        if any(x in text for x in ["dress", "saree", "lehenga", "gown"]):
            return "dress"
        return ""

    def _context_intent(self, context: Dict[str, Any]) -> str:
        text = " ".join(
            str(context.get(k) or "").lower()
            for k in ("occasion", "intent", "query", "user_query")
        )
        if any(x in text for x in ["office", "meeting", "client", "work"]):
            return "office"
        if any(x in text for x in ["date", "dinner", "night"]):
            return "date"
        if any(x in text for x in ["party", "club", "after-hours"]):
            return "party"
        return "daily"

    def _footwear_occasion_score(
        self, items: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        intent = self._context_intent(context or {})
        footwear = next((i for i in items if self._item_role(i) == "footwear"), None)
        text = self._item_text(footwear or {})
        if not footwear:
            return -4.0, [], ["missing footwear"]

        relaxed = any(x in text for x in ["slipper", "slider", "slides", "flip", "crocs", "birkenstock", "sandal"])
        athletic = any(x in text for x in ["running", "gym", "training", "sports", "chunky", "runner", "hiking", "trail"])
        polished = any(
            x in text
            for x in [
                "loafer",
                "formal",
                "leather",
                "chelsea",
                "boot",
                "clean sneaker",
                "white sneaker",
                "minimal sneaker",
            ]
        )

        if intent == "office":
            if relaxed or athletic:
                return -5.0, ["footwear weak for office"], ["office footwear mismatch"]
            if polished:
                return 1.8, ["office-ready footwear"], []
        if intent == "date":
            if relaxed or athletic:
                return -5.0, ["footwear weak for date night"], ["date footwear mismatch"]
            if polished:
                return 1.5, ["polished footwear"], []
        if intent == "party" and polished:
            return 0.7, ["strong footwear finish"], []
        return 0.0, [], []

    def _dna_score(self, items: List[Dict[str, Any]], dna: Dict[str, Any]) -> float:
        if not dna:
            return 0.0
        score = 0.0
        preferred_styles = {str(x).lower() for x in dna.get("preferred_styles", [])}
        preferred_colors = {str(x).lower() for x in dna.get("preferred_colors", [])}
        for item in items:
            style = str(item.get("style") or "").lower()
            color = str(item.get("color") or item.get("color_code") or "").lower()
            if style and style in preferred_styles:
                score += 0.6
            if color and color in preferred_colors:
                score += 0.5
        return score

    def _refinement_score(self, items: List[Dict[str, Any]], refinement: str) -> float:
        score = 0.0
        for item in items:
            style = str(item.get("style") or "").lower()
            pattern = str(item.get("pattern") or "").lower()
            if refinement == "sharp" and style in {"formal", "structured"}:
                score += 0.5
            if refinement == "relaxed" and style in {"casual", "loose"}:
                score += 0.5
            if refinement == "minimal" and pattern in {"solid", "plain"}:
                score += 0.4
        return score

    def _build_outfit_embedding(self, items: List[Dict[str, Any]]) -> List[float]:
        text = " ".join(
            f"{i.get('type','')} {i.get('sub_category','')} {i.get('color','')} {i.get('style','')}"
            for i in items
            if isinstance(i, dict)
        )
        try:
            return encode_metadata({"text": text}) or []
        except Exception:
            return []

    def _exploration_boost(self, items: List[Dict[str, Any]], factor: float) -> float:
        if factor <= 0:
            return 0.0
        colors = [i.get("color") or i.get("color_code") for i in items if i.get("color") or i.get("color_code")]
        styles = [i.get("style") for i in items if i.get("style")]
        score = 0.0
        if len(set(colors)) >= 3:
            score += 0.5 * factor
        if len(set(styles)) >= 2:
            score += 0.4 * factor
        return score

    def _aesthetic_score(self, items: List[Dict[str, Any]]) -> float:
        colors = [
            color_normalizer.normalize(i.get("color") or i.get("color_code"))
            for i in items
            if i.get("color") or i.get("color_code")
        ]
        unique = len(set(colors))
        if unique == 1:
            return 1.0
        if unique == 2:
            return 0.7
        if unique >= 3:
            return 0.5
        return 0.0

    def _label(self, score: float) -> str:
        if score >= 6:
            return "Excellent"
        if score >= 4:
            return "Strong"
        if score >= 2:
            return "Good"
        return "Basic"

    def _is_neutral(self, color: str) -> bool:
        return color in {"black", "white", "grey", "gray", "beige", "navy", "cream"}


style_scorer = UnifiedStyleScorer()
