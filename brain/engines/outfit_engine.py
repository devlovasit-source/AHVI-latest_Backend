
from typing import Any, Dict, List
import random
import itertools

from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.styling.palette_engine import palette_engine
from brain.engines.style_scorer import style_scorer


class OutfitEngine:
    """
    🔥 FINAL — DIVERSITY-AWARE STYLIST ENGINE

    - Generates combinations
    - Scores via UnifiedStyleScorer
    - Enforces diversity across outputs
    - Produces true stylist routes
    """

    def generate(self, wardrobe: List[Dict[str, Any]], context: Dict[str, Any]):

        if not wardrobe:
            return {"routes": []}

        # =========================
        # SPLIT
        # =========================
        tops = [i for i in wardrobe if i.get("category") in ["top", "tops"]]
        bottoms = [i for i in wardrobe if i.get("category") in ["bottom", "bottoms"]]
        shoes = [i for i in wardrobe if i.get("category") in ["shoes", "footwear"]]
        layers = [i for i in wardrobe if i.get("category") in ["outerwear"]]
        accessories = [i for i in wardrobe if i.get("category") in ["accessories", "bags", "jewelry"]]

        if not tops or not bottoms or not shoes:
            return {"routes": []}

        # =========================
        # GRAPH
        # =========================
        graph = style_graph_engine.build_graph({
            "tops": tops,
            "bottoms": bottoms,
            "shoes": shoes
        })

        # =========================
        # GENERATE
        # =========================
        candidates = self._generate_candidates(tops, bottoms, shoes, layers, accessories)

        # =========================
        # SCORING BLOCK (FIXED)
        # =========================
        scored = []

        for items in candidates:
            score_data = style_scorer.score_outfit(items, context, graph)

            scored.append({
                "items": items,
                "score": score_data.get("score", 0),   # ✅ FIX (numeric)
                "score_meta": score_data               # ✅ keep metadata
            })

        # =========================
        # SORTING
        # =========================
        scored.sort(key=lambda x: x["score"], reverse=True)

        # =========================
        # DIVERSITY SELECTION
        # =========================
        diverse = self._select_diverse_outfits(scored)

        # =========================
        # ROUTES
        # =========================
        routes = self._build_routes(diverse, context)

        return {"routes": routes}

    # =========================
    # CANDIDATES
    # =========================
    def _generate_candidates(self, tops, bottoms, shoes, layers, accessories):

        combos = list(itertools.product(tops, bottoms, shoes))
        random.shuffle(combos)
        combos = combos[:30]

        candidates = []

        for top, bottom, shoe in combos:

            items = [top, bottom, shoe]

            if layers and random.random() > 0.5:
                items.append(random.choice(layers))

            if accessories and random.random() > 0.5:
                items.append(random.choice(accessories))

            candidates.append(items)

        return candidates

    # =========================
    # DIVERSITY CORE
    # =========================
    def _select_diverse_outfits(self, scored: List[Dict[str, Any]]):

        if not scored:
            return []

        selected = []

        for candidate in scored:

            if not selected:
                selected.append(candidate)
                continue

            is_similar = False

            for existing in selected:
                sim = self._similarity(candidate["items"], existing["items"])

                if sim >= 3:
                    is_similar = True
                    break

            if not is_similar:
                selected.append(candidate)

            if len(selected) >= 3:
                break

        return selected

    # =========================
    # SIMILARITY
    # =========================
    def _similarity(self, items_a, items_b):

        types_a = set(i.get("type") for i in items_a)
        types_b = set(i.get("type") for i in items_b)

        colors_a = set(i.get("color") for i in items_a)
        colors_b = set(i.get("color") for i in items_b)

        return len(types_a & types_b) + len(colors_a & colors_b)

    # =========================
    # ROUTES
    # =========================
    def _build_routes(self, outfits, context):

        routes = []

        labels = [
            ("safe", "Easy Win"),
            ("elevated", "Sharp Upgrade"),
            ("bold", "Statement Move"),
        ]

        for i, outfit in enumerate(outfits):

            route_type, label = labels[i] if i < len(labels) else ("alt", "Option")

            items = outfit["items"]
            score = outfit["score"]
            meta = outfit.get("score_meta", {})

            routes.append({
                "type": route_type,
                "label": label,
                "outfit": {
                    "items": items,
                    "score": round(score, 3),
                    "aesthetic": self._build_aesthetic(items, context, route_type),
                    "description": self._build_description(route_type, context),

                    # ✅ ADD (non-breaking explainability)
                    "label": meta.get("label"),
                    "reasons": meta.get("reasons", [])
                }
            })

        return routes

    # =========================
    # AESTHETIC
    # =========================
    def _build_aesthetic(self, items, context, route_type):

        colors = [i.get("color", "").lower() for i in items if i.get("color")]

        if len(set(colors)) == 1:
            color_story = "monochrome"
        elif len(set(colors)) <= 2:
            color_story = "balanced"
        else:
            color_story = "contrast"

        vibe_map = {
            "safe": "clean_minimal",
            "elevated": "refined_structured",
            "bold": "expressive_statement"
        }

        return {
            "vibe": vibe_map.get(route_type),
            "color_story": color_story,
            "occasion": context.get("occasion")
        }

    # =========================
    # DESCRIPTION
    # =========================
    def _build_description(self, route_type, context):

        occasion = context.get("occasion", "your day")

        if route_type == "safe":
            return f"Reliable, clean, and effortless — perfect for {occasion}."

        if route_type == "elevated":
            return f"A sharper, more styled take — ideal when you want to stand out at {occasion}."

        if route_type == "bold":
            return f"Confident and expressive — this look makes a statement for {occasion}."

        return "Strong option."


# Singleton
outfit_engine = OutfitEngine()
