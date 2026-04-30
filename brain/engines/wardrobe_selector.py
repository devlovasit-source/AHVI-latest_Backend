
from typing import List, Dict, Any, Optional

from brain.engines.color_normalizer import color_normalizer
from brain.engines.memory_scorer import memory_scorer
from brain.engines.styling.palette_engine import palette_engine
from services.qdrant_service import qdrant_service


class WardrobeSelector:
    """
    🔥 ELITE WARDROBE SELECTOR

    Responsibilities:
    ✔ Smart item selection
    ✔ Embedding + heuristic scoring
    ✔ Robust fallback

    Guarantees:
    ✔ Always returns best possible item
    ✔ Never crashes on missing embeddings
    """

    # =========================
    # TYPE NORMALIZATION
    # =========================
    TYPE_MAP = {
        "tshirt": "top",
        "shirt": "top",
        "tee": "top",
        "top": "top",

        "jeans": "bottom",
        "pants": "bottom",
        "trousers": "bottom",
        "bottom": "bottom",

        "shoes": "footwear",
        "sneakers": "footwear",
        "loafers": "footwear",
        "heels": "footwear",
        "boots": "footwear",
        "sandals": "footwear",
        "footwear": "footwear",

        "dress": "dress",
        "dresses": "dress",
        "outerwear": "outerwear",
        "jacket": "outerwear",
        "coat": "outerwear",
    }

    def normalize_type(self, t: str) -> str:
        if not t:
            return ""
        t = t.lower().strip()
        return self.TYPE_MAP.get(t, t)

    def _context_score(self, item: Dict[str, Any], context: Dict[str, Any], outfit: List[Dict[str, Any]]) -> float:
        score = 0.0

        style_dna = context.get("style_dna", {}) or {}
        preferred_styles = style_dna.get("preferred_styles", []) or []
        preferred_styles = [str(x).strip().lower() for x in preferred_styles if str(x).strip()]

        item_style = str(item.get("style") or "").strip().lower()
        if item_style and item_style in preferred_styles:
            score += 0.4

        outfit_colors = []
        if isinstance(outfit, list):
            for i in outfit:
                if not isinstance(i, dict):
                    continue
                c = str(i.get("color") or i.get("color_code") or "").strip()
                if c:
                    outfit_colors.append(color_normalizer.normalize(c))

        item_color = color_normalizer.normalize(str(item.get("color") or item.get("color_code") or ""))
        if item_color and outfit_colors and item_color in outfit_colors:
            score += 0.5

        return score

    # =========================
    # MAIN API
    # =========================
    def find_best_match(
        self,
        target_type: str,
        context: Dict[str, Any],
        reference_embedding: Optional[List[float]] = None,
        preferred_colors: Optional[List[str]] = None,
        require_occasion: str | None = None,
    ) -> Optional[Dict]:

        wardrobe = context.get("wardrobe", [])
        if not wardrobe:
            print("⚠️ No wardrobe available")
            return None

        target_type = self.normalize_type(target_type)
        occasion = str(require_occasion or context.get("occasion") or "").strip().lower()

        # Palette-aware preferred colors (deterministic).
        palette_colors: List[str] = []
        try:
            palette = palette_engine.select_palette(
                {
                    "event": occasion or None,
                    "microtheme": (context.get("style_dna") or {}).get("primary_aesthetic"),
                }
            )
            palette_colors = [color_normalizer.normalize(c) for c in (palette.get("hex") or []) if c]
        except Exception:
            palette_colors = []

        preferred_norm = [color_normalizer.normalize(c) for c in (preferred_colors or []) if c]
        current_outfit = context.get("current_outfit", []) if isinstance(context.get("current_outfit"), list) else []

        # -------------------------
        # FILTER BY TYPE (SMART)
        # -------------------------
        def _get_item_type(row: Dict[str, Any]) -> str:
            return self.normalize_type(str(row.get("type") or row.get("sub_category") or ""))

        def _get_item_category(row: Dict[str, Any]) -> str:
            return self.normalize_type(str(row.get("category") or ""))

        candidates = []
        for w in wardrobe:
            if not isinstance(w, dict):
                continue
            item_type = _get_item_type(w)
            item_cat = _get_item_category(w)
            # Prefer strict matching first.
            if target_type and (item_type == target_type or item_cat == target_type):
                candidates.append(w)

        # Fallback: loosen matching if strict produces nothing.
        if not candidates:
            for w in wardrobe:
                if not isinstance(w, dict):
                    continue
                item_type = _get_item_type(w)
                item_cat = _get_item_category(w)
                if target_type and (target_type in item_type or target_type in item_cat):
                    candidates.append(w)

        if not candidates:
            print(f"⚠️ No candidates for type: {target_type}")
            return None

        # -------------------------
        # SCORING
        # -------------------------
        scored = []

        for item in candidates:
            score = 0.0

            # 🔥 EMBEDDING SCORE (PRIMARY)
            if reference_embedding and item.get("embedding"):
                try:
                    sim = qdrant_service.cosine_similarity(
                        reference_embedding,
                        item["embedding"]
                    )
                    score += sim * 0.8
                except Exception:
                    pass

            # 🔥 HEURISTIC BOOST
            if target_type in str(item.get("type", "")).lower():
                score += 0.2

            # slight preference for items with embeddings
            if item.get("embedding"):
                score += 0.05

            # Occasion match (when tags exist)
            if occasion:
                tags = item.get("occasion_tags") or item.get("occasions") or []
                if isinstance(tags, list):
                    tags_norm = [str(t).strip().lower() for t in tags if str(t).strip()]
                    if occasion in tags_norm:
                        score += 0.18

            # Palette / preferred color match (light boost)
            color_raw = item.get("color") or item.get("color_code") or ""
            color = color_normalizer.normalize(str(color_raw))
            if preferred_norm and color in preferred_norm:
                score += 0.22
            elif palette_colors and color in palette_colors:
                score += 0.12
            elif color in ["black", "white", "grey", "gray", "beige", "navy", "cream"]:
                score += 0.06

            # Context scoring: align with current outfit palette / user's style.
            score += self._context_score(item, context, current_outfit)

            # Memory: boost items aligned with user's past likes (bounded by MemoryScorer clamp).
            if item.get("embedding"):
                try:
                    mem = float(memory_scorer.score(item["embedding"], context))
                    # Keep bounded so memory doesn't drown out objective fit/palette signals.
                    mem = max(-1.0, min(mem, 1.0))
                    score += mem
                except Exception:
                    pass

            scored.append({
                "item": item,
                "score": score
            })

        # -------------------------
        # SORT
        # -------------------------
        scored.sort(key=lambda x: x["score"], reverse=True)

        best = scored[0]["item"]

        print(f"SELECTED ITEM → type: {target_type} | score: {scored[0]['score']:.3f}")

        return best

    # =========================
    # FALLBACK (SAFE)
    # =========================
    def fallback(self, wardrobe: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not wardrobe:
            return None

        # prefer items with embeddings
        for item in wardrobe:
            if item.get("embedding"):
                return item

        return wardrobe[0]


# singleton
wardrobe_selector = WardrobeSelector()
