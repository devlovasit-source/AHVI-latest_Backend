import re
from typing import List, Dict, Any, Optional

from brain.engines.color_normalizer import color_normalizer
from brain.engines.memory_scorer import memory_scorer
from brain.engines.styling.palette_engine import palette_engine
from services.qdrant_service import qdrant_service


class WardrobeSelector:
    """
    ELITE WARDROBE SELECTOR

    Responsibilities:
    - Smart item selection
    - Embedding + heuristic scoring
    - Robust fallback

    Guarantees:
    - Always returns best possible item
    - Never crashes on missing embeddings
    """

    # =========================
    # TYPE NORMALIZATION
    # =========================
    TYPE_MAP = {
        "tshirt": "top",
        "t-shirt": "top",
        "tshirts": "top",
        "tee": "top",
        "tees": "top",
        "shirt": "top",
        "shirts": "top",
        "top": "top",
        "tops": "top",
        "polo": "top",
        "polos": "top",
        "kurta": "top",
        "kurtas": "top",
        "hoodie": "top",
        "hoodies": "top",
        "blouse": "top",
        "sweater": "top",
        "tank": "top",
        "camisole": "top",

        "bottom": "bottom",
        "bottoms": "bottom",
        "pant": "bottom",
        "pants": "bottom",
        "trouser": "bottom",
        "trousers": "bottom",
        "jean": "bottom",
        "jeans": "bottom",
        "short": "bottom",
        "shorts": "bottom",
        "chino": "bottom",
        "chinos": "bottom",
        "skirt": "bottom",
        "leggings": "bottom",
        "joggers": "bottom",

        "shoe": "footwear",
        "shoes": "footwear",
        "sneaker": "footwear",
        "sneakers": "footwear",
        "loafer": "footwear",
        "loafers": "footwear",
        "heel": "footwear",
        "heels": "footwear",
        "boot": "footwear",
        "boots": "footwear",
        "sandal": "footwear",
        "sandals": "footwear",
        "slipper": "footwear",
        "slippers": "footwear",
        "footwear": "footwear",

        "dress": "dress",
        "dresses": "dress",
        "gown": "dress",
        "saree": "dress",
        "sari": "dress",
        "jumpsuit": "dress",

        "outerwear": "outerwear",
        "jacket": "outerwear",
        "jackets": "outerwear",
        "coat": "outerwear",
        "coats": "outerwear",
        "blazer": "outerwear",
        "blazers": "outerwear",
        "cardigan": "outerwear",
        "shrug": "outerwear",

        "accessory": "accessory",
        "accessories": "accessory",
        "watch": "accessory",
        "watches": "accessory",
        "belt": "accessory",
        "belts": "accessory",
        "bag": "accessory",
        "bags": "accessory",
        "cap": "accessory",
        "caps": "accessory",
        "hat": "accessory",
        "hats": "accessory",
        "jewelry": "accessory",
        "jewellery": "accessory",
        "bracelet": "accessory",
        "necklace": "accessory",
        "ring": "accessory",
        "rings": "accessory",
        "earring": "accessory",
        "earrings": "accessory",
        "sunglasses": "accessory",
        "eyewear": "accessory",
    }

    def normalize_type(self, t: str) -> str:
        if not t:
            return ""
        t = str(t).lower().strip()
        return self.TYPE_MAP.get(t, t)

    def normalize_item_role(self, row: Dict[str, Any]) -> str:
        """
        Normalize a wardrobe item into one of:
        top, bottom, footwear, accessory, dress, outerwear.

        This avoids substring bugs like:
        - "Short-Sleeved Shirt" being treated as bottom because of "short"
        - "Dress Shirt" being treated as dress
        """
        if not isinstance(row, dict):
            return ""

        explicit_fields = (
            "role",
            "slot",
            "type",
            "category",
            "cat",
            "category_group",
            "sub_category",
            "subcategory",
            "subCategory",
            "garment_type",
        )

        valid_roles = {
            "top",
            "bottom",
            "footwear",
            "accessory",
            "dress",
            "outerwear",
        }

        # 1. Trust explicit structured fields first.
        for key in explicit_fields:
            value = str(row.get(key) or "").strip().lower()
            normalized = self.normalize_type(value)
            if normalized in valid_roles:
                return normalized

        # 2. Fallback to text fields with token-aware matching.
        text_fields = (
            "name",
            "label",
            "title",
            "description",
        )

        blob = " ".join(str(row.get(k) or "") for k in text_fields).lower()
        tokens = set(re.sub(r"[^a-z0-9]+", " ", blob).split())

        # Important phrase guards.
        if "dress shirt" in blob:
            return "top"

        if "shirt dress" in blob or "maxi dress" in blob or "mini dress" in blob:
            return "dress"

        # Tops should win over misleading words like "short" or "dress" in shirt names.
        top_tokens = {
            "shirt",
            "shirts",
            "tshirt",
            "tshirts",
            "tee",
            "tees",
            "top",
            "tops",
            "polo",
            "polos",
            "kurta",
            "kurtas",
            "hoodie",
            "hoodies",
            "blouse",
            "sweater",
            "tank",
            "camisole",
        }

        bottom_tokens = {
            "jeans",
            "jean",
            "pants",
            "pant",
            "trousers",
            "trouser",
            "chinos",
            "chino",
            "shorts",
            "skirt",
            "leggings",
            "joggers",
            "bottom",
            "bottoms",
        }

        footwear_tokens = {
            "footwear",
            "shoe",
            "shoes",
            "sneaker",
            "sneakers",
            "boot",
            "boots",
            "loafer",
            "loafers",
            "sandal",
            "sandals",
            "heel",
            "heels",
            "slipper",
            "slippers",
        }

        accessory_tokens = {
            "watch",
            "watches",
            "belt",
            "belts",
            "bag",
            "bags",
            "cap",
            "caps",
            "hat",
            "hats",
            "jewelry",
            "jewellery",
            "bracelet",
            "necklace",
            "ring",
            "rings",
            "earring",
            "earrings",
            "sunglasses",
            "eyewear",
        }

        dress_tokens = {
            "dress",
            "dresses",
            "gown",
            "saree",
            "sari",
            "jumpsuit",
        }

        outerwear_tokens = {
            "jacket",
            "jackets",
            "coat",
            "coats",
            "outerwear",
            "blazer",
            "blazers",
            "cardigan",
            "shrug",
        }

        if tokens & top_tokens:
            return "top"

        if tokens & bottom_tokens:
            return "bottom"

        if tokens & footwear_tokens:
            return "footwear"

        if tokens & accessory_tokens:
            return "accessory"

        # Dress only after dress-shirt and top checks.
        if tokens & dress_tokens:
            return "dress"

        if tokens & outerwear_tokens:
            return "outerwear"

        return ""

    def _context_score(
        self,
        item: Dict[str, Any],
        context: Dict[str, Any],
        outfit: List[Dict[str, Any]],
    ) -> float:
        score = 0.0

        style_dna = context.get("style_dna", {}) or {}
        preferred_styles = style_dna.get("preferred_styles", []) or []
        preferred_styles = [
            str(x).strip().lower() for x in preferred_styles if str(x).strip()
        ]

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

        item_color = color_normalizer.normalize(
            str(item.get("color") or item.get("color_code") or "")
        )
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
            print("No wardrobe available")
            return None

        target_type = self.normalize_type(target_type)
        occasion = (
            str(require_occasion or context.get("occasion") or "").strip().lower()
        )

        # Palette-aware preferred colors.
        palette_colors: List[str] = []
        try:
            palette = palette_engine.select_palette(
                {
                    "event": occasion or None,
                    "microtheme": (context.get("style_dna") or {}).get(
                        "primary_aesthetic"
                    ),
                }
            )
            palette_colors = [
                color_normalizer.normalize(c) for c in (palette.get("hex") or []) if c
            ]
        except Exception:
            palette_colors = []

        preferred_norm = [
            color_normalizer.normalize(c) for c in (preferred_colors or []) if c
        ]

        current_outfit = (
            context.get("current_outfit", [])
            if isinstance(context.get("current_outfit"), list)
            else []
        )

        # -------------------------
        # FILTER BY TYPE
        # -------------------------
        def _get_item_type(row: Dict[str, Any]) -> str:
            return self.normalize_item_role(row)

        def _get_item_category(row: Dict[str, Any]) -> str:
            return self.normalize_item_role(row)

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

                if target_type and (
                    target_type in item_type or target_type in item_cat
                ):
                    candidates.append(w)

        if not candidates:
            print(f"No candidates for type: {target_type}")
            return None

        # -------------------------
        # SCORING
        # -------------------------
        scored = []

        for item in candidates:
            score = 0.0

            # Embedding score.
            if reference_embedding and item.get("embedding"):
                try:
                    sim = qdrant_service.cosine_similarity(
                        reference_embedding, item["embedding"]
                    )
                    score += sim * 0.8
                except Exception:
                    pass

            # Heuristic boost.
            item_role = self.normalize_item_role(item)
            if target_type and item_role == target_type:
                score += 0.2

            # Slight preference for items with embeddings.
            if item.get("embedding"):
                score += 0.05

            # Occasion match when tags exist.
            if occasion:
                tags = item.get("occasion_tags") or item.get("occasions") or []
                if isinstance(tags, list):
                    tags_norm = [str(t).strip().lower() for t in tags if str(t).strip()]
                    if occasion in tags_norm:
                        score += 0.18

            # Palette / preferred color match.
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

            # Memory: boost items aligned with user's past likes.
            if item.get("embedding"):
                try:
                    mem = float(memory_scorer.score(item["embedding"], context))
                    mem = max(-1.0, min(mem, 1.0))
                    score += mem
                except Exception:
                    pass

            scored.append({"item": item, "score": score})

        # -------------------------
        # SORT
        # -------------------------
        scored.sort(key=lambda x: x["score"], reverse=True)

        best = scored[0]["item"]

        print(f"SELECTED ITEM -> type: {target_type} | score: {scored[0]['score']:.3f}")

        return best

    # =========================
    # FALLBACK
    # =========================
    def fallback(self, wardrobe: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # Production-safe: do not fabricate a match by returning the first wardrobe item.
        # Missing match should surface as missing data, not random outfit selection.
        return None


wardrobe_selector = WardrobeSelector()
