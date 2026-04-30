
from typing import Dict, Any, List
import copy

from brain.engines.wardrobe_selector import wardrobe_selector
from brain.engines.style_scorer import style_scorer
from brain.engines.styling.palette_engine import palette_engine
from brain.engines.color_normalizer import color_normalizer


class RefinementEngine:
    """
    🔥 FINAL ELITE REFINEMENT ENGINE

    Responsibilities:
    ✔ apply chip transformations
    ✔ align with style DNA
    ✔ align with memory
    ✔ delegate item selection to wardrobe_selector

    DOES NOT:
    ❌ perform scoring
    ❌ rank candidates manually
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def apply(self, outfits: List[Dict], context: Dict[str, Any]) -> List[Dict]:

        if not outfits:
            return outfits

        mode = context.get("refinement")
        style_dna = context.get("style_dna", {}) or {}
        memory = context.get("user_memory", {}) or {}
        graph = context.get("style_graph", {}) or {}

        refined_outfits = []

        for outfit in outfits:
            new_outfit = copy.deepcopy(outfit)
            items = new_outfit.get("items", [])
            if not isinstance(items, list):
                items = []

            score_meta = new_outfit.get("score_meta", {}) if isinstance(new_outfit.get("score_meta"), dict) else {}
            reasons = score_meta.get("reasons", []) if isinstance(score_meta.get("reasons"), list) else []
            score = new_outfit.get("score", 0)
            try:
                score = float(score or 0.0)
            except Exception:
                score = 0.0

            base_snapshot = self._unified_snapshot(items, context, graph)
            try:
                base_score = float(base_snapshot.get("score") or 0.0)
            except Exception:
                base_score = 0.0

            # Prevent over-refinement: if already strong and no explicit mode requested, keep it.
            if (score > 7.5 or base_score >= 7.5) and not context.get("force_refine"):
                new_outfit["unified_style_refinement"] = base_snapshot
                refined_outfits.append(new_outfit)
                continue

            # -------------------------
            # 1. CHIP TRANSFORMATION
            # -------------------------
            if mode:
                items = self._apply_mode(items, mode)

            # -------------------------
            # 2. STYLE DNA ALIGNMENT
            # -------------------------
            items = self._apply_style_dna(items, style_dna)

            # -------------------------
            # 3. MEMORY ALIGNMENT
            # -------------------------
            items = self._apply_memory(items, memory)

            # -------------------------
            # 4. TARGETED CORRECTIONS (SCORE-AWARE)
            # -------------------------
            if "palette aligned" not in reasons:
                items = self._fix_palette(items)
            if "clean aesthetic balance" not in reasons:
                items = self._fix_silhouette(items)

            items = self._apply_targeted_corrections(
                items=items,
                outfit=new_outfit,
                context=context,
                style_dna=style_dna,
            )

            # Weather-triggered practical fix.
            if str((context.get("signals") or {}).get("weather_mode") or "").strip().lower() in ("hot", "summer", "heat", "warm"):
                items = self._make_breathable(items)
                # remove heavy fabrics if explicit (best-effort)
                items = [i for i in items if not (isinstance(i, dict) and str(i.get("fabric") or "").strip().lower() == "heavy")]

            # -------------------------
            # 5. REAL WARDROBE SWAP
            # -------------------------
            items = self._apply_wardrobe_swap(items, context)

            # Final: keep only improvements (or neutral changes) to avoid "random mutations".
            final_snapshot = self._unified_snapshot(items, context, graph)
            try:
                before = float(base_snapshot.get("score") or 0.0)
                after = float(final_snapshot.get("score") or 0.0)
            except Exception:
                before, after = 0.0, 0.0

            if after + 0.10 < before:
                # Revert when we made it worse.
                items = new_outfit.get("items", []) if isinstance(new_outfit.get("items"), list) else items
                final_snapshot = base_snapshot

            new_outfit["items"] = items
            new_outfit["refined"] = mode or "auto"
            new_outfit["unified_style_refinement"] = final_snapshot

            refined_outfits.append(new_outfit)

        return refined_outfits

    def _fix_palette(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        neutrals = {"black", "white", "beige", "grey", "gray", "navy", "cream"}
        for i in items:
            if not isinstance(i, dict):
                continue
            c = str(i.get("color") or i.get("color_code") or "").strip().lower()
            if c and c not in neutrals:
                # Hint for selector: keep the same base type but bias toward neutral colors.
                i["preferred_replacement"] = self._base_type(i) or "top"
                i["preferred_colors"] = ["black", "white", "beige", "grey", "navy", "cream"]
        return items

    def _fix_silhouette(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i in items:
            if not isinstance(i, dict):
                continue
            # Some wardrobes store fit tags; best-effort hinting.
            if str(i.get("fit") or "").strip().lower() == "oversized":
                i["preferred_fit"] = "regular"
        return items

    def _make_breathable(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Lightweight guidance: avoid heavy layering pieces when hot.
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            t = str(it.get("type") or "").lower()
            if any(x in t for x in ["coat", "jacket", "outerwear", "blazer"]):
                # Mark for swap to a lighter top instead of removing outright.
                it["preferred_replacement"] = "top"
            out.append(it)
        return out

    def _unified_snapshot(self, items: List[Dict[str, Any]], context: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return style_scorer.score_outfit(items=items, context=context, graph=graph)
        except Exception:
            return {"score": 0.0, "label": "Basic", "reasons": []}

    def _base_type(self, item: Dict[str, Any]) -> str:
        t = str(item.get("type", "") or "").lower().strip()
        if any(x in t for x in ["shoe", "sneaker", "heel", "boot", "loafer", "sandal"]):
            return "footwear"
        if "dress" in t:
            return "dress"
        if any(x in t for x in ["jacket", "coat", "blazer", "outerwear"]):
            return "outerwear"
        if any(x in t for x in ["pant", "jean", "trouser", "chino", "skirt", "bottom"]):
            return "bottom"
        if any(x in t for x in ["shirt", "tshirt", "t-shirt", "tee", "top", "hoodie", "sweater"]):
            return "top"
        return ""

    def _preferred_colors(self, context: Dict[str, Any], style_dna: Dict[str, Any]) -> List[str]:
        occasion = str(context.get("occasion") or "").strip().lower()
        palette_hexes: List[str] = []
        try:
            palette = palette_engine.select_palette(
                {"event": occasion or None, "microtheme": style_dna.get("primary_aesthetic")}
            )
            palette_hexes = [str(x).strip() for x in (palette.get("hex") or []) if str(x).strip()]
        except Exception:
            palette_hexes = []

        preferred = []
        if isinstance(style_dna.get("preferred_colors"), list):
            preferred.extend([str(x).strip() for x in style_dna.get("preferred_colors") if str(x).strip()])
        preferred.extend(palette_hexes)
        # Normalize for matching.
        return [color_normalizer.normalize(c) for c in preferred if c]

    def _pattern_list(self, items: List[Dict[str, Any]]) -> List[str]:
        pats = []
        for it in items:
            if not isinstance(it, dict):
                continue
            p = str(it.get("pattern") or "").strip().lower()
            if p:
                pats.append(p)
        return pats

    def _detect_weakness(self, outfit: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
        breakdown = outfit.get("score_breakdown") if isinstance(outfit.get("score_breakdown"), dict) else {}
        try:
            color_intel = float(breakdown.get("color_intelligence") or 0.0)
        except Exception:
            color_intel = 0.0
        try:
            occ = float(breakdown.get("occasion_rules") or 0.0)
        except Exception:
            occ = 0.0
        try:
            layering = float(breakdown.get("layering") or 0.0)
        except Exception:
            layering = 0.0

        pats = self._pattern_list(items)
        non_plain = [p for p in pats if p not in ("plain", "solid", "none")]
        pattern_busy = len(set(non_plain)) >= 2

        if color_intel < 0.5:
            return "color"
        if occ < 0.5:
            return "occasion"
        if pattern_busy:
            return "pattern"
        if layering < 0.2:
            return "layering"
        return ""

    def _apply_targeted_corrections(self, *, items: List[Dict[str, Any]], outfit: Dict[str, Any], context: Dict[str, Any], style_dna: Dict[str, Any]) -> List[Dict[str, Any]]:
        wardrobe = context.get("wardrobe", [])
        if not isinstance(wardrobe, list) or not wardrobe:
            return items

        weakness = self._detect_weakness(outfit, items)
        if not weakness:
            return items

        occasion = str(context.get("occasion") or "").strip().lower()
        preferred_colors = self._preferred_colors(context, style_dna)
        neutrals = ["black", "white", "grey", "gray", "beige", "navy", "cream"]

        # Decide which item to swap (bounded to 1 swap here; closed-loop will run again upstream if needed).
        swap_index = -1
        swap_target_type = ""

        if weakness == "layering":
            # If cold and no outerwear, try to add an outerwear piece.
            has_outer = any(self._base_type(it) == "outerwear" for it in items if isinstance(it, dict))
            if not has_outer and str((context.get("signals") or {}).get("weather_mode") or "").strip().lower() in ("cold", "winter", "chilly"):
                candidate = wardrobe_selector.find_best_match(
                    "outerwear",
                    context,
                    preferred_colors=preferred_colors or neutrals,
                    require_occasion=occasion or None,
                )
                if candidate:
                    return items + [candidate]
            return items

        # For color/occasion/pattern: pick the most "off" item.
        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            base = self._base_type(it)
            if not base:
                continue

            if weakness == "occasion":
                if base == "footwear":
                    swap_index = idx
                    swap_target_type = "footwear"
                    break
                continue

            if weakness == "pattern":
                p = str(it.get("pattern") or "").strip().lower()
                if p and p not in ("plain", "solid", "none"):
                    swap_index = idx
                    swap_target_type = base
                    break
                continue

            if weakness == "color":
                c = color_normalizer.normalize(str(it.get("color") or it.get("color_code") or ""))
                if preferred_colors and c and c not in preferred_colors and c not in neutrals:
                    swap_index = idx
                    swap_target_type = base
                    break

        if swap_index < 0 or not swap_target_type:
            return items

        candidate = wardrobe_selector.find_best_match(
            swap_target_type,
            context,
            reference_embedding=items[swap_index].get("embedding") if isinstance(items[swap_index], dict) else None,
            preferred_colors=(preferred_colors or neutrals) if weakness in ("color", "pattern", "occasion") else None,
            require_occasion=occasion if weakness == "occasion" else None,
        )
        if not candidate:
            return items

        updated = list(items)
        updated[swap_index] = candidate
        return updated

    # =========================
    # CHIP MODE
    # =========================
    def _apply_mode(self, items, mode):

        if mode == "sharp":
            return self._make_sharp(items)

        if mode == "relaxed":
            return self._make_relaxed(items)

        if mode == "bold":
            return self._make_bold(items)

        if mode == "minimal":
            return self._make_minimal(items)

        return items

    # -------------------------
    # SHARP
    # -------------------------
    def _make_sharp(self, items):
        for i in items:
            t = str(i.get("type", "")).lower()

            if "hoodie" in t or "tshirt" in t:
                i["preferred_replacement"] = "shirt"
                i["style"] = "structured"

            if "sneaker" in t:
                i["preferred_replacement"] = "loafers"
                i["style"] = "formal"

        return items

    # -------------------------
    # RELAXED
    # -------------------------
    def _make_relaxed(self, items):
        for i in items:
            t = str(i.get("type", "")).lower()

            if "shirt" in t:
                i["preferred_replacement"] = "tshirt"
                i["style"] = "casual"

            if "loafers" in t:
                i["preferred_replacement"] = "sneakers"
                i["style"] = "casual"

        return items

    # -------------------------
    # BOLD
    # -------------------------
    def _make_bold(self, items):
        for i in items:
            i["pattern"] = "statement"
            i["color"] = i.get("color") or "red"
        return items

    # -------------------------
    # MINIMAL
    # -------------------------
    def _make_minimal(self, items):
        for i in items:
            i["pattern"] = "solid"
            if i.get("color") not in ["black", "white", "beige", "grey"]:
                i["color"] = "neutral"
        return items

    # =========================
    # STYLE DNA
    # =========================
    def _apply_style_dna(self, items, dna):

        if not dna:
            return items

        preferred_styles = dna.get("preferred_styles", [])
        preferred_colors = dna.get("preferred_colors", [])

        for i in items:
            if preferred_styles:
                i["style_boost"] = i.get("style") in preferred_styles

            if preferred_colors:
                i["color_boost"] = i.get("color") in preferred_colors

        return items

    # =========================
    # MEMORY
    # =========================
    def _apply_memory(self, items, memory):

        signals = memory.get("memory_signals", {})

        liked_styles = signals.get("preferred_styles", [])
        disliked_items = signals.get("disliked_items", [])

        for i in items:
            t = str(i.get("type", "")).lower()

            if t in disliked_items:
                i["avoid"] = True

            if i.get("style") in liked_styles:
                i["memory_boost"] = True

        return items

    # =========================
    # WARDROBE SWAP (FIXED)
    # =========================
    def _apply_wardrobe_swap(self, items, context):

        swapped = []

        for i in items:

            replacement_type = i.get("preferred_replacement")
            preferred_colors = i.get("preferred_colors") if isinstance(i, dict) and isinstance(i.get("preferred_colors"), list) else None

            if not replacement_type:
                swapped.append(i)
                continue

            # 🔥 USE CENTRAL SELECTOR
            candidate = wardrobe_selector.find_best_match(
                replacement_type,
                context,
                reference_embedding=i.get("embedding") if isinstance(i, dict) else None,
                preferred_colors=preferred_colors,
                require_occasion=str(context.get("occasion") or "").strip().lower() or None,
            )

            if candidate:
                swapped.append(candidate)
            else:
                swapped.append(i)

        return swapped


# singleton
refinement_engine = RefinementEngine()
