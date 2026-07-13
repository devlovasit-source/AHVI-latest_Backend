"""Deterministic Style DNA derived from durable request context."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List


logger = logging.getLogger("ahvi.style_dna")


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_gender(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"m", "male", "man", "men", "mens", "boy"}:
        return "male"
    if raw in {"f", "female", "woman", "women", "womens", "girl", "ladies"}:
        return "female"
    if raw in {"unisex", "neutral", "genderless", "any"}:
        return "unisex"
    return ""


def _gender_from_profile(profile: Dict[str, Any], explicit_dna: Dict[str, Any]) -> str:
    prefs = _coerce_dict(profile.get("preferences"))
    style_prefs = _coerce_dict(
        profile.get("style_preferences") or profile.get("stylePreference")
    )
    for value in (
        profile.get("style_gender"), profile.get("gender"),
        profile.get("preferred_gender"), profile.get("target_gender"),
        prefs.get("style_gender"), prefs.get("gender"),
        prefs.get("preferred_gender"), prefs.get("target_gender"),
        style_prefs.get("style_gender"), style_prefs.get("gender"),
        style_prefs.get("preferred_gender"), style_prefs.get("target_gender"),
        explicit_dna.get("style_gender"), explicit_dna.get("gender"),
    ):
        gender = _normalize_gender(value)
        if gender:
            return gender
    return "unisex"


class StyleDNAEngine:
    """Stateless personalization derived from already-loaded durable inputs.

    No local file is read or written. Identical inputs therefore produce the
    same DNA after a restart and across independent Cloud Run instances.
    """

    def build(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = context if isinstance(context, dict) else {}
        profile = _coerce_dict(ctx.get("user_profile") or ctx.get("profile"))
        preferences = self._preferences(profile, ctx)
        explicit_dna = _coerce_dict(
            ctx.get("style_dna") or profile.get("style_dna") or profile.get("styleDNA")
        )
        memory = self._memory(ctx)
        wardrobe = self._list(ctx.get("wardrobe_items") or ctx.get("wardrobe"))
        dna = self._derive(profile, preferences, explicit_dna, memory, wardrobe)
        logger.info(
            "AHVI_STYLE_DNA_DERIVED dna_signal_count=%d dna_confidence=%.2f "
            "durable_feedback_used=%s saved_memory_used=%s wear_memory_used=%s "
            "personalization_degraded=%s",
            dna["dna_signal_count"], dna["confidence"],
            dna["durable_feedback_used"], dna["saved_memory_used"],
            dna["wear_memory_used"], dna["personalization_degraded"],
        )
        return dna

    def enrich_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = dict(context or {})
        ctx["style_dna"] = self.build(ctx)
        return ctx

    def _derive(
        self,
        profile: Dict[str, Any],
        prefs: Dict[str, Any],
        explicit_dna: Dict[str, Any],
        memory: Dict[str, Any],
        wardrobe: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        color_dna = _coerce_dict(explicit_dna.get("color_dna"))
        silhouette_dna = _coerce_dict(explicit_dna.get("silhouette_dna"))

        explicit_colors = self._merge_unique(
            self._values(profile.get("preferred_colors")),
            self._values(prefs.get("preferred_colors") or prefs.get("colors")),
            self._values(explicit_dna.get("preferred_colors")),
            self._values(color_dna.get("core_colors") or color_dna.get("power_colors")),
        )
        avoided_colors = self._merge_unique(
            self._values(profile.get("avoided_colors")),
            self._values(prefs.get("avoided_colors")),
            self._values(explicit_dna.get("avoided_colors")),
            self._values(color_dna.get("avoided_colors")),
        )
        explicit_styles = self._merge_unique(
            self._values(
                profile.get("preferred_styles")
                or profile.get("stylePreferences")
                or profile.get("style")
            ),
            self._values(prefs.get("preferred_styles") or prefs.get("archetypes")),
            self._values(prefs.get("style_keywords")),
            self._values(explicit_dna.get("style_archetypes")),
            self._values(explicit_dna.get("preferred_styles")),
            self._values(explicit_dna.get("preferred_style_keywords")),
        )
        explicit_types = self._merge_unique(
            self._values(profile.get("preferred_categories")),
            self._values(prefs.get("preferred_categories") or prefs.get("categories")),
            self._values(explicit_dna.get("preferred_types")),
        )
        explicit_dislikes = self._merge_unique(
            self._values(profile.get("disliked_items") or profile.get("avoided_categories")),
            self._values(prefs.get("disliked_items") or prefs.get("avoided_categories")),
            self._values(explicit_dna.get("disliked_items")),
        )
        explicit_avoid_keywords = self._merge_unique(
            self._values(prefs.get("avoid_keywords")),
            self._values(explicit_dna.get("avoid_style_keywords")),
        )

        liked_ids = set(self._values(memory.get("liked_item_ids")))
        disliked_ids = set(self._values(memory.get("disliked_item_ids")))
        wear_counts = _coerce_dict(memory.get("wear_counts"))
        wardrobe_by_id: Dict[str, Dict[str, Any]] = {}
        for item in wardrobe:
            if not isinstance(item, dict):
                continue
            raw = _coerce_dict(item.get("_raw"))
            item_id = self._clean(
                item.get("id") or item.get("$id") or raw.get("$id") or raw.get("id")
            )
            if item_id:
                wardrobe_by_id[item_id] = {**raw, **item}

        liked_colors: List[str] = []
        liked_types: List[str] = []
        disliked_types: List[str] = []
        worn_types: List[str] = []
        for item_id, item in wardrobe_by_id.items():
            category = self._clean(item.get("category") or item.get("type"))
            color = self._clean(item.get("color") or item.get("colour"))
            if item_id in liked_ids:
                liked_types.append(category)
                liked_colors.append(color)
            if item_id in disliked_ids:
                disliked_types.append(category or self._clean(item.get("name")))
            try:
                worn = int(wear_counts.get(item_id) or 0) > 0
            except Exception:
                worn = False
            if worn:
                worn_types.append(category)

        liked_patterns = self._values(memory.get("liked_board_patterns"))
        disliked_patterns = self._values(memory.get("disliked_board_patterns"))
        saved_patterns = self._values(memory.get("saved_board_patterns"))
        preferred_colors = self._merge_unique(
            explicit_colors, self._values(memory.get("favorite_colors")), liked_colors
        )
        preferred_styles = self._merge_unique(
            explicit_styles, liked_patterns, saved_patterns
        )
        preferred_types = self._merge_unique(
            explicit_types, self._values(memory.get("favorite_categories")),
            liked_types, worn_types,
        )
        disliked_items = self._merge_unique(explicit_dislikes, disliked_types)
        avoid_keywords = self._merge_unique(explicit_avoid_keywords, disliked_patterns)

        explicit_signals = self._tagged("explicit", [
            *explicit_colors, *avoided_colors, *explicit_styles, *explicit_types,
            *explicit_dislikes, *explicit_avoid_keywords,
        ])
        feedback_signals = self._tagged("feedback", [
            *liked_ids, *disliked_ids, *liked_patterns, *disliked_patterns,
        ])
        saved_signals = self._tagged("saved", [
            *self._values(memory.get("saved_item_ids")), *saved_patterns,
        ])
        wear_signals = self._tagged("wear", [
            *self._values(memory.get("recently_worn_ids")),
            *[key for key, value in wear_counts.items() if value],
        ])
        all_signals = explicit_signals | feedback_signals | saved_signals | wear_signals
        points = (
            3.0 * min(len(explicit_signals), 8)
            + 2.0 * min(len(feedback_signals), 8)
            + 1.5 * min(len(saved_signals), 6)
            + 1.0 * min(len(wear_signals), 6)
        )
        confidence = round(min(1.0, points / 30.0), 2) if all_signals else 0.0
        primary = preferred_styles[0] if preferred_styles else ""
        meta = _coerce_dict(memory.get("_personalization_meta"))
        gender = _gender_from_profile(profile, explicit_dna)

        return {
            "style": primary,
            "style_archetypes": preferred_styles[:8],
            "preferred_colors": preferred_colors[:10],
            "avoided_colors": avoided_colors[:10],
            "preferred_styles": preferred_styles[:8],
            "preferred_style_keywords": preferred_styles[:8],
            "avoid_style_keywords": avoid_keywords[:10],
            "preferred_silhouettes": self._merge_unique(
                self._values(prefs.get("silhouettes")),
                self._values(
                    silhouette_dna.get("preferred_fits")
                    or silhouette_dna.get("preferred_shapes")
                ),
            )[:8],
            "preferred_types": preferred_types[:10],
            "disliked_items": disliked_items[:10],
            "primary_aesthetic": primary,
            "secondary_aesthetics": preferred_styles[1:3],
            "confidence": confidence,
            "dna_signal_count": len(all_signals),
            "durable_feedback_used": bool(feedback_signals),
            "saved_memory_used": bool(saved_signals),
            "wear_memory_used": bool(wear_signals),
            "personalization_degraded": bool(meta.get("personalization_degraded")),
            "gender": gender,
            "style_gender": gender,
        }

    @staticmethod
    def _merge_unique(*groups: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for group in groups:
            if not isinstance(group, list):
                continue
            for value in group:
                key = " ".join(str(value or "").strip().lower().split())
                if key and key not in seen:
                    seen.add(key)
                    result.append(key)
        return result

    @staticmethod
    def _list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            for key in ("items", "documents", "wardrobe", "data"):
                if isinstance(value.get(key), list):
                    return list(value[key])
        return []

    @classmethod
    def _values(cls, value: Any) -> List[str]:
        if isinstance(value, dict):
            ranked = sorted(
                value.items(),
                key=lambda pair: float(pair[1])
                if isinstance(pair[1], (int, float)) else 0.0,
                reverse=True,
            )
            return [cls._clean(key) for key, weight in ranked if cls._clean(key) and weight]
        if isinstance(value, (list, tuple, set)):
            return [cls._clean(item) for item in value if cls._clean(item)]
        if isinstance(value, str) and value.strip():
            return [cls._clean(part) for part in value.split(",") if cls._clean(part)]
        return []

    @staticmethod
    def _clean(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @classmethod
    def _preferences(cls, profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        for candidate in (
            context.get("preferences"), profile.get("style_preferences"),
            profile.get("preferences"), profile.get("style"),
        ):
            parsed = _coerce_dict(candidate)
            if parsed:
                return parsed
        if isinstance(profile.get("stylePreferences"), list):
            return {"archetypes": profile.get("stylePreferences")}
        return {}

    @classmethod
    def _memory(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        memory = _coerce_dict(context.get("memory"))
        for key in (
            "recently_worn_ids", "underworn_ids", "wear_counts", "saved_item_ids",
            "saved_board_patterns", "favorite_colors", "favorite_categories",
            "liked_item_ids", "disliked_item_ids", "liked_board_patterns",
            "disliked_board_patterns", "_personalization_meta",
        ):
            if key in context and key not in memory:
                memory[key] = context[key]
        return memory

    @staticmethod
    def _tagged(source: str, values: Iterable[Any]) -> set[str]:
        return {
            f"{source}:{str(value).strip().lower()}"
            for value in values
            if str(value or "").strip()
        }


style_dna_engine = StyleDNAEngine()
