import json
import os
from collections import Counter
from threading import Lock
from typing import Any, Dict, List

from services.appwrite_proxy import AppwriteProxy


class StyleDNAEngine:
    """
    🔥 FINAL STYLE DNA ENGINE

    - Learns from profile + history + feedback + memory
    - Applies recency weighting
    - Extracts aesthetics
    - Produces confidence score
    """

    def __init__(self) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._dna_path = os.path.join(base_dir, "data", "style_dna_memory.json")
        self._feedback_memory_path = os.path.join(
            base_dir, "data", "outfit_memory.json"
        )
        self._lock = Lock()

    def build(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(context.get("user_id") or "anonymous")
        profile = context.get("user_profile") or {}
        history = context.get("history") or []
        memory = context.get("memory") or {}  # 🔥 NEW

        with self._lock:
            dna_state = self._load_json(self._dna_path, fallback={"users": {}})
            feedback_memory = self._load_json(
                self._feedback_memory_path, fallback={"users": {}}
            )

            learned_dna = self._build_dna(
                profile=profile,
                history=history,
                previous_dna=(
                    (dna_state.get("users", {}) or {}).get(user_id, {}) or {}
                ),
                feedback_user=(
                    (feedback_memory.get("users", {}) or {}).get(user_id, {}) or {}
                ),
                memory=memory,  # 🔥 NEW
            )

            dna_state.setdefault("users", {})[user_id] = learned_dna
            self._save_json(self._dna_path, dna_state)

            return learned_dna

    def enrich_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["style_dna"] = self.build(context)
        return context

    # =========================
    # 🔥 CORE DNA BUILDER
    # =========================
    def _build_dna(
        self,
        profile: Dict[str, Any],
        history: List[Dict[str, Any]],
        previous_dna: Dict[str, Any],
        feedback_user: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> Dict[str, Any]:

        profile = profile or {}
        previous_dna = previous_dna or {}
        feedback_user = feedback_user or {}

        memory_signals = memory.get("memory_signals", {})

        liked = feedback_user.get("liked_outfits", []) or []
        disliked = feedback_user.get("disliked_outfits", []) or []

        liked_colors = Counter()
        liked_fabrics = Counter()
        liked_types = Counter()
        disliked_types = Counter()

        # =========================
        # 🔥 RECENCY WEIGHTED LEARNING
        # =========================
        for i, outfit in enumerate(liked[:80]):
            weight = 1 + (0.02 * (80 - i))

            for part in ("top", "bottom", "shoes"):
                item = outfit.get(part, {}) if isinstance(outfit, dict) else {}

                liked_colors.update([str(item.get("color", "")).lower()] * int(weight))
                liked_fabrics.update(
                    [str(item.get("fabric", "")).lower()] * int(weight)
                )
                liked_types.update(
                    [str(item.get("type") or item.get("category") or "").lower()]
                    * int(weight)
                )

        for outfit in disliked[:80]:
            for part in ("top", "bottom", "shoes"):
                item = outfit.get(part, {}) if isinstance(outfit, dict) else {}
                disliked_types.update(
                    [str(item.get("type") or item.get("category") or "").lower()]
                )

        # =========================
        # HISTORY
        # =========================
        history_styles = Counter()

        for event in history[-40:]:
            if not isinstance(event, dict):
                continue

            style_value = str(event.get("style", "")).lower()
            if style_value:
                history_styles.update([style_value])

        def _top(counter: Counter, n: int = 5):
            return [k for k, _ in counter.most_common(n) if k]

        # =========================
        # 🔥 MERGE ALL SIGNALS
        # =========================
        preferred_colors = self._merge_unique(
            profile.get("preferred_colors", []),
            previous_dna.get("preferred_colors", []),
            _top(liked_colors, 6),
            memory_signals.get("liked_colors", []),  # 🔥 NEW
        )

        preferred_styles = self._merge_unique(
            profile.get("preferred_styles", []),
            previous_dna.get("preferred_styles", []),
            _top(history_styles, 4),
            memory_signals.get("preferred_styles", []),  # 🔥 NEW
        )

        preferred_types = self._merge_unique(
            previous_dna.get("preferred_types", []),
            _top(liked_types, 8),
        )

        disliked_items = self._merge_unique(
            profile.get("disliked_items", []),
            previous_dna.get("disliked_items", []),
            _top(disliked_types, 8),
            memory_signals.get("disliked_items", []),  # 🔥 NEW
        )

        # =========================
        # 🔥 AESTHETIC EXTRACTION
        # =========================
        aesthetic_counter = Counter(preferred_styles)

        primary_aesthetic = (
            aesthetic_counter.most_common(1)[0][0] if aesthetic_counter else "casual"
        )

        secondary_aesthetics = [k for k, _ in aesthetic_counter.most_common(3)][1:]

        # =========================
        # 🔥 CONFIDENCE
        # =========================
        confidence = min(1.0, len(liked) / 50)

        return {
            "style": primary_aesthetic,
            "preferred_colors": preferred_colors[:10],
            "preferred_styles": preferred_styles[:8],
            "preferred_types": preferred_types[:10],
            "disliked_items": disliked_items[:10],
            # 🔥 NEW
            "primary_aesthetic": primary_aesthetic,
            "secondary_aesthetics": secondary_aesthetics,
            "confidence": round(confidence, 2),
        }

    # =========================
    # HELPERS
    # =========================
    @staticmethod
    def _merge_unique(*groups: List[str]) -> List[str]:
        result = []
        seen = set()

        for group in groups:
            if not isinstance(group, list):
                continue

            for value in group:
                key = str(value).strip().lower()

                if not key or key in seen:
                    continue

                seen.add(key)
                result.append(key)

        return result

    @staticmethod
    def _load_json(path: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not os.path.exists(path):
            return dict(fallback)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

        return dict(fallback)

    @staticmethod
    def _save_json(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)


# Singleton
style_dna_engine = StyleDNAEngine()

# ================= AHVI STYLE DNA PATCH V2 BEGIN =================


def _ahvi_dna_coerce_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            import json as _json

            parsed = _json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _ahvi_dna_normalize_gender(value):
    raw = str(value or "").strip().lower()
    if raw in {"m", "male", "man", "men", "mens", "boy"}:
        return "male"
    if raw in {"f", "female", "woman", "women", "womens", "girl", "ladies"}:
        return "female"
    if raw in {"unisex", "neutral", "genderless", "any"}:
        return "unisex"
    return ""


def _ahvi_gender_from_profile(profile, previous_dna=None):
    profile = profile or {}
    previous_dna = previous_dna or {}

    prefs = _ahvi_dna_coerce_dict(profile.get("preferences"))
    style_prefs = _ahvi_dna_coerce_dict(
        profile.get("style_preferences") or profile.get("stylePreference")
    )

    candidates = [
        profile.get("style_gender"),
        profile.get("gender"),
        profile.get("preferred_gender"),
        profile.get("target_gender"),
        prefs.get("style_gender"),
        prefs.get("gender"),
        prefs.get("preferred_gender"),
        prefs.get("target_gender"),
        style_prefs.get("style_gender"),
        style_prefs.get("gender"),
        style_prefs.get("preferred_gender"),
        style_prefs.get("target_gender"),
        previous_dna.get("style_gender"),
        previous_dna.get("gender"),
    ]

    for value in candidates:
        gender = _ahvi_dna_normalize_gender(value)
        if gender:
            return gender

    return "unisex"


try:
    _AHVI_ORIGINAL_STYLE_DNA_ENRICH_CONTEXT = StyleDNAEngine.enrich_context
except Exception:
    _AHVI_ORIGINAL_STYLE_DNA_ENRICH_CONTEXT = None


if _AHVI_ORIGINAL_STYLE_DNA_ENRICH_CONTEXT and not getattr(
    StyleDNAEngine.enrich_context, "_ahvi_gender_patch_v2", False
):

    def _ahvi_style_dna_enrich_context(self, context):
        ctx = _AHVI_ORIGINAL_STYLE_DNA_ENRICH_CONTEXT(self, context) or context or {}

        if not isinstance(ctx, dict):
            ctx = {}

        profile = (
            ctx.get("user_profile") if isinstance(ctx.get("user_profile"), dict) else {}
        )
        style_dna = (
            ctx.get("style_dna") if isinstance(ctx.get("style_dna"), dict) else {}
        )

        gender = _ahvi_gender_from_profile(profile, style_dna)

        style_dna = dict(style_dna)
        style_dna.setdefault("gender", gender)
        style_dna.setdefault("style_gender", gender)

        ctx["style_dna"] = style_dna
        return ctx

    _ahvi_style_dna_enrich_context._ahvi_gender_patch_v2 = True
    StyleDNAEngine.enrich_context = _ahvi_style_dna_enrich_context

# ================= AHVI STYLE DNA PATCH V2 END =================
