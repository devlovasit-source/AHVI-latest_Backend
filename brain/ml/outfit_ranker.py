"""Deterministic outfit ranker with no mutable local learning state."""
from __future__ import annotations

import logging
import math
from types import MappingProxyType
from typing import Any, Dict, List


logger = logging.getLogger("ahvi.outfit_ranker")

# Preserve the legacy baseline weights exactly. Durable Batch 8 preference
# effects are already applied by UnifiedStyleScorer before this rank pass.
BASELINE_WEIGHTS = MappingProxyType({
    "occasion_rules": 0.9,
    "color_intelligence": 0.8,
    "layering": 0.7,
    "style_graph": 0.7,
    "memory": 0.8,
    "feedback": 1.0,
    "semantic_relevance": 0.9,
})


class OutfitRanker:
    """Apply immutable baseline feature weights.

    ``learn_from_feedback`` remains as a compatibility no-op. It had no live
    route caller; production preference learning now comes from durable Batch 8
    feedback in canonical scoring.
    """

    def rank(
        self, user_id: str, outfits: List[Dict[str, Any]], top_n: int = 3
    ) -> List[Dict[str, Any]]:
        del user_id  # Ranking is deterministic for identical durable inputs.
        if not outfits:
            return []
        scored: List[Dict[str, Any]] = []
        for outfit in outfits:
            feature_map = outfit.get("ml_features", {}) if isinstance(outfit, dict) else {}
            linear = sum(
                float(feature_map.get(key, 0.0)) * float(weight)
                for key, weight in BASELINE_WEIGHTS.items()
            )
            ml_score = self._sigmoid(linear)
            item = dict(outfit)
            item["ml_score"] = round(ml_score, 4)
            item["rank_score"] = round(
                (ml_score * 100.0) + float(item.get("score", 0.0)), 3
            )
            scored.append(item)
        scored.sort(key=lambda value: float(value.get("rank_score", 0.0)), reverse=True)
        return scored[: max(1, int(top_n))]

    def learn_from_feedback(
        self, user_id: str, features: Dict[str, Any], feedback: str
    ) -> None:
        """Legacy compatibility method; intentionally performs no mutation."""
        logger.info(
            "AHVI_LEGACY_RANKER_LEARNING_IGNORED user_present=%s feature_count=%d feedback_present=%s",
            bool(str(user_id or "").strip()), len(features or {}),
            bool(str(feedback or "").strip()),
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = max(-20.0, min(20.0, float(value)))
        return 1.0 / (1.0 + math.exp(-clipped))


outfit_ranker = OutfitRanker()
