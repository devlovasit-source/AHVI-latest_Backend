"""
tests/test_trend_context.py
Comprehensive test suite for AHVI Trend Intelligence V1.
Verifies strict provenance gates, freshness controls, region filtering, soft additive influence,
personalized explanations, and health diagnostics.
"""
import unittest
from unittest.mock import patch
from datetime import date
from services.trend_context_service import TrendContextService, annotate_board_with_trend, _canonicalize_occasion


class TestTrendContext(unittest.TestCase):

    def setUp(self):
        self.sample_valid_trend = {
            "trend_id": "relaxed_tailoring_2026",
            "label": "Relaxed Tailoring",
            "scope": "global",
            "region": ["india", "global"],
            "gender": ["men", "women", "unisex"],
            "categories": ["top", "bottom", "outerwear", "shoes"],
            "colors": ["navy", "brown", "olive"],
            "keywords": ["wide leg", "trouser", "pleated", "overshirt"],
            "occasions": ["smart_casual", "office"],
            "confidence": 0.85,
            "review_state": "approved",
            "valid_from": "2026-01-01",
            "valid_until": "2026-12-31",
            "source": {
                "publisher": "Vogue Fashion Index",
                "url": "https://vogue.com/trends/relaxed-tailoring",
                "published_at": "2026-01-05T00:00:00Z",
            },
        }

    # ─── PROVENANCE & FRESHNESS GATES ───

    def test_expired_trend_rejected(self):
        """Expired trend (valid_until in the past) must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["valid_until"] = "2025-12-31"
        target_date = date(2026, 8, 26)
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=target_date))

    def test_future_trend_rejected(self):
        """Future trend (valid_from in the future) must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["valid_from"] = "2027-01-01"
        target_date = date(2026, 8, 26)
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=target_date))

    def test_valid_current_trend_accepted(self):
        """Valid current trend with complete source provenance must be accepted."""
        target_date = date(2026, 8, 26)
        self.assertTrue(TrendContextService.is_trend_valid(self.sample_valid_trend, target_date=target_date))

    def test_missing_source_rejected(self):
        """Trend missing publisher or url in source must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["source"] = {"publisher": "", "url": "", "published_at": "2026-01-05T00:00:00Z"}
        self.assertFalse(TrendContextService.is_trend_valid(trend))

    def test_missing_publication_date_rejected(self):
        """Trend missing published_at timestamp in source must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["source"] = {"publisher": "Vogue", "url": "https://vogue.com", "published_at": ""}
        self.assertFalse(TrendContextService.is_trend_valid(trend))

    def test_low_confidence_trend_rejected(self):
        """Trend with confidence < 0.60 must be rejected/suppressed."""
        trend = dict(self.sample_valid_trend)
        trend["confidence"] = 0.45
        self.assertFalse(TrendContextService.is_trend_valid(trend))

    # ─── REGION & GENDER GATES ───

    def test_region_mismatch_rejected(self):
        """Pure global trend requested for India without allow_global_fallback must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["scope"] = "global"
        trend["region"] = ["global"]  # Does NOT list India
        self.assertFalse(
            TrendContextService.is_trend_valid(
                trend, target_region="india", allow_global_fallback=False
            )
        )

    def test_global_fallback_works_when_allowed(self):
        """Global trend is accepted for India request ONLY when allow_global_fallback=True."""
        trend = dict(self.sample_valid_trend)
        trend["scope"] = "global"
        trend["region"] = ["global"]
        self.assertTrue(
            TrendContextService.is_trend_valid(
                trend, target_region="india", allow_global_fallback=True
            )
        )

    def test_occasion_mismatch_rejected(self):
        """Trend filtering excludes trends that do not match requested occasion."""
        trends = TrendContextService.get_active_trends(
            target_date=date(2026, 8, 26), occasion="office_fit", target_region="india"
        )
        self.assertTrue(len(trends) > 0)
        for t in trends:
            self.assertIn("office", [occ.lower() for occ in t.get("occasions", [])])

    def test_gender_mismatch_rejected(self):
        """Trend filtering excludes trends incompatible with requested gender."""
        trend = dict(self.sample_valid_trend)
        trend["gender"] = ["women"]
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            active = TrendContextService.get_active_trends(
                target_date=date(2026, 8, 26), gender="men", target_region="india"
            )
            self.assertEqual(len(active), 0)

    # ─── WARDROBE MATCHING & REASONING ───

    def test_color_only_coincidence_does_not_falsely_match(self):
        """Gym clothes with matching colors but NO matching keywords must be rejected."""
        gym_wardrobe = [
            {"id": "1", "name": "Athletic shorts", "category": "bottom", "color": "navy"},
            {"id": "2", "name": "Gym tee", "category": "top", "color": "brown"},
        ]
        match = TrendContextService.match_board_trend(gym_wardrobe, occasion="smart_casual")
        self.assertIsNone(match)

    def test_strong_wardrobe_signal_matches(self):
        """Wardrobe with strong keyword matches returns match score and personalized explanation."""
        wardrobe = [
            {"id": "1", "name": "Wide leg trouser", "category": "bottom", "color": "navy", "tags": ["pleated"]},
            {"id": "2", "name": "Pleated overshirt", "category": "outerwear", "color": "brown"},
            {"id": "3", "name": "Minimal sneaker", "category": "shoes", "color": "white"},
        ]
        match = TrendContextService.match_board_trend(wardrobe, occasion="smart_casual", target_date=date(2026, 8, 26))
        self.assertIsNotNone(match)
        self.assertIn(match["label"], ["Relaxed Tailoring", "Elevated Basics"])
        self.assertGreaterEqual(match["match_score"], 0.55)
        self.assertIn("wide leg trouser", match["explanation"].lower())

    # ─── SOFT ADDITIVE INFLUENCE & IMMUTABILITY ───

    def test_trend_cannot_bypass_outfit_safety(self):
        """annotate_board_with_trend must never mutate items list or primary outfit selection."""
        original_items = [{"id": "top_1"}, {"id": "bottom_2"}]
        board = {
            "outfit_id": "outfit_123",
            "primary_outfit": {"top": "top_1", "bottom": "bottom_2"},
            "items": list(original_items),
        }
        annotated = annotate_board_with_trend(board)
        self.assertEqual(annotated["items"], original_items)
        self.assertEqual(annotated["primary_outfit"], {"top": "top_1", "bottom": "bottom_2"})

    def test_no_valid_trend_board_unchanged(self):
        """When no valid trend matches, trend keys are omitted and board is returned unchanged."""
        board = {
            "outfit_id": "123",
            "items": [{"id": "1", "name": "Generic item", "category": "other"}],
        }
        annotated = annotate_board_with_trend(board)
        self.assertNotIn("trend_label", annotated)
        self.assertNotIn("trend_explanation", annotated)
        self.assertNotIn("trend_meta", annotated)

    def test_trend_metadata_preserves_provenance(self):
        """trend_meta includes complete provenance fields: publisher, published_at, confidence, region, valid_until."""
        wardrobe = [
            {"id": "1", "name": "Wide leg trouser", "category": "bottom", "color": "navy", "tags": ["pleated"]},
            {"id": "2", "name": "Pleated overshirt", "category": "outerwear", "color": "brown"},
        ]
        board = {"outfit_id": "123", "items": wardrobe}
        annotated = annotate_board_with_trend(board, occasion="smart_casual", target_date=date(2026, 8, 26))
        self.assertIn("trend_label", annotated)
        self.assertIn("trend_meta", annotated)
        meta = annotated["trend_meta"]
        self.assertIn("publisher", meta)
        self.assertIn("published_at", meta)
        self.assertIn("confidence", meta)
        self.assertIn("region", meta)
        self.assertIn("valid_until", meta)
        self.assertIn("source_count", meta)

    def test_diagnostics_reporting(self):
        """get_diagnostics reports loaded status, record counts, latest ingestion, and source health."""
        diag = TrendContextService.get_diagnostics(target_date=date(2026, 8, 26), target_region="india")
        self.assertTrue(diag["trend_registry_loaded"])
        self.assertGreater(diag["total_record_count"], 0)
        self.assertGreaterEqual(diag["active_record_count"], 0)
        self.assertIn(diag["source_health"], ["healthy", "degraded"])


if __name__ == "__main__":
    unittest.main()
