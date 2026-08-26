"""
tests/test_trend_context.py
Comprehensive test suite for AHVI Trend Intelligence V1.
Verifies strict provenance gates, freshness controls, stale source rejection, missing review_state rejection,
gender normalization, region fallbacks, soft additive influence, personalized explanations, and health diagnostics.
"""
import unittest
from unittest.mock import patch
from datetime import date
from services.trend_context_service import TrendContextService, annotate_board_with_trend, _canonicalize_occasion, _normalize_gender


class TestTrendContext(unittest.TestCase):

    def setUp(self):
        self.sample_valid_trend = {
            "trend_id": "relaxed_tailoring_2026",
            "label": "Relaxed Tailoring",
            "scope": "global",
            "region": ["india", "global"],
            "gender": ["male", "female", "unisex"],
            "categories": ["top", "bottom", "outerwear", "shoes"],
            "colors": ["navy", "brown", "olive"],
            "keywords": ["wide leg", "trouser", "pleated", "overshirt"],
            "occasions": ["smart_casual", "office"],
            "confidence": 0.85,
            "review_state": "approved",
            "valid_from": "2026-01-01",
            "valid_until": "2026-12-31",
            "ingested_at": "2026-01-10T08:00:00Z",
            "verified_at": "2026-02-01T10:00:00Z",
            "source": {
                "publisher": "Vogue Fashion Index",
                "url": "https://www.vogue.com/article/spring-2026-fashion-trends",
                "published_at": "2026-01-15T00:00:00Z",
            },
        }

    # ─── PROVENANCE, FRESHNESS & REVIEW STATE GATES ───

    def test_missing_review_state_rejected(self):
        """Trend missing review_state or with unapproved state must be rejected."""
        trend_missing = dict(self.sample_valid_trend)
        trend_missing.pop("review_state", None)
        self.assertFalse(TrendContextService.is_trend_valid(trend_missing, target_date=date(2026, 8, 26)))

        trend_pending = dict(self.sample_valid_trend)
        trend_pending["review_state"] = "pending"
        self.assertFalse(TrendContextService.is_trend_valid(trend_pending, target_date=date(2026, 8, 26)))

    def test_future_published_at_rejected(self):
        """Source with published_at in the future relative to target_date must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["source"] = {
            "publisher": "Vogue",
            "url": "https://www.vogue.com/article/future-trends",
            "published_at": "2026-09-01T00:00:00Z",  # Future relative to 2026-08-26
        }
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_stale_source_rejected(self):
        """Source with published_at > 365 days older than valid_from/target_date must be rejected as stale."""
        trend = dict(self.sample_valid_trend)
        trend["source"] = {
            "publisher": "Vogue",
            "url": "https://www.vogue.com/article/old-trends",
            "published_at": "2024-12-01T00:00:00Z",  # > 365 days older
        }
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_expired_trend_rejected(self):
        """Expired trend (valid_until in the past) must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["valid_until"] = "2025-12-31"
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_future_trend_rejected(self):
        """Future trend (valid_from in the future) must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["valid_from"] = "2027-01-01"
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_valid_current_trend_accepted(self):
        """Valid current trend with complete source provenance must be accepted."""
        self.assertTrue(TrendContextService.is_trend_valid(self.sample_valid_trend, target_date=date(2026, 8, 26)))

    def test_missing_source_rejected(self):
        """Trend missing publisher or url in source must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["source"] = {"publisher": "", "url": "", "published_at": "2026-01-15T00:00:00Z"}
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_missing_publication_date_rejected(self):
        """Trend missing published_at timestamp in source must be rejected."""
        trend = dict(self.sample_valid_trend)
        trend["source"] = {"publisher": "Vogue", "url": "https://www.vogue.com", "published_at": ""}
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_low_confidence_trend_rejected(self):
        """Trend with confidence < 0.60 must be rejected/suppressed."""
        trend = dict(self.sample_valid_trend)
        trend["confidence"] = 0.45
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    # ─── GENDER NORMALIZATION & REGION GATES ───

    def test_male_vs_men_normalization(self):
        """'male', 'man', 'men', 'm' all resolve to 'male' and match male trends."""
        self.assertEqual(_normalize_gender("men"), "male")
        self.assertEqual(_normalize_gender("Man"), "male")
        self.assertEqual(_normalize_gender("MALE"), "male")

        trend = dict(self.sample_valid_trend)
        trend["gender"] = ["male"]
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            active = TrendContextService.get_active_trends(
                target_date=date(2026, 8, 26), gender="men", target_region="global"
            )
            self.assertEqual(len(active), 1)

    def test_female_vs_women_normalization(self):
        """'female', 'woman', 'women', 'w', 'f' all resolve to 'female' and match female trends."""
        self.assertEqual(_normalize_gender("women"), "female")
        self.assertEqual(_normalize_gender("Woman"), "female")
        self.assertEqual(_normalize_gender("FEMALE"), "female")

        trend = dict(self.sample_valid_trend)
        trend["gender"] = ["female"]
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            active = TrendContextService.get_active_trends(
                target_date=date(2026, 8, 26), gender="women", target_region="global"
            )
            self.assertEqual(len(active), 1)

    def test_global_fallback_disabled_and_enabled(self):
        """Global trend requested for India is rejected when allow_global_fallback=False, accepted when True."""
        trend = dict(self.sample_valid_trend)
        trend["scope"] = "global"
        trend["region"] = ["global"]

        # Disabled fallback -> Rejected for India
        self.assertFalse(
            TrendContextService.is_trend_valid(
                trend, target_date=date(2026, 8, 26), target_region="india", allow_global_fallback=False
            )
        )

        # Enabled fallback -> Accepted for India
        self.assertTrue(
            TrendContextService.is_trend_valid(
                trend, target_date=date(2026, 8, 26), target_region="india", allow_global_fallback=True
            )
        )

    # ─── REAL RUNTIME BOARD ANNOTATION & IMMUTABILITY ───

    def test_real_runtime_board_annotation(self):
        """Integration test: Real board receives trend_label, trend_meta, trend_explanation without mutating items."""
        original_items = [
            {"id": "1", "name": "Wide leg trouser", "category": "bottom", "color": "navy", "tags": ["pleated"]},
            {"id": "2", "name": "Pleated overshirt", "category": "outerwear", "color": "brown"},
        ]
        board = {
            "outfit_id": "board_999",
            "primary_outfit": {"top": "2", "bottom": "1"},
            "items": list(original_items),
        }

        annotated = annotate_board_with_trend(
            board,
            user_gender="men",
            occasion="smart_casual",
            target_region="india",
            allow_global_fallback=True,
            target_date=date(2026, 8, 26),
        )

        # 1. Trend fields populated
        self.assertIn("trend_label", annotated)
        self.assertIn("trend_explanation", annotated)
        self.assertIn("trend_meta", annotated)
        meta = annotated["trend_meta"]
        self.assertIn("publisher", meta)
        self.assertIn("published_at", meta)
        self.assertIn("confidence", meta)

        # 2. Safety & Outfit Immutability: items and primary_outfit MUST NOT change!
        self.assertEqual(annotated["items"], original_items)
        self.assertEqual(annotated["primary_outfit"], {"top": "2", "bottom": "1"})

    def test_no_valid_trend_board_unchanged(self):
        """When no valid trend matches, board keys are omitted and core fields remain intact."""
        board = {
            "outfit_id": "123",
            "items": [{"id": "1", "name": "Generic item", "category": "other"}],
        }
        annotated = annotate_board_with_trend(board)
        self.assertNotIn("trend_label", annotated)
        self.assertNotIn("trend_explanation", annotated)
        self.assertNotIn("trend_meta", annotated)

    def test_diagnostics_reporting(self):
        """get_diagnostics reports loaded status, record counts, latest ingestion, and source health."""
        diag = TrendContextService.get_diagnostics(target_date=date(2026, 8, 26), target_region="india")
        self.assertTrue(diag["trend_registry_loaded"])
        self.assertGreater(diag["total_record_count"], 0)
        self.assertIn(diag["source_health"], ["healthy", "degraded"])


if __name__ == "__main__":
    unittest.main()
