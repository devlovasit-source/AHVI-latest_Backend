"""
tests/test_trend_context.py
"""
import unittest
from unittest.mock import patch
from services.trend_context_service import TrendContextService
from routers.stylist import annotate_board_with_trend


class TestTrendContext(unittest.TestCase):

    def test_1_strict_occasion_filtering(self):
        trends = TrendContextService.get_active_trends(occasion="beach_party")
        # Should be empty or exclude strictly formal/office trends
        for t in trends:
            self.assertIn("beach_party", [occ.lower() for occ in t.get("occasions", [])])

    def test_2_minimum_evidence_rule_rejects_gym_clothes(self):
        # Gym clothes with matching colors but NO matching keywords
        gym_wardrobe = [
            {"id": "1", "name": "Athletic shorts", "category": "bottom", "color": "black"},
            {"id": "2", "name": "Gym tee", "category": "top", "color": "grey"}
        ]
        # Should return None because it lacks strong keyword/tag signals
        match = TrendContextService.match_board_trend(gym_wardrobe, occasion="smart_casual")
        self.assertIsNone(match)

    def test_3_matching_wardrobe_gets_trend(self):
        wardrobe = [
            {"id": "1", "name": "Wide leg trouser", "category": "bottom", "color": "navy", "tags": ["pleated"]},
            {"id": "2", "name": "Pleated overshirt", "category": "outerwear", "color": "brown"},
            {"id": "3", "name": "Minimal sneaker", "category": "shoes", "color": "white"}
        ]
        match = TrendContextService.match_board_trend(wardrobe, occasion="smart_casual")
        self.assertIsNotNone(match)
        self.assertIn(match["label"], ["Relaxed Tailoring", "Elevated Basics"])
        self.assertGreaterEqual(match["match_score"], 0.55)

    def test_4_trend_service_failure_does_not_break_flow(self):
        with patch.object(TrendContextService, "match_board_trend", side_effect=Exception("Simulated service crash")):
            board = {
                "outfit_id": "outfit_abc",
                "items": [{"id": "1", "name": "Basic Shirt", "category": "top"}]
            }
            
            # Run annotation with a broken trend service
            annotated = annotate_board_with_trend(board, occasion="casual")
            
            # Assert keys were completely omitted, and core fields remain intact
            self.assertNotIn("trend_label", annotated)
            self.assertNotIn("trend_meta", annotated)
            self.assertEqual(annotated["outfit_id"], "outfit_abc")

    def test_5_trend_context_must_not_change_selected_outfit(self):
        original_items = [{"id": "top_1"}, {"id": "bottom_2"}]
        board = {
            "outfit_id": "outfit_123",
            "primary_outfit": {"top": "top_1", "bottom": "bottom_2"},
            "items": list(original_items)
        }
        
        annotated = annotate_board_with_trend(board)
        
        # Regression check: ensure the items list and primary outfit were not mutated
        self.assertEqual(annotated["items"], original_items)
        self.assertEqual(annotated["primary_outfit"], {"top": "top_1", "bottom": "bottom_2"})

    def test_6_no_match_omits_keys(self):
        board = {
            "outfit_id": "123",
            "items": [{"id": "1", "name": "Generic item", "category": "other"}]
        }
        annotated = annotate_board_with_trend(board)
        self.assertNotIn("trend_label", annotated)
        self.assertNotIn("trend_meta", annotated)

    def test_7_partial_match_stays_below_threshold(self):
        """
        Tests that 1 strong matching item + 3 unrelated items 
        heavily dilutes the score and fails to reach the 0.55 threshold.
        """
        mixed_wardrobe = [
            {"id": "1", "name": "Wide leg trouser", "category": "bottom", "color": "navy", "tags": ["pleated"]}, # Strong match
            {"id": "2", "name": "Graphic tee", "category": "top", "color": "neon pink"}, # Unrelated
            {"id": "3", "name": "Windbreaker", "category": "outerwear", "color": "silver"}, # Unrelated
            {"id": "4", "name": "Flip flops", "category": "shoes", "color": "orange"} # Unrelated
        ]
        
        match = TrendContextService.match_board_trend(
            board_items=mixed_wardrobe, 
            occasion="smart_casual"
        )
        self.assertIsNone(match)

    def test_8_style_wardrobe_item_annotation(self):

        """Verify that style_wardrobe_item populates trend_label when matching items exist."""
        from routers.stylist import style_wardrobe_item, ItemStyleRequest
        
        anchor = {"id": "1", "name": "Wide leg trouser", "category": "bottom", "color": "navy", "sub_category": "wide leg trouser", "tags": ["pleated"]}
        wardrobe = [
            anchor,
            {"id": "2", "name": "Pleated overshirt", "category": "top", "sub_category": "overshirt", "color": "brown"},
            {"id": "3", "name": "Minimal sneaker", "category": "shoes", "sub_category": "minimal sneaker", "color": "white"}
        ]
        
        req = ItemStyleRequest(
            user_id="user_test",
            mode="build_outfit",
            occasion="smart_casual",
            anchor_item=anchor,
            wardrobe=wardrobe
        )
        res = style_wardrobe_item("1", req)
        self.assertTrue(res["success"])
        outfit = res.get("outfit", {})
        self.assertIn("trend_label", outfit)
        self.assertIn(outfit["trend_label"], ["Relaxed Tailoring", "Elevated Basics"])


if __name__ == "__main__":
    unittest.main()

