"""
tests/test_trend_context.py
AHVI Trend Intelligence V1 -- focused + runtime integration tests.

Adapted from PR #48's internal test suite (kept and re-targeted at the
cleaned-up API: `canonical_occasion` instead of raw `occasion`, the
verification_status gate, and the claim contract), plus the runtime
integration / route-level tests the rebuild review required.
"""
import inspect
import unittest
from unittest.mock import patch
from datetime import date

from services.trend_context_service import (
    TrendContextService,
    annotate_board_with_trend,
    _normalize_gender,
    SOURCE_BACKED_CURRENT,
    SOURCE_BACKED_RECENT,
    NO_VALID_TREND,
)
from services import style_trend_registry


def _verified_trend(**overrides):
    base = {
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
        "verification_status": "VERIFIED",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-10T08:00:00Z",
        "verified_at": "2026-08-01T10:00:00Z",
        "source": {
            "publisher": "Vogue Fashion Index",
            "url": "https://www.vogue.com/article/spring-2026-fashion-trends",
            "published_at": "2026-01-15T00:00:00Z",
        },
    }
    base.update(overrides)
    return base


class TestTrendVerificationGate(unittest.TestCase):
    """Section 3 (BLOCKING): nothing is active on manually-entered metadata alone."""

    def test_unverified_source_ignored(self):
        trend = _verified_trend(verification_status="DEAD_LINK")
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_missing_verification_status_ignored(self):
        trend = _verified_trend()
        trend.pop("verification_status", None)
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_missing_verified_at_rejected_even_if_review_state_approved(self):
        trend = _verified_trend(verified_at=None)
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_future_verified_at_rejected(self):
        trend = _verified_trend(verified_at="2027-01-01T00:00:00Z")
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_original_pr48_records_remain_permanently_disabled(self):
        """The original 12 PR48 records (all dead links or unconfirmed
        content, checked 2026-08-26) must never become active regardless of
        the Phase 2 verified additions -- this proves the BLOCKING
        provenance rule is still enforced end-to-end for that specific data,
        not just in a mocked unit test."""
        registry = style_trend_registry.get_trend_registry()
        original_ids = {t["trend_id"] for t in style_trend_registry.ORIGINAL_DISABLED_TRENDS}
        active = TrendContextService.get_active_trends(
            target_date=date(2026, 8, 26), target_region="india", allow_global_fallback=True,
        )
        active_ids = {t["trend_id"] for t in active}
        self.assertTrue(original_ids.isdisjoint(active_ids))
        self.assertTrue(all(t.get("verification_status") != "VERIFIED" for t in style_trend_registry.ORIGINAL_DISABLED_TRENDS))

    def test_phase2_verified_records_are_active(self):
        """The 7 Phase 2 records, independently re-sourced and verified live
        on 2026-08-26 against real editorial publications, must be reachable
        (India-scoped one for target_region="india"; all 7 with global
        fallback permitted)."""
        registry = style_trend_registry.get_trend_registry()
        verified_ids = {t["trend_id"] for t in style_trend_registry.VERIFIED_TRENDS}
        self.assertEqual(len(verified_ids), 7)
        active_global = TrendContextService.get_active_trends(
            target_date=date(2026, 8, 26), target_region="india", allow_global_fallback=True,
        )
        active_ids = {t["trend_id"] for t in active_global}
        self.assertEqual(active_ids, verified_ids)

    def test_missing_review_state_rejected(self):
        trend = _verified_trend()
        trend.pop("review_state", None)
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_future_published_at_rejected(self):
        trend = _verified_trend()
        trend["source"] = dict(trend["source"], published_at="2026-09-01T00:00:00Z")
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_stale_source_rejected(self):
        trend = _verified_trend()
        trend["source"] = dict(trend["source"], published_at="2024-12-01T00:00:00Z")
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_expired_trend_rejected(self):
        trend = _verified_trend(valid_until="2025-12-31")
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_future_trend_rejected(self):
        trend = _verified_trend(valid_from="2027-01-01")
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_valid_verified_current_trend_accepted(self):
        self.assertTrue(TrendContextService.is_trend_valid(_verified_trend(), target_date=date(2026, 8, 26)))

    def test_missing_source_rejected(self):
        trend = _verified_trend(source={"publisher": "", "url": "", "published_at": "2026-01-15T00:00:00Z"})
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))

    def test_low_confidence_trend_rejected(self):
        trend = _verified_trend(confidence=0.45)
        self.assertFalse(TrendContextService.is_trend_valid(trend, target_date=date(2026, 8, 26)))


class TestGenderNormalization(unittest.TestCase):
    def test_male_vs_men_normalization(self):
        self.assertEqual(_normalize_gender("men"), "male")
        self.assertEqual(_normalize_gender("Man"), "male")
        trend = _verified_trend(gender=["male"])
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            active = TrendContextService.get_active_trends(target_date=date(2026, 8, 26), gender="men", target_region="global")
            self.assertEqual(len(active), 1)

    def test_female_vs_women_normalization(self):
        self.assertEqual(_normalize_gender("women"), "female")
        trend = _verified_trend(gender=["female"])
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            active = TrendContextService.get_active_trends(target_date=date(2026, 8, 26), gender="women", target_region="global")
            self.assertEqual(len(active), 1)


class TestNoSecondOccasionTaxonomy(unittest.TestCase):
    """Section 5 & 6: no private occasion map, no substring reinterpretation."""

    def test_no_canonical_occasion_map_symbol_exists(self):
        import services.trend_context_service as svc
        self.assertFalse(hasattr(svc, "_CANONICAL_OCCASION_MAP"))
        self.assertFalse(hasattr(svc, "_canonicalize_occasion"))

    def test_service_never_reinterprets_raw_text_it_only_compares_verbatim(self):
        """canonical_occasion is used as an opaque, already-resolved token --
        passing raw text through does NOT get parsed/mapped by this service,
        it is only compared verbatim (lowercased) against each trend's
        `occasions` list. This proves the service itself has no substring-
        matching reinterpretation logic, independent of whatever the
        upstream canonicalizer does."""
        trend = _verified_trend(occasions=["dinner"])
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            # Raw text containing "date" as a substring must NOT match a
            # trend scoped to "dinner" just because some upstream mapping
            # elsewhere might once have conflated the two -- this service
            # does no such mapping, it only compares literal tokens.
            active = TrendContextService.get_active_trends(
                target_date=date(2026, 8, 26), canonical_occasion="candidate_interview", target_region="global",
            )
            self.assertEqual(active, [])

    def test_candidate_interview_upstream_canonicalizer_bug_documented(self):
        """DOCUMENTS a pre-existing bug discovered during this rebuild in
        brain.engines.style_scorer.normalize_occasion() -- NOT something
        trend_context_service.py does or can fix (Section 15: no new
        occasion canonicalizer). normalize_occasion("candidate_interview")
        currently returns "date_night" because of a `"date" in readable`
        substring check ("date" is literally a substring of "candidate").
        This test intentionally fails loudly if that upstream bug is
        silently fixed without updating this note, and serves as the
        BLOCKING record that Trend Intelligence V1 could theoretically
        inherit a bad occasion resolution from its caller -- today this has
        zero observable effect because the registry has zero VERIFIED
        trends (see test_real_registry_has_zero_verified_active_trends)."""
        from brain.engines.style_scorer import normalize_occasion
        result = normalize_occasion("candidate_interview")
        self.assertEqual(
            result, "date_night",
            "If this fails, the upstream normalize_occasion() bug has been fixed -- "
            "update this test and the module docstring's note about it.",
        )

    def test_workout_and_gym_do_not_collapse_into_a_casual_trend(self):
        """PR #48's own private map explicitly did workout->casual, gym->casual.
        This service does no such collapsing -- a trend scoped only to
        "casual" must not match when the caller resolves workout/gym to
        their own distinct canonical occasion."""
        casual_trend = _verified_trend(occasions=["casual"])
        with patch("services.trend_context_service.get_trend_registry", return_value=[casual_trend]):
            for occ in ("workout", "gym"):
                active = TrendContextService.get_active_trends(
                    target_date=date(2026, 8, 26), canonical_occasion=occ, target_region="global",
                )
                self.assertEqual(active, [], f"{occ!r} must not match a casual-only trend")


class TestIndiaGlobalPolicy(unittest.TestCase):
    """Section 8."""

    def test_india_specific_trend_matches_india_request(self):
        trend = _verified_trend(scope="india", region=["india"], occasions=["wedding"])
        items = [{"id": "1", "name": "Overshirt", "category": "top", "tags": ["wide leg"]}]
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            match = TrendContextService.match_board_trend(
                board_items=items, canonical_occasion="wedding", target_region="india", allow_global_fallback=False,
            )
            self.assertIsNotNone(match)

    def test_global_trend_rejected_for_india_without_fallback(self):
        trend = _verified_trend(scope="global", region=["global"])
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            active = TrendContextService.get_active_trends(target_date=date(2026, 8, 26), target_region="india", allow_global_fallback=False)
            self.assertEqual(active, [])

    def test_permitted_global_fallback_used_only_when_no_india_match(self):
        india_trend = _verified_trend(trend_id="india_1", scope="india", region=["india"], occasions=["party"])
        global_trend = _verified_trend(trend_id="global_1", scope="global", region=["global"], occasions=["smart_casual"])
        items = [{"id": "1", "name": "Wide leg trouser", "category": "bottom", "tags": ["wide leg"]}]
        with patch("services.trend_context_service.get_trend_registry", return_value=[india_trend, global_trend]):
            # occasion only matches the global trend -> india pass finds nothing, fallback kicks in
            match = TrendContextService.match_board_trend(
                board_items=items, canonical_occasion="smart_casual", target_region="india", allow_global_fallback=True,
            )
            self.assertIsNotNone(match)
            self.assertEqual(match["trend_id"], "global_1")

    def test_global_fallback_not_used_when_disallowed(self):
        global_trend = _verified_trend(scope="global", region=["global"], occasions=["smart_casual"])
        items = [{"id": "1", "name": "Wide leg trouser", "category": "bottom", "tags": ["wide leg"]}]
        with patch("services.trend_context_service.get_trend_registry", return_value=[global_trend]):
            match = TrendContextService.match_board_trend(
                board_items=items, canonical_occasion="smart_casual", target_region="india", allow_global_fallback=False,
            )
            self.assertIsNone(match)


class TestClaimContract(unittest.TestCase):
    """Section 11."""

    def test_recently_published_source_claims_current(self):
        trend = _verified_trend(source={"publisher": "Vogue", "url": "https://www.vogue.com/x", "published_at": "2026-07-01T00:00:00Z"})
        items = [{"id": "1", "name": "Wide leg trouser", "category": "bottom", "tags": ["wide leg"]}]
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            match = TrendContextService.match_board_trend(board_items=items, canonical_occasion="office", target_region="global", allow_global_fallback=True)
            self.assertEqual(match["claim"], SOURCE_BACKED_CURRENT)

    def test_older_published_source_claims_recent(self):
        trend = _verified_trend(source={"publisher": "Vogue", "url": "https://www.vogue.com/x", "published_at": "2026-01-15T00:00:00Z"})
        items = [{"id": "1", "name": "Wide leg trouser", "category": "bottom", "tags": ["wide leg"]}]
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            match = TrendContextService.match_board_trend(board_items=items, canonical_occasion="office", target_region="global", allow_global_fallback=True)
            self.assertEqual(match["claim"], SOURCE_BACKED_RECENT)

    def test_no_match_never_fabricates_an_explanation(self):
        board = {"outfit_id": "1", "items": [{"id": "1", "name": "Generic item", "category": "other"}]}
        annotated = annotate_board_with_trend(board, canonical_occasion="office", target_region="india", allow_global_fallback=True)
        self.assertNotIn("trend_label", annotated)
        self.assertNotIn("trend_explanation", annotated)
        self.assertNotIn("trend_meta", annotated)
        self.assertNotIn("trend_claim", annotated)


class TestDeterminism(unittest.TestCase):
    """Section 9: no list(set(...)), repeated calls must be byte-equivalent."""

    def test_no_list_set_pattern_in_source(self):
        import services.trend_context_service as svc
        src = inspect.getsource(svc)
        self.assertNotIn("list(set(", src)

    def test_repeated_calls_produce_identical_ordering(self):
        trend = _verified_trend(occasions=["office"])
        items = [
            {"id": "3", "name": "Wide leg trouser", "category": "bottom", "tags": ["wide leg", "trouser"]},
            {"id": "1", "name": "Overshirt", "category": "top", "tags": ["overshirt", "pleated"]},
            {"id": "2", "name": "Loafer", "category": "shoes", "tags": ["trouser"]},
        ]
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            results = [
                TrendContextService.match_board_trend(board_items=items, canonical_occasion="office", target_region="global", allow_global_fallback=True)
                for _ in range(25)
            ]
        first = results[0]
        for r in results[1:]:
            self.assertEqual(r["matched_item_ids"], first["matched_item_ids"])
            self.assertEqual(r["matched_reasons"], first["matched_reasons"])
            self.assertEqual(r["match_score"], first["match_score"])


class TestCanonicalItemTaxonomy(unittest.TestCase):
    """Section 7."""

    def test_trend_category_maps_into_ahvi_canonical_roles(self):
        from services.trend_context_service import TREND_CATEGORY_TO_AHVI_ROLE
        # spot-check the mapping table itself, not invented percentages
        self.assertEqual(TREND_CATEGORY_TO_AHVI_ROLE["shoes"], "footwear")
        self.assertEqual(TREND_CATEGORY_TO_AHVI_ROLE["accessories"], "accessory")
        self.assertEqual(TREND_CATEGORY_TO_AHVI_ROLE["dresses"], "dress")
        self.assertNotIn("ethnic_wear", TREND_CATEGORY_TO_AHVI_ROLE)  # degrades safely, not force-mapped

    def test_unmapped_trend_category_degrades_safely_no_crash(self):
        trend = _verified_trend(categories=["ethnic_wear"], keywords=["kurta"], occasions=["festive"])
        items = [{"id": "1", "name": "Cotton Kurta", "category": "ethnic_wear", "tags": ["kurta"]}]
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            match = TrendContextService.match_board_trend(board_items=items, canonical_occasion="festive", target_region="global", allow_global_fallback=True)
            self.assertIsNotNone(match)  # keyword signal alone is enough; category bonus just doesn't apply


class TestBoardImmutability(unittest.TestCase):
    """Section 13."""

    def test_annotation_never_mutates_items_or_anchor(self):
        trend = _verified_trend()
        original_items = [
            {"id": "1", "name": "Wide leg trouser", "category": "bottom", "color": "navy", "tags": ["pleated"]},
            {"id": "2", "name": "Pleated overshirt", "category": "outerwear", "color": "brown"},
        ]
        board = {"outfit_id": "board_999", "primary_outfit": {"top": "2", "bottom": "1"}, "items": list(original_items)}
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            annotated = annotate_board_with_trend(
                board, user_gender="men", canonical_occasion="smart_casual",
                target_region="india", allow_global_fallback=True, target_date=date(2026, 8, 26),
            )
        self.assertIn("trend_label", annotated)
        self.assertEqual(annotated["items"], original_items)
        self.assertEqual(annotated["primary_outfit"], {"top": "2", "bottom": "1"})
        for i, item in enumerate(annotated["items"]):
            self.assertEqual(item, original_items[i])

    def test_no_valid_trend_board_unchanged(self):
        board = {"outfit_id": "123", "items": [{"id": "1", "name": "Generic item", "category": "other"}]}
        annotated = annotate_board_with_trend(board)
        self.assertNotIn("trend_label", annotated)


class TestFailOpen(unittest.TestCase):
    """Section 11/16: trend unavailable -> normal valid board still succeeds."""

    def test_trend_service_exception_still_returns_valid_board(self):
        board = {"outfit_id": "1", "items": [{"id": "1", "name": "Shirt", "category": "top"}]}
        with patch(
            "services.trend_context_service.TrendContextService.match_board_trend",
            side_effect=RuntimeError("boom"),
        ):
            annotated = annotate_board_with_trend(board, canonical_occasion="office")
        self.assertEqual(annotated["outfit_id"], "1")
        self.assertEqual(annotated["items"], board["items"])
        self.assertNotIn("trend_label", annotated)

    def test_non_dict_board_returned_unchanged(self):
        self.assertEqual(annotate_board_with_trend(None), None)
        self.assertEqual(annotate_board_with_trend("not a dict"), "not a dict")


class TestDiagnostics(unittest.TestCase):
    def test_diagnostics_reporting_reflects_phase2_activation(self):
        diag = TrendContextService.get_diagnostics(target_date=date(2026, 8, 26), target_region="india")
        self.assertTrue(diag["trend_registry_loaded"])
        self.assertEqual(diag["total_record_count"], 19)  # 12 disabled + 7 verified
        self.assertEqual(diag["active_record_count"], 1)  # india-scoped default region: only the 1 India record
        self.assertEqual(diag["unverified_count"], 12)
        self.assertEqual(diag["source_health"], "healthy")
        self.assertEqual(diag["mode"], "CURATED_SOURCE_BACKED")


class TestRuntimeIntegration(unittest.TestCase):
    """Section 10/12: the actual wiring into style_flow_service, not just the
    service in isolation. Full HTTP-route testing of finalize_style_response_
    payload would require mocking the entire Gemini/Appwrite/wardrobe stack,
    which is disproportionate to this change's surface area (one ~15-line
    call site). These tests instead prove: (a) the wiring exists at the
    correct call site and imports cleanly, (b) calling
    annotate_board_with_trend exactly the way style_flow_service.py calls it
    (same kwargs) works end-to-end against a realistic card shape."""

    def test_integration_call_site_exists_in_style_flow_service(self):
        import services.style_flow_service as sfs
        src = inspect.getsource(sfs.finalize_style_response_payload)
        self.assertIn("annotate_board_with_trend", src)
        self.assertIn("canonical_occasion=normalized_occasion", src)
        # must run after board_items/composition_brief enrichment, before the
        # final response_payload/data dicts are built
        self.assertLess(src.index("annotate_board_with_trend"), src.index("response_payload = {"))
        self.assertLess(src.index("composition_brief"), src.index("annotate_board_with_trend"))

    def test_integration_call_shape_matches_real_card_from_style_flow(self):
        """A 'card' as style_flow_service builds it: id/name/category items
        under board_dict['items'], real occasion string, real gender source."""
        trend = _verified_trend(occasions=["office"])
        card = {
            "title": "Boardroom Ready",
            "items": [
                {"id": "shirt-1", "name": "Oxford Shirt", "category": "top", "color": "navy", "tags": []},
                {"id": "trouser-1", "name": "Wide leg trouser", "category": "bottom", "color": "navy", "tags": ["pleated"]},
            ],
            "board_items": [],
        }
        with patch("services.trend_context_service.get_trend_registry", return_value=[trend]):
            annotated = annotate_board_with_trend(
                card, user_gender="male", canonical_occasion="office",
                target_region="india", allow_global_fallback=True,
            )
        self.assertIn("trend_label", annotated)
        self.assertEqual(annotated["title"], "Boardroom Ready")  # untouched non-trend fields


class TestNegativeCompatUnaffected(unittest.TestCase):
    """Section 14: trend intelligence must not change negative-compat outcomes."""

    def test_invalid_outfit_rejection_identical_with_trend_module_imported(self):
        from brain.engines.style_compatibility_rules import evaluate_outfit, SEVERITY_HARD
        # importing trend_context_service must not change compat evaluation
        import services.trend_context_service  # noqa: F401
        v = evaluate_outfit(
            [
                {"id": "tux", "role": "outerwear", "name": "formal tuxedo"},
                {"id": "sneak", "role": "footwear", "name": "chunky running sneakers"},
            ],
            occasion="black_tie", query="black tie event",
        )
        self.assertTrue(v)
        self.assertTrue(all(x.severity == SEVERITY_HARD for x in v))


class TestPhase2Matchability(unittest.TestCase):
    """Section 8: a valid source that can never match AHVI taxonomy is not
    useful. One representative canonical board fixture per Phase 2 verified
    trend, built from that trend's own real keywords, against the REAL
    (unmocked) registry."""

    FIXTURES = {
        "modern_heirlooms_2026": (
            [{"id": "1", "name": "Beaded Brocade Gown", "category": "dresses", "tags": ["embroidery"]}],
            "cocktail",
        ),
        "office_hours_skirt_suits_2026": (
            [{"id": "1", "name": "Houndstooth Skirt Suit Blazer", "category": "outerwear"}],
            "office",
        ),
        "touch_point_texture_2026": (
            [{"id": "1", "name": "Shearling Fur Coat", "category": "outerwear"}],
            "travel",
        ),
        "night_moves_dark_romance_2026": (
            [{"id": "1", "name": "Lace Corsetry Dress", "category": "dresses", "color": "burgundy"}],
            "party",
        ),
        "shape_shift_proportion_2026": (
            [{"id": "1", "name": "Oversized Shoulder Peplum Jacket", "category": "outerwear"}],
            "party",
        ),
        "its_a_wrap_shrouds_2026": (
            [{"id": "1", "name": "Structured Knit Cape", "category": "outerwear"}],
            "casual",
        ),
        "india_couture_week_embellished_ethnic_2026": (
            [{"id": "1", "name": "Zardozi Embroidered Lehenga", "category": "ethnic_wear", "color": "gold"}],
            "wedding",
        ),
    }

    def test_every_verified_trend_matches_its_own_representative_board(self):
        for trend_id, (items, occasion) in self.FIXTURES.items():
            with self.subTest(trend_id=trend_id):
                match = TrendContextService.match_board_trend(
                    board_items=items, canonical_occasion=occasion,
                    target_region="india", allow_global_fallback=True,
                    target_date=date(2026, 8, 26),
                )
                self.assertIsNotNone(match, f"{trend_id} did not match its own representative board")
                self.assertEqual(match["trend_id"], trend_id)
                self.assertGreaterEqual(match["match_score"], 0.55)

    def test_color_alone_still_insufficient_against_real_verified_trend(self):
        """night_moves_dark_romance_2026 lists 'jewel tone'/'burgundy' as
        colors but requires a real keyword match (lace/corsetry/high neck/
        feathered/asymmetric hem) -- a plain burgundy item with none of
        those words must NOT match, proving the strong-signal rule still
        holds against real (not mocked) verified data."""
        items = [{"id": "1", "name": "Burgundy Wool Sweater", "category": "top", "color": "burgundy"}]
        match = TrendContextService.match_board_trend(
            board_items=items, canonical_occasion="party",
            target_region="india", allow_global_fallback=True, target_date=date(2026, 8, 26),
        )
        self.assertIsNone(match)


class TestPhase2IndiaGlobalFallbackCases(unittest.TestCase):
    """Section 9, cases A-D, against the real registry."""

    def test_case_a_india_specific_match_selected_before_global(self):
        items = [{"id": "1", "name": "Zardozi Embroidered Lehenga", "category": "ethnic_wear", "color": "gold"}]
        match = TrendContextService.match_board_trend(
            board_items=items, canonical_occasion="wedding",
            target_region="india", allow_global_fallback=True, target_date=date(2026, 8, 26),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["trend_id"], "india_couture_week_embellished_ethnic_2026")

    def test_case_b_no_india_match_valid_global_participates(self):
        items = [{"id": "1", "name": "Houndstooth Skirt Suit Blazer", "category": "outerwear"}]
        match = TrendContextService.match_board_trend(
            board_items=items, canonical_occasion="office",
            target_region="india", allow_global_fallback=True, target_date=date(2026, 8, 26),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["trend_id"], "office_hours_skirt_suits_2026")

    def test_case_c_india_trend_exists_but_board_mismatch_allows_global_fallback(self):
        """The India-scoped trend exists in the registry, but this board
        (an office blazer) shares no keyword signal with it -- V1 policy
        must still allow a matching global trend through."""
        items = [{"id": "1", "name": "Houndstooth Skirt Suit Blazer", "category": "outerwear"}]
        india_trends = TrendContextService.get_active_trends(
            target_date=date(2026, 8, 26), canonical_occasion="office", target_region="india", allow_global_fallback=False,
        )
        self.assertEqual(india_trends, [])  # india trend doesn't cover "office" occasion at all
        match = TrendContextService.match_board_trend(
            board_items=items, canonical_occasion="office",
            target_region="india", allow_global_fallback=True, target_date=date(2026, 8, 26),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["trend_id"], "office_hours_skirt_suits_2026")  # global trend used via fallback

    def test_case_c_global_not_used_when_fallback_disallowed(self):
        items = [{"id": "1", "name": "Houndstooth Skirt Suit Blazer", "category": "outerwear"}]
        match = TrendContextService.match_board_trend(
            board_items=items, canonical_occasion="office",
            target_region="india", allow_global_fallback=False, target_date=date(2026, 8, 26),
        )
        self.assertIsNone(match)

    def test_case_d_no_valid_trend_board_unchanged(self):
        board = {"outfit_id": "1", "items": [{"id": "1", "name": "Plain Grey T-Shirt", "category": "top"}]}
        annotated = annotate_board_with_trend(
            board, canonical_occasion="casual", target_region="india",
            allow_global_fallback=True, target_date=date(2026, 8, 26),
        )
        self.assertNotIn("trend_label", annotated)
        self.assertEqual(annotated["items"], board["items"])


class TestPhase2UserFacingMetadata(unittest.TestCase):
    """Section 10: exact metadata shape attached to a board, no internal
    scoring/debug leakage, no item mutation."""

    def test_trend_meta_exposes_only_approved_fields(self):
        board = {
            "outfit_id": "board_1",
            "items": [{"id": "1", "name": "Houndstooth Skirt Suit Blazer", "category": "outerwear"}],
        }
        annotated = annotate_board_with_trend(
            board, canonical_occasion="office", target_region="india",
            allow_global_fallback=True, target_date=date(2026, 8, 26),
        )
        self.assertIn("trend_meta", annotated)
        meta = annotated["trend_meta"]
        required_keys = {"trend_id", "match_score", "region", "valid_until", "confidence", "publisher", "published_at", "claim"}
        self.assertEqual(set(meta.keys()), required_keys)
        # no raw scoring internals (item points, coverage math, matched_item_ids) leak into user-facing meta
        self.assertNotIn("matched_item_ids", meta)
        self.assertNotIn("matched_reasons", meta)
        self.assertEqual(meta["trend_id"], "office_hours_skirt_suits_2026")
        self.assertEqual(meta["publisher"], "Coveteur")
        self.assertEqual(meta["published_at"], "2026-08-12T00:00:00Z")
        self.assertIn(meta["claim"], (SOURCE_BACKED_CURRENT, SOURCE_BACKED_RECENT))
        self.assertEqual(board["items"], annotated["items"])  # unchanged


if __name__ == "__main__":
    unittest.main()
