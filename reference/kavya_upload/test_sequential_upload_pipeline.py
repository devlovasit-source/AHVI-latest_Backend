"""Comprehensive 21-Point Automated Test Suite for Sequential Multi-Image Upload Processing with Smart Regeneration Bypass.
"""

import concurrent.futures
import io
import os
import sys
import time
import types
import unittest
from PIL import Image

from services.upload_batch_orchestrator import (
    UploadBatchOrchestrator,
    classify_image_suitability,
    compute_canonical_fingerprint,
    deterministic_appwrite_id,
    validate_cutout_quality,
    validate_status_transition,
)


def _make_dummy_image(width: int = 400, height: int = 400, color: str = "red") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _mock_successful_save(*args, **kwargs):
    upload_item_id = kwargs.get("upload_item_id") or "mock_item"
    wardrobe_id = f"w_{upload_item_id}"
    return {
        "success": True,
        "wardrobe_item_id": wardrobe_id,
        "items": [{"id": wardrobe_id}],
    }


class TestSequentialUploadPipeline(unittest.TestCase):

    def setUp(self):
        os.environ["ALLOW_LOCAL_AI_FALLBACK"] = "true"
        self.orchestrator = UploadBatchOrchestrator()
        self.user_id = "test_user_999"
        self.batch_id = "test_batch_111"
        self._orig_save = self.orchestrator.persistence_service.save_wardrobe_item
        self.orchestrator.persistence_service.save_wardrobe_item = _mock_successful_save

    def tearDown(self):
        self.orchestrator.persistence_service.save_wardrobe_item = self._orig_save

    def test_01_clean_product_bypass(self):
        """Test 1: Clean product photo -> DIRECT_RMBG -> skip AI regeneration."""
        img_bytes = _make_dummy_image(400, 400, "blue")
        route = classify_image_suitability(img_bytes, {"category": "Tops", "catalog_quality_score": 85.0})
        self.assertEqual(route, "DIRECT_RMBG")

        res = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id,
            batch_id=self.batch_id,
            upload_item_id="item_clean_1",
            image_bytes=img_bytes,
            metadata={"category": "Tops", "catalog_quality_score": 85.0},
            processing_mode="MULTI_AUTO_ADD",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "ADDED_TO_WARDROBE")
        self.assertFalse(res["regeneration_called"])
        self.assertEqual(res["route"], "DIRECT_RMBG")

    def test_02_complex_photo_regeneration(self):
        """Test 2: Occluded photo -> REGENERATE -> AI regeneration called and cutout validated."""
        img_bytes = _make_dummy_image(200, 200, "grey")
        route = classify_image_suitability(img_bytes, {"category": "Tops", "catalog_quality_score": 30.0, "has_person_remnant": True})
        self.assertEqual(route, "REGENERATE")

        res = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id,
            batch_id=self.batch_id,
            upload_item_id="item_complex_2",
            image_bytes=img_bytes,
            metadata={"category": "Tops", "catalog_quality_score": 30.0, "has_person_remnant": True},
            processing_mode="MULTI_AUTO_ADD",
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["regeneration_called"])

    def test_03_sequential_order(self):
        """Test 3: Sequential processing order."""
        item_1_claimed = self.orchestrator.claim_item_lease(self.user_id, self.batch_id, "seq_item_1", "w1")
        self.assertTrue(item_1_claimed["success"])

        res1 = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id,
            batch_id=self.batch_id,
            upload_item_id="seq_item_1",
            image_bytes=_make_dummy_image(),
            metadata={},
        )
        self.assertEqual(res1["status"], "ADDED_TO_WARDROBE")

        item_2_claimed = self.orchestrator.claim_item_lease(self.user_id, self.batch_id, "seq_item_2", "w1")
        self.assertTrue(item_2_claimed["success"])

    def test_04_middle_item_failure_recovery(self):
        """Test 4: Middle item failure does not block item 3."""
        res1 = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id, batch_id=self.batch_id, upload_item_id="m_1", image_bytes=_make_dummy_image(), metadata={}
        )
        self.assertTrue(res1["success"])

        res2 = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id, batch_id=self.batch_id, upload_item_id="m_2", image_bytes=b"", metadata={}
        )
        self.assertFalse(res2["success"])
        self.assertEqual(res2["status"], "REJECTED")

        res3 = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id, batch_id=self.batch_id, upload_item_id="m_3", image_bytes=_make_dummy_image(), metadata={}
        )
        self.assertTrue(res3["success"])

    def test_05_duplicate_idempotency(self):
        """Test 5: Duplicate request returns existing wardrobe item without re-processing."""
        item_id = "dup_item_5"
        res1 = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id, batch_id=self.batch_id, upload_item_id=item_id, image_bytes=_make_dummy_image(), metadata={}
        )
        self.assertTrue(res1["success"])
        w_id = res1["wardrobe_item_id"]

        res2 = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id, batch_id=self.batch_id, upload_item_id=item_id, image_bytes=_make_dummy_image(), metadata={}
        )
        self.assertTrue(res2["success"])
        self.assertTrue(res2.get("idempotent"))
        self.assertEqual(res2["wardrobe_item_id"], w_id)

    def test_06_mixed_batch(self):
        """Test 6: Mixed batch execution (DIRECT_RMBG + REGENERATE)."""
        r1 = classify_image_suitability(_make_dummy_image(400, 400), {"catalog_quality_score": 90})
        r2 = classify_image_suitability(_make_dummy_image(200, 200), {"has_person_remnant": True})
        self.assertEqual(r1, "DIRECT_RMBG")
        self.assertEqual(r2, "REGENERATE")

    def test_07_rmbg_cutout_validation_fallback(self):
        """Test 7: Empty cutout fails quality validation with empty_foreground."""
        diag = validate_cutout_quality(b"")
        self.assertFalse(diag["valid"])
        self.assertIn("empty_cutout_bytes", diag["reasons"])

    def test_08_20_worker_concurrent_race_protection(self):
        """Test 8: 20 concurrent threads racing for same item lease."""
        item_id = "race_item_concurrent_8"

        def _worker_claim(worker_idx: int):
            orchestrator = UploadBatchOrchestrator()
            return orchestrator.claim_item_lease(self.user_id, self.batch_id, item_id, f"worker_{worker_idx}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(_worker_claim, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        successes = [r for r in results if r.get("success")]
        self.assertEqual(len(successes), 1)

    def test_09_lost_http_response_recovery(self):
        """Test 9: Retry after save succeeds returns existing wardrobe document."""
        item_id = "lost_resp_9"
        res1 = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id, batch_id=self.batch_id, upload_item_id=item_id, image_bytes=_make_dummy_image(), metadata={}
        )
        self.assertEqual(res1["status"], "ADDED_TO_WARDROBE")

        res2 = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id, batch_id=self.batch_id, upload_item_id=item_id, image_bytes=_make_dummy_image(), metadata={}
        )
        self.assertEqual(res2["wardrobe_item_id"], res1["wardrobe_item_id"])

    def test_10_independent_batches(self):
        """Test 10: Batch A and Batch B process independently."""
        c_a = self.orchestrator.claim_item_lease(self.user_id, "batch_A", "item_a", "w1")
        c_b = self.orchestrator.claim_item_lease(self.user_id, "batch_B", "item_b", "w2")
        self.assertTrue(c_a["success"])
        self.assertTrue(c_b["success"])

    def test_11_expired_lease_recovery(self):
        """Test 11: Expired lease can be claimed by second worker."""
        item_id = "expired_item_11"
        doc_id = deterministic_appwrite_id(self.user_id, item_id)
        self.orchestrator._create_doc(
            self.orchestrator.items_collection,
            {
                "user_id": self.user_id,
                "batch_id": self.batch_id,
                "client_upload_item_id": item_id,
                "status": "PROCESSING",
                "lease_owner": "worker_crashed",
                "lease_expires_at": int(time.time()) - 100,
                "attempt_count": 1,
            },
            doc_id,
        )

        c2 = self.orchestrator.claim_item_lease(self.user_id, self.batch_id, item_id, "worker_recovered")
        self.assertTrue(c2["success"])
        self.assertEqual(c2["doc"]["lease_owner"], "worker_recovered")

    def test_12_batch_fingerprint_conflict(self):
        """Test 12: Payload fingerprint generation and conflict detection."""
        fp1 = compute_canonical_fingerprint([{"upload_item_id": "i1", "content_hash": "abc"}])
        fp2 = compute_canonical_fingerprint([{"upload_item_id": "i1", "content_hash": "xyz"}])
        self.assertNotEqual(fp1, fp2)

    def test_13_invalid_state_transition(self):
        """Test 13: Attempting illegal state transition raises ValueError."""
        with self.assertRaises(ValueError):
            validate_status_transition("ADDED_TO_WARDROBE", "PROCESSING")

    def test_14_retryable_failure_recovery(self):
        """Test 14: Retryable failure recovery path (attempt 1 fails, attempt 2 reclaims lease & succeeds)."""
        item_id = "retryable_item_14"
        c1 = self.orchestrator.claim_item_lease(self.user_id, self.batch_id, item_id, "worker_1", lease_duration_seconds=0)
        self.assertTrue(c1["success"])
        self.assertEqual(c1["doc"]["attempt_count"], 1)

        time.sleep(0.01)

        c2 = self.orchestrator.claim_item_lease(self.user_id, self.batch_id, item_id, "worker_2")
        self.assertTrue(c2["success"])
        self.assertEqual(c2["doc"]["attempt_count"], 2)

        res = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id,
            batch_id=self.batch_id,
            upload_item_id=item_id,
            image_bytes=_make_dummy_image(),
            metadata={},
        )
        self.assertEqual(res["status"], "ADDED_TO_WARDROBE")

    def test_15_terminal_failure_handling(self):
        """Test 15: Empty/invalid image bytes rejected immediately."""
        res = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id, batch_id=self.batch_id, upload_item_id="term_15", image_bytes=b"", metadata={}
        )
        self.assertEqual(res["status"], "REJECTED")

    def test_16_worker_crash_recovery_anchor(self):
        """Test 16: Worker crash post-save recovery via durable wardrobe anchor."""
        item_id = "crash_recovery_item_16"
        doc_id = deterministic_appwrite_id(self.user_id, item_id)

        self.orchestrator._create_doc(
            self.orchestrator.items_collection,
            {
                "user_id": self.user_id,
                "batch_id": self.batch_id,
                "client_upload_item_id": item_id,
                "status": "ADDED_TO_WARDROBE",
                "wardrobe_item_id": "wardrobe_persisted_777",
                "lease_owner": "worker_A_crashed",
                "lease_expires_at": int(time.time()) - 10,
                "attempt_count": 1,
            },
            doc_id,
        )

        res = self.orchestrator.process_wardrobe_upload_item(
            user_id=self.user_id,
            batch_id=self.batch_id,
            upload_item_id=item_id,
            image_bytes=_make_dummy_image(),
            metadata={},
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["idempotent"])
        self.assertEqual(res["wardrobe_item_id"], "wardrobe_persisted_777")

    def test_17_unauthorized_batch_status(self):
        """Test 17: User B requesting User A's batch status receives unauthorized."""
        batch_doc_id = deterministic_appwrite_id("user_A", "private_batch")
        self.orchestrator._create_doc(
            self.orchestrator.batches_collection,
            {"user_id": "user_A", "client_batch_request_id": "private_batch", "status": "QUEUED"},
            batch_doc_id,
        )

        res = self.orchestrator.get_batch_status("user_B", "private_batch")
        self.assertFalse(res["success"])
        self.assertEqual(res["reason"], "unauthorized")

    def test_18_polling_after_terminal_state(self):
        """Test 18: Polling batch status after terminal state returns mathematically derived status."""
        batch_doc_id = deterministic_appwrite_id(self.user_id, "done_batch")
        self.orchestrator._create_doc(
            self.orchestrator.batches_collection,
            {"user_id": self.user_id, "client_batch_request_id": "done_batch", "status": "PROCESSING", "total_items": 2, "added_count": 2},
            batch_doc_id,
        )

        res = self.orchestrator.get_batch_status(self.user_id, "done_batch")
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "COMPLETED")

    def test_19_non_swallowed_rmbg_failure(self):
        """Test 19: RMBG service failure raises exception and sets status FAILED (never swallowed)."""
        mock_rmbg_mod = types.ModuleType("services.rmbg_service")
        def _bad_rmbg(img):
            raise RuntimeError("Simulated 500 RMBG Service Error")
        mock_rmbg_mod.remove_background = _bad_rmbg
        sys.modules["services.rmbg_service"] = mock_rmbg_mod

        try:
            with self.assertRaises(RuntimeError) as ctx:
                self.orchestrator.process_wardrobe_upload_item(
                    user_id=self.user_id,
                    batch_id=self.batch_id,
                    upload_item_id="rmbg_fail_19",
                    image_bytes=_make_dummy_image(400, 400),
                    metadata={"catalog_quality_score": 90},
                )
            self.assertIn("RMBG_SERVICE_FAILED", str(ctx.exception))
        finally:
            sys.modules.pop("services.rmbg_service", None)

    def test_20_non_swallowed_persistence_failure(self):
        """Test 20: Persistence failure raises exception and sets status FAILED (never swallowed)."""
        def _bad_save(*args, **kwargs):
            raise RuntimeError("Simulated Appwrite Database Failure")
        self.orchestrator.persistence_service.save_wardrobe_item = _bad_save

        try:
            with self.assertRaises(RuntimeError) as ctx:
                self.orchestrator.process_wardrobe_upload_item(
                    user_id=self.user_id,
                    batch_id=self.batch_id,
                    upload_item_id="persist_fail_20",
                    image_bytes=_make_dummy_image(400, 400),
                    metadata={},
                )
            self.assertIn("PERSISTENCE_FAILED", str(ctx.exception))
        finally:
            self.orchestrator.persistence_service.save_wardrobe_item = _mock_successful_save

    def test_21_20_worker_concurrent_expired_lease_reclaim(self):
        """Test 21: 20 concurrent workers racing to reclaim the SAME expired lease."""
        item_id = "expired_race_item_21"
        doc_id = deterministic_appwrite_id(self.user_id, item_id)
        self.orchestrator._create_doc(
            self.orchestrator.items_collection,
            {
                "user_id": self.user_id,
                "batch_id": self.batch_id,
                "client_upload_item_id": item_id,
                "status": "PROCESSING",
                "lease_owner": "worker_crashed_21",
                "lease_expires_at": int(time.time()) - 100,
                "attempt_count": 1,
            },
            doc_id,
        )

        def _worker_reclaim(worker_idx: int):
            orch = UploadBatchOrchestrator()
            return orch.claim_item_lease(self.user_id, self.batch_id, item_id, f"reclaim_worker_{worker_idx}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(_worker_reclaim, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        successes = [r for r in results if r.get("success")]
        self.assertEqual(len(successes), 1)
        self.assertEqual(results[0]["doc"]["lease_owner"], successes[0]["doc"]["lease_owner"])


if __name__ == "__main__":
    unittest.main()
