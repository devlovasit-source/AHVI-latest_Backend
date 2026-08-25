"""Sequential wardrobe upload batch orchestration (AHVI P0 MVP).

Fixes the demonstrated multi-image upload problem: today the client sends
either one giant /analyze-batch request (all N images, one round trip, no
per-item progress, 240s client timeout) or one giant /save-selected request
(all approved items, one round trip, 120s client timeout). A slow/failed
image anywhere in the batch is invisible to the user until the whole request
returns or times out, and a timeout loses visibility into whichever items DID
succeed server-side.

This module lets the client instead process one image at a time
(client_upload_item_id + client_batch_request_id both deterministic), getting
an immediate per-item terminal result back after each call - "1 of N", "2 of
N", ... - while a durable per-item tracking document makes retries safe
(same item retried twice never persists twice) without inventing a second
domain pipeline: analysis, gating and persistence are the CURRENT production
functions (routers.wardrobe_capture.analyze_capture,
services.wardrobe_persistence_service.persist_selected_items), called once
per item instead of once per batch.

Adapted from a reference orchestrator design (deterministic IDs, item state
machine, sequential batch counters) - see reference/kavya_upload/ in this
worktree for the original. Two things were deliberately NOT ported:

1. Its own WardrobePersistenceService (writes a much thinner outfits document
   than current AHVI's schema - would silently regress image provenance,
   catalog generation, R2, Qdrant). CURRENT_AUTHORITY stays
   services.wardrobe_persistence_service.persist_selected_items.
2. Its own classify_image_suitability()/validate_cutout_quality() RMBG-route
   classifier - a second, competing AI pipeline. CURRENT_AUTHORITY stays
   analyze_capture()'s vision pipeline plus the existing
   _is_preview_item_save_approved()/_save_selected_block_reason() gating
   save-selected already applies.
3. Its database-level conditional-PATCH lease reclaim (CAS via Appwrite query
   predicates) - unverified against the actual deployed Appwrite server
   version within the MVP time budget. Lease reclaim here is a plain
   last-write-wins update_document instead (LEASE_MODE=MVP_SAFE_RETRY) -
   acceptable for a 30-user MVP; see the integration report for the tradeoff.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, Optional

from fastapi import Request

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError

logger = logging.getLogger("ahvi.upload_batch_orchestrator")


def _utcnow_iso() -> str:
    return datetime.now(dt_timezone.utc).isoformat()


def deterministic_appwrite_id(*parts: str) -> str:
    """Deterministic 36-character hex Appwrite document ID.

    Same inputs -> same document ID -> Appwrite's own create-with-id 409
    conflict IS the idempotency primitive: two callers racing to create the
    same (user, batch) or (user, item) tracking doc can never both succeed.
    """
    raw = ":".join(str(p).strip() for p in parts if p is not None).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:36]


# Required item/batch states (AHVI P0 upload MVP contract).
LEGAL_ITEM_TRANSITIONS: Dict[str, set] = {
    "PENDING": {"PROCESSING"},
    "PROCESSING": {"ADDED_TO_WARDROBE", "NEEDS_REVIEW", "REJECTED", "FAILED", "PROCESSING"},
    "ADDED_TO_WARDROBE": set(),
    "NEEDS_REVIEW": set(),
    "REJECTED": set(),
    "FAILED": set(),
}

LEGAL_BATCH_TRANSITIONS: Dict[str, set] = {
    "QUEUED": {"PROCESSING"},
    "PROCESSING": {"COMPLETED", "COMPLETED_WITH_ISSUES", "FAILED", "PROCESSING"},
    "COMPLETED": set(),
    "COMPLETED_WITH_ISSUES": set(),
    "FAILED": set(),
}

_TERMINAL_ITEM_STATUSES = {"ADDED_TO_WARDROBE", "NEEDS_REVIEW", "REJECTED", "FAILED"}


def _is_blocked_duplicate(item: Dict[str, Any], override_duplicate: bool) -> bool:
    """The one canonical duplicate signal is analyze_capture()'s own
    item["duplicate"] object (checked/is_duplicate/reason/confidence/
    matched_item_id) - see services.qdrant_service via
    routers.wardrobe_capture._find_upload_duplicate. This is a read-only
    check against that existing signal, never a second detector."""
    if override_duplicate:
        return False
    dup = item.get("duplicate") if isinstance(item.get("duplicate"), dict) else {}
    return bool(dup.get("checked")) and bool(dup.get("is_duplicate"))


def validate_status_transition(old_status: str, new_status: str, entity_type: str = "item") -> bool:
    old_s = str(old_status or "PENDING").upper().strip()
    new_s = str(new_status or "PENDING").upper().strip()
    if old_s == new_s:
        return True
    allowed_map = LEGAL_ITEM_TRANSITIONS if entity_type == "item" else LEGAL_BATCH_TRANSITIONS
    allowed = allowed_map.get(old_s, set())
    if new_s not in allowed:
        raise ValueError(f"Invalid {entity_type} status transition: '{old_s}' -> '{new_s}'. Allowed: {allowed}")
    return True


def _memory_fallback_allowed() -> bool:
    """Phase 9 contract: silent in-memory durability is a TEST convenience
    only. Production (or any unset/unknown ENVIRONMENT) must never pretend an
    unreachable Appwrite write succeeded durably."""
    env = str(os.getenv("ENVIRONMENT", "")).strip().lower()
    return env in {"local", "test", "testing"}


class UploadBatchInfraError(RuntimeError):
    """Typed infrastructure failure - Appwrite unavailable outside test/local.
    Never caught and converted into a fake success anywhere in this module."""


class UploadBatchOrchestrator:
    # Test/local-only fallback store - never used unless _memory_fallback_allowed().
    _memory_batches: Dict[str, Dict[str, Any]] = {}
    _memory_items: Dict[str, Dict[str, Any]] = {}

    def __init__(self) -> None:
        self.proxy = AppwriteProxy()
        self.batches_collection = "upload_batches"
        self.items_collection = "upload_batch_items"

    # ------------------------------------------------------------------ #
    # doc plumbing
    # ------------------------------------------------------------------ #

    def _is_proxy_configured(self) -> bool:
        return bool(self.proxy.endpoint and self.proxy.project_id and self.proxy.api_key and self.proxy.database_id)

    def _get_doc(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        if self._is_proxy_configured():
            try:
                doc = self.proxy.get_document(collection, doc_id)
                if isinstance(doc, dict) and doc:
                    return doc
            except AppwriteProxyError as exc:
                if exc.status_code == 404:
                    return None
                raise UploadBatchInfraError(f"UPLOAD_BATCH_STORE_UNAVAILABLE: {exc}") from exc
            return None
        if not _memory_fallback_allowed():
            raise UploadBatchInfraError(
                "UPLOAD_BATCH_STORE_UNAVAILABLE: Appwrite is not configured and "
                "ENVIRONMENT is not local/test/testing - refusing an in-memory fallback."
            )
        store = self._memory_batches if collection == self.batches_collection else self._memory_items
        return store.get(doc_id)

    def _create_doc(self, collection: str, data: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
        """Atomic initial claim: Appwrite create-with-id 409s if doc_id already
        exists - that 409 IS the idempotency guarantee, not a retry helper."""
        if self._is_proxy_configured():
            try:
                created = self.proxy.create_document(collection, data, document_id=doc_id)
            except AppwriteProxyError as exc:
                if exc.status_code == 409:
                    raise ValueError(f"Document {doc_id} already exists in Appwrite") from exc
                raise UploadBatchInfraError(f"UPLOAD_BATCH_STORE_UNAVAILABLE: {exc}") from exc
            merged = dict(data)
            if isinstance(created, dict):
                merged.update(created)
            return merged
        if not _memory_fallback_allowed():
            raise UploadBatchInfraError(
                "UPLOAD_BATCH_STORE_UNAVAILABLE: Appwrite is not configured and "
                "ENVIRONMENT is not local/test/testing - refusing an in-memory fallback."
            )
        store = self._memory_batches if collection == self.batches_collection else self._memory_items
        if doc_id in store:
            raise ValueError(f"Document {doc_id} already exists")
        data = dict(data)
        data["$id"] = doc_id
        store[doc_id] = data
        return dict(data)

    def _update_doc(self, collection: str, doc_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._is_proxy_configured():
            try:
                updated = self.proxy.update_document(collection, doc_id, data)
            except AppwriteProxyError as exc:
                raise UploadBatchInfraError(f"UPLOAD_BATCH_STORE_UNAVAILABLE: {exc}") from exc
            merged = dict(data)
            if isinstance(updated, dict):
                merged.update(updated)
            return merged
        if not _memory_fallback_allowed():
            raise UploadBatchInfraError(
                "UPLOAD_BATCH_STORE_UNAVAILABLE: Appwrite is not configured and "
                "ENVIRONMENT is not local/test/testing - refusing an in-memory fallback."
            )
        store = self._memory_batches if collection == self.batches_collection else self._memory_items
        existing = dict(store.get(doc_id) or {})
        existing.update(data)
        store[doc_id] = existing
        return dict(existing)

    # ------------------------------------------------------------------ #
    # batch lifecycle
    # ------------------------------------------------------------------ #

    def create_or_resume_batch(self, *, user_id: str, client_batch_request_id: str, total_items: int) -> Dict[str, Any]:
        """Same user + same client_batch_request_id => same logical batch
        document, whether this is the first call or a client retry."""
        doc_id = deterministic_appwrite_id(user_id, client_batch_request_id)
        data = {
            "user_id": user_id,
            "client_batch_request_id": client_batch_request_id,
            "status": "QUEUED",
            "total_items": int(total_items),
            "added_count": 0,
            "needs_review_count": 0,
            "rejected_count": 0,
            "failed_count": 0,
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
        }
        try:
            doc = self._create_doc(self.batches_collection, data, doc_id)
            return {"success": True, "batch_id": client_batch_request_id, "resumed": False, "doc": doc}
        except ValueError:
            existing = self._get_doc(self.batches_collection, doc_id)
            if not existing:
                raise UploadBatchInfraError(f"UPLOAD_BATCH_STORE_UNAVAILABLE: batch {doc_id} vanished after 409")
            if str(existing.get("user_id") or "") != str(user_id):
                return {"success": False, "reason": "unauthorized"}
            return {"success": True, "batch_id": client_batch_request_id, "resumed": True, "doc": existing}

    def _recompute_batch_status(self, batch_doc: Dict[str, Any]) -> str:
        total = int(batch_doc.get("total_items") or 0)
        added = int(batch_doc.get("added_count") or 0)
        needs_review = int(batch_doc.get("needs_review_count") or 0)
        rejected = int(batch_doc.get("rejected_count") or 0)
        failed = int(batch_doc.get("failed_count") or 0)
        terminal = added + needs_review + rejected + failed
        if total <= 0 or terminal < total:
            return "PROCESSING"
        if added == total:
            return "COMPLETED"
        if added > 0:
            return "COMPLETED_WITH_ISSUES"
        return "FAILED"

    def _bump_batch_counter(
        self, user_id: str, client_batch_request_id: str, field: str, *, migrate_from: Optional[str] = None
    ) -> None:
        """Increment `field` by one. migrate_from also decrements that other
        counter by one in the SAME update - used when an item that was
        already counted once (e.g. needs_review_count on a duplicate flag,
        or failed_count on a FAILED item) reaches a DIFFERENT terminal state
        on an explicit re-entry (e.g. "Add anyway" -> added_count, or a
        FAILED retry -> added_count), so the item is never double-counted
        across the batch's terminal-count total.

        migrate_from == field (e.g. a FAILED retry that fails AGAIN) is the
        one case that must NOT increment: the item was already counted in
        this exact bucket - re-affirming the same terminal state is a no-op
        for the count, not a second item."""
        batch_doc_id = deterministic_appwrite_id(user_id, client_batch_request_id)
        doc = self._get_doc(self.batches_collection, batch_doc_id)
        if not doc:
            return
        doc = dict(doc)
        update: Dict[str, Any] = {}
        if migrate_from != field:
            doc[field] = int(doc.get(field) or 0) + 1
            update[field] = doc[field]
            if migrate_from:
                doc[migrate_from] = max(0, int(doc.get(migrate_from) or 0) - 1)
                update[migrate_from] = doc[migrate_from]
        doc["status"] = self._recompute_batch_status(doc)
        update["status"] = doc["status"]
        update["updated_at"] = _utcnow_iso()
        self._update_doc(self.batches_collection, batch_doc_id, update)

    def get_batch_status(self, user_id: str, client_batch_request_id: str) -> Dict[str, Any]:
        batch_doc_id = deterministic_appwrite_id(user_id, client_batch_request_id)
        doc = self._get_doc(self.batches_collection, batch_doc_id)
        if not doc:
            return {"success": False, "reason": "batch_not_found"}
        if str(doc.get("user_id") or "") != str(user_id):
            return {"success": False, "reason": "unauthorized"}
        status = self._recompute_batch_status(doc)
        return {
            "success": True,
            "batch_id": client_batch_request_id,
            "batch_doc_id": batch_doc_id,
            "status": status,
            "total_items": int(doc.get("total_items") or 0),
            "added_count": int(doc.get("added_count") or 0),
            "needs_review_count": int(doc.get("needs_review_count") or 0),
            "rejected_count": int(doc.get("rejected_count") or 0),
            "failed_count": int(doc.get("failed_count") or 0),
            "poll_after_ms": 1000,
        }

    # ------------------------------------------------------------------ #
    # per-item claim (MVP_SAFE_RETRY - see module docstring point 3)
    # ------------------------------------------------------------------ #

    def claim_item(self, *, user_id: str, batch_id: str, client_upload_item_id: str) -> Dict[str, Any]:
        doc_id = deterministic_appwrite_id(user_id, client_upload_item_id)
        item_data = {
            "user_id": user_id,
            "batch_id": batch_id,
            "client_upload_item_id": client_upload_item_id,
            "status": "PROCESSING",
            "attempt_count": 1,
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
        }
        try:
            doc = self._create_doc(self.items_collection, item_data, doc_id)
            return {"success": True, "reason": "created", "doc": doc}
        except ValueError:
            pass

        existing = self._get_doc(self.items_collection, doc_id)
        if not existing:
            raise UploadBatchInfraError(f"UPLOAD_BATCH_STORE_UNAVAILABLE: item {doc_id} vanished after 409")

        cur_status = str(existing.get("status") or "PENDING").upper()
        if cur_status in _TERMINAL_ITEM_STATUSES:
            return {"success": False, "reason": f"terminal_status:{cur_status}", "doc": existing}

        # PROCESSING with no CAS-verified lease expiry: last-write-wins retry
        # claim. Safe for a 30-user MVP where the same item is not being
        # driven by two truly concurrent workers; see LEASE_MODE=MVP_SAFE_RETRY.
        validate_status_transition(cur_status, "PROCESSING")
        item_data["attempt_count"] = int(existing.get("attempt_count") or 1) + 1
        doc = self._update_doc(self.items_collection, doc_id, item_data)
        return {"success": True, "reason": "reclaimed", "doc": doc}

    # ------------------------------------------------------------------ #
    # single-item processing - CURRENT domain pipeline, not a new one
    # ------------------------------------------------------------------ #

    async def process_single_batch_item(
        self,
        *,
        http_request: Request,
        user_id: str,
        batch_id: str,
        client_upload_item_id: str,
        image_base64: str,
        metadata: Optional[Dict[str, Any]] = None,
        override_duplicate: bool = False,
        reviewed_item: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Canonical single-item processor: persist the ONE garment the user
        already reviewed and approved in preview, gate it with the SAME
        approval rule save-selected already applies, persist it through the
        SAME persist_selected_items() the rest of the app uses. No competing
        RMBG/catalog/persistence stack.

        reviewed_item, when supplied, IS that already-reviewed garment (the
        same detected-item dict shape analyze_capture()/save-selected already
        use) - it is used directly, WITHOUT calling analyze_capture() again,
        since the garment was already detected and approved once during
        preview. image_base64 is only re-analyzed as a backwards-compatible
        fallback when reviewed_item is absent (older clients).

        override_duplicate is this ONE item's explicit "Add anyway" - it is
        scoped to exactly this call/doc_id and can never affect any other
        item's processing, since every item is its own independent call.
        """
        t0 = time.time()
        metrics: Dict[str, Any] = {
            "queue_wait_ms": 0,
            "analysis_ms": 0,
            "persistence_ms": 0,
            "total_item_ms": 0,
        }
        doc_id = deterministic_appwrite_id(user_id, client_upload_item_id)
        # Set to the batch counter field this item was previously counted
        # under, whenever a terminal _update_doc below moves it to a NEW
        # terminal state - _bump_batch_counter decrements that field in the
        # SAME update so the item is never double-counted across the batch's
        # terminal total (added + needs_review + rejected + failed).
        migrate_from_field: Optional[str] = None

        existing = self._get_doc(self.items_collection, doc_id)
        existing_status = str(existing.get("status") or "").upper() if existing else ""

        if existing_status == "ADDED_TO_WARDROBE":
            wardrobe_item_id = str(existing.get("wardrobe_item_id") or "").strip()
            if not wardrobe_item_id:
                raise UploadBatchInfraError(
                    f"DATA_INTEGRITY_ERROR: item {doc_id} is ADDED_TO_WARDROBE with no wardrobe_item_id"
                )
            return {
                "success": True,
                "status": "ADDED_TO_WARDROBE",
                "wardrobe_item_id": wardrobe_item_id,
                "idempotent": True,
                "metrics": metrics,
            }
        elif existing_status == "REJECTED":
            # REJECTED (no item detected) stays terminal - nothing about a
            # bare retry changes what was actually in the image.
            return {
                "success": False,
                "status": existing.get("status"),
                "idempotent": True,
                "error_code": existing.get("error_code"),
                "metrics": metrics,
            }
        elif existing_status == "NEEDS_REVIEW":
            if not override_duplicate:
                # Ordinary NEEDS_REVIEW (not-auto-approved) AND a duplicate
                # flag without explicit override both stay terminal here -
                # the only way out of NEEDS_REVIEW is override_duplicate=True.
                return {
                    "success": False,
                    "status": existing.get("status"),
                    "idempotent": True,
                    "error_code": existing.get("error_code"),
                    "matched_item_id": existing.get("matched_item_id"),
                    "duplicate_reason": existing.get("duplicate_reason"),
                    "duplicate_confidence": existing.get("duplicate_confidence"),
                    "metrics": metrics,
                }
            # NEEDS_REVIEW + override_duplicate=True is the one deliberate
            # re-entry point for "Add anyway" arriving after this exact item
            # was already flagged a duplicate. NEEDS_REVIEW has no OTHER
            # legal exit (LEGAL_ITEM_TRANSITIONS keeps it terminal for every
            # other caller/poll), so this bypasses claim_item()'s generic
            # transition check deliberately rather than weakening the state
            # machine for everyone: this path only runs when the caller
            # explicitly asked to override duplicate-blocking on THIS item.
            self._update_doc(
                self.items_collection, doc_id,
                {"status": "PROCESSING", "attempt_count": int(existing.get("attempt_count") or 1) + 1, "updated_at": _utcnow_iso()},
            )
            migrate_from_field = "needs_review_count"
        elif existing_status == "FAILED":
            # FAILED is the one terminal status that IS retryable: unlike
            # REJECTED/NEEDS_REVIEW, a FAILED item never told the user
            # anything true about their photo - it's a transport/persistence
            # hiccup, and the exact same reviewed_item/image is worth trying
            # again. Same batch_id/client_upload_item_id, no new IDs. Clear
            # error_code so a stale error can never survive into whatever
            # terminal doc this attempt produces next.
            self._update_doc(
                self.items_collection, doc_id,
                {
                    "status": "PROCESSING",
                    "attempt_count": int(existing.get("attempt_count") or 1) + 1,
                    "error_code": None,
                    "updated_at": _utcnow_iso(),
                },
            )
            migrate_from_field = "failed_count"
        else:
            # No existing doc, or an existing PROCESSING lease (possibly
            # stale) - claim_item()'s existing create/reclaim logic handles
            # both, unchanged.
            claim = self.claim_item(user_id=user_id, batch_id=batch_id, client_upload_item_id=client_upload_item_id)
            if not claim["success"]:
                doc = claim.get("doc") or {}
                return {
                    "success": str(doc.get("status")).upper() == "ADDED_TO_WARDROBE",
                    "status": doc.get("status", "FAILED"),
                    "wardrobe_item_id": doc.get("wardrobe_item_id"),
                    "idempotent": True,
                    "reason": claim["reason"],
                    "metrics": metrics,
                }

        # Import locally: routers.wardrobe_capture imports many services at
        # module scope, so importing it back from a service module at import
        # time risks a circular import. Deferred import breaks that cycle.
        from routers.wardrobe_capture import (
            CaptureAnalyzeRequest,
            analyze_capture,
            _is_preview_item_save_approved,
            _try_upload_inline_images,
            _maybe_generate_catalog_image,
            _image_cache_enabled,
            _image_cache_get_sync,
        )
        from services.wardrobe_persistence_service import persist_selected_items

        if reviewed_item is not None:
            # The exact garment already detected + approved during preview -
            # reuse it as-is, never re-run detection on the source bytes.
            items = [dict(reviewed_item)] if isinstance(reviewed_item, dict) else []
            # WARDROBE_ANALYZE_IMAGE_CACHE: same restore save-selected already
            # does - the preview may have sent an image_cache_token instead of
            # inline base64 to avoid re-uploading ~MB payloads.
            if items and _image_cache_enabled():
                for _it in items:
                    _tok = str(_it.get("image_cache_token") or "").strip()
                    if _tok and not _it.get("raw_image_base64"):
                        _cached_b64 = _image_cache_get_sync(_tok)
                        if _cached_b64:
                            _it["raw_image_base64"] = _cached_b64
        else:
            t_an = time.time()
            try:
                analyze_result = await analyze_capture(
                    http_request,
                    CaptureAnalyzeRequest(user_id=user_id, image_base64=image_base64, auto_save=False),
                )
            except Exception as exc:
                metrics["analysis_ms"] = int((time.time() - t_an) * 1000)
                self._update_doc(
                    self.items_collection, doc_id,
                    {"status": "FAILED", "error_code": "ANALYSIS_FAILED", "updated_at": _utcnow_iso()},
                )
                self._bump_batch_counter(
                    user_id, batch_id, "failed_count",
                    migrate_from=migrate_from_field,
                )
                raise UploadBatchInfraError(f"ANALYSIS_FAILED: {exc}") from exc
            metrics["analysis_ms"] = int((time.time() - t_an) * 1000)

            items = analyze_result.get("items") if isinstance(analyze_result, dict) else []
            items = [i for i in (items or []) if isinstance(i, dict)]

        if not items:
            self._update_doc(
                self.items_collection, doc_id,
                {"status": "REJECTED", "error_code": "NO_ITEM_DETECTED", "updated_at": _utcnow_iso()},
            )
            self._bump_batch_counter(
                user_id, batch_id, "rejected_count",
                migrate_from=migrate_from_field,
            )
            metrics["total_item_ms"] = int((time.time() - t0) * 1000)
            return {"success": False, "status": "REJECTED", "reason": "no_item_detected", "metrics": metrics}

        for item in items:
            if isinstance(metadata, dict):
                item.setdefault("category", metadata.get("category"))
                item.setdefault("name", metadata.get("name"))

        save_approved = [i for i in items if _is_preview_item_save_approved(i)]
        duplicate_blocked = [i for i in save_approved if _is_blocked_duplicate(i, override_duplicate)]
        blocked_ids = {id(i) for i in duplicate_blocked}
        approved = [i for i in save_approved if id(i) not in blocked_ids]

        if not approved and duplicate_blocked:
            dup = duplicate_blocked[0].get("duplicate") or {}
            self._update_doc(
                self.items_collection, doc_id,
                {
                    "status": "NEEDS_REVIEW",
                    "error_code": "DUPLICATE_WARDROBE_ITEM",
                    "matched_item_id": dup.get("matched_item_id"),
                    "duplicate_reason": dup.get("reason"),
                    "duplicate_confidence": dup.get("confidence"),
                    "updated_at": _utcnow_iso(),
                },
            )
            self._bump_batch_counter(
                user_id, batch_id, "needs_review_count",
                migrate_from=migrate_from_field,
            )
            metrics["total_item_ms"] = int((time.time() - t0) * 1000)
            return {
                "success": False,
                "status": "NEEDS_REVIEW",
                "reason": "duplicate_wardrobe_item",
                "matched_item_id": dup.get("matched_item_id"),
                "duplicate_reason": dup.get("reason"),
                "duplicate_confidence": dup.get("confidence"),
                "metrics": metrics,
            }

        if not approved:
            self._update_doc(
                self.items_collection, doc_id,
                {"status": "NEEDS_REVIEW", "error_code": "NOT_AUTO_APPROVED", "updated_at": _utcnow_iso()},
            )
            self._bump_batch_counter(
                user_id, batch_id, "needs_review_count",
                migrate_from=migrate_from_field,
            )
            metrics["total_item_ms"] = int((time.time() - t0) * 1000)
            return {"success": False, "status": "NEEDS_REVIEW", "reason": "not_auto_approved", "items": items, "metrics": metrics}

        # Ensure raw_url/masked_url/normalized_url exist before persisting -
        # SAME mechanism the canonical save-selected flow forces
        # (allow_fast_mode_skip=False, prefer_inline=True). Without this,
        # persist_selected_items() silently drops any item missing all three
        # URL fields, which is exactly how a reviewed, already-approved item
        # used to vanish with a bare HTTP 200 and no persisted wardrobe row.
        approved = [
            _try_upload_inline_images(item, allow_fast_mode_skip=False, prefer_inline=True)
            for item in approved
        ]

        # Same canonical step save-selected's own Phase 2 runs on every
        # prepared item, unconditionally - the function itself decides
        # whether anything happens (catalog flags, category, crop source).
        # Required whenever WARDROBE_PRIVACY_CATALOG_ONLY blocks the inline
        # raw/masked upload above for a face-risk garment: normalized_url can
        # then ONLY come from here. Never raises; a failure just leaves the
        # item without a URL, which persist_selected_items/the check below
        # already turns into an explicit UPLOAD_ITEM_PERSISTENCE_FAILED.
        for item in approved:
            _maybe_generate_catalog_image(item)

        approved_ids = [str(i.get("item_id") or "").strip() for i in approved if str(i.get("item_id") or "").strip()]

        t_persist = time.time()
        try:
            save_result = persist_selected_items(
                user_id=user_id,
                selected_item_ids=approved_ids,
                detected_items=approved,
            )
        except Exception as exc:
            metrics["persistence_ms"] = int((time.time() - t_persist) * 1000)
            self._update_doc(
                self.items_collection, doc_id,
                {"status": "FAILED", "error_code": "PERSISTENCE_FAILED", "updated_at": _utcnow_iso()},
            )
            self._bump_batch_counter(
                user_id, batch_id, "failed_count",
                migrate_from=migrate_from_field,
            )
            raise UploadBatchInfraError(f"PERSISTENCE_FAILED: {exc}") from exc
        metrics["persistence_ms"] = int((time.time() - t_persist) * 1000)

        saved_items = save_result.get("items") if isinstance(save_result, dict) else []
        saved_items = [i for i in (saved_items or []) if isinstance(i, dict)]

        if not saved_items:
            # ZERO synthetic wardrobe IDs: a save that persisted nothing is a
            # failure, never a fabricated success.
            logger.warning(
                "ahvi.upload_batch.persistence_returned_no_items batch_id=%s item_id=%s user_id=%s "
                "has_raw_url=%s has_masked_url=%s has_normalized_url=%s",
                batch_id, client_upload_item_id, user_id,
                bool(approved[0].get("raw_url")) if approved else None,
                bool(approved[0].get("masked_url")) if approved else None,
                bool(approved[0].get("normalized_url")) if approved else None,
            )
            self._update_doc(
                self.items_collection, doc_id,
                {"status": "FAILED", "error_code": "UPLOAD_ITEM_PERSISTENCE_FAILED", "updated_at": _utcnow_iso()},
            )
            self._bump_batch_counter(
                user_id, batch_id, "failed_count",
                migrate_from=migrate_from_field,
            )
            metrics["total_item_ms"] = int((time.time() - t0) * 1000)
            return {
                "success": False,
                "status": "FAILED",
                "error_code": "UPLOAD_ITEM_PERSISTENCE_FAILED",
                "reason": "persistence_returned_no_items",
                "metrics": metrics,
            }

        wardrobe_item_id = str(saved_items[0].get("$id") or saved_items[0].get("id") or "").strip()
        if not wardrobe_item_id:
            self._update_doc(
                self.items_collection, doc_id,
                {"status": "FAILED", "error_code": "PERSISTENCE_FAILED", "updated_at": _utcnow_iso()},
            )
            self._bump_batch_counter(
                user_id, batch_id, "failed_count",
                migrate_from=migrate_from_field,
            )
            raise UploadBatchInfraError("PERSISTENCE_FAILED: saved item missing an id")

        self._update_doc(
            self.items_collection, doc_id,
            {"status": "ADDED_TO_WARDROBE", "wardrobe_item_id": wardrobe_item_id, "updated_at": _utcnow_iso()},
        )
        self._bump_batch_counter(
            user_id, batch_id, "added_count",
            migrate_from=migrate_from_field,
        )
        metrics["total_item_ms"] = int((time.time() - t0) * 1000)
        logger.info(
            "ahvi.upload_batch.item_added batch_id=%s item_id=%s analysis_ms=%s persistence_ms=%s total_item_ms=%s",
            batch_id, client_upload_item_id, metrics["analysis_ms"], metrics["persistence_ms"], metrics["total_item_ms"],
        )

        return {
            "success": True,
            "status": "ADDED_TO_WARDROBE",
            "wardrobe_item_id": wardrobe_item_id,
            "metrics": metrics,
        }


upload_batch_orchestrator = UploadBatchOrchestrator()
