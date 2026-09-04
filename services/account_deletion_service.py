"""45-day soft-delete and hard purge account lifecycle management service.

Handles:
- Scheduling soft account deletion with a 45-day grace period.
- Session revocation upon deletion request.
- Cancellation and reactivation within the 45-day window.
- Permanent cascading hard purge across Cloudflare R2, Qdrant, Appwrite DB, and Appwrite Auth.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ahvi.account_deletion")

GRACE_PERIOD_DAYS = 45
ACCOUNT_STATUS_ACTIVE = "active"
ACCOUNT_STATUS_PENDING_DELETION = "pending_deletion"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _now_utc().isoformat()


def _parse_iso(iso_str: Optional[str]) -> Optional[datetime]:
    if not iso_str or not isinstance(iso_str, str):
        return None
    try:
        clean = iso_str.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except Exception:
        return None


def request_account_deletion(
    user_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Schedules an account for permanent deletion after a 45-day grace period.

    - Sets account_status to 'pending_deletion'.
    - Marks deletion_requested_at as now.
    - Sets deletion_scheduled_at to exactly 45 days in the future.
    - Revokes active Appwrite sessions for the user.
    """
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id cannot be empty")

    now = _now_utc()
    now_iso = now.isoformat()
    scheduled_at = now + timedelta(days=GRACE_PERIOD_DAYS)
    scheduled_iso = scheduled_at.isoformat()

    update_payload: Dict[str, Any] = {
        "account_status": ACCOUNT_STATUS_PENDING_DELETION,
        "deletion_requested_at": now_iso,
        "deletion_scheduled_at": scheduled_iso,
        "deletion_reason": str(reason or "").strip() or None,
    }

    # 1. Update user profile in Appwrite database
    profile_updated = False
    try:
        from services.appwrite_proxy import AppwriteProxy

        proxy = AppwriteProxy()
        try:
            proxy.update_document("users", uid, update_payload)
            profile_updated = True
        except Exception:
            # Create document if it does not yet exist
            create_payload = {"userId": uid, **update_payload}
            proxy.create_document("users", create_payload, document_id=uid)
            profile_updated = True
        logger.info("Account deletion requested user_id=%s scheduled_at=%s", uid, scheduled_iso)
    except Exception as exc:
        logger.exception("Failed to update user profile deletion status user_id=%s: %s", uid, exc)
        raise RuntimeError(f"Could not persist account deletion request: {exc}") from exc

    # 2. Revoke active Appwrite sessions
    sessions_revoked = False
    try:
        from services.appwrite_service import get_admin_client, is_appwrite_configured

        if is_appwrite_configured():
            from appwrite.services.users import Users

            users_svc = Users(get_admin_client())
            users_svc.delete_sessions(uid)
            sessions_revoked = True
            logger.info("Successfully revoked Appwrite sessions user_id=%s", uid)
        else:
            logger.warning("Appwrite not configured; skipping session revocation for user_id=%s", uid)
    except Exception as exc:
        logger.warning("Session revocation error user_id=%s: %s", uid, exc)

    return {
        "success": True,
        "user_id": uid,
        "account_status": ACCOUNT_STATUS_PENDING_DELETION,
        "deletion_requested_at": now_iso,
        "deletion_scheduled_at": scheduled_iso,
        "deletion_reason": reason,
        "grace_period_days": GRACE_PERIOD_DAYS,
        "sessions_revoked": sessions_revoked,
        "message": (
            f"Account scheduled for permanent deletion on {scheduled_iso}. "
            f"You have a {GRACE_PERIOD_DAYS}-day grace period to cancel deletion by logging back in."
        ),
    }


def cancel_account_deletion(user_id: str) -> Dict[str, Any]:
    """Restores account_status to 'active' and clears deletion timestamps."""
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id cannot be empty")

    update_payload: Dict[str, Any] = {
        "account_status": ACCOUNT_STATUS_ACTIVE,
        "deletion_requested_at": "",
        "deletion_scheduled_at": "",
        "deletion_reason": "",
    }

    try:
        from services.appwrite_proxy import AppwriteProxy

        proxy = AppwriteProxy()
        try:
            proxy.update_document("users", uid, update_payload)
        except Exception:
            create_payload = {"userId": uid, **update_payload}
            proxy.create_document("users", create_payload, document_id=uid)
        logger.info("Account deletion cancelled and restored to active user_id=%s", uid)
    except Exception as exc:
        logger.exception("Failed to cancel account deletion user_id=%s: %s", uid, exc)
        raise RuntimeError(f"Could not cancel account deletion: {exc}") from exc

    return {
        "success": True,
        "user_id": uid,
        "account_status": ACCOUNT_STATUS_ACTIVE,
        "message": "Account deletion cancelled successfully. Your account is active.",
    }


def get_account_deletion_status(user_id: str) -> Dict[str, Any]:
    """Returns the current deletion state and remaining days in the grace period."""
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id cannot be empty")

    account_status = ACCOUNT_STATUS_ACTIVE
    requested_at_raw: Optional[str] = None
    scheduled_at_raw: Optional[str] = None
    reason: Optional[str] = None

    try:
        from services.data_access_service import get_user_profile

        profile = get_user_profile(user_id=uid) or {}
        account_status = str(profile.get("account_status") or ACCOUNT_STATUS_ACTIVE).strip().lower()
        requested_at_raw = profile.get("deletion_requested_at") or None
        scheduled_at_raw = profile.get("deletion_scheduled_at") or None
        reason = profile.get("deletion_reason") or None
    except Exception as exc:
        logger.warning("Could not read user profile for deletion status user_id=%s: %s", uid, exc)

    now = _now_utc()
    days_remaining: Optional[int] = None
    hours_remaining: Optional[int] = None
    is_expired = False

    if account_status == ACCOUNT_STATUS_PENDING_DELETION and scheduled_at_raw:
        scheduled_dt = _parse_iso(scheduled_at_raw)
        if scheduled_dt:
            delta = scheduled_dt - now
            total_seconds = delta.total_seconds()
            if total_seconds <= 0:
                days_remaining = 0
                hours_remaining = 0
                is_expired = True
            else:
                days_remaining = max(0, delta.days)
                hours_remaining = max(0, int(total_seconds // 3600))
                is_expired = False

    return {
        "user_id": uid,
        "account_status": account_status,
        "deletion_requested_at": requested_at_raw,
        "deletion_scheduled_at": scheduled_at_raw,
        "deletion_reason": reason,
        "days_remaining": days_remaining,
        "hours_remaining": hours_remaining,
        "is_expired": is_expired,
        "grace_period_days": GRACE_PERIOD_DAYS,
    }


def execute_hard_purge(user_id: str) -> Dict[str, Any]:
    """Permanently cascades cleanup across Cloudflare R2, Qdrant vectors,

    Appwrite database collections, and Appwrite Auth user identity.
    Uses defensive try/except blocks with structured logging for each target.
    """
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id cannot be empty")

    logger.info("Starting execute_hard_purge for user_id=%s", uid)
    started_at = _iso_now()

    summary: Dict[str, Any] = {
        "user_id": uid,
        "started_at": started_at,
        "success": False,
        "targets": {},
    }

    # =========================================================================
    # 1. Cloudflare R2 Storage (User Assets)
    # =========================================================================
    r2_summary: Dict[str, Any] = {
        "status": "pending",
        "objects_deleted": 0,
        "errors": [],
    }
    try:
        from services.r2_storage import R2Storage, R2StorageError

        storage = R2Storage()
        client = storage._client()
        deleted_count = 0

        # A. Remove avatar objects
        if storage.raw_bucket:
            try:
                for obj in client.list_objects(storage.raw_bucket, prefix=f"avatar_{uid}", recursive=True):
                    client.remove_object(storage.raw_bucket, obj.object_name)
                    deleted_count += 1
            except Exception as e:
                r2_summary["errors"].append(f"raw_bucket avatars: {e}")

        # B. Remove style board objects from style_boards_bucket and raw_bucket
        target_buckets = {b for b in [storage.style_boards_bucket, storage.raw_bucket] if b}
        for b in target_buckets:
            try:
                for obj in client.list_objects(b, prefix=f"style_board_{uid}", recursive=True):
                    client.remove_object(b, obj.object_name)
                    deleted_count += 1
            except Exception as e:
                r2_summary["errors"].append(f"style_boards in {b}: {e}")

        # C. Remove wardrobe image assets
        try:
            from services.appwrite_proxy import AppwriteProxy

            proxy = AppwriteProxy()
            outfits_doc = proxy.list_documents("outfits", user_id=uid, limit=500)
            items = (
                outfits_doc.get("documents")
                if isinstance(outfits_doc, dict)
                else (outfits_doc if isinstance(outfits_doc, list) else [])
            )
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("$id") or item.get("id") or "").strip()
                if item_id and storage.wardrobe_bucket:
                    for obj_name in [
                        storage.catalog_object_name(item_id),
                        storage.catalog_png_object_name(item_id),
                    ]:
                        try:
                            client.remove_object(storage.wardrobe_bucket, obj_name)
                            deleted_count += 1
                        except Exception:
                            pass
                raw_cand = item.get("raw_file_name") or item.get("raw_name")
                masked_cand = item.get("masked_file_name") or item.get("masked_name")
                norm_cand = item.get("normalized_file_name") or item.get("normalized_name")
                del_res = storage.delete_wardrobe_images(
                    raw_file_name=raw_cand or "",
                    masked_file_name=masked_cand or "",
                    normalized_file_name=norm_cand or "",
                )
                deleted_count += sum(1 for v in del_res.values() if v)
        except Exception as e:
            r2_summary["errors"].append(f"wardrobe images: {e}")

        r2_summary["objects_deleted"] = deleted_count
        r2_summary["status"] = "completed" if not r2_summary["errors"] else "partial"
        logger.info("Hard purge R2 completed user_id=%s objects_deleted=%d", uid, deleted_count)
    except Exception as exc:
        r2_summary["status"] = "failed"
        r2_summary["errors"].append(str(exc))
        logger.exception("Hard purge R2 failed user_id=%s: %s", uid, exc)

    summary["targets"]["r2_storage"] = r2_summary

    # =========================================================================
    # 2. Qdrant Vector Collections (Wardrobe & Memory Embeddings)
    # =========================================================================
    qdrant_summary: Dict[str, Any] = {
        "status": "pending",
        "collections_purged": [],
        "errors": [],
    }
    try:
        from services.qdrant_service import qdrant_service

        if qdrant_service.client:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            user_filter = Filter(
                should=[
                    FieldCondition(key="userId", match=MatchValue(value=uid)),
                    FieldCondition(key="user_id", match=MatchValue(value=uid)),
                ]
            )
            collections = [
                qdrant_service.collection,
                qdrant_service.image_collection,
                qdrant_service.memory_collection,
                qdrant_service.user_memory_collection,
            ]
            for col in collections:
                if not col:
                    continue
                try:
                    qdrant_service.client.delete(
                        collection_name=col,
                        points_selector=user_filter,
                    )
                    qdrant_summary["collections_purged"].append(col)
                except Exception as col_exc:
                    qdrant_summary["errors"].append(f"{col}: {col_exc}")
                    logger.warning("Hard purge Qdrant failed for col=%s user_id=%s: %s", col, uid, col_exc)

            qdrant_summary["status"] = "completed" if not qdrant_summary["errors"] else "partial"
            logger.info("Hard purge Qdrant completed user_id=%s collections=%s", uid, qdrant_summary["collections_purged"])
        else:
            qdrant_summary["status"] = "skipped: not_configured"
    except Exception as exc:
        qdrant_summary["status"] = "failed"
        qdrant_summary["errors"].append(str(exc))
        logger.exception("Hard purge Qdrant failed user_id=%s: %s", uid, exc)

    summary["targets"]["qdrant_vectors"] = qdrant_summary

    # =========================================================================
    # 3. Appwrite Database Collections (Cascading Cleanup)
    # =========================================================================
    db_summary: Dict[str, Any] = {
        "status": "pending",
        "documents_deleted": 0,
        "collections": {},
        "errors": [],
    }
    try:
        from services.appwrite_proxy import AppwriteProxy

        proxy = AppwriteProxy()

        target_collections = [
            "chat_threads",
            "chat_messages",
            "outfits",
            "wardrobe_style_metadata",
            "style_assets",
            "saved_boards",
            "style_board_states",
            "wear_events",
            "calendar_events",
            "plans",
            "skincare",
            "skincare_profiles",
            "skincare_logs",
            "workout_outfits",
            "bills",
            "coupons",
            "meds",
            "med_logs",
            "meal_plans",
            "life_goals",
            "life_boards",
            "contacts",
            "memories",
            "jobs",
            "notification_devices",
            "notification_reminders",
            "notification_preferences",
        ]

        total_db_deleted = 0
        for res in target_collections:
            del_in_res = 0
            try:
                while True:
                    docs_resp = proxy.list_documents(res, user_id=uid, limit=100)
                    docs = (
                        docs_resp.get("documents")
                        if isinstance(docs_resp, dict)
                        else (docs_resp if isinstance(docs_resp, list) else [])
                    )
                    if not docs:
                        break
                    for doc in docs:
                        doc_id = str((doc or {}).get("$id") or (doc or {}).get("id") or "").strip()
                        if doc_id:
                            try:
                                proxy.delete_document(res, doc_id)
                                del_in_res += 1
                            except Exception as de:
                                logger.warning("Failed to delete doc %s in %s: %s", doc_id, res, de)
                    if len(docs) < 100:
                        break
                if del_in_res > 0:
                    db_summary["collections"][res] = del_in_res
                    total_db_deleted += del_in_res
            except Exception as ce:
                db_summary["errors"].append(f"{res}: {ce}")
                logger.warning("Hard purge DB list/delete error res=%s user_id=%s: %s", res, uid, ce)

        # Remove the user document itself from "users"
        try:
            proxy.delete_document("users", uid)
            db_summary["collections"]["users"] = 1
            total_db_deleted += 1
        except Exception as ue:
            logger.info("Profile document delete for %s: %s", uid, ue)

        db_summary["documents_deleted"] = total_db_deleted
        db_summary["status"] = "completed" if not db_summary["errors"] else "partial"
        logger.info("Hard purge Appwrite DB completed user_id=%s docs_deleted=%d", uid, total_db_deleted)
    except Exception as exc:
        db_summary["status"] = "failed"
        db_summary["errors"].append(str(exc))
        logger.exception("Hard purge Appwrite DB failed user_id=%s: %s", uid, exc)

    summary["targets"]["appwrite_database"] = db_summary

    # =========================================================================
    # 4. Appwrite Auth User Identity
    # =========================================================================
    auth_summary: Dict[str, Any] = {
        "status": "pending",
        "errors": [],
    }
    try:
        from services.appwrite_service import get_admin_client, is_appwrite_configured

        if is_appwrite_configured():
            from appwrite.services.users import Users

            users_svc = Users(get_admin_client())
            try:
                users_svc.delete(uid)
                auth_summary["status"] = "deleted"
                logger.info("Hard purge Appwrite Auth deleted user_id=%s", uid)
            except Exception as ae:
                err_msg = str(ae)
                if "404" in err_msg or "user_not_found" in err_msg.lower():
                    auth_summary["status"] = "already_deleted"
                else:
                    auth_summary["status"] = "failed"
                    auth_summary["errors"].append(err_msg)
                    logger.warning("Hard purge Appwrite Auth user delete error user_id=%s: %s", uid, ae)
        else:
            auth_summary["status"] = "skipped: not_configured"
    except Exception as exc:
        auth_summary["status"] = "failed"
        auth_summary["errors"].append(str(exc))
        logger.exception("Hard purge Appwrite Auth failed user_id=%s: %s", uid, exc)

    summary["targets"]["appwrite_auth"] = auth_summary

    # Overall success if at least DB or Auth finished without fatal crash
    overall_success = (
        db_summary["status"] in {"completed", "partial"}
        or auth_summary["status"] in {"deleted", "already_deleted"}
    )
    summary["success"] = overall_success
    summary["completed_at"] = _iso_now()

    logger.info("execute_hard_purge completed user_id=%s overall_success=%s", uid, overall_success)
    return summary
