"""Wardrobe Item Service.

Handles explicit favorite state updates (with state-diffed preference signaling)
and resilient idempotent tombstone soft-deletion.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, Optional

from services.appwrite_proxy import AppwriteProxy
from services.qdrant_service import QdrantService
from services.r2_storage import R2Storage

logger = logging.getLogger("ahvi.wardrobe_item_service")


def _utcnow() -> datetime:
    return datetime.now(dt_timezone.utc)


class WardrobeItemService:
    def __init__(self) -> None:
        self.proxy = AppwriteProxy()
        self.collection = "outfits"  # Wardrobe items collection in Appwrite

    def require_owned_item(self, user_id: str, item_id: str, include_deleted: bool = False) -> Dict[str, Any]:
        """Fetch item and verify item.user_id == user_id. Raises ValueError/KeyError if unauthorized/not found."""
        uid = str(user_id or "").strip()
        iid = str(item_id or "").strip()
        if not uid or not iid:
            raise KeyError("Item not found")

        doc = self.proxy.get_document(self.collection, iid)
        if not isinstance(doc, dict):
            raise KeyError("Item not found")

        doc_user = str(doc.get("userId") or doc.get("user_id") or "").strip()
        if doc_user != uid:
            raise KeyError("Item not found")

        status = str(doc.get("status") or "active").lower()
        if status == "deleted" and not include_deleted:
            raise KeyError("Item not found")

        return doc

    def set_favorite(self, *, user_id: str, item_id: str, is_favorite: bool) -> Dict[str, Any]:
        """Set explicit favorite state. Compares old vs new state so retries don't duplicate affinity signals."""
        doc = self.require_owned_item(user_id, item_id)
        old_val = bool(doc.get("is_favorite") or doc.get("isLiked") or doc.get("isFavourite") or False)
        new_val = bool(is_favorite)

        patch = {
            "is_favorite": new_val,
            "isLiked": new_val,
            "isFavourite": new_val,
            "liked": new_val,
            "favoritedAtISO": _utcnow().isoformat() if new_val else None,
            "updatedAtISO": _utcnow().isoformat(),
        }

        self.proxy.update_document(self.collection, item_id, patch)

        # State-diff: emit preference signal ONLY when state actually changed
        if old_val != new_val:
            self._emit_favorite_preference(user_id, item_id, new_val)

        return {
            "item_id": item_id,
            "is_favorite": new_val,
            "updated_at": patch["updatedAtISO"],
        }

    def delete_item(self, *, user_id: str, item_id: str) -> bool:
        """Idempotent soft-delete orchestration. Sets status=deleted, cancels reminders,
        removes Qdrant vector and cleans up R2 storage. Retries return True (204)."""
        try:
            doc = self.require_owned_item(user_id, item_id, include_deleted=True)
        except KeyError:
            return True

        current_status = str(doc.get("status") or "active").lower()
        if current_status == "deleted":
            logger.info("ahvi.delete_item.already_deleted item_id=%s user_id=%s", item_id, user_id)
            return True

        # 1. Authoritative Appwrite soft-delete
        self.proxy.update_document(
            self.collection,
            item_id,
            {
                "status": "deleted",
                "deletedAtISO": _utcnow().isoformat(),
                "deletedBy": user_id,
            },
        )

        # 2. Cancel active wear/care reminders
        try:
            from services.notification_store import NotificationStore

            store = NotificationStore()
            reminders = store._appwrite.list_documents(store.reminders_resource, user_id=user_id, limit=100)
            for r in reminders if isinstance(reminders, list) else []:
                if isinstance(r, dict) and str(r.get("eventId") or r.get("entityId") or "").strip() == item_id:
                    rid = str(r.get("$id") or r.get("id") or "")
                    if rid:
                        store.mark_reminder(reminder_doc_id=rid, status="cancelled")
        except Exception as exc:
            logger.warning("ahvi.delete_item.cancel_reminders_failed item_id=%s err=%s", item_id, str(exc)[:140])

        # 3. Resilient Qdrant vector removal
        try:
            qdrant = QdrantService()
            qdrant.delete_item(item_id)
        except Exception as exc:
            logger.warning("ahvi.delete_item.qdrant_delete_failed item_id=%s err=%s", item_id, str(exc)[:140])

        # 4. Resilient R2 image cleanup
        try:
            r2 = R2Storage()
            r2.delete_wardrobe_images(raw_file_name=f"{user_id}/{item_id}.png")
        except Exception as exc:
            logger.warning("ahvi.delete_item.r2_delete_failed item_id=%s err=%s", item_id, str(exc)[:140])

        logger.info("ahvi.delete_item.completed item_id=%s user_id=%s", item_id, user_id)
        return True

    def _emit_favorite_preference(self, user_id: str, item_id: str, active: bool) -> None:
        try:
            logger.info(
                "AHVI_PREFERENCE_EMITTED user_id=%s item_id=%s signal=favorite active=%s",
                user_id, item_id, active,
            )
        except Exception as exc:
            logger.warning("ahvi.favorite_preference_failed err=%s", str(exc)[:140])
