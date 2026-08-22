"""Wardrobe Persistence Service.

Provides decoupled persistence operations for saving processed wardrobe items
to Appwrite database. ZERO synthetic fake wardrobe IDs in any environment.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.wardrobe_persistence")


class WardrobePersistenceService:
    def __init__(self) -> None:
        self.proxy = AppwriteProxy()
        self.collection_name = "outfits"

    def save_wardrobe_item(
        self,
        *,
        user_id: str,
        upload_item_id: str,
        name: str = "Wardrobe Item",
        category: str = "Tops",
        masked_image_base64: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a single processed garment item to wardrobe storage.
        
        Returns a dictionary containing the saved wardrobe document ID:
        {"success": True, "wardrobe_item_id": "outfit_doc_id", "items": [{"id": "outfit_doc_id"}]}
        
        ZERO synthetic fake wardrobe IDs in any environment. If database is unreachable
        or unconfigured, strictly raises RuntimeError("PERSISTENCE_FAILED").
        """
        if not user_id:
            raise ValueError("user_id is required for wardrobe persistence")

        meta = metadata or {}
        now_ts = int(time.time() * 1000)
        doc_data = {
            "userId": user_id,
            "name": name,
            "category": category,
            "masked_image_base64": masked_image_base64,
            "upload_item_id": upload_item_id,
            "created_at": now_ts,
            "source": meta.get("source") or "multi_garment_upload",
        }

        # Must persist to real Appwrite database. Synthetic fake IDs are strictly prohibited!
        if self.proxy.endpoint and self.proxy.project_id:
            try:
                res_doc = self.proxy.create_document(self.collection_name, doc_data)
                wardrobe_id = str(res_doc.get("$id") or res_doc.get("id") or "")
                if not wardrobe_id:
                    raise RuntimeError("Appwrite returned empty document ID for outfit document")
                return {
                    "success": True,
                    "wardrobe_item_id": wardrobe_id,
                    "items": [{"id": wardrobe_id, "name": name, "category": category}],
                }
            except Exception as exc:
                logger.error("ahvi.wardrobe_persistence.failed user_id=%s item_id=%s err=%s", user_id, upload_item_id, exc)
                raise RuntimeError(f"PERSISTENCE_FAILED: {exc}") from exc

        # Unconfigured/unreachable database strictly fails. NO synthetic fake IDs!
        logger.error("ahvi.wardrobe_persistence.unreachable user_id=%s item_id=%s", user_id, upload_item_id)
        raise RuntimeError("PERSISTENCE_FAILED: Appwrite database backend is unconfigured or unreachable")
