"""Wardrobe Care Rule Service.

Manages user intent CareRule entities separate from notification delivery occurrences.
Enforces double-entity ownership checks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional

from services.appwrite_proxy import AppwriteProxy
from services.wardrobe_item_service import WardrobeItemService

logger = logging.getLogger("ahvi.wardrobe_care_service")


def _utcnow() -> datetime:
    return datetime.now(dt_timezone.utc)


class WardrobeCareService:
    def __init__(self) -> None:
        self.proxy = AppwriteProxy()
        self.resource = "care_rules"
        self.item_service = WardrobeItemService()

    def create_care_rule(
        self,
        *,
        user_id: str,
        item_id: str,
        care_type: str = "wash",
        trigger_type: str = "date",
        scheduled_at: Optional[str] = None,
        repeat_every_wears: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Double-entity check: Validate wardrobe item ownership
        self.item_service.require_owned_item(user_id, item_id)

        clean_care_type = str(care_type or "wash").strip().lower()
        clean_trigger_type = str(trigger_type or "date").strip().lower()

        data = {
            "userId": user_id,
            "itemId": item_id,
            "careType": clean_care_type,
            "triggerType": clean_trigger_type,
            "scheduledAtISO": scheduled_at or _utcnow().isoformat(),
            "repeatEveryWears": repeat_every_wears,
            "status": "active",
            "createdAtISO": _utcnow().isoformat(),
            "updatedAtISO": _utcnow().isoformat(),
        }

        try:
            created = self.proxy.create_document(self.resource, data)
            doc_id = str(created.get("$id") or created.get("id") or "") if isinstance(created, dict) else ""
        except Exception:
            # Fallback inline doc_id generator
            doc_id = f"care_{user_id[:8]}_{item_id[:8]}_{clean_care_type}"
            self.proxy.create_document(self.resource, data, document_id=doc_id)

        return {"rule_id": doc_id, "user_id": user_id, "item_id": item_id, **data}

    def list_care_rules(self, *, user_id: str, item_id: str) -> List[Dict[str, Any]]:
        self.item_service.require_owned_item(user_id, item_id)
        try:
            rows = self.proxy.list_documents(self.resource, user_id=user_id, limit=100)
        except Exception:
            rows = []

        results = []
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            r_user = str(r.get("userId") or "").strip()
            r_item = str(r.get("itemId") or "").strip()
            if r_user == user_id and r_item == item_id and str(r.get("status") or "active").lower() == "active":
                results.append(
                    {
                        "rule_id": str(r.get("$id") or r.get("id") or ""),
                        "item_id": item_id,
                        "care_type": r.get("careType"),
                        "trigger_type": r.get("triggerType"),
                        "scheduled_at": r.get("scheduledAtISO"),
                        "repeat_every_wears": r.get("repeatEveryWears"),
                        "status": r.get("status"),
                    }
                )
        return results

    def patch_care_rule(
        self,
        *,
        user_id: str,
        item_id: str,
        rule_id: str,
        care_type: Optional[str] = None,
        scheduled_at: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.item_service.require_owned_item(user_id, item_id)
        self._get_owned_rule(user_id, item_id, rule_id)

        patch: Dict[str, Any] = {"updatedAtISO": _utcnow().isoformat()}
        if care_type:
            patch["careType"] = str(care_type).lower()
        if scheduled_at:
            patch["scheduledAtISO"] = scheduled_at
        if status:
            patch["status"] = str(status).lower()

        self.proxy.update_document(self.resource, rule_id, patch)
        return {"rule_id": rule_id, "item_id": item_id, **patch}

    def delete_care_rule(self, *, user_id: str, item_id: str, rule_id: str) -> bool:
        self.item_service.require_owned_item(user_id, item_id)
        self._get_owned_rule(user_id, item_id, rule_id)
        self.proxy.update_document(self.resource, rule_id, {"status": "inactive", "updatedAtISO": _utcnow().isoformat()})
        return True

    def _get_owned_rule(self, user_id: str, item_id: str, rule_id: str) -> Dict[str, Any]:
        doc = self.proxy.get_document(self.resource, rule_id)
        if not isinstance(doc, dict):
            raise KeyError("Care rule not found")
        r_user = str(doc.get("userId") or "").strip()
        r_item = str(doc.get("itemId") or "").strip()
        if r_user != user_id or r_item != item_id:
            raise KeyError("Care rule not found")
        return doc
