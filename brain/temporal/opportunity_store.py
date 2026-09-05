"""Decoupled Opportunity Store / Registry for AHVI Temporal Intelligence.

Manages persistence, querying, and status updates for Opportunity objects.
Strict boundary: Persists and queries opportunities; DOES NOT decide module actions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from brain.temporal.opportunity_models import Opportunity, OpportunityStatus
from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.temporal.opportunity_store")


class OpportunityStore:
    """Opportunity Store / Registry decoupled from decision engines."""

    def __init__(self) -> None:
        self._memory_cache: Dict[str, Opportunity] = {}
        self._idempotency_index: Set[str] = set()

    def save_opportunity(self, opportunity: Opportunity) -> bool:
        """Persist an opportunity to storage and update cache/indexes."""
        if not isinstance(opportunity, Opportunity):
            return False

        self._memory_cache[opportunity.id] = opportunity
        self._idempotency_index.add(opportunity.idempotency_key)

        # Appwrite persistence attempt (fail-open to local cache)
        try:
            proxy = AppwriteProxy()
            payload = {
                "user_id": opportunity.user_id,
                "opportunity_type": opportunity.opportunity_type,
                "timeline_item_id": opportunity.timeline_item_id,
                "trigger_window": opportunity.trigger_window,
                "idempotency_key": opportunity.idempotency_key,
                "status": opportunity.status.value,
                "created_at": opportunity.created_at.isoformat(),
                "expires_at": opportunity.expires_at.isoformat() if opportunity.expires_at else None,
                "payload": opportunity.payload,
            }
            proxy.create_document("opportunities", payload, document_id=opportunity.id)
        except Exception as exc:
            logger.debug(
                "AHVI_OPPORTUNITY_STORE_APPWRITE_PERSIST_DEBUG id=%s err=%s",
                opportunity.id,
                str(exc),
            )
        return True

    def get_opportunity(self, opportunity_id: str) -> Optional[Opportunity]:
        """Fetch an opportunity by ID."""
        if opportunity_id in self._memory_cache:
            return self._memory_cache[opportunity_id]

        try:
            proxy = AppwriteProxy()
            doc = proxy.get_document("opportunities", opportunity_id)
            if isinstance(doc, dict):
                opp = Opportunity(
                    id=doc.get("$id") or doc.get("id") or opportunity_id,
                    user_id=doc.get("user_id", ""),
                    opportunity_type=doc.get("opportunity_type", ""),
                    timeline_item_id=doc.get("timeline_item_id", ""),
                    trigger_window=doc.get("trigger_window", ""),
                    idempotency_key=doc.get("idempotency_key", ""),
                    status=OpportunityStatus(doc.get("status", "CREATED")),
                    payload=doc.get("payload") or {},
                )
                self._memory_cache[opp.id] = opp
                self._idempotency_index.add(opp.idempotency_key)
                return opp
        except Exception:
            pass

        return None

    def query_user_opportunities(
        self,
        user_id: str,
        status: Optional[OpportunityStatus] = None,
    ) -> List[Opportunity]:
        """Query opportunities for a given user, optionally filtered by status."""
        matching: List[Opportunity] = []
        for opp in self._memory_cache.values():
            if opp.user_id == user_id:
                if status is None or opp.status == status:
                    matching.append(opp)
        return matching

    def has_idempotency_key(self, idempotency_key: str) -> bool:
        """Check if an idempotency key already exists in store."""
        return idempotency_key in self._idempotency_index

    def update_status(self, opportunity_id: str, new_status: OpportunityStatus | str) -> bool:
        """Update the lifecycle status of an opportunity."""
        opp = self.get_opportunity(opportunity_id)
        if not opp:
            return False

        status_enum = OpportunityStatus(new_status) if isinstance(new_status, str) else new_status
        opp.status = status_enum
        self._memory_cache[opportunity_id] = opp

        try:
            proxy = AppwriteProxy()
            proxy.update_document(
                "opportunities", opportunity_id, {"status": status_enum.value}
            )
        except Exception:
            pass

        return True

    def delete_opportunity(self, opportunity_id: str) -> bool:
        """Remove an opportunity from storage."""
        opp = self._memory_cache.pop(opportunity_id, None)
        if opp:
            self._idempotency_index.discard(opp.idempotency_key)

        try:
            proxy = AppwriteProxy()
            proxy.delete_document("opportunities", opportunity_id)
        except Exception:
            pass

        return True


# Global OpportunityStore singleton
opportunity_store = OpportunityStore()
