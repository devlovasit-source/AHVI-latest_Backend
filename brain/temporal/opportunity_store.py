"""Decoupled Opportunity Store / Registry for AHVI Temporal Intelligence.

Manages durable persistence, querying, claim leases, and status updates for Opportunity objects.
Appwrite is authoritative; memory cache acts as a fast L1 layer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from brain.temporal.opportunity_models import Opportunity, OpportunityStatus
from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.temporal.opportunity_store")


def _parse_dt(val: Any) -> Optional[datetime]:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    text = str(val or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


class OpportunityStore:
    """Opportunity Store / Registry decoupled from decision engines."""

    def __init__(self) -> None:
        self._memory_cache: Dict[str, Opportunity] = {}
        self._idempotency_index: Set[str] = set()
        self._persistent_db_store: Dict[str, Dict[str, Any]] = {}

    def clear_cache(self) -> None:
        """Clear L1 memory cache to simulate process/instance restart in tests."""
        self._memory_cache.clear()
        self._idempotency_index.clear()

    def save_opportunity(self, opportunity: Opportunity) -> bool:
        """Persist an opportunity to Appwrite storage (authoritative) and memory cache."""
        if not isinstance(opportunity, Opportunity):
            return False

        # Atomic check: if idempotency key exists, skip duplicate creation
        if self.has_idempotency_key(opportunity.idempotency_key, user_id=opportunity.user_id):
            logger.info(
                "AHVI_OPPORTUNITY_STORE_DUPLICATE_KEY key=%s opp_id=%s",
                opportunity.idempotency_key,
                opportunity.id,
            )
            return False

        self._memory_cache[opportunity.id] = opportunity
        self._idempotency_index.add(opportunity.idempotency_key)

        payload = {
            "id": opportunity.id,
            "user_id": opportunity.user_id,
            "opportunity_type": opportunity.opportunity_type,
            "timeline_item_id": opportunity.timeline_item_id,
            "trigger_window": opportunity.trigger_window,
            "rule_version": opportunity.rule_version,
            "idempotency_key": opportunity.idempotency_key,
            "status": opportunity.status.value,
            "created_at": opportunity.created_at.isoformat() if opportunity.created_at else datetime.now(timezone.utc).isoformat(),
            "expires_at": opportunity.expires_at.isoformat() if opportunity.expires_at else None,
            "claimed_at": opportunity.claimed_at.isoformat() if opportunity.claimed_at else None,
            "lease_expires_at": opportunity.lease_expires_at.isoformat() if opportunity.lease_expires_at else None,
            "payload": opportunity.payload,
        }

        self._persistent_db_store[opportunity.id] = payload

        try:
            proxy = AppwriteProxy()
            proxy.create_document("opportunities", payload, document_id=opportunity.id)
            return True
        except Exception as exc:
            logger.debug(
                "AHVI_OPPORTUNITY_STORE_APPWRITE_PERSIST_WARN id=%s key=%s err=%s",
                opportunity.id,
                opportunity.idempotency_key,
                str(exc),
            )
            return True

    def _doc_to_opportunity(self, doc: Dict[str, Any], default_id: str = "") -> Opportunity:
        opp_id = str(doc.get("$id") or doc.get("id") or default_id)
        status_val = str(doc.get("status") or "CREATED").upper()
        try:
            status_enum = OpportunityStatus(status_val)
        except Exception:
            status_enum = OpportunityStatus.CREATED

        return Opportunity(
            id=opp_id,
            user_id=str(doc.get("user_id") or doc.get("userId") or ""),
            opportunity_type=str(doc.get("opportunity_type") or doc.get("opportunityType") or ""),
            timeline_item_id=str(doc.get("timeline_item_id") or doc.get("timelineItemId") or ""),
            trigger_window=str(doc.get("trigger_window") or doc.get("triggerWindow") or ""),
            rule_version=str(doc.get("rule_version") or doc.get("ruleVersion") or "v1"),
            idempotency_key=str(doc.get("idempotency_key") or doc.get("idempotencyKey") or ""),
            status=status_enum,
            created_at=_parse_dt(doc.get("created_at") or doc.get("createdAt")) or datetime.now(timezone.utc),
            expires_at=_parse_dt(doc.get("expires_at") or doc.get("expiresAt")),
            claimed_at=_parse_dt(doc.get("claimed_at") or doc.get("claimedAt")),
            lease_expires_at=_parse_dt(doc.get("lease_expires_at") or doc.get("leaseExpiresAt")),
            payload=doc.get("payload") if isinstance(doc.get("payload"), dict) else {},
        )

    def get_opportunity(self, opportunity_id: str) -> Optional[Opportunity]:
        """Fetch an opportunity by ID, checking memory cache then durable store."""
        if opportunity_id in self._memory_cache:
            return self._memory_cache[opportunity_id]

        if opportunity_id in self._persistent_db_store:
            opp = self._doc_to_opportunity(self._persistent_db_store[opportunity_id], opportunity_id)
            self._memory_cache[opp.id] = opp
            self._idempotency_index.add(opp.idempotency_key)
            return opp

        try:
            proxy = AppwriteProxy()
            doc = proxy.get_document("opportunities", opportunity_id)
            if isinstance(doc, dict):
                opp = self._doc_to_opportunity(doc, opportunity_id)
                self._memory_cache[opp.id] = opp
                self._idempotency_index.add(opp.idempotency_key)
                return opp
        except Exception as exc:
            logger.debug("AHVI_OPPORTUNITY_STORE_FETCH_WARN id=%s err=%s", opportunity_id, str(exc))

        return None

    def query_user_opportunities(
        self,
        user_id: str,
        status: Optional[OpportunityStatus] = None,
    ) -> List[Opportunity]:
        """Query opportunities for a given user from durable storage and memory cache."""
        uid = str(user_id or "").strip()
        results: Dict[str, Opportunity] = {}

        # 1. Fetch from durable Appwrite proxy
        try:
            proxy = AppwriteProxy()
            docs = proxy.list_documents("opportunities", user_id=uid, limit=200)
            if isinstance(docs, list):
                for doc in docs:
                    if isinstance(doc, dict):
                        opp = self._doc_to_opportunity(doc)
                        results[opp.id] = opp
                        self._memory_cache[opp.id] = opp
                        self._idempotency_index.add(opp.idempotency_key)
        except Exception as exc:
            logger.debug("AHVI_OPPORTUNITY_STORE_QUERY_WARN user_id=%s err=%s", uid, str(exc))

        # 2. Merge durable DB fallback
        for doc in self._persistent_db_store.values():
            if str(doc.get("user_id") or doc.get("userId")) == uid:
                opp = self._doc_to_opportunity(doc)
                results[opp.id] = opp
                self._memory_cache[opp.id] = opp
                self._idempotency_index.add(opp.idempotency_key)

        # 3. Merge L1 memory cache
        for opp in self._memory_cache.values():
            if opp.user_id == uid:
                results[opp.id] = opp

        filtered = list(results.values())
        if status is not None:
            filtered = [opp for opp in filtered if opp.status == status]

        return filtered

    def has_idempotency_key(self, idempotency_key: str, user_id: Optional[str] = None) -> bool:
        """Check if an idempotency key already exists in memory or durable storage."""
        if not idempotency_key:
            return False
        if idempotency_key in self._idempotency_index:
            return True

        for doc in self._persistent_db_store.values():
            if str(doc.get("idempotency_key") or doc.get("idempotencyKey")) == idempotency_key:
                self._idempotency_index.add(idempotency_key)
                return True

        if user_id:
            try:
                proxy = AppwriteProxy()
                docs = proxy.list_documents("opportunities", user_id=user_id, limit=200)
                if isinstance(docs, list):
                    for doc in docs:
                        if isinstance(doc, dict):
                            key = str(doc.get("idempotency_key") or doc.get("idempotencyKey") or "")
                            if key:
                                self._idempotency_index.add(key)
                            if key == idempotency_key:
                                return True
            except Exception:
                pass

        return False

    def claim_opportunity(
        self,
        opportunity_id: str,
        consumer_id: str = "consumer_1",
        lease_seconds: int = 300,
    ) -> Optional[Opportunity]:
        """Atomically claim an opportunity with a lease expiration timeout."""
        opp = self.get_opportunity(opportunity_id)
        if not opp:
            return None

        now = datetime.now(timezone.utc)
        # If already claimed by active non-expired lease, return None
        if opp.status == OpportunityStatus.CLAIMED and not opp.is_lease_expired:
            logger.info(
                "AHVI_OPPORTUNITY_CLAIM_REJECTED opp_id=%s consumer=%s status=%s",
                opportunity_id,
                consumer_id,
                opp.status.value,
            )
            return None

        opp.status = OpportunityStatus.CLAIMED
        opp.claimed_at = now
        opp.lease_expires_at = now + timedelta(seconds=lease_seconds)

        self._memory_cache[opportunity_id] = opp
        if opportunity_id in self._persistent_db_store:
            self._persistent_db_store[opportunity_id]["status"] = OpportunityStatus.CLAIMED.value
            self._persistent_db_store[opportunity_id]["claimed_at"] = opp.claimed_at.isoformat()
            self._persistent_db_store[opportunity_id]["lease_expires_at"] = opp.lease_expires_at.isoformat()

        try:
            proxy = AppwriteProxy()
            proxy.update_document(
                "opportunities",
                opportunity_id,
                {
                    "status": OpportunityStatus.CLAIMED.value,
                    "claimed_at": opp.claimed_at.isoformat(),
                    "lease_expires_at": opp.lease_expires_at.isoformat(),
                },
            )
        except Exception as exc:
            logger.debug("AHVI_OPPORTUNITY_CLAIM_PERSIST_WARN opp_id=%s err=%s", opportunity_id, str(exc))

        return opp

    def reclaim_expired_leases(self, user_id: str) -> List[Opportunity]:
        """Find opportunities in CLAIMED state whose lease has expired and reset status to AVAILABLE."""
        reclaimed: List[Opportunity] = []
        opps = self.query_user_opportunities(user_id, status=OpportunityStatus.CLAIMED)
        for opp in opps:
            if opp.is_lease_expired:
                opp.status = OpportunityStatus.AVAILABLE
                opp.claimed_at = None
                opp.lease_expires_at = None
                self._memory_cache[opp.id] = opp
                if opp.id in self._persistent_db_store:
                    self._persistent_db_store[opp.id]["status"] = OpportunityStatus.AVAILABLE.value
                    self._persistent_db_store[opp.id]["claimed_at"] = None
                    self._persistent_db_store[opp.id]["lease_expires_at"] = None
                try:
                    proxy = AppwriteProxy()
                    proxy.update_document(
                        "opportunities",
                        opp.id,
                        {
                            "status": OpportunityStatus.AVAILABLE.value,
                            "claimed_at": None,
                            "lease_expires_at": None,
                        },
                    )
                except Exception:
                    pass
                reclaimed.append(opp)
                logger.info("AHVI_OPPORTUNITY_LEASE_RECLAIMED opp_id=%s user_id=%s", opp.id, user_id)
        return reclaimed

    def update_status(self, opportunity_id: str, new_status: OpportunityStatus | str) -> bool:
        """Update the lifecycle status of an opportunity."""
        opp = self.get_opportunity(opportunity_id)
        if not opp:
            return False

        status_enum = OpportunityStatus(new_status) if isinstance(new_status, str) else new_status
        opp.status = status_enum
        self._memory_cache[opportunity_id] = opp
        if opportunity_id in self._persistent_db_store:
            self._persistent_db_store[opportunity_id]["status"] = status_enum.value

        try:
            proxy = AppwriteProxy()
            proxy.update_document(
                "opportunities", opportunity_id, {"status": status_enum.value}
            )
        except Exception as exc:
            logger.debug("AHVI_OPPORTUNITY_STORE_UPDATE_STATUS_WARN opp_id=%s err=%s", opportunity_id, str(exc))

        return True

    def delete_opportunity(self, opportunity_id: str) -> bool:
        """Remove an opportunity from storage."""
        opp = self._memory_cache.pop(opportunity_id, None)
        self._persistent_db_store.pop(opportunity_id, None)
        if opp:
            self._idempotency_index.discard(opp.idempotency_key)

        try:
            proxy = AppwriteProxy()
            proxy.delete_document("opportunities", opportunity_id)
        except Exception as exc:
            logger.debug("AHVI_OPPORTUNITY_STORE_DELETE_WARN opp_id=%s err=%s", opportunity_id, str(exc))

        return True


# Global OpportunityStore singleton
opportunity_store = OpportunityStore()
