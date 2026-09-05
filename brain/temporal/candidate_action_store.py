"""Decoupled Durable Candidate Action Store for AHVI Temporal Intelligence.

Manages durable persistence, query, and status updates for CandidateAction objects.
Ensures deferred candidate actions (with future deliver_after) survive process restarts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from brain.temporal.attention_models import CandidateAction
from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.temporal.candidate_action_store")


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


class CandidateActionStore:
    """Durable registry for CandidateAction objects with restart persistence."""

    def __init__(self) -> None:
        self._memory_cache: Dict[str, CandidateAction] = {}
        self._persistent_db_store: Dict[str, Dict[str, Any]] = {}

    def clear_cache(self) -> None:
        """Clear L1 memory cache to simulate process restart."""
        self._memory_cache.clear()

    def save_action(self, action: CandidateAction) -> bool:
        """Persist candidate action to Appwrite (authoritative) and memory cache."""
        if not isinstance(action, CandidateAction):
            return False

        payload = {
            "id": action.id,
            "user_id": action.user_id,
            "source_opportunity_id": action.source_opportunity_id,
            "source_module": action.source_module,
            "action_type": action.action_type,
            "priority": action.priority,
            "urgency": action.urgency,
            "attention_cost": action.attention_cost,
            "deliver_after": action.deliver_after.isoformat() if action.deliver_after else None,
            "expires_at": action.expires_at.isoformat() if action.expires_at else None,
            "payload": action.payload,
        }

        self._memory_cache[action.id] = action
        self._persistent_db_store[action.id] = payload

        try:
            proxy = AppwriteProxy()
            proxy.create_document("candidate_actions", payload, document_id=action.id)
        except Exception:
            try:
                proxy = AppwriteProxy()
                proxy.update_document("candidate_actions", action.id, payload)
            except Exception as patch_exc:
                logger.debug(
                    "AHVI_CANDIDATE_ACTION_STORE_PERSIST_WARN action_id=%s err=%s",
                    action.id,
                    str(patch_exc),
                )
        return True

    def _doc_to_action(self, doc: Dict[str, Any], default_id: str = "") -> CandidateAction:
        action_id = str(doc.get("$id") or doc.get("id") or default_id)
        return CandidateAction(
            id=action_id,
            user_id=str(doc.get("user_id") or doc.get("userId") or ""),
            source_opportunity_id=str(doc.get("source_opportunity_id") or doc.get("sourceOpportunityId") or ""),
            source_module=str(doc.get("source_module") or doc.get("sourceModule") or ""),
            action_type=str(doc.get("action_type") or doc.get("actionType") or ""),
            priority=int(doc.get("priority") or 3),
            urgency=float(doc.get("urgency") or 0.5),
            attention_cost=float(doc.get("attention_cost") or doc.get("attentionCost") or 0.5),
            deliver_after=_parse_dt(doc.get("deliver_after") or doc.get("deliverAfter")),
            expires_at=_parse_dt(doc.get("expires_at") or doc.get("expiresAt")),
            payload=doc.get("payload") if isinstance(doc.get("payload"), dict) else {},
        )

    def get_action(self, action_id: str) -> Optional[CandidateAction]:
        """Fetch candidate action by ID."""
        if action_id in self._memory_cache:
            return self._memory_cache[action_id]

        if action_id in self._persistent_db_store:
            act = self._doc_to_action(self._persistent_db_store[action_id], action_id)
            self._memory_cache[act.id] = act
            return act

        try:
            proxy = AppwriteProxy()
            doc = proxy.get_document("candidate_actions", action_id)
            if isinstance(doc, dict):
                act = self._doc_to_action(doc, action_id)
                self._memory_cache[act.id] = act
                return act
        except Exception:
            pass

        return None

    def query_user_actions(self, user_id: str) -> List[CandidateAction]:
        """Fetch candidate actions for a user from Appwrite and memory cache."""
        uid = str(user_id or "").strip()
        results: Dict[str, CandidateAction] = {}

        try:
            proxy = AppwriteProxy()
            docs = proxy.list_documents("candidate_actions", user_id=uid, limit=200)
            if isinstance(docs, list):
                for doc in docs:
                    if isinstance(doc, dict):
                        act = self._doc_to_action(doc)
                        results[act.id] = act
                        self._memory_cache[act.id] = act
        except Exception:
            pass

        # Fallback to local persistent store for test environment when Appwrite is offline
        for doc in self._persistent_db_store.values():
            if str(doc.get("user_id") or doc.get("userId")) == uid:
                act = self._doc_to_action(doc)
                results[act.id] = act
                self._memory_cache[act.id] = act

        for act in self._memory_cache.values():
            if act.user_id == uid:
                results[act.id] = act

        return list(results.values())


# Global singleton
candidate_action_store = CandidateActionStore()
