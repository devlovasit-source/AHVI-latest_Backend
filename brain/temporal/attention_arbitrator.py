"""Attention Arbitrator Layer for AHVI Temporal Intelligence.

Ranks, suppresses, merges, batches, or defers competing CandidateAction items
so individual module brains do not independently control user interruption.
Uses deliver_after and expires_at fields for batch/deferral behavior with durable persistence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from brain.temporal.attention_models import CandidateAction
from brain.temporal.candidate_action_store import candidate_action_store

logger = logging.getLogger("ahvi.temporal.attention_arbitrator")


class AttentionArbitrator:
    """Arbitrator layer managing candidate action ranking, suppression, and durable deferral."""

    def filter_deliverable(self, actions: List[CandidateAction]) -> List[CandidateAction]:
        """Filter out candidate actions that are expired or not yet deliverable."""
        return [a for a in actions if isinstance(a, CandidateAction) and a.is_deliverable]

    def suppress_duplicates(self, actions: List[CandidateAction]) -> List[CandidateAction]:
        """Suppress duplicate candidate actions with matching source_module and action_type."""
        seen_keys: set[str] = set()
        kept: List[CandidateAction] = []

        for action in actions:
            if not isinstance(action, CandidateAction):
                continue
            key = f"{action.user_id}:{action.source_module}:{action.action_type}"
            if key in seen_keys:
                logger.info(
                    "AHVI_ATTENTION_ACTION_SUPPRESSED user_id=%s key=%s action_id=%s",
                    action.user_id,
                    key,
                    action.id,
                )
                continue
            seen_keys.add(key)
            kept.append(action)

        return kept

    def rank(self, actions: List[CandidateAction]) -> List[CandidateAction]:
        """Rank candidate actions by composite score (priority, urgency, attention cost)."""
        return sorted(actions, key=lambda a: a.composite_score, reverse=True)

    def defer_action(self, action: CandidateAction, defer_duration: timedelta) -> CandidateAction:
        """Defer a candidate action by setting deliver_after and persisting to durable store."""
        now = datetime.now(timezone.utc)
        action.deliver_after = now + defer_duration
        candidate_action_store.save_action(action)
        logger.info(
            "AHVI_ATTENTION_ACTION_DEFERRED action_id=%s deliver_after=%s",
            action.id,
            action.deliver_after.isoformat(),
        )
        return action

    def arbitrate(
        self,
        actions: List[CandidateAction],
        max_delivery: int = 3,
    ) -> List[CandidateAction]:
        """Full arbitration pipeline: filter deliverable, suppress duplicates, rank, and cap."""
        if not actions:
            return []

        # Save all incoming candidate actions to durable store
        for act in actions:
            candidate_action_store.save_action(act)

        deliverable = self.filter_deliverable(actions)
        suppressed = self.suppress_duplicates(deliverable)
        ranked = self.rank(suppressed)

        # Defer any excess actions beyond max_delivery budget
        to_deliver = ranked[:max_delivery]
        to_defer = ranked[max_delivery:]

        for deferred in to_defer:
            self.defer_action(deferred, timedelta(minutes=30))

        return to_deliver

    def arbitrate_actions(
        self,
        actions: List[CandidateAction],
        max_delivery: int = 3,
    ) -> List[CandidateAction]:
        """Alias for arbitrate method."""
        return self.arbitrate(actions, max_delivery=max_delivery)

    def scan_and_deliver_due_deferred_actions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Scan stored candidate actions for user_id where deliver_after <= now and route them through DeliveryRouter.
        Ensures deferred actions are actively re-evaluated and delivered upon background sweep or restart.
        """
        from brain.temporal.delivery_router import delivery_router

        uid = str(user_id or "").strip()
        if not uid:
            return []

        user_actions = candidate_action_store.query_user_actions(uid)
        delivered_outputs: List[Dict[str, Any]] = []

        for action in user_actions:
            if action.status == "PENDING" and action.deliver_after is not None and action.is_deliverable:
                routed = delivery_router.route_candidate_action(action)
                action.status = "DELIVERED"
                action.deliver_after = None
                candidate_action_store.save_action(action)
                delivered_outputs.append(routed)
                logger.info(
                    "AHVI_ATTENTION_DEFERRED_ACTION_REDELIVERED action_id=%s user_id=%s channel=%s status=%s",
                    action.id,
                    uid,
                    routed.get("channel"),
                    action.status,
                )



        return delivered_outputs



# Global singleton
attention_arbitrator = AttentionArbitrator()
