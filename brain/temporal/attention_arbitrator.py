"""Attention Arbitrator Layer for AHVI Temporal Intelligence.

Ranks, suppresses, merges, batches, or defers competing CandidateAction items
so individual module brains do not independently control user interruption.
Uses deliver_after and expires_at fields for batch/deferral behavior.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from brain.temporal.attention_models import CandidateAction

logger = logging.getLogger("ahvi.temporal.attention_arbitrator")


class AttentionArbitrator:
    """Arbitrator layer managing candidate action ranking, suppression, and deferral."""

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
        """Defer a candidate action by setting its deliver_after timestamp."""
        now = datetime.now(timezone.utc)
        action.deliver_after = now + defer_duration
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



# Global singleton
attention_arbitrator = AttentionArbitrator()
