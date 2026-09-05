"""Opportunity Notifier for AHVI Temporal Intelligence.

Implements the "Push to Wake, Pull to Consume" event pattern, notifying subscribed
module brains and downstream consumers when new opportunities become available.
Supports durable crash recovery and replay of unconsumed opportunities from OpportunityStore.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional
from brain.temporal.opportunity_models import Opportunity, OpportunityStatus
from brain.temporal.opportunity_store import opportunity_store

logger = logging.getLogger("ahvi.temporal.opportunity_notifier")

OpportunityHandler = Callable[[Opportunity], None]


class OpportunityNotifier:
    """Event publisher notifying subscribers of available opportunities with crash recovery."""

    def __init__(self) -> None:
        self._subscribers: List[OpportunityHandler] = []

    def subscribe(self, handler: OpportunityHandler) -> None:
        """Register a callback handler for opportunity notifications."""
        if handler not in self._subscribers:
            self._subscribers.append(handler)

    def unsubscribe(self, handler: OpportunityHandler) -> None:
        """Remove a registered callback handler."""
        if handler in self._subscribers:
            self._subscribers.remove(handler)

    def notify_opportunity_available(self, opportunity: Opportunity) -> int:
        """Dispatch notification to all registered subscribers."""
        notified_count = 0
        for handler in list(self._subscribers):
            try:
                handler(opportunity)
                notified_count += 1
            except Exception as exc:
                logger.error(
                    "AHVI_OPPORTUNITY_NOTIFIER_HANDLER_FAILED opp_id=%s err=%s",
                    opportunity.id,
                    str(exc),
                    exc_info=True,
                )

        logger.info(
            "AHVI_OPPORTUNITY_NOTIFIER_DISPATCHED opp_id=%s subscriber_count=%d notified=%d",
            opportunity.id,
            len(self._subscribers),
            notified_count,
        )
        return notified_count

    def recover_and_replay_unclaimed(self, user_id: str) -> int:
        """Crash recovery: Re-read AVAILABLE opportunities from durable store and notify consumers."""
        unclaimed = opportunity_store.query_user_opportunities(user_id, status=OpportunityStatus.AVAILABLE)
        replayed_count = 0

        for opp in unclaimed:
            if not opp.is_lease_expired and opp.status == OpportunityStatus.AVAILABLE:
                self.notify_opportunity_available(opp)
                replayed_count += 1

        logger.info(
            "AHVI_OPPORTUNITY_NOTIFIER_RECOVERY_COMPLETE user_id=%s replayed=%d",
            user_id,
            replayed_count,
        )
        return replayed_count


# Global singleton
opportunity_notifier = OpportunityNotifier()
