"""Opportunity Engine for AHVI Temporal Intelligence.

Consumes TemporalSignals and TimelineItems, generates opportunities with deterministic
idempotency keys, persists them to OpportunityStore, and notifies subscribers via OpportunityNotifier.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from brain.temporal.models import TimelineItem
from brain.temporal.opportunity_models import Opportunity, OpportunityStatus, generate_idempotency_key
from brain.temporal.opportunity_notifier import opportunity_notifier
from brain.temporal.opportunity_store import opportunity_store
from brain.temporal.signals import TemporalSignal, TemporalSignalType

logger = logging.getLogger("ahvi.temporal.opportunity_engine")


class OpportunityEngine:
    """Deterministic opportunity generator with idempotency and notifier dispatch."""

    def evaluate_signal(self, signal: TemporalSignal) -> Optional[Opportunity]:
        """Evaluate a single TemporalSignal and create an opportunity if eligible."""
        trigger_window = f"win_{int(signal.timestamp.timestamp() // 3600)}"
        opportunity_type = f"opp_{signal.signal_type.value.lower()}"

        idempotency_key = generate_idempotency_key(
            user_id=signal.user_id,
            opportunity_type=opportunity_type,
            timeline_item_id=signal.timeline_item_id,
            trigger_window=trigger_window,
        )

        if opportunity_store.has_idempotency_key(idempotency_key):
            logger.debug(
                "AHVI_OPPORTUNITY_ENGINE_SKIP_DUPLICATE key=%s user_id=%s",
                idempotency_key,
                signal.user_id,
            )
            return None

        expires_at = signal.timestamp + timedelta(hours=6)
        opp = Opportunity.create(
            user_id=signal.user_id,
            opportunity_type=opportunity_type,
            timeline_item_id=signal.timeline_item_id,
            trigger_window=trigger_window,
            expires_at=expires_at,
            payload={"signal_id": signal.id, "signal_type": signal.signal_type.value, **signal.metadata},
        )
        opp.status = OpportunityStatus.AVAILABLE

        # Persist to OpportunityStore
        opportunity_store.save_opportunity(opp)

        # Notify consumers (Push to Wake, Pull to Consume)
        opportunity_notifier.notify_opportunity_available(opp)

        return opp

    def process_signals(self, signals: List[TemporalSignal]) -> List[Opportunity]:
        """Evaluate a batch of signals and return generated opportunities."""
        created: List[Opportunity] = []
        for sig in signals:
            opp = self.evaluate_signal(sig)
            if opp:
                created.append(opp)
        return created


# Global singleton
opportunity_engine = OpportunityEngine()
