"""Delivery Router for AHVI Temporal Intelligence.

Categorizes and routes arbitrated CandidateAction outputs into channel deliveries:
SYSTEM_REMINDER, MODULE_REMINDER, AHVI_OPPORTUNITY.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from brain.temporal.attention_models import CandidateAction, DeliveryChannel

logger = logging.getLogger("ahvi.temporal.delivery_router")


class DeliveryRouter:
    """Routes candidate actions to target surfaces based on delivery channel."""

    def categorize_channel(self, action: CandidateAction) -> DeliveryChannel:
        """Determine target DeliveryChannel for a CandidateAction."""
        if action.source_module in ("system", "calendar", "bill"):
            return DeliveryChannel.SYSTEM_REMINDER
        if action.source_module in ("workout", "meal", "medication", "skincare"):
            return DeliveryChannel.MODULE_REMINDER
        return DeliveryChannel.AHVI_OPPORTUNITY

    def route_action(self, action: CandidateAction) -> Dict[str, Any]:
        """Route a single arbitrated action to its delivery output payload."""
        channel = self.categorize_channel(action)
        delivery_payload = {
            "action_id": action.id,
            "user_id": action.user_id,
            "channel": channel.value,
            "source_module": action.source_module,
            "action_type": action.action_type,
            "priority": action.priority,
            "payload": action.payload,
            "target_surface": (
                "home_ui" if channel == DeliveryChannel.SYSTEM_REMINDER
                else ("module_dashboard" if channel == DeliveryChannel.MODULE_REMINDER else "chat_or_push")
            ),
        }
        logger.info(
            "AHVI_DELIVERY_ROUTED action_id=%s channel=%s surface=%s",
            action.id,
            channel.value,
            delivery_payload["target_surface"],
        )
        return delivery_payload

    def route_batch(self, actions: List[CandidateAction]) -> List[Dict[str, Any]]:
        """Route a batch of arbitrated candidate actions."""
        return [self.route_action(a) for a in actions if isinstance(a, CandidateAction)]


# Global singleton
delivery_router = DeliveryRouter()
