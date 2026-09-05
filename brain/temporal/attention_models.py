"""Attention Models and CandidateAction Schema for AHVI Temporal Intelligence."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class DeliveryChannel(str, Enum):
    """Channels through which candidate actions are delivered to the user."""

    SYSTEM_REMINDER = "SYSTEM_REMINDER"
    MODULE_REMINDER = "MODULE_REMINDER"
    AHVI_OPPORTUNITY = "AHVI_OPPORTUNITY"


class CandidateAction(BaseModel):
    """CandidateAction model representing a module intervention requesting user attention.

    Complete field set from master spec:
    id, user_id, source_opportunity_id, source_module, action_type, priority,
    urgency, attention_cost, deliver_after, expires_at, payload.
    """

    id: str = Field(..., description="Unique action candidate ID")
    user_id: str = Field(..., description="Target user ID")
    source_opportunity_id: str = Field(..., description="Originating opportunity ID")
    source_module: str = Field(..., description="Source module generating this action")
    action_type: str = Field(..., description="Specific action type identifier")
    priority: int = Field(default=3, ge=1, le=5, description="Importance priority (1-5)")
    urgency: float = Field(default=0.5, ge=0.0, le=1.0, description="Time urgency rating (0.0 to 1.0)")
    attention_cost: float = Field(default=0.5, ge=0.0, le=1.0, description="Estimated cognitive/interruption cost (0.0 to 1.0)")
    deliver_after: Optional[datetime] = Field(default=None, description="Earliest delivery timestamp for batching/deferral")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp after which action is void")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Action payload for UI rendering or execution")

    @property
    def is_expired(self) -> bool:
        """Check if candidate action is past its expiration timestamp."""
        if self.expires_at:
            now = datetime.now(timezone.utc)
            exp = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=timezone.utc)
            return now >= exp
        return False

    @property
    def is_deliverable(self) -> bool:
        """Check if candidate action has reached its deliver_after timestamp and is not expired."""
        if self.is_expired:
            return False
        if self.deliver_after:
            now = datetime.now(timezone.utc)
            after = self.deliver_after if self.deliver_after.tzinfo else self.deliver_after.replace(tzinfo=timezone.utc)
            return now >= after
        return True

    @property
    def composite_score(self) -> float:
        """Compute ranking score balancing priority, urgency, and attention cost."""
        return (self.priority * 2.0) + (self.urgency * 3.0) - (self.attention_cost * 1.5)
