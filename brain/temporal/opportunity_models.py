"""Opportunity Models and Idempotency Key Generation for AHVI Temporal Intelligence."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class OpportunityStatus(str, Enum):
    """Lifecycle states of an Opportunity."""

    CREATED = "CREATED"
    AVAILABLE = "AVAILABLE"
    CLAIMED = "CLAIMED"
    ACTIONED = "ACTIONED"
    DISMISSED = "DISMISSED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


def generate_idempotency_key(
    user_id: str,
    opportunity_type: str,
    timeline_item_id: str,
    trigger_window: str,
) -> str:
    """Generate a deterministic idempotency key for opportunity evaluation."""
    raw = f"{user_id}:{opportunity_type}:{timeline_item_id}:{trigger_window}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Opportunity(BaseModel):
    """Data model representing a contextual intervention opportunity."""

    id: str = Field(..., description="Unique opportunity ID")
    user_id: str = Field(..., description="Target user ID")
    opportunity_type: str = Field(..., description="Categorical type of opportunity")
    timeline_item_id: str = Field(..., description="Associated timeline item ID")
    trigger_window: str = Field(..., description="Evaluation trigger window string")
    idempotency_key: str = Field(..., description="Deterministic idempotency key")
    status: OpportunityStatus = Field(default=OpportunityStatus.CREATED, description="Current opportunity lifecycle status")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Custom payload passed to consumers")

    @classmethod
    def create(
        cls,
        user_id: str,
        opportunity_type: str,
        timeline_item_id: str,
        trigger_window: str,
        expires_at: Optional[datetime] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Opportunity:
        key = generate_idempotency_key(user_id, opportunity_type, timeline_item_id, trigger_window)
        opp_id = f"opp_{key[:16]}"
        return cls(
            id=opp_id,
            user_id=user_id,
            opportunity_type=opportunity_type,
            timeline_item_id=timeline_item_id,
            trigger_window=trigger_window,
            idempotency_key=key,
            status=OpportunityStatus.CREATED,
            expires_at=expires_at,
            payload=payload or {},
        )
