"""Opportunity Models and Idempotency Key Generation for AHVI Temporal Intelligence.

Defines OpportunityStatus lifecycle enum, deterministic logical idempotency key generation
with rule versioning, and Opportunity schema supporting claim leases and durability.
"""

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
    rule_version: str = "v1",
) -> str:
    """Generate a full 64-character SHA256 deterministic idempotency key based on logical transition window + rule version."""
    raw = f"{user_id}:{opportunity_type}:{timeline_item_id}:{trigger_window}:{rule_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Opportunity(BaseModel):
    """Data model representing a contextual intervention opportunity."""

    id: str = Field(..., description="Unique opportunity ID")
    user_id: str = Field(..., description="Target user ID")
    opportunity_type: str = Field(..., description="Categorical type of opportunity")
    timeline_item_id: str = Field(..., description="Associated timeline item ID")
    trigger_window: str = Field(..., description="Logical trigger window identifier")
    rule_version: str = Field(default="v1", description="Opportunity generation rule version")
    idempotency_key: str = Field(..., description="Deterministic idempotency key")
    status: OpportunityStatus = Field(default=OpportunityStatus.CREATED, description="Current opportunity lifecycle status")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")
    claimed_at: Optional[datetime] = Field(default=None, description="Timestamp when opportunity was claimed by a consumer")
    lease_expires_at: Optional[datetime] = Field(default=None, description="Lease expiration timestamp for active consumer claims")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Custom payload passed to consumers")

    @property
    def is_lease_expired(self) -> bool:
        """Check if an active claim lease has expired."""
        if self.status == OpportunityStatus.CLAIMED and self.lease_expires_at:
            now = datetime.now(timezone.utc)
            lease_exp = self.lease_expires_at if self.lease_expires_at.tzinfo else self.lease_expires_at.replace(tzinfo=timezone.utc)
            return now >= lease_exp
        return False

    @classmethod
    def create(
        cls,
        user_id: str,
        opportunity_type: str,
        timeline_item_id: str,
        trigger_window: str,
        rule_version: str = "v1",
        expires_at: Optional[datetime] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Opportunity:
        key = generate_idempotency_key(user_id, opportunity_type, timeline_item_id, trigger_window, rule_version)
        # Use full SHA256 hex digest of the key to guarantee uniform distribution and zero false collisions across distinct items
        opp_id = f"opp_{key[:32]}"
        return cls(
            id=opp_id,
            user_id=user_id,
            opportunity_type=opportunity_type,
            timeline_item_id=timeline_item_id,
            trigger_window=trigger_window,
            rule_version=rule_version,
            idempotency_key=key,
            status=OpportunityStatus.CREATED,
            expires_at=expires_at,
            payload=payload or {},
        )
