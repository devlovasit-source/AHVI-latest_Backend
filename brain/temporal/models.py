"""Timeline Contract Data Models for AHVI Temporal Intelligence.

Defines the normalized TimelineItem schema and supporting enums required across
all timeline adapters, the Temporal Context Engine, and signal generation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Flexibility(str, Enum):
    """Categorizes the scheduling flexibility of a timeline item."""

    FIXED = "FIXED"
    MOVABLE = "MOVABLE"
    FLEXIBLE_WINDOW = "FLEXIBLE_WINDOW"


class TimelineItemStatus(str, Enum):
    """Lifecycle status of a timeline item."""

    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class TimelineSourceType(str, Enum):
    """Source module originating the timeline item."""

    CALENDAR = "CALENDAR"
    WORKOUT = "WORKOUT"
    MEAL = "MEAL"
    MEDICATION = "MEDICATION"
    BILL = "BILL"
    SKINCARE = "SKINCARE"


class TimelineItem(BaseModel):
    """Normalized TimelineItem schema.

    Represents a unified temporal activity or requirement across any AHVI module.
    """

    id: str = Field(..., description="Unique timeline item ID")
    user_id: str = Field(..., description="ID of the user owning this item")
    source: TimelineSourceType = Field(..., description="Originating source module")
    source_id: str = Field(..., description="ID of the raw source record")
    type: str = Field(default="general", description="Primary activity classification")
    subtype: str = Field(default="general", description="Granular activity classification")
    title: str = Field(..., description="Human-readable item title")
    start_time: datetime = Field(..., description="Start timestamp of the activity")
    end_time: datetime = Field(..., description="End timestamp of the activity")
    priority: int = Field(default=3, ge=1, le=5, description="Priority level from 1 (low) to 5 (high)")
    status: TimelineItemStatus = Field(default=TimelineItemStatus.SCHEDULED, description="Current item status")
    flexibility: Flexibility = Field(default=Flexibility.FIXED, description="Scheduling flexibility constraint")
    preparation_required: bool = Field(default=False, description="Flag indicating if advance preparation is needed")
    preparation_start: Optional[datetime] = Field(default=None, description="Timestamp when preparation window opens")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary source-specific metadata payload")

    @property
    def duration_minutes(self) -> float:
        """Calculate item duration in minutes."""
        if self.start_time and self.end_time:
            return max(0.0, (self.end_time - self.start_time).total_seconds() / 60.0)
        return 0.0

    @property
    def preparation_lead_minutes(self) -> float:
        """Calculate preparation lead time in minutes prior to start_time."""
        if self.preparation_required and self.preparation_start and self.start_time:
            lead = (self.start_time - self.preparation_start).total_seconds() / 60.0
            return max(0.0, lead)
        return 0.0
