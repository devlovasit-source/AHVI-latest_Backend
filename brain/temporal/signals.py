"""Temporal Signals for AHVI Temporal Intelligence.

Defines the complete, spec-aligned TemporalSignalType enum and TemporalSignal model,
along with the TemporalSignalEmitter that generates signals from TimelineItem streams.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from brain.temporal.models import TimelineItem, TimelineItemStatus

logger = logging.getLogger("ahvi.temporal.signals")


class TemporalSignalType(str, Enum):
    """Categorical types of temporal signals emitted from timeline stream evaluations."""

    UPCOMING = "UPCOMING"
    STARTING_SOON = "STARTING_SOON"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    MISSED = "MISSED"
    CONFLICT_DETECTED = "CONFLICT_DETECTED"
    PREPARATION_WINDOW_OPEN = "PREPARATION_WINDOW_OPEN"
    FREE_WINDOW_AVAILABLE = "FREE_WINDOW_AVAILABLE"
    SCHEDULE_CHANGE = "SCHEDULE_CHANGE"


class TemporalSignal(BaseModel):
    """Data model representing a temporal signal event."""

    id: str = Field(..., description="Unique signal ID")
    user_id: str = Field(..., description="Target user ID")
    signal_type: TemporalSignalType = Field(..., description="Type of temporal signal")
    timeline_item_id: str = Field(..., description="Associated timeline item ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Emission timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Context payload")


class TemporalSignalEmitter:
    """Evaluates TimelineItem states against current time and emits TemporalSignals."""

    def emit_signals_for_items(
        self,
        timeline_items: List[TimelineItem],
        current_time: Optional[datetime] = None,
    ) -> List[TemporalSignal]:
        """Generate signals for a list of timeline items based on reference timestamp."""
        ref_time = current_time or datetime.now(timezone.utc)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        signals: List[TemporalSignal] = []

        for item in timeline_items:
            if item.status == TimelineItemStatus.CANCELLED:
                continue

            s_time = item.start_time if item.start_time.tzinfo else item.start_time.replace(tzinfo=timezone.utc)
            e_time = item.end_time if item.end_time.tzinfo else item.end_time.replace(tzinfo=timezone.utc)

            # Preparation window open signal
            if item.preparation_required and item.preparation_start:
                p_start = item.preparation_start if item.preparation_start.tzinfo else item.preparation_start.replace(tzinfo=timezone.utc)
                if p_start <= ref_time < s_time:
                    sig_id = f"sig_prep_{item.id}_{int(p_start.timestamp())}"
                    signals.append(
                        TemporalSignal(
                            id=sig_id,
                            user_id=item.user_id,
                            signal_type=TemporalSignalType.PREPARATION_WINDOW_OPEN,
                            timeline_item_id=item.id,
                            timestamp=ref_time,
                            metadata={"lead_minutes": item.preparation_lead_minutes, "title": item.title},
                        )
                    )

            # Starting soon signal (within 30 mins)
            mins_to_start = (s_time - ref_time).total_seconds() / 60.0
            if 0.0 <= mins_to_start <= 30.0:
                sig_id = f"sig_soon_{item.id}_{int(ref_time.timestamp() // 300)}"
                signals.append(
                    TemporalSignal(
                        id=sig_id,
                        user_id=item.user_id,
                        signal_type=TemporalSignalType.STARTING_SOON,
                        timeline_item_id=item.id,
                        timestamp=ref_time,
                        metadata={"minutes_to_start": mins_to_start, "title": item.title},
                    )
                )

            # Upcoming signal (between 30 mins and 24 hours)
            elif 30.0 < mins_to_start <= 1440.0:
                sig_id = f"sig_up_{item.id}_{int(s_time.timestamp())}"
                signals.append(
                    TemporalSignal(
                        id=sig_id,
                        user_id=item.user_id,
                        signal_type=TemporalSignalType.UPCOMING,
                        timeline_item_id=item.id,
                        timestamp=ref_time,
                        metadata={"minutes_to_start": mins_to_start, "title": item.title},
                    )
                )

            # Started signal (in progress)
            if s_time <= ref_time <= e_time:
                sig_id = f"sig_start_{item.id}_{int(s_time.timestamp())}"
                signals.append(
                    TemporalSignal(
                        id=sig_id,
                        user_id=item.user_id,
                        signal_type=TemporalSignalType.STARTED,
                        timeline_item_id=item.id,
                        timestamp=ref_time,
                        metadata={"title": item.title},
                    )
                )

        return signals


# Global singleton
signal_emitter = TemporalSignalEmitter()
