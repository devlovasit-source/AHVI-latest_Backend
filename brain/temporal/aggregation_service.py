"""Timeline Aggregation Service for AHVI Temporal Intelligence.

Aggregates TimelineItems across all active module adapters with per-adapter fault isolation.
A failure or exception in any single adapter is logged without crashing the overall stream.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from brain.temporal.adapters.base_adapter import TimelineSourceAdapter
from brain.temporal.adapters.bill_adapter import bill_adapter
from brain.temporal.adapters.calendar_adapter import calendar_adapter
from brain.temporal.adapters.meal_adapter import meal_adapter
from brain.temporal.adapters.medication_adapter import medication_adapter
from brain.temporal.adapters.skincare_adapter import skincare_adapter
from brain.temporal.adapters.workout_adapter import workout_adapter
from brain.temporal.models import TimelineItem

logger = logging.getLogger("ahvi.temporal.aggregation_service")


class TimelineAggregationService:
    """Unified timeline aggregator with fault isolation per adapter."""

    def __init__(self, adapters: Optional[List[TimelineSourceAdapter]] = None) -> None:
        self.adapters: List[TimelineSourceAdapter] = adapters or [
            calendar_adapter,
            workout_adapter,
            meal_adapter,
            medication_adapter,
            bill_adapter,
            skincare_adapter,
        ]

    def fetch_unified_timeline(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[TimelineItem]:
        """Fetch and aggregate normalized TimelineItems across all registered adapters."""
        all_items: List[TimelineItem] = []

        for adapter in self.adapters:
            source_name = adapter.source_type.value
            try:
                items = adapter.fetch_and_normalize(user_id, start_time, end_time)
                all_items.extend(items)
            except Exception as exc:
                logger.error(
                    "AHVI_AGGREGATION_SERVICE_ADAPTER_ERROR adapter=%s user_id=%s err=%s",
                    source_name,
                    user_id,
                    str(exc),
                    exc_info=True,
                )
                continue

        # Sort aggregated stream by start_time ascending, then priority descending
        all_items.sort(key=lambda x: (x.start_time, -x.priority))
        return all_items


# Global singleton
aggregation_service = TimelineAggregationService()
