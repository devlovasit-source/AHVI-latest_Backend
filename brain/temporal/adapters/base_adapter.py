"""Abstract Base Class for Timeline Source Adapters.

Enforces a strict contract across all module adapters:
Source Data -> Validate -> Normalize -> TimelineItem
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from brain.temporal.models import TimelineItem, TimelineSourceType

logger = logging.getLogger("ahvi.temporal.adapters")


class TimelineSourceAdapter(ABC):
    """Abstract base class for all module timeline adapters."""

    @property
    @abstractmethod
    def source_type(self) -> TimelineSourceType:
        """Return the TimelineSourceType enum identifying this source module."""
        pass

    @abstractmethod
    def fetch_raw(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch raw source documents for [user_id] within [start_time, end_time]."""
        pass

    @abstractmethod
    def validate(self, raw_item: Dict[str, Any]) -> bool:
        """Validate whether a raw source document contains necessary fields."""
        pass

    @abstractmethod
    def normalize(
        self,
        raw_item: Dict[str, Any],
        user_id: str,
    ) -> Optional[TimelineItem]:
        """Normalize a single validated raw source document into a TimelineItem."""
        pass

    def fetch_and_normalize(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[TimelineItem]:
        """Fault-isolated workflow to fetch, validate, and normalize source items."""
        normalized_items: List[TimelineItem] = []
        try:
            raw_records = self.fetch_raw(user_id, start_time, end_time)
        except Exception as exc:
            logger.error(
                "AHVI_ADAPTER_FETCH_FAILED source=%s user_id=%s err=%s",
                self.source_type.value,
                user_id,
                str(exc),
                exc_info=True,
            )
            return []

        for raw_item in raw_records or []:
            if not isinstance(raw_item, dict):
                continue
            try:
                if not self.validate(raw_item):
                    continue
                item = self.normalize(raw_item, user_id)
                if item is not None:
                    normalized_items.append(item)
            except Exception as exc:
                logger.warning(
                    "AHVI_ADAPTER_NORMALIZE_ITEM_FAILED source=%s user_id=%s item_id=%s err=%s",
                    self.source_type.value,
                    user_id,
                    raw_item.get("id") or raw_item.get("eventId") or "?",
                    str(exc),
                )
                continue

        return normalized_items

    def normalize_batch(
        self,
        raw_items: List[Dict[str, Any]],
        user_id: str,
    ) -> List[TimelineItem]:
        """Normalize a list of raw records safely with fault isolation."""
        normalized: List[TimelineItem] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            try:
                if self.validate(raw):
                    item = self.normalize(raw, user_id)
                    if item:
                        normalized.append(item)
            except Exception as exc:
                logger.warning(
                    "AHVI_ADAPTER_NORMALIZE_BATCH_ITEM_FAILED source=%s user_id=%s err=%s",
                    self.source_type.value,
                    user_id,
                    str(exc),
                )
        return normalized

