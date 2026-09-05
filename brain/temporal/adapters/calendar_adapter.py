"""Calendar Timeline Source Adapter for AHVI Temporal Intelligence.

Reads raw calendar events and normalizes them into TimelineItem objects.
Read-only and side-effect free.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from brain.engines.calendar_runtime import _infer_group_subtype_priority
from brain.temporal.adapters.base_adapter import TimelineSourceAdapter
from brain.temporal.models import Flexibility, TimelineItem, TimelineItemStatus, TimelineSourceType
from services.appwrite_proxy import AppwriteProxy


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _map_priority(val: Any) -> int:
    try:
        return max(1, min(5, int(val)))
    except Exception:
        text = str(val or "").strip().lower()
        mapping = {
            "urgent": 5,
            "critical": 5,
            "important": 4,
            "high": 4,
            "normal": 3,
            "medium": 3,
            "gentle": 2,
            "low": 2,
            "optional": 1,
        }
        return mapping.get(text, 3)


class CalendarTimelineAdapter(TimelineSourceAdapter):
    """Adapter normalizing calendar events into TimelineItem records."""

    @property
    def source_type(self) -> TimelineSourceType:
        return TimelineSourceType.CALENDAR

    def fetch_raw(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        proxy = AppwriteProxy()
        docs: List[Dict[str, Any]] = []

        try:
            raw_docs = proxy.list_documents("calendar_events", queries=[])
            if isinstance(raw_docs, list):
                docs = [d for d in raw_docs if isinstance(d, dict)]
        except Exception:
            docs = []

        # Filter by user_id if present in document
        user_docs = [
            d for d in docs
            if str(d.get("userId") or d.get("user_id") or user_id) == user_id
        ]

        if not user_docs and docs:
            user_docs = docs

        # Filter by date range if provided
        if start_time or end_time:
            filtered: List[Dict[str, Any]] = []
            for d in user_docs:
                st = _parse_datetime(d.get("startAtISO") or d.get("start_time") or d.get("startTime"))
                if not st:
                    filtered.append(d)
                    continue
                if start_time and st < start_time:
                    continue
                if end_time and st > end_time:
                    continue
                filtered.append(d)
            return filtered

        return user_docs

    def validate(self, raw_item: Dict[str, Any]) -> bool:
        if not isinstance(raw_item, dict):
            return False
        title = str(raw_item.get("title") or raw_item.get("name") or raw_item.get("label") or "").strip()
        if not title:
            return False
        start = _parse_datetime(raw_item.get("startAtISO") or raw_item.get("start_time") or raw_item.get("startTime"))
        return start is not None

    def normalize(
        self,
        raw_item: Dict[str, Any],
        user_id: str,
    ) -> Optional[TimelineItem]:
        if not self.validate(raw_item):
            return None

        source_id = str(
            raw_item.get("eventId")
            or raw_item.get("$id")
            or raw_item.get("id")
            or raw_item.get("title")
        ).strip()

        title = str(raw_item.get("title") or raw_item.get("name") or "").strip()
        group, subtype, confidence, matched, priority_val = _infer_group_subtype_priority(title)

        start_time = _parse_datetime(raw_item.get("startAtISO") or raw_item.get("start_time") or raw_item.get("startTime"))
        end_time = _parse_datetime(raw_item.get("endAtISO") or raw_item.get("end_time") or raw_item.get("endTime"))

        if not start_time:
            return None
        if not end_time or end_time <= start_time:
            end_time = start_time + timedelta(hours=1)

        # Flexibility determination
        group_lower = group.lower()
        if group_lower in ("meeting", "travel", "work", "health"):
            flexibility = Flexibility.FIXED
        elif group_lower in ("social", "party"):
            flexibility = Flexibility.MOVABLE
        else:
            flexibility = Flexibility.FLEXIBLE_WINDOW

        # Preparation requirements
        prep_required = group_lower in ("meeting", "work", "travel", "social", "party", "health")
        prep_start = None
        if prep_required:
            lead_hours = 2.0 if group_lower in ("travel", "party") else 1.0
            prep_start = start_time - timedelta(hours=lead_hours)

        return TimelineItem(
            id=f"cal_{source_id}",
            user_id=user_id,
            source=TimelineSourceType.CALENDAR,
            source_id=source_id,
            type=group_lower,
            subtype=subtype.lower(),
            title=title,
            start_time=start_time,
            end_time=end_time,
            priority=_map_priority(priority_val),
            status=TimelineItemStatus.SCHEDULED,
            flexibility=flexibility,
            preparation_required=prep_required,
            preparation_start=prep_start,
            metadata={
                "location": str(raw_item.get("location") or ""),
                "group": group,
                "confidenceScore": confidence,
                "matchedSignals": matched,
            },
        )


# Global singleton instance
calendar_adapter = CalendarTimelineAdapter()
