"""Calendar Timeline Source Adapter for AHVI Temporal Intelligence.

Reads raw calendar events and normalizes them into TimelineItem objects.
User-scoped, read-only, and reuses canonical calendar-local timezone lookup.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    _DEFAULT_TZ = ZoneInfo(os.getenv("CALENDAR_TIMEZONE", "Asia/Kolkata"))
except Exception:
    _DEFAULT_TZ = timezone(timedelta(hours=5, minutes=30))

from brain.engines.calendar_runtime import _infer_group_subtype_priority
from brain.temporal.adapters.base_adapter import TimelineSourceAdapter
from brain.temporal.models import Flexibility, TimelineItem, TimelineItemStatus, TimelineSourceType
from services.appwrite_proxy import AppwriteProxy


def resolve_calendar_timezone(raw_item: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None) -> Any:
    """Resolve configured calendar timezone from raw event payload, user settings, or default fallback."""
    if isinstance(raw_item, dict):
        tz_str = str(raw_item.get("timezone") or raw_item.get("tz") or raw_item.get("userTimezone") or "").strip()
        if tz_str:
            try:
                return ZoneInfo(tz_str)
            except Exception:
                pass
    try:
        from services.calendar_service import _CALENDAR_TZ
        return _CALENDAR_TZ
    except Exception:
        return _DEFAULT_TZ


def _parse_datetime(value: Any, raw_item: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None) -> Optional[datetime]:
    """Parse timestamp into UTC-aware datetime.

    Naive values are assigned the resolved calendar timezone (per-event/per-user config)
    and then converted to UTC. Aware values retain their timezone offsets.
    """
    if isinstance(value, datetime):
        tz = resolve_calendar_timezone(raw_item, user_id)
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=tz).astimezone(timezone.utc)

    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        tz = resolve_calendar_timezone(raw_item, user_id)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=tz).astimezone(timezone.utc)
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
        uid = str(user_id or "").strip()
        if not uid:
            return []

        proxy = AppwriteProxy()
        docs: List[Dict[str, Any]] = []

        try:
            raw_docs = proxy.list_documents("calendar_events", user_id=uid, limit=200)
            if isinstance(raw_docs, list):
                docs = [d for d in raw_docs if isinstance(d, dict)]
            elif isinstance(raw_docs, dict):
                docs = [d for d in raw_docs.get("documents", []) if isinstance(d, dict)]
        except Exception:
            docs = []

        # Strict user-scoping: NEVER fallback to all documents
        user_docs = [
            d for d in docs
            if str(d.get("userId") or d.get("user_id") or uid) == uid
        ]

        # Filter by date range if provided
        if start_time or end_time:
            filtered: List[Dict[str, Any]] = []
            for d in user_docs:
                st = _parse_datetime(d.get("startAtISO") or d.get("start_time") or d.get("startTime"), raw_item=d, user_id=uid)
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
        start = _parse_datetime(raw_item.get("startAtISO") or raw_item.get("start_time") or raw_item.get("startTime"), raw_item=raw_item)
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

        start_time = _parse_datetime(raw_item.get("startAtISO") or raw_item.get("start_time") or raw_item.get("startTime"), raw_item=raw_item, user_id=user_id)
        end_time = _parse_datetime(raw_item.get("endAtISO") or raw_item.get("end_time") or raw_item.get("endTime"), raw_item=raw_item, user_id=user_id)

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
