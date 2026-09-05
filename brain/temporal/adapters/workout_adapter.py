"""Workout Timeline Source Adapter for AHVI Temporal Intelligence.

Reads raw workout items and normalizes them into TimelineItem objects.
Read-only and fault-isolated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from brain.temporal.adapters.base_adapter import TimelineSourceAdapter
from brain.temporal.models import Flexibility, TimelineItem, TimelineItemStatus, TimelineSourceType
from services.appwrite_proxy import AppwriteProxy


def _parse_dt(val: Any) -> Optional[datetime]:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    text = str(val or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


class WorkoutTimelineAdapter(TimelineSourceAdapter):
    """Adapter normalizing workout routines/sessions into TimelineItems."""

    @property
    def source_type(self) -> TimelineSourceType:
        return TimelineSourceType.WORKOUT

    def fetch_raw(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        proxy = AppwriteProxy()
        try:
            docs = proxy.list_documents("workout_plans", queries=[])
            if isinstance(docs, list):
                return [d for d in docs if isinstance(d, dict)]
        except Exception:
            pass
        return []

    def validate(self, raw_item: Dict[str, Any]) -> bool:
        if not isinstance(raw_item, dict):
            return False
        title = str(raw_item.get("title") or raw_item.get("workout_type") or raw_item.get("name") or "").strip()
        if not title:
            return False
        start = _parse_dt(raw_item.get("scheduled_at") or raw_item.get("start_time") or raw_item.get("date"))
        return start is not None

    def normalize(self, raw_item: Dict[str, Any], user_id: str) -> Optional[TimelineItem]:
        if not self.validate(raw_item):
            return None

        source_id = str(raw_item.get("id") or raw_item.get("$id") or raw_item.get("workout_id") or "wrk_1").strip()
        title = str(raw_item.get("title") or raw_item.get("workout_type") or "Workout").strip()
        start_time = _parse_dt(raw_item.get("scheduled_at") or raw_item.get("start_time") or raw_item.get("date"))
        if not start_time:
            return None

        duration_mins = float(raw_item.get("duration_minutes") or 45)
        end_time = start_time + timedelta(minutes=duration_mins)

        return TimelineItem(
            id=f"wrk_{source_id}",
            user_id=user_id,
            source=TimelineSourceType.WORKOUT,
            source_id=source_id,
            type="fitness",
            subtype=title.lower(),
            title=title,
            start_time=start_time,
            end_time=end_time,
            priority=2,
            status=TimelineItemStatus.SCHEDULED,
            flexibility=Flexibility.MOVABLE,
            preparation_required=True,
            preparation_start=start_time - timedelta(minutes=15),
            metadata={"duration_minutes": duration_mins},
        )


# Global singleton
workout_adapter = WorkoutTimelineAdapter()
