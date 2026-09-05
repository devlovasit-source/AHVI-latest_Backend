"""Medication Timeline Source Adapter for AHVI Temporal Intelligence.

Reads raw med_logs items and normalizes them into TimelineItem objects.
User-scoped, read-only, and fault-isolated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    DEFAULT_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    DEFAULT_TZ = timezone.utc

from brain.temporal.adapters.base_adapter import TimelineSourceAdapter
from brain.temporal.models import Flexibility, TimelineItem, TimelineItemStatus, TimelineSourceType
from services.appwrite_proxy import AppwriteProxy


def _parse_dt(val: Any) -> Optional[datetime]:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=DEFAULT_TZ)
    text = str(val or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=DEFAULT_TZ)
    except Exception:
        return None


class MedicationTimelineAdapter(TimelineSourceAdapter):
    """Adapter normalizing medication reminders into TimelineItems."""

    @property
    def source_type(self) -> TimelineSourceType:
        return TimelineSourceType.MEDICATION

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
            raw_docs = proxy.list_documents("med_logs", user_id=uid, limit=200)
            if isinstance(raw_docs, list):
                docs = [d for d in raw_docs if isinstance(d, dict)]
            elif isinstance(raw_docs, dict):
                docs = [d for d in raw_docs.get("documents", []) if isinstance(d, dict)]
        except Exception:
            docs = []

        return [
            d for d in docs
            if str(d.get("userId") or d.get("user_id") or uid) == uid
        ]

    def validate(self, raw_item: Dict[str, Any]) -> bool:
        if not isinstance(raw_item, dict):
            return False
        title = str(raw_item.get("medName") or raw_item.get("medication_name") or raw_item.get("name") or raw_item.get("title") or "").strip()
        if not title:
            return False
        start = _parse_dt(raw_item.get("time") or raw_item.get("scheduled_at") or raw_item.get("dosage_time") or raw_item.get("$createdAt"))
        return start is not None

    def normalize(self, raw_item: Dict[str, Any], user_id: str) -> Optional[TimelineItem]:
        if not self.validate(raw_item):
            return None

        source_id = str(raw_item.get("id") or raw_item.get("$id") or raw_item.get("med_id") or "med_1").strip()
        name = str(raw_item.get("medName") or raw_item.get("medication_name") or raw_item.get("name") or "Medication").strip()
        start_time = _parse_dt(raw_item.get("time") or raw_item.get("scheduled_at") or raw_item.get("dosage_time") or raw_item.get("$createdAt"))
        if not start_time:
            return None

        end_time = start_time + timedelta(minutes=15)

        return TimelineItem(
            id=f"med_{source_id}",
            user_id=user_id,
            source=TimelineSourceType.MEDICATION,
            source_id=source_id,
            type="health",
            subtype="medication",
            title=f"Take {name}",
            start_time=start_time,
            end_time=end_time,
            priority=4,  # High priority for health
            status=TimelineItemStatus.SCHEDULED,
            flexibility=Flexibility.FIXED,
            preparation_required=False,
            metadata={"dosage": raw_item.get("dose") or raw_item.get("dosage", "")},
        )


# Global singleton
medication_adapter = MedicationTimelineAdapter()
