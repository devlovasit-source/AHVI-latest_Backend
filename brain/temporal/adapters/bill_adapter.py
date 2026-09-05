"""Bill Timeline Source Adapter for AHVI Temporal Intelligence.

Reads raw bill payment reminders and normalizes them into TimelineItem objects.
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


class BillTimelineAdapter(TimelineSourceAdapter):
    """Adapter normalizing bill due dates into TimelineItems."""

    @property
    def source_type(self) -> TimelineSourceType:
        return TimelineSourceType.BILL

    def fetch_raw(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        proxy = AppwriteProxy()
        try:
            docs = proxy.list_documents("bills", queries=[])
            if isinstance(docs, list):
                return [d for d in docs if isinstance(d, dict)]
        except Exception:
            pass
        return []

    def validate(self, raw_item: Dict[str, Any]) -> bool:
        if not isinstance(raw_item, dict):
            return False
        title = str(raw_item.get("bill_name") or raw_item.get("title") or raw_item.get("name") or "").strip()
        if not title:
            return False
        start = _parse_dt(raw_item.get("due_date") or raw_item.get("due_at") or raw_item.get("scheduled_at"))
        return start is not None

    def normalize(self, raw_item: Dict[str, Any], user_id: str) -> Optional[TimelineItem]:
        if not self.validate(raw_item):
            return None

        source_id = str(raw_item.get("id") or raw_item.get("$id") or raw_item.get("bill_id") or "bill_1").strip()
        title = str(raw_item.get("bill_name") or raw_item.get("title") or "Bill Payment").strip()
        due_date = _parse_dt(raw_item.get("due_date") or raw_item.get("due_at") or raw_item.get("scheduled_at"))
        if not due_date:
            return None

        end_time = due_date + timedelta(minutes=30)

        return TimelineItem(
            id=f"bill_{source_id}",
            user_id=user_id,
            source=TimelineSourceType.BILL,
            source_id=source_id,
            type="finance",
            subtype="payment",
            title=f"Pay {title}",
            start_time=due_date,
            end_time=end_time,
            priority=4,
            status=TimelineItemStatus.SCHEDULED,
            flexibility=Flexibility.FLEXIBLE_WINDOW,
            preparation_required=True,
            preparation_start=due_date - timedelta(hours=24),
            metadata={"amount": raw_item.get("amount", 0.0)},
        )


# Global singleton
bill_adapter = BillTimelineAdapter()
