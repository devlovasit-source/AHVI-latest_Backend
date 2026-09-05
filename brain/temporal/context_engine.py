"""Temporal Context Engine for AHVI Temporal Intelligence.

Read-only, deterministic service evaluating TimelineItem streams. Calculates current
activity, upcoming activities, free/busy blocks, schedule conflicts, and preparation lead time windows.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from brain.temporal.models import TimelineItem, TimelineItemStatus

logger = logging.getLogger("ahvi.temporal.context_engine")


class TemporalContextEngine:
    """Read-only deterministic engine for temporal context calculation."""

    def get_current_activity(
        self,
        user_id: str,
        timestamp: Optional[datetime] = None,
        items: Optional[List[TimelineItem]] = None,
    ) -> Optional[TimelineItem]:
        """Find the item currently in progress for a user at the given timestamp."""
        ref_time = timestamp or datetime.now(timezone.utc)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        user_items = [
            it for it in (items or [])
            if it.user_id == user_id and it.status != TimelineItemStatus.CANCELLED
        ]

        active_items = []
        for it in user_items:
            s = it.start_time if it.start_time.tzinfo else it.start_time.replace(tzinfo=timezone.utc)
            e = it.end_time if it.end_time.tzinfo else it.end_time.replace(tzinfo=timezone.utc)
            if s <= ref_time <= e:
                active_items.append(it)

        if not active_items:
            return None

        # Return highest priority item if multiple overlap
        active_items.sort(key=lambda x: x.priority, reverse=True)
        return active_items[0]

    def get_upcoming_activities(
        self,
        user_id: str,
        window_hours: float = 24.0,
        start_time: Optional[datetime] = None,
        items: Optional[List[TimelineItem]] = None,
    ) -> List[TimelineItem]:
        """Get upcoming items for a user within a lookahead window."""
        ref_start = start_time or datetime.now(timezone.utc)
        if ref_start.tzinfo is None:
            ref_start = ref_start.replace(tzinfo=timezone.utc)

        ref_end = ref_start + timedelta(hours=window_hours)

        user_items = [
            it for it in (items or [])
            if it.user_id == user_id and it.status != TimelineItemStatus.CANCELLED
        ]

        upcoming = []
        for it in user_items:
            s = it.start_time if it.start_time.tzinfo else it.start_time.replace(tzinfo=timezone.utc)
            if ref_start <= s <= ref_end:
                upcoming.append(it)

        upcoming.sort(key=lambda x: (x.start_time, -x.priority))
        return upcoming

    def calculate_free_busy_windows(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        items: Optional[List[TimelineItem]] = None,
    ) -> Dict[str, Any]:
        """Calculate free and busy time intervals for a user between start_time and end_time."""
        s_bound = start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)
        e_bound = end_time if end_time.tzinfo else end_time.replace(tzinfo=timezone.utc)

        if e_bound <= s_bound:
            return {"busy_blocks": [], "free_blocks": [], "total_free_minutes": 0.0}

        user_items = [
            it for it in (items or [])
            if it.user_id == user_id and it.status != TimelineItemStatus.CANCELLED
        ]

        # Extract busy intervals within window
        busy_intervals = []
        for it in user_items:
            s = it.start_time if it.start_time.tzinfo else it.start_time.replace(tzinfo=timezone.utc)
            e = it.end_time if it.end_time.tzinfo else it.end_time.replace(tzinfo=timezone.utc)
            
            # Clip interval to s_bound, e_bound
            clipped_s = max(s, s_bound)
            clipped_e = min(e, e_bound)
            if clipped_s < clipped_e:
                busy_intervals.append((clipped_s, clipped_e, it))

        # Merge overlapping busy intervals
        busy_intervals.sort(key=lambda x: x[0])
        merged_busy = []
        for cur_s, cur_e, item in busy_intervals:
            if not merged_busy:
                merged_busy.append([cur_s, cur_e])
            else:
                last_s, last_e = merged_busy[-1]
                if cur_s <= last_e:
                    merged_busy[-1][1] = max(last_e, cur_e)
                else:
                    merged_busy.append([cur_s, cur_e])

        # Compute free intervals
        free_blocks = []
        curr_ptr = s_bound

        for b_start, b_end in merged_busy:
            if b_start > curr_ptr:
                free_blocks.append({
                    "start": curr_ptr.isoformat(),
                    "end": b_start.isoformat(),
                    "duration_minutes": (b_start - curr_ptr).total_seconds() / 60.0,
                })
            curr_ptr = max(curr_ptr, b_end)

        if curr_ptr < e_bound:
            free_blocks.append({
                "start": curr_ptr.isoformat(),
                "end": e_bound.isoformat(),
                "duration_minutes": (e_bound - curr_ptr).total_seconds() / 60.0,
            })

        busy_blocks = [
            {
                "start": b_start.isoformat(),
                "end": b_end.isoformat(),
                "duration_minutes": (b_end - b_start).total_seconds() / 60.0,
            }
            for b_start, b_end in merged_busy
        ]

        total_free = sum(fb["duration_minutes"] for fb in free_blocks)

        return {
            "busy_blocks": busy_blocks,
            "free_blocks": free_blocks,
            "total_free_minutes": total_free,
        }

    def detect_schedule_conflicts(
        self,
        timeline_items: List[TimelineItem],
    ) -> List[Dict[str, Any]]:
        """Detect overlapping timeline items for the same user."""
        conflicts: List[Dict[str, Any]] = []

        valid_items = [it for it in timeline_items if it.status != TimelineItemStatus.CANCELLED]
        sorted_items = sorted(
            valid_items,
            key=lambda x: x.start_time if x.start_time.tzinfo else x.start_time.replace(tzinfo=timezone.utc),
        )

        n = len(sorted_items)
        for i in range(n):
            item_a = sorted_items[i]
            s_a = item_a.start_time if item_a.start_time.tzinfo else item_a.start_time.replace(tzinfo=timezone.utc)
            e_a = item_a.end_time if item_a.end_time.tzinfo else item_a.end_time.replace(tzinfo=timezone.utc)

            for j in range(i + 1, n):
                item_b = sorted_items[j]
                s_b = item_b.start_time if item_b.start_time.tzinfo else item_b.start_time.replace(tzinfo=timezone.utc)
                e_b = item_b.end_time if item_b.end_time.tzinfo else item_b.end_time.replace(tzinfo=timezone.utc)

                if s_b >= e_a:
                    break  # No further overlaps possible for item_a

                if item_a.user_id == item_b.user_id:
                    overlap_start = max(s_a, s_b)
                    overlap_end = min(e_a, e_b)
                    overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60.0

                    if overlap_minutes > 0:
                        conflicts.append({
                            "user_id": item_a.user_id,
                            "item_a_id": item_a.id,
                            "item_a_title": item_a.title,
                            "item_b_id": item_b.id,
                            "item_b_title": item_b.title,
                            "overlap_minutes": overlap_minutes,
                            "overlap_start": overlap_start.isoformat(),
                            "overlap_end": overlap_end.isoformat(),
                        })

        return conflicts

    def calculate_preparation_lead_time(
        self,
        timeline_item: TimelineItem,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Calculate preparation window status and lead time details for an item."""
        ref_time = current_time or datetime.now(timezone.utc)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        s_time = timeline_item.start_time if timeline_item.start_time.tzinfo else timeline_item.start_time.replace(tzinfo=timezone.utc)

        if not timeline_item.preparation_required or not timeline_item.preparation_start:
            return {
                "preparation_required": False,
                "preparation_lead_minutes": 0.0,
                "is_in_prep_window": False,
                "prep_window_open": False,
            }

        p_start = timeline_item.preparation_start if timeline_item.preparation_start.tzinfo else timeline_item.preparation_start.replace(tzinfo=timezone.utc)
        lead_minutes = max(0.0, (s_time - p_start).total_seconds() / 60.0)

        is_in_window = (p_start <= ref_time < s_time)
        is_window_open = (ref_time >= p_start)

        return {
            "preparation_required": True,
            "preparation_lead_minutes": lead_minutes,
            "preparation_start": p_start.isoformat(),
            "start_time": s_time.isoformat(),
            "is_in_prep_window": is_in_window,
            "prep_window_open": is_window_open,
        }


# Global singleton instance
context_engine = TemporalContextEngine()
