"""Shadow Mode Evaluator and Parity Gate for AHVI Temporal Intelligence.

Executes legacy Calendar Intelligence and the new Temporal Context Engine side-by-side
when `ENABLE_TEMPORAL_SHADOW_MODE=true`. Calculates diffs and enforces the Shadow Mode Parity Gate.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from models.calendar_models import CalendarEventInput
from brain.engines.calendar_runtime import run_calendar_runtime
from brain.temporal.adapters.calendar_adapter import calendar_adapter
from brain.temporal.context_engine import context_engine
from brain.temporal.models import TimelineItem

logger = logging.getLogger("ahvi.temporal.shadow_mode")


def is_shadow_mode_enabled() -> bool:
    """Check if Temporal Shadow Mode is enabled via environment variable."""
    return os.getenv("ENABLE_TEMPORAL_SHADOW_MODE", "false").lower() in ("true", "1", "yes")


class ShadowModeEvaluator:
    """Side-by-side evaluator comparing legacy Calendar Intelligence with Temporal Context Engine."""

    def evaluate_events(
        self,
        raw_events: List[Dict[str, Any]],
        user_id: str = "shadow_user",
        triaged_diff_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run shadow mode comparison on a batch of calendar events."""
        triaged_set = set(triaged_diff_keys or [])
        diffs: List[Dict[str, Any]] = []
        matching_count = 0

        # Step 1: Normalize via new Temporal Adapter
        timeline_items = calendar_adapter.normalize_batch(raw_events, user_id=user_id)
        item_map: Dict[str, TimelineItem] = {item.source_id: item for item in timeline_items}

        for raw_event in raw_events:
            event_id = raw_event.get("eventId") or raw_event.get("id") or "unknown"
            
            # Step 2: Legacy output
            c_input = CalendarEventInput(
                eventId=event_id,
                title=raw_event.get("title", ""),
                startAtISO=raw_event.get("startAtISO") or raw_event.get("start_time") or raw_event.get("startAt") or datetime.now(timezone.utc).isoformat(),
                endAtISO=raw_event.get("endAtISO") or raw_event.get("end_time") or raw_event.get("endAt") or datetime.now(timezone.utc).isoformat(),
                location=raw_event.get("location"),
                dressCode=raw_event.get("dressCode"),
                isAllDay=raw_event.get("isAllDay", False),
            )
            legacy_res = run_calendar_runtime(c_input, user_id=user_id)
            legacy_classified = legacy_res.classifiedEvent

            # Step 3: New Temporal Timeline Item output
            t_item = item_map.get(event_id)

            if not t_item:
                diffs.append({
                    "event_id": event_id,
                    "field": "missing_timeline_item",
                    "legacy_value": legacy_classified.group,
                    "temporal_value": None,
                    "triaged": ("missing_timeline_item" in triaged_set),
                })
                continue

            # Compare category / group classification
            event_diffs = []
            if legacy_classified.group.lower() != t_item.type.lower():
                event_diffs.append({
                    "event_id": event_id,
                    "field": "type_group_mismatch",
                    "legacy_value": legacy_classified.group,
                    "temporal_value": t_item.type,
                    "triaged": ("type_group_mismatch" in triaged_set),
                })

            # Compare subtype
            if legacy_classified.subtype.lower() != t_item.subtype.lower():
                event_diffs.append({
                    "event_id": event_id,
                    "field": "subtype_mismatch",
                    "legacy_value": legacy_classified.subtype,
                    "temporal_value": t_item.subtype,
                    "triaged": ("subtype_mismatch" in triaged_set),
                })

            if not event_diffs:
                matching_count += 1
            else:
                diffs.extend(event_diffs)

        total_events = len(raw_events)
        diff_count = len(diffs)
        untriaged_diffs = [d for d in diffs if not d.get("triaged")]

        diff_rate_percent = (diff_count / total_events * 100.0) if total_events > 0 else 0.0
        
        # Parity Gate criteria: 0% diff OR 100% of diffs are triaged
        parity_passed = (diff_count == 0) or (len(untriaged_diffs) == 0)

        result = {
            "total_events": total_events,
            "matching_events": matching_count,
            "diff_count": diff_count,
            "untriaged_diff_count": len(untriaged_diffs),
            "diff_rate_percent": round(diff_rate_percent, 2),
            "parity_passed": parity_passed,
            "diffs": diffs,
        }

        if is_shadow_mode_enabled():
            logger.info(
                "AHVI_TEMPORAL_SHADOW_MODE_EVALUATION total=%d matches=%d diffs=%d parity_passed=%s",
                total_events,
                matching_count,
                diff_count,
                parity_passed,
            )

        return result


# Global singleton
shadow_evaluator = ShadowModeEvaluator()
