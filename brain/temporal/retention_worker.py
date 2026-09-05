"""Retention and Expiry Worker for AHVI Temporal Intelligence.

Enforces configurable operational retention policies across TimelineItems,
TemporalSignals, and Opportunities. Automatically reclaims expired consumer claim leases
and replays unclaimed opportunities.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from brain.temporal.opportunity_models import OpportunityStatus
from brain.temporal.opportunity_notifier import opportunity_notifier
from brain.temporal.opportunity_store import opportunity_store

logger = logging.getLogger("ahvi.temporal.retention_worker")


def get_retention_settings() -> Dict[str, int]:
    """Fetch configurable operational retention settings from environment variables."""
    return {
        "TIMELINE_ITEM_RETENTION_PAST_DAYS": int(
            os.getenv("TIMELINE_ITEM_RETENTION_PAST_DAYS", "30")
        ),
        "TIMELINE_ITEM_RETENTION_FUTURE_DAYS": int(
            os.getenv("TIMELINE_ITEM_RETENTION_FUTURE_DAYS", "90")
        ),
        "TEMPORAL_SIGNAL_TTL_HOURS": int(
            os.getenv("TEMPORAL_SIGNAL_TTL_HOURS", "48")
        ),
        "OPPORTUNITY_RETENTION_POST_RESOLUTION_DAYS": int(
            os.getenv("OPPORTUNITY_RETENTION_POST_RESOLUTION_DAYS", "60")
        ),
    }


class RetentionWorker:
    """Worker handling periodic TTL cleanup and consumer lease recovery."""

    def __init__(self) -> None:
        self.settings = get_retention_settings()

    def run_cleanup(self, user_id: str) -> Dict[str, int]:
        """Execute cleanup pass, reclaim expired leases, and replay unclaimed opportunities."""
        now = datetime.now(timezone.utc)
        settings = get_retention_settings()

        # Step 1: Reclaim expired consumer leases
        reclaimed_leases = opportunity_store.reclaim_expired_leases(user_id)

        # Step 2: Auto recovery & replay unconsumed opportunities
        replayed_count = opportunity_notifier.recover_and_replay_unclaimed(user_id)

        # Step 3: Opportunity expiry & post-resolution retention
        cleaned_opps = 0
        opps = opportunity_store.query_user_opportunities(user_id)
        post_res_cutoff = now - timedelta(days=settings["OPPORTUNITY_RETENTION_POST_RESOLUTION_DAYS"])

        for opp in opps:
            # Expire if past expires_at
            if opp.expires_at:
                exp = opp.expires_at if opp.expires_at.tzinfo else opp.expires_at.replace(tzinfo=timezone.utc)
                if now >= exp and opp.status != OpportunityStatus.EXPIRED:
                    opportunity_store.update_status(opp.id, OpportunityStatus.EXPIRED)
                    cleaned_opps += 1
                    continue

            # Delete if post-resolution retention period exceeded
            if opp.created_at:
                created = opp.created_at if opp.created_at.tzinfo else opp.created_at.replace(tzinfo=timezone.utc)
                if created < post_res_cutoff:
                    opportunity_store.delete_opportunity(opp.id)
                    cleaned_opps += 1

        logger.info(
            "AHVI_TEMPORAL_RETENTION_CLEANUP user_id=%s reclaimed_leases=%d replayed=%d expired_opps=%d settings=%s",
            user_id,
            len(reclaimed_leases),
            replayed_count,
            cleaned_opps,
            settings,
        )

        return {
            "opportunities_cleaned": cleaned_opps,
            "reclaimed_leases": len(reclaimed_leases),
            "replayed_notifications": replayed_count,
            "past_days_limit": settings["TIMELINE_ITEM_RETENTION_PAST_DAYS"],
            "future_days_limit": settings["TIMELINE_ITEM_RETENTION_FUTURE_DAYS"],
            "signal_ttl_hours": settings["TEMPORAL_SIGNAL_TTL_HOURS"],
        }


# Global singleton
retention_worker = RetentionWorker()
