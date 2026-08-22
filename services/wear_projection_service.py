"""Wear Projection Service.

Provides projection queries for item wear history (total wears, last worn at,
days since last worn, 7d/30d counts, and monthly breakdown).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.wear_projection_service")


class WearProjectionService:
    def __init__(self) -> None:
        self.proxy = AppwriteProxy()
        self.resource = "wear_events"

    def get_item_wear_history_projection(self, *, user_id: str, item_id: str) -> Dict[str, Any]:
        uid = str(user_id or "").strip()
        iid = str(item_id or "").strip()
        if not uid or not iid:
            return {
                "item_id": iid,
                "summary": {
                    "total_wears": 0,
                    "last_worn_at": None,
                    "days_since_last_worn": None,
                    "wears_last_7_days": 0,
                    "wears_last_30_days": 0,
                },
                "monthly": [],
                "history": [],
            }

        try:
            rows = self.proxy.list_documents(self.resource, user_id=uid, limit=300)
        except Exception as exc:
            logger.warning("ahvi.wear_projection_failed user_id=%s err=%s", uid, str(exc)[:140])
            rows = []

        valid_events: List[Dict[str, Any]] = []
        monthly_counts: Dict[str, int] = {}

        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            if r.get("revokedAtISO"):
                continue
            item_ids = r.get("itemIds") or []
            if isinstance(item_ids, list) and iid in [str(x).strip() for x in item_ids]:
                valid_events.append(r)
                # Compute monthly group YYYY-MM
                local_date = str(r.get("localDate") or r.get("occurredAtISO") or "")[:7]
                if len(local_date) == 7:
                    monthly_counts[local_date] = monthly_counts.get(local_date, 0) + 1

        # Sort descending by occurredAtISO
        def _get_ts(doc):
            val = doc.get("occurredAtISO") or ""
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0

        valid_events.sort(key=_get_ts, reverse=True)

        now_dt = datetime.now(datetime.now().astimezone().tzinfo)
        latest_ts = _get_ts(valid_events[0]) if valid_events else 0.0
        last_worn_iso = valid_events[0].get("occurredAtISO") if valid_events else None
        days_since = max(0, int((now_dt.timestamp() - latest_ts) // 86400)) if latest_ts > 0 else None

        ts_7d = now_dt.timestamp() - (7 * 86400)
        ts_30d = now_dt.timestamp() - (30 * 86400)

        wears_7d = sum(1 for e in valid_events if _get_ts(e) >= ts_7d)
        wears_30d = sum(1 for e in valid_events if _get_ts(e) >= ts_30d)

        monthly_list = [
            {"month": m, "wears": cnt}
            for m, cnt in sorted(monthly_counts.items(), reverse=True)
        ]

        history_list = [
            {
                "wear_event_id": str(e.get("$id") or e.get("id") or ""),
                "worn_at": e.get("occurredAtISO"),
                "source": e.get("source"),
                "outfit_id": e.get("outfitId"),
            }
            for e in valid_events
        ]

        return {
            "item_id": iid,
            "summary": {
                "total_wears": len(valid_events),
                "last_worn_at": last_worn_iso,
                "days_since_last_worn": days_since,
                "wears_last_7_days": wears_7d,
                "wears_last_30_days": wears_30d,
            },
            "monthly": monthly_list,
            "history": history_list,
        }
