"""Clickstream Analytics Service.

Ingests user behavioral clickstream events, handles sessionization,
and computes metrics (DAU, Feature Usage, Funnel Drop-offs, User Journeys).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional

from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.clickstream_service")


def _utcnow_iso() -> str:
    return datetime.now(dt_timezone.utc).isoformat()


class ClickstreamService:
    def __init__(self) -> None:
        self.proxy = AppwriteProxy()
        self.collection = "clickstream_events"
        self._memory_store: List[Dict[str, Any]] = []

    def ingest_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Idempotent batch ingestion of clickstream events."""
        if not events:
            return {"success": True, "accepted": 0, "duplicates": 0}

        accepted = 0
        duplicates = 0

        for raw in events:
            if not isinstance(raw, dict):
                continue

            event_id = str(raw.get("event_id") or raw.get("id") or "").strip()
            if not event_id:
                # Generate fallback ID if missing
                event_id = f"evt_{int(datetime.now().timestamp() * 1000)}_{len(self._memory_store)}"

            user_id = str(raw.get("user_id") or "").strip()
            session_id = str(raw.get("session_id") or "").strip()
            event_name = str(raw.get("event_name") or "unknown_event").strip().lower()
            timestamp = str(raw.get("timestamp") or _utcnow_iso()).strip()
            screen = str(raw.get("screen") or "").strip()
            properties = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
            device = raw.get("device") or {}
            app_version = str(raw.get("app_version") or "1.0.0").strip()

            doc_data = {
                "event_id": event_id,
                "userId": user_id,
                "sessionId": session_id,
                "eventName": event_name,
                "timestampISO": timestamp,
                "screen": screen,
                "propertiesJSON": properties,
                "deviceJSON": device,
                "appVersion": app_version,
                "createdAtISO": _utcnow_iso(),
            }

            # In-memory store fallback & verification
            if any(e.get("event_id") == event_id for e in self._memory_store):
                duplicates += 1
                continue

            self._memory_store.append(doc_data)

            # Persist to Appwrite clickstream_events collection
            try:
                self.proxy.create_document(self.collection, doc_data, document_id=event_id)
                accepted += 1
            except Exception as exc:
                # Store in-memory gracefully if Appwrite collection is not yet provisioned
                logger.debug("ahvi.clickstream_ingest.appwrite_fallback event_id=%s err=%s", event_id, str(exc)[:100])
                accepted += 1

        return {
            "success": True,
            "accepted": accepted,
            "duplicates": duplicates,
            "total_received": len(events),
        }

    def _get_all_events(self) -> List[Dict[str, Any]]:
        try:
            rows = self.proxy.list_documents(self.collection, limit=1000)
            if isinstance(rows, list) and len(rows) > 0:
                combined = {e.get("event_id"): e for e in self._memory_store if isinstance(e, dict)}
                for r in rows:
                    if isinstance(r, dict):
                        eid = r.get("event_id") or r.get("$id")
                        if eid and eid not in combined:
                            combined[eid] = r
                return list(combined.values())
        except Exception:
            pass
        return self._memory_store

    def get_metrics(self) -> Dict[str, Any]:
        """Compute summary clickstream metrics (DAU, sessions, feature usage)."""
        events = self._get_all_events()

        unique_users = set()
        unique_sessions = set()
        feature_counts: Dict[str, int] = {}
        daily_users: Dict[str, set[str]] = {}

        for e in events:
            uid = str(e.get("userId") or e.get("user_id") or "").strip()
            sid = str(e.get("sessionId") or e.get("session_id") or "").strip()
            ename = str(e.get("eventName") or e.get("event_name") or "unknown").strip().lower()
            ts = str(e.get("timestampISO") or e.get("timestamp") or "")

            if uid:
                unique_users.add(uid)
                date_key = ts[:10] if len(ts) >= 10 else "unknown_date"
                if date_key not in daily_users:
                    daily_users[date_key] = set()
                daily_users[date_key].add(uid)

            if sid:
                unique_sessions.add(sid)

            feature_counts[ename] = feature_counts.get(ename, 0) + 1

        sorted_features = [
            {"event_name": name, "count": cnt}
            for name, cnt in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        dau_series = [
            {"date": d, "active_users": len(u_set)}
            for d, u_set in sorted(daily_users.items(), reverse=True)
        ]

        return {
            "total_events": len(events),
            "total_unique_users": len(unique_users),
            "total_unique_sessions": len(unique_sessions),
            "feature_usage": sorted_features,
            "daily_active_users": dau_series,
        }

    def get_funnel_analysis(self) -> Dict[str, Any]:
        """Compute conversion funnel stats across key stages."""
        events = self._get_all_events()

        funnel_stages = [
            ("1_app_open", {"app_open", "screen_view:home"}),
            ("2_wardrobe_view", {"wardrobe_opened", "wardrobe_item_viewed", "screen_view:wardrobe"}),
            ("3_wardrobe_liked", {"wardrobe_item_liked"}),
            ("4_outfit_generated", {"outfit_generated", "styling_request"}),
            ("5_outfit_saved", {"outfit_saved", "board_saved"}),
        ]

        stage_users: Dict[str, set[str]] = {stage[0]: set() for stage in funnel_stages}

        for e in events:
            uid = str(e.get("userId") or e.get("user_id") or e.get("sessionId") or "").strip()
            if not uid:
                continue

            ename = str(e.get("eventName") or e.get("event_name") or "").strip().lower()
            screen = str(e.get("screen") or "").strip().lower()
            comb_key = f"screen_view:{screen}" if ename == "screen_view" else ename

            for stage_name, matched_events in funnel_stages:
                if ename in matched_events or comb_key in matched_events:
                    stage_users[stage_name].add(uid)

        base_count = len(stage_users["1_app_open"]) or max(len(u) for u in stage_users.values()) or 1
        results = []

        for stage_name, u_set in funnel_stages:
            cnt = len(stage_users[stage_name])
            pct = round((cnt / base_count) * 100, 1) if base_count > 0 else 0.0
            results.append({
                "stage": stage_name,
                "unique_users": cnt,
                "conversion_rate_pct": pct,
            })

        return {"funnel": results, "base_cohort_users": base_count}

    def get_user_journeys(self, limit_sessions: int = 20) -> List[Dict[str, Any]]:
        """Compute sessionized user journey sequence flows."""
        events = self._get_all_events()

        sessions: Dict[str, List[Dict[str, Any]]] = {}

        for e in events:
            sid = str(e.get("sessionId") or e.get("session_id") or "").strip()
            if not sid:
                continue
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(e)

        journeys = []

        for sid, s_events in list(sessions.items())[:limit_sessions]:
            # Sort events chronologically by timestampISO
            def _ts(doc):
                val = doc.get("timestampISO") or doc.get("timestamp") or ""
                return val

            s_events.sort(key=_ts)
            sequence = [
                str(e.get("eventName") or e.get("event_name") or "").strip()
                for e in s_events
            ]
            uid = s_events[0].get("userId") or s_events[0].get("user_id") or "anonymous"

            journeys.append({
                "session_id": sid,
                "user_id": uid,
                "event_count": len(sequence),
                "journey_sequence": " ➔ ".join(sequence),
                "started_at": s_events[0].get("timestampISO") or s_events[0].get("timestamp"),
            })

        return journeys
