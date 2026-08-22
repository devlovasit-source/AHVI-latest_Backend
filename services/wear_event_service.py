"""Canonical Wear Event Service.

Source of truth for actual item/outfit wear events. Handles idempotency,
timezone local-date resolution, durable matching reminder completion,
and committed wear summary calculation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional
import zoneinfo

from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.wear_event_service")


def _proxy() -> AppwriteProxy:
    return AppwriteProxy()


def _utcnow() -> datetime:
    return datetime.now(dt_timezone.utc)


def _resolve_local_date(occurred_at_iso: str, tz_name: str) -> tuple[str, str, str]:
    """Given an ISO timestamp string and a timezone name, return (occurred_at_iso, local_date, tz_name)."""
    tz_str = (tz_name or "").strip() or "Asia/Kolkata"
    try:
        tz = zoneinfo.ZoneInfo(tz_str)
    except Exception:
        tz = zoneinfo.ZoneInfo("Asia/Kolkata")
        tz_str = "Asia/Kolkata"

    if occurred_at_iso and occurred_at_iso.strip():
        raw = occurred_at_iso.strip()
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            dt = _utcnow()
    else:
        dt = _utcnow()

    local_dt = dt.astimezone(tz)
    return dt.isoformat(), local_dt.strftime("%Y-%m-%d"), tz_str


def build_idempotency_key(
    user_id: str,
    source: str,
    entity_type: str,
    entity_id: str,
    local_date: str,
) -> str:
    """Format: {user_id}:{source}:{entity_type}:{entity_id}:{local_date}"""
    u = str(user_id or "").strip()
    src = str(source or "").strip().lower() or "wardrobe_item"
    etype = str(entity_type or "").strip().lower() or "item"
    eid = str(entity_id or "").strip()
    ldate = str(local_date or "").strip()
    return f"{u}:{src}:{etype}:{eid}:{ldate}"


class WearEventService:
    def __init__(self) -> None:
        self.proxy = _proxy()
        self.resource = "wear_events"

    def record_wear(
        self,
        *,
        user_id: str,
        item_ids: List[str],
        source: str = "wardrobe_item",
        entity_type: str = "item",
        entity_id: str = "",
        outfit_id: Optional[str] = None,
        board_id: Optional[str] = None,
        calendar_event_id: Optional[str] = None,
        occurred_at: str = "",
        timezone: str = "Asia/Kolkata",
    ) -> Dict[str, Any]:
        uid = str(user_id or "").strip()
        clean_item_ids = [str(x).strip() for x in (item_ids or []) if str(x or "").strip()]
        if not uid or not clean_item_ids:
            return {
                "recorded": False,
                "duplicate": False,
                "wear_event_id": "",
                "summary": {"total_wears": 0, "last_worn_at": "", "wears_last_7_days": 0, "wears_last_30_days": 0},
            }

        eid = entity_id.strip() if entity_id else clean_item_ids[0]
        iso_time, local_date, tz_used = _resolve_local_date(occurred_at, timezone)
        idem_key = build_idempotency_key(uid, source, entity_type, eid, local_date)

        # 1. Idempotency Check (persist or fetch existing)
        existing = self.find_by_idempotency_key(idem_key)
        duplicate = False
        event_doc: Dict[str, Any] = {}

        if existing:
            duplicate = True
            event_doc = existing
            logger.info("ahvi.wear_event.duplicate_detected key=%s id=%s", idem_key, existing.get("$id"))
        else:
            doc_data = {
                "userId": uid,
                "occurredAtISO": iso_time,
                "localDate": local_date,
                "timezone": tz_used,
                "itemIds": clean_item_ids,
                "source": str(source or "wardrobe_item").lower(),
                "entityType": str(entity_type or "item").lower(),
                "entityId": eid,
                "outfitId": outfit_id or "",
                "boardId": board_id or "",
                "calendarEventId": calendar_event_id or "",
                "idempotencyKey": idem_key,
                "revokedAtISO": None,
                "revokedReason": None,
                "createdAtISO": _utcnow().isoformat(),
            }
            try:
                created = self.proxy.create_document(self.resource, doc_data, document_id=idem_key.replace(":", "_"))
                event_doc = created if isinstance(created, dict) else doc_data
            except Exception as exc:
                # Fallback search if index conflict occurred concurrently
                logger.warning("ahvi.wear_event.create_failed error=%s retry_search=true", str(exc)[:140])
                existing_retry = self.find_by_idempotency_key(idem_key)
                if existing_retry:
                    duplicate = True
                    event_doc = existing_retry
                else:
                    raise exc

        event_id = str(event_doc.get("$id") or event_doc.get("id") or idem_key.replace(":", "_"))

        # 2. Durable Side Effect: Reconcile & complete matching wear_item reminders
        self.reconcile_matching_wear_reminders(user_id=uid, item_ids=clean_item_ids)

        # 3. Compute fresh summary from committed non-revoked wear_events (Truth)
        primary_item_id = clean_item_ids[0]
        summary = self.get_committed_item_summary(user_id=uid, item_id=primary_item_id)

        # 4. Trigger eventual background projections (StyleMemory / OutfitHistory)
        self._trigger_eventual_projections(user_id=uid, item_ids=clean_item_ids, worn_at=iso_time, board_id=board_id)

        return {
            "recorded": not duplicate,
            "duplicate": duplicate,
            "wear_event_id": event_id,
            "summary": summary,
        }

    def find_by_idempotency_key(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            # Query by document_id or idempotencyKey filter
            doc_id = key.replace(":", "_")
            doc = self.proxy.get_document(self.resource, doc_id)
            if isinstance(doc, dict) and doc.get("idempotencyKey") == key:
                return doc
        except Exception:
            pass
        return None

    def reconcile_matching_wear_reminders(self, *, user_id: str, item_ids: List[str]) -> int:
        """Durable synchronous completion of matching future wear_item reminders."""
        try:
            from services.notification_store import NotificationStore

            store = NotificationStore()
            rows = store._appwrite.list_documents(store.reminders_resource, user_id=user_id, limit=200)
            completed_count = 0
            item_set = set(item_ids)

            for r in rows if isinstance(rows, list) else []:
                if not isinstance(r, dict):
                    continue
                status = str(r.get("status") or "").lower()
                if status != "scheduled":
                    continue
                # Ensure source or subtype matches wear_item / style
                source = str(r.get("source") or "").lower()
                entity_id = str(r.get("eventId") or r.get("entityId") or "").strip()

                if source in ("style", "wardrobe_item", "wear_item") or entity_id in item_set:
                    doc_id = str(r.get("$id") or r.get("id") or "")
                    if doc_id:
                        store.mark_reminder(reminder_doc_id=doc_id, status="completed")
                        completed_count += 1
            return completed_count
        except Exception as exc:
            logger.warning("ahvi.reconcile_reminders_failed user_id=%s err=%s", user_id, str(exc)[:140])
            return 0

    def get_committed_item_summary(self, *, user_id: str, item_id: str) -> Dict[str, Any]:
        """Compute exact stats strictly from non-revoked wear_events."""
        try:
            rows = self.proxy.list_documents(self.resource, user_id=user_id, limit=300)
        except Exception:
            rows = []

        valid_events = []
        now_dt = _utcnow()

        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            if r.get("revokedAtISO"):
                continue
            iids = r.get("itemIds") or []
            if isinstance(iids, list) and item_id in [str(x).strip() for x in iids]:
                valid_events.append(r)

        if not valid_events:
            return {
                "total_wears": 0,
                "last_worn_at": None,
                "days_since_last_worn": None,
                "wears_last_7_days": 0,
                "wears_last_30_days": 0,
            }

        # Sort by occurredAtISO descending
        def _get_ts(doc):
            val = doc.get("occurredAtISO") or ""
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0

        valid_events.sort(key=_get_ts, reverse=True)

        latest_ts = _get_ts(valid_events[0])
        last_worn_iso = valid_events[0].get("occurredAtISO")
        days_since = max(0, int((now_dt.timestamp() - latest_ts) // 86400)) if latest_ts > 0 else None

        ts_7d = now_dt.timestamp() - (7 * 86400)
        ts_30d = now_dt.timestamp() - (30 * 86400)

        wears_7d = sum(1 for e in valid_events if _get_ts(e) >= ts_7d)
        wears_30d = sum(1 for e in valid_events if _get_ts(e) >= ts_30d)

        return {
            "total_wears": len(valid_events),
            "last_worn_at": last_worn_iso,
            "days_since_last_worn": days_since,
            "wears_last_7_days": wears_7d,
            "wears_last_30_days": wears_30d,
        }

    def revoke_wear(self, *, user_id: str, wear_event_id: str, reason: str = "user_correction") -> bool:
        """Mark a wear event as revoked (undo). Does NOT restore completed reminders."""
        try:
            doc = self.proxy.get_document(self.resource, wear_event_id)
            if not isinstance(doc, dict) or doc.get("userId") != user_id:
                return False
            self.proxy.update_document(
                self.resource,
                wear_event_id,
                {
                    "revokedAtISO": _utcnow().isoformat(),
                    "revokedReason": str(reason or "user_correction"),
                },
            )
            return True
        except Exception as exc:
            logger.warning("ahvi.revoke_wear_failed id=%s err=%s", wear_event_id, str(exc)[:140])
            return False

    def _trigger_eventual_projections(
        self, user_id: str, item_ids: List[str], worn_at: str, board_id: Optional[str]
    ) -> None:
        """Best-effort update of legacy outfit_history & StyleMemoryService."""
        try:
            from services.style_memory_service import record_wear as legacy_record_wear

            legacy_record_wear(
                user_id=user_id,
                item_ids=item_ids,
                board_id=board_id or "",
                worn_at=worn_at,
            )
        except Exception as exc:
            logger.warning("ahvi.eventual_projection_failed err=%s", str(exc)[:140])
