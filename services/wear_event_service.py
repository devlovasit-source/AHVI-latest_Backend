"""Canonical wear events — the single source of truth for "this user wore
this item on this day".

wear_events is an append-mostly log (one row per user+item+local-date,
status=active/revoked); the pre-existing outfit_history aggregate in
services/style_memory_service.py remains the read-optimized projection
Style This/DailyWear/the scorer already consume. This service is the only
writer that is allowed to advance that projection for a wear — both
POST /api/wardrobe/items/{item_id}/wear and the legacy
POST /api/style/wear-today delegate here so there is exactly one wear
implementation.

Idempotency: the wear_events document id is a deterministic hash of
(user_id, item_id, local_date[, source]) — see deterministic_appwrite_id.
A retried/duplicated request for the same logical wear resolves to the same
document id, so Appwrite's own "document already exists" (409) response is
the idempotency check: on 409 we fetch and return the existing event with
newly_created=False, and callers must skip re-driving the projection/style
memory for a non-newly-created event.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ahvi.wear_event_service")

WEAR_EVENTS_RESOURCE = "wear_events"


def deterministic_appwrite_id(*parts: str) -> str:
    """Stable <=36-char Appwrite document id from the given parts. Never
    derived by mangling an idempotency key's characters (e.g. replacing
    ':') — hashed, so any key shape is safe and length is fixed."""
    raw = ":".join(str(p or "") for p in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:36]


def _proxy():
    from services.appwrite_proxy import AppwriteProxy

    return AppwriteProxy()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_date(occurred_at_iso: str) -> str:
    """UTC calendar date (YYYY-MM-DD) of occurred_at_iso. No client timezone
    field exists on this contract (same constraint style_memory_service
    already documents), so UTC date is the narrowest safe boundary."""
    text = str(occurred_at_iso or "").strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date().isoformat()


def record_wear(
    *,
    user_id: str,
    item_id: str,
    occurred_at_iso: str = "",
    source: str = "wardrobe",
    entity_type: str = "wardrobe_item",
    entity_id: str = "",
    board_id: str = "",
    occasion: str = "",
) -> Dict[str, Any]:
    """Commit one canonical wear event. Idempotent per (user, item, local
    calendar day): a retried call for the same day returns the same event
    with newly_created=False and does NOT re-trigger downstream projection
    updates — the caller is expected to gate on newly_created.

    Returns {"event": {...}, "newly_created": bool}.
    """
    from services.appwrite_proxy import AppwriteProxyError

    uid = str(user_id or "").strip()
    iid = str(item_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    if not iid:
        raise ValueError("item_id is required")

    occurred_at = str(occurred_at_iso or "").strip() or _now_iso()
    local_date = _local_date(occurred_at)
    entity_id = str(entity_id or "").strip() or iid
    idempotency_key = f"{uid}:wear:{iid}:{local_date}"
    event_id = deterministic_appwrite_id(uid, "wear", iid, local_date)

    data = {
        "userId": uid,
        "itemId": iid,
        "localDate": local_date,
        "occurredAtISO": occurred_at,
        "source": str(source or "wardrobe"),
        "entityType": str(entity_type or "wardrobe_item"),
        "entityId": entity_id,
        "status": "active",
        "idempotencyKey": idempotency_key,
        "createdAtISO": _now_iso(),
        "revokedAtISO": "",
        "boardId": str(board_id or ""),
        "occasion": str(occasion or ""),
    }

    proxy = _proxy()
    try:
        event = proxy.create_document(WEAR_EVENTS_RESOURCE, data, document_id=event_id)
        newly_created = True
        logger.info(
            "AHVI_WEAR_EVENT_CREATED user_id=%s item_id=%s local_date=%s event_id=%s",
            uid, iid, local_date, event_id,
        )
    except AppwriteProxyError as exc:
        if exc.status_code != 409:
            raise
        event = proxy.get_document(WEAR_EVENTS_RESOURCE, event_id)
        newly_created = False
        logger.info(
            "AHVI_WEAR_EVENT_DUPLICATE user_id=%s item_id=%s local_date=%s event_id=%s",
            uid, iid, local_date, event_id,
        )

    if newly_created:
        # Bridge into the existing, unmodified style-memory projection —
        # never trigger this a second time for a duplicate/retried request.
        try:
            from services.style_memory_service import record_wear as _record_legacy_wear

            _record_legacy_wear(
                user_id=uid,
                item_ids=[iid],
                board_id=board_id,
                occasion=occasion,
                worn_at=occurred_at,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "AHVI_WEAR_EVENT_PROJECTION_FAILED user_id=%s item_id=%s", uid, iid
            )

        # Reconcile wear reminders: a newly-committed wear satisfies at most
        # its own item's reminders, never another item's. Retryable/logged
        # only — must never cause the wear itself to fail or re-run.
        try:
            from services.wardrobe_reminder_service import complete_reminders_for_item

            complete_reminders_for_item(user_id=uid, item_id=iid)
        except Exception:  # noqa: BLE001
            logger.warning(
                "AHVI_WEAR_REMINDER_RECONCILE_FAILED user_id=%s item_id=%s", uid, iid
            )

        # Keep the wardrobe item's own display-only `worn` counter in sync.
        # wear_events/outfit_history are the source of truth; this is purely
        # so the wardrobe grid/AI-insight copy (which reads the item's own
        # document, never wear_events) doesn't go stale after a relaunch —
        # it previously worked because the old direct-Appwrite-write path
        # patched this same field on every wear. Never blocks the wear.
        try:
            from services.wardrobe_persistence_service import _fetch_document, _patch_document

            total_wears = get_wear_history(user_id=uid, item_id=iid)["total_wears"]
            _, collection_id, database_id = _fetch_document(iid)
            _patch_document(
                iid, {"worn": total_wears}, collection_id=collection_id, database_id=database_id
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "AHVI_WEAR_ITEM_WORN_SYNC_FAILED user_id=%s item_id=%s", uid, iid
            )

    return {"event": event, "newly_created": newly_created}


def _parse_iso(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def get_wear_history(*, user_id: str, item_id: str) -> Dict[str, Any]:
    """Owner-scoped wear history for one item: aggregates + individual
    active events, newest first. Revoked events are excluded from every
    aggregate and from the returned events list."""
    uid = str(user_id or "").strip()
    iid = str(item_id or "").strip()
    empty = {
        "total_wears": 0,
        "last_worn_at": None,
        "days_since_last_worn": None,
        "wears_last_7_days": 0,
        "wears_last_30_days": 0,
        "monthly_summary": {},
        "events": [],
    }
    if not uid or not iid:
        return empty

    try:
        rows = _proxy().list_documents(WEAR_EVENTS_RESOURCE, user_id=uid, limit=500)
    except Exception:
        return empty
    if not isinstance(rows, list):
        return empty

    active: List[Dict[str, Any]] = [
        r
        for r in rows
        if isinstance(r, dict)
        and str(r.get("itemId") or "").strip() == iid
        and str(r.get("status") or "active").strip().lower() != "revoked"
    ]
    if not active:
        return empty

    def _occurred(r: Dict[str, Any]) -> datetime:
        return _parse_iso(r.get("occurredAtISO")) or datetime.min.replace(tzinfo=timezone.utc)

    active.sort(key=_occurred, reverse=True)

    now = datetime.now(timezone.utc)
    monthly: Dict[str, int] = {}
    last_7 = 0
    last_30 = 0
    for r in active:
        dt = _occurred(r)
        month_key = str(r.get("localDate") or "")[:7]
        if month_key:
            monthly[month_key] = monthly.get(month_key, 0) + 1
        age_days = (now - dt).total_seconds() / 86400.0
        if age_days <= 7:
            last_7 += 1
        if age_days <= 30:
            last_30 += 1

    last_worn_dt = _occurred(active[0])
    days_since = int((now - last_worn_dt).total_seconds() // 86400)

    return {
        "total_wears": len(active),
        "last_worn_at": active[0].get("occurredAtISO"),
        "days_since_last_worn": days_since,
        "wears_last_7_days": last_7,
        "wears_last_30_days": last_30,
        "monthly_summary": monthly,
        "events": [
            {
                "occurred_at": r.get("occurredAtISO"),
                "local_date": r.get("localDate"),
                "source": r.get("source"),
                "board_id": r.get("boardId"),
                "occasion": r.get("occasion"),
            }
            for r in active
        ],
    }
