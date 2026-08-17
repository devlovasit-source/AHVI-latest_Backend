from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.appwrite_proxy import AppwriteProxy


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


NOTIFICATION_PREFERENCE_CATEGORIES = {
    "medi",
    "calendar",
    "style",
}


def _hash_id(prefix: str, raw: str, *, length: int = 32) -> str:
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}_{digest[: max(8, min(length, 48))]}"


class NotificationStore:
    """
    Stores:
    - device tokens in Appwrite collection: notification_devices
    - reminders in Appwrite collection: notification_reminders

    This is intentionally simple (MVP). For scale, move due-reminder queries to a DB index.
    """

    def __init__(self) -> None:
        self._appwrite = AppwriteProxy()
        self.devices_resource = (
            os.getenv("APPWRITE_COLLECTION_NOTIFICATION_DEVICES", "")
            or os.getenv("APPWRITE_RESOURCE_NOTIFICATION_DEVICES", "")
            or "notification_devices"
        )
        self.reminders_resource = (
            os.getenv("APPWRITE_COLLECTION_NOTIFICATION_REMINDERS", "")
            or os.getenv("APPWRITE_RESOURCE_NOTIFICATION_REMINDERS", "")
            or "notification_reminders"
        )
        self.preferences_resource = (
            os.getenv("APPWRITE_COLLECTION_NOTIFICATION_PREFERENCES", "")
            or os.getenv("APPWRITE_RESOURCE_NOTIFICATION_PREFERENCES", "")
            or "notification_preferences"
        )
        # Claim rows are how dispatch prevents double-sends. One claim row per
        # reminder occurrence (same doc id as the reminder itself), created
        # atomically via Appwrite's unique-id create - see try_claim_reminder().
        self.dispatch_claims_resource = (
            os.getenv("APPWRITE_COLLECTION_NOTIFICATION_DISPATCH_CLAIMS", "")
            or os.getenv("APPWRITE_RESOURCE_NOTIFICATION_DISPATCH_CLAIMS", "")
            or "notification_dispatch_claims"
        )
        self.max_scan = max(
            50, int(os.getenv("NOTIFICATION_REMINDER_SCAN_LIMIT", "500"))
        )
        self._claim_ttl_seconds = max(
            30, int(os.getenv("NOTIFICATION_CLAIM_TTL_SECONDS", "300"))
        )

    def get_notification_preference(
        self,
        *,
        user_id: str,
        category: str,
    ) -> bool:
        uid = _safe_text(user_id)
        category = _safe_text(category).lower()

        if not uid or category not in NOTIFICATION_PREFERENCE_CATEGORIES:
            return True

        try:
            rows = self._appwrite.list_documents(
                self.preferences_resource,
                user_id=uid,
                limit=100,
            )
        except Exception:
            return True

        for row in rows or []:
            if not isinstance(row, dict):
                continue

            if _safe_text(row.get("category")).lower() == category:
                return bool(row.get("enabled", True))

        return True

    def get_notification_preference_for_dispatch(
        self,
        *,
        user_id: str,
        category: str,
    ) -> tuple[bool, str]:
        """
        Tri-state preference read for the dispatch path only (GET /preferences
        keeps using get_notification_preference() above, which is fine to
        fail-open for a settings screen).

        Distinguishes "no preference saved yet" from "the lookup itself
        errored", because those must NOT be treated the same when deciding
        whether to actually push a notification:

        Returns (should_send, status) where status is one of:
          - "no_record"       no preference row exists -> defaults to enabled
          - "enabled"         explicit preference found, enabled=True
          - "disabled"        explicit preference found, enabled=False
          - "invalid_category" category isn't medi/calendar/style -> FAILS
                              CLOSED (should_send=False). A producer is
                              sending an unrecognized category; that needs
                              fixing, not silently allowing the send.
          - "invalid_user"   no user_id on the reminder -> FAILS CLOSED
          - "lookup_failed"  Appwrite call raised -> should_send=False; the
                              caller must NOT send and should retry later,
                              not silently treat this as enabled.
        """
        uid = _safe_text(user_id)
        category = _safe_text(category).lower()

        if category not in NOTIFICATION_PREFERENCE_CATEGORIES:
            # Fail CLOSED: an unrecognized category must never be assumed
            # enabled. If this fires in practice, some producer is passing a
            # source/category outside medi/calendar/style and needs fixing -
            # see get_notification_preference_for_dispatch's docstring.
            return False, "invalid_category"

        if not uid:
            return False, "invalid_user"

        try:
            rows = self._appwrite.list_documents(
                self.preferences_resource,
                user_id=uid,
                limit=100,
            )
        except Exception:
            return False, "lookup_failed"

        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if _safe_text(row.get("category")).lower() == category:
                enabled = bool(row.get("enabled", True))
                return enabled, ("enabled" if enabled else "disabled")

        return True, "no_record"

    def set_notification_preference(
        self,
        *,
        user_id: str,
        category: str,
        enabled: bool,
    ) -> bool:
        uid = _safe_text(user_id)
        category = _safe_text(category).lower()

        if not uid or category not in NOTIFICATION_PREFERENCE_CATEGORIES:
            return False

        now = _utcnow().isoformat()

        doc_id = _hash_id(
            "pref",
            f"{uid}|{category}",
            length=28,
        )

        data = {
            "userId": uid,
            "category": category,
            "enabled": bool(enabled),
            "updatedAtISO": now,
        }

        try:
            self._appwrite.update_document(
                self.preferences_resource,
                doc_id,
                data,
            )
            return True
        except Exception:
            pass

        data["createdAtISO"] = now

        try:
            self._appwrite.create_document(
                self.preferences_resource,
                data,
                document_id=doc_id,
            )
            return True
        except Exception:
            return False

    # -------------------------
    # Devices
    # -------------------------
    def upsert_device(self, *, user_id: str, platform: str, token: str) -> str | None:
        uid = _safe_text(user_id)
        tok = _safe_text(token)
        plat = _safe_text(platform).lower() or "unknown"
        if not uid or not tok:
            return None

        # De-dupe by the real registration identity. The old token-only id could
        # still leave multiple rows when debug builds generated fresh dev ids;
        # this keeps repeated launches for the same device as updates.
        doc_id = _hash_id("dev", f"{uid}|{plat}|{tok}", length=28)
        data = {
            "userId": uid,
            "platform": plat,
            "token": tok,
            "updatedAtISO": _utcnow().isoformat(),
        }
        try:
            self._appwrite.update_document(self.devices_resource, doc_id, data)
        except Exception:
            try:
                self._appwrite.create_document(
                    self.devices_resource, data, document_id=doc_id
                )
            except Exception:
                return None
        return doc_id

    def delete_device(self, *, token: str) -> bool:
        tok = _safe_text(token)
        if not tok:
            return False

        # Registration keys a device by hash(userId|platform|token), but this
        # method only ever received `token` and was hashing that alone - a
        # different id that essentially never matched, so unregister silently
        # left the real row behind (stale FCM tokens keep getting sent to).
        # Look the row up by its actual `token` field instead of guessing an
        # id - this also cleans up any rows already orphaned by that bug.
        try:
            matches = self._appwrite.find_by_attribute(
                self.devices_resource, "token", tok, limit=10
            )
        except Exception:
            return False

        deleted_any = False
        for doc in matches or []:
            if not isinstance(doc, dict):
                continue
            doc_id = str(doc.get("$id") or doc.get("id") or "")
            if not doc_id:
                continue
            try:
                self._appwrite.delete_document(self.devices_resource, doc_id)
                deleted_any = True
            except Exception:
                continue

        return deleted_any

    def list_devices(self, *, user_id: str) -> List[Dict[str, Any]]:
        uid = _safe_text(user_id)
        if not uid:
            return []
        try:
            rows = self._appwrite.list_documents(
                self.devices_resource, user_id=uid, limit=200
            )
            return [r for r in rows if isinstance(r, dict)]
        except Exception:
            return []

    # -------------------------
    # Reminders
    # -------------------------
    def schedule_reminders(
        self,
        *,
        user_id: str,
        event_id: str,
        reminders: List[Dict[str, Any]],
        source: str = "calendar",
    ) -> Dict[str, Any]:
        uid = _safe_text(user_id)
        eid = _safe_text(event_id) or "event"
        if not uid:
            return {"success": False, "scheduled": 0}

        scheduled = 0
        for r in reminders or []:
            if not isinstance(r, dict):
                continue
            send_at = _safe_text(r.get("sendAtISO") or r.get("send_at") or "")
            message = _safe_text(r.get("message") or "")
            if not send_at or not message:
                continue

            offset_minutes = int(r.get("offsetMinutes") or 0)

            # Deliberately excludes message text: the occurrence identity is
            # "this user's this category's reminder for this event at this
            # send time/offset" - a wording change must update that same
            # occurrence, not create a second reminder for it.
            doc_id = _hash_id(
                "rem",
                f"{uid}|{_safe_text(source)}|{eid}|{send_at}|{offset_minutes}",
                length=32,
            )
            data = {
                "userId": uid,
                "eventId": eid,
                # Default scheduled (not pending) so the dispatch task, which
                # only picks status=="scheduled", can actually send them.
                # Confirmed reminders must use the state scanned by dispatch.
                "status": "scheduled",
                "priority": _safe_text(r.get("priority") or "normal"),
                "toneProfile": _safe_text(r.get("toneProfile") or ""),
                "offsetMinutes": offset_minutes,
                "message": message,
                "sendAtISO": send_at,
                "source": _safe_text(source),
                "lastError": _safe_text(r.get("lastError") or ""),
                "updatedAtISO": _utcnow().isoformat(),
            }
            try:
                self._appwrite.update_document(self.reminders_resource, doc_id, data)
                scheduled += 1
            except Exception:
                try:
                    self._appwrite.create_document(
                        self.reminders_resource, data, document_id=doc_id
                    )
                    scheduled += 1
                except Exception:
                    continue

        return {"success": scheduled > 0, "scheduled": scheduled}

    def list_due_reminders(
        self, *, now: Optional[datetime] = None, window_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        now_dt = now or _utcnow()
        cutoff = now_dt.timestamp() + float(max(5, int(window_seconds)))

        # MVP: scan recent scheduled reminders per user on demand.
        # For scale, use a real indexed query (or store reminders in Redis sorted sets).
        try:
            rows = self._appwrite.list_documents(
                self.reminders_resource, limit=self.max_scan
            )
        except Exception:
            return []

        due: List[Dict[str, Any]] = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            if str(r.get("status") or "").lower() != "scheduled":
                continue
            send_at = _safe_text(r.get("sendAtISO") or "")
            try:
                send_dt = datetime.fromisoformat(send_at.replace("Z", "+00:00"))
            except Exception:
                continue
            if send_dt.timestamp() <= cutoff:
                due.append(r)
        return due

    def mark_reminder(
        self, *, reminder_doc_id: str, status: str, error: str | None = None
    ) -> None:
        rid = _safe_text(reminder_doc_id)
        if not rid:
            return
        patch = {"status": _safe_text(status), "updatedAtISO": _utcnow().isoformat()}
        if error:
            patch["lastError"] = _safe_text(error)[:600]
        try:
            self._appwrite.update_document(self.reminders_resource, rid, patch)
        except Exception:
            return

    # -------------------------
    # Dispatch claims (duplicate prevention)
    # -------------------------
    #
    # Appwrite's document update is a blind overwrite - there is no native
    # "update only if status is still X" conditional operation to build a
    # safe claim on top of. The one operation Appwrite DOES guarantee
    # atomically is creating a document with a fixed id: if two callers race
    # to create the same id, only one succeeds and the other gets a 409.
    #
    # So a claim is its own small document, using the reminder's own
    # deterministic doc id as the claim's doc id too. That doc id is the
    # "deterministic occurrence/dedupe key" - at most one claim can ever
    # exist per reminder occurrence, so at most one dispatch run can ever
    # win the create and be allowed to send.

    def try_claim_reminder(self, *, reminder_doc_id: str) -> Dict[str, Any]:
        """
        Attempts to atomically claim a reminder occurrence for sending.

        Returns {"claimed": bool, "reason": str} where reason is one of:
          - "new_claim"          no claim existed; caller may proceed
          - "reclaimed_expired"  a stale ("processing" past its TTL) claim
                                  was taken over; caller may proceed
          - "already_sent"       this occurrence was already sent - caller
                                  must NOT send again
          - "already_processing" another in-flight dispatch run currently
                                  owns this occurrence - caller must NOT send
          - "already_suppressed" / "already_failed"  a prior run already
                                  reached a terminal state for this occurrence
          - "error"               claim attempt failed for a reason other
                                  than "already exists" (network/Appwrite
                                  error) - caller must NOT send, safe to
                                  retry next dispatch cycle
        """
        rid = _safe_text(reminder_doc_id)
        if not rid:
            return {"claimed": False, "reason": "error"}
        return self._attempt_claim(rid, allow_takeover=True)

    def _attempt_claim(self, rid: str, *, allow_takeover: bool) -> Dict[str, Any]:
        now = _utcnow()
        now_iso = now.isoformat()
        expires_at = now.timestamp() + self._claim_ttl_seconds

        try:
            self._appwrite.create_document(
                self.dispatch_claims_resource,
                {
                    "reminderId": rid,
                    "status": "processing",
                    "claimedAtISO": now_iso,
                    "expiresAt": expires_at,
                    "updatedAtISO": now_iso,
                },
                document_id=rid,
            )
            reason = "new_claim" if allow_takeover else "reclaimed_expired"
            return {"claimed": True, "reason": reason}
        except Exception as exc:
            if "(409)" not in str(exc):
                # A real error (network/auth/5xx), not "already exists".
                # Don't claim; the reminder stays "scheduled" and will be
                # retried on the next dispatch cycle.
                return {"claimed": False, "reason": "error"}

        # A claim already exists - inspect it before deciding anything.
        try:
            existing = self._appwrite.get_document(self.dispatch_claims_resource, rid)
        except Exception:
            return {"claimed": False, "reason": "error"}

        existing_status = _safe_text((existing or {}).get("status")).lower()

        if existing_status == "sent":
            return {"claimed": False, "reason": "already_sent"}

        if existing_status == "processing":
            try:
                existing_expires = float((existing or {}).get("expiresAt") or 0)
            except Exception:
                existing_expires = 0.0

            if existing_expires > now.timestamp():
                return {"claimed": False, "reason": "already_processing"}

            if not allow_takeover:
                # We already tried one takeover in this call and lost that
                # race too - stop here instead of looping forever.
                return {"claimed": False, "reason": "already_processing"}

            # Stale claim past its TTL (e.g. the process that claimed it
            # crashed mid-send). Appwrite's update is a blind overwrite, so
            # two workers racing here could BOTH pass the expiry check above
            # and both "successfully" update - both would then think they
            # won the claim. To keep the exclusivity guarantee, take over by
            # deleting the stale row and re-attempting a fresh create: the
            # create step is the one operation Appwrite guarantees is
            # exclusive (only one caller can ever create a given document
            # id), so only one of two racing takeovers can win. The other
            # lands back here via the 409 branch above and correctly backs
            # off as "already_processing".
            try:
                self._appwrite.delete_document(self.dispatch_claims_resource, rid)
            except Exception:
                pass  # already gone, or someone else deleted it - fine

            return self._attempt_claim(rid, allow_takeover=False)

        # suppressed/failed/anything else: a prior run already reached a
        # terminal state for this occurrence - do not resend automatically.
        return {"claimed": False, "reason": f"already_{existing_status or 'handled'}"}

    def finalize_claim(self, *, reminder_doc_id: str, status: str) -> None:
        """Marks a claim as sent/suppressed/failed once dispatch is done with it."""
        rid = _safe_text(reminder_doc_id)
        if not rid:
            return
        try:
            self._appwrite.update_document(
                self.dispatch_claims_resource,
                rid,
                {"status": _safe_text(status), "updatedAtISO": _utcnow().isoformat()},
            )
        except Exception:
            return

    def release_claim(self, *, reminder_doc_id: str) -> None:
        """
        Deletes a claim so the occurrence can be retried on the next dispatch
        cycle. Used only for transient failures (e.g. preference_lookup_failed)
        - never after a real user decision (disabled preference) or a real
        send outcome, which should stay as a terminal claim state instead.
        """
        rid = _safe_text(reminder_doc_id)
        if not rid:
            return
        try:
            self._appwrite.delete_document(self.dispatch_claims_resource, rid)
        except Exception:
            return


notification_store = NotificationStore()