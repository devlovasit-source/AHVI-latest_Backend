from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _hash_id(prefix: str, raw: str, *, length: int = 32) -> str:
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}_{digest[: max(8, min(length, 48))]}"


class ReminderStoreError(RuntimeError):
    """A reminder persistence failure that is not an idempotency conflict."""


class ReminderConflictError(ReminderStoreError):
    """A deterministic Appwrite document already exists (HTTP 409)."""


class ReminderSchemaError(ReminderStoreError):
    """The durable reminder attributes or indexes are not provisioned."""


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
        self.max_scan = max(
            50, int(os.getenv("NOTIFICATION_REMINDER_SCAN_LIMIT", "500"))
        )

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
        doc_id = self.device_record_id(user_id=uid, platform=plat, token=tok)
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

    def device_record_id(self, *, user_id: str, platform: str, token: str) -> str:
        return _hash_id(
            "dev",
            f"{_safe_text(user_id)}|{_safe_text(platform).lower() or 'unknown'}|{_safe_text(token)}",
            length=28,
        )

    def delete_device(
        self, *, token: str, user_id: str = "", platform: str = ""
    ) -> bool:
        tok = _safe_text(token)
        if not tok:
            return False
        uid = _safe_text(user_id)
        plat = _safe_text(platform)
        try:
            if uid and plat:
                self._appwrite.delete_document(
                    self.devices_resource,
                    self.device_record_id(user_id=uid, platform=plat, token=tok),
                )
                return True
            # Existing callers only provide a token. This is still indexed,
            # but new callers should provide the full registration identity.
            rows = self._appwrite.find_by_attribute(
                self.devices_resource, "token", tok, limit=20
            )
            for row in rows:
                if _safe_text(row.get("token")) != tok:
                    continue
                doc_id = _safe_text(row.get("$id") or row.get("id"))
                if doc_id:
                    self._appwrite.delete_document(self.devices_resource, doc_id)
                    return True
            return False
        except AppwriteProxyError:
            return False
        except Exception:
            return False

    # -------------------------
    # Durable medicine reminder records
    # -------------------------
    def reminder_record_id(self, notification_key: str) -> str:
        return _hash_id("medrem", _safe_text(notification_key), length=36)

    def claim_record_id(self, notification_key: str, attempt: int) -> str:
        return _hash_id("medclaim", f"{_safe_text(notification_key)}|{int(attempt)}", length=36)

    @staticmethod
    def _query(method: str, attribute: str, *values: str) -> Dict[str, Any]:
        return {"method": method, "attribute": attribute, "values": list(values)}

    def create_reminder_record(self, doc_id: str, data: Dict[str, Any]) -> None:
        try:
            self._appwrite.create_document(self.reminders_resource, dict(data), document_id=doc_id)
        except AppwriteProxyError as exc:
            if exc.status_code == 409:
                raise ReminderConflictError("reminder record already exists") from exc
            raise ReminderStoreError("reminder record create failed") from exc
        except Exception as exc:
            raise ReminderStoreError("reminder record create failed") from exc

    def get_reminder_record(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            row = self._appwrite.get_document(self.reminders_resource, _safe_text(doc_id))
            return row if isinstance(row, dict) else None
        except AppwriteProxyError as exc:
            if exc.status_code == 404:
                return None
            raise ReminderStoreError("reminder record read failed") from exc
        except Exception as exc:
            raise ReminderStoreError("reminder record read failed") from exc

    def update_reminder_record(self, doc_id: str, patch: Dict[str, Any]) -> None:
        try:
            self._appwrite.update_document(
                self.reminders_resource,
                _safe_text(doc_id),
                {**patch, "updatedAtISO": _utcnow().isoformat()},
            )
        except AppwriteProxyError as exc:
            raise ReminderStoreError("reminder record update failed") from exc
        except Exception as exc:
            raise ReminderStoreError("reminder record update failed") from exc

    def create_claim_marker(
        self, *, user_id: str, notification_key: str, attempt: int, claimed_at_iso: str
    ) -> None:
        try:
            self._appwrite.create_document(
                self.reminders_resource,
                {
                    "userId": _safe_text(user_id),
                    "eventId": _safe_text(notification_key),
                    "notificationKey": _safe_text(notification_key),
                    "kind": "claim",
                    "status": "claimed",
                    "source": "medicine",
                    "attemptCount": int(attempt),
                    "sendAtISO": _safe_text(claimed_at_iso),
                    "message": "",
                    "lastError": "",
                    "updatedAtISO": _safe_text(claimed_at_iso),
                },
                document_id=self.claim_record_id(notification_key, attempt),
            )
        except AppwriteProxyError as exc:
            if exc.status_code == 409:
                raise ReminderConflictError("claim marker already exists") from exc
            raise ReminderStoreError("claim marker create failed") from exc
        except Exception as exc:
            raise ReminderStoreError("claim marker create failed") from exc

    def list_due_medicine_records(
        self, *, kind: str, status: str, now_iso: str, earliest_iso: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            return self._appwrite.query_documents(
                self.reminders_resource,
                queries=[
                    self._query("equal", "kind", _safe_text(kind)),
                    self._query("equal", "status", _safe_text(status)),
                    self._query("greaterThanEqual", "sendAtISO", _safe_text(earliest_iso)),
                    self._query("lessThanEqual", "sendAtISO", _safe_text(now_iso)),
                    self._query("orderAsc", "sendAtISO"),
                ],
                limit=limit,
            )
        except AppwriteProxyError as exc:
            if exc.status_code in {400, 404}:
                raise ReminderSchemaError("due reminder index unavailable") from exc
            raise ReminderStoreError("due reminder query failed") from exc
        except Exception as exc:
            raise ReminderStoreError("due reminder query failed") from exc

    def list_medicine_occurrences_to_seed(
        self, *, earliest_iso: str, latest_iso: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Read scheduled medicine occurrences through the med_logs time index.

        This deliberately does not scan the collection. Batch 3 must provision
        the required index before the feature flag can be enabled in production.
        """
        try:
            return self._appwrite.query_documents(
                "med_logs",
                queries=[
                    self._query("greaterThanEqual", "time", _safe_text(earliest_iso)),
                    self._query("lessThanEqual", "time", _safe_text(latest_iso)),
                    self._query("orderAsc", "time"),
                ],
                limit=limit,
            )
        except AppwriteProxyError as exc:
            if exc.status_code in {400, 404}:
                raise ReminderSchemaError("medicine occurrence index unavailable") from exc
            raise ReminderStoreError("medicine occurrence query failed") from exc
        except Exception as exc:
            raise ReminderStoreError("medicine occurrence query failed") from exc

    def list_occurrence_logs(self, *, user_id: str, med_id: str, occurrence_id: str) -> List[Dict[str, Any]]:
        try:
            return self._appwrite.query_documents(
                "med_logs",
                queries=[
                    self._query("equal", "userId", _safe_text(user_id)),
                    self._query("equal", "occurrenceId", _safe_text(occurrence_id)),
                ],
                limit=10,
            )
        except AppwriteProxyError as exc:
            if exc.status_code == 400:
                return []  # Legacy schema has no occurrenceId attribute yet.
            raise ReminderStoreError("occurrence log query failed") from exc

    def list_legacy_dose_logs(
        self, *, user_id: str, med_id: str, earliest_iso: str, latest_iso: str
    ) -> List[Dict[str, Any]]:
        try:
            return self._appwrite.query_documents(
                "med_logs",
                queries=[
                    self._query("equal", "userId", _safe_text(user_id)),
                    self._query("equal", "medId", _safe_text(med_id)),
                    self._query("greaterThanEqual", "time", _safe_text(earliest_iso)),
                    self._query("lessThanEqual", "time", _safe_text(latest_iso)),
                ],
                limit=20,
            )
        except AppwriteProxyError as exc:
            raise ReminderStoreError("legacy dose log query failed") from exc

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

    def list_reminder_devices(self, *, user_id: str) -> List[Dict[str, Any]]:
        """Read dispatch tokens through the userId index without downgrading an
        Appwrite outage into an apparent no-token condition."""
        uid = _safe_text(user_id)
        if not uid:
            return []
        try:
            return self._appwrite.query_documents(
                self.devices_resource,
                queries=[self._query("equal", "userId", uid)],
                limit=200,
            )
        except AppwriteProxyError as exc:
            raise ReminderStoreError("device query failed") from exc
        except Exception as exc:
            raise ReminderStoreError("device query failed") from exc

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

            doc_id = _hash_id("rem", f"{uid}|{eid}|{send_at}|{message}", length=32)
            data = {
                "userId": uid,
                "eventId": eid,
                # Default scheduled (not pending) so the dispatch task, which
                # only picks status=="scheduled", can actually send them.
                "status": _safe_text(r.get("status") or "scheduled"),
                "priority": _safe_text(r.get("priority") or "normal"),
                "toneProfile": _safe_text(r.get("toneProfile") or ""),
                "offsetMinutes": int(r.get("offsetMinutes") or 0),
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

        return {"success": True, "scheduled": scheduled}

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


notification_store = NotificationStore()
