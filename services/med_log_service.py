from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from services.appwrite_proxy import AppwriteProxy

MED_LOG_RESOURCE = "med_logs"
VALID_STATUSES = {"taken", "skipped", "missed"}


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _normalize_time(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return datetime.now(timezone.utc).isoformat()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).isoformat()
    except Exception as exc:
        raise ValueError("time must be an ISO datetime") from exc


def _normalize_status(value: Any) -> str:
    status = _safe_str(value).lower()
    if status not in VALID_STATUSES:
        raise ValueError("status must be taken, skipped, or missed")
    return status


def _normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(doc or {})
    return {
        "id": _safe_str(row.get("$id") or row.get("id")),
        "userId": _safe_str(row.get("userId") or row.get("user_id")),
        "medId": _safe_str(row.get("medId") or row.get("med_id")),
        "medName": _safe_str(row.get("medName") or row.get("med_name")),
        "dose": _safe_str(row.get("dose")),
        "time": _safe_str(row.get("time")),
        "status": _safe_str(row.get("status")).lower(),
        "createdAt": _safe_str(row.get("$createdAt") or row.get("createdAt")),
        "updatedAt": _safe_str(row.get("$updatedAt") or row.get("updatedAt")),
    }


def _build_document(user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    med_id = _safe_str(payload.get("medId") or payload.get("med_id"))
    med_name = _safe_str(payload.get("medName") or payload.get("med_name"))
    dose = _safe_str(payload.get("dose"))
    if not _safe_str(user_id):
        raise PermissionError("Missing authenticated user")
    if not med_id:
        raise ValueError("medId is required")
    if not med_name:
        raise ValueError("medName is required")
    if not dose:
        raise ValueError("dose is required")
    return {
        "userId": _safe_str(user_id),
        "medId": med_id,
        "medName": med_name,
        "dose": dose,
        "time": _normalize_time(payload.get("time")),
        "status": _normalize_status(payload.get("status")),
    }


def create_med_log(user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    doc = AppwriteProxy().create_document(MED_LOG_RESOURCE, _build_document(user_id, payload))
    return _normalize_doc(doc)


def list_med_logs(user_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
    rows = AppwriteProxy().list_documents(
        MED_LOG_RESOURCE,
        user_id=_safe_str(user_id),
        limit=max(1, min(int(limit), 500)),
    )
    if isinstance(rows, dict):
        rows = rows.get("documents") or []
    user = _safe_str(user_id)
    return [
        _normalize_doc(row)
        for row in rows or []
        if isinstance(row, dict) and _safe_str(row.get("userId") or row.get("user_id")) == user
    ]


def update_med_log_status(user_id: str, log_id: str, status: str) -> Dict[str, Any]:
    existing = AppwriteProxy().get_document(MED_LOG_RESOURCE, log_id)
    row = _normalize_doc(existing)
    if row.get("userId") != _safe_str(user_id):
        raise PermissionError("Med log does not belong to this user")
    doc = AppwriteProxy().update_document(
        MED_LOG_RESOURCE,
        log_id,
        {"userId": _safe_str(user_id), "status": _normalize_status(status)},
    )
    return _normalize_doc(doc)
