"""Small, privacy-bounded persistence foundation for beta operations telemetry.

This module intentionally has no instrumentation hooks. Callers can validate
and persist an already-normalized event without making telemetry failures part
of their user-facing operation's failure path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger("ahvi.beta_ops_telemetry")

# Kept injectable for tests and lazy-loaded so validation remains usable in
# tooling that does not load the application's optional HTTP stack.
AppwriteProxy = None

BETA_OPS_EVENTS_RESOURCE = "beta_ops_events"
MAX_METADATA_BYTES = 2048

ALLOWED_EVENT_TYPES = frozenset(
    {
        "user.activity",
        "home.summary_requested",
        "wardrobe.upload_started",
        "style.requested",
        "style_board.behavior",
        "style.request_outcome",
        "llm.usage_attempt",
        "rmbg.completed",
        "rmbg.failed",
        "notification.attempt",
        "product.event",
    }
)

_STRING_LIMITS = {
    "event_type": 64,
    "user_id": 128,
    "status": 32,
    "request_id": 128,
    "operation_id": 128,
    "provider": 32,
    "model": 96,
    "usecase": 64,
    "error_code": 64,
}
_NUMERIC_FIELDS = (
    "attempt",
    "duration_ms",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
)
_METADATA_KEYS = frozenset(
    {
        "source",
        "flow",
        "batch_id",
        "item_id",
        "board_id",
        "requested_count",
        "generated_count",
        "rejected_count",
        "reason_counts",
        "failure_reason",
    }
)
_PROHIBITED_KEY_RE = re.compile(
    r"(?:prompt|chat|image|photo|board_payload|response|credential|secret|token|password|api[_-]?key)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?:bearer\s+|(?:api[_-]?key|secret|password|token)\s*[:=]|https?://[^\s]*[?&](?:key|token|secret|signature)=)",
    re.IGNORECASE,
)
_PROHIBITED_VALUE_RE = re.compile(
    r"(?:prompt|chat\s+content|base64|data:image|raw\s+provider\s+response|credentials?|passwords?)",
    re.IGNORECASE,
)
_STRUCTURED_CONTENT_RE = re.compile(
    r"(?:^\s*[\[{]|[\"'](?:items|cards|wardrobe|messages|contents)[\"']\s*:)",
    re.IGNORECASE,
)
_CREDENTIAL_URL_RE = re.compile(r"https?://[^\s/@]+:[^\s/@]+@", re.IGNORECASE)


def deterministic_appwrite_id(*parts: str) -> str:
    """Return AHVI's existing deterministic 36-character SHA256 ID."""
    raw = ":".join(str(part).strip() for part in parts if part is not None).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:36]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_datetime(value: Any) -> str:
    if value is None or value == "":
        return _utcnow().isoformat()
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("occurred_at must be a valid ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _scalar_string(name: str, value: Any, *, required: bool = False) -> Optional[str]:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{name} is required")
    if len(text) > _STRING_LIMITS[name]:
        raise ValueError(f"{name} exceeds {_STRING_LIMITS[name]} characters")
    if (
        _PROHIBITED_VALUE_RE.search(text)
        or _SECRET_VALUE_RE.search(text)
        or _STRUCTURED_CONTENT_RE.search(text)
        or _CREDENTIAL_URL_RE.search(text)
    ):
        raise ValueError(f"{name} contains prohibited content")
    return text or None


def _validate_metadata_value(key: str, value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError(f"metadata field {key} must not be boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"metadata field {key} must not be negative")
        return value
    if isinstance(value, str):
        text = value.strip()
        if (
            len(text) > 256
            or _PROHIBITED_VALUE_RE.search(text)
            or _SECRET_VALUE_RE.search(text)
            or _STRUCTURED_CONTENT_RE.search(text)
            or _CREDENTIAL_URL_RE.search(text)
        ):
            raise ValueError(f"metadata field {key} contains prohibited content")
        return text
    if isinstance(value, Mapping):
        if key != "reason_counts":
            raise ValueError(f"metadata field {key} must be scalar")
        result: Dict[str, int] = {}
        for reason, count in value.items():
            reason_text = str(reason).strip()
            if not reason_text or len(reason_text) > 64 or _PROHIBITED_KEY_RE.search(reason_text):
                raise ValueError("reason_counts contains a prohibited reason")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("reason_counts values must be non-negative integers")
            result[reason_text] = count
        return result
    raise ValueError(f"metadata field {key} has an unsupported type")


def normalize_metadata(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    normalized: Dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key).strip()
        if key_text not in _METADATA_KEYS or _PROHIBITED_KEY_RE.search(key_text):
            raise ValueError(f"metadata field {key_text or '<empty>'} is not allowed")
        normalized[key_text] = _validate_metadata_value(key_text, value)
    encoded = json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("metadata_json exceeds 2048 bytes")
    return encoded


def normalize_event(
    *,
    event_type: str,
    user_id: str,
    occurred_at: Any = None,
    status: Any = None,
    request_id: Any = None,
    operation_id: Any = None,
    attempt: Any = None,
    provider: Any = None,
    model: Any = None,
    usecase: Any = None,
    duration_ms: Any = None,
    input_tokens: Any = None,
    output_tokens: Any = None,
    cached_tokens: Any = None,
    error_code: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    event_name = str(event_type or "").strip()
    if event_name not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"unsupported beta ops event_type: {event_name or '<empty>'}")
    event: Dict[str, Any] = {
        "event_type": event_name,
        "user_id": _scalar_string("user_id", user_id, required=True),
        "occurred_at": normalize_datetime(occurred_at),
    }
    for name in _STRING_LIMITS:
        if name == "event_type" or name == "user_id":
            continue
        value = _scalar_string(name, locals()[name])
        if value is not None:
            event[name] = value
    for name, value in zip(_NUMERIC_FIELDS, (attempt, duration_ms, input_tokens, output_tokens, cached_tokens)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        event[name] = value
    metadata_json = normalize_metadata(metadata)
    if metadata_json is not None:
        event["metadata_json"] = metadata_json
    return event


def event_document_id(event: Mapping[str, Any], idempotency_key: str = "") -> str:
    key = str(idempotency_key or "").strip()
    if not key:
        key = "|".join(f"{name}={event.get(name, '')}" for name in sorted(event))
    return deterministic_appwrite_id("beta_ops", key)


def new_operation_id() -> str:
    """Create one opaque identifier for one logical provider operation."""
    return uuid.uuid4().hex


def record_llm_attempt(
    *,
    user_id: Optional[str],
    request_id: Optional[str],
    operation_id: str,
    attempt: int,
    provider: str,
    model: str,
    usecase: Optional[str],
    status: str,
    duration_ms: int,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cached_tokens: Optional[int] = None,
    error_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist one provider attempt using the Phase 1 observational boundary."""
    return record_event(
        event_type="llm.usage_attempt",
        user_id=user_id or "",
        request_id=request_id or None,
        operation_id=operation_id,
        attempt=attempt,
        provider=provider,
        model=model,
        usecase=usecase,
        status=status,
        duration_ms=max(0, int(duration_ms)),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        error_code=error_code,
        idempotency_key=f"{operation_id}|{attempt}|{provider}|{model}",
    )


def record_event(
    *,
    idempotency_key: str = "",
    **event_fields: Any,
) -> Dict[str, Any]:
    """Safely persist one event; validation or Appwrite failures never raise."""
    event_type = str(event_fields.get("event_type") or "").strip()
    safe_event_type = event_type if event_type in ALLOWED_EVENT_TYPES else "invalid"
    try:
        event = normalize_event(**event_fields)
        document_id = event_document_id(event, idempotency_key)
        proxy_class = AppwriteProxy
        if proxy_class is None:
            from services.appwrite_proxy import AppwriteProxy as proxy_class
        proxy = proxy_class()
        try:
            stored = proxy.create_document(
                BETA_OPS_EVENTS_RESOURCE,
                event,
                document_id=document_id,
            )
            logger.info(
                "ahvi.beta_ops.telemetry_persisted event_type=%s document_id=%s",
                event["event_type"],
                document_id,
            )
            return {"persisted": True, "duplicate": False, "event": stored, "document_id": document_id}
        except Exception as exc:
            if getattr(exc, "status_code", None) != 409:
                raise
            stored = proxy.get_document(BETA_OPS_EVENTS_RESOURCE, document_id)
            logger.info(
                "ahvi.beta_ops.telemetry_duplicate event_type=%s document_id=%s",
                event["event_type"],
                document_id,
            )
            return {"persisted": True, "duplicate": True, "event": stored, "document_id": document_id}
    except Exception as exc:
        logger.warning(
            "ahvi.beta_ops.telemetry_persistence_failed event_type=%s error_type=%s",
            safe_event_type,
            type(exc).__name__,
        )
        return {
            "persisted": False,
            "duplicate": False,
            "event": None,
            "document_id": None,
        }


def list_events(
    *,
    user_id: Optional[str] = None,
    event_type: Optional[str] = None,
    occurred_after: Any = None,
    occurred_before: Any = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Read primitive for future aggregation with explicit local scoping."""
    proxy_class = AppwriteProxy
    if proxy_class is None:
        from services.appwrite_proxy import AppwriteProxy as proxy_class
    proxy = proxy_class()
    safe_limit = max(1, min(int(limit), 100))
    safe_offset = max(0, int(offset))
    uid = str(user_id or "").strip()
    event_name = str(event_type or "").strip()
    if uid:
        rows = proxy.find_by_attribute(
            BETA_OPS_EVENTS_RESOURCE,
            "user_id",
            uid,
            limit=safe_limit,
        )
    elif event_name:
        rows = proxy.find_by_attribute(
            BETA_OPS_EVENTS_RESOURCE,
            "event_type",
            event_name,
            limit=safe_limit,
        )
    else:
        rows = proxy.list_documents(
            BETA_OPS_EVENTS_RESOURCE,
            limit=safe_limit,
            offset=safe_offset,
        )
    after = normalize_datetime(occurred_after) if occurred_after is not None else None
    before = normalize_datetime(occurred_before) if occurred_before is not None else None
    filtered = []
    for row in rows:
        if uid and str(row.get("user_id") or "") != uid:
            continue
        if event_name and row.get("event_type") != event_name:
            continue
        occurred = str(row.get("occurred_at") or "")
        if after and occurred < after:
            continue
        if before and occurred >= before:
            continue
        filtered.append(row)
    return filtered[safe_offset:] if (uid or event_name) and safe_offset else filtered


__all__ = [
    "ALLOWED_EVENT_TYPES",
    "BETA_OPS_EVENTS_RESOURCE",
    "deterministic_appwrite_id",
    "event_document_id",
    "list_events",
    "normalize_datetime",
    "normalize_event",
    "normalize_metadata",
    "new_operation_id",
    "record_llm_attempt",
    "record_event",
]
