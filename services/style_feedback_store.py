"""Durable, append-only Style feedback events.

Appwrite is authoritative. Qdrant and the legacy local JSON ranker are
non-authoritative. Reads are capped at 200 newest events per user; the latest
like/dislike for each item/board wins inside that window.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

RESOURCE = "style_feedback_events"
# Appwrite's default maximum page size is 100. Keep aggregation to one bounded,
# owner-filtered newest-first page so feedback is loaded once per Style request.
FEEDBACK_READ_LIMIT = 100
MAX_ITEM_IDS = 25
MAX_INPUT_BYTES = 100_000
ACTIVE_ACTIONS = {"like", "dislike", "saved", "skipped"}


class FeedbackStoreError(RuntimeError):
    """Durable feedback storage is unavailable."""


class FeedbackValidationError(ValueError):
    """Feedback is outside the bounded storage contract."""


def feedback_document_id(user_id: str, event_id: str) -> str:
    seed = f"{str(user_id).strip()}|{str(event_id).strip()}"
    return hashlib.sha256(seed.encode()).hexdigest()[:36]


def _text(value: Any, limit: int = 160) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _learning_text(value: Any, limit: int = 160) -> str:
    """Return compact non-media text suitable for the feedback learning payload."""
    clean = _text(value, limit)
    lowered = clean.lower()
    if lowered.startswith(("http://", "https://", "data:")) or "base64," in lowered:
        return ""
    return clean


def _unique(values: Iterable[Any], limit: int) -> List[str]:
    out, seen = [], set()
    for value in values:
        clean = _text(value, 128)
        marker = clean.lower()
        if clean and marker not in seen:
            seen.add(marker)
            out.append(clean)
        if len(out) >= limit:
            break
    return out


def _item_ids(board: Dict[str, Any], supplied: List[str]) -> List[str]:
    raw: List[Any] = list(supplied or [])
    for key in ("item_ids", "itemIds", "board_ids", "boardIds"):
        value = board.get(key)
        raw.extend(value if isinstance(value, list) else str(value).split(",") if value else [])
    for item in board.get("items") or []:
        raw.append(
            item.get("$id") or item.get("id") or item.get("item_id")
            if isinstance(item, dict) else item
        )
    if len([x for x in raw if str(x or "").strip()]) > MAX_ITEM_IDS:
        raise FeedbackValidationError(f"itemIds must contain at most {MAX_ITEM_IDS} values")
    return _unique(raw, MAX_ITEM_IDS)


def canonical_feedback_payload(board: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only small learning attributes; media URLs/base64 never enter storage."""
    if not isinstance(board, dict):
        raise FeedbackValidationError("board_payload must be an object")
    try:
        size = len(json.dumps(board, default=str, ensure_ascii=False).encode())
    except Exception as exc:
        raise FeedbackValidationError("board_payload is not JSON serializable") from exc
    if size > MAX_INPUT_BYTES:
        raise FeedbackValidationError("board_payload is too large")
    meta = board.get("style_metadata") if isinstance(board.get("style_metadata"), dict) else {}
    compact: Dict[str, Any] = {}
    aliases = {
        "style_signature": ("style_signature", "_style_signature"),
        "core_style_signature": ("core_style_signature", "_style_core_signature"),
        "style_archetype": ("style_archetype",), "style_direction": ("style_direction",),
        "aesthetic": ("aesthetic",), "vibe": ("vibe",),
    }
    for target, keys in aliases.items():
        value = next((board.get(k) for k in keys if board.get(k)), None) or meta.get(target)
        if _learning_text(value):
            compact[target] = _learning_text(value)
    colors, categories = [], []
    for item in board.get("items") or []:
        if isinstance(item, dict):
            colors.append(_learning_text(item.get("color") or item.get("colour")))
            categories.append(_learning_text(
                item.get("category") or item.get("type") or item.get("sub_category")
            ))
    if _unique(colors, 8):
        compact["colors"] = _unique(colors, 8)
    if _unique(categories, 8):
        compact["categories"] = _unique(categories, 8)
    return compact


def canonical_event(*, user_id: str, event_id: str, action: str,
                    board_payload: Dict[str, Any] | None = None,
                    item_ids: List[str] | None = None, board_id: str = "",
                    occasion: str = "", source_policy: str = "") -> Dict[str, Any]:
    uid, eid, act = _text(user_id, 128), _text(event_id, 128), _text(action, 16).lower()
    if not uid:
        raise FeedbackValidationError("authenticated user is required")
    if not eid or len(str(event_id or "").strip()) > 128:
        raise FeedbackValidationError("eventId is required and must be <= 128 characters")
    if act not in ACTIVE_ACTIONS:
        raise FeedbackValidationError("action must be like, dislike, saved, or skipped")
    board = board_payload if isinstance(board_payload, dict) else {}
    ids = _item_ids(board, item_ids or [])
    bid = _text(board_id or board.get("board_id") or board.get("boardId")
                or board.get("id") or board.get("card_id"), 128)
    compact = canonical_feedback_payload(board) if board else {}
    return {
        "userId": uid, "eventId": eid, "action": act, "boardId": bid,
        "itemIds": json.dumps(ids, separators=(",", ":"), ensure_ascii=False),
        "sourcePolicy": _text(source_policy or board.get("source_policy")
                              or board.get("sourcePolicy"), 64).lower(),
        "occasion": _text(occasion or board.get("occasion")
                          or board.get("canonical_occasion"), 80).lower(),
        "payload": json.dumps(compact, separators=(",", ":"), ensure_ascii=False),
        "createdAtISO": datetime.now(timezone.utc).isoformat(),
    }


class AppwriteStyleFeedbackStore:
    def __init__(self, proxy: Any = None):
        if proxy is None:
            from services.appwrite_proxy import AppwriteProxy
            proxy = AppwriteProxy()
        self._proxy = proxy

    def append(self, event: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = feedback_document_id(event.get("userId", ""), event.get("eventId", ""))
        try:
            doc = self._proxy.create_document(RESOURCE, event, document_id=doc_id)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 409:
                return {"event": dict(event), "idempotent": True, "document_id": doc_id}
            raise FeedbackStoreError(str(exc)) from exc
        return {"event": {**event, **(doc or {})}, "idempotent": False, "document_id": doc_id}

    def list_recent(self, user_id: str, limit: int = FEEDBACK_READ_LIMIT) -> List[Dict[str, Any]]:
        try:
            rows = self._proxy.list_documents(
                RESOURCE, user_id=str(user_id), limit=max(1, min(limit, FEEDBACK_READ_LIMIT))
            )
        except Exception as exc:
            raise FeedbackStoreError(str(exc)) from exc
        owned = [r for r in rows or [] if isinstance(r, dict)
                 and str(r.get("userId") or "") == str(user_id)]
        return sorted(owned, key=lambda r: str(r.get("createdAtISO")
                      or r.get("$createdAt") or ""), reverse=True)[:FEEDBACK_READ_LIMIT]


def _json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return _unique(value, 60)
    try:
        parsed = json.loads(value) if isinstance(value, str) else []
    except Exception:
        parsed = []
    return _unique(parsed if isinstance(parsed, list) else [], 60)


def _json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value) if isinstance(value, str) else {}
    except Exception:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def aggregate_feedback_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate newest-first; first like/dislike seen per entity wins."""
    item_action: Dict[str, tuple[str, str]] = {}
    board_action: Dict[str, tuple[str, str]] = {}
    saved_items, saved_patterns, liked_patterns, disliked_patterns = [], [], [], []
    positive_colors, positive_categories = [], []
    ordered = sorted(
        [e for e in events or [] if isinstance(e, dict)],
        key=lambda e: str(e.get("createdAtISO") or e.get("$createdAt") or ""),
        reverse=True,
    )
    for event in ordered[:FEEDBACK_READ_LIMIT]:
        action = str(event.get("action") or "").lower()
        if action not in ACTIVE_ACTIONS:
            continue
        ids, board_id = _json_list(event.get("itemIds")), _text(event.get("boardId"), 128).lower()
        payload = _json_dict(event.get("payload"))
        is_latest_board = bool(board_id and board_id not in board_action)
        if action in {"like", "dislike"}:
            for item_id in ids:
                item_action.setdefault(item_id.lower(), (item_id, action))
            if is_latest_board:
                original_board_id = _text(event.get("boardId"), 128)
                board_action.setdefault(board_id, (original_board_id, action))
        elif action == "saved":
            saved_items.extend(ids)
        patterns = _unique((payload.get(k) for k in (
            "style_signature", "core_style_signature", "style_archetype",
            "style_direction", "aesthetic", "vibe",
        )), 12)
        if action == "like" and is_latest_board:
            liked_patterns.extend(patterns)
            positive_colors.extend(payload.get("colors") or [])
            positive_categories.extend(payload.get("categories") or [])
        elif action == "dislike" and is_latest_board:
            disliked_patterns.extend(patterns)
        elif action == "saved":
            saved_patterns.extend(patterns)
    return {
        "liked_item_ids": _unique((item_id for item_id, action in item_action.values()
                                   if action == "like"), 60),
        "disliked_item_ids": _unique((item_id for item_id, action in item_action.values()
                                      if action == "dislike"), 60),
        "feedback_saved_item_ids": _unique(saved_items, 60),
        "feedback_saved_board_patterns": _unique(saved_patterns, 12),
        "liked_board_ids": _unique((board_id for board_id, action in board_action.values()
                                    if action == "like"), 30),
        "disliked_board_ids": _unique((board_id for board_id, action in board_action.values()
                                       if action == "dislike"), 30),
        "liked_board_patterns": _unique(liked_patterns, 12),
        "disliked_board_patterns": _unique(disliked_patterns, 12),
        "feedback_preferred_colors": _unique(positive_colors, 6),
        "feedback_preferred_categories": _unique(positive_categories, 6),
    }


def load_feedback_memory(user_id: str, *, store: Any = None) -> Dict[str, Any]:
    if not str(user_id or "").strip():
        return aggregate_feedback_events([])
    active_store = store or AppwriteStyleFeedbackStore()
    return aggregate_feedback_events(active_store.list_recent(user_id, FEEDBACK_READ_LIMIT))


__all__ = [
    "RESOURCE", "FEEDBACK_READ_LIMIT", "MAX_ITEM_IDS", "FeedbackStoreError",
    "FeedbackValidationError", "feedback_document_id", "canonical_event",
    "canonical_feedback_payload", "aggregate_feedback_events", "load_feedback_memory",
    "AppwriteStyleFeedbackStore",
]
