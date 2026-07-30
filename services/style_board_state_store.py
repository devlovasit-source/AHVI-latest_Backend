"""Durable, atomic style-board revision storage.

Model: one IMMUTABLE document per (board_id, revision) in the Appwrite
`style_board_states` collection, with a deterministic document ID derived
from board_id + revision. Creating revision N+1 is the atomic claim: Appwrite
document-ID uniqueness guarantees that of two concurrent shuffles from
revision N, exactly one create succeeds — no read-then-update CAS emulation.

Normalized record shape returned by every store method:
    {
        "user_id":  str,   # owner (authenticated user at registration)
        "board_id": str,
        "revision": int,
        "payload":  dict,  # scenario / source_policy / items+positions / ...
        "created_at": str, # ISO timestamp
    }

Production code must use AppwriteBoardStateStore. InMemoryBoardStateStore is
ONLY an explicitly injected test double (style_board_shuffle_service never
selects it implicitly).

Configuration (AppwriteBoardStateStore):
    APPWRITE_COLLECTION_STYLE_BOARD_STATES  — collection id
                                              (default: "style_board_states")
plus the standard AppwriteProxy endpoint/project/database/key variables.
Run scripts/create_style_board_states_collection.py once per environment to
create the collection, attributes and indexes (idempotent).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("ahvi.style_board_state_store")

RESOURCE = "style_board_states"


class BoardStateStoreError(RuntimeError):
    """Durable state storage is unavailable (transport/config failure)."""


class BoardRevisionExistsError(RuntimeError):
    """The (board_id, revision) document already exists — atomic claim lost."""


def revision_document_id(board_id: str, revision: int) -> str:
    """Deterministic Appwrite document ID for (board_id, revision).

    Appwrite custom IDs are limited to 36 chars from [a-zA-Z0-9._-], so a
    UUID board_id cannot simply be suffixed; a truncated SHA-1 of the pair is
    deterministic, collision-safe at this scale, and always valid.
    """
    seed = f"{str(board_id).strip()}|{int(revision)}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:36]


def _record(
    *, user_id: str, board_id: str, revision: int, payload: Dict[str, Any], created_at: str
) -> Dict[str, Any]:
    return {
        "user_id": str(user_id or ""),
        "board_id": str(board_id),
        "revision": int(revision),
        "payload": dict(payload or {}),
        "created_at": str(created_at or ""),
    }


class InMemoryBoardStateStore:
    """Test double ONLY — never used in production paths.

    Thread-safe so concurrency tests exercise the same "exactly one claim
    wins" contract as Appwrite document-ID uniqueness.
    """

    def __init__(self, backing: Optional[Dict[str, Dict[str, Any]]] = None):
        # backing may be shared between instances to model durable storage
        # surviving store re-instantiation.
        self._docs: Dict[str, Dict[str, Any]] = backing if backing is not None else {}
        self._lock = threading.Lock()

    def create_revision(
        self, *, user_id: str, board_id: str, revision: int, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        doc_id = revision_document_id(board_id, revision)
        rec = _record(
            user_id=user_id,
            board_id=board_id,
            revision=revision,
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            if doc_id in self._docs:
                raise BoardRevisionExistsError(doc_id)
            self._docs[doc_id] = rec
        return dict(rec)

    def get_revision(self, board_id: str, revision: int) -> Optional[Dict[str, Any]]:
        rec = self._docs.get(revision_document_id(board_id, revision))
        return dict(rec) if rec else None

    def get_latest(self, board_id: str) -> Optional[Dict[str, Any]]:
        board_id = str(board_id).strip()
        with self._lock:
            candidates = [r for r in self._docs.values() if r["board_id"] == board_id]
        if not candidates:
            return None
        return dict(max(candidates, key=lambda r: r["revision"]))

    def clear(self) -> None:
        with self._lock:
            self._docs.clear()


class AppwriteBoardStateStore:
    """Production store: immutable revision documents in Appwrite."""

    def __init__(self, proxy: Any = None):
        if proxy is None:
            from services.appwrite_proxy import AppwriteProxy

            proxy = AppwriteProxy()
        self._proxy = proxy

    def _collection_id(self) -> str:
        return (
            str(os.getenv("APPWRITE_COLLECTION_STYLE_BOARD_STATES") or "").strip()
            or RESOURCE
        )

    @staticmethod
    def _from_document(doc: Dict[str, Any]) -> Dict[str, Any]:
        payload_raw = doc.get("payload")
        if isinstance(payload_raw, str):
            try:
                payload = json.loads(payload_raw)
            except Exception:
                payload = {}
        elif isinstance(payload_raw, dict):
            payload = payload_raw
        else:
            payload = {}
        return _record(
            user_id=str(doc.get("userId") or ""),
            board_id=str(doc.get("boardId") or ""),
            revision=int(doc.get("revision") or 0),
            payload=payload,
            created_at=str(doc.get("createdAtISO") or doc.get("$createdAt") or ""),
        )

    def create_revision(
        self, *, user_id: str, board_id: str, revision: int, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        doc_id = revision_document_id(board_id, revision)
        data = {
            "userId": str(user_id or ""),
            "boardId": str(board_id).strip(),
            "revision": int(revision),
            "payload": json.dumps(payload or {}, separators=(",", ":"), ensure_ascii=False),
            "createdAtISO": datetime.now(timezone.utc).isoformat(),
        }
        try:
            doc = self._proxy.create_document(RESOURCE, data, document_id=doc_id)
        except Exception as exc:  # noqa: BLE001 — classify by status, not class
            # Any proxy failure is a storage failure; classification uses the
            # status_code attribute rather than exception-class identity so a
            # reloaded appwrite_proxy module (tests do this) cannot bypass it.
            if getattr(exc, "status_code", None) == 409:
                raise BoardRevisionExistsError(doc_id) from exc
            raise BoardStateStoreError(str(exc)) from exc
        return self._from_document({**data, **(doc or {})})

    def get_revision(self, board_id: str, revision: int) -> Optional[Dict[str, Any]]:
        doc_id = revision_document_id(board_id, revision)
        try:
            doc = self._proxy.get_document(RESOURCE, doc_id)
        except Exception as exc:  # noqa: BLE001 — classify by status, not class
            if getattr(exc, "status_code", None) == 404:
                return None
            raise BoardStateStoreError(str(exc)) from exc
        return self._from_document(doc or {})

    def get_latest(self, board_id: str) -> Optional[Dict[str, Any]]:
        board_id = str(board_id).strip()
        # Strict single-shot query — deliberately NOT the proxy's paged list
        # helper, whose compatibility fallbacks can drop query filters and
        # return other boards' documents. Any transport/syntax failure fails
        # closed as BOARD_STATE_UNAVAILABLE upstream.
        tokens = [
            {"method": "equal", "attribute": "boardId", "values": [board_id]},
            {"method": "orderDesc", "attribute": "revision"},
            {"method": "limit", "values": [1]},
        ]
        params = {"queries[]": [self._proxy._serialize_query_token(t) for t in tokens]}
        try:
            data = self._proxy._request(
                "GET", self._proxy._url(self._collection_id()), params=params
            )
        except Exception as exc:  # noqa: BLE001 — any proxy failure fails closed
            raise BoardStateStoreError(str(exc)) from exc
        docs = [
            d
            for d in (data or {}).get("documents", [])
            if isinstance(d, dict) and str(d.get("boardId") or "").strip() == board_id
        ]
        if not docs:
            return None
        return self._from_document(max(docs, key=lambda d: int(d.get("revision") or 0)))


__all__ = [
    "RESOURCE",
    "BoardStateStoreError",
    "BoardRevisionExistsError",
    "revision_document_id",
    "InMemoryBoardStateStore",
    "AppwriteBoardStateStore",
]
