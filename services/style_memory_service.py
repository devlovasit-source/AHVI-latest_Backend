"""Minimal style learning producers.

Writes/read the lightweight wear + saved-board memory that the scorer and
reasoner already consume. NOT a learning agent — just durable counters via the
existing Appwrite proxy. All reads are best-effort and degrade to neutral
(empty) memory so a missing collection never breaks styling.

outfit_history collection contract (proposed — env
APPWRITE_COLLECTION_OUTFIT_HISTORY, else literal "outfit_history"):
    userId      string   (indexed)
    itemId      string   (indexed)  one row per item
    wearCount   integer
    lastWornAt  datetime (ISO8601)
    lastBoardId string
    lastOccasion string
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ahvi.style_memory")

# Items worn within this window count as "recently worn" (repeat penalty).
RECENT_WINDOW_DAYS = 10
# Wear count at/under this = "under-worn" (gentle boost when occasion-fit).
UNDERWORN_MAX_WEARS = 1


def _proxy():
    from services.appwrite_proxy import AppwriteProxy

    return AppwriteProxy()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _item_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("$id") or value.get("id") or value.get("item_id") or value.get("name") or ""
        ).strip()
    return str(value or "").strip()


def _parse_iso(value: Any) -> Optional[datetime]:
    """Same tolerant parsing load_wear_memory already uses for lastWornAt.
    Naive timestamps are assumed UTC (never local-tz-converted) since the API
    has no client-timezone field — see _is_same_day."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _is_same_day(a: Any, b: Any) -> bool:
    """UTC-calendar-date equality. The wear-today contract has no client
    timezone field, so UTC date is the narrowest safe boundary available —
    matches the UTC 'now' load_wear_memory already uses for recency. An
    unparsable timestamp is treated as NOT the same day so a wear is never
    silently dropped."""
    da, db = _parse_iso(a), _parse_iso(b)
    if da is None or db is None:
        return False
    return da.astimezone(timezone.utc).date() == db.astimezone(timezone.utc).date()


# ---------------------------------------------------------------------------
# WRITE: wear-today
# ---------------------------------------------------------------------------
def record_wear(
    *,
    user_id: str,
    item_ids: List[str],
    board_id: str = "",
    occasion: str = "",
    worn_at: str = "",
) -> Dict[str, Any]:
    """Increment wearCount + set lastWornAt for each item. Upserts one
    outfit_history row per (user, item). Best-effort per item.

    Idempotent per (user, item, UTC calendar day): repeat taps on the same
    day refresh board/occasion context only, never double-count wearCount or
    overwrite the day's original lastWornAt.
    """
    proxy = _proxy()
    worn_at = (worn_at or "").strip() or _now_iso()
    clean_ids = [i for i in (_item_id(x) for x in (item_ids or [])) if i]
    recorded = 0
    for iid in clean_ids:
        try:
            existing = _find_history_row(proxy, user_id, iid)
            if existing and _is_same_day(existing.get("lastWornAt"), worn_at):
                metadata_patch: Dict[str, Any] = {}
                if board_id and board_id != existing.get("lastBoardId"):
                    metadata_patch["lastBoardId"] = board_id
                if occasion and occasion != existing.get("lastOccasion"):
                    metadata_patch["lastOccasion"] = occasion
                if metadata_patch:
                    proxy.update_document(
                        "outfit_history",
                        existing.get("$id") or existing.get("id"),
                        metadata_patch,
                    )
            elif existing:
                count = int(existing.get("wearCount") or 0) + 1
                proxy.update_document(
                    "outfit_history",
                    existing.get("$id") or existing.get("id"),
                    {
                        "wearCount": count,
                        "lastWornAt": worn_at,
                        "lastBoardId": board_id or existing.get("lastBoardId") or "",
                        "lastOccasion": occasion or existing.get("lastOccasion") or "",
                    },
                )
            else:
                proxy.create_document(
                    "outfit_history",
                    {
                        "userId": str(user_id),
                        "itemId": iid,
                        "wearCount": 1,
                        "lastWornAt": worn_at,
                        "lastBoardId": board_id or "",
                        "lastOccasion": occasion or "",
                    },
                )
            recorded += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("ahvi.wear_record_failed item=%s err=%s", iid, str(exc)[:140])
    logger.info(
        "AHVI_WEAR_TODAY_RECORDED user_id=%s items=%d recorded=%d board_id=%s occasion=%s",
        user_id, len(clean_ids), recorded, board_id, occasion,
    )
    return {"recorded": recorded, "item_ids": clean_ids, "worn_at": worn_at}


def _find_history_row(proxy, user_id: str, item_id: str) -> Optional[Dict[str, Any]]:
    try:
        rows = proxy.list_documents("outfit_history", user_id=user_id, limit=100)
    except Exception:
        return None
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict) and str(r.get("itemId") or "").strip() == item_id:
            return r
    return None


# ---------------------------------------------------------------------------
# READ: wear memory
# ---------------------------------------------------------------------------
def load_wear_memory(user_id: str, wardrobe: Any = None) -> Dict[str, Any]:
    """Return {recently_worn_ids, underworn_ids, wear_counts, last_worn_at}.
    Neutral (empty) when no history / no collection."""
    empty = {
        "recently_worn_ids": [],
        "underworn_ids": [],
        "wear_counts": {},
        "last_worn_at": {},
    }
    try:
        rows = _proxy().list_documents("outfit_history", user_id=user_id, limit=200)
    except Exception:
        return empty
    if not isinstance(rows, list) or not rows:
        # No history: every wardrobe item is "under-worn".
        underworn = [
            _item_id(it) for it in (wardrobe or []) if _item_id(it)
        ]
        empty["underworn_ids"] = underworn
        return empty

    wear_counts: Dict[str, int] = {}
    last_worn: Dict[str, str] = {}
    recent: List[str] = []
    now = datetime.now(timezone.utc)
    for r in rows:
        if not isinstance(r, dict):
            continue
        iid = str(r.get("itemId") or "").strip()
        if not iid:
            continue
        count = int(r.get("wearCount") or 0)
        wear_counts[iid] = count
        lw = str(r.get("lastWornAt") or "").strip()
        if lw:
            last_worn[iid] = lw
            try:
                dt = datetime.fromisoformat(lw.replace("Z", "+00:00"))
                if (now - dt).days <= RECENT_WINDOW_DAYS:
                    recent.append(iid)
            except Exception:
                pass

    # under-worn = wardrobe items with zero/low recorded wears.
    underworn: List[str] = []
    for it in wardrobe or []:
        iid = _item_id(it)
        if iid and wear_counts.get(iid, 0) <= UNDERWORN_MAX_WEARS:
            underworn.append(iid)

    return {
        "recently_worn_ids": recent,
        "underworn_ids": underworn,
        "wear_counts": wear_counts,
        "last_worn_at": last_worn,
    }


# ---------------------------------------------------------------------------
# READ: recommendation memory (what AHVI suggested — NOT actual wear).
# Actual wear lives exclusively in outfit_history above; this reads the
# separate memories.recent_outfits blob written by get_daily_outfits().
# ---------------------------------------------------------------------------
def _outfit_signature(items: Any) -> str:
    """Order/duplicate-independent id signature for a set of outfit items."""
    ids = {
        _item_id(it).lower() for it in (items or []) if _item_id(it)
    }
    return "|".join(sorted(ids)) if ids else ""


def load_recommendation_memory(user_id: str) -> Dict[str, Any]:
    """Return {recent_recommended_signatures}. Signatures of outfits AHVI
    recently recommended (memories.recent_outfits), so the scorer can avoid
    repeating an exact past recommendation. Neutral when no history."""
    empty = {"recent_recommended_signatures": []}
    user_id = str(user_id or "").strip()
    if not user_id:
        return empty
    try:
        from brain.outfit_pipeline import _load_user_memory

        user_memory = _load_user_memory(user_id)
    except Exception:
        return empty
    recent_outfits = user_memory.get("recent_outfits") if isinstance(user_memory, dict) else None
    if not isinstance(recent_outfits, list):
        return empty

    signatures: List[str] = []
    for outfit in recent_outfits:
        if not isinstance(outfit, dict):
            continue
        sig = _outfit_signature(outfit.get("items"))
        if sig:
            signatures.append(sig)
    return {"recent_recommended_signatures": list(dict.fromkeys(signatures))}


# ---------------------------------------------------------------------------
# READ: outfit-level rejection memory ("Not for me").
# Writer is routers/feedback.py's dislike action (Qdrant user_memory,
# memory_type="disliked", populated with item_ids/outfit_signature only when
# the caller supplies them — legacy dislike callers still work, they just
# never produce a signature). This is deliberately NOT disliked_item_ids:
# rejecting one outfit must never blacklist its individual garments.
# ---------------------------------------------------------------------------
def load_rejected_outfit_memory(user_id: str, limit: int = 50) -> Dict[str, Any]:
    """Return {rejected_outfit_signatures}. Neutral when no rejections or
    Qdrant is unavailable — same best-effort contract as every other loader
    in this module."""
    empty = {"rejected_outfit_signatures": []}
    user_id = str(user_id or "").strip()
    if not user_id:
        return empty
    try:
        from services.qdrant_service import qdrant_service

        rows = qdrant_service.list_user_memory(user_id, "disliked", limit=limit)
    except Exception:
        return empty
    signatures: List[str] = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        sig = str(r.get("outfit_signature") or "").strip()
        if sig:
            signatures.append(sig)
    return {"rejected_outfit_signatures": list(dict.fromkeys(signatures))}


# ---------------------------------------------------------------------------
# READ: localized replacement memory ("Change it").
# Writer is the change-item endpoint, only after a mutation is actually
# accepted/committed. from/to are intentionally separate weak signals, never
# a global blacklist of the replaced item.
# ---------------------------------------------------------------------------
def load_outfit_change_memory(user_id: str, limit: int = 50) -> Dict[str, Any]:
    """Return {recent_changed_items}: a list of {from_item_id, to_item_id,
    role, occasion} events, newest first. Kept as structured events (not
    flattened id sets) so the scorer can gate the from/to signal by the
    context it happened in — a "Change it" swap must stay local to its own
    role/occasion, never become a blanket opinion on the item. Neutral when
    no change history or Qdrant is unavailable."""
    empty = {"recent_changed_items": []}
    user_id = str(user_id or "").strip()
    if not user_id:
        return empty
    try:
        from services.qdrant_service import qdrant_service

        rows = qdrant_service.list_user_memory(user_id, "changed_item", limit=limit)
    except Exception:
        return empty
    events: List[Dict[str, str]] = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        fid = str(r.get("from_item_id") or "").strip()
        tid = str(r.get("to_item_id") or "").strip()
        if not fid and not tid:
            continue
        events.append(
            {
                "from_item_id": fid,
                "to_item_id": tid,
                "role": str(r.get("role") or "").strip(),
                "occasion": str(r.get("occasion") or "").strip(),
            }
        )
    return {"recent_changed_items": events}


# ---------------------------------------------------------------------------
# READ: saved-board memory
# ---------------------------------------------------------------------------
def load_saved_board_memory(user_id: str) -> Dict[str, Any]:
    """Return {saved_item_ids, saved_board_patterns, favorite_colors,
    favorite_categories}. Neutral when no saved boards."""
    empty = {
        "saved_item_ids": [],
        "saved_board_patterns": [],
        "favorite_colors": [],
        "favorite_categories": [],
    }
    try:
        from services.board_service import list_saved_boards

        boards = list_saved_boards(user_id=user_id, limit=50)
    except Exception:
        return empty
    if not isinstance(boards, list) or not boards:
        return empty

    saved_ids: List[str] = []
    patterns: List[str] = []
    colors: Dict[str, int] = {}
    cats: Dict[str, int] = {}
    for b in boards:
        if not isinstance(b, dict):
            continue
        for key in ("item_ids", "itemIds", "board_ids", "boardIds"):
            v = b.get(key)
            if isinstance(v, list):
                saved_ids.extend(str(x).strip() for x in v if str(x).strip())
            elif isinstance(v, str) and v.strip():
                saved_ids.extend(x.strip() for x in v.split(",") if x.strip())
        for pkey in ("aesthetic", "vibe", "occasion"):
            pv = str(b.get(pkey) or "").strip()
            if pv:
                patterns.append(pv.lower())
        for it in b.get("items") or []:
            if isinstance(it, dict):
                c = str(it.get("color") or it.get("colour") or "").strip().lower()
                if c:
                    colors[c] = colors.get(c, 0) + 1
                cat = str(it.get("category") or it.get("type") or "").strip().lower()
                if cat:
                    cats[cat] = cats.get(cat, 0) + 1

    def _top(d: Dict[str, int], n: int) -> List[str]:
        return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]]

    out = {
        "saved_item_ids": list(dict.fromkeys(saved_ids))[:60],
        "saved_board_patterns": list(dict.fromkeys(patterns))[:8],
        "favorite_colors": _top(colors, 6),
        "favorite_categories": _top(cats, 6),
    }
    logger.info(
        "AHVI_SAVED_BOARD_MEMORY_LOADED user_id=%s boards=%d saved_items=%d colors=%s",
        user_id, len(boards), len(out["saved_item_ids"]), out["favorite_colors"],
    )
    return out


# ---------------------------------------------------------------------------
# Combined memory context for the scorer/reasoner.
# ---------------------------------------------------------------------------
def build_style_memory_context(user_id: str, wardrobe: Any = None) -> Dict[str, Any]:
    """One call → the full memory slice the scorer + reasoner read. Always
    returns the keys (empty when no data) so callers never KeyError."""
    if not str(user_id or "").strip():
        return _neutral_memory()
    wear = load_wear_memory(user_id, wardrobe)
    saved = load_saved_board_memory(user_id)
    recommended = load_recommendation_memory(user_id)
    rejected = load_rejected_outfit_memory(user_id)
    changed = load_outfit_change_memory(user_id)
    ctx = {
        "recently_worn_ids": wear["recently_worn_ids"],
        "underworn_ids": wear["underworn_ids"],
        "wear_counts": wear["wear_counts"],
        "last_worn_at": wear["last_worn_at"],
        "saved_item_ids": saved["saved_item_ids"],
        "saved_board_patterns": saved["saved_board_patterns"],
        "favorite_colors": saved["favorite_colors"],
        "favorite_categories": saved["favorite_categories"],
        "disliked_item_ids": [],  # no producer yet — reserved.
        "recent_recommended_signatures": recommended["recent_recommended_signatures"],
        "rejected_outfit_signatures": rejected["rejected_outfit_signatures"],
        "recent_changed_items": changed["recent_changed_items"],
    }
    has_memory = bool(
        ctx["recently_worn_ids"] or ctx["saved_item_ids"] or ctx["wear_counts"]
        or ctx["recent_recommended_signatures"]
    )
    if has_memory:
        logger.info(
            "AHVI_STYLE_MEMORY_CONTEXT_USED recent=%d underworn=%d saved=%d",
            len(ctx["recently_worn_ids"]), len(ctx["underworn_ids"]), len(ctx["saved_item_ids"]),
        )
    return ctx


def _neutral_memory() -> Dict[str, Any]:
    return {
        "recently_worn_ids": [],
        "underworn_ids": [],
        "wear_counts": {},
        "last_worn_at": {},
        "saved_item_ids": [],
        "saved_board_patterns": [],
        "favorite_colors": [],
        "favorite_categories": [],
        "disliked_item_ids": [],
        "recent_recommended_signatures": [],
        "rejected_outfit_signatures": [],
        "recent_changed_items": [],
    }
