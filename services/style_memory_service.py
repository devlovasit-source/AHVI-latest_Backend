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
    outfit_history row per (user, item). Best-effort per item."""
    proxy = _proxy()
    worn_at = (worn_at or "").strip() or _now_iso()
    clean_ids = [i for i in (_item_id(x) for x in (item_ids or [])) if i]
    recorded = 0
    for iid in clean_ids:
        try:
            existing = _find_history_row(proxy, user_id, iid)
            if existing:
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
        "_degraded": False,
    }
    try:
        rows = _proxy().list_documents("outfit_history", user_id=user_id, limit=200)
    except Exception:
        return {**empty, "_degraded": True}
    if not isinstance(rows, list) or not rows:
        # No evidence is neutral; it must not create a positive memory score.
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
        "_degraded": False,
    }


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
        "_degraded": False,
    }
    try:
        from services.board_service import list_saved_boards

        boards = list_saved_boards(user_id=user_id, limit=50)
    except Exception:
        return {**empty, "_degraded": True}
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
        "_degraded": False,
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
    # Durable Appwrite feedback is authoritative. Read failures are neutral;
    # never fall back to the legacy local JSON ranker.
    try:
        from services.style_feedback_store import load_feedback_memory
        feedback = load_feedback_memory(user_id)
        feedback_degraded = False
    except Exception as exc:  # noqa: BLE001
        logger.warning("AHVI_STYLE_FEEDBACK_READ_FAILED err=%s", str(exc)[:140])
        feedback = {}
        feedback_degraded = True
    saved_item_ids = list(dict.fromkeys([
        *(saved.get("saved_item_ids") or []),
        *(feedback.get("feedback_saved_item_ids") or []),
    ]))[:60]
    ctx = {
        "recently_worn_ids": wear["recently_worn_ids"],
        "underworn_ids": wear["underworn_ids"],
        "wear_counts": wear["wear_counts"],
        "last_worn_at": wear["last_worn_at"],
        "saved_item_ids": saved_item_ids,
        "saved_board_patterns": list(dict.fromkeys([
            *(saved.get("saved_board_patterns") or []),
            *(feedback.get("feedback_saved_board_patterns") or []),
        ]))[:20],
        "favorite_colors": list(dict.fromkeys([
            *(feedback.get("feedback_preferred_colors") or []),
            *(saved.get("favorite_colors") or []),
        ]))[:6],
        "favorite_categories": list(dict.fromkeys([
            *(feedback.get("feedback_preferred_categories") or []),
            *(saved.get("favorite_categories") or []),
        ]))[:6],
        "liked_item_ids": feedback.get("liked_item_ids") or [],
        "liked_board_patterns": feedback.get("liked_board_patterns") or [],
        "disliked_board_patterns": feedback.get("disliked_board_patterns") or [],
        "disliked_item_ids": [],  # no producer yet — reserved.
    }
    ctx["disliked_item_ids"] = feedback.get("disliked_item_ids") or []
    ctx["_personalization_meta"] = {
        "durable_feedback_used": bool(
            ctx["liked_item_ids"] or ctx["disliked_item_ids"]
            or ctx["liked_board_patterns"] or ctx["disliked_board_patterns"]
        ),
        "saved_memory_used": bool(ctx["saved_item_ids"] or ctx["saved_board_patterns"]),
        "wear_memory_used": bool(
            ctx["recently_worn_ids"] or ctx["underworn_ids"] or ctx["wear_counts"]
        ),
        "personalization_degraded": bool(
            wear.get("_degraded") or saved.get("_degraded") or feedback_degraded
        ),
    }
    has_memory = bool(
        ctx["recently_worn_ids"] or ctx["saved_item_ids"] or ctx["wear_counts"]
        or ctx["liked_item_ids"] or ctx["disliked_item_ids"]
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
        "liked_item_ids": [],
        "disliked_item_ids": [],
        "liked_board_patterns": [],
        "disliked_board_patterns": [],
        "_personalization_meta": {
            "durable_feedback_used": False,
            "saved_memory_used": False,
            "wear_memory_used": False,
            "personalization_degraded": False,
        },
    }
