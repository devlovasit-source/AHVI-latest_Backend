"""Read-only lookup of *today's* persisted meal plan for the Home Eat card.

Unlike ``module_summary_service._meals`` (which picks the newest plan of any
date and would happily show a stale plan), this selects only a plan whose
Appwrite ``$createdAt``, converted into the caller's local timezone, lands on
the local date. No same-day plan -> unavailable. Never falls back to an older
day. Strictly read-only: only ``list_documents`` is called.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.appwrite_proxy import AppwriteProxy
# Reuse the existing defensive meal-entry parser + ordering — same schema.
from services.module_summary_service import _parse_meal_entry, _MEAL_ORDER

logger = logging.getLogger("ahvi.services.today_meal_plan_service")

_UNAVAILABLE: Dict[str, Any] = {
    "status": "unavailable",
    "reason": "today_meal_plan_missing",
    "plan_id": None,
    "plan_name": None,
    "meal": None,
}


def _txt(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _meal_period_for_hour(hour: int) -> str:
    """Hour -> meal period. Mirrors services/diet_service.get_diet_recommendation
    exactly; do not diverge or the Home card and Diet module disagree."""
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 16:
        return "lunch"
    if 16 <= hour < 19:
        return "snack"
    return "dinner"


def _rows(resp: Any) -> List[Dict[str, Any]]:
    if isinstance(resp, dict):
        resp = resp.get("documents") or resp.get("items") or []
    return [r for r in (resp or []) if isinstance(r, dict)]


def _parse_created_at(value: Any, tzinfo) -> Optional[datetime]:
    raw = _txt(value)
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tzinfo)


def _meal_from_plan(plan: Dict[str, Any], period: str) -> Optional[Dict[str, Any]]:
    """Pick the meal matching `period`; else the first valid meal. Never fabricate."""
    meals_raw = plan.get("meals")
    if not isinstance(meals_raw, list):
        return None

    parsed: List[Dict[str, Any]] = []
    for entry in meals_raw:
        try:
            m = _parse_meal_entry(entry)
        except Exception:
            continue
        if not isinstance(m, dict):
            continue
        if not _txt(m.get("name")):
            continue  # empty/malformed -> ignore
        parsed.append(m)

    if not parsed:
        return None

    def _shape(m: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": _txt(m.get("name")),
            "type": _txt(m.get("type") or m.get("mealType")),
            "cal": m.get("cal") or m.get("kcal") or m.get("calories"),
        }

    for m in parsed:
        if _txt(m.get("type") or m.get("mealType")).lower() == period:
            return _shape(m)

    # No exact period match — first valid meal, ordered by canonical meal order
    # so "first" is stable (breakfast < lunch < dinner < snack) rather than
    # insertion-order noise.
    parsed.sort(key=lambda m: _MEAL_ORDER.get(
        _txt(m.get("type") or m.get("mealType")).lower(), 99))
    return _shape(parsed[0])


def get_today_meal_plan(
    *,
    user_id: str,
    local_now: datetime,
    proxy_client: Any = None,
) -> Dict[str, Any]:
    """Return today's meal + owning plan, or an unavailable envelope.

    Read-only. `proxy_client` is an injection seam for tests; production passes
    None and a fresh AppwriteProxy is used.
    """
    if not _txt(user_id):
        return dict(_UNAVAILABLE)

    tzinfo = local_now.tzinfo or timezone.utc
    today = local_now.astimezone(tzinfo).date()

    try:
        proxy = proxy_client or AppwriteProxy()
        resp = proxy.list_documents("meal_plans", user_id=user_id, limit=20)
    except Exception:
        logger.exception("today_meal_plan.list_failed")
        return dict(_UNAVAILABLE)

    # Same-day only. `list_documents` already scopes by userId, but re-check the
    # owner field defensively so a mis-scoped row can never leak across users.
    same_day: List[tuple[datetime, Dict[str, Any]]] = []
    for doc in _rows(resp):
        owner = _txt(doc.get("userId") or doc.get("user_id"))
        if owner and owner != _txt(user_id):
            continue
        created = _parse_created_at(doc.get("$createdAt"), tzinfo)
        if created is None or created.date() != today:
            continue
        same_day.append((created, doc))

    if not same_day:
        return dict(_UNAVAILABLE)

    # Newest first.
    same_day.sort(key=lambda pair: pair[0], reverse=True)

    # Prefer newest daily plan; else newest same-day plan of any type.
    selected = next(
        (doc for _, doc in same_day if _txt(doc.get("planType")).lower() == "daily"),
        same_day[0][1],
    )

    period = _meal_period_for_hour(local_now.hour)
    meal = _meal_from_plan(selected, period)
    if meal is None:
        return dict(_UNAVAILABLE)

    return {
        "status": "ready",
        "reason": None,
        "plan_id": _txt(selected.get("$id")) or None,
        "plan_name": _txt(selected.get("name")) or None,
        "meal": meal,
    }
