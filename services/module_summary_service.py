"""Module summary cards.

Reads a user's real Appwrite data for a life module and returns a
`module_card` envelope the chat client renders as a summary card
(icon + count badge + item rows + "Open <module>" link).

This replaces the hardcoded demo cards that were baked into the
Flutter `_local` map.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from services.appwrite_proxy import AppwriteProxy


def _docs(resource: str, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    try:
        rows = AppwriteProxy().list_documents(resource, user_id=user_id, limit=limit)
    except Exception:
        return []
    if isinstance(rows, dict):
        rows = rows.get("documents") or rows.get("items") or []
    return [r for r in (rows or []) if isinstance(r, dict)]


def _txt(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _is_today(iso: str) -> bool:
    iso = _txt(iso)
    if not iso:
        return False
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return False
    now = datetime.now(dt.tzinfo or timezone.utc)
    return (dt.year, dt.month, dt.day) == (now.year, now.month, now.day)


def _card(
    *,
    module: str,
    title: str,
    icon: str,
    summary: str,
    count_done: int,
    count_total: int,
    rows: List[Dict[str, Any]],
    open_key: str,
) -> Dict[str, Any]:
    return {
        "success": True,
        "type": "module_card",
        "response_type": "module_card",
        "module": module,
        "domain": module,
        "card": {
            "title": title,
            "icon": icon,
            "summary": summary,
            "count_done": count_done,
            "count_total": count_total,
            "rows": rows,
            "open_key": open_key,
        },
        "message": summary,
        "message_text": summary,
        "response": summary,
        "cards": [],
        "style_boards": [],
        "chips": [],
        "data": {"module": module, "rows": rows, "message": summary},
        "meta": {"mode": "module_card", "module": module},
    }


# =========================
# MEDICINES
# =========================
def _medicines(user_id: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    taken = 0
    for doc in _docs("meds", user_id):
        name = _txt(doc.get("name")) or "Medicine"
        dose = _txt(doc.get("dose"))
        freq = _txt(doc.get("freq"))
        time = _txt(doc.get("time"))
        is_taken = _is_today(_txt(doc.get("lastTaken")))
        if is_taken:
            taken += 1
        main = f"{name} — {dose}" if dose else name
        sub = " · ".join(part for part in (freq, time) if part)
        rows.append(
            {
                "done": is_taken,
                "main": main,
                "sub": sub,
                "tag": "Taken" if is_taken else "Pending",
            }
        )
    total = len(rows)
    noun = "medicine" if total == 1 else "medicines"
    summary = (
        f"You have {total} {noun} tracked."
        if total
        else "No medicines tracked yet. Add one from the Medicines page."
    )
    return _card(
        module="medicines",
        title="Medicines",
        icon="medication",
        summary=summary,
        count_done=taken,
        count_total=total,
        rows=rows,
        open_key="medi",
    )


_BUILDERS = {
    "medicines": _medicines,
}


def build_module_summary(module: str, user_id: str) -> Dict[str, Any]:
    """Return a module_card envelope for `module`, or {} if unsupported."""
    builder = _BUILDERS.get(_txt(module).lower())
    if not builder or not _txt(user_id):
        return {}
    return builder(user_id)
