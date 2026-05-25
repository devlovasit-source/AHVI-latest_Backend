"""Module summary cards.

Reads a user's real Appwrite data for a life module and returns a
`module_card` envelope the chat client renders as a summary card
(icon + count badge + item rows + "Open <module>" link).

This replaces the hardcoded demo cards that were baked into the
Flutter `_local` map. One builder per module — all of them defensive
about field names + missing data, so an unwritten collection just
renders a clean "nothing yet" card instead of crashing.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from services.appwrite_proxy import AppwriteProxy


# =========================
# HELPERS
# =========================
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


def _parse_iso(value: Any) -> datetime | None:
    raw = _txt(value)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _is_today(value: Any) -> bool:
    dt = _parse_iso(value)
    if not dt:
        return False
    now = datetime.now(dt.tzinfo or timezone.utc)
    return (dt.year, dt.month, dt.day) == (now.year, now.month, now.day)


_QUICK_ACTIONS = {
    "meals": ["Create today's meal plan", "High-protein meals", "Light dinner ideas", "Open Meals"],
    "workout": ["Home workout", "Gym workout", "Workout outfit", "Recovery meal"],
    "medicines": ["Mark taken", "Set reminder", "Open Medicines"],
    "bills": ["Mark paid", "Set reminder", "Open Bills"],
    "events": ["Plan my day", "Add event", "Prep for tomorrow"],
    "skincare": ["Create routine", "Morning routine", "Evening routine", "Open Skincare"],
}


def _quick_actions(module: str) -> List[str]:
    return list(_QUICK_ACTIONS.get(module, []))


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
    quick_actions = _quick_actions(module)
    open_module = {
        "type": "open_module",
        "module": open_key,
        "route": f"/organize/{open_key}",
    }
    return {
        "success": True,
        "type": "module_card",
        "response_type": "module_card",
        "module": module,
        "domain": module,
        "intent": module,
        "card": {
            "title": title,
            "icon": icon,
            "summary": summary,
            "count_done": count_done,
            "count_total": count_total,
            "rows": rows,
            "open_key": open_key,
            "quick_actions": quick_actions,
            "open_module": open_module,
        },
        "message": summary,
        "message_text": summary,
        "response": summary,
        "cards": [],
        "style_boards": [],
        "chips": quick_actions,
        "quick_actions": quick_actions,
        "cta": open_module,
        "open_module": open_module,
        "data": {
            "module": module,
            "intent": module,
            "rows": rows,
            "message": summary,
            "quick_actions": quick_actions,
            "open_module": open_module,
        },
        "meta": {"mode": "module_card", "module": module},
    }


# =========================
# MEDICINES — `meds` collection
# =========================
def _medicines(user_id: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    taken = 0
    for doc in _docs("meds", user_id):
        name = _txt(doc.get("name")) or "Medicine"
        dose = _txt(doc.get("dose"))
        freq = _txt(doc.get("freq"))
        time = _txt(doc.get("time"))
        is_taken = _is_today(doc.get("lastTaken"))
        if is_taken:
            taken += 1
        main = f"{name} — {dose}" if dose else name
        sub = " · ".join(p for p in (freq, time) if p)
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
        else "No medicines are tracked yet. I can help add one and set a reminder."
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


# =========================
# BILLS — `bills` collection
# =========================
def _bills(user_id: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for doc in _docs("bills", user_id):
        store = _txt(doc.get("store") or doc.get("name") or doc.get("title"))
        amount = doc.get("amount")
        date_dt = _parse_iso(doc.get("date") or doc.get("dueDate"))
        category = _txt(doc.get("category"))
        date_short = date_dt.strftime("%b %d") if date_dt else ""
        sub_parts: List[str] = []
        if date_short:
            sub_parts.append(f"Due: {date_short}")
        if category:
            sub_parts.append(category)
        tag = ""
        if isinstance(amount, (int, float)):
            tag = f"₹{int(amount):,}" if amount >= 1 else f"₹{amount}"
        elif amount is not None:
            tag = f"₹{_txt(amount)}"
        rows.append(
            {
                "done": False,
                "main": store or "Bill",
                "sub": " · ".join(sub_parts),
                "tag": tag,
            }
        )
    total = len(rows)
    noun = "bill" if total == 1 else "bills"
    summary = (
        f"You have {total} unpaid {noun}."
        if total
        else "No pending bills are saved yet. I can help add one and set a due-date reminder."
    )
    return _card(
        module="bills",
        title="Bills",
        icon="receipt",
        summary=summary,
        count_done=0,
        count_total=total,
        rows=rows,
        open_key="bill",
    )


# =========================
# EVENTS — calendar_events (via calendar_service)
# =========================
def _events(user_id: str, *, upcoming: bool = False) -> Dict[str, Any]:
    try:
        from services.calendar_service import list_today_calendar_events, list_upcoming_calendar_events

        events = (
            list_upcoming_calendar_events(user_id=user_id, limit=8)
            if upcoming
            else list_today_calendar_events(user_id=user_id)
        )
    except Exception:
        events = []
    rows: List[Dict[str, Any]] = []
    done_count = 0
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        title = _txt(ev.get("title")) or "Event"
        start_dt = _parse_iso(ev.get("start_time") or ev.get("startTime"))
        venue = _txt(ev.get("venue_name") or ev.get("venueName") or ev.get("location"))
        kind = _txt(ev.get("type") or "Plan")
        if kind:
            kind = kind[:1].upper() + kind[1:]
        status = _txt(ev.get("status")).lower()
        when = start_dt.strftime("%b %d · %I:%M %p") if start_dt else ""
        sub_parts = [when, venue]
        sub = " · ".join(p for p in sub_parts if p)
        is_done = status in {"completed", "done"}
        if is_done:
            done_count += 1
        rows.append(
            {
                "done": is_done,
                "main": title,
                "sub": sub,
                "tag": kind,
            }
        )
    total = len(rows)
    if total:
        label = "upcoming" if upcoming else "today"
        summary = f"You have {total} event{'s' if total != 1 else ''} {label}."
    else:
        summary = (
            "No upcoming events are saved yet. Want me to add one or help plan what is next?"
            if upcoming
            else "Your day looks open. Want me to help plan it around a workout, errands, or prep for tomorrow?"
        )
    return _card(
        module="events",
        title="Upcoming Events" if upcoming else "Events",
        icon="event",
        summary=summary,
        count_done=done_count,
        count_total=total,
        rows=rows[:8],
        open_key="calendar",
    )


def _events_upcoming(user_id: str) -> Dict[str, Any]:
    return _events(user_id, upcoming=True)


# =========================
# MEALS — `meal_plans` collection
# =========================
# Schema is per-plan: one doc per saved plan, with `meals` as an array
# of JSON-encoded meal strings. The card picks the most recent daily
# plan (or the most recent plan of any type as fallback) and extracts
# the meals from that array.
_MEAL_ORDER = {"breakfast": 0, "lunch": 1, "dinner": 2, "snack": 3}


def _parse_meal_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, dict):
        return entry
    text = _txt(entry)
    if not text:
        return {}
    try:
        import json

        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"name": text}
    except Exception:
        return {"name": text}


def _meals(user_id: str) -> Dict[str, Any]:
    docs = _docs("meal_plans", user_id, limit=20)
    # Most recent first.
    docs.sort(key=lambda d: _txt(d.get("$createdAt")), reverse=True)

    selected_plan: Dict[str, Any] | None = None
    for doc in docs:
        if _txt(doc.get("planType")).lower() == "daily":
            selected_plan = doc
            break
    if selected_plan is None and docs:
        selected_plan = docs[0]

    rows: List[Dict[str, Any]] = []
    if selected_plan:
        meals_raw = selected_plan.get("meals") or []
        if isinstance(meals_raw, list):
            parsed_meals = [_parse_meal_entry(m) for m in meals_raw]
            parsed_meals.sort(
                key=lambda m: _MEAL_ORDER.get(
                    _txt(m.get("type") or m.get("mealType")).lower(), 99
                )
            )
            for meal in parsed_meals[:6]:
                name = _txt(meal.get("name")) or "Meal"
                meal_type = _txt(meal.get("type") or meal.get("mealType"))
                kcal = meal.get("cal") or meal.get("kcal") or meal.get("calories")
                sub_parts: List[str] = []
                if meal_type:
                    sub_parts.append(meal_type)
                if kcal not in (None, "", 0):
                    sub_parts.append(f"{kcal} kcal")
                rows.append(
                    {
                        "done": False,
                        "main": name,
                        "sub": " · ".join(sub_parts),
                        "tag": meal_type or "Meal",
                    }
                )

    total = len(rows)
    if total:
        plan_name = _txt(selected_plan.get("name")) if selected_plan else ""
        summary = (
            f"{plan_name}: {total} meal{'s' if total != 1 else ''} planned."
            if plan_name
            else f"You have {total} meal{'s' if total != 1 else ''} planned."
        )
    else:
        summary = "No meals are planned yet. I can create today's meal plan around your goal and diet preference."

    return _card(
        module="meals",
        title="Meals",
        icon="restaurant",
        summary=summary,
        count_done=0,
        count_total=total,
        rows=rows,
        open_key="meal",
    )


# =========================
# WORKOUT — fitness engine's today recommendation
# =========================
def _workout(user_id: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    summary = "No workout is planned yet. I can build a home or gym session for today."
    today = None
    try:
        # Lazy import — the workouts router knows how to assemble a card.
        from routers.workouts import _recommendations  # type: ignore
        from services.workout_context_service import build_workout_context  # type: ignore

        context = build_workout_context(
            user_id,
            {
                "goal": "general_fitness",
                "duration": 12,
                "location": "home",
                "equipment": "none",
                "profile": {},
            },
        )
        cards = _recommendations(context, limit=1)
        today = cards[0] if cards else None
    except Exception:
        today = None

    if isinstance(today, dict):
        exercises = today.get("exercises") or today.get("items") or []
        if isinstance(exercises, list):
            for ex in exercises[:6]:
                if not isinstance(ex, dict):
                    continue
                name = _txt(ex.get("name") or ex.get("title")) or "Exercise"
                kind = _txt(ex.get("type") or ex.get("category"))
                sets = ex.get("sets")
                reps = ex.get("reps")
                dur = ex.get("duration_minutes") or ex.get("duration")
                bits: List[str] = []
                if kind:
                    bits.append(kind)
                if sets:
                    bits.append(f"{sets} sets")
                if reps:
                    bits.append(f"{reps} reps")
                if dur:
                    bits.append(f"{dur} min")
                rows.append(
                    {
                        "done": False,
                        "main": name,
                        "sub": " · ".join(bits),
                        "tag": kind or "Workout",
                    }
                )
            if rows:
                summary = (
                    f"Today's workout has {len(rows)} exercise"
                    f"{'s' if len(rows) != 1 else ''}."
                )

    return _card(
        module="workout",
        title="Workout",
        icon="fitness",
        summary=summary,
        count_done=0,
        count_total=len(rows),
        rows=rows,
        open_key="workout",
    )


# =========================
# SKINCARE — `skincare_profiles` collection (one doc per user)
# =========================
def _skincare(user_id: str) -> Dict[str, Any]:
    docs = _docs("skincare_profiles", user_id) or _docs("skincare", user_id)
    rows: List[Dict[str, Any]] = []
    summary = "Your morning skincare routine is not set up yet. I can help create a simple 3-step routine."
    if docs:
        prof = docs[0]
        skin_type = _txt(prof.get("skinType"))
        day_steps = prof.get("daySteps") or []
        night_steps = prof.get("nightSteps") or []
        day_count = len(day_steps) if isinstance(day_steps, list) else 0
        night_count = len(night_steps) if isinstance(night_steps, list) else 0
        if skin_type:
            rows.append(
                {
                    "done": True,
                    "main": f"Skin type: {skin_type}",
                    "sub": "",
                    "tag": "",
                }
            )
        if day_count:
            rows.append(
                {
                    "done": True,
                    "main": "Morning routine",
                    "sub": f"{day_count} step{'s' if day_count != 1 else ''}",
                    "tag": "AM",
                }
            )
        if night_count:
            rows.append(
                {
                    "done": True,
                    "main": "Night routine",
                    "sub": f"{night_count} step{'s' if night_count != 1 else ''}",
                    "tag": "PM",
                }
            )
        steps_total = day_count + night_count
        if steps_total:
            summary = (
                f"Your routine has {day_count} morning + "
                f"{night_count} night step{'s' if steps_total != 1 else ''}."
            )
        elif skin_type:
            summary = "Your skin profile is set. Add routine steps to track."
    return _card(
        module="skincare",
        title="Skincare",
        icon="spa",
        summary=summary,
        count_done=len(rows),
        count_total=len(rows),
        rows=rows,
        open_key="skincare",
    )


# =========================
# DISPATCH
# =========================
_BUILDERS = {
    "medicines": _medicines,
    "bills": _bills,
    "events": _events,
    "events_upcoming": _events_upcoming,
    "meals": _meals,
    "workout": _workout,
    "skincare": _skincare,
}


def build_module_summary(module: str, user_id: str) -> Dict[str, Any]:
    """Return a module_card envelope for `module`, or {} if unsupported."""
    builder = _BUILDERS.get(_txt(module).lower())
    if not builder or not _txt(user_id):
        return {}
    return builder(user_id)
