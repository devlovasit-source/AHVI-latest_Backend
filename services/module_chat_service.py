from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from services.appwrite_proxy import AppwriteProxy


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_domain(value: Any) -> str:
    domain = _text(value).lower().replace("-", "_")
    aliases = {
        "meal": "diet",
        "meals": "diet",
        "meal_planner": "diet",
        "diet": "diet",
        "workout": "fitness",
        "fitness": "fitness",
        "medical": "medi",
        "meds": "medi",
        "medicine": "medi",
        "medicines": "medi",
        "planner": "calendar",
        "plan": "calendar",
        "planning": "calendar",
        "calendar": "calendar",
        "event": "calendar",
        "events": "calendar",
    }
    return aliases.get(domain, domain or "chat")


_QUICK_ACTIONS: Dict[str, List[str]] = {
    "diet": ["Create today's meal plan", "High-protein meals", "Light dinner ideas", "Open Meals"],
    "fitness": ["Home workout", "Gym workout", "Workout outfit", "Recovery meal"],
    "medi": ["Mark taken", "Set reminder", "Open Medicines"],
    "bills": ["Mark paid", "Set reminder", "Open Bills"],
    "calendar": ["Plan my day", "Add event", "Prep for tomorrow"],
    "skincare": ["Create routine", "Morning routine", "Evening routine", "Open Skincare"],
    "planner": ["Packing checklist", "Plan outfits", "Weather prep", "Save trip plan"],
    "chat": ["Style me", "Plan my day", "Summarize"],
}


_OPEN_MODULES: Dict[str, Dict[str, str]] = {
    "diet": {"module": "meals", "route": "/organize/meals"},
    "fitness": {"module": "workout", "route": "/organize/workout"},
    "medi": {"module": "medicines", "route": "/organize/medicines"},
    "bills": {"module": "bills", "route": "/organize/bills"},
    "calendar": {"module": "calendar", "route": "/organize/calendar"},
    "skincare": {"module": "skincare", "route": "/organize/skincare"},
    "planner": {"module": "planner", "route": "/organize/calendar"},
}


def _quick_actions(domain: str) -> List[str]:
    return list(_QUICK_ACTIONS.get(domain, _QUICK_ACTIONS["chat"]))


def _envelope(
    *,
    domain: str,
    message: str,
    chips: List[str] | None = None,
    cards: List[Dict[str, Any]] | None = None,
    data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = data or {}
    payload.setdefault("message", message)
    actions = chips or _quick_actions(domain)
    open_module = _OPEN_MODULES.get(domain)
    payload.setdefault("module", domain)
    payload.setdefault("intent", domain)
    if open_module:
        payload.setdefault("open_module", open_module)
    return {
        "success": True,
        "type": "module_response",
        "domain": domain,
        "module": domain,
        "intent": payload.get("intent") or domain,
        "message": message,
        "message_text": message,
        "response": message,
        "cards": cards or [],
        "chips": actions,
        "quick_actions": actions,
        "cta": open_module,
        "open_module": open_module,
        "data": payload,
    }


def _looks_like_event_create(message: str) -> bool:
    text = str(message or "").lower().strip()
    if not text or text in {"add event", "view events", "open events", "open calendar"}:
        return False
    event_tokens = (
        "appointment",
        "doctor",
        "dentist",
        "meeting with",
        "call with",
        "call at",
        "interview",
        "remind me",
        "schedule meeting",
        "birthday",
    )
    date_tokens = (
        "today",
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "jan",
        "january",
        "feb",
        "february",
        "mar",
        "march",
        "apr",
        "april",
        "may",
        "jun",
        "june",
        "jul",
        "july",
        "aug",
        "august",
        "sep",
        "sept",
        "september",
        "oct",
        "october",
        "nov",
        "november",
        "dec",
        "december",
    )
    has_date = any(token in text for token in date_tokens) or bool(re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\b", text))
    has_time = bool(re.search(r"\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", text)) or any(
        token in text for token in (" morning", " afternoon", " evening", " night")
    )
    return any(token in text for token in event_tokens) and (has_date or has_time)


def _format_event_when(start_time: Any) -> str:
    raw = _text(start_time)
    if not raw:
        return "your calendar"
    try:
        start = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return raw
    now = datetime.now(start.tzinfo or timezone.utc)
    if start.date() == now.date():
        day = "today"
    elif start.date() == (now + timedelta(days=1)).date():
        day = "tomorrow"
    else:
        day = start.strftime("%d %B")
    return f"{day} at {start.strftime('%I:%M %p').lstrip('0')}"


def _calendar_event_created_envelope(event: Dict[str, Any]) -> Dict[str, Any]:
    title = _text(event.get("title")) or "Event"
    when = _format_event_when(event.get("start_time"))
    message = f"{title} added for {when}."
    card = {
        "type": "module_card",
        "module": "calendar",
        "title": title,
        "subtitle": when,
        "summary": _text(event.get("description")) or when,
        "cta": {"label": "Open calendar", "module": "calendar", "route": "calendar"},
        "open_module": "calendar",
    }
    return _envelope(
        domain="calendar",
        message=message,
        chips=["View events", "Add reminder", "Plan my day"],
        cards=[card],
        data={
            "intent": "calendar_event_created",
            "event": event,
            "refresh": "calendar",
        },
    ) | {
        "intent": "calendar_event_created",
        "refresh": "calendar",
        "card": card,
    }


def _event_needs_time_envelope() -> Dict[str, Any]:
    message = "What time should I save this for?"
    return _envelope(
        domain="calendar",
        message=message,
        chips=["Today 6 PM", "Tomorrow 9 AM", "Open calendar"],
        data={"intent": "event_needs_time", "refresh": "calendar"},
    ) | {"intent": "event_needs_time", "refresh": "calendar"}


def _safe_docs(resource: str, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    try:
        rows = AppwriteProxy().list_documents(resource, user_id=user_id, limit=limit)
    except Exception:
        return []
    if isinstance(rows, dict):
        rows = rows.get("documents") or rows.get("items") or []
    return [row for row in rows or [] if isinstance(row, dict)]


def _row_title(row: Dict[str, Any]) -> str:
    return _text(
        row.get("title")
        or row.get("name")
        or row.get("label")
        or row.get("description")
        or row.get("summary")
    )


def _plan_my_day_envelope(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    from services.calendar_service import list_today_calendar_events

    try:
        events = list_today_calendar_events(user_id=user_id)
    except Exception:
        events = []
    plans = _safe_docs("plans", user_id, limit=20)
    meal_plans = _safe_docs("meal_plans", user_id, limit=10)

    morning: List[str] = []
    afternoon: List[str] = []
    evening: List[str] = []

    for event in events:
        title = _row_title(event) or "Calendar event"
        when = _format_event_when(event.get("start_time"))
        line = f"{title} - {when}"
        try:
            start = datetime.fromisoformat(_text(event.get("start_time")).replace("Z", "+00:00"))
            hour = start.hour
        except Exception:
            hour = 12
        if hour < 12:
            morning.append(line)
        elif hour < 17:
            afternoon.append(line)
        else:
            evening.append(line)

    for row in plans[:4]:
        title = _row_title(row)
        if title:
            afternoon.append(title)

    meal_title = _row_title(meal_plans[0]) if meal_plans else ""
    if meal_title:
        evening.append(f"Meal plan: {meal_title}")

    has_data = bool(events or plans or meal_plans)
    if has_data:
        parts = [f"You have {len(events)} calendar event{'s' if len(events) != 1 else ''} today."]
        if plans:
            parts.append(f"I also found {len(plans)} saved plan{'s' if len(plans) != 1 else ''}.")
        if meal_plans:
            parts.append("Your latest meal plan is available for food timing.")
        reply = " ".join(parts)
    else:
        reply = (
            "Your day is open in AHVI right now. Add a calendar event, meal plan, or priority task and I can turn it into a morning, afternoon, and evening flow."
        )

    cards = [
        {"title": "Morning", "items": morning or ["No saved morning events yet."]},
        {"title": "Afternoon", "items": afternoon or ["No saved afternoon plans yet."]},
        {"title": "Evening", "items": evening or ["No saved evening plans yet."]},
    ]
    return _envelope(
        domain="calendar",
        message=reply,
        chips=["Add event", "View events", "Eat today", "Workout today"],
        cards=cards,
        data={
            "intent": "plan_my_day",
            "refresh": "calendar",
            "events_count": len(events),
            "plans_count": len(plans),
            "meal_plans_count": len(meal_plans),
            "sections": {"morning": morning, "afternoon": afternoon, "evening": evening},
        },
    ) | {"refresh": "calendar"}


async def handle_diet_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    lower = message.lower()
    if "recovery meal" in lower or ("recovery" in lower and "meal" in lower):
        reply = (
            "For recovery, pair protein, carbs, and fluids: paneer/tofu/chicken or dal with rice, "
            "curd or yogurt, fruit, and water. Keep it easy to digest if you trained hard."
        )
    elif ("weekly" in lower or "week" in lower) and "vegan" in lower:
        reply = (
            "Weekly vegan plan: keep breakfast simple with oats, fruit, nuts, or tofu scramble. "
            "Rotate lunches between dal-rice bowls, chickpea salad, tofu wraps, and vegetable khichdi. "
            "For dinner, use lighter options like soup with beans, millet upma, stir-fry tofu, or lentil pasta. "
            "Add fruit, nuts, and enough water so the plan feels sustainable."
        )
    elif any(x in lower for x in ("what should i eat", "eat today", "meal today", "today")):
        reply = (
            "For today, keep it balanced: breakfast with protein plus fiber, lunch with dal/paneer/tofu/chicken plus rice or roti and vegetables, "
            "an evening fruit or yogurt snack, and a lighter dinner with protein, vegetables, and enough fluids."
        )
    elif "low" in lower and "carb" in lower and ("dinner" in lower or "night" in lower):
        reply = (
            "For a low-carb dinner, build the plate around protein plus vegetables: grilled paneer, tofu, eggs, fish, or chicken with sauteed greens, salad, or soup. "
            "Keep sauces light and skip large rice/roti portions tonight."
        )
    elif "weight gain" in lower or "gain weight" in lower or "bulk" in lower:
        reply = (
            "For healthy weight gain, add a steady calorie surplus with protein at each meal: milk or curd, eggs/paneer/tofu/chicken, dal, rice or roti, nuts, and smoothies. "
            "Aim for consistency rather than very large single meals."
        )
    elif "non vegetarian" in lower or "non-vegetarian" in lower or "chicken" in lower or "fish" in lower:
        reply = (
            "A non-vegetarian plan can use eggs or Greek yogurt at breakfast, chicken/fish with rice or roti at lunch, and lean protein with vegetables at dinner. "
            "Balance it with fiber, fluids, and not every meal needs to be heavy."
        )
    elif "weekly" in lower or "week" in lower:
        reply = (
            "For a weekly meal plan, rotate protein sources and repeat easy staples: two high-protein breakfasts, two simple lunches, two lighter dinners, and one flexible meal. "
            "Tell me veg/non-veg and goal if you want a day-by-day plan."
        )
    elif "protein" in lower and "breakfast" in lower:
        reply = (
            "For a high-protein breakfast, try Greek yogurt with fruit and nuts, eggs with "
            "whole-grain toast, paneer or tofu scramble, or oats with milk and seeds. Keep it "
            "balanced with fiber and fluids."
        )
    elif "pre" in lower and "workout" in lower:
        reply = (
            "Before a workout, choose easy carbs plus a little protein: banana with yogurt, "
            "toast with peanut butter, or oats. Leave enough time to digest before training."
        )
    elif "post" in lower and "workout" in lower:
        reply = (
            "After a workout, pair protein with carbs and fluids. Rice with dal, eggs and toast, "
            "yogurt with fruit, or tofu with a grain bowl are simple options."
        )
    elif "hydrat" in lower or "water" in lower:
        reply = (
            "Sip water through the day, and consider electrolytes when you sweat heavily or train "
            "in heat. Thirst, urine color, and workout intensity are useful practical cues."
        )
    elif "weight loss" in lower or "lose weight" in lower:
        reply = (
            "For weight-loss meal ideas, build plates around lean protein, vegetables or fruit, "
            "fiber-rich carbs, and a small amount of healthy fat. Keep it sustainable and avoid "
            "extreme restriction."
        )
    else:
        reply = (
            "I can help build a meal plan from here. Tell me your goal, diet preference, or meal "
            "timing, or choose a quick action to create today's meals."
        )
    return _envelope(domain="diet", message=reply, chips=_quick_actions("diet"))


async def handle_skincare_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    lower = message.lower()
    if "morning routine" in lower or "morning skincare" in lower:
        reply = "Your morning skincare routine is not set up yet. I can help create a simple 3-step routine."
    elif "evening routine" in lower or "night routine" in lower or "night skincare" in lower:
        reply = "Your evening skincare routine is not set up yet. I can help create a simple cleanse, treat, and moisturize routine."
    elif "create routine" in lower:
        reply = "Let us create a simple routine: choose your skin type, pick your main concern, then save morning and night steps."
    elif "open skincare" in lower:
        reply = "Opening Skincare."
    elif any(token in lower for token in ("spf", "sunscreen", "sun screen", "sunblock")):
        reply = (
            "For SPF, choose broad-spectrum SPF 30 or higher and apply it every morning. If your "
            "skin is oily or acne-prone, try a lightweight non-comedogenic gel or fluid. If your "
            "skin is dry or sensitive, look for a fragrance-free moisturizing sunscreen. Reapply "
            "every 2-3 hours outdoors."
        )
    elif "dry" in lower:
        reply = (
            "For dry skin, keep the routine simple: gentle cleanser, hydrating serum if you use one, "
            "moisturizer, and SPF in the morning. At night, cleanse and use a richer moisturizer."
        )
    elif "oily" in lower:
        reply = (
            "For oily skin, use a gentle cleanser, lightweight moisturizer, and non-comedogenic SPF. "
            "Avoid stripping the skin, because that can make oiliness feel worse."
        )
    elif "acne" in lower or "pimple" in lower or "breakout" in lower:
        reply = (
            "For acne-prone skin, keep it gentle: mild cleanser, lightweight moisturizer, and SPF. "
            "Introduce actives slowly. For painful, persistent, or worsening acne, check with a dermatologist."
        )
    elif "night" in lower:
        reply = (
            "A simple night routine: cleanse, apply a gentle hydrating step if you use one, then "
            "moisturize. If you use treatments, add only one at a time and watch how your skin reacts."
        )
    else:
        reply = (
            "Your skincare flow is ready to set up. I can create a simple morning or evening routine "
            "from your skin type, concern, and products."
        )
    return _envelope(domain="skincare", message=reply, chips=_quick_actions("skincare"))


async def handle_calendar_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    lower = message.lower()
    if "add event" in lower:
        reply = "Tell me the event name and date/time. For example: Birthday on 23 July or Doctor appointment tomorrow at 6 PM."
    elif "view events" in lower or "open events" in lower or "open calendar" in lower:
        reply = "Opening Calendar."
    elif "plan my day" in lower or "plan day" in lower:
        return _plan_my_day_envelope(message, context, user_id)
    elif "prioritize" in lower:
        reply = (
            "Prioritize by urgency and energy: do time-sensitive work first, then high-value tasks, then small admin. "
            "If you share your list, I will sort it into today, later, and optional."
        )
    elif _looks_like_event_create(message):
        from services.calendar_service import create_calendar_event, parse_plan_text_to_payload

        user_profile = context.get("user_profile") if isinstance(context.get("user_profile"), dict) else {}
        timezone_name = _text(context.get("timezone") or user_profile.get("timezone")) or "Asia/Kolkata"
        try:
            payload = parse_plan_text_to_payload(message, category="Plan", timezone_name=timezone_name)
            event = create_calendar_event(user_id, payload)
            return _calendar_event_created_envelope(event)
        except ValueError as exc:
            if str(exc) == "time_required":
                return _event_needs_time_envelope()
            reply = "I could not parse that event yet. Try: Doctor appointment tomorrow at 9 AM."
        except Exception:
            reply = "I tried to save that calendar event, but the calendar could not sync yet. Please try again."
    elif "add plan" in lower:
        reply = "Tell me the plan in one line, for example: 'Client meeting at 9 PM'."
    else:
        reply = (
            "Tell me what you are planning, or choose a quick action. I can organize your day, add an event, or prep tomorrow."
        )
    return _envelope(domain="calendar", message=reply, chips=_quick_actions("calendar"))


async def handle_medi_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    mark_result = _handle_medi_mark_taken(message, user_id)
    if mark_result:
        return mark_result

    meds = context.get("medications")
    count = len(meds) if isinstance(meds, list) else 0
    if "due" in message.lower() or "today" in message.lower():
        reply = (
            f"I can help review today's medication schedule. I see {count} medication entries in context. "
            "Use the tracker status as the source of truth, and check with your clinician before changing any dose."
        )
    else:
        reply = (
            "I can summarize your medication tracker, help spot missed logs, and organize reminder questions. "
            "I cannot diagnose or change dosages; for medical decisions, please consult your clinician."
        )
    return _envelope(domain="medi", message=reply, chips=_quick_actions("medi"))


def _handle_medi_mark_taken(message: str, user_id: str) -> Dict[str, Any] | None:
    text = _text(message)
    lower = text.lower()
    if not user_id or not any(
        token in lower
        for token in (
            "mark ",
            "taken",
            "took ",
            "i took",
            "medicine done",
            "meds done",
        )
    ):
        return None

    if not (
        "taken" in lower
        or "took" in lower
        or "done" in lower
        or "mark" in lower
    ):
        return None

    proxy = AppwriteProxy()
    try:
        docs = proxy.list_documents("meds", user_id=user_id, limit=100)
    except Exception:
        docs = []
    if not isinstance(docs, list) or not docs:
        return _envelope(
            domain="medi",
            message="I could not find a medicine to mark yet. Open Medicines to add one first.",
            chips=_quick_actions("medi"),
            data={"intent": "medicine_mark_taken_empty", "refresh": "medi"},
        )

    def med_name(doc: Dict[str, Any]) -> str:
        return _text(doc.get("name") or doc.get("title") or doc.get("medicine"))

    requested = lower
    for prefix in (
        "mark",
        "as taken",
        "taken",
        "took",
        "i took",
        "medicine",
        "medicines",
        "meds",
    ):
        requested = requested.replace(prefix, " ")
    requested = re.sub(r"[^a-z0-9 ]+", " ", requested)
    requested = re.sub(r"\s+", " ", requested).strip()

    selected = None
    if requested:
        for doc in docs:
            name = med_name(doc).lower()
            if name and (name in lower or any(part and part in name for part in requested.split())):
                selected = doc
                break
    if selected is None and len(docs) == 1:
        selected = docs[0]
    if selected is None:
        names = ", ".join(med_name(doc) for doc in docs[:3] if med_name(doc))
        return _envelope(
            domain="medi",
            message=f"Which medicine should I mark as taken? I found: {names or 'your tracked medicines'}.",
            chips=_quick_actions("medi"),
            data={"intent": "medicine_mark_taken_clarify"},
        )

    doc_id = _text(selected.get("$id") or selected.get("id"))
    name = med_name(selected) or "Medicine"
    now_iso = datetime.now(timezone.utc).isoformat()
    update: Dict[str, Any] = {"lastTaken": now_iso}
    left = selected.get("left")
    if isinstance(left, int):
        update["left"] = max(0, left - 1)
    try:
        if doc_id:
            proxy.update_document("meds", doc_id, update)
        proxy.create_document(
            "med_logs",
            {
                "userId": user_id,
                "medId": doc_id,
                "medName": name,
                "dose": _text(selected.get("dose")),
                "time": now_iso,
                "status": "taken",
            },
        )
    except Exception:
        return _envelope(
            domain="medi",
            message=f"I tried to mark {name} as taken, but the tracker could not sync yet. Refresh Medicines and try again.",
            chips=_quick_actions("medi"),
            data={"intent": "medicine_mark_taken_failed", "refresh": "medi"},
        )

    return _envelope(
        domain="medi",
        message=f"{name} marked as taken.",
        chips=_quick_actions("medi"),
        data={
            "intent": "medicine_marked_taken",
            "refresh": "medi",
            "medicine_id": doc_id,
            "medicine_name": name,
        },
    ) | {"refresh": "medi"}


async def handle_bills_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    bills = context.get("bills")
    coupons = context.get("coupons")
    bill_count = len(bills) if isinstance(bills, list) else 0
    coupon_count = len(coupons) if isinstance(coupons, list) else 0
    reply = (
        f"I can help review your bills. I see {bill_count} bills and {coupon_count} coupons in context. "
        "Ask me to summarize spending, find the top category, or plan which bills to handle first."
    )
    return _envelope(domain="bills", message=reply, chips=_quick_actions("bills"))


async def handle_fitness_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    lower = message.lower()
    if "home workout" in lower:
        reply = (
            "Home workout: 5-minute warm-up, then squats, push-ups, glute bridges, rows or towel pulls, "
            "and plank holds. Finish with stretching. Wear breathable tee, flexible bottoms, and training shoes."
        )
    elif "gym workout" in lower:
        reply = (
            "Gym workout: start with mobility, then leg press or squats, dumbbell rows, chest press, hamstring curl, "
            "and core. Keep supportive shoes and a breathable top."
        )
    elif "workout outfit" in lower:
        reply = (
            "Workout outfit: breathable tee, training shorts or joggers, supportive training shoes, and a water bottle. "
            "For outdoor heat, add a cap and lighter colors."
        )
    elif "hiit" in lower:
        reply = (
            "HIIT plan: warm up 5 minutes, then 30 seconds work / 30 seconds rest for squats, mountain climbers, push-ups, high knees, and plank jacks. "
            "Repeat 3-4 rounds, cool down 5 minutes, and wear breathable stretch pieces with secure footwear."
        )
    elif "strength" in lower or "muscle" in lower:
        reply = (
            "Strength plan: 3-4 sets each of squats, hip hinges, push movements, rows, and core. Keep reps controlled, rest 60-90 seconds, and progress slowly."
        )
    elif "yoga" in lower or "stretch" in lower:
        reply = (
            "Mobility plan: 5 minutes breathing, cat-cow, hip openers, hamstring stretch, thoracic rotations, and a gentle cooldown. Wear soft stretch pieces that do not restrict movement."
        )
    elif "outfit" in lower or "wardrobe" in lower:
        reply = (
            "For a workout outfit, choose breathable top, flexible bottom, and supportive footwear. If one slot is missing from wardrobe, add that piece before I pair a complete training look."
        )
    else:
        reply = (
            "I can build today's workout from your time, location, equipment, and intensity. Choose a quick action or tell me the session you want."
        )
    return _envelope(domain="fitness", message=reply, chips=_quick_actions("fitness"))


async def handle_style_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    reply = "I can help with that. Tell me the occasion, weather, and what you want to optimize for."
    return _envelope(domain="chat", message=reply, chips=["Style me", "Plan my day", "Summarize"])


async def handle_module_chat(payload: Dict[str, Any], user_id: str = "") -> Dict[str, Any]:
    domain = _normalize_domain(payload.get("domain") or payload.get("module"))
    message = _text(payload.get("message") or payload.get("text"))
    context = payload.get("context") or payload.get("context_data") or {}
    if not isinstance(context, dict):
        context = {"value": context}

    if domain == "diet":
        return await handle_diet_chat(message, context, user_id)
    if domain == "fitness":
        return await handle_fitness_chat(message, context, user_id)
    if domain == "skincare":
        return await handle_skincare_chat(message, context, user_id)
    if domain == "calendar":
        return await handle_calendar_chat(message, context, user_id)
    if domain == "medi":
        return await handle_medi_chat(message, context, user_id)
    if domain == "bills":
        return await handle_bills_chat(message, context, user_id)
    return await handle_style_chat(message, context, user_id)
