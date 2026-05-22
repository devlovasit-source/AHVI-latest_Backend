from __future__ import annotations

from typing import Any, Dict, List


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_domain(value: Any) -> str:
    domain = _text(value).lower().replace("-", "_")
    aliases = {
        "meal": "diet",
        "meals": "diet",
        "medical": "medi",
        "meds": "medi",
        "medicine": "medi",
        "planner": "calendar",
        "planning": "calendar",
        "event": "calendar",
        "events": "calendar",
    }
    return aliases.get(domain, domain or "chat")


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
    return {
        "success": True,
        "type": "module_response",
        "domain": domain,
        "module": domain,
        "message": message,
        "message_text": message,
        "response": message,
        "cards": cards or [],
        "chips": chips or [],
        "data": payload,
    }


async def handle_diet_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    lower = message.lower()
    if ("weekly" in lower or "week" in lower) and "vegan" in lower:
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
            "I can help with meal ideas. A balanced option is protein, fiber-rich carbs, vegetables "
            "or fruit, and enough fluids. Tell me your goal, diet preference, or meal timing for a "
            "sharper suggestion."
        )
    return _envelope(domain="diet", message=reply, chips=["High protein breakfast", "Pre-workout meal", "Hydration"])


async def handle_skincare_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    lower = message.lower()
    if any(token in lower for token in ("spf", "sunscreen", "sun screen", "sunblock")):
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
            "A safe skincare baseline is gentle cleanser, moisturizer, and broad-spectrum SPF in the "
            "morning, then cleanser and moisturizer at night. Tell me your skin type or concern for a "
            "more tailored routine."
        )
    return _envelope(domain="skincare", message=reply, chips=["Morning routine", "Night routine", "SPF help"])


async def handle_calendar_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    lower = message.lower()
    if "plan my day" in lower or "plan day" in lower:
        reply = (
            "Here is a simple day plan: handle the most important task first, keep one focused admin block, leave a buffer before evening commitments, "
            "and add only the items you truly want reminders for."
        )
    elif "prioritize" in lower:
        reply = (
            "Prioritize by urgency and energy: do time-sensitive work first, then high-value tasks, then small admin. "
            "If you share your list, I will sort it into today, later, and optional."
        )
    elif any(token in lower for token in ("meeting", "appointment", "call", "at ")):
        reply = (
            "I can turn that into a planner entry. Add the title, time, and any location or dress code, then tap Add plan to save it."
        )
    elif "add plan" in lower:
        reply = "Tell me the plan in one line, for example: 'Client meeting at 9 PM'."
    else:
        reply = (
            "Tell me what you are planning, the time, and the category. I will turn it into a clear planner item."
        )
    return _envelope(domain="calendar", message=reply, chips=["Plan my day", "Prioritize tasks", "Add plan"])


async def handle_medi_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
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
    return _envelope(domain="medi", message=reply, chips=["Due today", "Missed logs", "Reminder help"])


async def handle_bills_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    bills = context.get("bills")
    coupons = context.get("coupons")
    bill_count = len(bills) if isinstance(bills, list) else 0
    coupon_count = len(coupons) if isinstance(coupons, list) else 0
    reply = (
        f"I can help review your bills. I see {bill_count} bills and {coupon_count} coupons in context. "
        "Ask me to summarize spending, find the top category, or plan which bills to handle first."
    )
    return _envelope(domain="bills", message=reply, chips=["Summarize bills", "Top category", "Coupons"])


async def handle_fitness_chat(message: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    lower = message.lower()
    if "hiit" in lower:
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
            "Tell me workout type, time available, intensity, equipment, and any constraints. I will build the session around those details instead of giving a generic routine."
        )
    return _envelope(domain="fitness", message=reply, chips=["Quick workout", "Outfit help", "Stretching"])


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
