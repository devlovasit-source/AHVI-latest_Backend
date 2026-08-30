from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict

from services.location_weather_context import resolve_location_weather_context
from services.workout_card_service import get_today_workout_card
from services.workout_context_service import build_workout_context
from services.diet_service import get_diet_recommendation
from services.today_meal_plan_service import get_today_meal_plan
from services.adherence_read_service import (
    get_today_skincare_state_readonly,
    get_today_medicine_state_readonly,
)

logger = logging.getLogger("ahvi.services.home_summary_service")

CONTEXT_VERSION = "2.2.0"

def _str_or(value, fallback: str) -> str:
    text = str(value if value is not None else "").strip()
    return text or fallback


def _obfuscate_uid(user_id: str) -> str:
    if not user_id:
        return "anonymous"
    return f"usr_{user_id[:8]}..." if len(user_id) > 8 else user_id

async def generate_home_summary(
    user_id: str,
    user_profile: dict,
    timezone_override: str | None = None,
    request_id: str = ""
) -> dict:
    """
    GET /api/home/today-summary
    Assembles concise, read-only Home-ready summaries for all 5 cards.
    """
    start_time = time.perf_counter()
    obfuscated_uid = _obfuscate_uid(user_id)

    logger.info(
        "home.summary.requested request_id=%s user_id=%s timezone_override=%s",
        request_id, obfuscated_uid, timezone_override
    )

    # 1. Central Timezone and local date resolving using standard ZoneInfo
    timezone_candidate = timezone_override or user_profile.get("timezone") or "Asia/Kolkata"
    try:
        tz = ZoneInfo(timezone_candidate)
        resolved_timezone = timezone_candidate
    except Exception:
        tz = ZoneInfo("Asia/Kolkata")
        resolved_timezone = "Asia/Kolkata"

    local_now = datetime.now(timezone.utc).astimezone(tz)
    local_date = local_now.date()
    local_date_str = local_date.isoformat()
    local_hour = local_now.hour

    # 2. Canonical Location / Weather
    weather_used = False
    location_used = False
    location_source = "none"
    weather_status = "unavailable"
    weather_timestamp = None
    weather_provider = "open-meteo"
    fallback_used = False
    profile_used = False

    try:
        resolved_context = resolve_location_weather_context(
            user_id=user_id, request_data={}, profile=user_profile
        )
        weather_info = resolved_context.get("weather", {})
        location_info = resolved_context.get("location", {})

        weather_status = weather_info.get("status") or "unavailable"
        weather_timestamp = weather_info.get("weather_timestamp") or weather_info.get("observed_at")
        weather_provider = weather_info.get("provider") or "open-meteo"

        location_used = bool(location_info and location_info.get("status") == "available")
        location_source = location_info.get("source") or "none"

    except Exception as exc:
        logger.warning("Failed resolving weather/location context: %s", exc)
        weather_info = {"status": "unavailable"}
        resolved_context = {}

    cards = {}
    partial_failures = []

    # -------------------------------------------------------------
    # 1. WEAR CARD
    # -------------------------------------------------------------
    try:
        temp = weather_info.get("temperature") if weather_info.get("temperature") is not None else weather_info.get("temp_c")
        condition = weather_info.get("condition") or "clear"

        style_dna = user_profile.get("style_dna") or user_profile.get("styleDNA") or {}
        primary_aesthetic = style_dna.get("primary_aesthetic")
        pref_colors = style_dna.get("preferred_colors")
        fav_color = pref_colors[0] if pref_colors else "neutral"

        # Explicit None checks to preserve 0°C
        weather_ok = (weather_status == "available" and temp is not None)
        dna_ok = bool(primary_aesthetic or pref_colors)

        if dna_ok:
            profile_used = True

        if not weather_ok and not dna_ok:
            # Wear must not fabricate personalized ready content when both weather and style DNA are missing
            fallback_used = True
            headline = "Style recommendation"
            context_text = "Open the module to continue"

            cards["wear"] = {
                "headline": headline,
                "context": context_text,
                "status": "unavailable",
                "action": "open_daily_wear",
                "entity_id": None,
                "available": False,
                "reason": "style_context_unavailable",
                "source": "style_context"
            }
        else:
            aesthetic_label = str(primary_aesthetic or "minimalist").lower().strip()
            if weather_ok:
                weather_used = True
                if temp >= 32:
                    headline = "Light, breathable layers"
                    context_text = f"Very hot {temp}°C · Ready for today"
                elif temp <= 15:
                    headline = "Layered warm outerwear"
                    context_text = f"Cold {temp}°C · Ready for today"
                else:
                    headline = f"Classic {aesthetic_label} separates"
                    context_text = f"Mild {temp}°C · Ready for today"
            else:
                headline = f"Polished {aesthetic_label} look"
                context_text = "Comfortable classics · Ready for today"

            cards["wear"] = {
                "headline": headline,
                "context": context_text,
                "status": "ready",
                "action": "open_daily_wear",
                "entity_id": None,
                "available": True,
                "reason": None,
                "source": "style_context"
            }
    except Exception as exc:
        logger.exception("Wear card generation failed")
        partial_failures.append("wear")
        cards["wear"] = {
            "headline": "Recommendation unavailable",
            "context": "Open the module to continue",
            "status": "unavailable",
            "action": "open_daily_wear",
            "entity_id": None,
            "available": False,
            "reason": "style_context_error",
            "source": "style_context"
        }

    # -------------------------------------------------------------
    # 2. MOVE CARD (Workout)
    # -------------------------------------------------------------
    try:
        workout_context = build_workout_context(user_id, {
            "goal": "general_fitness",
            "duration": 12,
            "location": "home",
            "equipment": "none",
            "profile": user_profile,
            "weather_context": weather_info,
            "location_context": resolved_context.get("location"),
        })
        workout_card = get_today_workout_card(user_id=user_id, profile=user_profile, context=workout_context)
        if workout_card and workout_card.get("id"):
            cards["move"] = {
                "headline": workout_card.get("title") or "12-minute mobility session",
                "context": workout_card.get("why_this") or "A focused workout that fits your day",
                "status": "ready",
                "action": "open_fitness",
                "entity_id": workout_card.get("id"),
                "available": True,
                "reason": None,
                "source": "workout"
            }
        else:
            cards["move"] = {
                "headline": "Recommendation unavailable",
                "context": "Open the module to continue",
                "status": "unavailable",
                "action": "open_fitness",
                "entity_id": None,
                "available": False,
                "reason": "no_workout_found",
                "source": "workout"
            }
    except Exception as exc:
        logger.exception("Move card generation failed")
        partial_failures.append("move")
        cards["move"] = {
            "headline": "Recommendation unavailable",
            "context": "Open the module to continue",
            "status": "unavailable",
            "action": "open_fitness",
            "entity_id": None,
            "available": False,
            "reason": "workout_service_error",
            "source": "workout"
        }

    # -------------------------------------------------------------
    # 3. EAT CARD (Diet)
    # -------------------------------------------------------------
    try:
        eat_card = None

        # A. Today's persisted meal plan wins. Read-only; never crashes the card.
        try:
            plan_res = get_today_meal_plan(user_id=user_id, local_now=local_now)
        except Exception:
            logger.exception("Today meal plan lookup failed")
            plan_res = None

        if plan_res and plan_res.get("status") == "ready" and plan_res.get("meal"):
            meal = plan_res["meal"]
            meal_type_label = (_str_or(meal.get("type"), "Meal")).title()
            cards["eat"] = {
                "headline": f"{meal_type_label} from today's plan",
                "context": _str_or(meal.get("name"), "Today's meal"),
                "status": "ready",
                "action": "open_diet",
                "entity_id": plan_res.get("plan_id"),
                "available": True,
                "reason": None,
                "source": "meal_plan"
            }
            eat_card = cards["eat"]

        # C. No same-day plan -> preserve the existing recommendation path.
        if eat_card is None:
            diet_res = get_diet_recommendation(
                user_id=user_id,
                local_date=local_date,
                local_hour=local_hour,
                meal_type=None,
                profile=user_profile
            )
            if diet_res and diet_res.get("status") == "ready" and diet_res.get("recommendation"):
                profile_used = True
                rec = diet_res["recommendation"]
                cards["eat"] = {
                    "headline": f"Balanced {rec['meal_type']} recommendation",
                    "context": rec["title"],
                    "status": "ready",
                    "action": "open_diet",
                    "entity_id": rec["id"],
                    "available": True,
                    "reason": None,
                    "source": "diet_recommendation"
                }
                eat_card = cards["eat"]

        # D. Neither plan nor recommendation resolved.
        if eat_card is None:
            cards["eat"] = {
                "headline": "No meal planned yet",
                "context": "Open Diet to create today's plan",
                "status": "unavailable",
                "action": "open_diet",
                "entity_id": None,
                "available": False,
                "reason": "today_meal_plan_missing",
                "source": "meal_plan"
            }
    except Exception as exc:
        logger.exception("Eat card generation failed")
        partial_failures.append("eat")
        cards["eat"] = {
            "headline": "Recommendation unavailable",
            "context": "Open the module to continue",
            "status": "unavailable",
            "action": "open_diet",
            "entity_id": None,
            "available": False,
            "reason": "diet_service_error",
            "source": "diet_recommendation"
        }

    # -------------------------------------------------------------
    # 4. CARE CARD (Skincare only)
    # -------------------------------------------------------------
    try:
        skincare_state = get_today_skincare_state_readonly(user_id, local_now)
        if skincare_state.get("status") != "unavailable" and skincare_state.get("status") != "error":
            profile_used = True
            status_label = skincare_state["status"]
            routine = skincare_state["routine"]
            cards["care"] = {
                "headline": f"{str(routine).title()} skincare routine",
                "context": "Ready to start" if status_label == "ready" else "Completed for today",
                "status": status_label,
                "action": "open_skincare",
                "entity_id": skincare_state.get("entity_id"), # Projected is None
                "available": True,
                "reason": None,
                "source": skincare_state.get("source")
            }
        else:
            cards["care"] = {
                "headline": "Recommendation unavailable",
                "context": "Open the module to continue",
                "status": "unavailable",
                "action": "open_skincare",
                "entity_id": None,
                "available": False,
                "reason": skincare_state.get("reason") or "skincare_routine_unavailable",
                "source": "skincare"
            }
    except Exception as exc:
        logger.exception("Care card generation failed")
        partial_failures.append("care")
        cards["care"] = {
            "headline": "Recommendation unavailable",
            "context": "Open the module to continue",
            "status": "unavailable",
            "action": "open_skincare",
            "entity_id": None,
            "available": False,
            "reason": "skincare_service_error",
            "source": "skincare"
        }

    # -------------------------------------------------------------
    # 5. MEDICINE CARD
    # -------------------------------------------------------------
    try:
        med_state = get_today_medicine_state_readonly(user_id, local_now)
        if med_state.get("status") != "unavailable" and med_state.get("status") != "error":
            profile_used = True
            status_label = med_state["status"]

            # Formulate safe clinical-neutral headings
            if status_label == "overdue":
                count = med_state["overdue_count"]
                headline = "A medicine reminder is overdue" if count == 1 else f"{count} medicine reminders are overdue"
            elif status_label == "due_now":
                count = med_state["due_now_count"]
                headline = "A medicine reminder is due now" if count == 1 else f"{count} medicine reminders are due now"
            elif status_label == "due_soon":
                count = med_state["due_soon_count"]
                headline = "A medicine reminder is due soon" if count == 1 else f"{count} medicine reminders are due soon"
            elif status_label == "later":
                count = med_state["later_count"]
                headline = "A medicine reminder is scheduled for later today" if count == 1 else f"{count} medicine reminders are scheduled for later today"
            else: # completed
                headline = "Today's medicine reminders are completed"

            cards["medicine"] = {
                "headline": headline,
                "context": "Open Medi Tracker to review it" if status_label != "completed" else "Up to date for today",
                "status": status_label,
                "action": "open_medi",
                "entity_id": med_state.get("entity_id"), # Projected is None
                "available": True,
                "reason": None,
                "source": med_state.get("source")
            }
        else:
            cards["medicine"] = {
                "headline": "Recommendation unavailable",
                "context": "Open the module to continue",
                "status": "unavailable",
                "action": "open_medi",
                "entity_id": None,
                "available": False,
                "reason": med_state.get("reason") or "medicine_schedule_unavailable",
                "source": "medicine"
            }
    except Exception as exc:
        logger.exception("Medicine card generation failed")
        partial_failures.append("medicine")
        cards["medicine"] = {
            "headline": "Recommendation unavailable",
            "context": "Open the module to continue",
            "status": "unavailable",
            "action": "open_medi",
            "entity_id": None,
            "available": False,
            "reason": "medicine_service_error",
            "source": "medicine"
        }

    # 3. Formulate context usage
    context_usage = {
        "location_used": location_used,
        "location_source": location_source,
        "weather_used": weather_used,
        "weather_status": weather_status,
        "weather_reason": weather_info.get("reason"),
        "weather_timestamp": weather_timestamp,
        "weather_provider": weather_provider,
        "calendar_used": False,
        "profile_used": profile_used,
        "fallback_used": fallback_used
    }

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # Include unavailable cards in partial-result logging
    unavailable_cards = [name for name, card in cards.items() if card.get("status") == "unavailable"]
    logged_failures = sorted(list(set(partial_failures + unavailable_cards)))

    if logged_failures:
        logger.warning(
            "home.summary.partial request_id=%s user_id=%s local_date=%s timezone=%s failures=%s latency_ms=%s",
            request_id, obfuscated_uid, local_date_str, resolved_timezone, logged_failures, elapsed_ms
        )
    else:
        logger.info(
            "home.summary.generated request_id=%s user_id=%s local_date=%s timezone=%s latency_ms=%s",
            request_id, obfuscated_uid, local_date_str, resolved_timezone, elapsed_ms
        )

    return {
        "date": local_date_str,
        "timezone": resolved_timezone,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "context_version": CONTEXT_VERSION,
        "cards": cards,
        "context_usage": context_usage
    }
