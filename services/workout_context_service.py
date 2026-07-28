from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


def _time_of_day(now: datetime | None = None) -> str:
    hour = (now or datetime.now(timezone.utc)).hour
    if hour < 11:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 22:
        return "evening"
    return "late"


def _weather_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("weather") or payload.get("weather_context") or {}
    if not isinstance(raw, dict):
        raw = {"condition": str(raw)}
    condition = str(
        raw.get("condition")
        or raw.get("weather_type")
        or raw.get("weatherType")
        or ""
    ).lower()
    temp = raw.get("temperature")
    if temp is None:
        temp = raw.get("temp_c")
    humidity = raw.get("humidity")
    recommendation = "normal"
    if "rain" in condition or "storm" in condition:
        recommendation = "indoor"
    elif "humid" in condition or (isinstance(humidity, (int, float)) and humidity >= 70):
        recommendation = "indoor"
        condition = condition or "humid"
    elif isinstance(temp, (int, float)) and temp >= 32:
        recommendation = "shorter"
        condition = condition or "hot"
    return {
        "status": "available" if condition or temp is not None or humidity is not None else "unavailable",
        "condition": condition or "unknown",
        "temperature": temp,
        "humidity": humidity,
        "recommendation": recommendation,
    }


def _normalize_gender(value: Any) -> str | None:
    raw = str(value or "").lower().strip()
    if not raw:
        return None
    if raw in {"m", "male", "man", "men", "mens", "boy"}:
        return "men"
    if raw in {"f", "female", "woman", "women", "womens", "girl"}:
        return "women"
    if raw in {"unisex", "neutral", "genderless", "any", "all", "universal"}:
        return "universal"
    return None


def _gender_from_payload(payload: Dict[str, Any]) -> str:
    direct = _normalize_gender(payload.get("gender"))
    if direct:
        return direct

    profile = payload.get("profile") or {}
    if not isinstance(profile, dict):
        profile = {}
    style_prefs = profile.get("stylePreferences") or profile.get("style_preferences") or {}
    if not isinstance(style_prefs, dict):
        style_prefs = {}

    for source in (profile, style_prefs):
        for key in ("fitness_gender", "style_gender", "gender", "preferred_gender", "target_gender"):
            gender = _normalize_gender(source.get(key))
            if gender:
                return gender

    return "universal"


def build_workout_context(user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    weather = _weather_context(payload)
    duration = payload.get("duration") or payload.get("duration_minutes") or 20
    try:
        duration = int(duration)
    except Exception:
        duration = 20

    location_value = payload.get("location")
    location = str(location_value if isinstance(location_value, str) else "").lower().strip()
    if not location:
        location = "home" if weather["recommendation"] in {"indoor", "shorter"} else "home"

    equipment = str(payload.get("equipment") or "none").lower().strip()
    if equipment in {"no", "nothing", "none available"}:
        equipment = "none"

    return {
        "user_id": user_id,
        "goal": str(payload.get("goal") or "general_fitness").lower().strip(),
        "gender": _gender_from_payload(payload),
        "duration": duration,
        "location": location,
        "equipment": equipment,
        "constraint": str(payload.get("constraint") or "").lower().strip() or None,
        "time_of_day": str(payload.get("time_of_day") or _time_of_day()).lower(),
        "weather_context": weather,
        "recent_skipped_workout_ids": payload.get("recent_skipped_workout_ids") or [],
        "available_time": payload.get("available_time"),
        "profile": payload.get("profile") or {},
        "calendar": payload.get("calendar") or {},
    }
