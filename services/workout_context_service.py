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
    condition = str(raw.get("condition") or "").lower()
    temp = raw.get("temperature") or raw.get("temp_c")
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
        "condition": condition or "unknown",
        "temperature": temp,
        "humidity": humidity,
        "recommendation": recommendation,
    }


def build_workout_context(user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    weather = _weather_context(payload)
    duration = payload.get("duration") or payload.get("duration_minutes") or 20
    try:
        duration = int(duration)
    except Exception:
        duration = 20

    location = str(payload.get("location") or "").lower().strip()
    if not location:
        location = "home" if weather["recommendation"] in {"indoor", "shorter"} else "home"

    equipment = str(payload.get("equipment") or "none").lower().strip()
    if equipment in {"no", "nothing", "none available"}:
        equipment = "none"

    return {
        "user_id": user_id,
        "goal": str(payload.get("goal") or "general_fitness").lower().strip(),
        "gender": str(payload.get("gender") or "").lower().strip() or None,
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
