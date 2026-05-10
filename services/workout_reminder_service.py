from __future__ import annotations

from typing import Any, Dict, List


def build_workout_reminders(context: Dict[str, Any], workout: Dict[str, Any]) -> List[str]:
    reminders: List[str] = []
    time_of_day = str(context.get("time_of_day") or "")
    weather = context.get("weather_context") or {}
    weather_reco = str(weather.get("recommendation") or "")

    if time_of_day == "evening":
        reminders.append("Start before the evening slips away.")
    elif time_of_day == "morning":
        reminders.append("Do this before the day gets crowded.")

    if weather_reco == "indoor":
        reminders.append("Keep it indoors today and keep water nearby.")
    elif weather_reco == "shorter":
        reminders.append("Keep the session short and hydrate before you start.")
    else:
        reminders.append("Keep water nearby.")

    return reminders[:2]
