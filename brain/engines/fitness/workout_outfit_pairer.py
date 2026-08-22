from __future__ import annotations

from typing import Any, Dict, List


def pair_workout_outfit(context: Dict[str, Any], workout: Dict[str, Any]) -> Dict[str, Any]:
    location = str(context.get("location") or "home").lower()
    weather = str((context.get("weather_context") or {}).get("condition") or "").lower()
    intensity = str(workout.get("intensity") or workout.get("energy_level") or "medium")
    duration = int(workout.get("duration_minutes") or workout.get("duration_min") or 20)

    top = "breathable tee"
    bottom = "training shorts"
    footwear = "training shoes"
    accessories: List[str] = ["water bottle"]

    if location in {"yoga", "home"} and intensity in {"low", "gentle", "easy"}:
        bottom = "stretch joggers"
        footwear = "barefoot or light trainers"
    if location in {"outdoor", "park", "run"}:
        accessories.append("cap if outdoors")
    if "rain" in weather:
        accessories.append("light layer")
    if "humid" in weather or "hot" in weather:
        top = "lightweight breathable tee"
        if duration > 20:
            accessories.append("small towel")

    return {
        "top": top,
        "bottom": bottom,
        "footwear": footwear,
        "accessories": accessories,
        "why": "Breathable, low-friction pieces suit the session intensity and keep prep simple.",
    }
