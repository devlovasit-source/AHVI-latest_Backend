from __future__ import annotations

import re
from typing import Any, Dict, List

from brain.engines.fitness.workout_outfit_pairer import pair_workout_outfit
from services.workout_reminder_service import build_workout_reminders


def _exercise_from_text(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    reps = None
    seconds = None
    reps_match = re.search(r"(\d+)\s*(?:reps?|each side|x)?", text, re.I)
    sec_match = re.search(r"(\d+)\s*(?:sec|second)", text, re.I)
    if sec_match:
        seconds = int(sec_match.group(1))
    elif reps_match:
        reps = int(reps_match.group(1))
    name = re.sub(r"\s+\d+.*$", "", text).strip() or text
    key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return {
        "name": name,
        "duration_seconds": seconds,
        "reps": reps,
        "sets": None,
        "asset_url": f"https://r2.ahvi.assets/workouts/exercises/{key}.gif",
    }


def _exercises(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    exercises: List[Dict[str, Any]] = []
    for block in session.get("cards") or []:
        for item in block.get("items") or []:
            exercises.append(_exercise_from_text(item))
    return exercises[:8]


def build_workout_card(session: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    duration = int(session.get("duration_min") or context.get("duration") or 20)
    equipment = ", ".join(session.get("equipment") or []) or "none"
    location = ", ".join(session.get("location") or [context.get("location") or "home"])
    intensity = session.get("energy_level") or "medium"
    card = {
        "id": session.get("key"),
        "type": "workout_card",
        "title": session.get("title") or "Today’s Workout",
        "subtitle": f"{duration} min · {equipment}",
        "duration_minutes": duration,
        "intensity": intensity,
        "location": location,
        "equipment": equipment,
        "hero_asset_url": f"https://r2.ahvi.assets/workouts/heroes/{session.get('key')}.png",
        "exercises": _exercises(session),
        "repeat_instruction": "Repeat the blocks with clean form.",
        "why_this": _why_this(session, context),
        "prep_notes": _prep_notes(context),
    }
    card["outfit_pairing"] = pair_workout_outfit(context, card)
    card["reminders"] = build_workout_reminders(context, card)
    return card


def _why_this(session: Dict[str, Any], context: Dict[str, Any]) -> str:
    duration = session.get("duration_min") or context.get("duration") or 20
    location = context.get("location") or "home"
    weather = context.get("weather_context") or {}
    if weather.get("recommendation") == "indoor":
        return f"A short indoor session fits the weather and keeps movement realistic today."
    if context.get("time_of_day") == "morning":
        return f"{duration} minutes gives you movement without stealing the morning."
    return f"A focused {duration}-minute {location} session that fits your day."


def _prep_notes(context: Dict[str, Any]) -> List[str]:
    notes = ["Keep water nearby", "Use training shoes if the floor is hard"]
    weather = context.get("weather_context") or {}
    if weather.get("recommendation") == "indoor":
        notes.append("Avoid outdoor cardio today")
    if weather.get("condition") in {"humid", "hot"}:
        notes.append("Wear breathable fabric")
    return notes[:4]
