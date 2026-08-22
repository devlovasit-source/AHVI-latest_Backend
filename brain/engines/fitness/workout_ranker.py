from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).lower() for v in value]
    return [str(value).lower()]


def _duration_distance(session: Dict[str, Any], target: Any) -> int:
    try:
        wanted = int(target)
        actual = int(session.get("duration_min") or 0)
        return abs(actual - wanted)
    except Exception:
        return 999


class WorkoutRanker:
    def score(self, session: Dict[str, Any], context: Dict[str, Any]) -> float:
        score = 0.0
        goal = str(context.get("goal") or "").lower()
        location = str(context.get("location") or "").lower()
        equipment = str(context.get("equipment") or "").lower()
        constraint = str(context.get("constraint") or "").lower()
        weather_reco = str(
            (context.get("weather_context") or {}).get("recommendation") or ""
        ).lower()
        time_of_day = str(context.get("time_of_day") or "").lower()

        if goal and goal in _as_list(session.get("goal_tags")):
            score += 24
        if location and location in _as_list(session.get("location")):
            score += 20
        if equipment and equipment in _as_list(session.get("equipment")):
            score += 18
        if constraint and (
            constraint in _as_list(session.get("tags"))
            or constraint in _as_list(session.get("style"))
            or constraint in _as_list(session.get("goal_tags"))
        ):
            score += 12

        dist = _duration_distance(session, context.get("duration"))
        if dist <= 5:
            score += 16
        elif dist <= 10:
            score += 8

        style = str(session.get("style") or "").lower()
        energy = str(session.get("energy_level") or "").lower()
        if weather_reco == "indoor" and location in {"home", "hotel_room", "gym"}:
            score += 10
        if weather_reco == "shorter" and int(session.get("duration_min") or 0) <= 20:
            score += 10
        if time_of_day == "morning" and style in {"yoga", "mobility", "low_impact"}:
            score += 8
        if time_of_day == "evening" and style in {"strength", "cardio", "dance_fitness"}:
            score += 6
        if time_of_day == "late" and energy in {"low", "gentle", "easy"}:
            score += 10

        recently_skipped = set(context.get("recent_skipped_workout_ids") or [])
        if session.get("key") in recently_skipped:
            score -= 30

        return score

    def rank(
        self,
        sessions: Iterable[Dict[str, Any]],
        context: Dict[str, Any],
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        indexed = list(enumerate(sessions))
        ranked = sorted(
            indexed,
            key=lambda pair: (self.score(pair[1], context), -pair[0]),
            reverse=True,
        )
        return [session for _, session in ranked[:limit]]


workout_ranker = WorkoutRanker()
