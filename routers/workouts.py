from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain.engines.fitness.fitness_engine import fitness_engine
from brain.engines.fitness.workout_ranker import workout_ranker
from middleware.auth_middleware import get_current_user
from services.workout_card_service import build_workout_card
from services.workout_context_service import build_workout_context

router = APIRouter(prefix="/workouts", tags=["workouts"])

_WORKOUT_HISTORY: Dict[str, List[Dict[str, Any]]] = {}


class WorkoutRecommendRequest(BaseModel):
    user_id: str | None = None
    goal: str | None = "general_fitness"
    gender: str | None = None
    duration: int | None = Field(default=20, ge=5, le=90)
    duration_minutes: int | None = Field(default=None, ge=5, le=90)
    location: str | None = "home"
    equipment: str | None = "none"
    constraint: str | None = None
    weather: Dict[str, Any] | str | None = None
    calendar: Dict[str, Any] | None = None
    profile: Dict[str, Any] | None = None
    recent_skipped_workout_ids: List[str] | None = None


class WorkoutFeedbackRequest(BaseModel):
    user_id: str | None = None
    workout_id: str = Field(..., min_length=1, max_length=128)
    completed: bool | None = None
    skipped: bool | None = None
    difficulty_feedback: str | None = None
    reason: str | None = None


def _user_id(user: Any, fallback: str | None = None) -> str:
    uid = str(
        (user or {}).get("user_id")
        or (user or {}).get("$id")
        or fallback
        or ""
    ).strip()
    if not uid:
        raise HTTPException(status_code=401, detail="Missing authenticated user")
    return uid


def _history_for(user_id: str) -> List[Dict[str, Any]]:
    return _WORKOUT_HISTORY.setdefault(user_id, [])


def _recommendations(context: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    raw = fitness_engine.filter_sessions(context)
    if not raw:
        raw = fitness_engine.relaxed_fallback(context, limit=max(limit, 3))
    ranked = workout_ranker.rank(raw, context, limit=limit)
    return [build_workout_card(session, context) for session in ranked]


@router.post("/recommend")
def recommend_workout(
    req: WorkoutRecommendRequest,
    user=Depends(get_current_user),
):
    user_id = _user_id(user, req.user_id)
    skipped = [
        item["workout_id"]
        for item in _history_for(user_id)[-10:]
        if item.get("skipped")
    ]
    payload = req.model_dump(exclude_none=True)
    payload["recent_skipped_workout_ids"] = list(
        dict.fromkeys((payload.get("recent_skipped_workout_ids") or []) + skipped)
    )
    context = build_workout_context(user_id, payload)
    cards = _recommendations(context)
    return {
        "type": "fitness_recommendation",
        "recommendations": cards,
        "meta": {
            "mode": "prep",
            "context": context,
            "count": len(cards),
        },
    }


@router.get("/today")
def today_workout(user=Depends(get_current_user)):
    user_id = _user_id(user)
    context = build_workout_context(
        user_id,
        {
            "goal": "general_fitness",
            "duration": 12,
            "location": "home",
            "equipment": "none",
        },
    )
    cards = _recommendations(context, limit=1)
    first = cards[0] if cards else None
    return {
        "type": "fitness_today",
        "today_workout": first,
        "outfit_pairing": (first or {}).get("outfit_pairing") or {},
        "reminders": (first or {}).get("reminders") or [],
        "meta": {"mode": "prep", "context": context},
    }


@router.post("/complete")
def complete_workout(req: WorkoutFeedbackRequest, user=Depends(get_current_user)):
    user_id = _user_id(user, req.user_id)
    entry = {
        "user_id": user_id,
        "workout_id": req.workout_id,
        "completed": True if req.completed is None else bool(req.completed),
        "skipped": False,
        "feedback": req.difficulty_feedback,
        "difficulty": req.difficulty_feedback,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _history_for(user_id).append(entry)
    return {"success": True, "history": entry}


@router.post("/skip")
def skip_workout(req: WorkoutFeedbackRequest, user=Depends(get_current_user)):
    user_id = _user_id(user, req.user_id)
    entry = {
        "user_id": user_id,
        "workout_id": req.workout_id,
        "completed": False,
        "skipped": True,
        "feedback": req.reason,
        "difficulty": req.difficulty_feedback,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _history_for(user_id).append(entry)
    return {"success": True, "history": entry}


@router.get("/program/{key}")
def workout_program(key: str, user=Depends(get_current_user)):
    _user_id(user)
    program = fitness_engine.get_weekly_program(key)
    if not program:
        raise HTTPException(status_code=404, detail="Workout program not found")
    return {"type": "fitness_program", "program": program}
