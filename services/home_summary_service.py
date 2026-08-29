from __future__ import annotations

from typing import Any, Dict
from services.workout_card_service import get_today_workout_card


def generate_home_summary(user_id: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    context = context or {}
    workout_card = get_today_workout_card(user_id, context)
    return {
        "user_id": user_id,
        "move_card": workout_card,
        "cards": {
            "move": workout_card,
        },
    }
