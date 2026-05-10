import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class WorkoutApiTests(unittest.TestCase):
    def test_recommendation_pipeline_returns_contextual_cards(self):
        from brain.engines.fitness.fitness_engine import fitness_engine
        from brain.engines.fitness.workout_ranker import workout_ranker
        from services.workout_card_service import build_workout_card
        from services.workout_context_service import build_workout_context

        context = build_workout_context(
            "user_1",
            {
                "goal": "fat_loss",
                "duration": 20,
                "location": "home",
                "equipment": "none",
                "weather": {"condition": "humid", "humidity": 82},
            },
        )
        raw = fitness_engine.filter_sessions(context)
        ranked = workout_ranker.rank(raw or fitness_engine.relaxed_fallback(context), context)
        card = build_workout_card(ranked[0], context)

        self.assertEqual(card["type"], "workout_card")
        self.assertIn("outfit_pairing", card)
        self.assertIn("reminders", card)
        self.assertIn("why_this", card)

    def test_today_workout_shape(self):
        from brain.engines.fitness.fitness_engine import fitness_engine
        from brain.engines.fitness.workout_ranker import workout_ranker
        from services.workout_card_service import build_workout_card
        from services.workout_context_service import build_workout_context

        context = build_workout_context(
            "user_1",
            {"goal": "general_fitness", "duration": 12, "location": "home"},
        )
        ranked = workout_ranker.rank(
            fitness_engine.filter_sessions(context)
            or fitness_engine.relaxed_fallback(context),
            context,
            limit=1,
        )
        first = build_workout_card(ranked[0], context)
        result = {
            "type": "fitness_today",
            "today_workout": first,
            "outfit_pairing": first["outfit_pairing"],
            "reminders": first["reminders"],
        }

        self.assertEqual(result["type"], "fitness_today")
        self.assertIn("today_workout", result)
        self.assertIn("outfit_pairing", result)
        self.assertIn("reminders", result)

    def test_ranker_demotes_recently_skipped_workouts(self):
        from brain.engines.fitness.workout_ranker import WorkoutRanker

        ranker = WorkoutRanker()
        context = {
            "goal": "general_fitness",
            "duration": 10,
            "location": "home",
            "equipment": "none",
            "recent_skipped_workout_ids": ["skip_me"],
        }
        sessions = [
            {
                "key": "skip_me",
                "goal_tags": ["general_fitness"],
                "duration_min": 10,
                "location": ["home"],
                "equipment": ["none"],
            },
            {
                "key": "keep_me",
                "goal_tags": ["general_fitness"],
                "duration_min": 10,
                "location": ["home"],
                "equipment": ["none"],
            },
        ]

        self.assertEqual(ranker.rank(sessions, context, limit=1)[0]["key"], "keep_me")


if __name__ == "__main__":
    unittest.main()
