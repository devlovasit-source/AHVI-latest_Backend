from __future__ import annotations

import logging
import hashlib
from datetime import date, datetime
from typing import Any, Dict, List
from brain.engines.meals.meal_planner_engine import meal_planner_engine

logger = logging.getLogger("ahvi.diet_service")

CATALOGUE_VERSION = "v1.0"

# Allowlisted allergy-alias mapping
ALLERGY_MAP: Dict[str, List[str]] = {
    "peanut": ["peanut", "peanuts", "groundnuts", "groundnut"],
    "dairy": ["milk", "cheese", "paneer", "yogurt", "butter", "curd"],
    "egg": ["egg", "eggs"],
    "gluten": ["oats", "wheat", "flour", "semolina"]
}

# Curated deterministic list of food items with explicit allergen_tags
CURATED_DIET_CATALOGUE: List[Dict[str, Any]] = [
    {
        "id": "rec_veg_oats",
        "title": "Vegetable Masala Oats",
        "diet_type": ["veg", "vegetarian"],
        "goal_tags": ["balanced"],
        "time_min": 15,
        "ingredients": ["oats", "onion", "tomato", "chilli", "carrots"],
        "steps": ["Boil oats", "Sauté vegetables", "Mix together"],
        "meal_type": "breakfast",
        "allergen_tags": ["gluten"]
    },
    {
        "id": "rec_veg_poha",
        "title": "Traditional Poha",
        "diet_type": ["veg", "vegetarian"],
        "goal_tags": ["balanced"],
        "time_min": 15,
        "ingredients": ["flattened rice", "turmeric", "onion", "peanuts"],
        "steps": ["Wash poha", "Sauté spices and onion with peanuts", "Mix together"],
        "meal_type": "breakfast",
        "allergen_tags": ["peanut"]
    },
    {
        "id": "rec_veg_pulao",
        "title": "Paneer Vegetable Pulao",
        "diet_type": ["veg", "vegetarian"],
        "goal_tags": ["balanced"],
        "time_min": 25,
        "ingredients": ["rice", "paneer", "peas", "spices"],
        "steps": ["Wash rice", "Sauté paneer and spices", "Cook together in pot"],
        "meal_type": "lunch",
        "allergen_tags": ["dairy", "gluten"]
    },
    {
        "id": "rec_nonveg_chicken",
        "title": "Grilled Chicken and Rice Bowl",
        "diet_type": ["non-veg", "non-vegetarian"],
        "goal_tags": ["balanced"],
        "time_min": 25,
        "ingredients": ["chicken breast", "rice", "broccoli", "olive oil"],
        "steps": ["Sauté chicken", "Steam broccoli", "Serve over boiled rice"],
        "meal_type": "lunch",
        "allergen_tags": ["gluten"]
    },
    {
        "id": "rec_veg_salad",
        "title": "Spicy Chickpea Salad",
        "diet_type": ["veg", "vegetarian"],
        "goal_tags": ["balanced"],
        "time_min": 10,
        "ingredients": ["chickpeas", "cucumber", "tomato", "onion", "lemon"],
        "steps": ["Mix ingredients in bowl", "Squeeze lemon on top"],
        "meal_type": "snack",
        "allergen_tags": []
    },
    {
        "id": "rec_veg_curry",
        "title": "Spiced Chickpea Curry",
        "diet_type": ["veg", "vegetarian"],
        "goal_tags": ["balanced"],
        "time_min": 30,
        "ingredients": ["chickpeas", "tomato puree", "spices", "onion", "garlic"],
        "steps": ["Sauté onion and garlic", "Add spices and tomato puree", "Simmer chickpeas"],
        "meal_type": "dinner",
        "allergen_tags": []
    },
    {
        "id": "rec_nonveg_fish",
        "title": "Baked Fish with Sautéed Vegetables",
        "diet_type": ["non-veg", "non-vegetarian"],
        "goal_tags": ["balanced"],
        "time_min": 20,
        "ingredients": ["fish fillet", "bell peppers", "lemon", "soy sauce"],
        "steps": ["Bake fish", "Sauté vegetables", "Serve with lemon squeeze"],
        "meal_type": "dinner",
        "allergen_tags": []
    }
]

def get_diet_recommendation(
    user_id: str,
    local_date: date,
    local_hour: int,
    meal_type: str | None = None,
    profile: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Retrieves one deterministic, non-persisted recommendation for the user.
    Uses MealPlannerEngine for processing and scoring candidates based on preferences.
    """
    profile = profile or {}

    # 1. Determine meal type based on local_hour if not specified
    if not meal_type:
        if 5 <= local_hour < 11:
            meal_type = "breakfast"
        elif 11 <= local_hour < 16:
            meal_type = "lunch"
        elif 16 <= local_hour < 19:
            meal_type = "snack"
        else:
            meal_type = "dinner"

    meal_type = str(meal_type).lower().strip()
    if meal_type not in {"breakfast", "lunch", "snack", "dinner"}:
        meal_type = "dinner"

    # 2. Identify and validate diet preferences
    user_pref = str(profile.get("diet_type") or profile.get("diet") or "").lower().strip()

    is_veg = False
    is_non_veg = False

    if any(v in user_pref for v in ("vegetarian", "veg")) and "non" not in user_pref:
        is_veg = True
    elif "non" in user_pref and "veg" in user_pref:
        is_non_veg = True

    if not is_veg and not is_non_veg:
        logger.warning("Ambiguous or missing diet preference - returning unavailable")
        return {
            "status": "unavailable",
            "reason": "diet_constraints_unresolved",
            "recommendation": None
        }

    # 3. Normalize allergies / restrictions whether stored as a string or list
    raw_allergies = profile.get("allergies") or profile.get("restrictions") or []
    if isinstance(raw_allergies, str):
        raw_allergies = [raw_allergies]

    allergies = []
    if isinstance(raw_allergies, list):
        for item in raw_allergies:
            cleaned = str(item or "").lower().strip()
            if cleaned:
                allergies.append(cleaned)

    # 4. Map user allergies to normalised allergen tags in our allowlist.
    # Return diet_constraints_unresolved for unsupported or ambiguous restrictions.
    normalized_user_allergens = set()
    for allergy in allergies:
        # Find if this allergy is in our mapped allowlist
        matched_tag = None
        for tag, aliases in ALLERGY_MAP.items():
            if allergy == tag or allergy in aliases:
                matched_tag = tag
                break

        if matched_tag:
            normalized_user_allergens.add(matched_tag)
        else:
            # Unsupported or ambiguous allergy constraint - reject safely!
            logger.warning("Unsupported allergy constraint - returning unavailable")
            return {
                "status": "unavailable",
                "reason": "diet_constraints_unresolved",
                "recommendation": None
            }

    # 5. Filter the curated catalogue based on preferences and allergens
    candidates = []
    for recipe in CURATED_DIET_CATALOGUE:
        if recipe["meal_type"] != meal_type:
            continue

        # Respect vegetarian preference
        recipe_diets = [d.lower() for d in recipe["diet_type"]]
        is_recipe_veg = any("non" not in d and "veg" in d for d in recipe_diets)

        if is_veg and not is_recipe_veg:
            continue
        if is_non_veg and is_recipe_veg:
            if is_recipe_veg:
                continue

        # Respect explicit allergen_tags
        recipe_allergens = recipe.get("allergen_tags") or []
        if normalized_user_allergens.intersection(set(recipe_allergens)):
            continue

        candidates.append(recipe)

    if not candidates:
        logger.warning("No safe diet candidates found after constraints resolution")
        return {
            "status": "unavailable",
            "reason": "diet_constraints_unresolved",
            "recommendation": None
        }

    # 6. Use MealPlannerEngine to rank/pick best candidates
    try:
        input_data = {
            "recipes": candidates,
            "goals": {"focus": "balanced"},
            "user": {
                "diet_type": "veg" if is_veg else "non-veg",
                "allergies": list(normalized_user_allergens)
            },
            "constraints": {"cooking_time_cap_min": 45}
        }

        top_recipes = meal_planner_engine.pick_top(candidates, input_data, n=len(candidates))
        if not top_recipes:
            top_recipes = candidates
    except Exception as exc:
        logger.exception("MealPlannerEngine failed scoring: %s", exc)
        top_recipes = candidates

    # 7. Deterministic selection from ranked candidates using SHA-256
    date_str = local_date.isoformat()
    seed_str = f"{user_id}:{date_str}:{meal_type}:{CATALOGUE_VERSION}"
    hasher = hashlib.sha256(seed_str.encode("utf-8"))
    seed_val = int(hasher.hexdigest(), 16)

    selected_recipe = top_recipes[seed_val % len(top_recipes)]
    deterministic_key = f"rec_{hasher.hexdigest()[:16]}"

    recommendation = {
        "id": deterministic_key,
        "meal_type": meal_type,
        "title": selected_recipe["title"],
        "context": f"A practical {meal_type} recommendation for today",
        "source": "daily_recommendation",
        "persisted": False
    }

    return {
        "status": "ready",
        "recommendation": recommendation
    }
