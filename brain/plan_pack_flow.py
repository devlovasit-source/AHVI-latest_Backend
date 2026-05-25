import math
import re
from typing import Any, Dict, List, Optional

from brain.engines.packing.packing_engine import packing_engine


_ASSET_ICONS = {
    "sunscreen": "assets/icons/sunscreen.svg",
    "sunglasses": "assets/icons/sunglasses.svg",
    "charger": "assets/icons/charger.svg",
    "phone": "assets/icons/charger.svg",
    "power bank": "assets/icons/power_bank.svg",
    "moisturizer": "assets/icons/moisturizer.svg",
    "toiletries": "assets/icons/toiletries.svg",
    "water bottle": "assets/icons/water_bottle.svg",
    "hydration bottle": "assets/icons/water_bottle.svg",
    "shoes": "assets/icons/shoes.svg",
    "footwear": "assets/icons/shoes.svg",
    "jacket": "assets/icons/jacket.svg",
    "outer layer": "assets/icons/jacket.svg",
    "towel": "assets/icons/towel.svg",
    "medicine": "assets/icons/medicine_kit.svg",
    "first-aid": "assets/icons/first_aid_kit.svg",
    "first aid": "assets/icons/first_aid_kit.svg",
}


def _asset_icon_for(label: str) -> Optional[str]:
    lowered = str(label or "").lower()
    for key, icon in _ASSET_ICONS.items():
        if key in lowered:
            return icon
    return None


def _image_url_from_item(item: Dict[str, Any]) -> str:
    for key in ("imageUrl", "image_url", "image", "url", "thumbnail", "photoUrl"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _matches_wardrobe(label: str, item: Dict[str, Any]) -> bool:
    needle = str(label or "").lower()
    if not needle:
        return False
    searchable = " ".join(
        str(item.get(k) or "")
        for k in (
            "name",
            "category",
            "sub_category",
            "subcategory",
            "type",
            "tags",
            "color",
            "occasion",
        )
    ).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", needle) if len(t) >= 4]
    return any(token in searchable for token in tokens)


def _visual_item(
    label: str,
    *,
    category: str,
    wardrobe: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    clean = str(label or "").strip()
    for item in wardrobe or []:
        if not isinstance(item, dict):
            continue
        if _matches_wardrobe(clean, item):
            image_url = _image_url_from_item(item)
            if image_url:
                return {
                    "label": clean,
                    "category": category,
                    "checked": False,
                    "imageUrl": image_url,
                    "assetIcon": None,
                    "source": "wardrobe",
                    "wardrobeItemId": item.get("$id") or item.get("id"),
                }
    asset = _asset_icon_for(clean)
    return {
        "label": clean,
        "category": category,
        "checked": False,
        "imageUrl": None,
        "assetIcon": asset,
        "source": "asset" if asset else "text",
    }


def _parse_days(text: str) -> int:
    lowered = (text or "").lower()
    patterns = [
        r"(\d+)\s*[- ]?\s*day",
        r"(\d+)\s*days",
        r"for\s+(\d+)\s*days",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return max(1, min(21, int(match.group(1))))
            except Exception:
                pass

    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "week": 7,
    }
    for word, value in words.items():
        if f"{word} day" in lowered or f"{word}-day" in lowered:
            return value
    if "carry-on" in lowered or "carry on" in lowered:
        return 1
    return 3


def _has_explicit_duration(text: str) -> bool:
    lowered = (text or "").lower()
    if re.search(r"\d+\s*[- ]?\s*days?", lowered):
        return True
    return any(
        token in lowered
        for token in (
            "one day",
            "two day",
            "three day",
            "four day",
            "five day",
            "six day",
            "seven day",
            "week",
        )
    )


def _detect_scenario(text: str) -> str:
    lowered = (text or "").lower()
    if any(k in lowered for k in ["camping", "camp", "trek", "hike", "hiking"]):
        return "camping"
    if any(k in lowered for k in ["birthday party", "birthday", "party plan"]):
        return "birthday"
    if any(k in lowered for k in ["wedding", "marriage", "bride", "groom"]):
        return "wedding"
    if any(
        k in lowered for k in ["business", "work trip", "conference", "client meeting"]
    ):
        return "business"
    if any(k in lowered for k in ["goa", "beach", "vacation", "holiday", "trip"]):
        return "travel"
    if any(k in lowered for k in ["pack", "packing"]):
        return "travel"
    return "general"


def _extract_destination(text: str) -> str:
    lowered = (text or "").lower()
    if "goa" in lowered:
        return "Goa"
    if "carry-on" in lowered or "carry on" in lowered:
        return "Carry-On Trip"
    if "camp" in lowered:
        return "Camping"
    if "birthday" in lowered:
        return "Birthday Party"
    if "business" in lowered:
        return "Business Travel"
    if "wedding" in lowered:
        return "Wedding Event"
    m = re.search(
        r"(?:to|for)\s+([a-z ]{2,30})(?:trip|travel|vacation|holiday|wedding)?", lowered
    )
    if m:
        candidate = re.sub(r"\b(a|an|the|\d+|day|days)\b", " ", m.group(1))
        candidate = re.sub(r"\s+", " ", candidate).strip(" -")
        if candidate:
            return candidate.title()
    return "Your Trip"


def _packing_clothes(days: int, scenario: str) -> List[str]:
    tops = days + 1
    bottoms = max(2, math.ceil(days * 0.6))
    innerwear = days + 1
    socks = days + 1
    sleepwear = max(1, math.ceil(days / 3))
    footwear = 2
    outerwear = 1

    if scenario == "business":
        footwear = 2
        outerwear = 1
    if scenario == "wedding":
        footwear = 3
        outerwear = 1
    if scenario == "travel":
        footwear = 2
        outerwear = 1

    return [
        f"Tops x{tops}",
        f"Bottoms x{bottoms}",
        f"Innerwear x{innerwear}",
        f"Socks x{socks}",
        f"Sleepwear x{sleepwear}",
        f"Footwear x{footwear}",
        f"Outer layer x{outerwear}",
    ]


def _scenario_addons(scenario: str) -> List[str]:
    if scenario == "camping":
        return [
            "Torch/headlamp",
            "Power bank",
            "Reusable water bottle",
            "Insect repellent",
            "First-aid kit",
            "Weather-safe outer layer",
        ]
    if scenario == "birthday":
        return [
            "Guest list",
            "Cake order",
            "Decor checklist",
            "Music/playlist",
            "Food and drinks plan",
            "Gift or return favors",
        ]
    if scenario == "business":
        return [
            "Formal shirt/blouse",
            "Blazer",
            "Laptop + charger",
            "Business cards",
            "Meeting-ready shoes",
        ]
    if scenario == "wedding":
        return [
            "Main wedding outfit",
            "Backup festive outfit",
            "Jewelry/accessories",
            "Ethnic footwear",
            "Gift envelope",
        ]
    if scenario == "travel":
        return [
            "Sunscreen",
            "Sunglasses",
            "Beachwear/swimwear",
            "Toiletries kit",
            "Power bank",
        ]
    return [
        "Toiletries kit",
        "Phone + charger",
        "Personal medicine",
    ]


def _normalize_weather(context: Dict[str, Any]) -> str:
    weather = str(context.get("weather") or "").lower()
    if not weather:
        weather = str(
            (context.get("weather_data") or {}).get("condition") or ""
        ).lower()
    if any(k in weather for k in ["rain", "storm", "drizzle"]):
        return "rainy"
    if any(k in weather for k in ["cold", "chill", "winter"]):
        return "cold"
    if any(k in weather for k in ["hot", "heat", "humid", "warm", "summer"]):
        return "hot"
    return "mild"


def _time_of_day(context: Dict[str, Any]) -> str:
    value = str(context.get("time_of_day") or context.get("time") or "").lower()
    if value in ("morning", "afternoon", "evening", "night"):
        return value
    return "daytime"


def _weather_layer_items(weather: str) -> List[str]:
    if weather == "rainy":
        return ["Compact rain jacket", "Waterproof footwear", "Quick-dry bag cover"]
    if weather == "cold":
        return ["Warm jacket", "Thermal innerwear", "Socks x2 extra"]
    if weather == "hot":
        return ["Breathable cotton/linen", "Cap/hat", "Hydration bottle"]
    return ["Light layer for evenings"]


def _time_based_tasks(time_of_day: str) -> List[str]:
    if time_of_day == "morning":
        return [
            "Pack documents and chargers the night before",
            "Keep a quick breakfast/snack ready",
        ]
    if time_of_day == "evening":
        return [
            "Keep one ready-to-wear outfit on top",
            "Add travel-size freshening kit",
        ]
    if time_of_day == "night":
        return ["Keep sleepwear and essentials accessible", "Add eye mask/comfort kit"]
    return ["Keep first-day essentials in carry-on"]


def _timeline_checklist(days: int, scenario: str) -> List[str]:
    if scenario == "birthday":
        return [
            "Confirm date and time",
            "Finalize guest list",
            "Set venue or home setup",
            "Order cake",
            "Plan decor",
            "Plan food and drinks",
            "Send invitations",
            "Prepare music/playlist",
            "Arrange return gifts/favors",
            "Add to calendar",
        ]

    base = [
        "Confirm travel/event dates",
        "Book transport and stay",
        "Prepare outfits by day",
        "Pack essentials the night before",
    ]
    if scenario == "business":
        base.extend(["Prepare meeting deck", "Keep IDs and booking invoices ready"])
    if scenario == "wedding":
        base.extend(["Confirm ceremony timeline", "Coordinate with family/group"])
    if scenario == "camping":
        base.extend(["Check campsite rules", "Confirm route and emergency contacts"])
    if days >= 5:
        base.append("Add laundry plan for longer stay")
    return base


def _ui_cards(
    days: int,
    destination: str,
    scenario: str,
    weather: str,
    time_of_day: str,
    *,
    explicit_duration: bool = True,
    wardrobe: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    clothes = _packing_clothes(days=days, scenario=scenario)
    smart_packing = packing_engine.build_packing(
        {
            "days": days,
            "purpose": scenario,
            "destination": destination,
            "weather": weather,
            "gender": "universal",
        }
    )
    addons: List[str] = []
    for card in smart_packing.get("cards", []):
        title = str((card or {}).get("title") or (card or {}).get("category") or "").lower()
        if title not in {"purpose", "weather", "activity"}:
            continue
        items = card.get("items") if isinstance(card, dict) else []
        if isinstance(items, list):
            addons.extend([str(item) for item in items if str(item).strip()])
    if not addons:
        addons = _scenario_addons(scenario=scenario)
    timeline = _timeline_checklist(days=days, scenario=scenario)
    weather_items = _weather_layer_items(weather=weather)
    if scenario != "birthday":
        timeline = timeline + _time_based_tasks(time_of_day=time_of_day)

    if scenario == "birthday":
        primary_title = "Birthday Party Plan"
    elif scenario == "camping":
        primary_title = "Camping Prep Checklist"
    elif scenario == "wedding":
        primary_title = "Wedding Prep Checklist"
    elif scenario == "business":
        primary_title = "Business Travel Plan"
    elif destination == "Goa":
        primary_title = f"{days}-Day Goa Trip"
    elif destination == "Carry-On Trip":
        primary_title = "Carry-on Packing Checklist"
    else:
        primary_title = f"{days}-Day Plan"

    duration_label = f"{days} days" if explicit_duration else "Short trip"
    primary_subtitle = (
        "Birthday event"
        if scenario == "birthday"
        else f"{destination} · {duration_label}"
        if destination not in {"Your Trip", "Carry-On Trip"}
        else "Short carry-on trip"
        if destination == "Carry-On Trip"
        else duration_label
    )
    calendar_action = {
        "type": "open_module",
        "module": "calendar",
        "intent": "open_calendar",
        "route": "/organize/calendar",
        "label": "Open calendar",
    }
    checklist_action = {
        "type": "plan_pack_action",
        "module": "plan_pack",
        "intent": "open_checklist",
        "route": "plan_pack_checklist",
        "label": "Open checklist",
    }

    cards = [
        {
            "id": "trip_plan",
            "title": primary_title,
            "kind": "checklist",
            "subtitle": primary_subtitle,
            "items": [
                _visual_item(item, category="plan", wardrobe=wardrobe)
                for item in timeline
            ],
            "action": calendar_action if scenario == "birthday" else {
                "type": "plan_pack_action",
                "module": "plan_pack",
                "intent": "view_plan",
                "route": "plan_pack",
                "label": "View plan",
            },
        },
    ]
    if scenario == "birthday":
        return cards

    cards.extend([
        {
            "id": "packing_clothes",
            "title": "Packing List - Clothes",
            "kind": "checklist",
            "subtitle": duration_label,
            "items": [
                _visual_item(item, category="clothes", wardrobe=wardrobe)
                for item in clothes
            ],
            "action": checklist_action,
        },
        {
            "id": "packing_essentials",
            "title": "Packing List - Essentials",
            "kind": "checklist",
            "subtitle": destination if destination != "Your Trip" else scenario.title(),
            "items": [
                _visual_item(item, category="essentials", wardrobe=wardrobe)
                for item in addons
            ],
            "action": checklist_action,
        },
        {
            "id": "weather_time_adjustments",
            "title": "Weather & Time Adjustments",
            "kind": "checklist",
            "subtitle": f"{weather.title()} | {time_of_day.title()}",
            "items": [
                _visual_item(item, category="weather", wardrobe=wardrobe)
                for item in weather_items
            ],
            "action": {
                "type": "plan_pack_action",
                "module": "plan_pack",
                "intent": "weather_prep",
                "route": "plan_pack_weather",
                "label": "Weather prep",
            },
        },
    ])
    return cards


def build_plan_pack_response(
    text: str, context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    context = context or {}
    wardrobe = context.get("wardrobe") if isinstance(context.get("wardrobe"), list) else []
    explicit_duration = _has_explicit_duration(text)
    days = _parse_days(text)
    scenario = _detect_scenario(text)
    destination = _extract_destination(text)
    weather = _normalize_weather(context=context)
    time_of_day = _time_of_day(context=context)

    cards = _ui_cards(
        days=days,
        destination=destination,
        scenario=scenario,
        weather=weather,
        time_of_day=time_of_day,
        explicit_duration=explicit_duration,
        wardrobe=wardrobe,
    )
    if scenario == "birthday":
        message = "Built your birthday party plan."
    elif destination == "Carry-On Trip" and not explicit_duration:
        message = "Built your carry-on packing checklist for a short trip."
    else:
        message = f"Built your {scenario} plan and weather-aware packing checklist for {days} days."

    return {
        "intent": "plan_pack",
        "message": message,
        "board": "plan_pack",
        "type": "checklists",
        "chips": ["Open checklist", "Plan outfits", "Weather prep", "Save trip plan"],
        "quick_actions": [
            {"label": "Open checklist", "module": "plan_pack", "intent": "open_checklist"},
            {"label": "Plan outfits", "module": "style", "intent": "plan_outfits"},
            {"label": "Weather prep", "module": "plan_pack", "intent": "weather_prep"},
            {"label": "Save trip plan", "module": "plan_pack", "intent": "save_plan"},
        ],
        "cards": cards,
        "data": {
            "days": days,
            "duration_label": f"{days} days" if explicit_duration else "Short trip",
            "destination": destination,
            "scenario": scenario,
            "weather": weather,
            "time_of_day": time_of_day,
            "can_save_to_life_board": True,
            "source_text": text,
        },
    }
