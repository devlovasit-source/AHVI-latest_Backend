import base64
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError
from services.r2_storage import R2Storage, R2StorageError
from services.image_normalizer import normalize_style_board_image_bytes


# =========================
# HELPERS
# =========================
def clean_occasion(raw: str) -> str:
    v = (raw or "").strip().lower()
    mapping = {
        "party looks": "Party",
        "party": "Party",
        "office fit": "Office",
        "office": "Office",
        "vacation": "Vacation",
        "occasion": "Occasion",
    }
    return mapping.get(v, (raw or "Occasion").strip().title())


def decode_image_base64(value: str) -> tuple[bytes, str]:
    text = (value or "").strip()
    if not text:
        return b"", "png"

    extension = "png"

    # detect data URI
    if text.startswith("data:image/"):
        match = re.match(r"^data:image/([a-zA-Z0-9]+);base64,", text)
        if match:
            extension = match.group(1).lower()
        text = text.split(",", 1)[1] if "," in text else text

    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")

    if not data:
        raise HTTPException(status_code=400, detail="image_base64 is empty")

    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image_base64 too large (max 12MB)")

    return data, extension


# =========================
# READ APIs
# =========================
def list_saved_boards(
    *, user_id: str, occasion: Optional[str] = None, limit: int = 100
):
    proxy = AppwriteProxy()
    return proxy.list_documents(
        "saved_boards",
        user_id=user_id,
        occasion=clean_occasion(occasion) if occasion else None,
        limit=limit,
    )


def list_life_boards(*, user_id: str, limit: int = 100):
    return AppwriteProxy().list_documents("life_boards", user_id=user_id, limit=limit)


# =========================
# SAVE STYLE BOARD
# =========================
def save_board(
    *,
    user_id: str,
    occasion: str,
    image_url: str = "",
    image_base64: str = "",
    board_ids: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
):
    proxy = AppwriteProxy()
    payload = payload or {}

    final_image_url = (image_url or "").strip()

    # Upload if base64 present.
    # Long-term AHVI rule: saved board covers are normalized into a
    # consistent portrait preview before R2 upload so Planner -> Boards
    # thumbnails do not appear cropped/uneven across devices.
    if str(image_base64 or "").strip():
        image_bytes, _extension = decode_image_base64(image_base64)

        try:
            normalized_bytes = normalize_style_board_image_bytes(
                image_bytes,
                size=(1080, 1350),
                output_format="JPEG",
                quality=92,
            )
            storage = R2Storage()
            uploaded = storage.upload_style_board_image(
                user_id=user_id,
                image_bytes=normalized_bytes,
                extension="jpg",
            )
            final_image_url = uploaded.get("image_url", final_image_url)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Image upload failed: {exc}")

    # =========================
    # ITEM IDS EXTRACTION
    # =========================
    item_ids: list[str] = []

    if board_ids:
        item_ids = [x.strip() for x in board_ids.split(",") if x.strip()]
    else:
        raw_item_ids = payload.get("itemIds") or payload.get("boardIds") or []
        if isinstance(raw_item_ids, list):
            item_ids = [str(x).strip() for x in raw_item_ids if str(x).strip()]

    # =========================
    # ðŸ”¥ ELITE STRUCTURED BOARD
    # =========================
    # Demo-safe schema: Appwrite saved_boards currently supports only:
    # userId, imageUrl, itemIds, occasion.
    # Do not send aesthetic/vibe/colorStory/layout/items/styleScore/createdAt/updatedAt
    # unless those attributes are added to the Appwrite collection.
    doc = {
        "userId": user_id,
        "occasion": clean_occasion(occasion),
        "imageUrl": final_image_url,
        "itemIds": item_ids,
    }
    return proxy.create_document("saved_boards", doc)


# =========================
# SAVE LIFE BOARD
# =========================
def save_life_board(
    *,
    user_id: str,
    title: str,
    board_type: str,
    description: str,
    payload: Dict[str, Any],
):
    now_iso = datetime.now(timezone.utc).isoformat()

    doc = {
        "userId": user_id,
        "title": (title or "").strip() or "Life Board",
        "boardType": (board_type or "").strip() or "daily_wear",
        "description": (description or "").strip(),
        "payload": payload or {},
        "createdAt": now_iso,
        "updatedAt": now_iso,
    }

    return AppwriteProxy().create_document("life_boards", doc)


# =========================
# DELETE
# =========================
def delete_saved_board(*, document_id: str, user_id: str = ""):
    """Delete a saved board after verifying the requester owns it.

    user_id is the authenticated user's id. When supplied, the document's
    `userId` field must match. Empty user_id skips the check and is only
    safe for trusted callers that have already verified ownership.
    """
    proxy = AppwriteProxy()
    if user_id:
        try:
            doc = proxy.get_document("saved_boards", document_id)
        except AppwriteProxyError:
            raise
        owner = str(doc.get("userId") or doc.get("user_id") or "").strip()
        if owner and owner != user_id:
            raise AppwriteProxyError("Forbidden")
    proxy.delete_document("saved_boards", document_id)


# =========================
# VISUAL BOARD ADAPTERS
# =========================
# Reshape existing module engine output into the AHVI "visual_board"
# response contract rendered by lib/widgets/ahvi_visual_board.dart.
# These adapters do NOT duplicate engine logic: callers pass the engine
# result in, and the adapter maps it (or falls back to a safe default
# board when no engine result is available).


def _visual_board(
    board_type: str,
    title: str,
    subtitle: str,
    sections: list,
    *,
    principles: Optional[list] = None,
    why_this_plan: str = "",
) -> Dict[str, Any]:
    return {
        "response_type": "visual_board",
        "board_type": board_type,
        "title": title,
        "subtitle": subtitle,
        "principles": principles or [],
        "sections": sections or [],
        "why_this_plan": why_this_plan or "",
    }


def _checklist_items(values: Any) -> list:
    items = []
    for value in values or []:
        label = str(value or "").strip()
        if label:
            items.append({"label": label})
    return items


# Per-variant diet board templates. Each variant tunes the title,
# principles, meal options and rationale so a "vegan weekly plan" no
# longer renders the same balanced template a generic "diet plan"
# prompt would.
_DIET_VARIANTS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "title": "Balanced Day Meal Plan",
        "subtitle": "Simple meals built around protein, vegetables, healthy fats and whole-food carbs",
        "why": (
            "This keeps the day balanced without strict dieting. Each meal includes "
            "protein, fibre and slow carbs so energy stays steady."
        ),
        "principles": [
            {"label": "Vegetables", "value": "50% of plate"},
            {"label": "Protein", "value": "Palm-size"},
            {"label": "Fat", "value": "Healthy fat"},
            {"label": "Carbs", "value": "Whole-food carbs"},
        ],
        "breakfast": [
            {"name": "Overnight oats", "pairing": "berries and nuts"},
            {"name": "Chia pudding", "pairing": "coconut and fruit"},
            {"name": "Eggs", "pairing": "roasted vegetables"},
        ],
        "lunch_protein": ["chicken", "paneer", "fish", "beans", "lentils"],
        "lunch_carbs": ["rice", "potatoes", "quinoa"],
        "lunch_veggies": ["salad", "sautéed", "roasted"],
        "turn_into": ["wrap", "bowl", "loaded salad", "burrito bowl"],
        "dinner": [
            {"name": "Chickpea curry", "pairing": "rice"},
            {"name": "Chicken skillet", "pairing": "broccoli and quinoa"},
            {"name": "Stir-fry", "pairing": "vegetables and rice"},
        ],
    },
    "vegan": {
        "title": "Vegan Day Meal Plan",
        "subtitle": "Plant-based meals built around legumes, whole grains, vegetables and healthy fats",
        "why": (
            "All-plant meals deliver protein from legumes, tofu and tempeh, paired with "
            "whole grains and vegetables so the day stays satisfying and energy-stable."
        ),
        "principles": [
            {"label": "Vegetables", "value": "Half the plate"},
            {"label": "Protein", "value": "Beans / tofu / tempeh"},
            {"label": "Fat", "value": "Nuts, seeds, olive oil"},
            {"label": "Carbs", "value": "Whole grains"},
        ],
        "breakfast": [
            {"name": "Tofu scramble", "pairing": "sautéed spinach and avocado"},
            {"name": "Overnight oats", "pairing": "soy milk, peanut butter and berries"},
            {"name": "Chia pudding", "pairing": "coconut yogurt and fruit"},
        ],
        "lunch_protein": ["chickpeas", "tofu", "tempeh", "lentils", "black beans"],
        "lunch_carbs": ["brown rice", "quinoa", "millet"],
        "lunch_veggies": ["salad", "roasted", "stir-fried"],
        "turn_into": ["buddha bowl", "wrap", "loaded salad", "grain bowl"],
        "dinner": [
            {"name": "Chickpea curry", "pairing": "brown rice"},
            {"name": "Lentil dal", "pairing": "quinoa and greens"},
            {"name": "Tofu stir-fry", "pairing": "vegetables and noodles"},
        ],
    },
    "high_protein": {
        "title": "High-Protein Meal Plan",
        "subtitle": "Protein-forward meals to support muscle, satiety and recovery",
        "why": (
            "Each meal anchors on 25–40 g of lean protein so you hit a strong daily "
            "intake without overloading on carbs or fat."
        ),
        "principles": [
            {"label": "Protein", "value": "25–40 g per meal"},
            {"label": "Veggies", "value": "Always 1–2 portions"},
            {"label": "Carbs", "value": "Around training"},
            {"label": "Fat", "value": "Modest"},
        ],
        "breakfast": [
            {"name": "Greek yogurt bowl", "pairing": "berries, oats and nuts"},
            {"name": "3-egg omelette", "pairing": "spinach and feta"},
            {"name": "Cottage cheese", "pairing": "fruit and seeds"},
        ],
        "lunch_protein": ["grilled chicken", "lean beef", "tuna", "paneer", "tofu"],
        "lunch_carbs": ["rice", "sweet potato", "quinoa"],
        "lunch_veggies": ["greens", "roasted vegetables", "salad"],
        "turn_into": ["protein bowl", "wrap", "loaded salad", "grain bowl"],
        "dinner": [
            {"name": "Grilled chicken", "pairing": "quinoa and roasted veg"},
            {"name": "Baked salmon", "pairing": "salad and sweet potato"},
            {"name": "Paneer tikka", "pairing": "greens and lentils"},
        ],
    },
    "keto": {
        "title": "Keto Day Meal Plan",
        "subtitle": "Very-low-carb meals with healthy fats and quality protein",
        "why": (
            "Carbs stay low so the body taps fat for fuel. Each meal centres on "
            "non-starchy vegetables, protein and healthy fats."
        ),
        "principles": [
            {"label": "Net carbs", "value": "Under 30 g/day"},
            {"label": "Protein", "value": "Moderate"},
            {"label": "Fat", "value": "Olive oil, butter, avocado"},
            {"label": "Veggies", "value": "Leafy / above-ground"},
        ],
        "breakfast": [
            {"name": "Avocado and eggs", "pairing": "sautéed greens"},
            {"name": "Cheese omelette", "pairing": "mushrooms and spinach"},
            {"name": "Chia pudding", "pairing": "coconut milk and berries (low)"},
        ],
        "lunch_protein": ["chicken thighs", "salmon", "paneer", "eggs", "ground beef"],
        "lunch_carbs": ["cauliflower rice", "zoodles", "leafy greens"],
        "lunch_veggies": ["salad", "roasted low-carb veg", "stir-fried"],
        "turn_into": ["lettuce wrap", "salad bowl", "cheese plate"],
        "dinner": [
            {"name": "Steak", "pairing": "buttered green beans"},
            {"name": "Grilled fish", "pairing": "creamed spinach"},
            {"name": "Chicken curry", "pairing": "cauliflower rice"},
        ],
    },
    "weight_loss": {
        "title": "Weight-Loss Meal Plan",
        "subtitle": "Lean meals built around protein, fibre and lower-calorie volume",
        "why": (
            "Built for a gentle calorie deficit: high protein keeps hunger down, lots of "
            "vegetables add volume, and carbs sit modest around your active hours."
        ),
        "principles": [
            {"label": "Protein", "value": "Every meal"},
            {"label": "Veggies", "value": "Half the plate"},
            {"label": "Carbs", "value": "Modest portions"},
            {"label": "Fat", "value": "Small, healthy"},
        ],
        "breakfast": [
            {"name": "Greek yogurt", "pairing": "fruit and chia"},
            {"name": "Veggie omelette", "pairing": "fresh salad"},
            {"name": "Oats", "pairing": "milk, fruit and nuts (small)"},
        ],
        "lunch_protein": ["chicken breast", "fish", "tofu", "egg whites", "dal"],
        "lunch_carbs": ["small rice", "millet", "roti (1–2)"],
        "lunch_veggies": ["salad (large)", "steamed", "roasted"],
        "turn_into": ["bowl", "salad plate", "wrap (whole-grain)"],
        "dinner": [
            {"name": "Grilled fish", "pairing": "salad and lemon"},
            {"name": "Chicken curry", "pairing": "vegetables (skip extra rice)"},
            {"name": "Tofu and dal", "pairing": "greens"},
        ],
    },
    "mediterranean": {
        "title": "Mediterranean Day Plan",
        "subtitle": "Olive oil, fish, whole grains, vegetables and legumes",
        "why": (
            "The Mediterranean pattern emphasises plant foods, olive oil, fish, and "
            "modest dairy — well-studied for heart and metabolic health."
        ),
        "principles": [
            {"label": "Olive oil", "value": "Primary fat"},
            {"label": "Fish", "value": "2–3 times/week"},
            {"label": "Veggies", "value": "Every meal"},
            {"label": "Whole grains", "value": "Often"},
        ],
        "breakfast": [
            {"name": "Greek yogurt", "pairing": "honey, walnuts and figs"},
            {"name": "Whole-grain toast", "pairing": "olive oil, tomato and feta"},
            {"name": "Oats", "pairing": "almonds and fruit"},
        ],
        "lunch_protein": ["chickpeas", "fish", "feta", "lentils", "chicken"],
        "lunch_carbs": ["whole-grain pasta", "farro", "couscous"],
        "lunch_veggies": ["greek salad", "roasted veg", "grilled"],
        "turn_into": ["grain bowl", "mezze plate", "salad"],
        "dinner": [
            {"name": "Baked fish", "pairing": "tomatoes, olives and capers"},
            {"name": "Chickpea stew", "pairing": "whole-grain bread"},
            {"name": "Grilled chicken", "pairing": "tabbouleh and greens"},
        ],
    },
}


def _detect_diet_variant(text: str) -> str:
    """Pick a diet variant key from the user's prompt."""
    lowered = str(text or "").lower()
    if any(k in lowered for k in ("vegan", "plant based", "plant-based")):
        return "vegan"
    if "high protein" in lowered or "high-protein" in lowered or "highprotein" in lowered:
        return "high_protein"
    if "keto" in lowered or "ketogenic" in lowered or "low carb" in lowered:
        return "keto"
    if (
        "weight loss" in lowered
        or "lose weight" in lowered
        or "fat loss" in lowered
        or "cutting" in lowered
    ):
        return "weight_loss"
    if "mediterr" in lowered:
        return "mediterranean"
    return "balanced"


def build_diet_visual_board(
    engine_result=None, user_context=None, diet_variant: str = "balanced"
) -> Dict[str, Any]:
    variant = _DIET_VARIANTS.get(diet_variant) or _DIET_VARIANTS["balanced"]

    why = ""
    if isinstance(engine_result, dict):
        why = str(
            engine_result.get("why_this_plan")
            or engine_result.get("message")
            or ""
        ).strip()
    if not why and isinstance(user_context, dict):
        why = str(user_context.get("diet_summary") or "").strip()
    if not why:
        why = variant["why"]

    sections = [
        {
            "title": "Breakfast",
            "layout": "meal_options",
            "items": list(variant["breakfast"]),
        },
        {
            "title": "Lunch",
            "layout": "batch_prep",
            "items": [
                {"category": "Protein", "options": list(variant["lunch_protein"])},
                {"category": "Carbs", "options": list(variant["lunch_carbs"])},
                {"category": "Veggies", "options": list(variant["lunch_veggies"])},
            ],
            "turn_into": list(variant["turn_into"]),
        },
        {
            "title": "Dinner",
            "layout": "simple_combinations",
            "items": list(variant["dinner"]),
        },
    ]
    return _visual_board(
        "diet_plan",
        variant["title"],
        variant["subtitle"],
        sections,
        principles=list(variant["principles"]),
        why_this_plan=why,
    )


def build_pack_visual_board(engine_result=None, user_context=None) -> Dict[str, Any]:
    sections: list = []
    if isinstance(engine_result, dict):
        cards = engine_result.get("cards")
        for card in cards if isinstance(cards, list) else []:
            if not isinstance(card, dict):
                continue
            title = str(card.get("title") or card.get("category") or "").strip()
            items = _checklist_items(card.get("items"))
            if title and items:
                sections.append(
                    {"title": title, "layout": "checklist", "items": items}
                )

    if not sections:
        sections = [
            {
                "title": "Travel Essentials",
                "layout": "checklist",
                "items": _checklist_items(
                    ["Passport", "Boarding pass", "Wallet", "Phone charger", "Water bottle"]
                ),
            },
            {
                "title": "Comfort",
                "layout": "checklist",
                "items": _checklist_items(
                    ["Scarf", "Socks", "Eye mask", "Neck pillow"]
                ),
            },
            {
                "title": "Health",
                "layout": "checklist",
                "items": _checklist_items(
                    ["Medicines", "Tissues", "Wet wipes", "Sanitizer"]
                ),
            },
            {
                "title": "Grooming",
                "layout": "checklist",
                "items": _checklist_items(
                    ["Lip balm", "Moisturizer", "Sunscreen", "Compact mirror"]
                ),
            },
        ]

    why = ""
    if isinstance(engine_result, dict):
        why = str(engine_result.get("why_this_plan") or "").strip()
    if not why:
        why = (
            "This keeps security, boarding and in-flight needs accessible without "
            "overpacking the cabin bag."
        )
    return _visual_board(
        "packing_checklist",
        "Travel Carry-On Checklist",
        "Everything you need within reach",
        sections,
        why_this_plan=why,
    )


def build_plan_visual_board(engine_result=None, user_context=None) -> Dict[str, Any]:
    sections: list = []
    if isinstance(engine_result, dict):
        raw_sections = engine_result.get("sections")
        for section in raw_sections if isinstance(raw_sections, list) else []:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "").strip()
            items = _checklist_items(
                [
                    (i.get("label") if isinstance(i, dict) else i)
                    for i in (section.get("items") or [])
                ]
            )
            if title and items:
                sections.append(
                    {
                        "title": title,
                        "layout": "timeline_checklist",
                        "items": items,
                    }
                )

    if not sections:
        sections = [
            {
                "title": "Tonight",
                "layout": "timeline_checklist",
                "items": _checklist_items(
                    [
                        "Check weather",
                        "Charge phone and power bank",
                        "Pack bag",
                        "Keep outfit ready",
                    ]
                ),
            },
            {
                "title": "Tomorrow Morning",
                "layout": "timeline_checklist",
                "items": _checklist_items(
                    [
                        "Eat a light breakfast",
                        "Carry water",
                        "Check essentials",
                        "Leave buffer time",
                    ]
                ),
            },
        ]

    why = ""
    if isinstance(engine_result, dict):
        why = str(engine_result.get("why_this_plan") or "").strip()
    if not why:
        why = (
            "The order avoids morning decision fatigue by handling packing, charging "
            "and outfit decisions the night before."
        )
    return _visual_board(
        "trip_prep",
        "Tomorrow Prep Plan",
        "A simple timeline so nothing is rushed",
        sections,
        why_this_plan=why,
    )


__all__ = [
    "AppwriteProxyError",
    "R2StorageError",
    "clean_occasion",
    "list_saved_boards",
    "save_board",
    "list_life_boards",
    "save_life_board",
    "delete_saved_board",
    "build_diet_visual_board",
    "build_pack_visual_board",
    "build_plan_visual_board",
]
