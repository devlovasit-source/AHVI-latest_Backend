from __future__ import annotations

import re
from typing import Any, Dict, List


GENERAL_STYLE_ADVICE = "general_style_advice"
BODY_TYPE_ADVICE = "body_type_advice"
SKIN_TONE_ADVICE = "skin_tone_advice"
WARDROBE_STYLE = "wardrobe_style"
SHOPPING_ASSIST = "shopping_assist"


def _norm(text: Any) -> str:
    q = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", q).strip()


def _has_any(q: str, phrases: tuple[str, ...]) -> bool:
    return any(p in q for p in phrases)


def classify_stylist_intent(text: Any) -> str:
    q = _norm(text)
    if not q:
        return ""

    wardrobe_markers = (
        "use my wardrobe",
        "from my wardrobe",
        "with my wardrobe",
        "build a look from my wardrobe",
        "build an outfit from my wardrobe",
        "style from my wardrobe",
        "use my clothes",
        "use my items",
        "my blue shirt",
        "my shirt",
        "my pants",
        "my dress",
        "my blazer",
        "my jacket",
        "my shoes",
        "my saree",
        "my kurta",
    )
    if _has_any(q, wardrobe_markers):
        return WARDROBE_STYLE

    shopping_markers = (
        "what should i buy",
        "what to buy",
        "recommend shoes",
        "recommend footwear",
        "recommend missing",
        "complete this look",
        "shopping suggestions",
        "show shopping",
        "buy for this outfit",
        "similar looks",
    )
    if _has_any(q, shopping_markers):
        return SHOPPING_ASSIST

    skin_markers = (
        "skin tone",
        "warm skin",
        "cool skin",
        "neutral skin",
        "colors suit",
        "colours suit",
        "what colors",
        "what colours",
        "avoid wearing",
    )
    if _has_any(q, skin_markers):
        return SKIN_TONE_ADVICE

    body_markers = (
        "body type",
        "body shape",
        "pear body",
        "apple body",
        "rectangle body",
        "hourglass",
        "athletic body",
        "broad shoulders",
        "narrow shoulders",
        "wide hips",
        "what suits my body",
    )
    if _has_any(q, body_markers):
        return BODY_TYPE_ADVICE

    style_advice_markers = (
        "what should i wear",
        "what do i wear",
        "what to wear",
        "wear to",
        "wear for",
        "dress for",
        "outfit for",
        "style advice",
        "fashion advice",
    )
    occasion_markers = (
        "coffee date",
        "date",
        "funeral",
        "christian funeral",
        "work",
        "office",
        "wedding",
        "dinner",
        "party",
        "interview",
        "travel",
        "casual weekend",
    )
    if _has_any(q, style_advice_markers) and _has_any(q, occasion_markers):
        return GENERAL_STYLE_ADVICE

    return ""


def _occasion_from_text(text: Any) -> str:
    q = _norm(text)
    if "christian funeral" in q:
        return "Christian funeral"
    if "funeral" in q:
        return "funeral"
    if "coffee date" in q:
        return "coffee date"
    if "date" in q:
        return "date"
    if "wedding" in q:
        return "wedding"
    if "interview" in q:
        return "interview"
    if "work" in q or "office" in q:
        return "office"
    if "dinner" in q:
        return "dinner"
    if "party" in q:
        return "party"
    if "travel" in q:
        return "travel"
    return "occasion"


def _advice_blocks(kind: str, text: Any) -> Dict[str, List[str] | str]:
    occasion = _occasion_from_text(text)
    q = _norm(text)

    if kind == SKIN_TONE_ADVICE:
        tone = "warm" if "warm" in q else "cool" if "cool" in q else "neutral"
        return {
            "title": "Color Guidance",
            "intro": f"For {tone} skin, the strongest colors usually echo your undertone instead of fighting it.",
            "recommended": [
                "Warm: ivory, camel, olive, rust, tomato red, warm navy",
                "Cool: optic white, charcoal, cobalt, emerald, berry, icy pastels",
                "Neutral: stone, navy, soft black, teal, dusty rose, balanced earth tones",
            ],
            "outfit": [
                "Keep one color close to your face that flatters your undertone.",
                "Use neutrals as anchors, then add one controlled accent.",
            ],
            "avoid": [
                "Colors that make your skin look grey, tired, or overly yellow",
                "Too many competing brights near the face",
            ],
        }

    if kind == BODY_TYPE_ADVICE:
        body = "pear" if "pear" in q else "apple" if "apple" in q else "rectangle" if "rectangle" in q else "athletic" if "athletic" in q else "your body shape"
        return {
            "title": "Body Type Advice",
            "intro": f"For {body}, the goal is balance, clean proportion, and an intentional focal point.",
            "recommended": [
                "Use structured shoulders or open necklines to frame the upper body.",
                "Choose rises, hems, and layers that create a clear waist or vertical line.",
                "Repeat one color vertically when you want a longer, cleaner silhouette.",
            ],
            "outfit": [
                "A fitted or lightly structured top",
                "A bottom that skims rather than clings",
                "One polished layer or accessory to finish the line",
            ],
            "avoid": [
                "Cuts that stop at the widest point without structure",
                "Bulky layers with no waist or vertical break",
            ],
        }

    if kind == SHOPPING_ASSIST:
        return {
            "title": "Shopping Assist",
            "intro": "I can help complete the look by identifying the missing piece instead of forcing a wardrobe board.",
            "recommended": [
                "Start with the piece you already like, then buy only the gap.",
                "Prioritize shoes, outer layer, or one accessory before adding more clothes.",
                "Choose versatile neutrals unless the outfit needs a deliberate statement.",
            ],
            "outfit": [
                "Similar looks",
                "Missing-piece recommendations",
                "Shopping suggestions by budget",
            ],
            "avoid": [
                "Buying a full outfit when one grounding piece would solve it",
                "Trend pieces that do not match your real lifestyle",
            ],
        }

    if "funeral" in occasion.lower():
        return {
            "title": "Stylist Advice",
            "intro": f"{occasion.title()}s generally call for respectful, understated attire.",
            "recommended": ["Black", "Charcoal", "Navy", "Deep grey"],
            "outfit": [
                "Plain shirt, blouse, kurta, or modest dress",
                "Tailored trousers or a simple skirt",
                "Closed shoes with minimal detailing",
            ],
            "avoid": [
                "Bright colors",
                "Loud prints",
                "Excessive accessories",
                "Very casual footwear",
            ],
        }

    if "coffee date" in occasion.lower() or occasion == "date":
        return {
            "title": "Stylist Advice",
            "intro": "A coffee date works best when the look feels easy, considered, and not overly formal.",
            "recommended": ["Soft neutrals", "Denim blue", "Olive", "Cream", "Warm brown"],
            "outfit": [
                "Clean shirt, knit, or relaxed blouse",
                "Straight denim, chinos, or tailored casual trousers",
                "Loafers, minimal sneakers, or neat flats",
            ],
            "avoid": [
                "Overly corporate styling",
                "Very loud prints unless that is your signature",
                "Anything uncomfortable to sit and walk in",
            ],
        }

    return {
        "title": "Stylist Advice",
        "intro": f"For {occasion}, start with the dress code, setting, and how visible you want the outfit to feel.",
        "recommended": ["Clean neutrals", "One accent color", "Polished textures", "Occasion-appropriate shoes"],
        "outfit": [
            "A strong base piece",
            "A balanced second piece",
            "Shoes that match the setting",
            "One finishing detail",
        ],
        "avoid": [
            "Pieces that fight the occasion",
            "Too many statement items at once",
            "Shoes that are uncomfortable or too casual for the setting",
        ],
    }


def build_stylist_advice_response(
    *,
    query: str,
    intent: str,
    module_context: str = "",
) -> Dict[str, Any]:
    blocks = _advice_blocks(intent, query)
    title = str(blocks["title"])
    intro = str(blocks["intro"])
    recommended = [str(x) for x in blocks["recommended"]]  # type: ignore[index]
    outfit = [str(x) for x in blocks["outfit"]]  # type: ignore[index]
    avoid = [str(x) for x in blocks["avoid"]]  # type: ignore[index]

    message = "\n\n".join(
        [
            title,
            intro,
            "Recommended:\n" + "\n".join(f"- {x}" for x in recommended),
            "Outfit direction:\n" + "\n".join(f"- {x}" for x in outfit),
            "Avoid:\n" + "\n".join(f"- {x}" for x in avoid),
        ]
    )

    chips = [
        {"label": "Generate Moodboard", "value": f"Generate moodboard for: {query}"},
        {"label": "Use My Wardrobe", "value": f"Use my wardrobe for: {query}"},
        {"label": "Show Shopping Ideas", "value": f"Show shopping ideas for: {query}"},
    ]

    return {
        "success": True,
        "ok": True,
        "type": "stylist_advice",
        "intent": intent,
        "message": {"role": "assistant", "content": message},
        "message_text": message,
        "response": message,
        "text": message,
        "cards": [
            {
                "type": "moodboard",
                "title": title,
                "subtitle": intro,
                "palette": recommended[:6],
                "items": outfit,
                "avoid": avoid,
            }
        ],
        "style_boards": [],
        "chips": chips,
        "board_ids": "",
        "data": {
            "intent": intent,
            "stylist_mode": True,
            "moodboard": {
                "title": title,
                "palette": recommended[:6],
                "items": outfit,
                "avoid": avoid,
            },
            "cta_actions": chips,
        },
        "meta": {
            "mode": "stylist_knowledge",
            "intent": intent,
            "module_context": module_context or "chat",
            "wardrobe_lookup": False,
        },
        "audio_job_id": "offline",
    }
