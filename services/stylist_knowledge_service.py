from __future__ import annotations

import re
from typing import Any, Dict, List


STYLE_ADVICE = "style_advice"
WARDROBE_STYLE = "wardrobe_style"
SHOPPING_ASSIST = "shopping_assist"
STYLE_EDUCATION = "style_education"
COLOR_BODY_ADVICE = "color_body_advice"

STYLE_MODES = {
    STYLE_ADVICE,
    WARDROBE_STYLE,
    SHOPPING_ASSIST,
    STYLE_EDUCATION,
    COLOR_BODY_ADVICE,
}


def _norm(text: Any) -> str:
    q = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", q).strip()


def _has_any(q: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in q for phrase in phrases)


def _token_after(q: str, phrase: str) -> str:
    idx = q.find(phrase)
    if idx < 0:
        return ""
    return q[idx + len(phrase):].strip()


def is_wardrobe_style_request(
    text: Any,
    *,
    module_context: str = "",
    style_action: str = "",
) -> bool:
    q = _norm(text)
    module = _norm(module_context)
    action = _norm(style_action)

    if module in {"wardrobe", "closet"} and q:
        return True
    if action in {"use_my_wardrobe", "wardrobe_style", "style_saved_item"}:
        return True

    explicit = (
        "use my wardrobe",
        "use wardrobe",
        "from my wardrobe",
        "with my wardrobe",
        "with my clothes",
        "use my clothes",
        "with my closet",
        "from my closet",
        "build a look from my closet",
        "build a look from my wardrobe",
        "build an outfit from my wardrobe",
        "style from my wardrobe",
        "style my uploaded item",
        "style uploaded item",
        "style this uploaded item",
        "style this item",
        "style saved item",
        "use my items",
        "use my wardrobe for",
    )
    if _has_any(q, explicit):
        return True

    # "style my X" is wardrobe-specific only when X is a concrete item,
    # not when the user says "style my day" or asks for abstract advice.
    item_markers = (
        "shirt",
        "t shirt",
        "tee",
        "pants",
        "trousers",
        "jeans",
        "dress",
        "skirt",
        "blazer",
        "jacket",
        "coat",
        "shoes",
        "sneakers",
        "loafers",
        "saree",
        "kurta",
        "top",
        "hoodie",
    )
    return "style my" in q and _has_any(q, item_markers)


def is_shopping_assist_request(text: Any) -> bool:
    q = _norm(text)
    return _has_any(
        q,
        (
            "what should i buy",
            "what to buy",
            "find this",
            "shop similar",
            "shop this",
            "show similar",
            "similar pieces",
            "recommend shoes",
            "recommend footwear",
            "complete this look",
            "missing piece",
            "missing pieces",
            "shopping suggestions",
            "show shopping",
            "buy for this outfit",
        ),
    )


def is_color_body_advice_request(text: Any) -> bool:
    q = _norm(text)
    return _has_any(
        q,
        (
            "what colors suit",
            "what colours suit",
            "colors suit me",
            "colours suit me",
            "skin tone",
            "warm skin",
            "cool skin",
            "neutral skin",
            "body type",
            "body shape",
            "what works for my body",
            "what suits my body",
            "pear body",
            "apple body",
            "rectangle body",
            "hourglass",
            "broad shoulders",
            "wide hips",
            "avoid wearing",
        ),
    )


def is_style_education_request(text: Any) -> bool:
    q = _norm(text)
    return _has_any(
        q,
        (
            "what is",
            "explain",
            "teach me",
            "how does",
            "what does",
            "difference between",
            "why does",
            "style rule",
            "dress code",
        ),
    ) and _has_any(
        q,
        (
            "style",
            "fashion",
            "dress code",
            "silhouette",
            "color harmony",
            "colour harmony",
            "quiet luxury",
            "old money",
            "smart casual",
            "business casual",
            "modesty",
            "proportion",
        ),
    )


def is_style_advice_request(text: Any) -> bool:
    q = _norm(text)
    if not q:
        return False

    advice_markers = (
        "what should i wear",
        "what do i wear",
        "what to wear",
        "what is appropriate",
        "is appropriate",
        "how should i dress",
        "how to dress",
        "what works for",
        "wear to",
        "wear for",
        "dress for",
        "dress to",
        "outfit for",
        "outfit to",
        "look for",
        "style me for",
        "style me",
        "style advice",
        "fashion advice",
        "suggest an outfit",
        "suggest outfit",
        "recommend an outfit",
    )
    if _has_any(q, advice_markers):
        return True

    # Compact style requests should still get advice first unless wardrobe
    # usage is explicit. This covers "office outfit", "party look", etc.
    compact_markers = (
        "outfit",
        "look",
        "wear",
        "dress",
        "style",
    )
    occasionish = (
        "office",
        "client",
        "date",
        "party",
        "wedding",
        "funeral",
        "temple",
        "lunch",
        "pitch",
        "meeting",
        "travel",
        "interview",
        "dinner",
        "brunch",
        "work",
        "college",
        "school",
        "beach",
        "gym",
        "workout",
        "today",
        "tomorrow",
    )
    if _has_any(q, compact_markers) and _has_any(q, occasionish):
        return True

    short_occasionish = tuple(x for x in occasionish if x not in {"today", "tomorrow"})
    if len(q.split()) <= 4 and _has_any(q, short_occasionish):
        return True

    return False


def classify_style_mode(
    text: Any,
    *,
    module_context: str = "",
    style_action: str = "",
) -> str:
    if is_wardrobe_style_request(
        text,
        module_context=module_context,
        style_action=style_action,
    ):
        return WARDROBE_STYLE
    if is_shopping_assist_request(text):
        return SHOPPING_ASSIST
    if is_color_body_advice_request(text):
        return COLOR_BODY_ADVICE
    if is_style_education_request(text):
        return STYLE_EDUCATION
    if is_style_advice_request(text):
        return STYLE_ADVICE
    return ""


def _extract_context_phrase(query: str) -> str:
    q = str(query or "").strip()
    normalized = _norm(q)
    for phrase in (
        "what should i wear to",
        "what should i wear for",
        "what do i wear to",
        "what do i wear for",
        "what to wear to",
        "what to wear for",
        "how should i dress for",
        "how should i dress to",
        "how to dress for",
        "dress for",
        "outfit for",
        "outfit to",
        "look for",
    ):
        rest = _token_after(normalized, phrase)
        if rest:
            return rest
    return q or "this situation"


def _principle_cards(mode: str, query: str) -> Dict[str, List[str] | str]:
    context_phrase = _extract_context_phrase(query)

    if mode == COLOR_BODY_ADVICE:
        return {
            "title": "Color and Body Guidance",
            "intro": "Start with undertone, proportion, and where you want the eye to land.",
            "recommended": [
                "Choose one color near the face that makes your skin look clear and awake.",
                "Use vertical color continuity when you want a longer, cleaner line.",
                "Balance proportions with structure: shoulder, waist, hem, and footwear all matter.",
                "Repeat one material or color so the outfit feels intentional.",
            ],
            "outfit": [
                "Face-framing color or neckline",
                "A base silhouette that skims rather than fights the body",
                "A grounding shoe that matches the outfit weight",
                "One controlled accessory, not a cluster",
            ],
            "avoid": [
                "Colors that make the skin look grey, dull, or overly yellow",
                "Cuts that stop at the widest point without structure",
                "Too many focal points competing at once",
            ],
        }

    if mode == STYLE_EDUCATION:
        return {
            "title": "Style Education",
            "intro": "The useful style rule is not a fixed formula; it is matching visual signals to the setting.",
            "recommended": [
                "Formality: match the room before adding personality.",
                "Silhouette: decide whether the outfit should read sharp, soft, relaxed, or elongated.",
                "Color harmony: use neutrals as anchors and one deliberate accent.",
                "Restraint: remove one detail if the outfit feels busy.",
            ],
            "outfit": [
                "A clear base",
                "One proportion choice",
                "One texture or color decision",
                "Footwear that confirms the dress code",
            ],
            "avoid": [
                "Following an aesthetic without considering venue or comfort",
                "Mixing multiple loud signals when the setting needs clarity",
            ],
        }

    if mode == SHOPPING_ASSIST:
        return {
            "title": "Shopping Assist",
            "intro": "Treat shopping as filling a specific styling gap, not buying a whole new identity.",
            "recommended": [
                "Identify the missing role: shoe, layer, bottom, accessory, or color anchor.",
                "Buy the most reusable missing piece first.",
                "Choose quality and fit over novelty when the piece must work often.",
                "Use shopping ideas as options, then validate against comfort and lifestyle.",
            ],
            "outfit": [
                "Similar-look direction",
                "Missing-piece shortlist",
                "Budget-friendly alternatives",
                "One higher-quality anchor if it solves many outfits",
            ],
            "avoid": [
                "Buying duplicates of pieces you already own",
                "Statement items that only work for one narrow scenario",
            ],
        }

    return {
        "title": "Stylist Advice",
        "intro": f"For {context_phrase}, style it by reading the setting first, then choosing pieces that respect the room and still feel like you.",
        "recommended": [
            "Formality: choose a base that is one step more polished than casual if the setting is mixed.",
            "Cultural sensitivity and modesty: keep coverage and fit respectful when the venue or people around you call for it.",
            "Time, weather, and venue: let fabric weight, layering, and footwear match the real conditions.",
            "Color harmony: use calm neutrals with one controlled accent, or keep the palette tonal.",
            "Silhouette and comfort: pick shapes you can sit, walk, and move in without adjusting constantly.",
        ],
        "outfit": [
            "A clean base piece appropriate to the venue",
            "A second piece that balances the silhouette",
            "Footwear that matches formality and walking comfort",
            "One restrained accessory or layer",
        ],
        "avoid": [
            "Anything too loud for the setting before you understand the dress code",
            "Footwear that contradicts the venue or comfort needs",
            "Too many accessories, shiny details, or competing statement pieces",
        ],
    }


def build_stylist_advice_response(
    *,
    query: str,
    mode: str,
    module_context: str = "",
) -> Dict[str, Any]:
    blocks = _principle_cards(mode, query)
    title = str(blocks["title"])
    intro = str(blocks["intro"])
    recommended = [str(x) for x in blocks["recommended"]]  # type: ignore[index]
    outfit = [str(x) for x in blocks["outfit"]]  # type: ignore[index]
    avoid = [str(x) for x in blocks["avoid"]]  # type: ignore[index]

    message = "\n\n".join(
        [
            title,
            intro,
            "Styling principles:\n" + "\n".join(f"- {x}" for x in recommended),
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
        "intent": mode,
        "message": {"role": "assistant", "content": message},
        "message_text": message,
        "response": message,
        "text": message,
        "cards": [
            {
                "type": "style_principles",
                "title": title,
                "subtitle": intro,
                "principles": recommended,
                "items": outfit,
                "avoid": avoid,
            }
        ],
        "style_boards": [],
        "chips": chips,
        "board_ids": "",
        "data": {
            "intent": mode,
            "style_mode": mode,
            "stylist_mode": True,
            "moodboard": {
                "title": title,
                "principles": recommended,
                "items": outfit,
                "avoid": avoid,
            },
            "cta_actions": chips,
        },
        "meta": {
            "mode": "stylist_knowledge",
            "style_mode": mode,
            "intent": mode,
            "module_context": module_context or "chat",
            "wardrobe_lookup": False,
        },
        "audio_job_id": "offline",
    }
