from __future__ import annotations

import re
from typing import Any, Dict, List

from prompts.core_prompts import AHVI_SYSTEM_PROMPT
from prompts.styling_prompts import OCCASION_INTERPRETER_PROMPT
from services.ai_gateway import generate_text, parse_json_object


STYLE_ADVICE = "style_advice"
WARDROBE_STYLE = "wardrobe_style"
SHOPPING_ASSIST = "shopping_assist"
STYLE_EDUCATION = "style_education"
COLOR_BODY_ADVICE = "color_body_advice"
VISUAL_INSPIRATION = "visual_inspiration"

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
    safe_query = str(query or "").strip()
    safe_mode = _coerce_advice_mode(mode)
    try:
        payload = _generate_stylist_json(
            query=safe_query,
            mode=safe_mode,
            module_context=module_context,
        )
    except Exception:
        payload = _fallback_stylist_json(query=safe_query, mode=safe_mode)

    payload_mode = _coerce_advice_mode(payload.get("mode") or safe_mode)
    # stylist_reasoning leads; accept legacy stylist_advice as alias.
    advice = str(payload.get("stylist_reasoning") or payload.get("stylist_advice") or "").strip()
    if not advice:
        advice = str(_fallback_stylist_json(query=safe_query, mode=payload_mode)["stylist_reasoning"])
    visual_directions = _normalize_visual_directions(
        payload.get("visual_directions"),
        payload_mode,
        safe_query,
    )
    what_to_avoid = _string_list(payload.get("what_to_avoid"), limit=6)
    goal = str(payload.get("goal") or "").strip() or _fallback_goal(payload_mode, safe_query)
    impression = str(payload.get("impression") or "").strip() or _fallback_impression(safe_query)
    atmosphere = str(payload.get("atmosphere") or "").strip() or _fallback_atmosphere(safe_query)
    missing_piece_reasoning = (
        str(payload.get("missing_piece_reasoning") or "").strip()
        or _fallback_missing_piece(safe_query)
    )
    emotion_state = _coerce_emotion(payload.get("emotion_state"))
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.84))))
    except Exception:
        confidence = 0.84
    occasion = _nullable_text(payload.get("occasion")) or _extract_context_phrase(safe_query)
    follow_up = _nullable_text(payload.get("follow_up_question"))

    chips = _stylist_cta(safe_query)
    cards = [
        {
            "type": "visual_direction",
            "title": direction.get("title"),
            "strategy": direction.get("strategy"),
            "description": direction.get("description"),
            "palette": direction.get("palette"),
            "pieces": direction.get("pieces"),
            "why_it_works": direction.get("why_it_works"),
            "style_note": direction.get("style_note"),
        }
        for direction in visual_directions
    ]

    return {
        "success": True,
        "ok": True,
        "type": "stylist_advice",
        "intent": payload_mode,
        "message": {"role": "assistant", "content": advice},
        "message_text": advice,
        "response": advice,
        "text": advice,
        "cards": cards,
        "style_boards": [],
        "chips": chips,
        "board_ids": "",
        "data": {
            "intent": payload_mode,
            "style_mode": payload_mode,
            "stylist_mode": True,
            "stylist_reasoning": advice,
            "goal": goal,
            "impression": impression,
            "atmosphere": atmosphere,
            "missing_piece_reasoning": missing_piece_reasoning,
            "visual_directions": visual_directions,
            "what_to_avoid": what_to_avoid,
            "follow_up_question": follow_up,
            "cta_actions": chips,
        },
        "meta": {
            "mode": "stylist_knowledge_gemini",
            "style_mode": payload_mode,
            "intent": payload_mode,
            "module_context": module_context or "chat",
            "wardrobe_lookup": False,
            "goal": goal,
            "impression": impression,
            "atmosphere": atmosphere,
            "missing_piece_reasoning": missing_piece_reasoning,
            "emotion_state": emotion_state,
            "confidence": confidence,
            "occasion": occasion,
        },
        "audio_job_id": "offline",
    }


def _coerce_advice_mode(mode: Any) -> str:
    value = _norm(mode).replace(" ", "_")
    if value in {
        STYLE_ADVICE,
        COLOR_BODY_ADVICE,
        STYLE_EDUCATION,
        SHOPPING_ASSIST,
        VISUAL_INSPIRATION,
    }:
        return value
    return STYLE_ADVICE


def _nullable_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return None if not text or text.lower() == "null" else text


def _string_list(value: Any, *, limit: int = 8) -> List[str]:
    if isinstance(value, list):
        out = [str(item or "").strip() for item in value if str(item or "").strip()]
    elif str(value or "").strip():
        out = [str(value).strip()]
    else:
        out = []
    return out[:limit]


def _coerce_emotion(value: Any) -> str:
    emotion = _norm(value)
    if emotion in {"neutral", "excited", "frustrated", "vulnerable", "professional", "social"}:
        return emotion
    q = emotion
    if any(token in q for token in ("client", "meeting", "presentation", "work")):
        return "professional"
    return "neutral"


def _stylist_cta(query: str) -> List[Dict[str, str]]:
    return [
        {"label": "Show visual inspiration", "value": f"Show visual inspiration for: {query}"},
        {"label": "Use My Wardrobe", "value": f"Use my wardrobe for: {query}"},
        {"label": "Show Shopping Ideas", "value": f"Show shopping ideas for: {query}"},
    ]


def _generate_stylist_json(*, query: str, mode: str, module_context: str) -> Dict[str, Any]:
    prompt = f"""
{AHVI_SYSTEM_PROMPT}

{OCCASION_INTERPRETER_PROMPT}

You are AHVI's senior stylist — a real human stylist thinking out loud, not a
fashion database. Classification is already done; do not change wardrobe routing.

Before recommending clothes, decide: (1) what impression the user should
create, (2) what social outcome matters, (3) the styling strategy for the
moment, (4) what to avoid and why, (5) the one missing piece that would most
improve this direction. Lead with the opinion and reasoning; the directions
only support it.

Return ONLY valid JSON matching this schema:
{{
  "mode": "style_advice | color_body_advice | style_education | shopping_assist | visual_inspiration",
  "occasion": string|null,
  "goal": string,
  "impression": string,
  "atmosphere": string,
  "emotion_state": "neutral | excited | frustrated | vulnerable | professional | social",
  "stylist_reasoning": string,
  "what_to_avoid": [string],
  "missing_piece_reasoning": string,
  "visual_directions": [
    {{
      "title": string,
      "strategy": string,
      "description": string,
      "palette": [string],
      "pieces": [string],
      "why_it_works": string,
      "style_note": string
    }}
  ],
  "follow_up_question": string|null,
  "confidence": float
}}

Writing rules:
- stylist_reasoning speaks like a stylist explaining a decision: "This works
  because...", "I would avoid...", "The priority here is...", "The risk is...".
  Social strategy first, clothing logic second. 2-4 sentences, occasion-specific.
- why_it_works explains styling LOGIC, not the pieces. Bad: "Oxford with denim."
  Good: "The shirt creates structure while the denim keeps it approachable."
- missing_piece_reasoning justifies ONE piece and where else it earns its place.
- Ban filler unless unavoidable: "balanced silhouette", "color harmony",
  "approachable and tidy", "elevated aesthetic", "perfect for".

Hard rules:
- Do not generate images. visual_directions are structured cards only.
- For style_advice and visual_inspiration, return exactly 3 visual_directions.
- Each direction must be visibly different by mood, silhouette, palette, or formality.
- Do not output headings named "Styling principles", "Outfit direction", or "Color harmony".
- Do not sound like a fashion textbook. Do not reuse one skeleton per occasion.
- Christian funeral: respectful presence, understated, no bright/flashy.
- Coffee date: approachable confidence, intentional not corporate.
- Client presentation plus drinks: credibility first then social ease,
  transitional styling, avoid full formal that feels awkward later.
- Wedding guest: celebratory restraint, festive without competing, no bridal.
- Beach dinner: relaxed evening polish, no swimwear/flip-flops after sunset.
- Wardrobe lookup is false. Do not mention missing wardrobe items unless asked.

Mode: {mode}
Module context: {module_context or "chat"}
User query: {query}
"""
    raw = generate_text(
        prompt,
        options={"temperature": 0.55, "max_output_tokens": 900},
        signals={"context_mode": "stylist_knowledge", "style_mode": mode},
        usecase="stylist_knowledge",
    )
    parsed = parse_json_object(raw)
    if not isinstance(parsed, dict):
        raise ValueError("stylist knowledge response was not an object")
    return parsed


def _fallback_goal(mode: str, query: str) -> str:
    q = _norm(query)
    if mode == COLOR_BODY_ADVICE:
        return "Find flattering colors and proportions without overcomplicating the outfit."
    if mode == STYLE_EDUCATION:
        return "Explain the style idea in a practical way."
    if mode == SHOPPING_ASSIST:
        return "Identify the right missing piece."
    if "meeting" in q and "party" in q:
        return "Stay credible first, then shift easily into the social setting."
    return "Give useful style advice for the real setting."


def _fallback_atmosphere(query: str) -> str:
    q = _norm(query)
    if any(word in q for word in ("funeral", "memorial", "condolence")):
        return "quiet, respectful, and understated"
    if "coffee" in q and "date" in q:
        return "relaxed, warm, and intentional"
    if "meeting" in q and "party" in q:
        return "professional first, social second"
    if "beach" in q and "dinner" in q:
        return "breathable, relaxed, and evening-aware"
    return "considered and human"


def _fallback_impression(query: str) -> str:
    q = _norm(query)
    if any(word in q for word in ("funeral", "memorial", "condolence")):
        return "understated and considerate"
    if "coffee" in q and "date" in q:
        return "intentional but relaxed"
    if "meeting" in q and "party" in q:
        return "competent but not stiff"
    if "wedding" in q:
        return "festive without competing"
    if "beach" in q and "dinner" in q:
        return "effortless vacation elegance"
    return "considered and self-assured"


def _fallback_missing_piece(query: str) -> str:
    q = _norm(query)
    if any(word in q for word in ("funeral", "memorial", "condolence")):
        return (
            "A pair of clean, closed leather shoes would anchor this and carry "
            "across other formal or respectful settings."
        )
    if "meeting" in q and "party" in q:
        return (
            "A well-cut neutral blazer does the most work here — it sharpens the "
            "look for the room and relaxes for drinks afterward."
        )
    if "coffee" in q and "date" in q:
        return (
            "A brown suede loafer would elevate this and earn its place across "
            "coffee dates, weekend dinners, and smart-casual office days."
        )
    return (
        "One refined pair of shoes would shift this from fine to intentional, "
        "and it would carry across several other settings too."
    )


def _fallback_stylist_json(*, query: str, mode: str) -> Dict[str, Any]:
    blocks = _principle_cards(mode, query)
    intro = str(blocks["intro"])
    recommended = [str(x) for x in blocks["recommended"]]  # type: ignore[index]
    outfit = [str(x) for x in blocks["outfit"]]  # type: ignore[index]
    avoid = [str(x) for x in blocks["avoid"]]  # type: ignore[index]
    reasoning = " ".join([intro, "Try " + ", ".join(outfit[:3]).lower() + "."])
    return {
        "mode": mode,
        "occasion": _extract_context_phrase(query),
        "goal": _fallback_goal(mode, query),
        "impression": _fallback_impression(query),
        "atmosphere": _fallback_atmosphere(query),
        "emotion_state": "professional" if "meeting" in _norm(query) else "neutral",
        "stylist_reasoning": reasoning,
        "stylist_advice": reasoning,
        "what_to_avoid": avoid,
        "missing_piece_reasoning": _fallback_missing_piece(query),
        "visual_directions": _fallback_visual_directions(query, recommended, outfit),
        "follow_up_question": None,
        "confidence": 0.72,
    }


def _fallback_visual_directions(
    query: str,
    recommended: List[str],
    outfit: List[str],
) -> List[Dict[str, Any]]:
    q = _norm(query)
    if "funeral" in q:
        return [
            {
                "title": "Quiet Formal",
                "description": "Dark, clean, and respectful with closed footwear.",
                "palette": ["black", "charcoal", "deep navy"],
                "pieces": ["plain top", "tailored trousers", "closed shoes"],
                "style_note": "Let the outfit stay in the background.",
            },
            {
                "title": "Soft Traditional",
                "description": "Modest coverage with muted color and gentle texture.",
                "palette": ["ink blue", "soft white", "deep grey"],
                "pieces": ["modest shirt", "straight bottom", "simple layer"],
                "style_note": "Culturally aware without looking ceremonial unless the setting asks for it.",
            },
            {
                "title": "Minimal Polished",
                "description": "A tonal outfit with structure and almost no shine.",
                "palette": ["navy", "stone", "black"],
                "pieces": ["structured shirt", "clean bottom", "low-profile shoes"],
                "style_note": "Respectful, calm, and easy to sit in.",
            },
        ]
    if "coffee" in q and "date" in q:
        return [
            {
                "title": "Relaxed Oxford",
                "description": "A crisp shirt with dark denim and clean sneakers.",
                "palette": ["navy", "white", "tan"],
                "pieces": ["Oxford shirt", "dark denim", "clean sneakers"],
                "style_note": "Approachable without trying too hard.",
            },
            {
                "title": "Knit Polo Polish",
                "description": "Soft texture with sharper bottoms and refined shoes.",
                "palette": ["cream", "olive", "brown"],
                "pieces": ["knit polo", "straight trousers", "loafers"],
                "style_note": "Warm, intentional, and still relaxed.",
            },
            {
                "title": "Soft Layered Casual",
                "description": "A light layer over a simple base.",
                "palette": ["stone", "blue", "charcoal"],
                "pieces": ["light jacket", "plain tee", "chinos"],
                "style_note": "Good when the date moves from cafe to a walk.",
            },
        ]
    return [
        {
            "title": "Clean Base",
            "description": recommended[0] if recommended else "Start with a clean base that matches the setting.",
            "palette": ["navy", "white", "tan"],
            "pieces": outfit[:3] or ["simple top", "clean bottom", "appropriate shoes"],
            "style_note": "The safest direction when the room is unfamiliar.",
        },
        {
            "title": "Soft Polish",
            "description": recommended[1] if len(recommended) > 1 else "Add one polished detail without making the outfit stiff.",
            "palette": ["cream", "olive", "brown"],
            "pieces": outfit[1:4] or ["soft layer", "straight trousers", "refined shoes"],
            "style_note": "Useful when you want confidence without formality.",
        },
        {
            "title": "Easy Statement",
            "description": recommended[2] if len(recommended) > 2 else "Use one controlled color or texture as the focal point.",
            "palette": ["charcoal", "blue", "stone"],
            "pieces": outfit[2:5] or ["hero piece", "quiet base", "grounded footwear"],
            "style_note": "Keeps the outfit human rather than textbook.",
        },
    ]


def _ensure_direction_logic(direction: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee why_it_works + strategy so fallback cards still read like a
    stylist explaining logic, not a database listing pieces."""
    pieces = direction.get("pieces") or []
    title = str(direction.get("title") or "this direction").strip()
    if not direction.get("why_it_works"):
        if len(pieces) >= 2:
            direction["why_it_works"] = (
                f"The {str(pieces[0]).lower()} sets the structure while the "
                f"{str(pieces[1]).lower()} keeps it easy — so the look reads "
                "intentional without trying too hard."
            )
        else:
            direction["why_it_works"] = (
                f"{title} keeps one clear focal point so the outfit feels "
                "deliberate rather than busy."
            )
    if not direction.get("strategy"):
        direction["strategy"] = str(direction.get("style_note") or "").strip()
    return direction


def _normalize_visual_directions(value: Any, mode: str, query: str) -> List[Dict[str, Any]]:
    fallback = _fallback_stylist_json(query=query, mode=mode).get("visual_directions", [])
    rows = value if isinstance(value, list) else []
    normalized = []
    for idx in range(3):
        source = rows[idx] if idx < len(rows) and isinstance(rows[idx], dict) else {}
        fb = fallback[idx] if idx < len(fallback) and isinstance(fallback[idx], dict) else {}
        normalized.append(
            _ensure_direction_logic(
                {
                    "title": str(source.get("title") or fb.get("title") or "Style Direction").strip(),
                    "strategy": str(source.get("strategy") or fb.get("strategy") or "").strip(),
                    "description": str(source.get("description") or fb.get("description") or "").strip(),
                    "palette": _string_list(source.get("palette") or fb.get("palette"), limit=5),
                    "pieces": _string_list(source.get("pieces") or fb.get("pieces"), limit=6),
                    "why_it_works": str(source.get("why_it_works") or fb.get("why_it_works") or "").strip(),
                    "style_note": str(source.get("style_note") or source.get("styleNote") or fb.get("style_note") or "").strip(),
                }
            )
        )
    return normalized
