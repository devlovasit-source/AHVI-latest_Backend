from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from prompts.core_prompts import AHVI_SYSTEM_PROMPT
from prompts.styling_prompts import OCCASION_INTERPRETER_PROMPT
from services.ai_gateway import generate_text, parse_json_object


STYLE_ADVICE = "style_advice"
WARDROBE_STYLE = "wardrobe_style"
SHOPPING_ASSIST = "shopping_assist"
STYLE_EDUCATION = "style_education"
COLOR_BODY_ADVICE = "color_body_advice"
VISUAL_INSPIRATION = "visual_inspiration"
STYLE_PAIRING = "style_pairing"
BODY_PROPORTION_ADVICE = "body_proportion_advice"
COLOR_ADVICE = "color_advice"
OCCASION_ADVICE = "occasion_advice"

STYLE_MODES = {
    STYLE_ADVICE,
    WARDROBE_STYLE,
    SHOPPING_ASSIST,
    STYLE_EDUCATION,
    COLOR_BODY_ADVICE,
    STYLE_PAIRING,
    BODY_PROPORTION_ADVICE,
    COLOR_ADVICE,
    OCCASION_ADVICE,
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
    action = _norm(style_action)

    # module_context alone must never decide this (CORE PRINCIPLE: module is a
    # hint, resolved content/intent is authority) -- callers that legitimately
    # want "and it's the wardrobe surface" already check module_context
    # themselves (see e.g. _should_default_visual_inspiration). What follows
    # is content-only.
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
        # Imperative "style me <preposition>" is an execution request, not
        # advice, regardless of whether a specific owned item is named ("style
        # me for a Pooja") or is ("style me using my Red Top and ...") -- it
        # was previously falling through to the generic style_advice default
        # because none of the phrases above matched a bare "style me for/with/
        # using ..." construction.
        "style me for",
        "style me using",
        "style me with",
        "style me in",
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


def is_style_pairing_request(text: Any) -> bool:
    q = _norm(text)
    if not q:
        return False
    return _has_any(
        q,
        (
            "what to pair with",
            "what pairs with",
            "what pairs best with",
            "what pairs",
            "what works with",
            "what goes with",
            "how to style",
            "how do i style",
            "how do i wear",
            "ways to wear",
            "ways to style",
            "how can i wear",
            "what matches",
            "what should i wear with",
            "style this",
            "pair this with",
            "pair with",
            "wear with",
        ),
    )


def is_body_proportion_request(text: Any) -> bool:
    q = _norm(text)
    return _has_any(
        q,
        (
            "look taller", "look shorter", "appear taller", "appear slimmer",
            "look slimmer", "look leaner", "elongate", "proportion", "proportions",
            "body type", "body shape", "what works for my body", "what suits my body",
            "pear body", "apple body", "rectangle body", "hourglass",
            "broad shoulders", "wide hips", "short legs", "long torso",
            "petite", "tall frame", "balance my", "make my legs",
        ),
    )


def is_color_advice_request(text: Any) -> bool:
    q = _norm(text)
    return _has_any(
        q,
        (
            "what colors suit", "what colours suit", "colors suit me", "colours suit me",
            "what colors work", "what colours work", "color that suit", "colors for my",
            "skin tone", "warm skin", "cool skin", "neutral skin", "olive skin",
            "fair skin", "dark skin", "undertone", "color palette for me",
            "what colors should i wear", "best colors for",
        ),
    )


def is_occasion_advice_request(text: Any) -> bool:
    q = _norm(text)
    has_avoid = _has_any(q, ("what to avoid", "what should i avoid", "avoid for",
                             "avoid on a", "dos and don", "do and don", "is it ok to wear",
                             "appropriate for", "what not to wear"))
    has_occasion = _has_any(q, ("date", "coffee", "interview", "wedding", "funeral",
                                "office", "party", "dinner", "meeting", "brunch", "beach"))
    return has_avoid and has_occasion


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
            "why do",
            "why is",
            "what makes",
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
            "work with",
            "go with",
            "pair with",
            "loafer",
            "chino",
            "sneaker",
            "blazer",
            "denim",
            "trouser",
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
        "vacation",
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
        # Speaking / professional occasions — catches 2-word prompts like
        # "conference talk" or "keynote" before they fall to LLM fallback.
        "conference",
        "conference talk",
        "presentation",
        "seminar",
        "panel discussion",
        "panel",
        "keynote",
        # Indian wedding / cultural / festive ceremonies — bare 1-2 word
        # prompts ("Haldi", "Sangeet", "Diwali party") must reach the visual
        # stylist path, not general chat.
        "haldi",
        "mehendi",
        "mehndi",
        "sangeet",
        "engagement",
        "reception",
        "shaadi",
        "baraat",
        "roka",
        "nikah",
        "diwali",
        "eid",
        "navratri",
        "holi",
        "pongal",
        "onam",
        "festive",
        "festival",
        "puja",
        "pooja",
        "mandir",
        "darshan",
        "memorial",
        "condolence",
    )
    if _has_any(q, compact_markers) and _has_any(q, occasionish):
        return True

    # Bare 1-2 word occasion names are a deliberate, narrow shortcut for the
    # specific ceremony/ceremony-adjacent and professional-speaking terms
    # called out above ("Haldi", "Sangeet", "Diwali party", "conference
    # talk", "keynote") -- product already treats naming one of these alone
    # as a styling ask. It must NOT extend to generic occasion words (office,
    # work, dinner, date, party, wedding, meeting, ...): those appear just as
    # often in ordinary emotional/conversational sentences ("Work was
    # horrible today", "The wedding was exhausting") that have zero styling
    # intent, so requiring an explicit style word for them (line above) is
    # the only safe rule.
    _bare_occasion_shortcut_terms = (
        "conference", "conference talk", "presentation", "seminar",
        "panel discussion", "panel", "keynote",
        "haldi", "mehendi", "mehndi", "sangeet", "engagement", "reception",
        "shaadi", "baraat", "roka", "nikah", "diwali", "eid", "navratri",
        "holi", "pongal", "onam", "festive", "festival", "puja", "pooja",
        "mandir", "darshan", "memorial", "condolence",
    )
    if len(q.split()) <= 2 and _has_any(q, _bare_occasion_shortcut_terms):
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
    if is_style_pairing_request(text):
        return STYLE_PAIRING
    if is_shopping_assist_request(text):
        return SHOPPING_ASSIST
    # Specific advice modes BEFORE the broad color_body fallback.
    if is_body_proportion_request(text):
        return BODY_PROPORTION_ADVICE
    if is_occasion_advice_request(text):
        return OCCASION_ADVICE
    if is_color_advice_request(text):
        return COLOR_ADVICE
    if is_color_body_advice_request(text):
        return COLOR_BODY_ADVICE
    if is_style_education_request(text):
        return STYLE_EDUCATION
    if is_style_advice_request(text):
        return STYLE_ADVICE
    # Date / social occasion catch — runs regardless of sentence length so a
    # natural-language prompt like "first coffee date at a cafe on Saturday
    # afternoon" still produces style boards instead of falling to generic
    # chat. Matches an occasion phrase anywhere in the query.
    if _has_any(
        _norm(text),
        (
            "coffee date",
            "cafe date",
            "café date",
            "first date",
            "date night",
            "dinner date",
            "brunch date",
            "lunch date",
            "coffee catch up",
            "coffee catchup",
        ),
    ):
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
        STYLE_PAIRING,
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
  "mode": "style_advice | color_body_advice | style_education | shopping_assist | visual_inspiration | style_pairing",
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
        options={"temperature": 0.55, "max_output_tokens": 1600},
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
        pieces = _string_list(source.get("pieces") or source.get("items") or fb.get("pieces"), limit=6)
        palette = _string_list(source.get("palette") or source.get("colors") or fb.get("palette"), limit=5)
        style_note = str(
            source.get("styling_tip")
            or source.get("style_note")
            or source.get("styleNote")
            or fb.get("styling_tip")
            or fb.get("style_note")
            or ""
        ).strip()
        normalized.append(
            _ensure_direction_logic(
                {
                    "title": str(source.get("title") or fb.get("title") or "Style Direction").strip(),
                    "subtitle": str(source.get("subtitle") or source.get("style_direction") or source.get("styleDirection") or "").strip(),
                    "strategy": str(source.get("strategy") or fb.get("strategy") or "").strip(),
                    "description": str(source.get("description") or fb.get("description") or "").strip(),
                    "hero_piece": str(source.get("hero_piece") or source.get("heroPiece") or (pieces[0] if pieces else "")).strip(),
                    "hero_piece_reasoning": str(source.get("hero_piece_reasoning") or source.get("heroPieceReasoning") or "").strip(),
                    "palette": palette,
                    "colors": palette,
                    "pieces": pieces,
                    "items": pieces,
                    "why_it_works": str(source.get("why_it_works") or fb.get("why_it_works") or "").strip(),
                    "style_note": style_note,
                    "styling_tip": style_note[:80],
                }
            )
        )
    return normalized


# ============================================================================
# AHVI ARCHETYPE LIBRARY (compact in-code registry for persona-aware pairing).
# Gemini selects routes from these — it does not invent random archetypes
# (fallback only). gender_fit: male | female | neutral.
# ============================================================================
import logging as _logging

_archetype_logger = _logging.getLogger("ahvi.archetypes")

ARCHETYPE_LIBRARY = [
    {"name": "Quiet Luxury", "impression": ["refined", "understated"], "formality": 7,
     "best_for": ["dinner", "office", "evening"], "preferred_items": ["fine knit", "tailored trousers", "loafers", "cashmere", "overcoat"],
     "avoid_items": ["loud logos", "athleisure"], "palette": ["camel", "cream", "charcoal", "navy"],
     "gender_fit": "neutral", "style_keywords": ["timeless", "premium", "minimal"]},
    {"name": "Creative Executive", "impression": ["confident", "expressive"], "formality": 6,
     "best_for": ["work", "gallery", "meeting"], "preferred_items": ["unstructured blazer", "knit polo", "pleated trousers", "leather sneakers"],
     "avoid_items": ["stiff suiting"], "palette": ["ink", "rust", "stone"], "gender_fit": "neutral",
     "style_keywords": ["editorial", "modern", "directional"]},
    {"name": "Modern Professional", "impression": ["credible", "polished"], "formality": 8,
     "best_for": ["office", "client", "presentation"], "preferred_items": ["crisp shirt", "tailored trousers", "blazer", "leather loafers", "derbies"],
     "avoid_items": ["shorts", "flip flops"], "palette": ["navy", "white", "grey"], "gender_fit": "neutral",
     "style_keywords": ["sharp", "clean", "boardroom"]},
    {"name": "Power Casual", "impression": ["assured", "relaxed"], "formality": 5,
     "best_for": ["weekend", "smart casual", "drinks"], "preferred_items": ["blazer", "tee", "dark denim", "clean sneakers", "knitwear"],
     "avoid_items": ["full suit"], "palette": ["black", "indigo", "white"], "gender_fit": "neutral",
     "style_keywords": ["effortless", "elevated"]},
    {"name": "Refined Weekend", "impression": ["easy", "intentional"], "formality": 4,
     "best_for": ["weekend", "coffee", "casual"], "preferred_items": ["overshirt", "chinos", "polo", "white sneakers", "knit"],
     "avoid_items": ["formal suiting"], "palette": ["olive", "cream", "tan"], "gender_fit": "neutral",
     "style_keywords": ["soft", "approachable"]},
    {"name": "Off-Duty Tailoring", "impression": ["sharp", "relaxed"], "formality": 6,
     "best_for": ["weekend", "creative office"], "preferred_items": ["blazer", "tee", "denim", "chinos", "loafers", "sneakers", "knitwear"],
     "avoid_items": ["track pants"], "palette": ["navy", "grey", "white"], "gender_fit": "neutral",
     "style_keywords": ["tailored-casual", "modern"]},
    {"name": "Urban Minimalist", "impression": ["clean", "modern"], "formality": 5,
     "best_for": ["city", "weekend", "travel"], "preferred_items": ["monochrome tee", "straight trousers", "sneakers", "overshirt", "bomber"],
     "avoid_items": ["busy prints"], "palette": ["black", "grey", "white"], "gender_fit": "neutral",
     "style_keywords": ["monochrome", "structured"]},
    {"name": "Modern Utility", "impression": ["functional", "rugged"], "formality": 4,
     "best_for": ["weekend", "travel", "casual"], "preferred_items": ["field jacket", "cargo trousers", "tee", "boots", "sneakers"],
     "avoid_items": ["delicate fabrics"], "palette": ["olive", "khaki", "black"], "gender_fit": "neutral",
     "style_keywords": ["workwear", "practical"]},
    {"name": "Relaxed Executive", "impression": ["calm", "senior"], "formality": 6,
     "best_for": ["office", "dinner"], "preferred_items": ["soft blazer", "fine knit", "trousers", "loafers"],
     "avoid_items": ["athleisure"], "palette": ["stone", "navy", "brown"], "gender_fit": "neutral",
     "style_keywords": ["assured", "soft-tailoring"]},
    {"name": "Smart Casual Edge", "impression": ["current", "confident"], "formality": 5,
     "best_for": ["drinks", "weekend", "date"], "preferred_items": ["overshirt", "dark denim", "knit", "leather sneakers", "chelsea boots"],
     "avoid_items": ["formal suiting"], "palette": ["charcoal", "indigo", "black"], "gender_fit": "neutral",
     "style_keywords": ["edge", "modern"]},
    {"name": "Modern Romantic", "impression": ["warm", "considered"], "formality": 6,
     "best_for": ["date", "dinner", "evening"], "preferred_items": ["textured shirt", "trousers", "loafers", "fine knit"],
     "avoid_items": ["harsh utility"], "palette": ["burgundy", "cream", "brown"], "gender_fit": "neutral",
     "style_keywords": ["soft", "intentional"]},
    {"name": "Gallery Night", "impression": ["artful", "sharp"], "formality": 7,
     "best_for": ["evening", "gallery", "event"], "preferred_items": ["black knit", "tailored trousers", "minimal boots", "overcoat"],
     "avoid_items": ["bright logos"], "palette": ["black", "charcoal", "white"], "gender_fit": "neutral",
     "style_keywords": ["monochrome", "editorial"]},
    {"name": "Textural Depth", "impression": ["rich", "tactile"], "formality": 6,
     "best_for": ["autumn", "dinner", "weekend"], "preferred_items": ["corduroy", "wool knit", "suede loafers", "flannel trousers"],
     "avoid_items": ["flat synthetics"], "palette": ["tobacco", "olive", "cream"], "gender_fit": "neutral",
     "style_keywords": ["texture", "layered"]},
    {"name": "Italian Summer", "impression": ["breezy", "polished"], "formality": 5,
     "best_for": ["summer", "resort", "lunch"], "preferred_items": ["linen shirt", "pleated shorts", "loafers", "polo", "linen trousers"],
     "avoid_items": ["heavy wool"], "palette": ["white", "sky", "tan"], "gender_fit": "neutral",
     "style_keywords": ["linen", "vacation"]},
    {"name": "Resort Sophisticate", "impression": ["elegant", "relaxed"], "formality": 5,
     "best_for": ["resort", "beach dinner", "summer"], "preferred_items": ["camp collar shirt", "tailored shorts", "linen trousers", "espadrilles", "loafers"],
     "avoid_items": ["office trousers"], "palette": ["cream", "sand", "teal"], "gender_fit": "neutral",
     "style_keywords": ["resort", "refined"]},
    {"name": "Scandinavian Minimalist", "impression": ["clean", "quiet"], "formality": 5,
     "best_for": ["weekend", "city", "office"], "preferred_items": ["crew knit", "straight trousers", "white sneakers", "wool coat"],
     "avoid_items": ["loud prints"], "palette": ["grey", "ecru", "black"], "gender_fit": "neutral",
     "style_keywords": ["minimal", "functional"]},
    {"name": "Contemporary Classic", "impression": ["timeless", "neat"], "formality": 7,
     "best_for": ["office", "dinner", "event"], "preferred_items": ["oxford shirt", "chinos", "loafers", "navy blazer"],
     "avoid_items": ["trend-chasing pieces"], "palette": ["navy", "white", "tan"], "gender_fit": "neutral",
     "style_keywords": ["classic", "versatile"]},
    {"name": "Elevated Essentials", "impression": ["clean", "intentional"], "formality": 5,
     "best_for": ["everyday", "weekend", "smart casual"], "preferred_items": ["premium tee", "straight trousers", "overshirt", "clean sneakers"],
     "avoid_items": ["fast-fashion logos"], "palette": ["white", "grey", "navy"], "gender_fit": "neutral",
     "style_keywords": ["basics", "quality"]},
    {"name": "Polished Casual", "impression": ["tidy", "approachable"], "formality": 5,
     "best_for": ["coffee", "weekend", "casual office"], "preferred_items": ["polo", "chinos", "loafers", "knit", "clean sneakers"],
     "avoid_items": ["sloppy fits"], "palette": ["cream", "olive", "navy"], "gender_fit": "neutral",
     "style_keywords": ["neat", "easy"]},
    {"name": "Approachable Executive", "impression": ["credible", "warm"], "formality": 7,
     "best_for": ["office", "client", "meeting"], "preferred_items": ["soft blazer", "shirt", "trousers", "loafers", "fine knit"],
     "avoid_items": ["stiff formality"], "palette": ["navy", "grey", "brown"], "gender_fit": "neutral",
     "style_keywords": ["professional", "human"]},
    {"name": "Coastal Minimalist", "impression": ["calm", "clean"], "formality": 4,
     "best_for": ["resort", "summer", "weekend"], "preferred_items": ["linen shirt", "white trousers", "espadrilles", "tee", "loafers"],
     "avoid_items": ["heavy layers"], "palette": ["white", "sky", "sand"], "gender_fit": "neutral",
     "style_keywords": ["minimal", "coastal", "breezy"]},
    {"name": "Modern Prep", "impression": ["neat", "youthful"], "formality": 6,
     "best_for": ["weekend", "smart casual", "campus"], "preferred_items": ["oxford shirt", "chinos", "knit polo", "loafers", "blazer"],
     "avoid_items": ["distressed pieces"], "palette": ["navy", "white", "green"], "gender_fit": "neutral",
     "style_keywords": ["preppy", "classic", "clean"]},
    {"name": "Monochrome Dressing", "impression": ["sharp", "intentional"], "formality": 6,
     "best_for": ["city", "evening", "event"], "preferred_items": ["black knit", "black trousers", "minimal sneakers", "overcoat"],
     "avoid_items": ["busy prints", "clashing colors"], "palette": ["black", "white", "grey"], "gender_fit": "neutral",
     "style_keywords": ["monochrome", "tonal", "minimal"]},
    {"name": "Weekend Explorer", "impression": ["easy", "ready"], "formality": 3,
     "best_for": ["weekend", "outdoor", "travel"], "preferred_items": ["overshirt", "tee", "cargo trousers", "trainers", "denim"],
     "avoid_items": ["formal shoes"], "palette": ["olive", "stone", "rust"], "gender_fit": "neutral",
     "style_keywords": ["casual", "rugged", "practical"]},
    {"name": "Soft Power", "impression": ["assured", "approachable"], "formality": 7,
     "best_for": ["office", "client", "dinner"], "preferred_items": ["soft tailoring", "fine knit", "trousers", "loafers"],
     "avoid_items": ["harsh structure"], "palette": ["taupe", "navy", "cream"], "gender_fit": "neutral",
     "style_keywords": ["refined", "calm", "tailored"]},
    {"name": "Creative Casual", "impression": ["expressive", "relaxed"], "formality": 4,
     "best_for": ["weekend", "creative", "gallery"], "preferred_items": ["printed shirt", "denim", "overshirt", "sneakers", "tee"],
     "avoid_items": ["corporate suiting"], "palette": ["rust", "teal", "ecru"], "gender_fit": "neutral",
     "style_keywords": ["creative", "expressive", "colorful"]},
    {"name": "Travel Uniform", "impression": ["effortless", "composed"], "formality": 4,
     "best_for": ["travel", "airport", "weekend"], "preferred_items": ["knit", "straight trousers", "clean sneakers", "overshirt", "tee"],
     "avoid_items": ["fussy layers"], "palette": ["grey", "navy", "black"], "gender_fit": "neutral",
     "style_keywords": ["comfortable", "minimal", "versatile"]},
    {"name": "Minimal Luxe", "impression": ["premium", "quiet"], "formality": 7,
     "best_for": ["dinner", "evening", "office"], "preferred_items": ["cashmere knit", "tailored trousers", "leather loafers", "overcoat"],
     "avoid_items": ["logos", "loud color"], "palette": ["camel", "charcoal", "cream"], "gender_fit": "neutral",
     "style_keywords": ["luxury", "minimal", "timeless"]},
    {"name": "Contemporary Workwear", "impression": ["modern", "functional"], "formality": 5,
     "best_for": ["creative office", "weekend"], "preferred_items": ["chore jacket", "straight trousers", "tee", "boots", "sneakers"],
     "avoid_items": ["formal suiting"], "palette": ["indigo", "olive", "ecru"], "gender_fit": "neutral",
     "style_keywords": ["workwear", "utility", "modern"]},
    {"name": "Effortless Sophistication", "impression": ["polished", "easy"], "formality": 6,
     "best_for": ["dinner", "smart casual", "evening"], "preferred_items": ["fine knit", "tailored trousers", "loafers", "overshirt"],
     "avoid_items": ["overdone layering"], "palette": ["stone", "navy", "brown"], "gender_fit": "neutral",
     "style_keywords": ["effortless", "refined", "minimal"]},

    # --- Festive / ethnic / ceremony archetypes -------------------------------
    # Added so cultural occasions (haldi, mehendi, sangeet, reception, church
    # wedding, temple, funeral) have real candidates instead of collapsing onto
    # the Western list-head (Quiet Luxury / Creative Executive / Modern Professional).
    {"name": "Festive Heritage", "impression": ["celebratory", "rooted"], "formality": 7,
     "best_for": ["wedding", "reception", "sangeet", "festive", "diwali"], "preferred_items": ["bandhgala", "silk kurta", "nehru jacket", "embroidered jacket", "mojari"],
     "avoid_items": ["western suit", "sneakers"], "palette": ["maroon", "gold", "ivory", "deep green"], "gender_fit": "neutral",
     "style_keywords": ["festive", "ethnic", "heritage", "traditional"]},
    {"name": "Refined Traditional", "impression": ["understated", "elegant"], "formality": 6,
     "best_for": ["wedding", "engagement", "temple", "festive", "puja"], "preferred_items": ["linen kurta", "nehru jacket", "tailored kurta", "churidar", "loafers"],
     "avoid_items": ["loud sequins", "athleisure"], "palette": ["ivory", "sand", "muted gold", "sage"], "gender_fit": "neutral",
     "style_keywords": ["traditional", "ethnic", "understated", "refined"]},
    {"name": "Celebration Kurta", "impression": ["warm", "approachable"], "formality": 5,
     "best_for": ["haldi", "mehendi", "festive", "daytime celebration", "puja"], "preferred_items": ["cotton kurta", "kurta pajama", "printed kurta", "kolhapuri sandals"],
     "avoid_items": ["heavy embroidery", "dark suiting"], "palette": ["yellow", "ochre", "white", "coral"], "gender_fit": "neutral",
     "style_keywords": ["festive", "daytime", "kurta", "ethnic", "relaxed"]},
    {"name": "Sunlit Traditional", "impression": ["fresh", "celebratory"], "formality": 5,
     "best_for": ["haldi", "mehendi", "daytime", "garden", "festive"], "preferred_items": ["light kurta", "linen kurta", "bandi vest", "mojari", "kolhapuri sandals"],
     "avoid_items": ["heavy layers", "dark palette"], "palette": ["marigold", "ivory", "mint", "peach"], "gender_fit": "neutral",
     "style_keywords": ["daytime", "festive", "bright", "ethnic"]},
    {"name": "Wedding Day Ease", "impression": ["composed", "celebratory"], "formality": 6,
     "best_for": ["wedding", "reception", "festive", "guest"], "preferred_items": ["silk kurta", "nehru jacket", "tailored trousers", "loafers", "embroidered stole"],
     "avoid_items": ["casual tee", "sportswear"], "palette": ["champagne", "navy", "wine", "gold"], "gender_fit": "neutral",
     "style_keywords": ["wedding", "festive", "guest", "elevated"]},
    {"name": "Sangeet Statement", "impression": ["expressive", "festive"], "formality": 6,
     "best_for": ["sangeet", "reception", "party", "evening", "festive"], "preferred_items": ["embroidered jacket", "velvet bandi", "silk kurta", "loafers"],
     "avoid_items": ["plain office wear"], "palette": ["emerald", "wine", "ink", "gold"], "gender_fit": "neutral",
     "style_keywords": ["festive", "statement", "evening", "ethnic"]},
    {"name": "Church Formal", "impression": ["dignified", "polished"], "formality": 8,
     "best_for": ["church wedding", "christian wedding", "ceremony", "formal"], "preferred_items": ["two-piece suit", "crisp shirt", "tie", "oxfords", "pocket square"],
     "avoid_items": ["denim", "sneakers", "loud print"], "palette": ["navy", "charcoal", "white", "burgundy"], "gender_fit": "neutral",
     "style_keywords": ["formal", "ceremony", "suit", "classic"]},
    {"name": "Understated Elegance", "impression": ["refined", "quiet"], "formality": 8,
     "best_for": ["ceremony", "funeral", "formal", "church wedding"], "preferred_items": ["dark suit", "fine shirt", "leather oxfords", "tie"],
     "avoid_items": ["bright color", "casual fabric"], "palette": ["charcoal", "black", "slate", "white"], "gender_fit": "neutral",
     "style_keywords": ["formal", "restrained", "elegant", "ceremony"]},
    {"name": "Modern Gentleman", "impression": ["sharp", "warm"], "formality": 7,
     "best_for": ["christian wedding", "ceremony", "evening", "guest"], "preferred_items": ["soft-tailored suit", "knit tie", "loafers", "fine shirt"],
     "avoid_items": ["sportswear", "heavy logos"], "palette": ["navy", "sage", "tan", "white"], "gender_fit": "neutral",
     "style_keywords": ["formal", "modern", "ceremony", "tailored"]},
    {"name": "Temple Modest", "impression": ["calm", "respectful"], "formality": 5,
     "best_for": ["temple", "puja", "modest", "daytime", "festive"], "preferred_items": ["cotton kurta", "linen shirt", "churidar", "loafers", "mojari"],
     "avoid_items": ["shorts", "sleeveless", "loud print"], "palette": ["white", "ivory", "saffron", "sand"], "gender_fit": "neutral",
     "style_keywords": ["modest", "temple", "traditional", "covered"]},
]

# Occasion-family resolver: maps a raw or canonical occasion string to a family
# whose candidate archetype pool is ranked first. This is what stops the
# substring-zero-score collapse — without a family hit every archetype scored 0
# and Python's stable sort handed back the list head every time.
_OCCASION_FAMILY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    # Order matters: more specific cultural cues win over the generic "wedding".
    (("christian wedding", "church wedding", "church", "cathedral", "white wedding"), "christian_ceremony"),
    (("funeral", "memorial", "condolence", "prayer meeting"), "somber_formal"),
    (("temple", "puja", "pooja", "mandir", "darshan"), "temple_modest"),
    (("haldi", "mehendi", "mehndi"), "festive_daytime"),
    (("sangeet", "cocktail sangeet"), "festive_evening"),
    # Music/social "festival" is NOT an ethnic occasion. Match the concert/rave
    # family BEFORE festive_general so "music festival" never resolves to ethnic.
    # Multi-word so it never swallows a cultural phrase like "diwali festival".
    (("music festival", "music fest", "concert", "gig", "rave", "edm",
      "club night", "live show", "house party", "festival ground"), "social_party"),
    # festive_general keeps the genuine cultural cues. Bare "festival" removed —
    # it was pulling music festivals into kurta/bandhgala/mojari.
    (("reception", "engagement", "roka", "wedding", "shaadi", "baraat",
      "diwali", "eid", "navratri", "festive", "ethnic"), "festive_general"),
    # Western occasion families — also pooled so they draw from intentional
    # candidates instead of falling to seeded-tiebreak noise when no keyword
    # substring matches (e.g. "conference", "date night" scored 0 everywhere).
    (("conference", "seminar", "keynote", "presentation", "interview", "client meeting",
      "business meeting", "office meeting", "boardroom", "office", "work"), "professional"),
    (("date", "romantic", "dinner date", "anniversary"), "evening_date"),
    (("party", "club", "cocktail", "night out", "nightout", "celebration drinks"), "social_party"),
    (("beach", "resort", "vacation", "holiday", "poolside"), "resort_summer"),
    (("travel", "airport", "trip", "commute"), "travel_easy"),
    (("brunch", "coffee", "weekend", "casual", "daily", "everyday"), "relaxed_casual"),
)

# Each family lists its candidate archetypes in priority order. Members are
# boosted above the rest of the library; ties inside the pool break by an
# occasion-seeded hash so different occasions don't all collapse to one head.
_FAMILY_ARCHETYPE_POOL: Dict[str, tuple[str, ...]] = {
    "festive_daytime": ("Sunlit Traditional", "Celebration Kurta", "Refined Traditional",
                        "Festive Heritage", "Wedding Day Ease"),
    "festive_evening": ("Sangeet Statement", "Festive Heritage", "Wedding Day Ease",
                        "Refined Traditional", "Celebration Kurta"),
    "festive_general": ("Festive Heritage", "Refined Traditional", "Wedding Day Ease",
                        "Celebration Kurta", "Sangeet Statement", "Sunlit Traditional"),
    "christian_ceremony": ("Church Formal", "Modern Gentleman", "Understated Elegance",
                           "Contemporary Classic", "Modern Professional"),
    "somber_formal": ("Understated Elegance", "Quiet Luxury", "Contemporary Classic",
                      "Modern Professional"),
    "temple_modest": ("Temple Modest", "Refined Traditional", "Celebration Kurta",
                      "Sunlit Traditional"),
    "professional": ("Modern Professional", "Approachable Executive", "Relaxed Executive",
                     "Contemporary Classic", "Soft Power"),
    "evening_date": ("Modern Romantic", "Smart Casual Edge", "Effortless Sophistication",
                     "Power Casual", "Gallery Night"),
    "social_party": ("Monochrome Dressing", "Gallery Night", "Smart Casual Edge",
                     "Power Casual", "Off-Duty Tailoring"),
    "resort_summer": ("Italian Summer", "Resort Sophisticate", "Coastal Minimalist",
                      "Elevated Essentials"),
    "travel_easy": ("Travel Uniform", "Urban Minimalist", "Scandinavian Minimalist",
                    "Elevated Essentials"),
    "relaxed_casual": ("Refined Weekend", "Elevated Essentials", "Polished Casual",
                       "Creative Casual", "Weekend Explorer"),
}

_ARCHETYPE_BY_NAME = {a["name"]: a for a in ARCHETYPE_LIBRARY}


def _resolve_occasion_family(occasion: str) -> str:
    """Map an occasion string (raw 'haldi' or canonical 'wedding') to a family
    key with a candidate archetype pool. Returns '' when no cultural/ceremony
    family applies (the Western library handles those via normal scoring).

    Word-boundary matched: a needle must hit a whole word/phrase, not a
    substring inside another word. Prevents 'rave' matching 't-rave-l'
    (airport travel was wrongly resolving to social_party)."""
    import re as _re

    o = _norm(occasion)
    if not o:
        return ""
    for needles, family in _OCCASION_FAMILY_RULES:
        for n in needles:
            if _re.search(r"(?<![a-z0-9])" + _re.escape(n) + r"(?![a-z0-9])", o):
                return family
    return ""


def _occasion_pool(occasion: str) -> tuple[str, ...]:
    family = _resolve_occasion_family(occasion)
    return _FAMILY_ARCHETYPE_POOL.get(family, ())

# Feminine-only items to strip for a male persona (unless explicitly asked).
FEMININE_ONLY_ITEMS = (
    "skirt", "midi skirt", "maxi skirt", "dress", "midi dress", "gown",
    "camisole", "cami", "blouse", "heels", "stiletto", "pumps", "saree",
    "lehenga", "crop top", "bodycon", "peplum", "bralette",
)


def get_archetype_library() -> List[Dict[str, Any]]:
    _archetype_logger.info("AHVI_ARCHETYPE_LIBRARY_LOADED count=%d", len(ARCHETYPE_LIBRARY))
    return list(ARCHETYPE_LIBRARY)


def _archetype_score(arch, *, anchor_blob, occasion, style_keywords, formality_hint, style_dna):
    text = " ".join(
        str(x) for x in (arch.get("best_for") or []) + (arch.get("style_keywords") or [])
        + (arch.get("preferred_items") or []) + (arch.get("impression") or [])
    ).lower()
    arch_palette = " ".join(str(x) for x in (arch.get("palette") or [])).lower()
    score = 0.0
    # occasion + anchor + keyword base score
    if occasion and occasion in text:
        score += 3.0
    for kw in style_keywords:
        if kw and kw.lower() in text:
            score += 1.5
    for w in anchor_blob.split():
        if w and len(w) > 3 and w in text:
            score += 1.0
    if formality_hint:
        score -= abs(int(arch.get("formality") or 5) - formality_hint) * 0.4

    # Style DNA weighting — this is what personalizes the same anchor for
    # different users (minimalist vs creative).
    dna = style_dna if isinstance(style_dna, dict) else {}
    for aesthetic in dna.get("aesthetics") or dna.get("style_archetypes") or []:
        a = _norm(aesthetic)
        if a and (a in text or a in _norm(arch.get("name"))):
            score += 2.5
    for color in dna.get("preferred_palette") or dna.get("preferred_colors") or dna.get("colors") or []:
        if _norm(color) and _norm(color) in arch_palette:
            score += 1.0
    for sil in dna.get("silhouettes") or dna.get("preferred_silhouettes") or []:
        if _norm(sil) and _norm(sil) in text:
            score += 1.0
    fp = dna.get("formality_preference")
    if isinstance(fp, (int, float)) and fp:
        score -= abs(int(arch.get("formality") or 5) - int(fp)) * 0.5
    return score


def _tiebreak_seed(occasion: str, name: str) -> int:
    """Deterministic, occasion-seeded tiebreak. Replaces list-order tie-breaking
    so equal-scoring archetypes don't always collapse to the library head."""
    import hashlib
    h = hashlib.md5(f"{_norm(occasion)}|{_norm(name)}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _archetype_allowed_for_gender(arch: Dict[str, Any], gender: str) -> bool:
    """Drop archetypes whose gender_fit conflicts with a known persona gender.
    Neutral archetypes always pass; unknown gender disables the filter."""
    g = _norm(gender)
    if g not in {"male", "female"}:
        return True
    fit = _norm(arch.get("gender_fit") or "neutral")
    return fit in {"", "neutral", "unisex", g}


def select_archetypes(*, anchor=None, occasion="", style_keywords=None,
                      formality_hint=0, style_dna=None, gender="unknown", limit=5):
    """Select 4-5 best archetypes.

    occasion -> occasion family -> candidate archetype pool -> ranked.
    A resolved family pool (e.g. haldi -> festive_daytime) is ranked first so
    cultural occasions stop collapsing onto the Western list head. style_dna
    re-ranks within that so the same occasion still personalises per user.
    gender enforces persona context — archetypes whose gender_fit conflicts
    are dropped before ranking.
    """
    anchor = anchor or {}
    anchor_blob = " ".join(
        str(anchor.get(k) or "") for k in ("name", "category", "color")
    ).lower()
    occ = _norm(occasion)
    kws = [k for k in (style_keywords or []) if k]
    dna = style_dna if isinstance(style_dna, dict) else {}
    has_dna = bool(dna.get("aesthetics") or dna.get("style_archetypes") or dna.get("preferred_palette") or dna.get("silhouettes"))

    candidates = [a for a in ARCHETYPE_LIBRARY if _archetype_allowed_for_gender(a, gender)]
    pool = _occasion_pool(occasion)
    pool_rank = {name: i for i, name in enumerate(pool)}

    def _sort_key(a):
        name = a.get("name") or ""
        base = _archetype_score(
            a, anchor_blob=anchor_blob, occasion=occ, style_keywords=kws,
            formality_hint=formality_hint, style_dna=dna,
        )
        in_pool = name in pool_rank
        # Pool members first (priority order), then DNA/occasion score, then a
        # deterministic seeded tiebreak — never raw list order.
        return (
            0 if in_pool else 1,
            pool_rank.get(name, 0),
            -base,
            _tiebreak_seed(occ, name),
        )

    ranked = sorted(candidates, key=_sort_key)
    chosen = ranked[: max(4, min(int(limit), 5))]
    if pool:
        _archetype_logger.info(
            "AHVI_ARCHETYPE_FAMILY occasion=%s family=%s pool=%s",
            occ, _resolve_occasion_family(occasion), list(pool),
        )
    if has_dna:
        _archetype_logger.info("AHVI_STYLE_DNA_APPLIED aesthetics=%s palette=%s", dna.get("aesthetics") or dna.get("style_archetypes"), dna.get("preferred_palette") or dna.get("preferred_colors"))
        _archetype_logger.info("AHVI_ARCHETYPE_RERANKED names=%s", [a["name"] for a in chosen])
    _archetype_logger.info(
        "AHVI_ARCHETYPE_SELECTED anchor=%r occasion=%s dna=%s names=%s",
        anchor_blob[:40], occ, has_dna, [a["name"] for a in chosen],
    )
    return chosen


def resolve_style_archetypes(
    context: Any,
    anchor_item: Optional[Dict[str, Any]] = None,
    direction_count: int = 3,
) -> List[Dict[str, Any]]:
    """Normalize the existing archetype selector for direct styling flows.

    This is intentionally a thin adapter: ``ARCHETYPE_LIBRARY`` and
    ``select_archetypes`` remain the definition and scoring authorities.
    Unknown anchor metadata is neutral rather than restrictive.
    """
    ctx = context if isinstance(context, dict) else {}
    anchor = dict(anchor_item or {})
    count = max(1, min(int(direction_count or 3), 5))
    occasion = str(
        ctx.get("canonical_occasion") or ctx.get("occasion") or "daily"
    ).strip()
    style_tags = _string_list(
        anchor.get("style_tags")
        or anchor.get("tags")
        or ctx.get("style_keywords"),
        limit=8,
    )
    formality = anchor.get("formality") or ctx.get("formality") or 0
    try:
        formality_hint = int(float(formality))
    except (TypeError, ValueError):
        formality_hint = 0
    style_dna = ctx.get("style_dna") if isinstance(ctx.get("style_dna"), dict) else {}
    gender = str(ctx.get("gender") or "unknown")

    selected = select_archetypes(
        anchor=anchor,
        occasion=occasion,
        style_keywords=style_tags,
        formality_hint=formality_hint,
        style_dna=style_dna,
        gender=gender,
        limit=max(count, 4),
    )

    anchor_color = _norm(
        anchor.get("color") or anchor.get("colour") or anchor.get("color_name")
    )
    anchor_role = _norm(
        anchor.get("role")
        or anchor.get("category")
        or anchor.get("sub_category")
        or anchor.get("subcategory")
    )
    anchor_terms = set(_norm(" ".join(style_tags)).split())

    def _compatibility(arch: Dict[str, Any]) -> float:
        score = 0.0
        palette = {_norm(value) for value in arch.get("palette") or []}
        if anchor_color and any(
            anchor_color == color or anchor_color in color or color in anchor_color
            for color in palette if color
        ):
            score += 2.0
        arch_terms = set(
            _norm(
                " ".join(
                    str(value)
                    for value in (
                        (arch.get("style_keywords") or [])
                        + (arch.get("impression") or [])
                    )
                )
            ).split()
        )
        score += 0.5 * len(anchor_terms & arch_terms)
        preferred = _norm(" ".join(str(value) for value in arch.get("preferred_items") or []))
        if anchor_role and any(term in preferred for term in anchor_role.split() if len(term) > 3):
            score += 1.0
        if formality_hint:
            score -= abs(int(arch.get("formality") or 5) - formality_hint) * 0.05
        return score

    # Stable sort preserves the existing selector order when metadata provides
    # no additional compatibility signal.
    selected = sorted(selected, key=_compatibility, reverse=True)[:count]
    out: List[Dict[str, Any]] = []
    used_titles: set[str] = set()
    generic_titles = {"", "general", "default", "style direction", "curated direction"}
    for index, arch in enumerate(selected):
        name = str(arch.get("name") or "").strip()
        title = name
        if _norm(title) in generic_titles:
            impression = [str(v).strip().title() for v in arch.get("impression") or [] if str(v).strip()]
            title = f"{impression[0]} Edit" if impression else f"Style Edit {index + 1}"
        if title.lower() in used_titles:
            title = f"{title} {index + 1}"
        used_titles.add(title.lower())
        intent_parts = [str(v).strip() for v in arch.get("impression") or [] if str(v).strip()]
        if not intent_parts:
            intent_parts = [str(v).strip() for v in arch.get("style_keywords") or [] if str(v).strip()]
        out.append(
            {
                "archetype_id": _norm(name).replace(" ", "_"),
                "direction_title": title,
                "formality": arch.get("formality"),
                "palette": [str(v) for v in arch.get("palette") or [] if str(v).strip()],
                "avoid": [str(v) for v in arch.get("avoid_items") or [] if str(v).strip()],
                "reasoning_intent": ", ".join(intent_parts[:2]) or "intentional styling",
            }
        )
    return out


def filter_items_for_persona(items, gender):
    """Drop feminine-only items for a male persona. Returns (kept, removed)."""
    if _norm(gender) != "male":
        return list(items or []), []
    kept, removed = [], []
    for it in items or []:
        blob = _norm(it)
        if any(f in blob for f in FEMININE_ONLY_ITEMS):
            removed.append(it)
        else:
            kept.append(it)
    return kept, removed


# Loose category synonyms so a route's "cream trousers" matches an owned
# "beige chinos" as a substitution.
_REALITY_CATEGORY = {
    "top": ("shirt", "tee", "t shirt", "polo", "knit", "sweater", "blouse", "overshirt", "kurta", "top"),
    "bottom": ("trouser", "trousers", "chino", "chinos", "jean", "jeans", "pant", "pants", "short", "shorts", "skirt"),
    "footwear": ("shoe", "shoes", "sneaker", "sneakers", "loafer", "loafers", "boot", "boots", "heel", "sandal", "espadrille", "derby", "brogue"),
    "outerwear": ("blazer", "jacket", "coat", "overcoat", "cardigan", "chore"),
}


def _reality_category(blob: str) -> str:
    b = _norm(blob)
    for cat, words in _REALITY_CATEGORY.items():
        if any(w in b for w in words):
            return cat
    return "other"


def score_route_against_wardrobe(route, wardrobe):
    """How realistic is a route for the user's actual wardrobe.
    Returns {match_score, owned_items, missing_items, substitutions, confidence}.
    Matches by exact-ish name first, else same-category substitution."""
    items = route.get("items") if isinstance(route, dict) else route
    items = [str(x).strip() for x in (items or []) if str(x).strip()]
    owned_blobs = []
    for w in wardrobe or []:
        if isinstance(w, dict):
            owned_blobs.append((_norm(w.get("name") or w.get("label")), w))
        elif w:
            owned_blobs.append((_norm(w), {"name": str(w)}))

    owned_items, missing_items, substitutions = [], [], []
    for need in items:
        n = _norm(need)
        need_cat = _reality_category(need)
        # 1) direct keyword overlap
        direct = next(
            (w for blob, w in owned_blobs if blob and (n in blob or any(tok in blob for tok in n.split() if len(tok) > 3))),
            None,
        )
        if direct:
            owned_items.append(need)
            continue
        # 2) same-category substitution
        sub = next(
            (w for blob, w in owned_blobs if blob and _reality_category(blob) == need_cat and need_cat != "other"),
            None,
        )
        if sub:
            sub_name = str(sub.get("name") or sub.get("label") or "owned piece").strip()
            substitutions.append(f"{sub_name} instead of {need}")
            owned_items.append(need)
        else:
            missing_items.append(need)

    total = max(1, len(items))
    covered = len(owned_items)
    match_score = int(round((covered / total) * 100))
    confidence = round(min(1.0, 0.4 + 0.6 * (len(owned_blobs) >= 3)), 2) if owned_blobs else 0.0
    result = {
        "match_score": match_score,
        "owned_items": owned_items,
        "missing_items": missing_items,
        "substitutions": substitutions[:4],
        "confidence": confidence,
    }
    _archetype_logger.info(
        "AHVI_ROUTE_MATCH_SCORED route=%r match=%d owned=%d missing=%d subs=%d",
        _norm(route.get("title") if isinstance(route, dict) else "")[:30],
        match_score, covered, len(missing_items), len(substitutions),
    )
    return result
