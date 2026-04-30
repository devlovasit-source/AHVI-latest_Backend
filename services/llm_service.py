
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from brain.tone.tone_engine import tone_engine

# =========================
# CONFIG
# =========================
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_MODEL_FALLBACKS",
        "llama3.1:latest,llama3.1",
    ).split(",")
    if m.strip()
]

DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "300"))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


DEFAULT_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.2)

# =========================
# SESSION
# =========================
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))


# =========================
# CORE CALL
# =========================
def _call_ollama(payload, timeout=30):

    for model in [payload.get("model"), *MODEL_FALLBACKS]:
        try:
            payload["model"] = model
            options = dict(payload.get("options") or {})
            options.setdefault("num_ctx", DEFAULT_NUM_CTX)
            options.setdefault("num_predict", DEFAULT_NUM_PREDICT)
            options.setdefault("temperature", DEFAULT_TEMPERATURE)
            payload["options"] = options

            res = session.post(f"{OLLAMA_URL}/generate", json=payload, timeout=timeout)

            if res.status_code == 200:
                return res.json()

        except Exception:
            continue

    return None


# =========================
# WEATHER OVERLAY
# =========================
def _select_weather_overlay(signals: dict) -> str:

    if not signals:
        return ""

    weather = str(signals.get("weather", "")).lower()

    if "summer" in weather:
        return "The lighter structure keeps this comfortable in heat."
    if "rain" in weather:
        return "A slightly more structured base will handle weather shifts better."
    if "winter" in weather:
        return "Layering would elevate this look."

    return ""


# =========================
# BASE GENERATOR
# =========================
def generate_text(
    prompt,
    user_profile=None,
    signals=None,
    model=None,
    timeout_seconds=30,
    options=None,
    usecase=None,
    **kwargs,
):

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    full_prompt = f"""
You are AHVI, a premium AI fashion stylist.

Tone:
{tone.get("tone_instruction", "")}

STRICT RULES:
- Use provided system reasoning only
- Do NOT hallucinate new reasons
- Be natural and human
- Be concise but insightful
- Sound premium and confident

{prompt}
"""

    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False,
    }
    if options:
        payload["options"] = dict(options)

    data = _call_ollama(payload, timeout=timeout_seconds)

    if not data:
        return "This looks well put together and balanced."

    response = data.get("response", "").strip()

    return tone_engine.apply(response, user_profile=user_profile, signals=signals)


def chat_completion(
    messages,
    system_instruction="",
    model=None,
    user_profile=None,
    signals=None,
    timeout_seconds=30,
    options=None,
    usecase=None,
    **kwargs,
):
    lines = []
    if system_instruction:
        lines.append(f"System:\n{system_instruction}")
    for message in messages or []:
        role = str(message.get("role") or "user").strip()
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role.title()}: {content}")
    lines.append("Assistant:")
    return generate_text(
        "\n\n".join(lines),
        user_profile=user_profile,
        signals=signals,
        model=model,
        timeout_seconds=timeout_seconds,
        options=options,
        usecase=usecase,
    )


# =========================
# 🔥 FOLLOW-UP SUGGESTIONS
# =========================
def generate_followup_suggestions(context: dict) -> list:
    """
    🔥 Interactive UI chip suggestions
    """

    if not context:
        return ["Show outfits", "Try something new"]

    intent = context.get("intent", "general")
    occasion = context.get("occasion")
    aesthetic = context.get("aesthetic")

    suggestions = []

    if intent == "styling":
        suggestions = [
            "Make it sharper",
            "Make it more casual",
            "Change colors",
            "Try another vibe"
        ]

        if occasion:
            suggestions.append(f"More {occasion} appropriate")

        if aesthetic:
            suggestions.append(f"More {aesthetic}")

    elif intent == "refinement":
        suggestions = [
            "Make it cleaner",
            "Add layering",
            "Switch footwear",
            "Tone it down"
        ]

    elif intent == "explore_styles":
        suggestions = [
            "Show minimal styles",
            "Show bold looks",
            "Show streetwear",
            "Try new aesthetics"
        ]

    else:
        suggestions = [
            "Show outfit ideas",
            "Help me style this",
            "Suggest something new",
            "What works better?"
        ]

    return list(dict.fromkeys(suggestions))[:4]


# =========================
# 🔥 OUTFIT EXPLANATION
# =========================
def generate_outfit_explanation(outfits, context="", user_profile=None, signals=None):

    overlay = _select_weather_overlay(signals or {})

    item_explanations = signals.get("item_explanations") if signals else None
    reasons = signals.get("reasons") if signals else None

    prompt = f"""
User wardrobe:
{context}

Outfit options:
{outfits}

SYSTEM REASONING:
Item-level:
{item_explanations}

Score reasons:
{reasons}

Explain:
- why each outfit works
- key differences
- when to wear each

STRICT:
- Use system reasoning as truth
- Do NOT invent reasons
- Rephrase naturally

Optional styling note:
{overlay}
"""

    return generate_text(prompt, user_profile, signals)


# =========================
# 🔥 ITEM EXPLANATION
# =========================
def generate_item_level_explanation(outfit, user_profile=None, signals=None):

    if not outfit:
        return []

    prompt = f"""
Outfit:
{outfit}

System signals:
{signals}

Explain EACH item:
- why it works
- role (base / highlight / balance)
- pairing logic

STRICT:
- Use signals if present
- Do not hallucinate

Return JSON.
"""

    raw = generate_text(prompt, user_profile, signals)

    try:
        import json
        return json.loads(raw)
    except Exception:
        return []


# =========================
# STYLE ADVICE
# =========================
def generate_style_advice(user_input, wardrobe_summary, user_profile=None, signals=None):

    prompt = f"""
User request:
{user_input}

Wardrobe:
{wardrobe_summary}

Give sharp, practical advice.
"""

    return generate_text(prompt, user_profile, signals)


# =========================
# MAIN ENTRY
# =========================
def generate_ai_response(user_input, outfits, wardrobe_items, user_profile=None, signals=None):

    wardrobe_summary = format_wardrobe_for_llm(wardrobe_items)

    if outfits:
        return generate_outfit_explanation(
            outfits,
            wardrobe_summary,
            user_profile,
            signals
        )

    return generate_style_advice(
        user_input,
        wardrobe_summary,
        user_profile,
        signals
    )


# =========================
# FORMATTER
# =========================
def format_wardrobe_for_llm(items):

    if not items:
        return "Wardrobe is empty."

    msg = "Wardrobe:\n"

    for item in items[:50]:
        msg += f"- {item.get('color')} {item.get('type')}\n"

    return msg
