import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from google import genai
    from google.genai import types
except (
    Exception
):  # google-genai may be absent in local/dev until requirements are installed
    genai = None
    types = None

from brain.tone.tone_engine import tone_engine
from prompts.core_prompts import AHVI_SYSTEM_PROMPT

logger = logging.getLogger("ahvi.llm_service")

# =========================
# CONFIG
# =========================

load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()

# Gemini / Vertex config
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_CLOUD_PROJECT = (
    os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "ahvi-485510"
)
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "45"))

# Ollama config kept as fallback
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_MODEL_FALLBACKS",
        "llama3.2:3b,llama3.2:latest,llama3.1:latest,llama3.1",
    ).split(",")
    if m.strip()
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


DEFAULT_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 1024)
DEFAULT_NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", 300)
DEFAULT_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.2)


# =========================
# HTTP SESSION FOR OLLAMA
# =========================

session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))


# =========================
# GEMINI CLIENT
# =========================

_gemini_client = None


def _gemini_enabled() -> bool:
    return AI_PROVIDER in {"gemini", "vertex", "vertexai", "google"}


def _get_gemini_client():
    global _gemini_client

    if not _gemini_enabled():
        return None

    if genai is None or types is None:
        logger.warning("Gemini requested but google-genai is not installed")
        return None

    if _gemini_client is not None:
        return _gemini_client

    try:
        # Production path: Cloud Run service account + Vertex AI IAM
        _gemini_client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
            http_options=types.HttpOptions(api_version="v1"),
        )
        return _gemini_client
    except Exception as exc:
        logger.warning("Gemini client init failed: %s", exc)
        return None


def _call_gemini_text(
    prompt: str,
    *,
    user_profile: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
    temperature: float = 0.35,
    max_output_tokens: int = 450,
    system_instruction: Optional[str] = None,
) -> Optional[str]:
    client = _get_gemini_client()
    if client is None:
        return None

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    full_prompt = f"""
{AHVI_SYSTEM_PROMPT}

Tone instruction:
{tone.get("tone_instruction", "")}

Strict rules:
- Use only the provided wardrobe/outfit/system reasoning.
- Do not invent garments, colors, weather, user preferences, or reasons.
- Do not override AHVI safety/category decisions.
- Be concise, premium, and practical.

Task:
{prompt}
""".strip()

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
            or AHVI_SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=config,
        )

        text = (response.text or "").strip()
        if not text:
            return None

        return tone_engine.apply(text, user_profile=user_profile, signals=signals)
    except Exception as exc:
        logger.warning("Gemini text call failed: %s", exc)
        return None


# =========================
# OLLAMA FALLBACK
# =========================


def _call_ollama(
    payload: Dict[str, Any], timeout: int = 30
) -> Optional[Dict[str, Any]]:
    models = [payload.get("model") or DEFAULT_MODEL, *MODEL_FALLBACKS]
    seen = set()

    for model in models:
        if not model or model in seen:
            continue
        seen.add(model)

        try:
            current = dict(payload)
            current["model"] = model

            options = dict(current.get("options") or {})
            options.setdefault("num_ctx", DEFAULT_NUM_CTX)
            options.setdefault("num_predict", DEFAULT_NUM_PREDICT)
            options.setdefault("temperature", DEFAULT_TEMPERATURE)
            current["options"] = options

            res = session.post(f"{OLLAMA_URL}/generate", json=current, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            logger.warning(
                "Ollama call failed status=%s model=%s body=%s",
                res.status_code,
                model,
                res.text[:200],
            )
        except Exception as exc:
            logger.warning("Ollama call exception model=%s error=%s", model, exc)

    return None


# =========================
# WEATHER OVERLAY
# =========================


def _select_weather_overlay(signals: Optional[Dict[str, Any]]) -> str:
    if not signals:
        return ""

    weather = str(signals.get("weather", "")).lower()
    weather_mode = str(signals.get("weather_mode", "")).lower()

    joined = f"{weather} {weather_mode}"

    if "hot" in joined or "summer" in joined:
        return "The lighter structure keeps this comfortable in heat."
    if "rain" in joined:
        return "A slightly more structured base will handle weather shifts better."
    if "winter" in joined or "cold" in joined:
        return "Layering would elevate this look."

    return ""


# =========================
# BASE GENERATOR
# =========================


def generate_text(
    prompt: str,
    user_profile: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    timeout_seconds: int = 30,
    options: Optional[Dict[str, Any]] = None,
    usecase: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Public entry used across AHVI.

    Provider order:
    1. Gemini/Vertex when AI_PROVIDER=vertex/gemini/google
    2. Ollama fallback
    3. deterministic safe text
    """

    if _gemini_enabled():
        gemini_text = _call_gemini_text(
            prompt,
            user_profile=user_profile,
            signals=signals,
            temperature=float((options or {}).get("temperature", 0.35)),
            max_output_tokens=int((options or {}).get("max_output_tokens", 450)),
        )
        if gemini_text:
            logger.info("llm.generate_text provider=gemini model=%s usecase=%s", GEMINI_MODEL, usecase)
            return gemini_text
        logger.warning("llm.generate_text provider=gemini returned empty; falling back usecase=%s", usecase)

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    full_prompt = f"""
{AHVI_SYSTEM_PROMPT}

Tone: {tone.get("tone_instruction", "")}

STRICT RULES:
- Use provided system reasoning only
- Do NOT hallucinate new reasons
- Be natural and human
- Be concise but insightful
- Sound premium and confident

{prompt}
""".strip()

    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False,
    }

    if options:
        payload["options"] = dict(options)

    data = _call_ollama(payload, timeout=timeout_seconds)
    if not data:
        logger.warning("llm.generate_text provider=ollama_unavailable usecase=%s", usecase)
        return "This looks well put together and balanced."

    response = str(data.get("response", "")).strip()
    if not response:
        return "This looks well put together and balanced."

    return tone_engine.apply(response, user_profile=user_profile, signals=signals)


def chat_completion(
    messages: List[Dict[str, str]],
    system_instruction: str = "",
    model: Optional[str] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 30,
    options: Optional[Dict[str, Any]] = None,
    usecase: Optional[str] = None,
    **kwargs,
) -> str:
    lines: List[str] = []

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
# FOLLOW-UP SUGGESTIONS
# =========================


def generate_followup_suggestions(context: Optional[Dict[str, Any]]) -> List[str]:
    if not context:
        return ["Show outfits", "Try something new"]

    intent = context.get("intent", "general")
    occasion = context.get("occasion")
    aesthetic = context.get("aesthetic")

    if intent == "styling":
        suggestions = [
            "Make it sharper",
            "Make it more casual",
            "Change colors",
            "Try another vibe",
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
            "Tone it down",
        ]
    elif intent == "explore_styles":
        suggestions = [
            "Show minimal styles",
            "Show bold looks",
            "Show streetwear",
            "Try new aesthetics",
        ]
    else:
        suggestions = [
            "Show outfit ideas",
            "Help me style this",
            "Suggest something new",
            "What works better?",
        ]

    return list(dict.fromkeys(suggestions))[:4]


# =========================
# OUTFIT EXPLANATION
# =========================


def generate_outfit_explanation(
    outfits: Any,
    context: Any = "",
    user_profile: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
) -> str:
    signals = signals or {}
    overlay = _select_weather_overlay(signals)

    item_explanations = signals.get("item_explanations")
    reasons = signals.get("reasons")

    prompt = f"""
User wardrobe/context:
{context}

Outfit options:
{outfits}

System reasoning:
Item-level:
{item_explanations}

Score reasons:
{reasons}

Explain:
- why the selected outfit works
- why the pieces belong together
- when to wear it

Strict:
- Use system reasoning as truth
- Do not invent unavailable items
- Do not mention garments not present in the outfit
- Keep it concise

Optional styling note:
{overlay}
""".strip()

    return generate_text(
        prompt,
        user_profile=user_profile,
        signals=signals,
        options={"temperature": 0.35, "max_output_tokens": 420},
        usecase="outfit_explanation",
    )


# =========================
# ITEM EXPLANATION
# =========================


def generate_item_level_explanation(
    outfit: Any,
    user_profile: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not outfit:
        return []

    prompt = f"""
Outfit:
{outfit}

System signals:
{signals}

Explain EACH item:
- why it works
- role: base / highlight / balance
- pairing logic

Return JSON list only.
""".strip()

    raw = generate_text(
        prompt,
        user_profile=user_profile,
        signals=signals,
        options={"temperature": 0.2, "max_output_tokens": 500},
        usecase="item_level_explanation",
    )

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


# =========================
# STYLE ADVICE
# =========================


def generate_style_advice(
    user_input: str,
    wardrobe_summary: str,
    user_profile: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = f"""
User request:
{user_input}

Wardrobe:
{wardrobe_summary}

Give sharp, practical advice.
Do not invent wardrobe items.
Keep it premium and concise.
""".strip()

    return generate_text(
        prompt,
        user_profile=user_profile,
        signals=signals,
        options={"temperature": 0.4, "max_output_tokens": 450},
        usecase="style_advice",
    )


# =========================
# MAIN ENTRY
# =========================


def generate_ai_response(
    user_input: str,
    outfits: Any,
    wardrobe_items: List[Dict[str, Any]],
    user_profile: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
) -> str:
    wardrobe_summary = format_wardrobe_for_llm(wardrobe_items)

    if outfits:
        return generate_outfit_explanation(
            outfits,
            wardrobe_summary,
            user_profile,
            signals,
        )

    return generate_style_advice(
        user_input,
        wardrobe_summary,
        user_profile,
        signals,
    )


# =========================
# FORMATTER
# =========================


def format_wardrobe_for_llm(items: Optional[List[Dict[str, Any]]]) -> str:
    if not items:
        return "Wardrobe is empty."

    msg = "Wardrobe:\n"

    for item in items[:50]:
        if not isinstance(item, dict):
            continue

        color = item.get("color") or item.get("dominant_color") or "unknown color"
        item_type = (
            item.get("type")
            or item.get("category")
            or item.get("garment_type")
            or "item"
        )
        name = item.get("name") or item.get("title") or item.get("label") or ""

        line = f"- {color} {item_type}"
        if name:
            line += f" ({name})"
        msg += line + "\n"

    return msg
