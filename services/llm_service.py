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
from brain.response_validator import (
    polish_final_text,
    validate_final_text,
)
from prompts.core_prompts import AHVI_SYSTEM_PROMPT
from services.advice_text_guard import protect_newlines_through

logger = logging.getLogger("ahvi.llm_service")

# =========================
# CONFIG
# =========================

load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()

# Gemini / Vertex config
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
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


DEFAULT_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 4096)
DEFAULT_NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", 512)
DEFAULT_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.2)


# =========================
# RESPONSE TOKEN BUDGETS — keep AHVI output complete by default.
# Env-overridable per response type.
# =========================
TOKEN_BUDGETS = {
    "quick_chat": _env_int("AHVI_LLM_TOKENS_QUICK_CHAT", 500),
    "style_advice": _env_int("AHVI_LLM_TOKENS_STYLE_ADVICE", 900),
    "outfit_explanation": _env_int("AHVI_LLM_TOKENS_OUTFIT_EXPLANATION", 1200),
    "board_explanation": _env_int("AHVI_LLM_TOKENS_BOARD_EXPLANATION", 1400),
    "clarification": _env_int("AHVI_LLM_TOKENS_CLARIFICATION", 350),
}
RETRY_ON_TRUNCATION = _env_int("AHVI_LLM_RETRY_ON_TRUNCATION", 1) > 0


def _budget_for(usecase: Optional[str], default: int = 700) -> int:
    if not usecase:
        return default
    key = str(usecase).strip().lower()
    return int(TOKEN_BUDGETS.get(key, default))


def looks_truncated_safe(text: str) -> bool:
    try:
        from brain.response_validator import looks_truncated
        return bool(looks_truncated(text))
    except Exception:
        return False


_GUARD_TRUNCATION_SAFE_FALLBACK = "This looks well put together and balanced."


def _guard_truncation(text: str, *, usecase: Optional[str]) -> str:
    """Validate LLM output; trim to last complete sentence if truncated.

    Never returns the original malformed text as its own fallback -- doing
    so both ships the known-bad text and (since callers compare `guarded !=
    original` to decide whether to retry) silently suppresses
    RETRY_ON_TRUNCATION. If `polish_final_text`'s own salvage/fallback still
    looks truncated, escalate to a hardcoded deterministic safe string.
    """
    if not text:
        return text
    try:
        result = validate_final_text(text)
        if result.get("looks_truncated"):
            logger.warning(
                "ahvi.llm.response_truncated usecase=%s len=%d",
                usecase,
                len(text or ""),
            )
            repaired = polish_final_text(text)
            if looks_truncated_safe(repaired) or repaired == text:
                repaired = _GUARD_TRUNCATION_SAFE_FALLBACK
            return repaired
        return result.get("text") or text
    except Exception:
        return text


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

_gemini_clients: Dict[int, Any] = {}


def _gemini_enabled() -> bool:
    return AI_PROVIDER in {"gemini", "vertex", "vertexai", "google"}


def _get_gemini_client(timeout_seconds: Optional[int] = None):
    if not _gemini_enabled():
        return None

    if genai is None or types is None:
        logger.warning("Gemini requested but google-genai is not installed")
        return None

    timeout = max(1, int(timeout_seconds or GEMINI_TIMEOUT_SECONDS))
    if timeout in _gemini_clients:
        return _gemini_clients[timeout]

    try:
        # Production path: Cloud Run service account + Vertex AI IAM
        client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
            # google-genai expects milliseconds. The public generate_text
            # timeout was previously ignored for Gemini, allowing requests to
            # run well past the caller's latency budget.
            http_options=types.HttpOptions(
                api_version="v1", timeout=timeout * 1000
            ),
        )
        _gemini_clients[timeout] = client
        return client
    except Exception as exc:
        logger.warning("Gemini client init failed: %s", exc)
        return None


_THINKING_DISABLE_MODELS = ("2.5", "2-5")


def _thinking_config_disabled():
    """Return a ThinkingConfig with thinking disabled for models that support
    it (gemini-2.5-*), else None. Guarded so older SDKs / 2.0 models that lack
    the field never break generation."""
    model = str(GEMINI_MODEL or "").lower()
    if not any(tag in model for tag in _THINKING_DISABLE_MODELS):
        return None
    try:
        return types.ThinkingConfig(thinking_budget=0)
    except Exception:
        return None


def _call_gemini_text(
    prompt: str,
    *,
    user_profile: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
    temperature: float = 0.35,
    max_output_tokens: int = 700,
    system_instruction: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
) -> Optional[str]:
    client = _get_gemini_client(timeout_seconds)
    if client is None:
        return None

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    full_prompt = f"""
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
        config_kwargs: Dict[str, Any] = dict(
            system_instruction=system_instruction or AHVI_SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        # gemini-2.5-* models think by default, and those thinking tokens are
        # billed against max_output_tokens. With our larger JSON schemas that
        # starves the visible response (~100 chars), the JSON gets cut off and
        # parsing falls back to deterministic copy. Disable thinking so the full
        # token budget goes to the actual answer.
        thinking_cfg = _thinking_config_disabled()
        if thinking_cfg is not None:
            config_kwargs["thinking_config"] = thinking_cfg
        config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=config,
        )

        # Diagnostic only -- never used to alter model config or behavior.
        # No prompt/user content logged, only the SDK's own termination/usage
        # metadata (when the SDK exposes it) so a future truncation incident
        # can be told apart from a genuinely malformed completion.
        try:
            candidates = getattr(response, "candidates", None) or []
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            usage = getattr(response, "usage_metadata", None)
            logger.info(
                "ahvi.llm.gemini_generation_diagnostics model=%s configured_max_output_tokens=%d "
                "finish_reason=%s prompt_token_count=%s candidates_token_count=%s",
                GEMINI_MODEL,
                max_output_tokens,
                finish_reason,
                getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None),
            )
        except Exception:
            pass

        text = (response.text or "").strip()
        if not text:
            return None

        # tone_engine.apply() flattens all whitespace including newlines,
        # which would destroy the bullet line breaks advice-mode callers
        # (STYLE_ADVICE_FORMAT_CONTRACT) are instructed to return. This is
        # the shared entry point every Gemini text completion in the app
        # goes through -- protect newlines here rather than only in the one
        # caller (services.style_reasoning_engine) that remembered to.
        return protect_newlines_through(
            text,
            lambda t: tone_engine.apply(t, user_profile=user_profile, signals=signals),
        )
    except Exception as exc:
        # Loud log so we can SEE Gemini auth/model/region failures in
        # Cloud Logging instead of silently falling through to Ollama
        # (which doesn't exist on Cloud Run, leaving every user with the
        # canned "This looks well put together and balanced." text).
        logger.error(
            "llm.gemini_call_failed model=%s location=%s err_type=%s err=%s",
            GEMINI_MODEL, GOOGLE_CLOUD_LOCATION, type(exc).__name__, str(exc)[:300],
        )
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
    system_instruction: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Public entry used across AHVI.

    Provider order:
    1. Gemini/Vertex when AI_PROVIDER=vertex/gemini/google
    2. Ollama fallback
    3. deterministic safe text
    """

    # Resolve token budget: explicit options > usecase map > 700.
    requested_tokens = int(
        (options or {}).get("max_output_tokens") or _budget_for(usecase, 700)
    )
    logger.info(
        "ahvi.llm.token_budget usecase=%s max_output_tokens=%d",
        usecase,
        requested_tokens,
    )

    if _gemini_enabled():
        gemini_text = _call_gemini_text(
            prompt,
            user_profile=user_profile,
            signals=signals,
            temperature=float((options or {}).get("temperature", 0.35)),
            max_output_tokens=requested_tokens,
            system_instruction=system_instruction,
            timeout_seconds=timeout_seconds,
        )
        if gemini_text:
            logger.info("llm.generate_text provider=gemini model=%s usecase=%s", GEMINI_MODEL, usecase)
            guarded = _guard_truncation(gemini_text, usecase=usecase)
            if RETRY_ON_TRUNCATION and guarded != gemini_text and looks_truncated_safe(guarded):
                retry_tokens = int(requested_tokens * 1.5)
                logger.info(
                    "ahvi.llm.retry_on_truncation usecase=%s retry_tokens=%d",
                    usecase,
                    retry_tokens,
                )
                retry_text = _call_gemini_text(
                    prompt,
                    user_profile=user_profile,
                    signals=signals,
                    temperature=float((options or {}).get("temperature", 0.35)),
                    max_output_tokens=retry_tokens,
                    system_instruction=system_instruction,
                    timeout_seconds=timeout_seconds,
                )
                if retry_text:
                    return _guard_truncation(retry_text, usecase=usecase)
            return guarded
        logger.warning("llm.generate_text provider=gemini returned empty; falling back usecase=%s", usecase)

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    effective_system_instruction = system_instruction or AHVI_SYSTEM_PROMPT
    full_prompt = f"""
System:
{effective_system_instruction}

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

    # Match Ollama budget to the same usecase-aware target.
    payload.setdefault("options", {})
    if "num_predict" not in payload["options"]:
        payload["options"]["num_predict"] = requested_tokens

    data = _call_ollama(payload, timeout=timeout_seconds)
    if not data:
        logger.warning("llm.generate_text provider=ollama_unavailable usecase=%s", usecase)
        return "This looks well put together and balanced."

    response = str(data.get("response", "")).strip()
    if not response:
        return "This looks well put together and balanced."

    toned = tone_engine.apply(response, user_profile=user_profile, signals=signals)
    return _guard_truncation(toned, usecase=usecase)


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
        system_instruction=system_instruction or None,
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
        options={
            "temperature": 0.35,
            "max_output_tokens": TOKEN_BUDGETS["outfit_explanation"],
        },
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
        options={
            "temperature": 0.4,
            "max_output_tokens": TOKEN_BUDGETS["style_advice"],
        },
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
    # Strip non-fashion rows (chargers, skincare, travel gear) before they
    # can pollute the prompt and resurface as styling suggestions.
    from services.wardrobe_sanitizer import sanitize_fashion_wardrobe_items

    items = sanitize_fashion_wardrobe_items(
        list(items or []), source="format_wardrobe_for_llm"
    )
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
