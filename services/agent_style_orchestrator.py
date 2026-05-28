"""AHVI Style Orchestrator — thin reasoning layer in front of the existing
style engine.

This module calls the Google Agent Studio / Gemini-backed AHVI Style
Orchestrator Agent to produce a small structured intent payload that the
existing backend (style_flow_service → orchestrator → outfit_pipeline →
style_scorer → outfit_quality_guard) uses as additional signal.

The integration is fully behind the env flag ``ENABLE_AGENT_STYLE_ORCHESTRATOR``
and falls back silently to the legacy backend flow when:

* the flag is disabled
* the agent call fails
* the returned payload is malformed

This file intentionally does NOT replace any existing style logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ahvi.agent.style_orchestrator")


# ---------------------------------------------------------------------------
# Env knobs
# ---------------------------------------------------------------------------

ENV_ENABLE = "ENABLE_AGENT_STYLE_ORCHESTRATOR"
ENV_MODEL = "AGENT_STYLE_ORCHESTRATOR_MODEL"
ENV_TIMEOUT = "AGENT_STYLE_ORCHESTRATOR_TIMEOUT_SECONDS"
ENV_ENDPOINT = "AGENT_STYLE_ORCHESTRATOR_ENDPOINT"
ENV_API_KEY = "AGENT_STYLE_ORCHESTRATOR_API_KEY"

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_TIMEOUT = 12.0


def is_enabled() -> bool:
    return str(os.getenv(ENV_ENABLE, "")).strip().lower() in {"1", "true", "yes", "on"}


def _model_name() -> str:
    return os.getenv(ENV_MODEL, DEFAULT_MODEL) or DEFAULT_MODEL


def _timeout_seconds() -> float:
    try:
        return float(os.getenv(ENV_TIMEOUT, str(DEFAULT_TIMEOUT)))
    except Exception:
        return DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Schema + validation
# ---------------------------------------------------------------------------

_DEFAULT_PAYLOAD: Dict[str, Any] = {
    "occasion": "casual",
    "sub_intent": "outfit_generation",
    "formality": "mid",
    "dress_code": "",
    "style_direction": "elevated_basics",
    "wardrobe_usage": "preferred",
    "avoid_items": [],
    "required_slots": ["top", "bottom", "footwear"],
    "palette_direction": [],
    "accessory_policy": {},
    "clarification_needed": False,
    "clarification_reason": "",
    "confidence": 0.0,
}


def _coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _coerce_str_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _coerce_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_confidence(value: Any) -> float:
    try:
        c = float(value)
    except Exception:
        return 0.0
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    return c


def validate_agent_style_payload(payload: Any) -> Dict[str, Any]:
    """Normalize and defensively coerce an agent payload to the documented schema.

    Missing or invalid fields fall back to safe defaults. Always returns a
    full payload — callers can rely on every key being present.
    """
    raw = payload if isinstance(payload, dict) else {}

    validated: Dict[str, Any] = {
        "occasion": _coerce_str(raw.get("occasion"), _DEFAULT_PAYLOAD["occasion"]),
        "sub_intent": _coerce_str(raw.get("sub_intent"), _DEFAULT_PAYLOAD["sub_intent"]),
        "formality": _coerce_str(raw.get("formality"), _DEFAULT_PAYLOAD["formality"]),
        "dress_code": _coerce_str(raw.get("dress_code"), _DEFAULT_PAYLOAD["dress_code"]),
        "style_direction": _coerce_str(
            raw.get("style_direction"), _DEFAULT_PAYLOAD["style_direction"]
        ),
        "wardrobe_usage": _coerce_str(
            raw.get("wardrobe_usage"), _DEFAULT_PAYLOAD["wardrobe_usage"]
        ),
        "avoid_items": _coerce_str_list(raw.get("avoid_items")),
        "required_slots": _coerce_str_list(raw.get("required_slots"))
        or list(_DEFAULT_PAYLOAD["required_slots"]),
        "palette_direction": _coerce_str_list(raw.get("palette_direction")),
        "accessory_policy": _coerce_dict(raw.get("accessory_policy")),
        "clarification_needed": _coerce_bool(raw.get("clarification_needed"), False),
        "clarification_reason": _coerce_str(raw.get("clarification_reason"), ""),
        "confidence": _coerce_confidence(raw.get("confidence")),
    }
    return validated


def default_agent_payload() -> Dict[str, Any]:
    """Return a deep-copy of the safe default payload."""
    return validate_agent_style_payload(None)


# ---------------------------------------------------------------------------
# Agent transport (Gemini / Agent Studio)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are the AHVI Style Orchestrator. Convert the user's styling request, "
    "wardrobe context, weather and profile into a STRICT JSON object with "
    "exactly these keys: occasion, sub_intent, formality, dress_code, "
    "style_direction, wardrobe_usage, avoid_items, required_slots, "
    "palette_direction, accessory_policy, clarification_needed, "
    "clarification_reason, confidence. Return ONLY the JSON object, no prose."
)


def _build_user_prompt(
    *,
    message: str,
    chips: Optional[List[str]],
    wardrobe_items: Optional[List[Dict[str, Any]]],
    weather: Optional[Dict[str, Any]],
    profile: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> str:
    safe_wardrobe: List[Dict[str, Any]] = []
    for item in (wardrobe_items or [])[:60]:
        if not isinstance(item, dict):
            continue
        safe_wardrobe.append(
            {
                "id": item.get("id") or item.get("$id") or item.get("item_id"),
                "name": item.get("name") or item.get("title") or item.get("label"),
                "category": item.get("category") or item.get("type"),
                "color": item.get("color") or item.get("dominant_color"),
                "material": item.get("material") or item.get("fabric"),
            }
        )

    safe_profile = {
        k: v
        for k, v in (profile or {}).items()
        if k
        in {
            "gender",
            "style_gender",
            "stylePreferences",
            "style_preferences",
            "skinTone",
            "skin_tone",
            "bodyShape",
            "body_shape",
            "locationLabel",
        }
    }

    payload = {
        "message": message or "",
        "chips": list(chips or []),
        "wardrobe": safe_wardrobe,
        "weather": dict(weather or {}),
        "profile": safe_profile,
        "existing_intent": (context or {}).get("occasion")
        or (context or {}).get("intent")
        or "",
    }
    return json.dumps(payload, ensure_ascii=False)


async def _call_gemini_agent(prompt: str, *, model: str, timeout: float) -> Optional[Dict[str, Any]]:
    """Call the configured AHVI Style Orchestrator Agent endpoint.

    The transport is implementation-flexible: prefer an existing AI gateway,
    fall back to a generic HTTP call when an explicit endpoint is supplied,
    otherwise return ``None`` so callers fall back to legacy logic.
    """
    endpoint = os.getenv(ENV_ENDPOINT, "").strip()
    api_key = os.getenv(ENV_API_KEY, "").strip()

    # Preferred path: reuse the project's AI gateway if it exposes a
    # compatible agent call. This keeps auth/retries consistent with the
    # rest of the backend.
    try:
        from services import ai_gateway  # type: ignore

        call = getattr(ai_gateway, "call_agent", None) or getattr(
            ai_gateway, "call_gemini", None
        )
        if callable(call):
            result = call(
                model=model,
                system=_SYSTEM_PROMPT,
                prompt=prompt,
                timeout=timeout,
                response_format="json",
            )
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=timeout)
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except Exception:
                    return None
    except Exception:
        logger.debug("ahvi.agent.gateway_unavailable", exc_info=True)

    if not endpoint:
        return None

    try:
        import httpx  # type: ignore
    except Exception:
        logger.debug("ahvi.agent.httpx_unavailable")
        return None

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "system": _SYSTEM_PROMPT,
        "input": prompt,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("ahvi.agent.transport_failed error=%s", str(exc)[:200])
        return None

    if isinstance(data, dict):
        # Common Gemini-style envelope: { output: { content: "..." } }
        if isinstance(data.get("output"), dict):
            content = data["output"].get("content") or data["output"].get("text")
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except Exception:
                    return None
        if isinstance(data.get("content"), str):
            try:
                return json.loads(data["content"])
            except Exception:
                return None
        return data
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def orchestrate_style_request(
    message: str,
    user_id: Optional[str] = None,
    wardrobe_items: Optional[List[Dict[str, Any]]] = None,
    chips: Optional[List[str]] = None,
    weather: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the AHVI Style Orchestrator agent and return a validated payload.

    Always returns a payload matching the documented schema. If the agent is
    disabled or fails for any reason, returns the safe default payload so the
    existing backend flow continues to operate.
    """
    if not is_enabled():
        return default_agent_payload()

    model = _model_name()
    timeout = _timeout_seconds()
    prompt = _build_user_prompt(
        message=message,
        chips=chips,
        wardrobe_items=wardrobe_items,
        weather=weather,
        profile=profile,
        context=context,
    )

    try:
        raw = await asyncio.wait_for(
            _call_gemini_agent(prompt, model=model, timeout=timeout),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("ahvi.agent.timeout user_id=%s model=%s", user_id, model)
        return default_agent_payload()
    except Exception as exc:
        logger.warning(
            "ahvi.agent.call_failed user_id=%s error=%s", user_id, str(exc)[:200]
        )
        return default_agent_payload()

    if not raw:
        return default_agent_payload()

    validated = validate_agent_style_payload(raw)
    logger.info(
        "ahvi.agent.style_orchestration user_id=%s occasion=%s sub_intent=%s "
        "formality=%s style_direction=%s clarification_needed=%s confidence=%.2f",
        user_id,
        validated.get("occasion"),
        validated.get("sub_intent"),
        validated.get("formality"),
        validated.get("style_direction"),
        validated.get("clarification_needed"),
        validated.get("confidence"),
    )
    return validated


def orchestrate_style_request_sync(
    message: str,
    user_id: Optional[str] = None,
    wardrobe_items: Optional[List[Dict[str, Any]]] = None,
    chips: Optional[List[str]] = None,
    weather: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sync convenience wrapper for callers that are not async.

    Safe to call from inside synchronous code paths (e.g. the existing
    ``build_style_flow_response`` function). Falls back to the default payload
    if no usable event loop strategy is available.
    """
    if not is_enabled():
        return default_agent_payload()

    coro = orchestrate_style_request(
        message=message,
        user_id=user_id,
        wardrobe_items=wardrobe_items,
        chips=chips,
        weather=weather,
        profile=profile,
        context=context,
    )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside an active loop (e.g. FastAPI request) — spawn a
            # short-lived loop in a worker thread so we do not block the
            # surrounding loop.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=_timeout_seconds() + 2)
        return loop.run_until_complete(coro)
    except RuntimeError:
        try:
            return asyncio.run(coro)
        except Exception as exc:
            logger.warning("ahvi.agent.sync_wrapper_failed error=%s", str(exc)[:200])
            return default_agent_payload()
    except Exception as exc:
        logger.warning("ahvi.agent.sync_wrapper_failed error=%s", str(exc)[:200])
        return default_agent_payload()


# ---------------------------------------------------------------------------
# Context merge helpers (used by style_flow_service and orchestrator)
# ---------------------------------------------------------------------------

def merge_agent_payload_into_context(
    style_context: Dict[str, Any], agent_payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge a validated agent payload into the existing style context.

    Existing keys take precedence only when the agent payload's value is the
    default fallback — this keeps backward compatibility but still lets the
    agent's signal upgrade weak/missing context.
    """
    if not isinstance(style_context, dict):
        return style_context
    if not isinstance(agent_payload, dict):
        return style_context

    style_context["agent_orchestration"] = agent_payload

    # These keys are direct intent signals — the agent payload wins when it
    # produced a meaningful value, otherwise we leave the existing context
    # untouched.
    for key in (
        "occasion",
        "sub_intent",
        "formality",
        "style_direction",
    ):
        value = agent_payload.get(key)
        if value:
            style_context.setdefault(key, value)
            # Promote agent value when current context value is empty/default.
            if not style_context.get(key):
                style_context[key] = value

    # Lists/dicts: always expose them so downstream engines can read uniformly.
    style_context["avoid_items"] = list(agent_payload.get("avoid_items") or [])
    style_context["required_slots"] = list(
        agent_payload.get("required_slots") or ["top", "bottom", "footwear"]
    )
    style_context["palette_direction"] = list(
        agent_payload.get("palette_direction") or []
    )
    style_context["accessory_policy"] = dict(
        agent_payload.get("accessory_policy") or {}
    )
    return style_context


__all__ = [
    "orchestrate_style_request",
    "orchestrate_style_request_sync",
    "validate_agent_style_payload",
    "default_agent_payload",
    "merge_agent_payload_into_context",
    "is_enabled",
]
