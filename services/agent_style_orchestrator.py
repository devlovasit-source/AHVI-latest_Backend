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
import hashlib
import json
import logging
import os
import time
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


# --- agent result cache (kills the ~20-40s Vertex Agent Runtime call on repeat)
_AGENT_CACHE: Dict[str, "tuple[float, Dict[str, Any]]"] = {}
_AGENT_CACHE_TTL = 600.0  # 10 minutes
_AGENT_CACHE_MAX = 256


def _agent_cache_key(message: Any, wardrobe_items: Any, chips: Any) -> str:
    ids = sorted(
        str(it.get("$id") or it.get("id") or it.get("name") or "")
        for it in (wardrobe_items or [])
        if isinstance(it, dict)
    )
    chip_s = ",".join(sorted(str(c) for c in (chips or [])))
    blob = f"{str(message or '').strip().lower()}|{'|'.join(ids)}|{chip_s}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _agent_cache_get(key: str) -> Optional[Dict[str, Any]]:
    row = _AGENT_CACHE.get(key)
    if not row:
        return None
    ts, payload = row
    if time.time() - ts > _AGENT_CACHE_TTL:
        _AGENT_CACHE.pop(key, None)
        return None
    return dict(payload)


def _agent_cache_set(key: str, payload: Dict[str, Any]) -> None:
    if len(_AGENT_CACHE) >= _AGENT_CACHE_MAX:
        # drop oldest
        oldest = min(_AGENT_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _AGENT_CACHE.pop(oldest, None)
    _AGENT_CACHE[key] = (time.time(), dict(payload))


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


# Keys whose presence (non-empty) signals real agent reasoning happened.
_CONFIDENCE_SIGNAL_KEYS_WEIGHTED = (
    ("occasion", 0.20),
    ("sub_intent", 0.15),
    ("style_direction", 0.20),
    ("required_slots", 0.10),
    ("avoid_items", 0.10),
    ("palette_direction", 0.05),
    ("accessory_policy", 0.05),
)


def _has_value(raw: Dict[str, Any], key: str) -> bool:
    """True when key is present AND non-empty / non-default."""
    if not isinstance(raw, dict) or key not in raw:
        return False
    val = raw.get(key)
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, (list, tuple, set, dict)):
        return bool(val)
    return True


def _infer_confidence(raw: Dict[str, Any]) -> float:
    """Score signal richness when the agent forgot to include confidence.

    Each populated orchestration key contributes per
    _CONFIDENCE_SIGNAL_KEYS_WEIGHTED; the sum is clamped to [0.50, 0.90].
    """
    if not isinstance(raw, dict) or not raw:
        return 0.0
    score = 0.0
    for key, weight in _CONFIDENCE_SIGNAL_KEYS_WEIGHTED:
        if _has_value(raw, key):
            score += weight
    if score <= 0.0:
        return 0.0
    if score < 0.50:
        score = 0.50
    if score > 0.90:
        score = 0.90
    return round(score, 2)


def validate_agent_style_payload(payload: Any) -> Dict[str, Any]:
    """Normalize and defensively coerce an agent payload to the documented schema.

    Missing or invalid fields fall back to safe defaults. Always returns a
    full payload — callers can rely on every key being present.

    Confidence handling:
    - If the agent supplied an explicit confidence number, honour it.
    - If the field is missing AND the payload carries real orchestration
      signals (occasion / sub_intent / style_direction / required_slots /
      avoid_items / palette_direction / accessory_policy), infer a
      richness-weighted confidence in [0.50, 0.90] and log it. ADK agents
      frequently omit the confidence field even when their output is
      strong, which otherwise causes merge_low_confidence to discard the
      whole payload.
    """
    raw = payload if isinstance(payload, dict) else {}

    explicit_confidence = "confidence" in raw and raw.get("confidence") is not None
    if explicit_confidence:
        confidence_value = _coerce_confidence(raw.get("confidence"))
    else:
        confidence_value = _infer_confidence(raw)
        if confidence_value > 0.0:
            try:
                logger.info(
                    "ahvi.agent.confidence_inferred confidence=%.2f keys=%s",
                    confidence_value,
                    sorted(raw.keys())[:12],
                )
            except Exception:
                pass

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
        "confidence": confidence_value,
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

    # Vertex AI Reasoning Engine resource id (e.g.
    # projects/<num>/locations/<region>/reasoningEngines/<id>) — call the
    # Vertex SDK instead of treating the string as an HTTP URL.
    try:
        from services._reasoning_engine_client import (
            call_reasoning_engine,
            looks_like_resource_id,
        )

        if looks_like_resource_id(endpoint):
            engine_result = await call_reasoning_engine(
                endpoint,
                system=_SYSTEM_PROMPT,
                prompt=prompt,
                timeout=timeout,
            )
            if isinstance(engine_result, dict):
                return engine_result
            return None
    except Exception:
        logger.debug("ahvi.agent.reasoning_engine_path_failed", exc_info=True)

    try:
        import httpx  # type: ignore
    except Exception:
        logger.debug("ahvi.agent.httpx_unavailable")
        return None

    if not endpoint.startswith(("http://", "https://")):
        logger.warning(
            "ahvi.agent.endpoint_not_http endpoint=%s — skipping HTTP fallback",
            endpoint[:80],
        )
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

    # Cache: the Vertex Agent Runtime call costs ~20-40s. The result depends on
    # the occasion/message + wardrobe contents, so cache per (message, wardrobe
    # id-set, chips). Repeated "coffee date" asks with an unchanged wardrobe
    # then skip the agent entirely.
    cache_key = _agent_cache_key(message, wardrobe_items, chips)
    cached = _agent_cache_get(cache_key)
    if cached is not None:
        logger.info("ahvi.agent.cache_hit user_id=%s key=%s", user_id, cache_key[:16])
        return cached

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

    # Inject inferred confidence on the RAW agent response before
    # validation. Some ADK-deployed agents return every orchestration
    # key but omit the documented `confidence` field, which would
    # otherwise leave validated.confidence at 0.0 and trip the
    # merge_low_confidence gate that discards the whole payload.
    if isinstance(raw, dict) and "confidence" not in raw:
        signal_keys = {
            "occasion",
            "sub_intent",
            "formality",
            "style_direction",
            "required_slots",
            "avoid_items",
            "palette_direction",
            "accessory_policy",
        }
        if any(k in raw and raw.get(k) for k in signal_keys):
            raw["confidence"] = _infer_confidence(raw) or 0.75
            logger.info(
                "ahvi.agent.confidence_inferred confidence=%.2f keys=%s",
                raw["confidence"],
                sorted(raw.keys())[:12],
            )

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
    # Only cache real (non-default) agent results so a transient failure is not
    # cached as the answer.
    if validated.get("confidence", 0.0) > 0.0:
        _agent_cache_set(cache_key, validated)
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

    # Confidence-aware merge: a low-confidence agent payload (e.g. confidence
    # 0.0 or below threshold, common when the agent times out or falls back to
    # defaults) must NOT downgrade router-derived intent like occasion or
    # sub_intent. Only non-conflicting hints (avoid_items, required_slots,
    # palette_direction, accessory_policy, style_direction when missing) merge
    # through.
    try:
        confidence = float(agent_payload.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    # 0.8 floor — only a high-confidence, explicit-conflict agent payload may
    # override a router-derived occasion/sub_intent. Spec: wrong board is
    # worse than no board.
    confidence_floor = 0.8
    high_confidence = confidence >= confidence_floor

    for key in ("occasion", "sub_intent", "formality"):
        value = agent_payload.get(key)
        if not value:
            continue
        existing = style_context.get(key)
        if existing:
            # Router/caller already supplied this key. Only overwrite when the
            # agent is confident; otherwise preserve the router value.
            if high_confidence and value != existing:
                style_context[key] = value
        else:
            style_context[key] = value

    # style_direction is a softer hint — let it merge even at low confidence
    # when the router did not pick one.
    if agent_payload.get("style_direction") and not style_context.get("style_direction"):
        style_context["style_direction"] = agent_payload["style_direction"]

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

    if not high_confidence:
        logger.info(
            "ahvi.agent.merge_low_confidence preserved_router_intent=%s confidence=%.2f",
            {k: style_context.get(k) for k in ("occasion", "sub_intent", "formality") if style_context.get(k)},
            confidence,
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
