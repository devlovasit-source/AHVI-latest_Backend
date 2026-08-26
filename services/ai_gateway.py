import json
import logging
import os
import re
import time
import ast
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Tuple

import requests

from services import llm_service
from services.request_context import get_request_id
from services.beta_ops_telemetry import new_operation_id, record_llm_attempt

logger = logging.getLogger("ahvi.ai_gateway")


@dataclass(frozen=True)
class GatewayPolicy:
    timeout_seconds: int
    model: str | None = None


_DEFAULT_POLICY = GatewayPolicy(timeout_seconds=35, model=None)
_POLICIES: Dict[str, GatewayPolicy] = {
    "general": GatewayPolicy(timeout_seconds=35, model=None),
    "styling": GatewayPolicy(timeout_seconds=45, model=None),
    "intent": GatewayPolicy(timeout_seconds=20, model=None),
    "vision": GatewayPolicy(
        timeout_seconds=int(os.getenv("OLLAMA_VISION_TIMEOUT_SECONDS", "8")),
        model=None,
    ),
}
_BREAKER_FAIL_THRESHOLD = max(
    1, int(os.getenv("AI_GATEWAY_BREAKER_FAIL_THRESHOLD", "4"))
)
_BREAKER_COOLDOWN_SECONDS = max(
    3, int(os.getenv("AI_GATEWAY_BREAKER_COOLDOWN_SECONDS", "20"))
)
_breaker_lock = Lock()
_breaker_state: Dict[str, Dict[str, float]] = {}


def _policy(usecase: str | None) -> GatewayPolicy:
    key = str(usecase or "general").strip().lower()
    return _POLICIES.get(key, _DEFAULT_POLICY)


def _breaker_key(usecase: str | None, op: str) -> str:
    return f"{str(usecase or 'general').strip().lower()}:{op}"


def _breaker_allows(key: str) -> bool:
    now = time.monotonic()
    with _breaker_lock:
        row = _breaker_state.get(key) or {}
        opened_until = float(row.get("opened_until") or 0.0)
        return now >= opened_until


def _breaker_mark_failure(key: str) -> None:
    now = time.monotonic()
    with _breaker_lock:
        row = _breaker_state.setdefault(key, {"failures": 0.0, "opened_until": 0.0})
        row["failures"] = float(row.get("failures") or 0.0) + 1.0
        if row["failures"] >= _BREAKER_FAIL_THRESHOLD:
            row["opened_until"] = now + _BREAKER_COOLDOWN_SECONDS


def _breaker_mark_success(key: str) -> None:
    with _breaker_lock:
        _breaker_state[key] = {"failures": 0.0, "opened_until": 0.0}


def _trace(
    event: str,
    *,
    request_id: str,
    usecase: str,
    op: str,
    details: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "event": event,
        "request_id": request_id,
        "usecase": usecase,
        "op": op,
    }
    if details:
        payload.update(details)
    logger.info("ai_gateway %s", payload)


def log_control_event(
    event: str,
    *,
    request_id: str = "",
    usecase: str = "general",
    details: Dict[str, Any] | None = None,
) -> None:
    _trace(
        event,
        request_id=str(request_id or ""),
        usecase=str(usecase or "general"),
        op="control_plane",
        details=details,
    )


def generate_text(
    prompt: str,
    *,
    options: Dict[str, Any] | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
    model: str | None = None,
    timeout_seconds: int | None = None,
    usecase: str | None = None,
    request_id: str | None = None,
) -> str:
    rid = str(request_id or get_request_id() or "")
    case = str(usecase or (signals or {}).get("context_mode") or "general")
    p = _policy(case)
    op_key = _breaker_key(case, "generate_text")
    if not _breaker_allows(op_key):
        _trace("breaker_open", request_id=rid, usecase=case, op="generate_text")
        return "none"
    started = time.perf_counter()
    try:
        result = llm_service.generate_text(
            prompt=prompt,
            options=options,
            user_profile=user_profile,
            signals=signals,
            model=model or p.model,
            timeout_seconds=timeout_seconds or p.timeout_seconds,
            usecase=case,
            request_id=rid,
        )
        _breaker_mark_success(op_key)
        _trace(
            "success",
            request_id=rid,
            usecase=case,
            op="generate_text",
            details={"latency_ms": int((time.perf_counter() - started) * 1000)},
        )
        return result
    except Exception:
        _breaker_mark_failure(op_key)
        _trace(
            "error",
            request_id=rid,
            usecase=case,
            op="generate_text",
            details={"latency_ms": int((time.perf_counter() - started) * 1000)},
        )
        raise


def chat_completion(
    messages: List[Dict[str, Any]],
    *,
    system_instruction: str = "",
    model: str | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
    usecase: str | None = None,
    request_id: str | None = None,
) -> str:
    rid = str(request_id or get_request_id() or "")
    case = str(usecase or (signals or {}).get("context_mode") or "general")
    p = _policy(case)
    op_key = _breaker_key(case, "chat_completion")
    if not _breaker_allows(op_key):
        _trace("breaker_open", request_id=rid, usecase=case, op="chat_completion")
        return "I'm temporarily overloaded. Please try again in a moment."
    started = time.perf_counter()
    try:
        result = llm_service.chat_completion(
            messages=messages,
            system_instruction=system_instruction,
            model=model or p.model or llm_service.DEFAULT_MODEL,
            user_profile=user_profile,
            signals=signals,
            timeout_seconds=timeout_seconds or p.timeout_seconds,
            usecase=case,
            request_id=rid,
        )
        _breaker_mark_success(op_key)
        _trace(
            "success",
            request_id=rid,
            usecase=case,
            op="chat_completion",
            details={"latency_ms": int((time.perf_counter() - started) * 1000)},
        )
        return result
    except Exception:
        _breaker_mark_failure(op_key)
        _trace(
            "error",
            request_id=rid,
            usecase=case,
            op="chat_completion",
            details={"latency_ms": int((time.perf_counter() - started) * 1000)},
        )
        raise


def extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty response")

    clean = (
        re.sub(r"```(?:json|python|text)?", "", raw, flags=re.IGNORECASE)
        .replace("```", "")
        .strip()
    )
    clean = clean.strip()

    try:
        return json.loads(clean)
    except Exception:
        pass

    obj_start = clean.find("{")
    obj_end = clean.rfind("}")
    arr_start = clean.find("[")
    arr_end = clean.rfind("]")

    candidates: List[str] = []
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(clean[obj_start : obj_end + 1])
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(clean[arr_start : arr_end + 1])

    def _remove_trailing_commas(value: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", value)

    def _json_to_python_literals(value: str) -> str:
        out = value
        out = re.sub(r"\btrue\b", "True", out, flags=re.IGNORECASE)
        out = re.sub(r"\bfalse\b", "False", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnull\b", "None", out, flags=re.IGNORECASE)
        return out

    def _try_parse(candidate: str) -> Any | None:
        text_candidate = str(candidate or "").strip()
        if not text_candidate:
            return None

        try:
            return json.loads(text_candidate)
        except Exception:
            pass

        repaired = _remove_trailing_commas(text_candidate)
        if repaired != text_candidate:
            try:
                return json.loads(repaired)
            except Exception:
                pass

        try:
            return ast.literal_eval(_json_to_python_literals(repaired))
        except Exception:
            return None

    for candidate in candidates:
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    raise ValueError("no valid JSON found in model response")


def parse_json_object(text: str) -> Dict[str, Any]:
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def parse_json_array(text: str) -> List[Any]:
    parsed = extract_json(text)
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON array")
    return parsed


def generate_json_object(
    prompt: str,
    *,
    options: Dict[str, Any] | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    raw = generate_text(
        prompt=prompt,
        options=options,
        user_profile=user_profile,
        signals=signals,
    )
    try:
        return parse_json_object(raw)
    except Exception as exc:
        logger.warning(
            "generate_json_object parse failed request_id=%s error=%s raw=%s",
            get_request_id(),
            str(exc),
            str(raw)[:240],
        )
        return {}


def chat_json_object(
    messages: List[Dict[str, Any]],
    *,
    system_instruction: str = "",
    model: str | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    raw = chat_completion(
        messages=messages,
        system_instruction=system_instruction,
        model=model,
        user_profile=user_profile,
        signals=signals,
    )
    try:
        return parse_json_object(raw)
    except Exception as exc:
        logger.warning(
            "chat_json_object parse failed request_id=%s error=%s raw=%s",
            get_request_id(),
            str(exc),
            str(raw)[:240],
        )
        return {}


def _vision_model_candidates() -> List[str]:
    preferred = str(
        os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:latest") or ""
    ).strip()
    if str(os.getenv("OLLAMA_VISION_ENABLE_FALLBACKS", "false")).strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return [preferred] if preferred else []

    fallback_raw = str(
        os.getenv(
            "OLLAMA_VISION_MODEL_FALLBACKS",
            "llama3.2-vision:latest,llama3.2-vision",
        )
        or ""
    ).strip()
    ordered: List[str] = []
    for model in [preferred, *[m.strip() for m in fallback_raw.split(",")]]:
        if model and model not in ordered:
            ordered.append(model)
    return ordered


def _ollama_generate_url() -> str:
    # Vision can run on a dedicated Ollama instance/port.
    base = (
        str(
            os.getenv(
                "OLLAMA_VISION_URL",
                os.getenv("OLLAMA_URL", "http://localhost:11434/api"),
            )
            or ""
        )
        .strip()
        .rstrip("/")
    )
    return f"{base}/generate" if base.endswith("/api") else f"{base}/api/generate"


def _normalize_vision_image_base64(value: str) -> str:
    text = str(value or "").strip()
    if "," in text:
        # Ollama expects raw base64 payload, not data URI prefix.
        text = text.split(",", 1)[1].strip()
    return text


def ollama_vision_json(
    *,
    prompt: str,
    image_base64: str,
    timeout_seconds: int | None = None,
    request_id: str | None = None,
    usecase: str | None = "vision",
    user_id: str | None = None,
) -> Tuple[Dict[str, Any], str]:
    rid = str(request_id or get_request_id() or "")
    case = str(usecase or "vision")
    op_key = _breaker_key(case, "vision_json")
    if not _breaker_allows(op_key):
        raise RuntimeError("vision_json circuit breaker open")

    p = _policy(case)
    timeout = int(
        timeout_seconds
        or p.timeout_seconds
        or int(os.getenv("OLLAMA_VISION_TIMEOUT_SECONDS", "45"))
    )
    vision_num_ctx = max(256, int(os.getenv("OLLAMA_VISION_NUM_CTX", "512")))
    normalized_image = _normalize_vision_image_base64(image_base64)
    keep_alive = str(os.getenv("OLLAMA_VISION_KEEP_ALIVE", "10m") or "10m")
    num_predict = max(96, int(os.getenv("OLLAMA_VISION_NUM_PREDICT", "256")))
    payload = {
        "prompt": prompt,
        "images": [normalized_image],
        "stream": False,
        "format": "json",
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": vision_num_ctx,
            "num_predict": num_predict,
        },
    }

    last_error: Exception | None = None
    started = time.perf_counter()
    operation_id = new_operation_id()
    attempt = 0
    for model in _vision_model_candidates():
        attempt_started = None
        provider_call_started = False
        provider_recorded = False
        try:
            url = _ollama_generate_url()
            request_payload = {**payload, "model": model}
            attempt += 1
            attempt_started = time.perf_counter()
            provider_call_started = True
            response = requests.post(
                url,
                json=request_payload,
                timeout=(min(2, timeout), timeout),
            )
            if response.status_code >= 400:
                body = (response.text or "").strip()
                body = body[:500] if body else ""
                raise RuntimeError(
                    f"Ollama vision request failed model={model} status={response.status_code} body={body}"
                )
            try:
                usage = response.json()
            except Exception:
                usage = {}
                try:
                    record_llm_attempt(
                        user_id=user_id,
                        request_id=rid,
                        operation_id=operation_id,
                        attempt=attempt,
                        provider="ollama",
                        model=model,
                        usecase=case,
                        status="success",
                        duration_ms=round((time.perf_counter() - attempt_started) * 1000),
                    )
                except Exception:
                    logger.warning("ahvi.llm.telemetry_failed provider=ollama")
                provider_recorded = True
                raise
            raw = usage.get("response", "{}")
            parsed = parse_json_object(raw)
            try:
                record_llm_attempt(
                    user_id=user_id,
                    request_id=rid,
                    operation_id=operation_id,
                    attempt=attempt,
                    provider="ollama",
                    model=model,
                    usecase=case,
                    status="success",
                    duration_ms=round((time.perf_counter() - attempt_started) * 1000),
                    input_tokens=usage.get("prompt_eval_count", usage.get("prompt_tokens")),
                    output_tokens=usage.get("eval_count", usage.get("completion_tokens")),
                    cached_tokens=usage.get("cached_tokens"),
                )
            except Exception:
                logger.warning("ahvi.llm.telemetry_failed provider=ollama")
            provider_recorded = True
            _trace(
                "success",
                request_id=rid,
                usecase=case,
                op="vision_json",
                details={
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "model_used": model,
                },
            )
            _breaker_mark_success(op_key)
            return parsed, model
        except Exception as exc:
            if provider_call_started and not provider_recorded and attempt and attempt_started is not None:
                try:
                    record_llm_attempt(
                        user_id=user_id,
                        request_id=rid,
                        operation_id=operation_id,
                        attempt=attempt,
                        provider="ollama",
                        model=model,
                        usecase=case,
                        status="failed",
                        duration_ms=round((time.perf_counter() - attempt_started) * 1000),
                        error_code=(
                            "timeout" if isinstance(exc, requests.Timeout)
                            else "connection_error" if isinstance(exc, requests.exceptions.ConnectionError)
                            else "http_4xx" if re.search(r"status=4\d\d", str(exc))
                            else "http_5xx" if re.search(r"status=5\d\d", str(exc))
                            else "provider_error"
                        ),
                    )
                except Exception:
                    logger.warning("ahvi.llm.telemetry_failed provider=ollama")
            last_error = exc
            continue

    _breaker_mark_failure(op_key)
    _trace(
        "error",
        request_id=rid,
        usecase=case,
        op="vision_json",
        details={"latency_ms": int((time.perf_counter() - started) * 1000)},
    )
    raise RuntimeError(str(last_error or "vision generation failed"))
