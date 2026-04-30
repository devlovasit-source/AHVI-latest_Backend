import base64
import json
import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin

import requests


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalize_base64(value: str) -> str:
    text = str(value or "").strip()
    return text.split(",", 1)[1].strip() if "," in text else text


def _extract_image_base64(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    direct_keys = [
        "image_base64",
        "output_image_base64",
        "result_image_base64",
        "edited_image_base64",
    ]
    for key in direct_keys:
        value = str(payload.get(key, "") or "").strip()
        if value:
            return _normalize_base64(value)

    data = payload.get("data")
    if isinstance(data, dict):
        for key in direct_keys:
            value = str(data.get(key, "") or "").strip()
            if value:
                return _normalize_base64(value)

    return ""


def _extract_image_url(payload: Any) -> str:
    if isinstance(payload, str) and payload.startswith(("http://", "https://")):
        return payload

    if isinstance(payload, dict):
        for key in ("url", "image_url", "path"):
            value = str(payload.get(key, "") or "").strip()
            if value:
                return value
        data = payload.get("data")
        if isinstance(data, (dict, list, str)):
            return _extract_image_url(data)

    if isinstance(payload, list):
        for item in payload:
            found = _extract_image_url(item)
            if found:
                return found
    return ""


def _download_image_as_base64(image_url: str, *, timeout_seconds: int) -> str:
    response = requests.get(image_url, timeout=timeout_seconds)
    if response.status_code >= 400:
        return ""
    content = response.content or b""
    if not content:
        return ""
    return base64.b64encode(content).decode("utf-8")


def _gradio_call_infer(
    *,
    base_endpoint: str,
    image_base64: str,
    prompt: str,
    headers: Dict[str, str],
    timeout_seconds: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    base = str(base_endpoint or "").rstrip("/")
    post_url = f"{base}/gradio_api/call/infer"
    image_data_uri = f"data:image/png;base64,{_normalize_base64(image_base64)}"

    init_payload = {
        "data": [image_data_uri, prompt],
    }
    init_resp = requests.post(post_url, json=init_payload, headers=headers, timeout=timeout_seconds)
    if init_resp.status_code >= 400:
        return None, {
            "enabled": True,
            "status_code": init_resp.status_code,
            "reason": f"gradio init failed: {init_resp.text[:300]}",
        }

    try:
        init_data = init_resp.json()
    except Exception:
        init_data = {}

    event_id = str((init_data or {}).get("event_id", "") or "").strip()
    if not event_id:
        return None, {"enabled": True, "reason": "gradio init missing event_id"}

    stream_url = f"{post_url}/{event_id}"
    stream_resp = requests.get(stream_url, headers=headers, timeout=timeout_seconds)
    if stream_resp.status_code >= 400:
        return None, {
            "enabled": True,
            "status_code": stream_resp.status_code,
            "reason": f"gradio stream failed: {stream_resp.text[:300]}",
        }

    data_lines = [line.strip() for line in (stream_resp.text or "").splitlines() if line.strip().startswith("data:")]
    for line in reversed(data_lines):
        raw = line[5:].strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue

        as_b64 = _extract_image_base64(parsed)
        if as_b64:
            return as_b64, {"enabled": True, "reason": "ok_gradio_base64"}

        image_url = _extract_image_url(parsed)
        if image_url:
            if image_url.startswith("/"):
                image_url = urljoin(f"{base}/", image_url.lstrip("/"))
            downloaded_b64 = _download_image_as_base64(image_url, timeout_seconds=timeout_seconds)
            if downloaded_b64:
                return downloaded_b64, {"enabled": True, "reason": "ok_gradio_url"}

    return None, {"enabled": True, "reason": "gradio output missing image"}


def regenerate_image(
    *,
    image_base64: str,
    request_id: str = "",
    prompt: str = "",
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Calls an external Qwen image-edit service endpoint.
    Expected response shape includes one of:
    - image_base64
    - output_image_base64
    - result_image_base64
    - edited_image_base64
    """
    enabled = _env_bool("HF_QWEN_REGEN_ENABLED", False)
    if not enabled:
        return None, {"enabled": False, "reason": "HF_QWEN_REGEN_ENABLED=false"}

    endpoint = str(os.getenv("HF_QWEN_REGEN_URL", "") or "").strip()
    if not endpoint:
        return None, {"enabled": True, "reason": "HF_QWEN_REGEN_URL missing"}

    timeout_seconds = int(os.getenv("HF_QWEN_REGEN_TIMEOUT_SECONDS", "120"))
    task_prompt = (
        str(prompt or "").strip()
        or str(
            os.getenv(
                "HF_QWEN_REGEN_PROMPT",
                "Keep only the garment item. Remove human body/person/mannequin and keep realistic cloth details.",
            )
            or ""
        ).strip()
    )

    headers = {"Content-Type": "application/json"}
    hf_token = str(os.getenv("HF_TOKEN", "") or "").strip()
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    if request_id:
        headers["X-Request-Id"] = str(request_id)

    payload = {
        "image_base64": _normalize_base64(image_base64),
        "prompt": task_prompt,
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_seconds)
        if response.status_code < 400:
            data = response.json() if response.text else {}
            edited = _extract_image_base64(data)
            if edited:
                return edited, {"enabled": True, "status_code": response.status_code, "reason": "ok_http_json"}
            image_url = _extract_image_url(data)
            if image_url:
                downloaded = _download_image_as_base64(image_url, timeout_seconds=timeout_seconds)
                if downloaded:
                    return downloaded, {"enabled": True, "status_code": response.status_code, "reason": "ok_http_url"}

        base_endpoint = endpoint
        if "/gradio_api/call/" in base_endpoint:
            base_endpoint = base_endpoint.split("/gradio_api/call/", 1)[0]
        edited2, meta2 = _gradio_call_infer(
            base_endpoint=base_endpoint,
            image_base64=image_base64,
            prompt=task_prompt,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
        if edited2:
            return edited2, meta2

        reason = str((meta2 or {}).get("reason", "") or "").strip()
        if response.status_code >= 400:
            reason = reason or f"qwen endpoint error: {response.text[:300]}"
            return None, {"enabled": True, "status_code": response.status_code, "reason": reason}
        return None, {"enabled": True, "reason": reason or "qwen response missing image"}
    except Exception as exc:
        return None, {"enabled": True, "reason": f"qwen request failed: {exc}"}
