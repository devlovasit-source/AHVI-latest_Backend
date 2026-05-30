"""Vertex AI Reasoning Engine transport for AHVI agent calls.

Both `agent_style_orchestrator` and `agent_metadata_validator` started
shipping with `AGENT_*_ENDPOINT` set to a Vertex Reasoning Engine
*resource id* (e.g. `projects/631493992863/locations/us-west1/reasoningEngines/6180706703449784320`)
rather than a real `https://...` URL. The old httpx path on those
strings raised:

    Request URL is missing an 'http://' or 'https://' protocol.

This helper detects that pattern and calls the Reasoning Engine via the
official Vertex AI Python SDK. Never raises — returns parsed dict or
None so callers fall back to legacy logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("ahvi.agent.reasoning_engine")


_RESOURCE_RE = re.compile(
    r"^projects/[^/]+/locations/([^/]+)/reasoningEngines/[^/]+$"
)


def looks_like_resource_id(endpoint: Any) -> bool:
    """True if `endpoint` is a Vertex Reasoning Engine resource id."""
    if not isinstance(endpoint, str):
        return False
    text = endpoint.strip()
    if not text.startswith("projects/"):
        return False
    return "/reasoningEngines/" in text


def _location_from_resource(resource_id: str, fallback: str = "us-central1") -> str:
    m = _RESOURCE_RE.match(resource_id.strip())
    return m.group(1) if m else fallback


def _project_from_env() -> str:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("AHVI_GCP_PROJECT")
        or ""
    )


def _parse_engine_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """Normalize the various Reasoning Engine / ADK response envelopes
    into a flat dict ready for the agent schema validator."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        # ADK-style: {"output": {"content": "..."}} or {"output": {"text": "..."}}
        output = payload.get("output")
        if isinstance(output, dict):
            content = output.get("content") or output.get("text")
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except Exception:
                    return None
            if isinstance(content, dict):
                return content
        # Plain content string at the top level.
        if isinstance(payload.get("content"), str):
            try:
                return json.loads(payload["content"])
            except Exception:
                return None
        # Already a flat dict — return as-is. The schema validator will
        # coerce missing keys.
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return None
    return None


def _invoke_reasoning_engine_sync(
    resource_id: str,
    *,
    system: str,
    prompt: str,
) -> Optional[Dict[str, Any]]:
    """Synchronous Reasoning Engine call.

    Wrapped in `asyncio.to_thread` by the public async helper so the
    surrounding event loop is not blocked.
    """
    try:
        import vertexai  # type: ignore
        from vertexai.preview import reasoning_engines  # type: ignore
    except Exception as exc:
        logger.warning("ahvi.agent.reasoning_engine_sdk_unavailable err=%s", str(exc)[:200])
        return None

    project = _project_from_env()
    location = _location_from_resource(resource_id, fallback="us-central1")

    try:
        if project:
            vertexai.init(project=project, location=location)
        engine = reasoning_engines.ReasoningEngine(resource_id.strip())
        # ReasoningEngine.query signatures differ across SDK versions.
        # Try the documented `input` keyword first, then fall back to
        # positional, then to a dict payload.
        try:
            raw = engine.query(input={"system": system, "prompt": prompt})
        except TypeError:
            try:
                raw = engine.query({"system": system, "prompt": prompt})
            except Exception:
                raw = engine.query(input=prompt)
        return _parse_engine_payload(raw)
    except Exception as exc:
        logger.warning(
            "ahvi.agent.reasoning_engine_call_failed resource=%s err=%s",
            resource_id,
            str(exc)[:200],
        )
        return None


async def call_reasoning_engine(
    resource_id: str,
    *,
    system: str,
    prompt: str,
    timeout: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """Async wrapper. Returns parsed dict or None on any failure."""
    if not looks_like_resource_id(resource_id):
        return None
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _invoke_reasoning_engine_sync,
                resource_id,
                system=system,
                prompt=prompt,
            ),
            timeout=max(1.0, float(timeout)),
        )
    except asyncio.TimeoutError:
        logger.warning("ahvi.agent.reasoning_engine_timeout resource=%s", resource_id)
        return None
    except Exception as exc:
        logger.warning(
            "ahvi.agent.reasoning_engine_unexpected resource=%s err=%s",
            resource_id,
            str(exc)[:200],
        )
        return None


__all__ = ["call_reasoning_engine", "looks_like_resource_id"]
